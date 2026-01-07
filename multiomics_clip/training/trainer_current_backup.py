"""Trainer class for Multi-Omics CLIP"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from typing import Optional, Dict
import logging
from tqdm import tqdm

from ..model import MultiOmicsCLIP
from ..config import CLIPConfig
from .distributed import gather_features, is_main_process, reduce_tensor
from ..utils.logging_utils import setup_wandb, log_metrics
from ..utils.checkpoint import save_checkpoint, load_checkpoint


class CLIPTrainer:
    """
    Trainer for Multi-Omics CLIP model

    Args:
        model: MultiOmicsCLIP model
        config: CLIPConfig instance
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader (optional)
        device: Device to use
        distributed: Whether using distributed training
    """

    def __init__(
        self,
        model: MultiOmicsCLIP,
        config: CLIPConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
        distributed: bool = False
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.distributed = distributed

        # Move model to device (handle both int and string device specifications)
        if isinstance(device, int):
            self.device = f"cuda:{device}"
        else:
            self.device = device
        self.model.to(self.device)

        # ============================================================
        # DDP (DistributedDataParallel) Setup for Multi-GPU Training
        # ============================================================
        # This wraps the model with PyTorch DDP for data-parallel training across GPUs.
        #
        # Key DDP behaviors:
        # 1. Each GPU has a replica of the model
        # 2. Forward pass is local to each GPU
        # 3. Gradients are automatically averaged across GPUs during backward()
        # 4. All replicas stay synchronized via collective communication (NCCL)
        #
        # For torchrun compatibility:
        # - Process group must be initialized via dist.init_process_group() (done in distributed.py)
        # - Environment variables (RANK, LOCAL_RANK, WORLD_SIZE) are set by torchrun
        # - device_ids must match the local GPU assigned to this process
        #
        # find_unused_parameters=False: We use all model parameters (encoders + projection heads).
        # Set to True only if some parameters don't receive gradients (reduces performance).
        if distributed:
            # Resolve device index for DDP
            # device can be: int (e.g., 0), string (e.g., "cuda:0"), or torch.device
            if isinstance(device, int):
                device_index = device
            elif isinstance(self.device, str) and self.device.startswith("cuda"):
                device_index = torch.device(self.device).index
                if device_index is None:
                    device_index = 0
            elif isinstance(self.device, torch.device) and self.device.type == "cuda":
                device_index = self.device.index or 0
            else:
                device_index = None

            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[device_index] if device_index is not None else None,
                output_device=device_index if device_index is not None else None,
                find_unused_parameters=False
            )
            # Keep reference to unwrapped model for accessing methods/attributes
            self.model_module = self.model.module
        else:
            self.model_module = self.model

        # Setup optimizer
        self._setup_optimizer()

        # Setup learning rate scheduler
        self._setup_scheduler()

        # Setup mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None

        # Setup wandb logging
        if config.use_wandb and is_main_process():
            setup_wandb(config)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Cached batches and features for gradient accumulation
        self.accum_batches = []
        self.accum_features = {'rna': [], 'atac': []}

        # Setup logging (add rank to log messages in distributed training)
        rank_str = f"[Rank {dist.get_rank()}] " if distributed and dist.is_initialized() else ""
        log_format = f'%(asctime)s - {rank_str}%(levelname)s - %(message)s'

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(config.save_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for different components"""
        param_groups = self.model_module.get_parameters_by_lr(
            base_lr=self.config.learning_rate,
            encoder_lr_multiplier=self.config.encoder_lr_multiplier
        )

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler with linear warmup and cosine decay"""
        import math

        def cosine_schedule_with_warmup(step):
            """Linear warmup followed by cosine decay"""
            warmup_steps = self.config.warmup_steps
            num_steps = self.config.num_steps

            if step < warmup_steps:
                # Linear warmup: 0 -> 1
                return step / max(1, warmup_steps)
            else:
                # Cosine decay: 1 -> 0
                progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=cosine_schedule_with_warmup)

    def train_step(self, batch) -> Optional[Dict[str, float]]:
        """
        Training step with OpenCLIP-style gradient accumulation

        Strategy:
        1. Cache features WITHOUT gradients for all micro-batches
        2. For each micro-batch j, re-compute its features WITH gradients
        3. Use cached features from other batches as negatives (no grad)
        4. Call backward() for each micro-batch (gradients accumulate)
        5. Step optimizer only once at the end

        DDP behavior with gradient accumulation:
        - Each GPU processes local_batch_size samples per micro-batch
        - gather_features() collects features from all GPUs (world_size * local_batch_size)
        - Total negatives per micro-batch: accum_freq * world_size * local_batch_size
        - DDP averages gradients across GPUs automatically during backward()
        - Label indices must account for: micro-batch position + GPU rank + local position

        Example with 2 GPUs, accum_freq=4, local_batch_size=8:
        - GPU 0 processes samples [0-7, 16-23, 32-39, 48-55]
        - GPU 1 processes samples [8-15, 24-31, 40-47, 56-63]
        - Total batch size: 4 * 2 * 8 = 64 samples
        - For micro-batch j=1 on GPU 0: labels are [16, 17, ..., 23]

        Args:
            batch: Tuple of (rna_data, rna_gene_ids, atac_input_ids)

        Returns:
            Dictionary of metrics (None if still accumulating, dict when optimizer step is taken)
        """
        self.model.train()

        # Move batch to device
        rna_data, rna_gene_ids, atac_input_ids = [x.to(self.device) for x in batch]
        # print(f"Batch sizes: RNA {rna_data.size()}, ATAC {atac_input_ids.size()}")
        # Store batch on CPU for later re-computation (avoid GPU OOM)
        self.accum_batches.append((
            rna_data.cpu(),
            rna_gene_ids.cpu(),
            atac_input_ids.cpu()
        ))

        # Cache features WITHOUT gradients
        with torch.no_grad():
            rna_features = self.model_module.encode_rna(rna_data, rna_gene_ids)
            atac_features = self.model_module.encode_atac(atac_input_ids)

            # Gather across GPUs if distributed
            if self.distributed:
                rna_features = gather_features(rna_features, with_grad=False)
                atac_features = gather_features(atac_features, with_grad=False)

        # Add to cache on CPU (detached, avoid GPU OOM)
        self.accum_features['rna'].append(rna_features.cpu())
        self.accum_features['atac'].append(atac_features.cpu())

        # If not ready, return None
        if len(self.accum_batches) < self.config.accum_freq:
            return None

        # === Now we have all micro-batches cached, do the actual training ===
        # print(f"Accumulated {len(self.accum_batches)} micro-batches, starting training step...")
        self.optimizer.zero_grad()
        total_loss = 0.0
        total_accuracy = 0.0

        # For each micro-batch j, re-compute its features WITH gradients
        for j in range(self.config.accum_freq):
            # Move cached batch from CPU to GPU
            rna_data_j, rna_gene_ids_j, atac_input_ids_j = self.accum_batches[j]
            rna_data_j = rna_data_j.to(self.device)
            rna_gene_ids_j = rna_gene_ids_j.to(self.device)
            atac_input_ids_j = atac_input_ids_j.to(self.device)

            with autocast(enabled=self.config.mixed_precision):
                # Re-compute features for batch j WITH gradients
                rna_features_j = self.model_module.encode_rna(rna_data_j, rna_gene_ids_j)
                atac_features_j = self.model_module.encode_atac(atac_input_ids_j)

                # Gather across GPUs if distributed (WITH gradients)
                if self.distributed:
                    rna_features_j = gather_features(rna_features_j, with_grad=True)
                    atac_features_j = gather_features(atac_features_j, with_grad=True)

                # Concatenate: cached features (moved from CPU) + current batch (with grad)
                rna_features_all = torch.cat(
                    [f.to(self.device) for f in self.accum_features['rna'][:j]] +
                    [rna_features_j] +
                    [f.to(self.device) for f in self.accum_features['rna'][j + 1:]]
                )
                atac_features_all = torch.cat(
                    [f.to(self.device) for f in self.accum_features['atac'][:j]] +
                    [atac_features_j] +
                    [f.to(self.device) for f in self.accum_features['atac'][j + 1:]]
                )

                # Verify feature shapes for gradient accumulation with DDP
                local_batch_size = rna_data_j.shape[0]
                expected_total_samples = self.config.accum_freq * local_batch_size
                if self.distributed:
                    world_size = torch.distributed.get_world_size()
                    expected_total_samples *= world_size

                assert rna_features_all.shape[0] == expected_total_samples, \
                    f"RNA features shape mismatch: got {rna_features_all.shape[0]}, expected {expected_total_samples}"
                assert atac_features_all.shape[0] == expected_total_samples, \
                    f"ATAC features shape mismatch: got {atac_features_all.shape[0]}, expected {expected_total_samples}"

                # Compute contrastive loss
                logits_rna = torch.matmul(rna_features_all, atac_features_all.T) / self.model_module.temperature
                logits_atac = logits_rna.T

                # Determine labels for batch j
                # In DDP: labels must index into the global gathered feature matrix
                if self.distributed:
                    rank = torch.distributed.get_rank()
                    world_size = torch.distributed.get_world_size()
                    # Global indices for this batch
                    start_idx = j * (local_batch_size * world_size) + rank * local_batch_size
                    end_idx = start_idx + local_batch_size
                else:
                    start_idx = j * local_batch_size
                    end_idx = start_idx + local_batch_size

                labels = torch.arange(start_idx, end_idx, device=self.device)

                # Verify labels are within valid range
                assert labels.max() < expected_total_samples, \
                    f"Label index {labels.max()} out of range for {expected_total_samples} samples"

                # Extract logits for batch j samples
                logits_rna_j = logits_rna[start_idx:end_idx]
                logits_atac_j = logits_atac[start_idx:end_idx]

                # Symmetric InfoNCE loss
                loss_rna = nn.functional.cross_entropy(logits_rna_j, labels)
                loss_atac = nn.functional.cross_entropy(logits_atac_j, labels)
                loss = 0.5 * (loss_rna + loss_atac)

            # Compute accuracy for this batch
            with torch.no_grad():
                preds = torch.argmax(logits_rna_j, dim=1)
                accuracy = (preds == labels).float().mean()
                total_accuracy += accuracy.item()

            # Backward for this micro-batch (gradients accumulate)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.detach().item()

            # Explicitly delete large intermediate tensors to prevent memory fragmentation
            del rna_features_all, atac_features_all
            del logits_rna, logits_atac, logits_rna_j, logits_atac_j
            del rna_features_j, atac_features_j
            del rna_data_j, rna_gene_ids_j, atac_input_ids_j
            if j < self.config.accum_freq - 1:
                torch.cuda.empty_cache()

        # Average loss and accuracy
        total_loss /= self.config.accum_freq
        total_accuracy /= self.config.accum_freq

        # Optimizer step (only once after all backward calls)
        if self.scaler is not None:
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

        self.scheduler.step()

        # Clear caches
        self.accum_batches = []
        self.accum_features = {'rna': [], 'atac': []}

        metrics = {
            'loss': total_loss,
            'accuracy': total_accuracy,
            'temperature': float(self.model_module.temperature.detach().item()),
            'lr': self.scheduler.get_last_lr()[0],
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Run validation

        Strategy (batch-based processing):
        1. Process each batch independently (no feature accumulation)
        2. Compute contrastive loss per batch (batch_size × batch_size)
        3. Accumulate loss and accuracy metrics
        4. Average metrics across all batches

        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}

        # Clear gradient accumulation caches before validation to free memory
        self.accum_batches = []
        self.accum_features = {'rna': [], 'atac': []}

        # Clear CUDA cache to release fragmented memory
        torch.cuda.empty_cache()
        print("Starting validation...")
        self.model.eval()

        # Metrics accumulation
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        num_batches = 0

        # Limit validation steps if specified
        max_steps = self.config.validation_steps if self.config.validation_steps is not None else float('inf')

        # Process each batch independently
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if batch_idx >= max_steps:
                    break

                rna_data, rna_gene_ids, atac_input_ids = [x.to(self.device) for x in batch]
                batch_size = rna_data.shape[0]

                # Encode features on GPU
                rna_features = self.model_module.encode_rna(rna_data, rna_gene_ids)
                atac_features = self.model_module.encode_atac(atac_input_ids)

                # Note: No gathering for distributed training since we compute per-batch metrics
                # Each GPU will compute its own metrics and they'll be averaged across GPUs later

                # Compute contrastive loss for this batch
                # Logits shape: [batch_size, batch_size]
                temperature = self.model_module.temperature
                logits_rna = torch.matmul(rna_features, atac_features.T) / temperature
                logits_atac = logits_rna.T

                # Labels: diagonal elements (i.e., matching pairs)
                labels = torch.arange(batch_size, device=self.device)

                # Symmetric InfoNCE loss
                loss_rna = nn.functional.cross_entropy(logits_rna, labels)
                loss_atac = nn.functional.cross_entropy(logits_atac, labels)
                batch_loss = 0.5 * (loss_rna + loss_atac)

                # Compute accuracy (RNA->ATAC direction)
                preds = torch.argmax(logits_rna, dim=1)
                batch_accuracy = (preds == labels).float().mean()

                # Accumulate metrics (weighted by batch size for correct averaging)
                total_loss += batch_loss.item() * batch_size
                total_accuracy += batch_accuracy.item() * batch_size
                total_samples += batch_size
                num_batches += 1

                # Explicitly free GPU memory immediately after each batch
                del rna_data, rna_gene_ids, atac_input_ids
                del rna_features, atac_features
                del logits_rna, logits_atac, labels
                del batch_loss, batch_accuracy, preds

        if num_batches == 0:
            return {}

        # Average metrics across all batches
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples

        # Prepare metrics
        val_metrics = {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy
        }

        # Reduce metrics across GPUs if distributed
        if self.distributed:
            for key in val_metrics:
                val_metrics[key] = reduce_tensor(
                    torch.tensor(val_metrics[key], device=self.device)
                ).item()

        # Clear cache after validation
        torch.cuda.empty_cache()

        return val_metrics

    def train(self):
        """
        Main training loop
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Total steps: {self.config.num_steps}")
        self.logger.info(f"Effective batch size per GPU: {self.config.batch_size}")
        self.logger.info(f"Local batch size per GPU: {self.config.local_batch_size}")
        self.logger.info(f"Gradient accumulation frequency: {self.config.accum_freq}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")

        running_metrics = {}
        num_accumulation = 0

        while self.global_step < self.config.num_steps:
            for batch in self.train_dataloader:
                if self.global_step >= self.config.num_steps:
                    break

                # Training step (returns None if still accumulating micro-batches)
                metrics = self.train_step(batch)

                # Only process metrics if optimizer step was taken
                if metrics is not None:
                    # Accumulate metrics
                    for key, value in metrics.items():
                        running_metrics[key] = running_metrics.get(key, 0.0) + value
                    num_accumulation += 1

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_steps == 0 and is_main_process():
                        avg_metrics = {
                            key: value / num_accumulation
                            for key, value in running_metrics.items()
                        }

                        self.logger.info(
                            f"Step {self.global_step}/{self.config.num_steps} | "
                            f"Loss: {avg_metrics['loss']:.4f} | "
                            f"Acc: {avg_metrics['accuracy']:.4f} | "
                            f"Temp: {avg_metrics['temperature']:.4f} | "
                            f"LR: {avg_metrics['lr']:.2e}"
                        )

                        if self.config.use_wandb:
                            log_metrics(avg_metrics, step=self.global_step)

                        running_metrics = {}
                        num_accumulation = 0

                    # Validation
                    if self.global_step % self.config.eval_steps == 0:
                        val_metrics = self.validate()

                        if is_main_process() and val_metrics:
                            self.logger.info(
                                f"Validation | "
                                f"Loss: {val_metrics['val_loss']:.4f} | "
                                f"Acc: {val_metrics['val_accuracy']:.4f}"
                            )

                            if self.config.use_wandb:
                                log_metrics(val_metrics, step=self.global_step)

                            # Save best model
                            if val_metrics['val_loss'] < self.best_val_loss:
                                self.best_val_loss = val_metrics['val_loss']
                                self.save(os.path.join(self.config.save_dir, 'best_model.pt'))

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0 and is_main_process():
                        checkpoint_path = os.path.join(
                            self.config.save_dir,
                            f'checkpoint_step_{self.global_step}.pt'
                        )
                        self.save(checkpoint_path)

                        # Cleanup old checkpoints (keep only last + best)
                        self._cleanup_old_checkpoints(keep_best=True)

            self.current_epoch += 1

        self.logger.info("Training completed!")

        # Save final model
        if is_main_process():
            self.save(os.path.join(self.config.save_dir, 'final_model.pt'))

    def save(self, checkpoint_path: str):
        """
        Save checkpoint

        Args:
            checkpoint_path: Path to save checkpoint
        """
        # Get encoder configs for resuming without original encoder checkpoints
        encoder_configs = self.model_module.get_encoder_configs()

        save_checkpoint(
            model=self.model_module,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.global_step,
            epoch=self.current_epoch,
            config=self.config,
            checkpoint_path=checkpoint_path,
            additional_info=encoder_configs
        )
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load(self, checkpoint_path: str):
        """
        Load checkpoint

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = load_checkpoint(checkpoint_path)

        self.model_module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.global_step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from step {self.global_step}, epoch {self.current_epoch}")

    def _cleanup_old_checkpoints(self, keep_best: bool = True):
        """
        Remove old checkpoints, keeping only the last checkpoint and best model

        Args:
            keep_best: Whether to preserve best_model.pt
        """
        import glob

        # Find all checkpoint files
        checkpoint_pattern = os.path.join(self.config.save_dir, 'checkpoint_step_*.pt')
        all_checkpoints = glob.glob(checkpoint_pattern)

        if not all_checkpoints:
            return

        # Sort by step number (extract from filename)
        def get_step_from_filename(path):
            basename = os.path.basename(path)
            # Extract step number from 'checkpoint_step_XXXX.pt'
            step_str = basename.replace('checkpoint_step_', '').replace('.pt', '')
            return int(step_str)

        all_checkpoints.sort(key=get_step_from_filename)

        # Keep only the last checkpoint, remove all others
        if len(all_checkpoints) > 1:
            to_remove = all_checkpoints[:-1]
            for checkpoint_path in to_remove:
                try:
                    os.remove(checkpoint_path)
                    self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {checkpoint_path}: {e}")
