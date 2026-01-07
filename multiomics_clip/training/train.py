"""Training entry points for single-GPU and multi-GPU training"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from ..model import MultiOmicsCLIP
from ..config import CLIPConfig
from ..dataset import PairedMultiOmicsDataset, collate_fn
from .trainer import CLIPTrainer
from .distributed import setup_distributed, cleanup_distributed, is_main_process


def train_single_gpu(
    config: CLIPConfig,
    train_dataset: PairedMultiOmicsDataset,
    val_dataset: PairedMultiOmicsDataset = None,
    resume: str | None = None,
):
    """
    Train Multi-Omics CLIP on a single GPU

    Args:
        config: CLIPConfig instance
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
    """
    print("Starting single-GPU training...")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.local_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.validation_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,  # Prevent worker memory leaks
            drop_last=False
        )

    # Create model
    model = MultiOmicsCLIP(config)

    # Create trainer
    trainer = CLIPTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=config.device,
        distributed=False
    )

    # Resume if requested
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        trainer.load(resume)

    # Train
    trainer.train()

    return trainer


def train_multi_gpu(
    config: CLIPConfig,
    train_dataset: PairedMultiOmicsDataset,
    val_dataset: PairedMultiOmicsDataset = None,
    resume: str | None = None,
):
    """
    Train Multi-Omics CLIP on multiple GPUs with DDP

    Args:
        config: CLIPConfig instance
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
    """
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed(backend="nccl")

    # Only log on main process
    if rank == 0:
        print("Starting multi-GPU training...")

    # Update config for distributed training
    config.distributed = True
    config.local_rank = local_rank
    config.world_size = world_size
    # Pass local_rank as integer for cleaner device handling in trainer
    config.device = local_rank

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    val_sampler = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.local_batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.validation_batch_size,
            sampler=val_sampler,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,  # Prevent worker memory leaks
            drop_last=False
        )

    # Create model
    model = MultiOmicsCLIP(config)

    # Create trainer
    trainer = CLIPTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=config.device,
        distributed=True
    )

    # Resume if requested (on all ranks to keep optimizer/scheduler in sync)
    if resume:
        if rank == 0:
            print(f"Resuming from checkpoint: {resume}")
        trainer.load(resume)

    # Train
    trainer.train()

    # Cleanup
    cleanup_distributed()

    return trainer if is_main_process() else None


def train_from_config(config_path: str, distributed: bool = False):
    """
    Train from YAML configuration file

    Args:
        config_path: Path to YAML config file
        distributed: Whether to use distributed training

    Returns:
        Trained model (on main process) or None (on other processes)
    """
    # Load config
    config = CLIPConfig.from_yaml(config_path)

    # Load datasets (this should be customized based on your data)
    # For now, we'll raise an error to remind users to implement data loading
    raise NotImplementedError(
        "Please implement dataset loading in train_from_config(). "
        "Load your preprocessed RNA and ATAC AnnData objects and create "
        "PairedMultiOmicsDataset instances for training and validation."
    )

    # Example implementation:
    """
    import scanpy as sc
    from ..preprocessing import pair_rna_atac_by_barcode, split_train_val_test

    # Load preprocessed data
    rna_adata = sc.read_h5ad(config.rna_data_path)
    atac_adata = sc.read_h5ad(config.atac_data_path)

    # Pair modalities
    rna_paired, atac_paired, _ = pair_rna_atac_by_barcode(rna_adata, atac_adata)

    # Split train/val/test
    train_rna, train_atac, val_rna, val_atac, test_rna, test_atac = split_train_val_test(
        rna_paired, atac_paired
    )

    # Load gene list
    gene_list = load_gene_list(config.gene_list_path)

    # Create datasets
    train_dataset = PairedMultiOmicsDataset(train_rna, train_atac, gene_list)
    val_dataset = PairedMultiOmicsDataset(val_rna, val_atac, gene_list)

    # Train
    if distributed:
        trainer = train_multi_gpu(config, train_dataset, val_dataset)
    else:
        trainer = train_single_gpu(config, train_dataset, val_dataset)

    return trainer
    """
