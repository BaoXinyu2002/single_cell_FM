#!/usr/bin/env python3
"""
Training script for Multi-Omics CLIP

This script trains the Multi-Omics CLIP model on paired RNA-ATAC data.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import scanpy as sc
import pandas as pd
from multiomics_clip import MultiOmicsCLIP, CLIPConfig, PairedMultiOmicsDataset
from multiomics_clip.training import train_single_gpu, train_multi_gpu
from multiomics_clip.preprocessing import load_gene_list


# Define default values for all overrideable arguments
OVERRIDE_DEFAULTS = {
    # Training parameters
    'batch_size': 256,
    'local_batch_size': 8,
    'learning_rate': 5e-5,
    'encoder_lr_multiplier': 0.1,
    'num_steps': 100000,
    'warmup_steps': 10000,
    'weight_decay': 0.001,
    'grad_clip_norm': 4.0,
    'adam_beta1': 0.9,
    'adam_beta2': 0.98,
    'adam_epsilon': 1e-6,

    # Model parameters
    'projection_dim': 512,
    'projection_dropout': 0.0,
    'temperature': 0.07,
    'freeze_encoders': False,
    'freeze_rna_encoder': False,
    'freeze_atac_encoder': False,
    'gradient_checkpoint': False,

    # Data/Eval parameters
    'max_atac_length': 8192,
    'target_resolution': 4.0,
    'eval_batch_size': 128,
    'validation_batch_size': 128,
    'validation_steps': None,

    # Logging parameters
    'save_dir': '/nfs/turbo/umms-drjieliu/usr/xinyubao/sclip/multiomics_clip/models',
    'log_steps': 100,
    'save_steps': 500,
    'eval_steps': 100,
    'wandb_project': 'multiomics-clip',
    'wandb_entity': None,
    'wandb_run_name': None,
    'use_wandb': True,

    # Device parameters
    'device': 'cuda',
    'mixed_precision': True,
    'distributed': False,
}

# Map argument names to config attribute names (when different)
ARG_TO_CONFIG_MAP = {
    'gradient_checkpoint': 'use_gradient_checkpoint',
}


def apply_command_line_overrides(config: CLIPConfig, args: argparse.Namespace):
    """
    Apply command-line argument overrides to config.

    Overrides config values when command-line argument differs from current config value.
    Prints override messages for transparency.

    Args:
        config: Config object to update
        args: Command-line arguments

    Returns:
        Updated config object
    """
    overridden = []

    for arg_name in OVERRIDE_DEFAULTS.keys():
        # Get argument value
        arg_value = getattr(args, arg_name, None)
        if arg_value is None:
            continue

        # Map argument name to config attribute name
        config_attr = ARG_TO_CONFIG_MAP.get(arg_name, arg_name)

        # Get current config value
        config_value = getattr(config, config_attr, None)

        # Check if value differs from current config
        if arg_value != config_value:
            # Override config value
            setattr(config, config_attr, arg_value)

            # Track override for logging
            overridden.append((config_attr, config_value, arg_value))

    # Handle inverted boolean flags
    if hasattr(args, 'no_mixed_precision') and args.no_mixed_precision:
        if config.mixed_precision:  # Only override if currently True
            config.mixed_precision = False
            overridden.append(('mixed_precision', True, False))

    if hasattr(args, 'no_wandb') and args.no_wandb:
        if config.use_wandb:  # Only override if currently True
            config.use_wandb = False
            overridden.append(('use_wandb', True, False))

    # Print all overrides (only on rank 0)
    if overridden:
        # Check if we're in distributed mode and not rank 0
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return config

        # Also check environment variable before distributed is initialized
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            print("\n  Command-line overrides:")
            for attr, old_val, new_val in overridden:
                print(f"    {attr}: {old_val} → {new_val}")

    return config


def main():
    # Set CUDA memory allocator config to reduce fragmentation
    # Keep settings compatible across PyTorch versions (avoid expandable_segments)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'

    # Determine if we're the main process (rank 0) for logging
    # This needs to be done early, before distributed initialization
    rank = int(os.environ.get('RANK', 0))
    is_main = (rank == 0)

    parser = argparse.ArgumentParser(
        description="Train Multi-Omics CLIP model"
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file (optional)'
    )

    # Data paths
    parser.add_argument(
        '--train_rna',
        type=str,
        required=True,
        help='Path to training RNA h5ad file'
    )
    parser.add_argument(
        '--train_atac',
        type=str,
        required=True,
        help='Path to training ATAC h5ad file'
    )
    parser.add_argument(
        '--val_rna',
        type=str,
        help='Path to validation RNA h5ad file (optional)'
    )
    parser.add_argument(
        '--val_atac',
        type=str,
        help='Path to validation ATAC h5ad file (optional)'
    )

    # Model paths
    parser.add_argument(
        '--rna_encoder_path',
        type=str,
        default='',
        help='Path to scFoundation checkpoint (not required if using --clip-checkpoint)'
    )
    parser.add_argument(
        '--atac_encoder_path',
        type=str,
        default='',
        help='Path to EpiAgent checkpoint (not required if using --clip-checkpoint)'
    )
    parser.add_argument(
        '--gene_list_path',
        type=str,
        default='/nfs/turbo/umms-drjieliu/usr/xinyubao/sclip/scFoundation/OS_scRNA_gene_index.19264.tsv',
        help='Path to scFoundation gene list'
    )
    parser.add_argument(
        '--ccre_bed_path',
        type=str,
        default='/nfs/turbo/umms-drjieliu/usr/xinyubao/sclip/EpiAgent/data/cCRE.bed',
        help='Path to EpiAgent cCRE BED file'
    )
    parser.add_argument(
        '--target_resolution',
        type=float,
        default=4.0,
        help='scFoundation target resolution parameter'
    )
    parser.add_argument(
        '--max_atac_length',
        type=int,
        default=8192,
        help='Maximum ATAC sequence length'
    )

    # Training parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size (effective batch size per GPU)'
    )
    parser.add_argument(
        '--local_batch_size',
        type=int,
        default=8,
        help='Actual batch size per GPU per step (for memory constraints, enables gradient accumulation)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--encoder_lr_multiplier',
        type=float,
        default=0.1,
        help='Learning rate multiplier for encoders (if unfrozen)'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=100000,
        help='Number of training steps'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=10000,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.001,
        help='AdamW weight decay'
    )
    parser.add_argument(
        '--grad_clip_norm',
        type=float,
        default=4.0,
        help='Gradient clipping max norm'
    )
    parser.add_argument(
        '--adam_beta1',
        type=float,
        default=0.9,
        help='Adam beta1 parameter'
    )
    parser.add_argument(
        '--adam_beta2',
        type=float,
        default=0.98,
        help='Adam beta2 parameter'
    )
    parser.add_argument(
        '--adam_epsilon',
        type=float,
        default=1e-6,
        help='Adam epsilon parameter'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/nfs/turbo/umms-drjieliu/usr/xinyubao/sclip/multiomics_clip/models',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=128,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=128,
        help='Batch size for validation (default: 1 to reduce memory usage)'
    )
    parser.add_argument(
        '--validation_steps',
        type=int,
        default=None,
        help='Number of validation steps to run (default: None for full validation)'
    )

    # Model parameters
    parser.add_argument(
        '--projection_dim',
        type=int,
        default=512,
        help='Projection dimension'
    )
    parser.add_argument(
        '--projection_dropout',
        type=float,
        default=0.0,
        help='Dropout rate for projection heads'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.07,
        help='Temperature for contrastive loss'
    )
    parser.add_argument(
        '--freeze_encoders',
        action='store_true',
        help='Freeze encoder weights'
    )
    parser.add_argument(
        '--freeze_rna_encoder',
        action='store_true',
        help='Freeze only RNA encoder (if freeze_encoders not set)'
    )
    parser.add_argument(
        '--freeze_atac_encoder',
        action='store_true',
        help='Freeze only ATAC encoder (if freeze_encoders not set)'
    )
    parser.add_argument(
        '--gradient_checkpoint',
        action='store_true',
        help='Enable gradient checkpointing to reduce memory usage'
    )

    # Device configuration
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        default=True,
        help='Enable automatic mixed precision (AMP)'
    )
    parser.add_argument(
        '--no_mixed_precision',
        action='store_true',
        help='Disable automatic mixed precision'
    )

    # Distributed training
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use distributed training (DDP)'
    )

    # Logging
    parser.add_argument(
        '--log_steps',
        type=int,
        default=100,
        help='Log metrics every N steps'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help='Save checkpoint every N steps'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=100,
        help='Run evaluation every N steps'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='multiomics-clip',
        help='Wandb project name'
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='Wandb entity name'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        help='Wandb run name'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable wandb logging'
    )

    # Resume training
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to CLIP checkpoint to resume training from (contains model weights + optimizer state)'
    )
    # Legacy support (deprecated)
    parser.add_argument(
        '--resume',
        type=str,
        help='[DEPRECATED] Use --checkpoint instead'
    )
    parser.add_argument(
        '--clip-checkpoint',
        type=str,
        dest='clip_checkpoint',
        help='[DEPRECATED] Use --checkpoint instead'
    )

    args = parser.parse_args()

    # Handle legacy flags (merge into --checkpoint)
    if args.clip_checkpoint:
        if args.checkpoint:
            parser.error("Cannot use both --checkpoint and --clip-checkpoint (deprecated). Use --checkpoint only.")
        args.checkpoint = args.clip_checkpoint
        print("WARNING: --clip-checkpoint is deprecated. Use --checkpoint instead.")
    if args.resume:
        if args.checkpoint:
            parser.error("Cannot use both --checkpoint and --resume (deprecated). Use --checkpoint only.")
        args.checkpoint = args.resume
        print("WARNING: --resume is deprecated. Use --checkpoint instead.")

    # Validate arguments
    # Only require encoder paths when training from scratch (not resuming/loading config)
    if not args.checkpoint and not args.config:
        if not args.rna_encoder_path or not args.atac_encoder_path:
            parser.error("--rna_encoder_path and --atac_encoder_path are required when training from scratch")

    if is_main:
        print("=" * 80)
        print("Multi-Omics CLIP: Training")
        print("=" * 80)

    # Load or create config
    if args.checkpoint:
        # Load config from checkpoint
        if is_main:
            print(f"\nLoading from checkpoint: {args.checkpoint}")
        import torch
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        checkpoint_config = checkpoint.get('config', {})

        # Create config from checkpoint
        config = CLIPConfig(**checkpoint_config)
        config.checkpoint_path = args.checkpoint

        if is_main:
            print(f"  Resuming from step: {checkpoint.get('step', 0)}")
            print(f"  Resuming from epoch: {checkpoint.get('epoch', 0)}")

    elif args.config:
        if is_main:
            print(f"\nLoading config from: {args.config}")
        config = CLIPConfig.from_yaml(args.config)
    else:
        if is_main:
            print("\nCreating config from command-line arguments...")
        config = CLIPConfig(
            # Model paths
            rna_encoder_path=args.rna_encoder_path,
            atac_encoder_path=args.atac_encoder_path,
            # Architecture
            projection_dim=args.projection_dim,
            projection_dropout=args.projection_dropout,
            temperature=args.temperature,
            freeze_encoders=args.freeze_encoders,
            freeze_rna_encoder=args.freeze_rna_encoder,
            freeze_atac_encoder=args.freeze_atac_encoder,
            use_gradient_checkpoint=args.gradient_checkpoint,
            # Training hyperparameters
            batch_size=args.batch_size,
            local_batch_size=args.local_batch_size,
            learning_rate=args.learning_rate,
            encoder_lr_multiplier=args.encoder_lr_multiplier,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
            # Data configuration
            max_atac_length=args.max_atac_length,
            target_resolution=args.target_resolution,
            gene_list_path=args.gene_list_path,
            ccre_bed_path=args.ccre_bed_path,
            # Logging and checkpointing
            log_steps=args.log_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_dir=args.save_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            use_wandb=not args.no_wandb,
            # Distributed training
            distributed=args.distributed,
            # Device
            device=args.device,
            mixed_precision=not args.no_mixed_precision if args.no_mixed_precision else args.mixed_precision,
            # Evaluation
            eval_batch_size=args.eval_batch_size,
            validation_batch_size=args.validation_batch_size,
            validation_steps=args.validation_steps
        )

    # Apply command-line overrides (applies to all loading modes: checkpoint, config, or from scratch)
    config = apply_command_line_overrides(config, args)

    # Handle batch_size/accum_freq relationship
    expected_accum = args.batch_size / args.local_batch_size
    if config.accum_freq > expected_accum:
        if is_main:
            print(f"  WARNING: Reducing accum_freq from {config.accum_freq} to {expected_accum} to prevent OOM")
        config.accum_freq = int(expected_accum)
        config.batch_size = config.local_batch_size * config.accum_freq
        if is_main:
            print(f"  New effective batch size: {config.batch_size}")

    # Save config
    config_path = os.path.join(config.save_dir, 'config.yaml')
    os.makedirs(config.save_dir, exist_ok=True)
    config.to_yaml(config_path)
    if is_main:
        print(f"Saved config to: {config_path}")

    # Load gene list
    if is_main:
        print("\nLoading gene list...")
    gene_list = load_gene_list(config.gene_list_path)

    # Load training data
    if is_main:
        print(f"\nLoading training data...")
        print(f"  RNA:  {args.train_rna}")
        print(f"  ATAC: {args.train_atac}")

    train_rna = sc.read_h5ad(args.train_rna)
    train_atac = sc.read_h5ad(args.train_atac)

    if is_main:
        print(f"  Training samples: {len(train_rna)}")

    # Create training dataset
    train_dataset = PairedMultiOmicsDataset(
        rna_adata=train_rna,
        atac_adata=train_atac,
        gene_list=gene_list,
        target_resolution=config.target_resolution,
        max_atac_length=config.max_atac_length,
        random_truncate=False
    )

    # Load validation data if provided
    val_dataset = None
    if args.val_rna and args.val_atac:
        if is_main:
            print(f"\nLoading validation data...")
            print(f"  RNA:  {args.val_rna}")
            print(f"  ATAC: {args.val_atac}")

        val_rna = sc.read_h5ad(args.val_rna)
        val_atac = sc.read_h5ad(args.val_atac)

        if is_main:
            print(f"  Validation samples: {len(val_rna)}")

        val_dataset = PairedMultiOmicsDataset(
            rna_adata=val_rna,
            atac_adata=val_atac,
            gene_list=gene_list,
            target_resolution=config.target_resolution,
            max_atac_length=config.max_atac_length,
            random_truncate=False
        )

    # Train
    if is_main:
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80)

    # Determine resume path
    resume_path = args.checkpoint

    if args.distributed:
        trainer = train_multi_gpu(config, train_dataset, val_dataset, resume=resume_path)
    else:
        trainer = train_single_gpu(config, train_dataset, val_dataset, resume=resume_path)

    if is_main:
        print("\n" + "=" * 80)
        print("Training complete!")
        print("=" * 80)


if __name__ == '__main__':
    main()
