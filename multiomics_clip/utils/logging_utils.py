"""Logging utilities for Multi-Omics CLIP training"""

import os
from typing import Dict, Optional
import torch


def setup_wandb(config):
    """
    Setup Weights & Biases logging

    Args:
        config: CLIPConfig instance
    """
    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Skipping wandb logging.")
        return

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        config=config.to_dict(),
        dir=config.save_dir
    )

    print(f"Wandb initialized: {wandb.run.name}")


def log_metrics(metrics: Dict[str, float], step: int):
    """
    Log metrics to wandb

    Args:
        metrics: Dictionary of metric name to value
        step: Current training step
    """
    try:
        import wandb
        wandb.log(metrics, step=step)
    except ImportError:
        pass  # Silently skip if wandb not available


def log_model_checkpoint(checkpoint_path: str, step: int, is_best: bool = False):
    """
    Log model checkpoint to wandb

    Args:
        checkpoint_path: Path to checkpoint file
        step: Training step
        is_best: Whether this is the best model
    """
    try:
        import wandb
        artifact = wandb.Artifact(
            f"model-{wandb.run.id}",
            type='model',
            metadata={'step': step, 'is_best': is_best}
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
    except ImportError:
        pass


def finish_wandb():
    """Finish wandb run"""
    try:
        import wandb
        wandb.finish()
    except ImportError:
        pass
