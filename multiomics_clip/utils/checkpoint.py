"""Checkpoint saving and loading utilities"""

import os
import torch
from typing import Optional, Dict


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    epoch: int,
    config,
    checkpoint_path: str,
    additional_info: Optional[Dict] = None
):
    """
    Save training checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for AMP
        step: Current training step
        epoch: Current epoch
        config: Configuration object
        checkpoint_path: Path to save checkpoint
        additional_info: Additional information to save
    """
    # Create checkpoint directory if needed
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Create checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'config': config.to_dict() if hasattr(config, 'to_dict') else config
    }

    # Add scheduler state if available
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Add scaler state if available
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Add additional info
    if additional_info is not None:
        checkpoint.update(additional_info)

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu'
) -> Dict:
    """
    Load checkpoint

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load checkpoint to

    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    return checkpoint


def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    device: str = 'cpu'
) -> torch.nn.Module:
    """
    Load model weights from checkpoint

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint
        strict: Whether to strictly enforce key matching
        device: Device to load model to

    Returns:
        Model with loaded weights
    """
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    model.to(device)

    return model


def resume_from_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Resume training from checkpoint

    Args:
        checkpoint_path: Path to checkpoint
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        scaler: Gradient scaler instance
        device: Device to load to

    Returns:
        Dictionary with training state (step, epoch, etc.)
    """
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Load optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load scaler
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Return training state
    training_state = {
        'step': checkpoint.get('step', 0),
        'epoch': checkpoint.get('epoch', 0),
        'config': checkpoint.get('config', None)
    }

    return training_state
