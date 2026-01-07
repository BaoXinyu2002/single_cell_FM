"""Distributed training utilities for Multi-Omics CLIP"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed(backend: str = "nccl"):
    """
    Initialize distributed training

    Args:
        backend: Backend for distributed training ('nccl' for GPU, 'gloo' for CPU)

    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if not dist.is_initialized():
        # Get rank from environment variables
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank
        )

        # Set device for this process
        torch.cuda.set_device(local_rank)

        # Only print on rank 0 to avoid duplicate logs
        if rank == 0:
            print(f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")

        return rank, world_size, local_rank
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank


def cleanup_distributed():
    """
    Clean up distributed training
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0)

    Returns:
        True if main process, False otherwise
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def gather_features(
    features: torch.Tensor,
    world_size: Optional[int] = None,
    with_grad: bool = False
) -> torch.Tensor:
    """
    Gather features from all processes for global contrastive loss

    This is essential for CLIP-style contrastive learning in distributed settings,
    as it allows computing similarity across all samples in the global batch.

    Args:
        features: Local features [batch_size, dim]
        world_size: Number of processes (if None, uses dist.get_world_size())
        with_grad: If True, maintain gradient flow through gathered features
                   (required for correct contrastive learning)

    Returns:
        Gathered features [world_size * batch_size, dim]
    """
    if not dist.is_initialized():
        return features

    if world_size is None:
        world_size = dist.get_world_size()

    if with_grad:
        # Use gradient-aware gathering (import here to avoid circular dependency)
        from .gradient_accumulation import GatherWithGrad
        return GatherWithGrad.apply(features)
    else:
        # Standard gathering without gradients (for validation/metrics)
        gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
        dist.all_gather(gathered_features, features)
        gathered_features = torch.cat(gathered_features, dim=0)
        return gathered_features


def reduce_tensor(tensor: torch.Tensor, world_size: Optional[int] = None) -> torch.Tensor:
    """
    Reduce tensor across all processes (average)

    Args:
        tensor: Input tensor
        world_size: Number of processes

    Returns:
        Averaged tensor
    """
    if not dist.is_initialized():
        return tensor

    if world_size is None:
        world_size = dist.get_world_size()

    tensor_reduced = tensor.clone()
    dist.all_reduce(tensor_reduced, op=dist.ReduceOp.SUM)
    tensor_reduced /= world_size

    return tensor_reduced


def barrier():
    """
    Synchronization barrier for all processes
    """
    if dist.is_initialized():
        dist.barrier()


def get_world_size() -> int:
    """
    Get number of processes in distributed training

    Returns:
        World size (1 if not distributed)
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """
    Get rank of current process

    Returns:
        Rank (0 if not distributed)
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0
