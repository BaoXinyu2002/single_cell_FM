"""Gradient accumulation utilities for memory-efficient contrastive learning

Implements OpenCLIP-style gradient accumulation:
1. Cache features WITHOUT gradients for all micro-batches
2. For each micro-batch j, re-compute its features WITH gradients
3. Use cached features from other batches as negatives (no grad)
4. Call backward() for each micro-batch
5. Step optimizer only once at the end
"""

import torch
import torch.distributed as dist
from typing import Optional, List, Dict


class GatherWithGrad(torch.autograd.Function):
    """
    Custom autograd function to gather tensors across GPUs while maintaining gradients.
    """

    @staticmethod
    def forward(ctx, input_tensor):
        """Gather tensors from all processes."""
        ctx.save_for_backward(input_tensor)

        if not dist.is_available() or not dist.is_initialized():
            return input_tensor

        # Gather from all processes
        gathered_tensors = [torch.zeros_like(input_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_tensors, input_tensor)

        return torch.cat(gathered_tensors, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Scatter gradients back to the corresponding process."""
        if not dist.is_available() or not dist.is_initialized():
            return grad_output

        # Each process only needs gradients for its own samples
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Split gradients by process
        grad_chunks = grad_output.chunk(world_size, dim=0)

        return grad_chunks[rank]


def gather_features_with_grad(features: torch.Tensor, distributed: bool = False) -> torch.Tensor:
    """Gather features from all GPUs with gradient support."""
    if distributed and dist.is_available() and dist.is_initialized():
        return GatherWithGrad.apply(features)
    else:
        return features
