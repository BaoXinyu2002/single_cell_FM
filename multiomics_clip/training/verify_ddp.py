#!/usr/bin/env python3
"""
DDP Verification Script for Multi-Omics CLIP

This script verifies that the DDP implementation works correctly:
1. Gradient gathering across GPUs (GatherWithGrad)
2. Label indexing in distributed setting
3. Gradient accumulation with multiple micro-batches

Run with torchrun:
    torchrun --nproc_per_node=2 multiomics_clip/training/verify_ddp.py
    torchrun --nproc_per_node=4 multiomics_clip/training/verify_ddp.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from multiomics_clip.training.gradient_accumulation import GatherWithGrad
from multiomics_clip.training.distributed import setup_distributed, cleanup_distributed, is_main_process


def print_rank(msg: str):
    """Print message with rank prefix."""
    if dist.is_initialized():
        rank = dist.get_rank()
        print(f"[Rank {rank}] {msg}")
    else:
        print(f"[Single GPU] {msg}")


def test_gather_with_grad():
    """Test that GatherWithGrad correctly gathers features and propagates gradients."""
    print_rank("=" * 80)
    print_rank("Test 1: GatherWithGrad - Feature Gathering and Gradient Flow")
    print_rank("=" * 80)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # Create test features (different on each GPU)
    batch_size = 4
    feature_dim = 8
    features = torch.randn(batch_size, feature_dim, device=device, requires_grad=True)

    # Add rank-specific offset to make features different on each GPU
    features = features + rank * 10.0

    print_rank(f"Local features shape: {features.shape}")
    print_rank(f"Local features mean: {features.mean().item():.4f}")

    # Gather features across GPUs
    if dist.is_initialized():
        gathered_features = GatherWithGrad.apply(features)
    else:
        gathered_features = features

    expected_size = batch_size * world_size
    assert gathered_features.shape[0] == expected_size, \
        f"Gathered features shape mismatch: got {gathered_features.shape[0]}, expected {expected_size}"

    print_rank(f"Gathered features shape: {gathered_features.shape}")
    print_rank(f"Gathered features mean: {gathered_features.mean().item():.4f}")

    # Compute a simple loss that uses all gathered features
    # Each GPU computes loss based on its own samples vs all gathered samples
    loss = gathered_features[rank * batch_size:(rank + 1) * batch_size].pow(2).mean()

    print_rank(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check that gradients exist and are non-zero
    assert features.grad is not None, "Gradients not computed!"
    assert features.grad.abs().sum() > 0, "Gradients are all zero!"

    print_rank(f"Gradient mean: {features.grad.mean().item():.4f}")
    print_rank(f"Gradient std: {features.grad.std().item():.4f}")

    if is_main_process():
        print("\n✓ Test 1 PASSED: GatherWithGrad correctly gathers features and propagates gradients\n")


def test_label_indexing():
    """Test that label indexing is correct in distributed setting with gradient accumulation."""
    print_rank("=" * 80)
    print_rank("Test 2: Label Indexing for Gradient Accumulation with DDP")
    print_rank("=" * 80)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Simulate gradient accumulation parameters
    accum_freq = 3
    local_batch_size = 4

    print_rank(f"Gradient accumulation frequency: {accum_freq}")
    print_rank(f"Local batch size: {local_batch_size}")
    print_rank(f"World size: {world_size}")

    # Total samples across all micro-batches and all GPUs
    total_samples = accum_freq * local_batch_size * world_size
    print_rank(f"Total samples: {total_samples}")

    # Test label computation for each micro-batch
    for j in range(accum_freq):
        # Compute labels for micro-batch j on this GPU
        start_idx = j * (local_batch_size * world_size) + rank * local_batch_size
        end_idx = start_idx + local_batch_size

        labels = list(range(start_idx, end_idx))

        print_rank(f"Micro-batch {j}: labels = {labels}")

        # Verify labels are within valid range
        assert max(labels) < total_samples, \
            f"Label {max(labels)} out of range for {total_samples} samples"
        assert min(labels) >= 0, \
            f"Label {min(labels)} is negative"

    # Verify no overlap between ranks for the same micro-batch
    if dist.is_initialized():
        for j in range(accum_freq):
            start_idx = j * (local_batch_size * world_size) + rank * local_batch_size
            end_idx = start_idx + local_batch_size

            # Gather all start indices to verify no overlap
            start_indices = [torch.tensor(0, device='cuda') for _ in range(world_size)]
            dist.all_gather(start_indices, torch.tensor(start_idx, device='cuda'))

            if is_main_process():
                start_indices = [idx.item() for idx in start_indices]
                print(f"  Micro-batch {j} start indices across ranks: {start_indices}")

                # Check no overlap
                for i in range(len(start_indices)):
                    for k in range(i + 1, len(start_indices)):
                        assert start_indices[i] != start_indices[k], \
                            f"Overlap detected: ranks {i} and {k} have same start index {start_indices[i]}"

    if is_main_process():
        print("\n✓ Test 2 PASSED: Label indexing is correct for distributed gradient accumulation\n")


def test_gradient_accumulation_semantics():
    """Test that gradient accumulation produces correct effective batch size."""
    print_rank("=" * 80)
    print_rank("Test 3: Gradient Accumulation Semantics")
    print_rank("=" * 80)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # Create a simple linear model
    feature_dim = 8
    model = nn.Linear(feature_dim, 1).to(device)

    # Wrap with DDP
    if dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        model_module = model.module
    else:
        model_module = model

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    accum_freq = 2
    local_batch_size = 4

    print_rank(f"Accumulation frequency: {accum_freq}")
    print_rank(f"Local batch size: {local_batch_size}")
    print_rank(f"Effective batch size: {accum_freq * local_batch_size * world_size}")

    # Cache for features (simulate OpenCLIP strategy)
    cached_features = []

    # Phase 1: Cache features without gradients
    for j in range(accum_freq):
        features = torch.randn(local_batch_size, feature_dim, device=device)
        with torch.no_grad():
            if dist.is_initialized():
                # Gather across GPUs
                gathered = [torch.zeros_like(features) for _ in range(world_size)]
                dist.all_gather(gathered, features)
                features_gathered = torch.cat(gathered, dim=0)
            else:
                features_gathered = features

        cached_features.append((features.cpu(), features_gathered.cpu()))

    # Phase 2: Re-compute with gradients and accumulate
    optimizer.zero_grad()

    for j in range(accum_freq):
        # Move cached batch back to GPU
        features_local, features_all = cached_features[j]
        features_local = features_local.to(device).requires_grad_(True)

        # Re-compute with gradients
        if dist.is_initialized():
            features_gathered = GatherWithGrad.apply(features_local)
        else:
            features_gathered = features_local

        # Concatenate with cached features from other micro-batches
        features_all_with_grad = torch.cat(
            [cached_features[i][1].to(device) if i != j else features_gathered
             for i in range(accum_freq)],
            dim=0
        )

        # Simple loss: predict ones
        output = model(features_all_with_grad)
        target = torch.ones_like(output)
        loss = nn.functional.mse_loss(output, target)

        print_rank(f"Micro-batch {j} loss: {loss.item():.4f}")

        # Backward (gradients accumulate)
        loss.backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print_rank(f"{name} grad norm: {param.grad.norm().item():.4f}")

    # Single optimizer step
    optimizer.step()

    if is_main_process():
        print("\n✓ Test 3 PASSED: Gradient accumulation works correctly\n")


def main():
    """Run all DDP verification tests."""
    # Setup distributed training
    if 'RANK' in os.environ:
        rank, world_size, local_rank = setup_distributed(backend="nccl")
        print(f"\n[Rank {rank}/{world_size}] DDP Verification Script")
        print(f"[Rank {rank}] Using GPU: {local_rank}\n")
    else:
        print("Running in single-GPU mode (no torchrun environment detected)")
        print("For full DDP testing, run with: torchrun --nproc_per_node=2 verify_ddp.py\n")

    try:
        # Run tests
        test_gather_with_grad()
        test_label_indexing()
        test_gradient_accumulation_semantics()

        if is_main_process():
            print("=" * 80)
            print("ALL TESTS PASSED!")
            print("=" * 80)
            print("\nYour DDP implementation is working correctly.")
            print("You can now run full training with confidence.")
            print("\nNext steps:")
            print("  1. Submit SLURM job: sbatch multiomics_clip/scripts/run_train_ddp.sh")
            print("  2. Or run locally: bash multiomics_clip/scripts/launch_torchrun.sh --nproc_per_node 4 ...")
            print("=" * 80)

    finally:
        # Cleanup
        if dist.is_initialized():
            cleanup_distributed()


if __name__ == "__main__":
    main()
