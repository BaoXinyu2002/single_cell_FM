"""Training infrastructure for Multi-Omics CLIP"""

from .trainer import CLIPTrainer
from .train import train_single_gpu, train_multi_gpu
from .distributed import setup_distributed, cleanup_distributed, gather_features

__all__ = [
    "CLIPTrainer",
    "train_single_gpu",
    "train_multi_gpu",
    "setup_distributed",
    "cleanup_distributed",
    "gather_features",
]
