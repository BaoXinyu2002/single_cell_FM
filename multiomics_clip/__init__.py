"""
Multi-Omics CLIP: Contrastive Learning for RNA-seq and ATAC-seq Integration

This package implements CLIP-style contrastive learning to align single-cell
RNA-seq (scFoundation) and scATAC-seq (EpiAgent) modalities in a shared
embedding space.
"""

__version__ = "0.1.0"

from .model import MultiOmicsCLIP
from .config import CLIPConfig
from .dataset import PairedMultiOmicsDataset, collate_fn

__all__ = [
    "MultiOmicsCLIP",
    "CLIPConfig",
    "PairedMultiOmicsDataset",
    "collate_fn",
]
