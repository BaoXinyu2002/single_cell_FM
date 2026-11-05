"""MUTUAL: multiome transfer in a unified aligned latent space (RNA + ATAC cell embeddings on frozen foundation models)."""
from .model import MUTUAL, bundle_paths, cross_modal_metrics

__version__ = "0.1.0"
__all__ = ["MUTUAL", "bundle_paths", "cross_modal_metrics"]
