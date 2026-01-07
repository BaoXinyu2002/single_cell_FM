"""Encoder wrappers for scFoundation (RNA) and EpiAgent (ATAC)"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional

# Add paths for model imports relative to repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(_REPO_ROOT, "scFoundation"))
sys.path.append(os.path.join(_REPO_ROOT, "scFoundation", "model"))
sys.path.append(os.path.join(_REPO_ROOT, "EpiAgent"))

try:
    from load import load_model_frommmf, gatherData
    from epiagent.model import EpiAgent
except ImportError as e:
    print(f"Warning: Could not import required models: {e}")
    print("Make sure scFoundation and EpiAgent are properly installed")
    raise


class RNAEncoder(nn.Module):
    """Wrapper for scFoundation RNA encoder

    This encoder processes gene expression data and returns cell embeddings
    using the pretrained scFoundation model.

    Args:
        checkpoint_path: Path to scFoundation checkpoint
        freeze: Whether to freeze encoder weights
        skip_pretrained: Skip loading pretrained weights (for loading from CLIP checkpoint)
        saved_config: Saved config dict (required if skip_pretrained=True)
        use_gradient_checkpoint: Whether to use gradient checkpointing to save memory
    """

    def __init__(
        self,
        checkpoint_path: str,
        freeze: bool = True,
        skip_pretrained: bool = False,
        saved_config: Optional[dict] = None,
        use_gradient_checkpoint: bool = False
    ):
        super().__init__()

        self.use_gradient_checkpoint = use_gradient_checkpoint

        if skip_pretrained and saved_config is not None:
            # Initialize from saved config without loading pretrained weights
            self.config = saved_config
            # Need to import select_model from scFoundation
            from pretrainmodels import select_model
            self.model = select_model(self.config)
        else:
            # Load pretrained scFoundation model
            self.model, self.config = load_model_frommmf(checkpoint_path, key='gene')

        # Get embedding dimension (4x hidden_dim for pooled cell embedding)
        self.embedding_dim = self.config['encoder']['hidden_dim'] * 4

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def get_config(self) -> dict:
        """Return encoder config for saving in checkpoint"""
        return self.config

    def forward(self, rna_data: torch.Tensor, rna_gene_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode RNA expression data to cell embeddings

        Args:
            rna_data: Gene expression tensor [batch_size, 19266]
                      Last 2 values are [target_resolution, log10(total_count)]
            rna_gene_ids: Gene position IDs [batch_size, 19266]

        Returns:
            Cell embeddings [batch_size, embedding_dim]
        """
        # Determine which values are non-zero (valid genes)
        value_labels = rna_data > 0

        # Gather non-zero data and corresponding position IDs
        x, x_padding = gatherData(rna_data, value_labels, self.config['pad_token_id'])
        position_gene_ids, _ = gatherData(
            rna_gene_ids, value_labels, self.config['pad_token_id']
        )

        # Token embedding
        x = self.model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)

        # Add positional embeddings
        position_emb = self.model.pos_emb(position_gene_ids)
        x = x + position_emb

        # Encode with transformer (use gradient checkpointing if enabled)
        if self.use_gradient_checkpoint and self.training:
            # Gradient checkpointing recomputes forward pass during backward
            # to save memory at the cost of extra computation
            gene_embeddings = checkpoint(
                self.model.encoder,
                x,
                x_padding,
                use_reentrant=False
            )
        else:
            gene_embeddings = self.model.encoder(x, x_padding)

        # Pool gene embeddings to get cell embedding
        # Strategy: concatenate [last_token, second_last_token, max_pool, mean_pool]
        emb1 = gene_embeddings[:, -1, :]  # Last special token
        emb2 = gene_embeddings[:, -2, :]  # Second-to-last special token
        emb3, _ = torch.max(gene_embeddings[:, :-2, :], dim=1)  # Max pool over genes
        emb4 = torch.mean(gene_embeddings[:, :-2, :], dim=1)  # Mean pool over genes

        cell_embedding = torch.cat([emb1, emb2, emb3, emb4], dim=1)

        return cell_embedding


class ATACEncoder(nn.Module):
    """Wrapper for EpiAgent ATAC encoder

    This encoder processes chromatin accessibility data (cCRE sequences)
    and returns cell embeddings using the pretrained EpiAgent model.

    Args:
        checkpoint_path: Path to EpiAgent checkpoint
        vocab_size: Vocabulary size (number of cCREs + special tokens)
        num_layers: Number of transformer layers
        embedding_dim: Hidden dimension
        num_attention_heads: Number of attention heads
        max_rank_embeddings: Maximum sequence length
        use_flash_attn: Whether to use FlashAttention
        freeze: Whether to freeze encoder weights
        skip_pretrained: Skip loading pretrained weights (for loading from CLIP checkpoint)
        use_gradient_checkpoint: Whether to use gradient checkpointing to save memory
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_size: int = 1355449,
        num_layers: int = 18,
        embedding_dim: int = 512,
        num_attention_heads: int = 8,
        max_rank_embeddings: int = 8192,
        use_flash_attn: bool = True,
        freeze: bool = True,
        skip_pretrained: bool = False,
        use_gradient_checkpoint: bool = False
    ):
        super().__init__()

        self.use_gradient_checkpoint = use_gradient_checkpoint

        # Store architecture config for saving
        self._arch_config = {
            'vocab_size': vocab_size,
            'num_layers': num_layers,
            'embedding_dim': embedding_dim,
            'num_attention_heads': num_attention_heads,
            'max_rank_embeddings': max_rank_embeddings,
            'use_flash_attn': use_flash_attn
        }

        # Initialize EpiAgent model
        self.model = EpiAgent(
            vocab_size=vocab_size,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            max_rank_embeddings=max_rank_embeddings,
            use_flash_attn=use_flash_attn
        )

        # Load pretrained weights (skip if loading from CLIP checkpoint)
        if not skip_pretrained:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Handle different checkpoint formats
            # Use strict=False to ignore training artifacts (criterion weights) not needed for inference
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

        self.embedding_dim = embedding_dim

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def get_config(self) -> dict:
        """Return encoder architecture config for saving in checkpoint"""
        return self._arch_config

    def forward(self, atac_input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode ATAC accessibility data to cell embeddings

        Args:
            atac_input_ids: cCRE indices [batch_size, seq_len]
                           Format: [CLS, ccre_1, ccre_2, ..., SEP, PAD, ...]

        Returns:
            Cell embeddings [batch_size, embedding_dim]
        """
        from torch.cuda.amp import autocast

        # Forward pass through EpiAgent with mixed precision (required for FlashAttention)
        # FlashAttention requires fp16/bf16 dtype, autocast handles this automatically
        with autocast(enabled=atac_input_ids.is_cuda):
            if self.use_gradient_checkpoint and self.training:
                # Gradient checkpointing recomputes forward pass during backward
                # to save memory at the cost of extra computation
                def forward_fn(input_ids):
                    return self.model(input_ids, return_transformer_output=True)

                outputs = checkpoint(
                    forward_fn,
                    atac_input_ids,
                    use_reentrant=False
                )
            else:
                outputs = self.model(atac_input_ids, return_transformer_output=True)

        # Extract CLS token embedding (first token)
        cell_embedding = outputs['transformer_outputs'][:, 0, :]

        return cell_embedding


def load_encoders(
    rna_checkpoint: str,
    atac_checkpoint: str,
    freeze_rna: bool = True,
    freeze_atac: bool = True
) -> Tuple[RNAEncoder, ATACEncoder]:
    """
    Load both RNA and ATAC encoders

    Args:
        rna_checkpoint: Path to scFoundation checkpoint
        atac_checkpoint: Path to EpiAgent checkpoint
        freeze_rna: Whether to freeze RNA encoder
        freeze_atac: Whether to freeze ATAC encoder

    Returns:
        Tuple of (RNAEncoder, ATACEncoder)
    """
    rna_encoder = RNAEncoder(rna_checkpoint, freeze=freeze_rna)
    atac_encoder = ATACEncoder(atac_checkpoint, freeze=freeze_atac)

    return rna_encoder, atac_encoder
