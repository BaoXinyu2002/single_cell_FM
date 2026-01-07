"""Multi-Omics CLIP model for contrastive learning between RNA and ATAC modalities"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .encoders import RNAEncoder, ATACEncoder
from .config import CLIPConfig
from .utils.checkpoint import load_checkpoint


class ProjectionHead(nn.Module):
    """
    Projection head to map encoder outputs to shared embedding space

    Args:
        input_dim: Input dimension from encoder
        hidden_dim: Hidden dimension for MLP
        output_dim: Output dimension (shared embedding space)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to shared space

        Args:
            x: Input embeddings [batch_size, input_dim]

        Returns:
            Projected embeddings [batch_size, output_dim]
        """
        return self.projection(x)


class MultiOmicsCLIP(nn.Module):
    """
    Multi-Omics CLIP model for contrastive learning between RNA-seq and ATAC-seq

    This model aligns scRNA-seq (via scFoundation) and scATAC-seq (via EpiAgent)
    in a shared embedding space using contrastive learning (CLIP-style).

    Args:
        config: CLIPConfig instance with model and training parameters
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()

        self.config = config

        # Check if loading from CLIP checkpoint
        if config.clip_checkpoint_path:
            # Load checkpoint to get encoder configs
            checkpoint = load_checkpoint(config.clip_checkpoint_path, device='cpu')

            # Get saved encoder configs
            rna_encoder_config = checkpoint.get('rna_encoder_config')
            atac_encoder_config = checkpoint.get('atac_encoder_config')

            if rna_encoder_config is None or atac_encoder_config is None:
                raise ValueError(
                    "CLIP checkpoint does not contain encoder configs. "
                    "This checkpoint may have been saved before this feature was added. "
                    "Please use the original encoder checkpoint paths instead."
                )

            # Initialize encoders with saved configs (skip loading pretrained weights)
            self.rna_encoder = RNAEncoder(
                checkpoint_path="",  # Not used when skip_pretrained=True
                freeze=config.freeze_rna_encoder,
                skip_pretrained=True,
                saved_config=rna_encoder_config,
                use_gradient_checkpoint=config.use_gradient_checkpoint
            )

            self.atac_encoder = ATACEncoder(
                checkpoint_path="",  # Not used when skip_pretrained=True
                freeze=config.freeze_atac_encoder,
                skip_pretrained=True,
                use_gradient_checkpoint=config.use_gradient_checkpoint,
                **atac_encoder_config  # Pass architecture params
            )
        else:
            # Load pretrained encoders from original checkpoints
            self.rna_encoder = RNAEncoder(
                config.rna_encoder_path,
                freeze=config.freeze_rna_encoder,
                use_gradient_checkpoint=config.use_gradient_checkpoint
            )

            self.atac_encoder = ATACEncoder(
                config.atac_encoder_path,
                freeze=config.freeze_atac_encoder,
                use_gradient_checkpoint=config.use_gradient_checkpoint
            )

        # Get encoder output dimensions
        rna_dim = self.rna_encoder.embedding_dim
        atac_dim = self.atac_encoder.embedding_dim

        # Projection heads
        hidden_dim = config.projection_dim * 2

        self.rna_projection = ProjectionHead(
            input_dim=rna_dim,
            hidden_dim=hidden_dim,
            output_dim=config.projection_dim,
            dropout=config.projection_dropout
        )

        self.atac_projection = ProjectionHead(
            input_dim=atac_dim,
            hidden_dim=hidden_dim,
            output_dim=config.projection_dim,
            dropout=config.projection_dropout
        )

        # Learnable temperature parameter (using log parameterization to ensure positivity)
        # temperature = exp(log_temperature) ensures temperature > 0 always
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(config.temperature)))

        # Load weights from CLIP checkpoint if provided
        if config.clip_checkpoint_path:
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights from CLIP checkpoint: {config.clip_checkpoint_path}")

    def get_encoder_configs(self) -> Dict[str, dict]:
        """
        Get encoder configurations for saving in checkpoint

        Returns:
            Dictionary with 'rna_encoder_config' and 'atac_encoder_config'
        """
        return {
            'rna_encoder_config': self.rna_encoder.get_config(),
            'atac_encoder_config': self.atac_encoder.get_config()
        }

    @property
    def temperature(self) -> torch.Tensor:
        """
        Get temperature value (always positive via exponential parameterization)

        Returns:
            Temperature value (> 0)
        """
        return torch.exp(self.log_temperature)

    def encode_rna(self, rna_data: torch.Tensor, rna_gene_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode RNA expression to projected embeddings

        Args:
            rna_data: Gene expression [batch_size, 19266]
            rna_gene_ids: Gene position IDs [batch_size, 19266]

        Returns:
            L2-normalized projected embeddings [batch_size, projection_dim]
        """
        embeddings = self.rna_encoder(rna_data, rna_gene_ids)
        projected = self.rna_projection(embeddings)
        return F.normalize(projected, dim=-1)

    def encode_atac(self, atac_input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode ATAC accessibility to projected embeddings

        Args:
            atac_input_ids: cCRE indices [batch_size, seq_len]

        Returns:
            L2-normalized projected embeddings [batch_size, projection_dim]
        """
        embeddings = self.atac_encoder(atac_input_ids)
        projected = self.atac_projection(embeddings)
        return F.normalize(projected, dim=-1)

    def compute_contrastive_loss(
        self,
        rna_features: torch.Tensor,
        atac_features: torch.Tensor,
        gathered_rna_features: Optional[torch.Tensor] = None,
        gathered_atac_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute symmetric contrastive loss (InfoNCE)

        Args:
            rna_features: RNA embeddings [batch_size, projection_dim]
            atac_features: ATAC embeddings [batch_size, projection_dim]
            gathered_rna_features: All RNA features from DDP (optional)
            gathered_atac_features: All ATAC features from DDP (optional)

        Returns:
            Tuple of (total_loss, logits)
        """
        # Use gathered features if provided (for DDP)
        if gathered_rna_features is not None and gathered_atac_features is not None:
            # Compute similarity with all gathered features
            logits_rna_to_atac = torch.matmul(
                rna_features, gathered_atac_features.T
            ) / self.temperature

            logits_atac_to_rna = torch.matmul(
                atac_features, gathered_rna_features.T
            ) / self.temperature

            # Create labels for gathered batch
            batch_size = rna_features.shape[0]
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            labels = torch.arange(batch_size, device=rna_features.device) + rank * batch_size

        else:
            # Single-GPU: compute similarity within batch
            logits_rna_to_atac = torch.matmul(
                rna_features, atac_features.T
            ) / self.temperature

            logits_atac_to_rna = logits_rna_to_atac.T

            # Labels: diagonal (paired samples)
            batch_size = rna_features.shape[0]
            labels = torch.arange(batch_size, device=rna_features.device)

        # Symmetric cross-entropy loss
        loss_rna_to_atac = F.cross_entropy(logits_rna_to_atac, labels)
        loss_atac_to_rna = F.cross_entropy(logits_atac_to_rna, labels)

        total_loss = (loss_rna_to_atac + loss_atac_to_rna) / 2

        return total_loss, logits_rna_to_atac

    def forward(
        self,
        rna_data: torch.Tensor,
        rna_gene_ids: torch.Tensor,
        atac_input_ids: torch.Tensor,
        gathered_rna_features: Optional[torch.Tensor] = None,
        gathered_atac_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing contrastive loss

        Args:
            rna_data: Gene expression [batch_size, 19266]
            rna_gene_ids: Gene position IDs [batch_size, 19266]
            atac_input_ids: cCRE indices [batch_size, seq_len]
            gathered_rna_features: All RNA features from DDP (optional)
            gathered_atac_features: All ATAC features from DDP (optional)

        Returns:
            Dictionary containing:
                - loss: Contrastive loss
                - logits: Similarity matrix
                - rna_features: RNA embeddings
                - atac_features: ATAC embeddings
                - temperature: Current temperature value
        """
        # Encode both modalities
        rna_features = self.encode_rna(rna_data, rna_gene_ids)
        atac_features = self.encode_atac(atac_input_ids)

        # Compute contrastive loss
        loss, logits = self.compute_contrastive_loss(
            rna_features,
            atac_features,
            gathered_rna_features,
            gathered_atac_features
        )

        return {
            'loss': loss,
            'logits': logits,
            'rna_features': rna_features,
            'atac_features': atac_features,
            'temperature': self.temperature.item()
        }

    def get_parameters_by_lr(self, base_lr: float, encoder_lr_multiplier: float):
        """
        Get parameter groups with different learning rates

        Args:
            base_lr: Base learning rate for projection heads
            encoder_lr_multiplier: LR multiplier for encoders

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []

        # RNA encoder parameters (if not frozen)
        if not self.config.freeze_rna_encoder:
            param_groups.append({
                'params': self.rna_encoder.parameters(),
                'lr': base_lr * encoder_lr_multiplier,
                'name': 'rna_encoder'
            })

        # ATAC encoder parameters (if not frozen)
        if not self.config.freeze_atac_encoder:
            param_groups.append({
                'params': self.atac_encoder.parameters(),
                'lr': base_lr * encoder_lr_multiplier,
                'name': 'atac_encoder'
            })

        # Projection heads (always trainable)
        param_groups.append({
            'params': list(self.rna_projection.parameters()) +
                     list(self.atac_projection.parameters()),
            'lr': base_lr,
            'name': 'projections'
        })

        # Temperature parameter (log_temperature is the actual learnable parameter)
        param_groups.append({
            'params': [self.log_temperature],
            'lr': base_lr,
            'name': 'temperature'
        })

        return param_groups
