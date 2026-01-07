"""Configuration dataclass for Multi-Omics CLIP training"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import yaml
import os


@dataclass
class CLIPConfig:
    """Configuration for Multi-Omics CLIP model and training

    Model Configuration:
        rna_encoder_path: Path to scFoundation checkpoint
        atac_encoder_path: Path to EpiAgent checkpoint
        projection_dim: Dimension of shared embedding space
        temperature: Initial temperature for contrastive loss
        freeze_encoders: Whether to freeze pretrained encoders
        freeze_rna_encoder: Freeze only RNA encoder (if freeze_encoders=False)
        freeze_atac_encoder: Freeze only ATAC encoder (if freeze_encoders=False)

    Training Configuration:
        batch_size: Batch size per GPU (effective batch size, can be larger than memory allows)
        local_batch_size: Actual batch size per GPU per step (for memory constraints)
        accum_freq: Gradient accumulation frequency (batch_size / local_batch_size)
        learning_rate: Base learning rate
        encoder_lr_multiplier: LR multiplier for encoders (if unfrozen)
        warmup_steps: Number of warmup steps
        num_steps: Total training steps
        weight_decay: AdamW weight decay
        grad_clip_norm: Gradient clipping max norm

    Data Configuration:
        max_atac_length: Maximum ATAC sequence length
        target_resolution: scFoundation target resolution parameter
        gene_list_path: Path to scFoundation gene list TSV
        ccre_bed_path: Path to EpiAgent cCRE BED file

    Logging & Checkpointing:
        log_steps: Log metrics every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Run evaluation every N steps
        save_dir: Directory for checkpoints
        wandb_project: Wandb project name
        wandb_entity: Wandb entity name
        wandb_run_name: Wandb run name
        use_wandb: Enable wandb logging

    Distributed Training:
        distributed: Enable DDP training
        local_rank: Local rank for DDP
        world_size: Total number of processes

    Device Configuration:
        device: Device to use ('cuda' or 'cpu')
        mixed_precision: Enable AMP
    """

    # Model paths
    rna_encoder_path: str = ""
    atac_encoder_path: str = ""
    clip_checkpoint_path: Optional[str] = None  # Path to trained CLIP checkpoint for resume

    # Architecture
    projection_dim: int = 256
    projection_dropout: float = 0.0  # Dropout rate for projection heads
    temperature: float = 0.07
    freeze_encoders: bool = True
    freeze_rna_encoder: bool = False
    freeze_atac_encoder: bool = False
    use_gradient_checkpoint: bool = False  # Enable gradient checkpointing to reduce memory

    # Training hyperparameters
    batch_size: int = 32  # Effective batch size per GPU
    local_batch_size: Optional[int] = None  # Actual batch size per step (defaults to batch_size)
    accum_freq: int = 1  # Gradient accumulation frequency (auto-computed if local_batch_size is set)
    learning_rate: float = 5e-5
    encoder_lr_multiplier: float = 0.1  # LR for encoders if unfrozen
    warmup_steps: int = 10000
    num_steps: int = 100000
    weight_decay: float = 0.001
    grad_clip_norm: float = 1.0

    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-6

    # Data configuration
    max_atac_length: int = 8192
    target_resolution: float = 4.0
    gene_list_path: str = "scFoundation/model/OS_scRNA_gene_index.19264.tsv"
    ccre_bed_path: str = "EpiAgent/data/cCRE.bed"

    # Logging and checkpointing
    log_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 100
    save_dir: str = "./checkpoints"
    wandb_project: Optional[str] = "multiomics-clip"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True

    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1

    # Device
    device: str = "cuda"
    mixed_precision: bool = True

    # Evaluation
    eval_batch_size: int = 128
    validation_batch_size: int = 512  # Match effective training batch size for consistent contrastive learning
    validation_steps: Optional[int] = None  # Number of validation steps (None = full validation)
    retrieval_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 50])

    def __post_init__(self):
        """Validate configuration"""
        if self.freeze_encoders:
            self.freeze_rna_encoder = True
            self.freeze_atac_encoder = True

        # Auto-compute gradient accumulation parameters
        if self.local_batch_size is None:
            self.local_batch_size = self.batch_size
            self.accum_freq = 1
        else:
            # Compute accumulation frequency
            if self.batch_size % self.local_batch_size != 0:
                raise ValueError(
                    f"batch_size ({self.batch_size}) must be divisible by "
                    f"local_batch_size ({self.local_batch_size})"
                )
            self.accum_freq = self.batch_size // self.local_batch_size

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CLIPConfig":
        """Load configuration from YAML file

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            CLIPConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file

        Args:
            yaml_path: Path to save YAML configuration
        """
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self):
        """Convert configuration to dictionary

        Returns:
            Dictionary of configuration parameters
        """
        return asdict(self)
