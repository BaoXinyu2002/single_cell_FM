# Multi-Omics CLIP: Contrastive Learning for RNA-seq and ATAC-seq Integration

Multi-Omics CLIP implements CLIP-style contrastive learning to align single-cell RNA-seq (via scFoundation) and scATAC-seq (via EpiAgent) modalities in a shared embedding space. This enables cross-modal retrieval, label transfer, and integrated analysis of multi-omics data.

## Overview

This package provides:
- **Paired multi-omics training**: Contrastive learning on paired RNA-ATAC data from 10x Multiome
- **Pretrained encoders**: Leverages scFoundation (100M parameters, RNA) and EpiAgent (18-layer transformer, ATAC)
- **Flexible architecture**: Freeze or fine-tune encoders with learnable projection heads
- **Distributed training**: Multi-GPU support with DDP and global contrastive loss
- **Comprehensive evaluation**: Retrieval metrics (Recall@K, MRR), clustering (ARI, NMI), and batch correction (kBET, iLISI)
- **Full preprocessing pipeline**: From raw 10x Multiome to model-ready paired datasets

## Installation

### Prerequisites

- Python 3.11+
- CUDA 11.7+ (for GPU support)
- PyTorch 2.0+
- FlashAttention v2

### Install Dependencies

```bash
# Create conda environment
conda create -n multiomics-clip python=3.11
conda activate multiomics-clip

# Install PyTorch (CUDA 11.7)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Install FlashAttention
pip uninstall -y ninja && pip install ninja
pip install flash-attn==2.5.8 --no-build-isolation

# Install Multi-Omics CLIP
cd multiomics_clip
pip install -r requirements.txt

# Install scFoundation and EpiAgent
pip install -e ../EpiAgent  # or pip install epiagent
```

### Download Pretrained Models

- **scFoundation**: Download from [SharePoint](https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/EmUQnvZMETlDvoCaBduCNeIBQArcOrd8T8iEpiGofFZ9CQ) → `scFoundation/model/models/models.ckpt`
- **EpiAgent**: Download from [Google Drive](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX) → `EpiAgent/model/EpiAgent/EpiAgent.pth`

## Quick Start

### 1. Preprocess 10x Multiome Data

```bash
python multiomics_clip/scripts/preprocess_multiome.py \
  --rna_h5 /path/to/filtered_feature_bc_matrix.h5 \
  --atac_fragments /path/to/fragments.tsv.gz \
  --gene_list scFoundation/model/OS_scRNA_gene_index.19264.tsv \
  --ccre_bed EpiAgent/data/cCRE.bed \
  --output_dir ./preprocessed_data
```

This will:
1. Preprocess RNA: normalize, log1p, align to 19,264 genes
2. Preprocess ATAC: intersect with cCREs, TF-IDF, tokenize
3. Pair modalities by cell barcodes
4. Split into train/val/test sets

### 2. Train Multi-Omics CLIP

**Single-GPU training:**

```bash
python multiomics_clip/scripts/train_clip.py \
  --train_rna ./preprocessed_data/train_rna.h5ad \
  --train_atac ./preprocessed_data/train_atac.h5ad \
  --val_rna ./preprocessed_data/val_rna.h5ad \
  --val_atac ./preprocessed_data/val_atac.h5ad \
  --rna_encoder_path scFoundation/model/models/models.ckpt \
  --atac_encoder_path EpiAgent/model/EpiAgent/EpiAgent.pth \
  --batch_size 64 \
  --learning_rate 5e-5 \
  --num_steps 100000 \
  --save_dir ./checkpoints \
  --wandb_project multiomics-clip
```

**Multi-GPU training (DDP):**

```bash
torchrun --nproc_per_node=4 multiomics_clip/scripts/train_clip.py \
  --distributed \
  --train_rna ./preprocessed_data/train_rna.h5ad \
  --train_atac ./preprocessed_data/train_atac.h5ad \
  --val_rna ./preprocessed_data/val_rna.h5ad \
  --val_atac ./preprocessed_data/val_atac.h5ad \
  --rna_encoder_path scFoundation/model/models/models.ckpt \
  --atac_encoder_path EpiAgent/model/EpiAgent/EpiAgent.pth \
  --batch_size 64 \
  --save_dir ./checkpoints
```

**Training with config file:**

```bash
python multiomics_clip/scripts/train_clip.py \
  --config multiomics_clip/configs/example_multiome.yaml \
  --train_rna ./preprocessed_data/train_rna.h5ad \
  --train_atac ./preprocessed_data/train_atac.h5ad \
  --val_rna ./preprocessed_data/val_rna.h5ad \
  --val_atac ./preprocessed_data/val_atac.h5ad
```

### 3. Evaluate Model

```bash
python multiomics_clip/scripts/evaluate_clip.py \
  --checkpoint ./checkpoints/best_model.pt \
  --test_rna ./preprocessed_data/test_rna.h5ad \
  --test_atac ./preprocessed_data/test_atac.h5ad \
  --celltype_column cell_type \
  --batch_column batch \
  --output_dir ./evaluation_results
```

This computes:
- **Retrieval metrics**: RNA→ATAC and ATAC→RNA Recall@K, MR, MRR
- **Clustering metrics**: kNN label transfer, ARI, NMI, F1
- **Batch correction**: kBET, iLISI, silhouette scores

## Usage Examples

### Python API

```python
import scanpy as sc
from multiomics_clip import MultiOmicsCLIP, CLIPConfig, PairedMultiOmicsDataset
from multiomics_clip.preprocessing import load_gene_list, pair_rna_atac_by_barcode
from multiomics_clip.training import train_single_gpu

# Load preprocessed data
rna_adata = sc.read_h5ad('train_rna.h5ad')
atac_adata = sc.read_h5ad('train_atac.h5ad')
gene_list = load_gene_list('scFoundation/model/OS_scRNA_gene_index.19264.tsv')

# Create dataset
dataset = PairedMultiOmicsDataset(
    rna_adata=rna_adata,
    atac_adata=atac_adata,
    gene_list=gene_list,
    target_resolution=4.0,
    max_atac_length=8192
)

# Configure model
config = CLIPConfig(
    rna_encoder_path='scFoundation/model/models/models.ckpt',
    atac_encoder_path='EpiAgent/model/EpiAgent/EpiAgent.pth',
    projection_dim=256,
    freeze_encoders=True,
    batch_size=64,
    learning_rate=5e-5,
    num_steps=100000
)

# Train
trainer = train_single_gpu(config, dataset)
```

### Extract Embeddings

```python
import torch
from multiomics_clip import MultiOmicsCLIP, CLIPConfig
from multiomics_clip.utils.checkpoint import load_model_from_checkpoint

# Load model
config = CLIPConfig(...)
model = MultiOmicsCLIP(config)
model = load_model_from_checkpoint(model, 'checkpoints/best_model.pt')
model.eval()

# Encode RNA
rna_data = ...  # [batch, 19266]
rna_gene_ids = torch.arange(19266).unsqueeze(0).repeat(batch_size, 1)
rna_embeddings = model.encode_rna(rna_data, rna_gene_ids)

# Encode ATAC
atac_input_ids = ...  # [batch, seq_len]
atac_embeddings = model.encode_atac(atac_input_ids)
```

## Architecture

### Model Components

1. **RNA Encoder (scFoundation)**
   - Input: 19,264 genes + 2 control values (resolution, total count)
   - Output: Cell embedding (4 × hidden_dim = 2048)
   - Architecture: Masked autoencoder with auto-binning

2. **ATAC Encoder (EpiAgent)**
   - Input: Variable-length cCRE sequences (tokenized)
   - Output: CLS token embedding (512-dim)
   - Architecture: 18-layer BERT with FlashAttention

3. **Projection Heads**
   - 2-layer MLP: Linear → GELU → LayerNorm → Dropout → Linear
   - Projects both modalities to shared embedding space (default 256-dim)
   - L2-normalized outputs

4. **Contrastive Loss**
   - Symmetric InfoNCE loss: (RNA→ATAC + ATAC→RNA) / 2
   - Learnable temperature parameter
   - Global batch negatives with DDP all_gather

### Training Strategy

- **Default**: Freeze encoders, train projection heads only
- **Fine-tuning**: Unfreeze encoders with lower LR (encoder_lr_multiplier=0.1)
- **Optimizer**: AdamW with Noam LR scheduler (warmup + cosine decay)
- **Mixed Precision**: AMP with GradScaler for efficiency

## Configuration

All parameters can be set via YAML config files. Key parameters:

### Model Architecture
- `projection_dim`: Shared embedding dimension (default: 256)
- `temperature`: Contrastive loss temperature (default: 0.07)
- `freeze_encoders`: Freeze pretrained encoders (default: true)

### Training
- `batch_size`: Batch size per GPU (default: 64)
- `learning_rate`: Base learning rate (default: 5e-5)
- `warmup_steps`: LR warmup steps (default: 10000)
- `num_steps`: Total training steps (default: 100000)
- `grad_clip_norm`: Gradient clipping (default: 1.0)

### Data
- `max_atac_length`: Maximum ATAC sequence length (default: 8192)
- `target_resolution`: scFoundation resolution parameter (default: 4.0)

See `configs/default.yaml` for full configuration options.

## Evaluation Metrics

### Retrieval Metrics
- **Recall@K**: Fraction of queries with correct match in top-K
- **Mean Rank (MR)**: Average rank of correct match
- **Mean Reciprocal Rank (MRR)**: Average 1/rank of correct match

Computed bidirectionally: RNA→ATAC and ATAC→RNA

### Clustering Metrics
- **ARI (Adjusted Rand Index)**: Similarity of predicted vs true labels
- **NMI (Normalized Mutual Information)**: Shared information between clusterings
- **F1 Score**: Macro and weighted F1 for multi-class

Evaluated via kNN label transfer in joint embedding space

### Batch Correction Metrics
- **kBET**: k-nearest neighbor Batch Effect Test (acceptance rate)
- **iLISI**: Integration Local Inverse Simpson's Index (higher = better mixing)
- **Silhouette Score**: Batch separation (lower = better mixing)

## Data Requirements

### Input Format

**10x Multiome:**
- RNA: `filtered_feature_bc_matrix.h5` (Gene Expression)
- ATAC: `fragments.tsv.gz` (ATAC Fragments)
- Genome: GRCh38/hg38 (use liftOver for hg19)

### Preprocessed Format

**RNA (AnnData):**
- `.X`: Normalized log1p counts [n_cells, 19264 genes]
- `.obs`: Cell metadata (barcodes, batch, cell type, etc.)
- `.var`: Gene names (aligned to scFoundation gene list)

**ATAC (AnnData):**
- `.obs['cell_sentences']`: JSON-formatted cCRE indices (TF-IDF ranked)
- `.obs`: Cell metadata (barcodes, batch, cell type, etc.)

**Pairing:**
- RNA and ATAC must have matching cell barcodes in `.obs_names`
- Same cell order in both modalities

## File Structure

```
multiomics_clip/
├── __init__.py
├── config.py                  # Configuration dataclass
├── model.py                   # MultiOmicsCLIP model
├── encoders.py                # RNA/ATAC encoder wrappers
├── dataset.py                 # PairedMultiOmicsDataset
├── preprocessing/             # Data preprocessing
│   ├── preprocess_rna.py
│   ├── preprocess_atac.py
│   └── pair_modalities.py
├── training/                  # Training infrastructure
│   ├── trainer.py
│   ├── train.py
│   └── distributed.py
├── evaluation/                # Evaluation metrics
│   ├── retrieval.py
│   ├── clustering.py
│   └── batch_correction.py
├── utils/                     # Utilities
│   ├── logging_utils.py
│   └── checkpoint.py
├── scripts/                   # Runnable scripts
│   ├── preprocess_multiome.py
│   ├── train_clip.py
│   └── evaluate_clip.py
├── configs/                   # Configuration files
│   ├── default.yaml
│   └── example_multiome.yaml
└── demo/                      # Demo notebooks
```

## Dependencies

Core requirements:
- PyTorch 2.0+
- FlashAttention 2.5.8
- scanpy, anndata
- scikit-learn
- pybedtools
- wandb (optional, for logging)

See `requirements.txt` for complete list.

## Citation

If you use Multi-Omics CLIP in your research, please cite:

**scFoundation:**
```
scFoundation: Large Scale Foundation Model on Single-cell Transcriptomics
Nature Methods, 2024
```

**EpiAgent:**
```
EpiAgent: Foundation model for single-cell epigenomics
Nature Methods, 2025
```

## License

This code is licensed under Apache-2.0. See `LICENSE` for details.

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your-email@example.com]

## Acknowledgments

This project builds on:
- [scFoundation](https://github.com/biomap-research/scFoundation) - RNA foundation model
- [EpiAgent](https://github.com/xy-chen16/EpiAgent) - ATAC foundation model
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - CLIP training framework inspiration
