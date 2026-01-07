# Multi-Omics Foundation Models: scFoundation + EpiAgent + CLIP Integration

This repository contains **scFoundation** (single-cell RNA-seq foundation model), **EpiAgent** (scATAC-seq foundation model), and **Multi-Omics CLIP** (contrastive learning framework for RNA-ATAC integration).

## Overview

### scFoundation: Large-Scale Foundation Model on Single-Cell Transcriptomics

**Published in *Nature Methods* (2024)**

scFoundation is a 100M-parameter pretrained foundation model for single-cell RNA sequencing data. Built on the xTrimoGene architecture and trained on over 50 million human single-cell transcriptomes, scFoundation achieves state-of-the-art performance across diverse downstream tasks including:

- Cell type annotation
- Gene expression enhancement
- Drug response prediction (SCAD, DeepCDR)
- Perturbation prediction (GEARS)
- Gene module inference
- Cell mapping

**Key Features:**
- 100M trainable parameters
- Pretrained on 50M+ human cells
- Fixed gene set: 19,264 genes
- Supports both cell-level and gene-level embeddings
- Available via [API](https://aigp.biomap.com/) and local inference

**Publication:** [Nature Methods, 2024](https://www.nature.com/articles/s41592-024-02305-7)
**Documentation:** See [scFoundation/README.md](scFoundation/README.md)

---

### EpiAgent: Foundation Model for Single-Cell Epigenomics

**Published in *Nature Methods* (2025)**

EpiAgent is a foundation model pretrained on 5.4M+ cells from the Human-scATAC-Corpus, designed to capture chromatin accessibility patterns and regulatory networks in single-cell ATAC-seq data. EpiAgent represents cells as concise "cell sentences" using cCRE-based tokenization and employs bidirectional attention to model cellular heterogeneity.

**Key Features:**
- 18-layer transformer architecture with FlashAttention v2
- Pretrained on 5.4M+ human scATAC-seq cells
- 1,355,449 vocabulary size (cCREs + special tokens)
- Zero-shot and fine-tuned modes for all major tasks
- Supports unsupervised feature extraction, cell type annotation, data imputation, perturbation prediction, and in-silico cCRE knockout

**Publication:** [Nature Methods, 2025](https://www.nature.com/articles/s41592-025-02822-z)
**Database:** [Human-scATAC-Corpus](https://health.tsinghua.edu.cn/human-scatac-corpus)
**Documentation:** See [EpiAgent/README.md](EpiAgent/README.md)

---

### Multi-Omics CLIP: Contrastive Learning for RNA-ATAC Integration

Multi-Omics CLIP implements CLIP-style contrastive learning to align scRNA-seq (via scFoundation) and scATAC-seq (via EpiAgent) in a shared embedding space. This enables cross-modal retrieval, label transfer, and integrated multi-omics analysis.

**Key Features:**
- Paired RNA-ATAC training on 10x Multiome datasets
- Symmetric InfoNCE contrastive loss
- Freeze or fine-tune pretrained encoders
- Distributed training with DDP and global batch negatives
- Comprehensive evaluation: retrieval metrics, clustering, batch correction

**Documentation:** See [multiomics_clip/README.md](multiomics_clip/README.md)

---

## Quick Start

### Installation

#### Prerequisites

- Python 3.11+
- CUDA 11.7+ (for GPU support)
- PyTorch 2.0+

#### Environment Setup

```bash
# Create conda environment
conda create -n multiomics python=3.11
conda activate multiomics

# Install PyTorch (CUDA 11.7)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Install FlashAttention (required for EpiAgent)
pip uninstall -y ninja && pip install ninja
pip install flash-attn==2.5.8 --no-build-isolation

# Install EpiAgent
pip install epiagent

# Install scFoundation dependencies
pip install scanpy anndata einops local_attention pandas numpy scipy

# Install Multi-Omics CLIP dependencies
cd multiomics_clip
pip install -r requirements.txt
```

#### Download Pretrained Models

**scFoundation:**
- Model weights: [SharePoint](https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/EmUQnvZMETlDvoCaBduCNeIBQArcOrd8T8iEpiGofFZ9CQ)
- Save to: `scFoundation/model/models/models.ckpt`

**EpiAgent:**
- Model weights: [Google Drive](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX)
- Save to: `EpiAgent/model/EpiAgent/EpiAgent.pth`

---

## Usage Examples

### scFoundation: Generate Cell Embeddings

```bash
python scFoundation/model/get_embedding.py \
  --task_name my_task \
  --input_type singlecell \
  --output_type cell \
  --pool_type all \
  --tgthighres a5 \
  --data_path /path/to/data.h5ad \
  --save_path ./output \
  --pre_normalized F \
  --version ce
```

**Python API:**

```python
from scFoundation.model.get_embedding import main_gene_selection
import pandas as pd

# Align gene list to scFoundation's 19,264 genes
gene_list_df = pd.read_csv('scFoundation/model/OS_scRNA_gene_index.19264.tsv',
                           header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])
X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)
```

See [scFoundation/model/README.md](scFoundation/model/README.md) for detailed documentation.

---

### EpiAgent: Zero-Shot Feature Extraction

```python
from epiagent.inference import extract_features

# Load preprocessed AnnData with cell_sentences
import scanpy as sc
adata = sc.read_h5ad('preprocessed_atac.h5ad')

# Extract features with pretrained EpiAgent
embeddings = extract_features(
    adata,
    model_path='EpiAgent/model/EpiAgent/EpiAgent.pth',
    batch_size=64
)
```

**Fine-tuning for Supervised Cell Type Annotation:**

```bash
python EpiAgent/epiagent/train.py \
  --task annotation \
  --data_path /path/to/atac.h5ad \
  --save_dir ./results \
  --num_epochs 50 \
  --batch_size 32
```

See [EpiAgent/README.md](EpiAgent/README.md) and demo notebooks in [EpiAgent/demo/](EpiAgent/demo/).

---

### Multi-Omics CLIP: Train RNA-ATAC Alignment

**1. Preprocess 10x Multiome Data**

```bash
python multiomics_clip/scripts/preprocess_multiome.py \
  --rna_h5 /path/to/filtered_feature_bc_matrix.h5 \
  --atac_fragments /path/to/fragments.tsv.gz \
  --gene_list scFoundation/model/OS_scRNA_gene_index.19264.tsv \
  --ccre_bed EpiAgent/data/cCRE.bed \
  --output_dir ./preprocessed_data
```

**2. Train CLIP Model**

```bash
python multiomics_clip/scripts/train_clip.py \
  --train_rna ./preprocessed_data/train_rna.h5ad \
  --train_atac ./preprocessed_data/train_atac.h5ad \
  --val_rna ./preprocessed_data/val_rna.h5ad \
  --val_atac ./preprocessed_data/val_atac.h5ad \
  --rna_encoder_path scFoundation/model/models/models.ckpt \
  --atac_encoder_path EpiAgent/model/EpiAgent/EpiAgent.pth \
  --batch_size 64 \
  --save_dir ./checkpoints
```

**3. Evaluate Cross-Modal Retrieval**

```bash
python multiomics_clip/scripts/evaluate_clip.py \
  --checkpoint ./checkpoints/best_model.pt \
  --test_rna ./preprocessed_data/test_rna.h5ad \
  --test_atac ./preprocessed_data/test_atac.h5ad \
  --celltype_column cell_type \
  --output_dir ./evaluation
```

See [multiomics_clip/README.md](multiomics_clip/README.md) for detailed documentation.

---

## Repository Structure

```
.
├── scFoundation/                # RNA-seq foundation model
│   ├── model/                   # Model code and pretrained weights
│   ├── GEARS/                   # Perturbation prediction
│   ├── SCAD/                    # Single-cell drug sensitivity
│   ├── DeepCDR/                 # Cancer drug response
│   ├── enhancement/             # Read depth enhancement
│   ├── annotation/              # Cell type annotation
│   ├── mapping/                 # Cell mapping
│   ├── genemodule/              # Gene module inference
│   └── README.md
│
├── EpiAgent/                    # ATAC-seq foundation model
│   ├── epiagent/                # Core library (pip install epiagent)
│   ├── model/                   # Pretrained model weights
│   ├── data/                    # cCRE reference (hg38)
│   ├── demo/                    # Demo notebooks
│   └── README.md
│
├── multiomics_clip/             # CLIP-style RNA-ATAC integration
│   ├── config.py                # Configuration dataclass
│   ├── model.py                 # MultiOmicsCLIP model
│   ├── encoders.py              # RNA/ATAC encoder wrappers
│   ├── dataset.py               # Paired dataset loader
│   ├── preprocessing/           # Data preprocessing pipeline
│   ├── training/                # Training infrastructure (DDP support)
│   ├── evaluation/              # Retrieval, clustering, batch metrics
│   ├── scripts/                 # Runnable scripts
│   ├── configs/                 # YAML configuration files
│   └── README.md
│
├── data/                        # Shared data resources
├── configs/                     # Configuration files
├── docs/                        # Documentation
│   ├── MultiOmics-CLIP-Pretraining.md
│   ├── Preprocessing.md
│   └── ...
├── preprocessed_data/           # Preprocessed datasets
├── CLAUDE.md                    # Project instructions for AI assistants
└── README.md                    # This file
```

---

## Downstream Tasks

### scFoundation Downstream Applications

| Task | Description | Path | Documentation |
|------|-------------|------|---------------|
| **GEARS** | Perturbation prediction | `scFoundation/GEARS/` | [README](scFoundation/GEARS/README.md) |
| **SCAD** | Single-cell drug sensitivity | `scFoundation/SCAD/` | [README](scFoundation/SCAD/README.md) |
| **DeepCDR** | Cancer drug response (IC50) | `scFoundation/DeepCDR/` | [README](scFoundation/DeepCDR/README.md) |
| **Enhancement** | Read depth enhancement | `scFoundation/enhancement/` | [README](scFoundation/enhancement/README.md) |
| **Annotation** | Cell type annotation | `scFoundation/annotation/` | [README](scFoundation/annotation/README.md) |
| **Mapping** | Cell mapping (organoid → in vivo) | `scFoundation/mapping/` | [README](scFoundation/mapping/README.md) |
| **Gene Module** | Gene module inference | `scFoundation/genemodule/` | [README](scFoundation/genemodule/README.md) |

### EpiAgent Downstream Applications

- **Zero-shot feature extraction**: Unsupervised cell embeddings without fine-tuning
- **Fine-tuned feature extraction**: Task-specific unsupervised embeddings
- **Cell type annotation**: Supervised classification (zero-shot with EpiAgent-B/NT)
- **Data imputation**: Reconstruct missing accessibility signals
- **Reference integration & mapping**: Align query data to reference atlases
- **Perturbation prediction**: Predict cellular responses to stimulations
- **In-silico cCRE knockout**: Model regulatory element perturbations

See [EpiAgent/demo/](EpiAgent/demo/) for Jupyter notebooks demonstrating all tasks.

---

## Key Publications

### scFoundation

**Title:** scFoundation: Large Scale Foundation Model on Single-cell Transcriptomics
**Journal:** *Nature Methods*, 2024
**DOI:** [10.1038/s41592-024-02305-7](https://doi.org/10.1038/s41592-024-02305-7)

```bibtex
@article{scFoundation2024,
  title={scFoundation: Large Scale Foundation Model on Single-cell Transcriptomics},
  journal={Nature Methods},
  year={2024},
  doi={10.1038/s41592-024-02305-7}
}
```

### EpiAgent

**Title:** EpiAgent: foundation model for single-cell epigenomics
**Journal:** *Nature Methods*, 2025
**DOI:** [10.1038/s41592-025-02822-z](https://doi.org/10.1038/s41592-025-02822-z)

```bibtex
@article{Chen2025EpiAgent,
  title={EpiAgent: foundation model for single-cell epigenomics},
  author={Chen, X. and Li, K. and Cui, X. and Wang, Z. and Jiang, Q. and Lin, J. and Li, Z. and Gao, Z. and Hai, L. and Jiang, R.},
  journal={Nature Methods},
  year={2025},
  doi={10.1038/s41592-025-02822-z}
}
```

### Human-scATAC-Corpus

**Title:** Human-scATAC-Corpus: a comprehensive database of scATAC-seq data
**Preprint:** *bioRxiv*, 2025
**DOI:** [10.1101/2025.09.05.674505](https://doi.org/10.1101/2025.09.05.674505)
**Database:** [health.tsinghua.edu.cn/human-scatac-corpus](https://health.tsinghua.edu.cn/human-scatac-corpus)

```bibtex
@article{Chen2025HumanscATACCorpus,
  title={Human-scATAC-Corpus: a comprehensive database of scATAC-seq data},
  author={Chen, X. and Gao, Z. and Li, K. and Wang, Z. and Jiang, Q. and Cui, X. and Li, Z. and Jiang, R.},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.09.05.674505}
}
```

---

## Data Resources

### scFoundation
- **Model Weights:** [SharePoint](https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/EmUQnvZMETlDvoCaBduCNeIBQArcOrd8T8iEpiGofFZ9CQ)
- **Example Data:** [Figshare](https://doi.org/10.6084/m9.figshare.24049200)
- **Gene List:** `scFoundation/model/OS_scRNA_gene_index.19264.tsv`
- **API Access:** [aigp.biomap.com](https://aigp.biomap.com/)

### EpiAgent
- **Pretrained Models:** [Google Drive](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX)
- **Human-scATAC-Corpus:** [health.tsinghua.edu.cn/human-scatac-corpus](https://health.tsinghua.edu.cn/human-scatac-corpus)
- **cCRE Reference (hg38):** `EpiAgent/data/cCRE.bed`

---

## Requirements

### Core Dependencies

**scFoundation:**
```
torch>=2.0, einops, numpy, pandas, scipy, scanpy, anndata, local_attention
```

**EpiAgent:**
```
torch>=2.0, transformers, flash-attn==2.5.8, torch_geometric, scanpy, anndata,
pybedtools, numpy, pandas, scipy
```

**Multi-Omics CLIP:**
```
torch>=2.0, flash-attn==2.5.8, scanpy, anndata, scikit-learn, pybedtools, wandb (optional)
```

### Hardware Recommendations

- **GPU:** NVIDIA GPU with CUDA 11.7+ (FlashAttention requirement)
- **Memory:** 16GB+ RAM for inference, 32GB+ for training
- **Storage:** 50GB+ for pretrained models and example datasets

---

## License

This code is licensed under the **Apache-2.0 License**. See [LICENSE](LICENSE) for details.

**Note:** Use of pretrained model weights may be subject to separate model license agreements.

---

## Contact

### scFoundation
- **API Support:** aigp-support@biomap.com
- **GitHub Issues:** [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation)

### EpiAgent
- **Questions:** xychen20@mails.tsinghua.edu.cn
- **GitHub Issues:** [xy-chen16/EpiAgent](https://github.com/xy-chen16/EpiAgent)

### Multi-Omics CLIP
- Open an issue in this repository for questions about the CLIP integration framework

---

## Acknowledgments

This project builds upon and integrates:

- **[scFoundation](https://github.com/biomap-research/scFoundation)** - RNA foundation model
- **[EpiAgent](https://github.com/xy-chen16/EpiAgent)** - ATAC foundation model
- **[OpenCLIP](https://github.com/mlfoundations/open_clip)** - CLIP training framework inspiration
- **[xTrimoGene](https://proceedings.neurips.cc/paper_files/paper/2023/hash/db68f1c25678f72561ab7c97ce15d912-Abstract-Conference.html)** - scFoundation architecture basis
- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** - Efficient attention implementation

**Third-party software used:**
- PyTorch, PyTorch Lightning, DeepSpeed
- Scanpy, scvi-tools, anndata
- NumPy, Pandas, Scipy, einops
- bedtools, pybedtools

Thanks to all contributors and maintainers of these projects!

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

---

## Development Status

This repository is under active development. The multi-omics CLIP framework is currently in testing phase. Please report any issues via GitHub Issues.
