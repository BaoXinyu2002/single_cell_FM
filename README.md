# MUTUAL

MUTUAL (multiome transfer in a unified aligned latent space) maps single-cell RNA profiles and single-cell ATAC
profiles into one 256-dimensional embedding space. An RNA-only cell and an ATAC-only cell of the same state land close
together, so the space supports cross-modal matching, label transfer and joint analysis. MUTUAL is built on two frozen
foundation models, scFoundation (RNA) and EpiAgent (ATAC), plus corpus-fitted linear summaries and small trained heads.
It was trained on 3,009,609 paired 10x Multiome cells (400 dataset entries, 385 distinct libraries, from 50 studies;
human, hg38).

This repository holds the model code, the inference pipeline from raw 10x Multiome files, a runnable example with
data, and a fidelity check. **The model weights are not in the repository** (see [Model files](#1-model-files)).

| | |
|---|---|
| Model files | trained heads (3 seeds) + fitted PCA/LSI + feature panels: `MUTUAL_model.tar.gz`; architecture code: `mutual/` |
| Sample programs | `examples/run_example.py` (Python API walk-through), `scripts/mutual_embed.py` (command line) |
| Example data | `examples/data/`: 104 cells of a public normal-breast Multiome library, raw RNA counts + ATAC fragments |
| Expected output | `examples/expected_metrics.json`; formats in [Expected output](#5-expected-output) |
| Compute | GPU optional; see [Compute requirements](#6-compute-requirements) |

---

## 1. Model files

### MUTUAL weights (not in this repository)

`MUTUAL_model.tar.gz`: 255 MB packed, 312 MB unpacked, md5 `ca3282cfce1f2e480e234c796db07719`. A public download link
will be added here. Lab members can copy it from Turbo: `/nfs/turbo/umms-drjieliu1/usr/xinyubao/MUTUAL_model.tar.gz`.

```bash
tar -xzf MUTUAL_model.tar.gz                  # -> MUTUAL_model/
(cd MUTUAL_model && md5sum -c MD5SUMS)
```

```
MUTUAL_model/                   pass this folder as MODEL_DIR / model_dir
  seed1/ seed2/ seed3/          one trained model per training seed (the paper reports all three)
    step_400000.pt              trained heads: {step, knobs, state, config}; 6,226,099 parameters (fp16)
  fit/model.npz                 corpus-fitted linear summaries: RNA PCA-256 (V_rna, rna_mu/sd/mz), ATAC LSI-256
                                (atac_feats, atac_idf, V_atac)
  raw/selection.npz             raw-view panels: 4,000 HVGs (columns of the 19,264-gene list), 10,000 cCREs (+ idf)
  reference/                    OS_scRNA_gene_index.19264.tsv (scFoundation gene list), cCRE.bed and
                                cCRE_document_frequency.npy (EpiAgent cCREs, hg38)
  foundation/                   empty: put the two foundation-model checkpoints here (see below)
  README.md, MD5SUMS
```

### Frozen foundation models (download from their authors)

MUTUAL does not redistribute the foundation-model weights. Download them from the original releases:

| model | file | source | sha256 |
|---|---|---|---|
| scFoundation | `models1.ckpt` (key `gene`, 1,432,587,886 bytes) | [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation) (SharePoint link in its README) | `9f40bf324d3d0084c4b288d06f5af4fddd12206e2a3f022551d12e89e33a0ea9` |
| EpiAgent | `pretrained_EpiAgent.pth` (5,801,822,387 bytes) | [xy-chen16/EpiAgent](https://github.com/xy-chen16/EpiAgent) (Google Drive link in its README) | `7fffb9225c3b4b33d005c250c2b8d5990635221397ddf1a1d1004ae117a0a645` |

Check a download with `sha256sum models1.ckpt pretrained_EpiAgent.pth`. Put both files in `MUTUAL_model/foundation/`
under these names; every script then finds them. Alternatively, pass their paths: `scf_ckpt=` / `epiagent_ckpt=` in
`MUTUAL.from_pretrained`, or `--scf_ckpt` / `--epiagent_ckpt` for `scripts/mutual_embed.py` and
`examples/run_example.py`. Only the encoders are used (scFoundation 99.9 M parameters, EpiAgent 754.9 M); the decoders
in the checkpoint files are ignored. Their use follows the scFoundation and EpiAgent licences.

### Architecture (code: `mutual/`)

```
RNA counts --QC, CP10k, log1p--> 19,264-gene row --scFoundation (frozen)--> 3,072-d -+
                                                 --PCA-256 (fitted)-----------> 256-d -+-> FM view MLP  -> 128-d -+
                                                 --4,000 HVG values----------> raw view bag encoder -> 128-d -+-> 256-d
ATAC fragments --cCRE counts, TF-IDF--> sentence --EpiAgent [CLS] (frozen)--> 512-d -+                              (unit norm)
                                                 --LSI-256 (fitted)-----------> 256-d -+-> FM view MLP  -> 128-d -+
                                                 --10,000 cCREs accessible---> raw view bag encoder -> 128-d -+-> 256-d
```

| file | content |
|---|---|
| `mutual/preprocess.py` | raw counts / fragments / peaks -> model inputs (same steps as training) |
| `mutual/foundation.py` | scFoundation + EpiAgent encoders in plain PyTorch, loaded from the original checkpoints |
| `mutual/views.py` | PCA / LSI fold-in and raw-view panels |
| `mutual/heads.py` | the trained MUTUAL heads (two views per modality, fused to 256-d) |
| `mutual/model.py` | `MUTUAL.from_pretrained(...)`, `embed_rna`, `embed_atac`, `cross_modal_metrics` |

Parameters: 891.4 M in total (scFoundation encoder 99.9 M + EpiAgent 754.9 M + fitted PCA/LSI 26.7 M + trained heads
9.8 M, of which 6.2 M are used at inference). The model configuration is stored in each checkpoint (`knobs`) and is
checked on load (`mutual/heads.py: EXPECTED_KNOBS`).

---

## 2. Sample programs

**Python API** (`examples/run_example.py` does exactly this on the bundled data):

```python
import numpy as np
from mutual import MUTUAL, cross_modal_metrics
from mutual import preprocess as P

model = MUTUAL.from_pretrained("MUTUAL_model", seed=1, device="cuda")   # or "cpu"
# foundation models elsewhere: add scf_ckpt="/path/models1.ckpt", epiagent_ckpt="/path/pretrained_EpiAgent.pth"

rna = P.read_rna("filtered_feature_bc_matrix.h5")                     # 10x .h5 or .h5ad of raw counts
barcodes, X, _ = P.rna_log1p(rna, P.load_gene_list(model.paths["gene_list"]))
ccre = P.CCRE(model.paths["ccre_bed"])
_, counts = P.fragments_to_ccre_counts("atac_fragments.tsv.gz", ccre, barcodes=barcodes)
keep, sentences = P.ccre_counts_to_sentences(counts, np.load(model.paths["ccre_df"]))

z_rna = model.embed_rna(X[keep])           # [n, 256] float32, unit norm
z_atac = model.embed_atac(sentences)       # [n, 256] float32, same space
print(cross_modal_metrics(z_rna, z_atac))
```

**Command line** (any 10x Multiome sample, or one modality alone):

```bash
python scripts/mutual_embed.py --model_dir MUTUAL_model \
    --rna filtered_feature_bc_matrix.h5 --atac_fragments atac_fragments.tsv.gz --out sample.npz
python scripts/mutual_embed.py --model_dir ... --rna counts.h5ad --out rna_only.npz
python scripts/mutual_embed.py --model_dir ... --atac_fragments fragments.tsv.gz --barcodes cells.txt --out atac_only.npz
python scripts/mutual_embed.py --model_dir ... --atac_counts peaks.h5ad --out atac_only.npz     # cell x peak counts
```

Options: `--seed {1,2,3}`, `--device {cuda,cpu}`, `--scf_ckpt`, `--epiagent_ckpt`, `--min_genes 500`, `--min_cells 3`,
`--min_fragments 200` (training defaults).

---

## 3. Example data and input format

`examples/data/` (28 MB) is one public normal-breast 10x Multiome library, GEO GSE244594 (GSM7821209 RNA,
GSM7821193 ATAC, "KTB pool-3"). Breast tissue was **not** in the training corpus. It holds 104 annotated cells, up to 20
per cell type (Adipocytes, Endothelial, Epithelial, Fibroblasts, Macrophages, T-cells):

| file | content |
|---|---|
| `breast_pool3_rna_counts.h5ad` | raw UMI counts, all 36,601 Gene Expression features, var_names = gene symbols |
| `breast_pool3_atac_fragments.tsv.gz` | the 10x ATAC fragments of the same cells (chrom, start, end, barcode, count) |
| `breast_pool3_labels.csv` | barcode, cell_type, donor (the data authors' annotation) |

`examples/run_full_sample.sh MODEL_DIR` downloads the whole library from GEO (824 MB) and embeds every cell.
`examples/make_example_data.py` documents how the subset was cut.

**Inputs the pipeline accepts**

- RNA: raw UMI counts, gene symbols (HGNC, as in Cell Ranger GRCh38 references). QC removes cells with fewer than 500
  detected genes and genes detected in fewer than 3 cells. Counts are then normalised to 10,000 per cell, log1p
  transformed and aligned to the 19,264 scFoundation genes; genes missing from the data are left at 0.
  Gene filtering runs over the cells you pass in, so embed a whole sample at once rather than cell by cell.
- ATAC: hg38 only (lift hg19 data over first).
  - Fragments, or a cell x peak count matrix, are counted on EpiAgent's 1,355,445 cCREs. A count is any overlap of at
    least 1 bp; this reproduces `bedtools intersect`.
  - Cells need at least 200 fragment counts in cCREs.
  - EpiAgent global TF-IDF uses the corpus document frequencies.
  - Cell sentence: the accessible cCREs ordered by decreasing TF-IDF.
  - A cell with more than 8,190 accessible cCREs is split into interleaved subsets. Each subset goes through EpiAgent
    separately and the [CLS] outputs are averaged (the "T*" rule used in training).
- Precomputed inputs work too: `model.embed_atac(sentences)` takes EpiAgent cell sentences (`obs['cell_sentences']`
  strings, parsed with `P.sentences_from_strings`), and `model.embed_from_features(...)` runs the heads alone on cached
  foundation-model features.

---

## 4. Environment and commands

```bash
git clone <this repo> && cd MUTUAL
conda env create -f environment.yml && conda activate mutual      # or: pip install -r requirements.txt
# GPU: install the torch build for your CUDA first, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128
tar -xzf MUTUAL_model.tar.gz            # weights, see Model files; then put the two foundation models in MUTUAL_model/foundation/
python examples/run_example.py --model_dir MUTUAL_model                 # GPU if available
python examples/run_example.py --model_dir MUTUAL_model --device cpu
```

Tested with Python 3.10, torch 2.9.1 (CUDA 12.8), numpy 1.26.4, scipy 1.15.3, pandas 2.3.3, anndata 0.11.4,
scanpy 1.11.5, h5py 3.16.0. No flash-attn and no bedtools are needed. Keep numpy below 2.0: numpy 2 may order cCREs
with equal TF-IDF differently, which slightly changes the EpiAgent input.

---

## 5. Expected output

`run_example.py` writes `examples/output/`:

- `embeddings.npz` holds the following arrays:
  - `barcodes` [104];
  - `cell_type` [104];
  - `rna_embedding` [104, 256] float32;
  - `atac_embedding` [104, 256] float32.

  Row *i* of the two matrices is the same cell. Embeddings are unit-norm, so the cosine similarity is the dot product.
  Dimensions 0–127 come from the foundation-model view and dimensions 128–255 from the raw-data view, each scaled by
  sqrt(0.5).
- `metrics.json` compares each cell's RNA embedding with all ATAC embeddings:
  - `foscttm`: the mean fraction of cells closer than the true partner, averaged over both directions. 0 is perfect
    and 0.5 is chance.
  - `r1_rna_to_atac`, `r1_atac_to_rna`: the fraction of cells whose nearest cross-modal neighbour is their own partner.
    Chance is 1/104.
  - `label_transfer_*`: accuracy of copying the cell type of the nearest cell in the other modality.

  The script then compares these values with `examples/expected_metrics.json` and prints PASS or DIFFERENT.

Expected values (seed 1):

EXPECTED_TABLE

`scripts/mutual_embed.py` writes one `.npz` holding `rna_barcodes`, `rna_embedding`, `atac_barcodes`,
`atac_embedding`, `model_seed` and `model_step`. When both modalities are given, the rows are paired: only cells
present in both modalities after QC are kept, in the same order.

---

## 6. Compute requirements

COMPUTE_TABLE

- GPU: any CUDA GPU with at least 8 GB. scFoundation runs under bf16 autocast where the GPU supports bf16; elsewhere
  (e.g. V100) it falls back to fp32. EpiAgent runs under fp16 autocast. Batches adapt to GPU memory: on an
  out-of-memory error a batch is split in two and retried.
- RAM: about 10 GB to load both foundation models (EpiAgent's cCRE embedding table alone is 2.8 GB). Load one
  modality with `MUTUAL.from_pretrained(..., modalities=("rna",))` to save memory.
- Run time grows with ATAC depth. Deep nuclei (> 8,190 accessible cCREs) need several EpiAgent passes, and the
  example's breast nuclei are deep (median about 14,000).

---

## 7. Fidelity to the training code

`tools/check_against_reference.py` checks this package against the training-time code (lab-internal reference arrays).
It starts from the raw GEO files of one breast library (695 annotated cells) and compares the result with the
embeddings the training-time code produced for the same cells:

FIDELITY_TABLE

Preprocessing reproduces the training inputs exactly:
- RNA rows are bit-identical (maximum difference 0).
- ATAC cell sentences have the same tokens in the same order, for 695 of 695 cells.

## Notes

- The three seeds are three independently trained heads on the same inputs. The paper reports metrics for each seed
  and does not average embeddings. `--seed 1` is the default.
- This is the leave-Millman-out model: the Millman SC-islet study (GSE199636) is excluded from training, as is one
  eye study (GSE310245).
- Embeddings of different samples share one space with no batch correction. The paper's benchmarks report each
  metric twice: on the raw embeddings, and after donor centring (each donor's mean embedding subtracted).
- `mutual/foundation.py` re-implements the scFoundation and EpiAgent encoders so their released weights load without
  their packages. Their weights are not redistributed here or in `MUTUAL_model.tar.gz`; their use follows the
  scFoundation and EpiAgent licences.
