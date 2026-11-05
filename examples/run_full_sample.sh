#!/bin/bash
# Full public sample: download one normal-breast 10x Multiome library from GEO (GSE244594; 21.7 MB RNA .h5 +
# 802 MB ATAC fragments) and embed every cell with MUTUAL through the command-line tool.
#   bash examples/run_full_sample.sh /path/to/MUTUAL_LMO_XCAT_step400k [workdir]
set -euo pipefail
MODEL_DIR=${1:?usage: run_full_sample.sh MODEL_DIR [WORKDIR]}
WORK=${2:-breast_pool3}
GEO=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7821nnn
mkdir -p "$WORK"
[ -s "$WORK/rna.h5" ] || curl -L -o "$WORK/rna.h5" "$GEO/GSM7821209/suppl/GSM7821209_KTB_pool-3_filtered_feature_bc_matrix.h5"
[ -s "$WORK/atac_fragments.tsv.gz" ] || curl -L -o "$WORK/atac_fragments.tsv.gz" "$GEO/GSM7821193/suppl/GSM7821193_KTB_pool-3_atac_fragments.tsv.gz"
python "$(dirname "$0")/../scripts/mutual_embed.py" --model_dir "$MODEL_DIR" \
    --rna "$WORK/rna.h5" --atac_fragments "$WORK/atac_fragments.tsv.gz" --out "$WORK/mutual_embeddings.npz"
