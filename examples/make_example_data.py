"""Build the bundled example (examples/data/) from public GEO files. Run once by the maintainers; users do not need it.

Source: normal-breast 10x Multiome, GEO GSE244594 (sub-series GSE244585), samples GSM7821209 (KTB pool-3,
filtered_feature_bc_matrix.h5) and GSM7821193 (KTB pool-3 ATAC fragments). Breast tissue is NOT in MUTUAL's training
corpus. Cell-type / donor labels: the data authors' annotation released with the study.

Output: up to 20 annotated cells per cell type of the pool (seed 0), 104 cells in total.
  data/breast_pool3_rna_counts.h5ad      raw UMI counts, every Gene Expression feature (var_names = gene symbols)
  data/breast_pool3_atac_fragments.tsv.gz   the 10x ATAC fragments of the same cells
  data/breast_pool3_labels.csv           barcode, cell_type, donor

  python examples/make_example_data.py --rna_h5 GSM7821209_KTB_pool-3_filtered_feature_bc_matrix.h5 \
      --fragments GSM7821193_KTB_pool-3_atac_fragments.tsv.gz --labels labels.csv
labels.csv: barcode,cell_type,donor for the annotated cells of the pool.
"""
import argparse
import gzip
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mutual import preprocess as P  # noqa: E402

PER_TYPE = 20
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna_h5", required=True)
    ap.add_argument("--fragments", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lab = pd.read_csv(a.labels).set_index("barcode")
    rng = np.random.RandomState(SEED)
    pick = []
    for ct in sorted(lab.cell_type.unique()):
        bc = lab.index[lab.cell_type == ct].to_numpy()
        pick.append(bc if len(bc) <= PER_TYPE else rng.choice(bc, PER_TYPE, replace=False))
    pick = np.sort(np.concatenate(pick))
    lab.loc[pick, ["cell_type", "donor"]].reset_index().to_csv(f"{a.out}/breast_pool3_labels.csv", index=False)

    r = P.read_rna(a.rna_h5)
    r.var_names_make_unique()
    r = r[pick].copy()
    r.X = sp.csr_matrix(r.X, dtype=np.float32)
    r.obs = lab.loc[pick, ["cell_type", "donor"]].copy()
    r.var = r.var[[]]
    r.write_h5ad(f"{a.out}/breast_pool3_rna_counts.h5ad", compression="gzip")

    keep = set(pick)
    n = 0
    with gzip.open(a.fragments, "rt") as fi, gzip.open(f"{a.out}/breast_pool3_atac_fragments.tsv.gz", "wt") as fo:
        for line in fi:
            if not line.startswith("#") and line.split("\t", 4)[3] in keep:
                fo.write(line)
                n += 1
    print("cells", len(pick), lab.loc[pick].cell_type.value_counts().to_dict(), "fragments", n)


if __name__ == "__main__":
    main()
