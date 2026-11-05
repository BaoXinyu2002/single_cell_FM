"""Embed RNA and / or ATAC cells with MUTUAL.

  # paired 10x Multiome sample (the usual case)
  python scripts/mutual_embed.py --model_dir MODEL_DIR \
      --rna filtered_feature_bc_matrix.h5 --atac_fragments atac_fragments.tsv.gz --out sample.npz

  # RNA only / ATAC only (an ATAC-only run should give the cell barcodes, else every barcode with >= 200 fragments
  # in the fragments file is treated as a cell)
  python scripts/mutual_embed.py --model_dir MODEL_DIR --rna counts.h5ad --out rna.npz
  python scripts/mutual_embed.py --model_dir MODEL_DIR --atac_fragments fragments.tsv.gz --barcodes cells.txt --out atac.npz

  # ATAC as a cell x peak count matrix instead of fragments (var_names chr:start-end, hg38)
  python scripts/mutual_embed.py --model_dir MODEL_DIR --atac_counts peaks.h5ad --out atac.npz

Output (.npz): rna_barcodes, rna_embedding [n, 256] float32, atac_barcodes, atac_embedding [n, 256] float32 (unit
norm; cosine = dot product), plus model_seed / model_step. When both modalities are given, cells are restricted to the
barcodes present in both (after QC) and rows are in the same order.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mutual import MUTUAL  # noqa: E402
from mutual import preprocess as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True, help="MUTUAL checkpoint bundle (see README)")
    ap.add_argument("--seed", type=int, default=1, choices=(1, 2, 3), help="which of the 3 trained seeds")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--scf_ckpt", default=None, help="scFoundation models1.ckpt (default MODEL_DIR/foundation/)")
    ap.add_argument("--epiagent_ckpt", default=None, help="pretrained_EpiAgent.pth (default MODEL_DIR/foundation/)")
    ap.add_argument("--rna", help="10x filtered_feature_bc_matrix.h5, or .h5ad of raw UMI counts (var_names = symbols)")
    ap.add_argument("--atac_fragments", help="10x atac_fragments.tsv.gz (hg38)")
    ap.add_argument("--atac_counts", help=".h5ad cell x peak (or cCRE) counts, var_names chr:start-end (hg38)")
    ap.add_argument("--barcodes", help="text file, one cell barcode per line: restrict to these cells")
    ap.add_argument("--min_genes", type=int, default=500, help="RNA QC: min detected genes per cell")
    ap.add_argument("--min_cells", type=int, default=3, help="RNA QC: min cells per gene")
    ap.add_argument("--min_fragments", type=int, default=200, help="ATAC QC: min fragment counts in cCREs per cell")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not (a.rna or a.atac_fragments or a.atac_counts):
        ap.error("give --rna and/or --atac_fragments / --atac_counts")
    t0 = time.time()
    mods = (["rna"] if a.rna else []) + (["atac"] if (a.atac_fragments or a.atac_counts) else [])
    m = MUTUAL.from_pretrained(a.model_dir, seed=a.seed, device=a.device, scf_ckpt=a.scf_ckpt,
                               epiagent_ckpt=a.epiagent_ckpt, modalities=mods)
    want = None if not a.barcodes else [l.strip() for l in open(a.barcodes) if l.strip()]
    out = {"model_seed": a.seed, "model_step": m.heads.step}
    info = {}

    rna_bc = X = None
    if a.rna:
        ad_ = P.read_rna(a.rna)
        if want is not None:
            ad_ = ad_[ad_.obs_names.isin(want)].copy()
        rna_bc, X, n_hit = P.rna_log1p(ad_, P.load_gene_list(m.paths["gene_list"]), a.min_genes, a.min_cells)
        info["rna"] = {"cells_in": int(ad_.n_obs), "cells_after_qc": len(rna_bc), "scfoundation_genes_found": n_hit}
        print(f"[rna] {len(rna_bc)} cells after QC, {n_hit} of 19,264 scFoundation genes in the data", flush=True)

    atac_bc = sents = None
    if a.atac_fragments or a.atac_counts:
        cc = P.CCRE(m.paths["ccre_bed"])
        if a.atac_fragments:
            restrict = want if want is not None else (list(rna_bc) if rna_bc is not None else None)
            names, M = P.fragments_to_ccre_counts(a.atac_fragments, cc, barcodes=restrict)
        else:
            import anndata as ad
            A = ad.read_h5ad(a.atac_counts)
            if want is not None:
                A = A[A.obs_names.isin(want)].copy()
            names, M = np.asarray(A.obs_names.astype(str)), P.peaks_to_ccre_counts(A.X, A.var_names, cc)
        keep, sents = P.ccre_counts_to_sentences(M, np.load(m.paths["ccre_df"]), a.min_fragments)
        atac_bc = names[keep]
        L = np.array([len(s) for s in sents])
        info["atac"] = {"cells_in": len(names), "cells_after_qc": len(atac_bc),
                        "median_ccres_per_cell": float(np.median(L)) if len(L) else 0.0}
        print(f"[atac] {len(atac_bc)} cells after QC, median {info['atac']['median_ccres_per_cell']:.0f} cCREs/cell",
              flush=True)

    if rna_bc is not None and atac_bc is not None:          # paired: keep the common barcodes, same row order
        in_atac = set(atac_bc)
        common = [b for b in rna_bc if b in in_atac]
        ri = {b: i for i, b in enumerate(rna_bc)}
        ai = {b: i for i, b in enumerate(atac_bc)}
        X = X[[ri[b] for b in common]]
        sents = [sents[ai[b]] for b in common]
        rna_bc = atac_bc = np.asarray(common)
        info["paired_cells"] = len(common)
        print(f"[pair] {len(common)} cells with both modalities", flush=True)

    if rna_bc is not None:
        out["rna_barcodes"], out["rna_embedding"] = rna_bc, m.embed_rna(X)
    if atac_bc is not None:
        out["atac_barcodes"], out["atac_embedding"] = atac_bc, m.embed_atac(sents)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    info["seconds"] = round(time.time() - t0, 1)
    info["device"] = a.device
    print(json.dumps(info))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
