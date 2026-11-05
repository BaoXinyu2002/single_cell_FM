"""Fidelity check: this package, run from RAW 10x files, vs the embeddings computed by the training-time code for the
same cells (lab-internal reference arrays, one row per cell of a reference meta table).

  python tools/check_against_reference.py --model_dir MODEL_DIR --rna_h5 GSM7821209_KTB_pool-3_filtered_feature_bc_matrix.h5 \
      --fragments pool3_fragments.tsv.gz --barcodes pool3_barcodes.txt \
      --ref_dir .../emb/XCAT_s{seed}/step_400000 --ref_set breast --ref_meta breast_meta.csv --ref_prefix Pool3: \
      --out check.json

Pass rule (the one used for every re-implementation in the project): median per-cell cosine >= 0.9999 and 1st
percentile >= 0.999, for every seed and modality.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mutual import MUTUAL, bundle_paths, cross_modal_metrics  # noqa: E402
from mutual import preprocess as P  # noqa: E402
from mutual.heads import MUTUALHeads  # noqa: E402


def cos_rows(A, B):
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)).clip(1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rna_h5", required=True)
    ap.add_argument("--fragments", required=True)
    ap.add_argument("--barcodes", required=True)
    ap.add_argument("--ref_dir", required=True, help="contains {seed} placeholder")
    ap.add_argument("--ref_set", required=True)
    ap.add_argument("--ref_meta", required=True)
    ap.add_argument("--ref_prefix", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    bcs = [l.strip() for l in open(a.barcodes) if l.strip()]
    m = MUTUAL.from_pretrained(a.model_dir, seed=1, device=a.device)
    rna = P.read_rna(a.rna_h5)
    rbc, X, _ = P.rna_log1p(rna, P.load_gene_list(m.paths["gene_list"]))      # whole sample, as in training
    pos = pd.Series(np.arange(len(rbc)), index=rbc)
    X = X[pos.loc[bcs].to_numpy()]
    names, M = P.fragments_to_ccre_counts(a.fragments, P.CCRE(m.paths["ccre_bed"]), barcodes=bcs)
    keep, sents = P.ccre_counts_to_sentences(M, np.load(m.paths["ccre_df"]))
    assert len(keep) == len(bcs)
    _, rin = m.embed_rna(X, return_inputs=True)
    _, ain = m.embed_atac(sents, return_inputs=True)
    t_fm = time.time() - t0
    meta = pd.read_csv(a.ref_meta)
    ri = pd.Series(np.arange(len(meta)), index=meta.cell_barcode.astype(str)).loc[[a.ref_prefix + b for b in bcs]].to_numpy()
    res = {"cells": len(bcs), "device": a.device, "deep_cells_gt_8190": int(sum(len(s) > 8190 for s in sents)),
           "seconds_features": round(t_fm, 1)}
    for seed in (1, 2, 3):
        h = MUTUALHeads.from_checkpoint(bundle_paths(a.model_dir, seed)["heads"], a.device)
        zr = h.encode("rna", rin["scfoundation"], rin["pca"], rin["raw"])
        za = h.encode("atac", ain["epiagent"], ain["lsi"], ain["raw"])
        d = a.ref_dir.format(seed=seed)
        for mod, z in (("rna", zr), ("atac", za)):
            ref = np.load(f"{d}/{a.ref_set}_{mod}.npy", mmap_mode="r")[ri].astype(np.float32)
            c = cos_rows(z, ref)
            res[f"seed{seed}_{mod}"] = {"median_cos": float(np.median(c)), "p1_cos": float(np.percentile(c, 1)),
                                       "min_cos": float(c.min()), "pass": bool(np.median(c) >= 0.9999 and np.percentile(c, 1) >= 0.999)}
        rr = np.load(f"{d}/{a.ref_set}_rna.npy", mmap_mode="r")[ri].astype(np.float32)
        ra = np.load(f"{d}/{a.ref_set}_atac.npy", mmap_mode="r")[ri].astype(np.float32)
        res[f"seed{seed}_foscttm"] = {"this_package": cross_modal_metrics(zr, za)["foscttm"],
                                      "reference": cross_modal_metrics(rr, ra)["foscttm"]}
    res["status"] = "PASS" if all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v) else "FAIL"
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
