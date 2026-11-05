"""MUTUAL example: load the model, read raw 10x Multiome inputs, run inference, inspect the embeddings.

Data: 104 annotated cells of a normal-breast 10x Multiome sample (examples/data/, public GEO GSE244594: GSM7821209 / GSM7821193;
breast was not in MUTUAL's training corpus). 6 cell types, 5 donors.

  python examples/run_example.py --model_dir /path/to/MUTUAL_model [--device cuda|cpu] [--seed 1]
      [--scf_ckpt models1.ckpt] [--epiagent_ckpt pretrained_EpiAgent.pth]   (default: MODEL_DIR/foundation/)

Writes examples/output/{embeddings.npz, metrics.json} and checks the metrics against examples/expected_metrics.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mutual import MUTUAL, cross_modal_metrics  # noqa: E402
from mutual import preprocess as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--scf_ckpt", default=None, help="scFoundation models1.ckpt (default MODEL_DIR/foundation/)")
    ap.add_argument("--epiagent_ckpt", default=None, help="pretrained_EpiAgent.pth (default MODEL_DIR/foundation/)")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    a = ap.parse_args()
    t0 = time.time()

    # 1. load the model: frozen scFoundation + EpiAgent, corpus-fitted PCA / LSI, trained MUTUAL heads
    model = MUTUAL.from_pretrained(a.model_dir, seed=a.seed, device=a.device, scf_ckpt=a.scf_ckpt,
                                   epiagent_ckpt=a.epiagent_ckpt)

    # 2. read and preprocess the inputs
    labels = pd.read_csv(f"{HERE}/data/breast_pool3_labels.csv").set_index("barcode")
    rna = P.read_rna(f"{HERE}/data/breast_pool3_rna_counts.h5ad")                 # raw UMI counts, gene symbols
    rna_bc, X, n_genes = P.rna_log1p(rna, P.load_gene_list(model.paths["gene_list"]))
    ccre = P.CCRE(model.paths["ccre_bed"])
    atac_bc, counts = P.fragments_to_ccre_counts(f"{HERE}/data/breast_pool3_atac_fragments.tsv.gz", ccre,
                                                 barcodes=rna_bc)                   # cell x cCRE fragment counts
    keep, sentences = P.ccre_counts_to_sentences(counts, np.load(model.paths["ccre_df"]))
    assert len(keep) == len(rna_bc), "every example cell passes ATAC QC"
    print(f"{len(rna_bc)} cells; {n_genes} scFoundation genes in the data; "
          f"median {np.median([len(s) for s in sentences]):.0f} accessible cCREs per cell")

    # 3. inference: one 256-d embedding per cell and modality, in one shared space
    z_rna = model.embed_rna(X)
    z_atac = model.embed_atac(sentences)

    # 4. results: cross-modal matching of the same cells, and cell-type transfer between modalities
    y = labels.loc[rna_bc, "cell_type"].to_numpy()
    met = cross_modal_metrics(z_rna, z_atac, y)
    met["seconds"] = round(time.time() - t0, 1)
    met["device"], met["seed"] = a.device, a.seed
    if a.device.startswith("cuda"):
        import torch
        met["gpu_peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/embeddings.npz", barcodes=rna_bc, cell_type=y, rna_embedding=z_rna, atac_embedding=z_atac)
    json.dump(met, open(f"{a.out}/metrics.json", "w"), indent=1)
    print(json.dumps(met, indent=1))

    if not os.path.exists(f"{HERE}/expected_metrics.json"):
        return
    exp = json.load(open(f"{HERE}/expected_metrics.json"))[f"seed{a.seed}"]
    tol = exp.pop("tolerance")
    bad = {k: (met[k], v) for k, v in exp.items() if abs(met[k] - v) > tol}
    print("CHECK vs expected_metrics.json:", "PASS" if not bad else f"DIFFERENT {bad}")


if __name__ == "__main__":
    main()
