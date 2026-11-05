"""MUTUAL end to end: preprocessed cells -> 256-d embeddings in the shared RNA / ATAC space.

    from mutual import MUTUAL
    m = MUTUAL.from_pretrained("/path/to/MUTUAL_model", seed=1, device="cuda")
    z_rna  = m.embed_rna(X_log1p)        # X_log1p: log1p(CP10k) rows over the 19,264 scFoundation genes
    z_atac = m.embed_atac(sentences)     # sentences: EpiAgent token-id arrays, highest TF-IDF first

Model directory layout (the checkpoint bundle; see README):
    seed{1,2,3}/step_400000.pt          trained heads (one file per training seed)
    fit/model.npz                        corpus-fitted PCA (RNA) / LSI (ATAC)
    raw/selection.npz                    raw-view panels (4,000 HVGs, 10,000 cCREs)
    reference/OS_scRNA_gene_index.19264.tsv, cCRE.bed, cCRE_document_frequency.npy
    foundation/models1.ckpt              scFoundation (key 'gene')     -- or pass scf_ckpt=
    foundation/pretrained_EpiAgent.pth   EpiAgent                      -- or pass epiagent_ckpt=
"""
from __future__ import annotations

import os

import numpy as np

from .foundation import FoundationModels
from .heads import MUTUALHeads
from .views import LinearViews

STEP = 400000


def bundle_paths(model_dir, seed=1, scf_ckpt=None, epiagent_ckpt=None):
    d = os.path.abspath(model_dir)
    p = {"heads": f"{d}/seed{seed}/step_{STEP:06d}.pt", "fit": f"{d}/fit/model.npz",
         "selection": f"{d}/raw/selection.npz", "gene_list": f"{d}/reference/OS_scRNA_gene_index.19264.tsv",
         "ccre_bed": f"{d}/reference/cCRE.bed", "ccre_df": f"{d}/reference/cCRE_document_frequency.npy",
         "scf_ckpt": scf_ckpt or f"{d}/foundation/models1.ckpt",
         "epiagent_ckpt": epiagent_ckpt or f"{d}/foundation/pretrained_EpiAgent.pth"}
    return p


class MUTUAL:
    def __init__(self, heads, views, foundation, paths):
        self.heads, self.views, self.fm, self.paths = heads, views, foundation, paths

    @classmethod
    def from_pretrained(cls, model_dir, seed=1, device="cuda", scf_ckpt=None, epiagent_ckpt=None,
                        modalities=("rna", "atac"), verbose=True):
        """modalities: load only the foundation model(s) you need (scFoundation ~0.5 GB, EpiAgent ~3 GB of RAM)."""
        p = bundle_paths(model_dir, seed, scf_ckpt, epiagent_ckpt)
        for k in ("heads", "fit", "selection"):
            if not os.path.exists(p[k]):
                raise FileNotFoundError(f"{k}: {p[k]}")
        fm = FoundationModels(p["scf_ckpt"] if "rna" in modalities else None,
                              p["epiagent_ckpt"] if "atac" in modalities else None, device=device, verbose=verbose)
        return cls(MUTUALHeads.from_checkpoint(p["heads"], device), LinearViews(p["fit"], p["selection"]), fm, p)

    def embed_rna(self, X, return_inputs=False):
        """X: log1p(CP10k) rows [n, 19264] (scipy.sparse or dense). -> float32 [n, 256]."""
        feat = self.fm.rna_features(X)
        pca, raw = self.views.rna(X)
        z = self.heads.encode("rna", feat, pca, raw)
        return (z, {"scfoundation": feat, "pca": pca, "raw": raw}) if return_inputs else z

    def embed_atac(self, sentences, return_inputs=False):
        """sentences: list of EpiAgent token-id arrays. -> float32 [n, 256]."""
        feat = self.fm.atac_features(sentences)
        lsi, raw = self.views.atac(sentences)
        z = self.heads.encode("atac", feat, lsi, raw)
        return (z, {"epiagent": feat, "lsi": lsi, "raw": raw}) if return_inputs else z

    def embed_from_features(self, modality, fm_feature, linear_summary, raw):
        """Heads only, for precomputed foundation features (rna: 3,072-d scFoundation, atac: 512-d EpiAgent T*)."""
        return self.heads.encode(modality, fm_feature, linear_summary, raw)


def cross_modal_metrics(z_rna, z_atac, labels=None):
    """Paired cells (row i of z_rna and z_atac = the same cell): FOSCTTM (mean fraction of cells closer than the true
    partner, both directions; 0 = perfect, 0.5 = random), top-1 partner retrieval, and if labels are given the
    accuracy of transferring the label of the nearest cell of the other modality."""
    S = np.asarray(z_rna, np.float64) @ np.asarray(z_atac, np.float64).T
    n = S.shape[0]
    d = np.diag(S)
    fos = 0.5 * (((S > d[:, None]).sum(1) / (n - 1)).mean() + ((S > d[None, :]).sum(0) / (n - 1)).mean())
    out = {"n_cells": int(n), "foscttm": float(fos), "r1_rna_to_atac": float((S.argmax(1) == np.arange(n)).mean()),
           "r1_atac_to_rna": float((S.argmax(0) == np.arange(n)).mean())}
    if labels is not None:
        y = np.asarray(labels)
        out["label_transfer_rna_to_atac"] = float((y[S.argmax(1)] == y).mean())
        out["label_transfer_atac_to_rna"] = float((y[S.argmax(0)] == y).mean())
    return out
