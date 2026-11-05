"""Corpus-fitted linear summaries and raw-view panels (fixed after fitting on the training corpus; not trained).

  linear summary (concatenated to the FM feature)
    RNA   256-d PCA: the cell's log1p values at the 4,000 panel HVGs, z-scored with the corpus mean / SD, clipped at
          +-10, centred (rna_mz), projected on V_rna [4000, 256].
    ATAC  256-d LSI: the cell's accessible cCREs among the top-100,000 corpus cCREs (binary), TF-IDF
          log1p(1e4 / #selected) * idf, projected on V_atac.T [100000, 256] (first SVD component already dropped).
  raw view (the bag-of-features encoder input)
    RNA   log1p values at the 4,000 panel HVGs.
    ATAC  the cell's tokens at the 10,000 panel cCREs (count x idf; the model only reads accessible / not).
All outputs are float16, the dtype the model was trained on.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class LinearViews:
    def __init__(self, fit_npz, selection_npz):
        z = np.load(fit_npz)
        self.rna_mu, self.rna_sd, self.rna_mz = z["rna_mu"], z["rna_sd"], z["rna_mz"]
        self.V_rna = np.ascontiguousarray(z["V_rna"])                       # [4000, 256]
        self.atac_feats, self.atac_idf = z["atac_feats"].astype(np.int64), z["atac_idf"]
        self.V_atac_T = np.ascontiguousarray(z["V_atac"].T)                 # [100000, 256]
        s = np.load(selection_npz)
        self.hvg = s["hvg"].astype(np.int64)                                # columns of the 19,264-gene panel
        self.ccre = s["ccre"].astype(np.int64)                              # EpiAgent token ids
        self.ccre_idf = s["idf"].astype(np.float32)
        self.lut_raw = self._lut(self.ccre)
        self.lut_lsi = self._lut(self.atac_feats)

    @staticmethod
    def _lut(tokens):
        lut = np.full(int(tokens.max()) + 1, -1, np.int32)
        lut[tokens] = np.arange(len(tokens), dtype=np.int32)
        return lut

    def rna(self, X):
        """X: log1p(CP10k) rows over the 19,264 genes (dense or scipy.sparse). -> (pca f16 [n, 256], raw f16 [n, 4000])."""
        X = X.tocsr() if sp.issparse(X) else np.asarray(X)
        raw = np.asarray(X[:, self.hvg].toarray() if sp.issparse(X) else X[:, self.hvg], np.float32).astype(np.float16)
        n = raw.shape[0]
        pca = np.zeros((n, self.V_rna.shape[1]), np.float32)
        for i in range(0, n, 50000):
            z = (raw[i:i + 50000].astype(np.float64) - self.rna_mu) / self.rna_sd
            np.clip(z, -10.0, 10.0, out=z)
            z -= self.rna_mz
            pca[i:i + 50000] = (z @ self.V_rna).astype(np.float32)
        return pca.astype(np.float16), raw

    def atac(self, sentences):
        """sentences: EpiAgent token-id arrays. -> (lsi f16 [n, 256], raw f16 [n, 10000])."""
        n = len(sentences)
        K = len(self.atac_feats)
        lsi = np.zeros((n, self.V_atac_T.shape[1]), np.float32)
        raw = np.zeros((n, len(self.ccre)), np.float16)
        for c0 in range(0, n, 5000):
            rr = range(c0, min(n, c0 + 5000))
            cols, nsel = [], []
            for i in rr:
                u = np.unique(np.asarray(sentences[i], np.int64))
                c = self.lut_lsi[u[u < len(self.lut_lsi)]]
                c = c[c >= 0]
                cols.append(c)
                nsel.append(len(c))
                t = np.asarray(sentences[i], np.int64)
                t = self.lut_raw[t[t < len(self.lut_raw)]]
                t = t[t >= 0]
                if t.size:
                    raw[i] = (np.bincount(t, minlength=len(self.ccre)).astype(np.float32) * self.ccre_idf).astype(np.float16)
            nsel = np.asarray(nsel, np.int64)
            indptr = np.concatenate([[0], np.cumsum(nsel)])
            idx = np.concatenate(cols).astype(np.int32) if cols else np.zeros(0, np.int32)
            data = np.repeat(np.log1p(1e4 / np.maximum(nsel, 1)), nsel) * self.atac_idf[idx]
            T = sp.csr_matrix((data, idx, indptr), shape=(len(nsel), K))
            lsi[c0:c0 + len(nsel)] = np.asarray(T @ self.V_atac_T, np.float32)
        return lsi.astype(np.float16), raw
