"""Raw 10x Multiome data -> MUTUAL model inputs.

RNA   raw UMI counts -> QC (min genes per cell, min cells per gene) -> normalise to 10,000 per cell -> log1p ->
      aligned to the 19,264 scFoundation genes by symbol (missing genes = 0). Same steps and defaults as the
      training-time pipeline (multiomics_clip/preprocessing/preprocess_rna.py).
ATAC  fragments (or a peak / cCRE count matrix) -> cell x cCRE counts over EpiAgent's 1,355,445 hg38 cCREs
      -> QC (min fragments per cell) -> EpiAgent global TF-IDF (corpus document frequencies) -> cell sentence =
      the cell's accessible cCREs as token ids (cCRE.bed row + 4), highest TF-IDF first. Same steps as the training
      pipeline (multiomics_clip/preprocessing/preprocess_atac.py + EpiAgent's global_TFIDF / tokenization). bedtools
      is replaced by an exact interval lookup (cCREs are sorted and non-overlapping), so no external binary is needed.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .foundation import N_CCRE, N_GENES, TOKEN_OFFSET


# ----------------------------------------------------------------------------------------------------------------------
# RNA
# ----------------------------------------------------------------------------------------------------------------------
def load_gene_list(path):
    genes = pd.read_csv(path, sep="\t")["gene_name"].astype(str).tolist()
    assert len(genes) == N_GENES and len(set(genes)) == N_GENES, "expected the 19,264-gene scFoundation list"
    return genes


def read_rna(path):
    """10x filtered_feature_bc_matrix.h5 (Gene Expression features only) or an .h5ad of raw counts (var_names =
    gene symbols)."""
    import scanpy as sc
    if path.endswith(".h5ad"):
        import anndata as ad
        return ad.read_h5ad(path)
    return sc.read_10x_h5(path, gex_only=True)


def rna_log1p(adata, gene_list, min_genes=500, min_cells=3, target_sum=1e4):
    """Raw-count AnnData -> (barcodes, float32 CSR [n, 19264] of log1p(CP10k) on the scFoundation genes).
    min_cells filters genes over the cells given (as in training), which sets the library size used for CP10k."""
    import scanpy as sc
    a = adata.copy()
    if min_genes:
        sc.pp.filter_cells(a, min_genes=min_genes)
    if min_cells:
        sc.pp.filter_genes(a, min_cells=min_cells)
    sc.pp.normalize_total(a, target_sum=target_sum)
    sc.pp.log1p(a)
    names = pd.Index(a.var_names.astype(str))
    first = ~names.duplicated(keep="first")                 # duplicated symbols: the first occurrence is kept
    pos = {g: j for j, g in enumerate(gene_list)}
    src = np.nonzero(first & names.isin(gene_list))[0]
    dst = np.array([pos[g] for g in names[src]], np.int64)
    X = sp.csr_matrix(a.X) if not sp.issparse(a.X) else a.X.tocsr()
    X = X[:, src].tocoo()
    out = sp.csr_matrix((X.data.astype(np.float32), (X.row, dst[X.col])), shape=(a.n_obs, N_GENES))
    out.sum_duplicates()
    out.eliminate_zeros()
    n_hit = len(src)
    return np.asarray(a.obs_names.astype(str)), out, n_hit


# ----------------------------------------------------------------------------------------------------------------------
# cCRE reference
# ----------------------------------------------------------------------------------------------------------------------
class CCRE:
    """EpiAgent's cCRE.bed (hg38, 1,355,445 rows, sorted and non-overlapping within each chromosome)."""

    def __init__(self, bed_path):
        b = pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"],
                        dtype={"chrom": str, "start": np.int64, "end": np.int64})
        assert len(b) == N_CCRE, len(b)
        self.chroms = {}
        for c, g in b.groupby("chrom", sort=False):
            idx = g.index.to_numpy()
            s, e = g.start.to_numpy(), g.end.to_numpy()
            assert (np.diff(idx) == 1).all() and (np.diff(s) > 0).all() and (s[1:] >= e[:-1]).all(), c
            self.chroms[c] = (int(idx[0]), s, e)

    def overlaps(self, chrom, start, end):
        """Intervals (BED, half-open) on one chromosome -> (interval index, cCRE row) for every pair overlapping by
        >= 1 bp, as `bedtools intersect -wa -wb` reports them."""
        if chrom not in self.chroms:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        off, s, e = self.chroms[chrom]
        lo = np.searchsorted(e, start, side="right")         # first cCRE ending after the interval starts
        hi = np.searchsorted(s, end, side="left")            # cCREs starting before the interval ends
        k = np.maximum(hi - lo, 0)
        iv = np.repeat(np.arange(len(start), dtype=np.int64), k)
        first = np.repeat(lo - np.concatenate([[0], np.cumsum(k)[:-1]]), k)
        return iv, off + first + np.arange(k.sum(), dtype=np.int64)


def fragments_to_ccre_counts(fragments_path, ccre, barcodes=None, chunksize=5_000_000, verbose=True):
    """10x fragments.tsv.gz -> (barcodes, float32 CSR [n, 1,355,445]): per cell and cCRE, the summed read count
    (column 5) of the fragments overlapping the cCRE. barcodes: keep only these (in this order; missing -> empty
    rows); None = every barcode in the file (sorted)."""
    keep = None if barcodes is None else pd.Index(np.asarray(barcodes).astype(str))
    seen = {}                                                # barcodes -> row ids when no barcode list is given
    rows_all, cols_all, vals_all = [], [], []
    opener = gzip.open if str(fragments_path).endswith(".gz") else open
    n_frag = 0
    with opener(fragments_path, "rt") as fh:
        reader = pd.read_csv(fh, sep="\t", header=None, comment="#", usecols=[0, 1, 2, 3, 4],
                             names=["chrom", "start", "end", "barcode", "count"],
                             dtype={"chrom": str, "start": np.int64, "end": np.int64, "barcode": str,
                                    "count": np.float32}, chunksize=chunksize)
        for ch in reader:
            n_frag += len(ch)
            if keep is not None:
                rid = keep.get_indexer(ch.barcode.to_numpy())
            else:
                codes, uniq = pd.factorize(ch.barcode.to_numpy())
                ids = np.array([seen.setdefault(b, len(seen)) for b in uniq], np.int64)
                rid = ids[codes]
            ch = ch.assign(rid=rid)
            ch = ch[ch.rid >= 0]
            for c, g in ch.groupby("chrom", sort=False):
                iv, col = ccre.overlaps(c, g.start.to_numpy(), g.end.to_numpy())
                if not len(iv):
                    continue
                rows_all.append(g.rid.to_numpy()[iv])
                cols_all.append(col)
                vals_all.append(g["count"].to_numpy()[iv])
            if verbose:
                print(f"[fragments] {n_frag:,} fragments read", flush=True)
    names = keep if keep is not None else pd.Index(list(seen))
    rows = np.concatenate(rows_all) if rows_all else np.zeros(0, np.int64)
    cols = np.concatenate(cols_all) if cols_all else np.zeros(0, np.int64)
    vals = np.concatenate(vals_all) if vals_all else np.zeros(0, np.float32)
    M = sp.coo_matrix((vals, (rows, cols)), shape=(len(names), N_CCRE), dtype=np.float32).tocsr()
    if keep is None:                                         # barcodes in sorted order
        o = np.argsort(np.asarray(names.astype(str)), kind="stable")
        names, M = names[o], M[o]
    return np.asarray(names.astype(str)), M


def parse_interval(name):
    """'chr1:100-200', 'chr1-100-200' or 'chr1_100_200' -> (chrom, start, end)."""
    s = str(name)
    if ":" in s:
        c, r = s.split(":", 1)
        a, b = r.replace("_", "-").split("-")[:2]
        return c, int(a), int(b)
    sep = "_" if s.count("_") >= 2 else "-"
    c, a, b = s.rsplit(sep, 2)
    return c, int(a), int(b)


def peaks_to_ccre_counts(peak_counts, peak_names, ccre):
    """cell x peak counts -> float32 CSR [n, 1,355,445]: a cCRE receives the counts of every peak that overlaps it
    (multiomics_clip map_peaks_to_ccres followed by counts @ mapping). Peaks must be hg38."""
    iv = [parse_interval(p) for p in peak_names]
    df = pd.DataFrame(iv, columns=["chrom", "start", "end"])
    pr, cr = [], []
    for c, g in df.groupby("chrom", sort=False):
        i, col = ccre.overlaps(c, g.start.to_numpy(), g.end.to_numpy())
        pr.append(g.index.to_numpy()[i])
        cr.append(col)
    pr, cr = np.concatenate(pr), np.concatenate(cr)
    Mp = sp.csr_matrix((np.ones(len(pr), np.float32), (pr, cr)), shape=(len(peak_names), N_CCRE))
    return sp.csr_matrix(sp.csr_matrix(peak_counts, dtype=np.float32) @ Mp)


def ccre_counts_to_sentences(counts, document_frequency, min_fragments=200):
    """cell x cCRE counts -> (kept row indices, list of token-id arrays). QC keeps cells whose summed counts are
    >= min_fragments. Then EpiAgent global TF-IDF, tf-idf = log1p(1e4 * x / rowsum / df), and the sentence = nonzero
    cCREs sorted by decreasing tf-idf (np.argsort default kind, as epiagent.tokenization) + 4."""
    X = sp.csr_matrix(counts, dtype=np.float32)
    X.sum_duplicates()
    X.eliminate_zeros()
    tot = np.asarray(X.sum(axis=1)).ravel()
    keep = np.nonzero(tot >= min_fragments)[0]
    X = X[keep]
    X.sort_indices()
    # EpiAgent global_TFIDF, operation for operation (dtypes included): rows / rowsum (float32), then / df (float64)
    row_inv = sp.csr_matrix(np.reciprocal(X.sum(axis=1).A.flatten()))
    X = X.multiply(row_inv.T)
    X = sp.csr_matrix(X.multiply(sp.csr_matrix(np.reciprocal(np.asarray(document_frequency)))))
    X.data = np.log1p(10000 * X.data)
    X.sort_indices()
    sents = []
    for i in range(X.shape[0]):
        a, b = X.indptr[i], X.indptr[i + 1]
        idx, val = X.indices[a:b], X.data[a:b]
        nz = val != 0
        idx, val = idx[nz], val[nz]
        sents.append((idx[np.argsort(-val)] + TOKEN_OFFSET).astype(np.int64))
    return keep, sents


def sentences_from_strings(cell_sentences):
    """obs['cell_sentences'] strings as written by epiagent.tokenization ('[12,5,...]') -> list of int arrays."""
    import json
    return [np.asarray(json.loads(s.decode() if isinstance(s, bytes) else s), np.int64) for s in cell_sentences]
