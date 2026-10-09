#!/usr/bin/env python3
"""Replay the `fmtok-v1` frozen-FM token cache as a torch Dataset -- ZERO live FM
forwards.

This is the READ side of the cache whose format lives in `fm_token_cache.py` and whose
layout/parity contract lives in `CACHE_DESIGN.md`.  Nothing here re-implements the
format: every byte comes back through `fm_token_cache.CacheReader`, so there is exactly
one definition of what a shard means.

WHAT IT REPLACES.  `train_filip_combined.py`'s inner loop is

    for _ in range(accum):
        b = next(di)
        out = fm_tokens(model, b, device, include_summary=..., return_ids=...)
        RC.append(out[0]); RT.append(out[1]); ...
    rt, rm, gs = pad_stack3(RT, RM, GS); at, am, cs = pad_stack3(AT, AM, CS)

which costs 33 ms/cell/GPU of FROZEN forward (24 ms scFoundation + 9 ms EpiAgent, both
measured).  With this module the same tensors come from disk:

    ds = CachedTokenDataset(cache_dir, "train")
    dl = DataLoader(ds, batch_sampler=..., collate_fn=ds.collate_fn, num_workers=8)
    for batch in dl:
        batch = move_batch_to_device(batch, device)
        rc, rt, rm, ac, at, am, gs, cs = make_fm_tokens_from_cache(batch)
        G, C = compute_refined(rna_ref, atac_ref, rt, rm, at, am)

`make_fm_tokens_from_cache` returns EXACTLY what `fm_tokens` returns -- the 8-tuple, and
the 10-tuple under `return_ids=True` -- so the swap is one line at the call site.

THREE THINGS THIS MODULE REFUSES TO DO, each for a scar in this project's history:

  * It never derives validity from a token id.  `pad_token_id = 103` IS the real gene
    ABLIM1 (OS_scRNA_gene_index.19264.tsv line 105); the cache stores explicit lengths
    and packs only valid tokens, so validity is arithmetic on offsets and nothing else.
    103 appears here only as a FILL value under `pad_fill="live"`, never as a detector.
  * It never locates the two RNA summary tokens at a global position.  They are the last
    two entries of each PACKED cell slice, at `(n_i - 2, n_i - 1)`, exactly where
    `fm_pool_rna`/`filip_masks` look for them via `vc = (~rm).sum(1)`.  The
    `model.py:215-218` global-slice bug is unrepresentable in a packed layout: there
    is no global last position to read by mistake.
  * It never joins by row position.  Whitelists, labels and every filter go through the
    cache index's `barcode` column, which pass 0 resolved through the dataset's own
    two-level map `bc_all[_rna_keep_orig[_valid_idx]]`.  QC drops SCATTERED rows; this
    project has retracted a result to that bug class.

MASK POLARITY IS `True == PAD`, both modalities, everywhere -- the polarity `fm_tokens`
returns.  The ONLY inversion in the whole pipeline lives inside `ATACRefinerFM.forward`
(`kpm = ~pad_mask`, flash-attn wants True == VALID).  A reader that "helpfully" inverts
reintroduces the bug whose tombstone is `model.py:229-243` (rna_pad=1.0, rna_valid=0.0,
rna_proj_norm=0.0).  `cached_collate` asserts `(~rm).sum(1) == rna_len` before it
returns.

TOKENS ARE RAW ACTIVATIONS.  Not L2-normalised.  FineCLS feeds raw activations to
`FixedSlotPooler` and concatenates a UNIT-NORM cell block before a single Linear -- that
scale mismatch is part of the trained function.  There is no `l2` argument anywhere in
this file, by design (cf. the project's `l2_input_convention_bug`, where `load_set`
defaulted `l2=True`).

MEMORY.  Payloads stay fp16 from the memmap all the way to `.to(device)`; the model
casts (or autocast does).  At batch 128 with the measured post-QC means (RNA n = 2268,
ATAC m = 6343) one collated batch is

    ATAC  128 * 6343 * 512 * 2 B = 831 MB      RNA  128 * 2268 * 768 * 2 B = 446 MB
    ids/scores/cell vectors                    ~   11 MB
    ------------------------------------------------------------------------------
    TOTAL                                      ~ 1.29 GB   (fp32 would be 2.58 GB)

and with `max_atac_length=8192` forced (the parity mode that matches the live
`collate_fn(fixed_atac_length=8192)`) the ATAC block grows to 1.07 GB.  Casting to fp32
at `make_fm_tokens_from_cache` happens AFTER the H2D copy, so the fp32 copy lives on the
GPU and the PCIe traffic is halved.
"""

from __future__ import annotations

import csv
import os
import sys
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # the repo's standard sibling-import pattern
    sys.path.insert(0, _HERE)

from fm_token_cache import (  # noqa: E402  (path must be set first)
    CACHE_FORMAT_VERSION, LIVE_RNA_PAD_FILL, MASK_TRUE_IS_PAD, MODALITIES, CacheReader,
    CacheSpec,
)

#: `filip_align.NEG` -- the finite "minus inf" `pad_stack3`/`pad_stack4` write into the
#: padded tail of the per-token score, and that `select_topk_by_score` re-applies with
#: `score.masked_fill(padding_mask, NEG)`.  Mirrored rather than imported because
#: importing `filip_align` drags in the model package, and this file must stay loadable
#: on a login node.  `_assert_constants_match_the_consumer()` checks it when the real
#: module IS importable, so the mirror cannot drift silently.
NEG = -1e4

#: Fill values for the padded tail, per mode.  See `cached_collate`'s docstring for why
#: BOTH are mathematically inert and why "pad_stack" is the default.
_PAD_FILLS = {
    # `pad_stack`/`pad_stack3`/`pad_stack4`, train_filip_combined.py:321-356:
    # tokens <- 0, mask <- True, score <- NEG, ids <- -1 ("unmapped" for build_cis).
    "pad_stack": {"rna_tok": "zero", "gene_id": -1, "ccre_id": -1, "gs": NEG},
    # what the FROZEN FM itself emits inside one micro-batch: gatherData pads the value
    # AND id tensors with pad_token_id, and the encoder emits a real activation at those
    # columns; collate_fn pads ATAC ids with 0 and flash-attn writes exact zeros there.
    "live": {"rna_tok": "pad_emb", "gene_id": int(LIVE_RNA_PAD_FILL), "ccre_id": None,
             "gs": float(LIVE_RNA_PAD_FILL)},
}


# ------------------------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------------------------ #


class CachedTokenDataset(Dataset):
    """Per-cell view of a `fmtok-v1` cache.  One `__getitem__` = one disk-backed cell.

    Args:
        cache_dir:      cache root (the directory holding `manifest.json`/`index.csv`).
        split:          "train" / "val" -- selects the index rows AND the shard subtree.
        shards:         "all" (default), an int, a sequence of ints, or a dict
                        `{"rna": [...], "atac": [...]}`.  A bare int/sequence filters on
                        `rna_shard` ONLY.  See the note in `_apply_shard_filter` on why
                        an RNA shard subset does NOT localise ATAC reads.
        cell_whitelist: path to a one-barcode-per-line file, or any iterable of
                        barcodes.  Joined on the index's `barcode` column, never on row
                        position.
        allow_unlabeled: permit an all -1 label column.  OFF by default: an unlabeled
                        cache makes every label-conditioned loss term silently inert.
        allow_partial_index: permit a root index that is SHORT relative to the shards on
                        disk (a `--merge_index --shard_set 0-47` that was never redone
                        after 48-95 landed).  OFF by default, for the same reason
                        `modalities` is: a cache serving half its cells with every
                        barcode real is invisible downstream.  See
                        `CacheReader._check_index_completeness`.
        labels_csv:     `master_labels.csv`-style file; rebuilds the barcode -> int map
                        with the SAME recipe as `train_filip_combined.build_label_dict`
                        (sorted unique `cell_type` -> contiguous ints, missing = -1).
        subsample:      OFF by default.  `{"kg": int|None, "kc": int|None}` keeps only
                        the top-kg gene tokens (by expression) and the top-kc cCRE
                        tokens (by TF-IDF rank).  READ THE WARNING IN
                        `_apply_subsample`: this is NOT FILIP selection, and it changes
                        what the refiner is shown.
        token_dtype:    storage dtype carried out of `__getitem__` (fp16 by default; the
                        cast back from `CacheReader`'s fp32 is bit-exact because the
                        bytes on disk are fp16).

    Not thread-safe and deliberately not fork-safe-by-inheritance: memmaps are opened
    lazily and keyed by pid (see `_reader`).
    """

    def __init__(self, cache_dir: str, split: str,
                 shards: Union[str, int, Sequence[int], Dict] = "all",
                 cell_whitelist: Optional[Union[str, Iterable[str]]] = None,
                 labels_csv: Optional[str] = None,
                 spec: Optional[CacheSpec] = None, require_done: bool = True,
                 subsample: Optional[Dict[str, Optional[int]]] = None,
                 token_dtype=np.float16, keep_rna_pad: Optional[bool] = None,
                 label_barcode_column: str = "cell_barcode",
                 label_celltype_column: str = "cell_type",
                 allow_unlabeled: bool = False,
                 modalities: Optional[Sequence[str]] = None,
                 allow_partial_index: bool = False,
                 verbose: bool = True) -> None:
        self.cache_dir, self.split = cache_dir, split
        self.require_done = require_done
        self.token_dtype = np.dtype(token_dtype)
        self.verbose = verbose

        # Build the index ONCE here, in the parent.  CacheReader opens no memmap until
        # `_arr()` is called, so this probe holds nothing that must not cross a fork --
        # but we still drop it rather than keep a reader object around, so there is no
        # way for a child to end up using the parent's handles.
        # WHICH MODALITIES.  The corpus is built in two passes and, on a volume that
        # cannot hold it whole, the RNA half can be complete days before the ATAC half.
        # An RNA-only cache is a legitimate state, but it is NOT a state the cross-modal
        # consumer of record can train on -- so the default is to REFUSE IT AT
        # CONSTRUCTION, loudly, naming the missing pass, rather than to construct
        # happily and die on the first __getitem__ (which is what the unconditional
        # ATAC read used to do: a run that looked healthy through setup and failed at
        # step 0).  Passing `modalities` explicitly is the opt-in for RNA-side work.
        # SHORT BY SHARD RANGE is refused here too, and for the same reason as SHORT BY
        # MODALITY one line below: a stale `--shard_set` merge serves half the corpus
        # with every barcode real.  `allow_partial_index=True` is the opt-in.
        self.allow_partial_index = bool(allow_partial_index)
        probe = CacheReader(cache_dir, split, spec=spec, require_done=require_done,
                            modalities=modalities,
                            allow_partial_index=self.allow_partial_index)
        self.spec: CacheSpec = probe.spec
        self.manifest: Dict = probe.manifest
        self.index_completeness: Dict = probe.index_completeness
        self.modalities: Tuple[str, ...] = probe.modalities
        index: Dict[str, np.ndarray] = probe.index
        del probe
        if modalities is None:
            missing = [m for m in MODALITIES if m not in self.modalities]
            assert not missing, (
                f"{cache_dir} split={split!r} holds only {list(self.modalities)}: the "
                f"{missing} pass has not been built.  This is a PARTIAL cache by "
                f"design, not a broken one -- see manifest['splits']['{split}']"
                f"['completeness'].  A cross-modal consumer cannot train on it.  If "
                f"you genuinely want the {list(self.modalities)} side only, say so: "
                f"CachedTokenDataset(..., modalities={tuple(self.modalities)!r}).")

        assert self.manifest.get("format_version", CACHE_FORMAT_VERSION) == \
            CACHE_FORMAT_VERSION, "cache format version mismatch"
        n0 = len(index["barcode"])
        assert n0 > 0, f"{cache_dir}: split {split!r} has no rows in the index"

        index = self._apply_shard_filter(index, shards)
        index = self._apply_whitelist(index, cell_whitelist)
        self.index = index
        self.n = len(index["barcode"])
        assert self.n > 0, "every cell was filtered out"

        self.barcodes = np.asarray(index["barcode"]).astype(str)
        self.allow_unlabeled = bool(allow_unlabeled)
        self.labels = self._resolve_labels(labels_csv, label_barcode_column,
                                           label_celltype_column)

        if keep_rna_pad is None:
            keep_rna_pad = bool(self.spec.store_rna_pad) and "rna" in self.modalities
        assert not (keep_rna_pad and not self.spec.store_rna_pad), \
            "keep_rna_pad=True but the cache was built with store_rna_pad=False"
        assert not (keep_rna_pad and "rna" not in self.modalities), \
            "keep_rna_pad=True but this reader holds no RNA"
        self.keep_rna_pad = bool(keep_rna_pad)

        self.subsample = self._check_subsample(subsample)

        #: pid -> CacheReader.  NEVER populated in __init__; see `_reader`.
        self._readers: Dict[int, CacheReader] = {}
        self._reader_pid: int = -1

        if verbose:
            print(f"[CachedTokenDataset] {cache_dir} split={split} "
                  f"{n0} -> {self.n} cells, spec {CACHE_FORMAT_VERSION}, "
                  f"rna_pad={'on' if self.keep_rna_pad else 'off'}, "
                  f"subsample={self.subsample or 'off'}", flush=True)

    # --- construction-time filters -------------------------------------------------- #

    def _apply_shard_filter(self, index: Dict[str, np.ndarray], shards) -> Dict:
        """Restrict to a subset of shards.  LIVENESS-ASSERTED: a filter that removes
        nothing is a silently-inert flag, and this project has shipped two arms that
        came out bit-identical (`max|delta| = 0.000e+00`) in a single day.

        NOTE, and it matters for I/O planning: RNA and ATAC are built in TWO PASSES WITH
        DIFFERENT SHARDING KEYS (CACHE_DESIGN.md section 3.2 -- RNA by contiguous source
        row at 796 MB/s, ATAC by contiguous category code at 0.44 ms/cell).  A cell's
        `rna_shard` and `atac_shard` are therefore UNRELATED.  Filtering on `rna_shard`
        localises the RNA reads only; the ATAC reads still scatter across every ATAC
        shard.  Pass the dict form to constrain both.
        """
        if shards == "all" or shards is None:
            return index
        if isinstance(shards, dict):
            want = {m: np.asarray(sorted(set(int(x) for x in v)), dtype=np.int64)
                    for m, v in shards.items()}
            assert set(want) <= {"rna", "atac"}, f"bad shard dict keys {sorted(want)}"
        else:
            if isinstance(shards, (int, np.integer)):
                shards = [int(shards)]
            want = {"rna": np.asarray(sorted(set(int(x) for x in shards)),
                                      dtype=np.int64)}
        keep = np.ones(len(index["barcode"]), dtype=bool)
        for modality, ids in want.items():
            col = f"{modality}_shard"
            assert col in index, f"index has no {col!r} column"
            keep &= np.isin(np.asarray(index[col], dtype=np.int64), ids)
        n_after = int(keep.sum())
        assert n_after > 0, (
            f"shard filter {shards!r} selected 0 cells -- those shard ids do not "
            f"exist in this split's index")
        assert n_after < len(keep), (
            f"shard filter {shards!r} removed NOTHING ({n_after} of {len(keep)} cells) "
            f"-- the flag is INERT; pass shards='all' if that is what you meant")
        return {k: v[keep] for k, v in index.items()}

    def _apply_whitelist(self, index: Dict[str, np.ndarray], wl) -> Dict:
        """Join a barcode whitelist against the index's `barcode` column.

        BY BARCODE, never by position: QC drops scattered rows so cache row i is not
        source row i, and `PairedMultiOmicsDataset` documents that `cell_barcodes` can
        be longer than `len(ds)` outright.  The failure mode this assert exists for is a
        barcode NAMESPACE mismatch (a whitelist written without the `-1` suffix, or with
        a different `_{dataset_id}` suffix), which silently yields an empty dataset.
        """
        if wl is None:
            return index
        if isinstance(wl, str):
            # exactly train_filip_combined.py:1154's reader
            with open(wl) as f:
                wl_set = {ln.strip() for ln in f if ln.strip()}
            src = os.path.basename(wl)
        else:
            wl_set = {str(x) for x in wl}
            src = f"<{len(wl_set)} barcodes>"
        bc = np.asarray(index["barcode"]).astype(str)
        keep = np.fromiter((b in wl_set for b in bc), dtype=bool, count=len(bc))
        n_after = int(keep.sum())
        assert n_after > 0, (
            f"whitelist {src} matched 0 of {len(bc)} cache barcodes -- barcode "
            f"NAMESPACE mismatch (suffix?). cache e.g. {bc[0]!r}; "
            f"whitelist e.g. {next(iter(wl_set))!r}")
        # The join is a set intersection; proving it here means the filter really ran on
        # barcodes and not on some incidental ordering.
        assert n_after == len(wl_set & set(bc.tolist())), "whitelist join is not a join"
        if self.verbose:
            print(f"[CachedTokenDataset] whitelist {src}: {n_after}/{len(bc)} cells "
                  f"kept ({n_after}/{len(wl_set)} of the whitelist found)", flush=True)
        return {k: v[keep] for k, v in index.items()}

    def _resolve_labels(self, labels_csv: Optional[str], bc_col: str,
                        ct_col: str) -> np.ndarray:
        """Per-cell int label aligned to __getitem__ order; -1 = unknown.

        With `labels_csv`, rebuilds the map with `build_label_dict`'s exact recipe
        (train_filip_combined.py:223-230): sorted unique cell_type -> contiguous int.
        Without it, falls back to the index's own `label` column (written by pass 0 from
        `ds.get_labels()`, which resolves barcodes through `_valid_idx` /
        `_rna_keep_orig`).  Read with the csv module rather than pandas, so this file
        keeps no heavy import.
        """
        if labels_csv is None:
            lab = (np.asarray(self.index["label"], dtype=np.int64)
                   if "label" in self.index else np.full(self.n, -1, np.int64))
            cov = float((lab >= 0).mean()) if self.n else 0.0
            # THE SAME liveness rule as the labels_csv branch, and it belongs here even
            # more: the builder's `--labels_csv` used to default to None, so an all -1
            # column is a plausible thing to inherit from an older cache.  Every
            # label-conditioned term (supcon_xmodal, label_positives, hierarchical
            # positives) DROPS a row with no positive rather than raising, so half the
            # DECIDED objective would contribute exactly zero while the loss curve
            # still looked fine.
            assert cov > 0.0 or self.allow_unlabeled, (
                f"the cache index's `label` column is -1 for all {self.n} cells: this "
                f"cache was built without --labels_csv.  Every label-conditioned loss "
                f"term would be SILENTLY INERT.  Fix it with `build_fm_token_cache.py "
                f"--index_only --labels_csv ...` per shard (no payload is rebuilt), "
                f"pass labels_csv= here, or pass allow_unlabeled=True on purpose.")
            if self.verbose:
                print(f"[CachedTokenDataset] labels from the index: coverage "
                      f"{cov:.3f}", flush=True)
            return lab
        with open(labels_csv) as f:
            rows = list(csv.DictReader(f))
        assert rows and bc_col in rows[0] and ct_col in rows[0], \
            f"{labels_csv}: need columns {bc_col!r} and {ct_col!r}, got {list(rows[0])}"
        cts = sorted({str(r[ct_col]) for r in rows})
        ct2id = {c: i for i, c in enumerate(cts)}
        d = {str(r[bc_col]): ct2id[str(r[ct_col])] for r in rows}
        lab = np.asarray([d.get(b, -1) for b in self.barcodes], dtype=np.int64)
        cov = float((lab >= 0).mean())
        # Liveness: a labels_csv whose barcodes miss the cache namespace produces an
        # all -1 array, and every label-conditioned loss term then silently vanishes
        # (supcon drops rows with no positive rather than erroring).
        assert cov > 0.0 or self.allow_unlabeled, (
            f"{labels_csv}: 0/{self.n} cache barcodes carry a label -- barcode "
            f"namespace mismatch. cache e.g. {self.barcodes[0]!r}; csv e.g. "
            f"{next(iter(d))!r}")
        if self.verbose:
            print(f"[CachedTokenDataset] labels {os.path.basename(labels_csv)}: "
                  f"{len(cts)} classes, coverage {cov:.3f}", flush=True)
        return lab

    def _check_subsample(self, sub) -> Optional[Dict[str, Optional[int]]]:
        """Validate + LIVENESS-ASSERT the optional kg/kc token subsample.

        Off by default and it must stay that way: the FILIP pipeline runs the refiner
        over ALL tokens and selects top-kg/kc AFTERWARDS (train_filip_combined.py:1535
        -> :1537), so dropping tokens here changes the refiner's attention context for
        every surviving token.  That is a DIFFERENT MODEL, not a faster one.  The hook
        exists so a future arm can buy speed with fidelity deliberately, without
        rebuilding 8 TB of cache.
        """
        if sub is None:
            return None
        assert isinstance(sub, dict) and set(sub) <= {"kg", "kc"}, \
            f"subsample must be {{'kg':int|None,'kc':int|None}}, got {sub!r}"
        kg, kc = sub.get("kg"), sub.get("kc")
        assert kg is not None or kc is not None, "subsample={} is a no-op; pass None"
        out = {"kg": None if kg is None else int(kg),
               "kc": None if kc is None else int(kc)}
        # Liveness: assert the cap actually bites on THIS index.  A kg above the corpus
        # max is an inert flag that would report "subsample on" and change nothing.
        if out["kg"] is not None and "rna_len" in self.index:
            n_gene = np.asarray(self.index["rna_len"], np.int64) - 2
            assert int((n_gene > out["kg"]).sum()) > 0, (
                f"subsample kg={out['kg']} truncates 0 cells (max gene tokens "
                f"{int(n_gene.max())}) -- INERT flag")
        if out["kc"] is not None and "atac_len" in self.index:
            n_ccre = np.asarray(self.index["atac_len"], np.int64) - 2
            assert int((n_ccre > out["kc"]).sum()) > 0, (
                f"subsample kc={out['kc']} truncates 0 cells (max cCRE tokens "
                f"{int(n_ccre.max())}) -- INERT flag")
        warnings.warn(
            f"CachedTokenDataset: per-cell token subsample {out} is ACTIVE. This is "
            "NOT FILIP selection: the refiner self-attends over all tokens before "
            "FILIP picks top-kg/kc, so a pre-truncated batch is a different model. "
            "rc/ac stay full-token pooled.", RuntimeWarning, stacklevel=3)
        return out

    # --- per-worker memmaps --------------------------------------------------------- #

    def _reader(self) -> CacheReader:
        """The CacheReader owned by THIS process.

        np.memmap objects must not be inherited across a fork: the child would share the
        parent's file offsets and page mappings, which on NFS shows up as intermittent
        short/garbage reads rather than as an error.  So the handles open lazily on
        first use and are keyed by pid, and the dict is REPLACED (not extended) so a
        child can never reach a parent's entry even by accident.  The index is passed
        in, so a worker pays a manifest.json read and nothing else.
        """
        pid = os.getpid()
        r = self._readers.get(pid)
        if r is None:
            r = CacheReader(self.cache_dir, self.split, spec=self.spec,
                            index=self.index, require_done=self.require_done,
                            modalities=self.modalities,
                            allow_partial_index=self.allow_partial_index)
            self._readers = {pid: r}
            self._reader_pid = pid
        return r

    def reader_pid(self) -> int:
        """pid that owns the currently-open memmaps (-1 before the first read)."""
        return self._reader_pid

    def close(self) -> None:
        for r in self._readers.values():
            r.close()
        self._readers = {}
        self._reader_pid = -1

    def __getstate__(self):
        # Whatever the start method, a pickled dataset must not carry live memmaps.
        st = dict(self.__dict__)
        st["_readers"] = {}
        st["_reader_pid"] = -1
        return st

    # --- access --------------------------------------------------------------------- #

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict:
        """One cell, keyed with `fm_tokens`' own names so a call site ports by deletion.

        Returns numpy (`cached_collate` builds the torch tensors, and it runs inside
        the DataLoader worker, so these per-cell arrays never cross a process boundary):

            rt      [n, 768] f16   raw scFoundation tokens, meta at (n-2, n-1)
            rm      [n] bool       ALL FALSE -- a packed slice has no padding
            gs / rv [n] f16        gathered log1p expression = the leak-free `bio` score
                                   (`rv` is fm_tokens' internal name for the same array;
                                   both keys are the SAME object, so they cannot drift)
            gene_id [n] i64        scFoundation panel column; 19264/19265 = meta pair
            rc      [3072] f16     VALID-COUNT pooled FM cell vector -- NOT the
                                   encoder's batch-dependent global-slice rc
            at      [m, 512] f16   raw EpiAgent tokens, CLS at 0, SEP at m-1
            am      [m] bool       ALL FALSE
            ccre_id [m] i64        RAW token id (cCRE.bed row = id - 4; FixedSlotPooler
                                   subtracts the 4 itself, so do not pre-subtract)
            ac      [512] f16      EpiAgent CLS -- byte-identical to at[0]
            rna_pad [768] f16      encoder output at a pad column (only when kept); the
                                   one thing that makes pad_fill="live" reconstructible

        plus `barcode`, `label`, `dataset_id`/`batch_id` when the index carries them.
        """
        i = int(i)
        c = self._reader().get(i)

        # CacheReader.get upcasts to fp32 for the format's sake; casting back is
        # BIT-EXACT (the bytes on disk are already fp16) and halves what the DataLoader
        # has to move.  The model -- or autocast -- owns the widening.
        td = self.token_dtype
        out = {"barcode": c["barcode"], "label": int(self.labels[i])}
        # A modality this cache does not hold contributes NO KEYS.  Never zeros: a
        # zero-filled ATAC token is indistinguishable from a genuine flash-attn pad
        # output (CACHE_DESIGN L10), so a half cache would train silently.
        if "rna" in self.modalities:
            n = int(c["n_rna"])
            gs = c["gs"].astype(td, copy=False)
            out.update({
                "n_rna": n,
                "rt": c["rt"].astype(td, copy=False),
                "rm": np.zeros(n, dtype=bool),   # packed => no padding, by construction
                "gs": gs,
                "rv": gs,                        # same object; fm_tokens' internal name
                "gene_id": c["gene_id"],
                "rc": c["rc"].astype(td, copy=False),
            })
            if self.keep_rna_pad:
                out["rna_pad"] = c["rna_pad"].astype(td, copy=False)
        if "atac" in self.modalities:
            m = int(c["n_atac"])
            out.update({
                "n_atac": m,
                "at": c["at"].astype(td, copy=False),
                "am": np.zeros(m, dtype=bool),
                "ccre_id": c["ccre_id"],
                "ac": c["ac"].astype(td, copy=False),
            })
        for k in ("dataset_id", "batch_id", "orig_row"):
            if k in c:
                out[k] = c[k]
        if self.subsample is not None:
            _apply_subsample(out, self.subsample["kg"], self.subsample["kc"])
        return out

    # --- sampler / loss helpers (mirror PairedMultiOmicsDataset's API) -------------- #

    @property
    def collate_fn(self):
        """A `collate_fn` bound to this cache's spec, ready for `DataLoader(...)`."""
        from functools import partial
        return partial(cached_collate, spec=self.spec)

    def get_labels(self) -> np.ndarray:
        """Per-cell int labels aligned to __getitem__ order; -1 = unknown."""
        return self.labels.copy()

    def get_dataset_ids(self, column: str = "dataset_id") -> np.ndarray:
        """Per-cell obs value aligned to __getitem__ order (for
        SameDatasetBlockedBatchSampler and center_*_by_dataset).  Resolved by pass 0
        through the dataset's own index map, then carried in the cache index.
        """
        assert column in self.index, \
            f"cache index has no {column!r} column; have {sorted(self.index)}"
        return np.asarray(self.index[column]).astype(str)

    def get_aligned_barcodes(self) -> np.ndarray:
        return self.barcodes.copy()

    def lengths(self, modality: str = "rna") -> np.ndarray:
        """Per-cell token count, from the index (for a length-aware sampler)."""
        col = f"{modality}_len"
        assert col in self.index, f"cache index has no {col!r} column"
        return np.asarray(self.index[col], dtype=np.int64)


# ------------------------------------------------------------------------------------ #
# Optional per-cell token subsample (OFF by default -- see _check_subsample)
# ------------------------------------------------------------------------------------ #


def _apply_subsample(out: Dict, kg: Optional[int], kc: Optional[int]) -> None:
    """In-place per-cell top-k, preserving every structural invariant of the format.

    RNA: keep the top-kg GENE tokens by expression (`gs`), then RE-SORT by position so
    the gene ids stay strictly ascending, then re-append the two meta tokens -- they are
    always kept, because `fm_pool_rna`/`filip_masks` locate them at (vc-2, vc-1) and
    dropping them would silently repurpose two real genes as the summary pair.

    ATAC: keep CLS + the first kc cCREs + SEP.  A PREFIX is exactly the top-kc by TF-IDF
    accessibility (the cell sentence is TF-IDF-DESCENDING from EpiAgent tokenization),
    which is also what `cs = -arange` scores -- so the rank semantics survive untouched.

    `rc`/`ac` are deliberately NOT recomputed: they are the FROZEN FM's cell vectors,
    pooled over all tokens, and they do not depend on what the refiner is later shown.
    `kg`/`kc` here count GENE / cCRE tokens; the meta pair and CLS/SEP are extra.
    """
    # A modality this cache does not hold contributes no keys at all (see
    # CachedTokenDataset.__getitem__), so the corresponding leg is simply absent rather
    # than operating on a fabricated tensor.
    if kg is not None and "n_rna" in out:
        n = int(out["n_rna"])
        n_gene = n - 2
        if n_gene > kg:
            sc = np.asarray(out["gs"][:n_gene], dtype=np.float32)
            keep = np.argpartition(-sc, kg - 1)[:kg]
            keep.sort()  # positions ascending => gene ids ascending (packed invariant)
            idx = np.concatenate([keep, np.asarray([n_gene, n_gene + 1])])
            for k in ("rt", "gene_id", "gs", "rm"):
                out[k] = np.ascontiguousarray(out[k][idx])
            out["rv"] = out["gs"]
            out["n_rna"] = int(idx.size)
    if kc is not None and "n_atac" in out:
        m = int(out["n_atac"])
        n_ccre = m - 2
        if n_ccre > kc:
            idx = np.concatenate([np.asarray([0]), np.arange(1, kc + 1),
                                  np.asarray([m - 1])])
            for k in ("at", "ccre_id", "am"):
                out[k] = np.ascontiguousarray(out[k][idx])
            out["n_atac"] = int(idx.size)


# ------------------------------------------------------------------------------------ #
# Collate
# ------------------------------------------------------------------------------------ #


def cached_collate(batch: Sequence[Dict], max_atac_length: Optional[int] = None, *,
                   pad_fill: str = "pad_stack", rc_mode: str = "valid_count",
                   spec: Optional[CacheSpec] = None,
                   dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
    """Pad a list of cached cells to the BATCH max and rebuild `fm_tokens`' tensors.

    Returns a dict whose {rc, rt, rm, ac, at, am, gs, cs, gene_id, ccre_id} are the
    10-tuple of `fm_tokens(..., include_summary=True, return_ids=True)`, element for
    element, plus `labels`, `barcode`, `rna_len`, `atac_len`.

    WIDTH.  RNA pads to the batch max, which is what the live path does too (gatherData
    pads to the micro-batch max).  ATAC pads to the batch max by default; pass
    `max_atac_length=8192` to match `collate_fn(fixed_atac_length=8192)` exactly.  The
    narrower default is mathematically inert for the refiner (masked attention) and for
    every pooling (mask-weighted), but it is NOT inert for FILIP selection, because
    `select_topk_by_score` takes `k = min(kc, Na)` -- a batch whose longest cell falls
    under kc would select fewer tokens than a live run.  Parity runs must force it.
    A width BELOW the batch max is refused rather than truncated: silent ATAC truncation
    is the exact deficit this cache exists to remove.

    PAD FILL.  Neither mode reproduces the live tensor byte-for-byte, and neither can:
    the live `rt` is padded TWICE -- once by the FM inside a micro-batch (gatherData
    fills the value and id tensors with pad_token_id and the encoder emits an activation
    there; ATAC ids get 0 and flash-attn writes exact zeros), then again by `pad_stack3`
    /`pad_stack4` ACROSS micro-batches (tokens 0, mask True, score NEG, ids -1).  So a
    live batch carries a MIXTURE that depends on the accumulation geometry.  Both fills
    are inert -- every consumer re-masks: the refiners mask padding in attention,
    `select_topk_by_score` does `score.masked_fill(padding_mask, NEG)`, `masked_mean`
    and `fm_pool_*` weight by `~mask`.  Default is "pad_stack" because that is the
    semantics of the tensor this call REPLACES (the outer stack).  "live" reproduces the
    micro-batch fill and is what the parity tests compare against.
    103 under "live" is a FILL value.  It is also the real gene ABLIM1.  It is never,
    anywhere, used to DETECT padding.

    rc_mode
        "valid_count" (default) -- the cached, correct pooling (`fm_pool_rna` recipe).
        "live_global" -- reproduces `model.py:215-218` / `fixed_slot_model_mixin.py:
        464-477`: emb1/emb2 at GLOBAL [-1]/[-2] and max/mean over [:, :-2], i.e. over
        this cell's meta tokens AND its pad columns.  Needs `pad_fill="live"` and a
        cache built with `store_rna_pad`, because the pad columns must carry their real
        encoder output.  Provided so the FineCLS fixed-slot path stays reproducible, and
        so the T8 liveness probe has something to prove a difference against.
    """
    spec = spec or CacheSpec()
    assert pad_fill in _PAD_FILLS, f"pad_fill must be one of {sorted(_PAD_FILLS)}"
    assert rc_mode in ("valid_count", "live_global")
    fills = _PAD_FILLS[pad_fill]
    b = len(batch)
    assert b > 0, "empty batch"

    # WHICH MODALITIES this batch carries.  `__getitem__` OMITS the keys of a modality
    # its cache does not hold (never zero-fills them: a zeros ATAC block is
    # indistinguishable from a genuine flash-attn pad output, CACHE_DESIGN L10), so the
    # presence of the key IS the answer.  It must be the same answer for every cell --
    # a mixed batch would mean two caches were spliced.
    has_rna = "n_rna" in batch[0]
    has_atac = "n_atac" in batch[0]
    assert has_rna or has_atac, "every cell in the batch is empty"
    assert all(("n_rna" in c) == has_rna and ("n_atac" in c) == has_atac
               for c in batch), \
        "the batch mixes cells from a single-modality and a both-modality cache"

    out: Dict[str, torch.Tensor] = {}
    need_pad_emb = pad_fill == "live" or rc_mode == "live_global"

    if has_rna:
        nr = np.asarray([int(c["n_rna"]) for c in batch], dtype=np.int64)
        n_r = int(nr.max())
        d_r = spec.rna_dim
        assert batch[0]["rt"].shape[1] == d_r, "rna dim disagrees with the spec"
        rt = torch.zeros((b, n_r, d_r), dtype=dtype)
        gs = torch.full((b, n_r), float(fills["gs"]), dtype=dtype)
        gid = torch.full((b, n_r), int(fills["gene_id"]), dtype=torch.long)
        # True == PAD.  Start fully padded and clear the valid PREFIX: packing
        # guarantees the valid region is a prefix (gatherData packs valid-first;
        # flash-attn pads at the end), so there is no interior padding to represent.
        rm = torch.ones((b, n_r), dtype=torch.bool)
        rc = torch.zeros((b, spec.rna_cell_dim), dtype=dtype)
        for j, c in enumerate(batch):
            n = int(nr[j])
            rt[j, :n] = torch.as_tensor(np.asarray(c["rt"]))
            gs[j, :n] = torch.as_tensor(np.asarray(c["gs"]))
            gid[j, :n] = torch.as_tensor(np.asarray(c["gene_id"], dtype=np.int64))
            rm[j, :n] = False
            rc[j] = torch.as_tensor(np.asarray(c["rc"]))
            if need_pad_emb and n < n_r:
                assert "rna_pad" in c, (
                    "pad_fill='live'/rc_mode='live_global' need the per-cell pad "
                    "embedding; build with keep_rna_pad=True on a store_rna_pad cache")
                # Every pad column of a cell produces the SAME encoder output (identical
                # input token_emb(103.0)+pos_emb(103), identical masked key set), so one
                # vector broadcasts over the whole padded tail.
                rt[j, n:] = torch.as_tensor(np.asarray(c["rna_pad"]))
        # The cheap version of the tombstone at model.py:229-243.  If a future edit
        # inverts a mask, this fires here instead of showing up as rna_proj_norm=0.0
        # six hours in.
        assert torch.equal((~rm).sum(1), torch.from_numpy(nr)), \
            "rm polarity/length broken (True must be PAD)"
        out.update({"rt": rt, "rm": rm, "gs": gs, "gene_id": gid, "rc": rc,
                    "rna_len": torch.from_numpy(nr)})
    else:
        assert rc_mode == "valid_count", "rc_mode='live_global' needs the RNA side"

    if has_atac:
        na = np.asarray([int(c["n_atac"]) for c in batch], dtype=np.int64)
        n_a = int(na.max()) if max_atac_length is None else int(max_atac_length)
        assert n_a >= int(na.max()), (
            f"max_atac_length={n_a} < batch max {int(na.max())}: refusing to TRUNCATE "
            "ATAC tokens. Silent truncation is the deficit this cache exists to remove "
            "(pipeline_term_is_atac_truncation); widen the request or drop the cell.")
        assert n_a <= spec.max_atac_length, \
            f"max_atac_length={n_a} exceeds the cache cap {spec.max_atac_length}"
        d_a = spec.atac_dim
        assert batch[0]["at"].shape[1] == d_a, "atac dim disagrees with the spec"
        at = torch.zeros((b, n_a, d_a), dtype=dtype)
        ccre_fill = (spec.atac_pad_id if fills["ccre_id"] is None
                     else int(fills["ccre_id"]))
        cid = torch.full((b, n_a), ccre_fill, dtype=torch.long)
        am = torch.ones((b, n_a), dtype=torch.bool)
        ac = torch.zeros((b, d_a), dtype=dtype)
        for j, c in enumerate(batch):
            m = int(na[j])
            at[j, :m] = torch.as_tensor(np.asarray(c["at"]))
            cid[j, :m] = torch.as_tensor(np.asarray(c["ccre_id"], dtype=np.int64))
            am[j, :m] = False
            ac[j] = torch.as_tensor(np.asarray(c["ac"]))
        assert torch.equal((~am).sum(1), torch.from_numpy(na)), \
            "am polarity/length broken (True must be PAD)"
        # cs is EXACTLY -arange(Na): the negative token POSITION, i.e. TF-IDF
        # accessibility rank (the sentence is TF-IDF-descending).  Zero bits of
        # information -> always recomputed from the REPLAY width, never stored.
        # `.expand` matches fm_tokens:305 and costs Na floats, not b*Na.
        cs = -torch.arange(n_a, dtype=torch.float32).unsqueeze(0).expand(b, n_a)
        out.update({"at": at, "am": am, "ac": ac, "cs": cs, "ccre_id": cid,
                    "atac_len": torch.from_numpy(na)})
    elif max_atac_length is not None:
        # A width for a modality the batch does not carry is a silently-inert flag.
        raise AssertionError(
            f"max_atac_length={max_atac_length} on an RNA-only batch: the flag can "
            f"reach nothing.  Drop it, or collate a cache that holds ATAC.")

    if rc_mode == "live_global":
        assert pad_fill == "live", "rc_mode='live_global' needs pad_fill='live'"
        # model.py:215-218 verbatim, on the padded batch: emb1/emb2 from the GLOBAL last
        # two columns (padding for every cell but the batch-longest) and max/mean over
        # [:, :-2] (which leaks this cell's meta tokens AND its pad columns).
        f = out["rt"].float()
        out["rc"] = torch.cat(
            [f[:, -1, :], f[:, -2, :], f[:, :-2, :].max(dim=1).values,
             f[:, :-2, :].mean(dim=1)], dim=1).to(dtype)

    out["labels"] = torch.as_tensor([int(c.get("label", -1)) for c in batch],
                                    dtype=torch.long)
    out["barcode"] = [str(c["barcode"]) for c in batch]
    return out


def move_batch_to_device(batch: Dict, device, non_blocking: bool = True) -> Dict:
    """`.to(device)` every tensor, leave the python columns alone.

    Do this BEFORE `make_fm_tokens_from_cache(..., cast_float=True)`: the fp16 payload
    crosses PCIe at half the bytes and the fp32 copy is materialised on the GPU.
    """
    out = dict(batch)
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
    return out


# ------------------------------------------------------------------------------------ #
# fm_tokens() drop-in
# ------------------------------------------------------------------------------------ #


def make_fm_tokens_from_cache(batch, return_ids: bool = False, cast_float: bool = True,
                              device=None, **collate_kw) -> Tuple:
    """Return EXACTLY what `fm_tokens()` returns, from the cache instead of the FMs.

    `fm_tokens` (train_filip_combined.py:284) returns

        (rc, rt, rm, ac, at, am, rv, cs)                      # 8-tuple
        (rc, rt, rm, ac, at, am, rv, cs, gene_id, ccre_id)    # return_ids=True

    with rc/rt/ac/at/rv float32 (`.float()`), rm/am bool True==PAD, cs float32.  So the
    swap in the training loop is::

        - out = fm_tokens(model, b, device, include_summary=..., return_ids=cis)
        + out = make_fm_tokens_from_cache(b, return_ids=cis, device=device)

    `batch` may be the dict `cached_collate` produced, or the raw list of per-cell dicts
    (which is collated here with `**collate_kw`).  `cast_float=False` keeps fp16 and
    lets autocast widen, which halves activation memory on the refiner's input.
    """
    d = batch if isinstance(batch, dict) else cached_collate(batch, **collate_kw)
    if device is not None:
        d = move_batch_to_device(d, device)
    # `fm_tokens` is a CROSS-MODAL signature: there is no half of it.  A single-modality
    # cache cannot produce one, and the honest failure is here, naming the missing pass
    # -- not a tuple with zeros in six of ten slots.
    missing = [m for m, k in (("rna", "rt"), ("atac", "at")) if k not in d]
    assert not missing, (
        f"make_fm_tokens_from_cache needs both modalities; this batch carries no "
        f"{missing} (a PARTIAL cache -- see manifest['splits'][...]['completeness']).  "
        f"Build the {missing} pass before replacing fm_tokens() with the cache.")
    cast = (lambda t: t.float()) if cast_float else (lambda t: t)
    rc, rt, rm = cast(d["rc"]), cast(d["rt"]), d["rm"]
    ac, at, am = cast(d["ac"]), cast(d["at"]), d["am"]
    rv, cs = cast(d["gs"]), d["cs"]
    # fm_tokens' own shape asserts (lines 314-315), kept at the point of truth: an id
    # array one token wider or narrower than the tokens shifts every cCRE's genome
    # coordinate SILENTLY, because the ids stay valid -- they just belong to the
    # neighbouring cCRE.
    assert rm.shape == rt.shape[:2] and am.shape == at.shape[:2]
    assert cs.shape == at.shape[:2], f"cs {tuple(cs.shape)} vs tokens {at.shape[1]}"
    if not return_ids:
        return rc, rt, rm, ac, at, am, rv, cs
    gid, cid = d["gene_id"], d["ccre_id"]
    assert cid.shape[1] == at.shape[1], f"cCRE ids {cid.shape[1]} vs {at.shape[1]}"
    assert gid.shape[1] == rt.shape[1], f"gene ids {gid.shape[1]} vs {rt.shape[1]}"
    return rc, rt, rm, ac, at, am, rv, cs, gid, cid


# ------------------------------------------------------------------------------------ #
# Drift guards
# ------------------------------------------------------------------------------------ #


def _assert_constants_match_the_consumer() -> List[str]:
    """Check the mirrored constants against the real modules when they are importable.

    NEG and the mask polarity are copied here so this file loads without the model
    package.  A copy that drifts is worse than an import, so prove it has not: this is
    called by the self-test and is a no-op (returning what it could not check) on a node
    where the consumer is not importable.
    """
    unchecked = []
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    for p in (repo, os.path.join(repo, "haoyun", "multiomics_clip_finelip")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from modules.filip_align import NEG as REAL_NEG  # type: ignore
        assert NEG == REAL_NEG, f"NEG drifted: {NEG} vs filip_align {REAL_NEG}"
    except Exception:
        unchecked.append("filip_align.NEG")
    assert MASK_TRUE_IS_PAD, "True==PAD is the pipeline-wide convention"
    return unchecked


if __name__ == "__main__":  # pragma: no cover
    print(f"cached_token_dataset for {CACHE_FORMAT_VERSION}; "
          f"unchecked constants: {_assert_constants_match_the_consumer()}")
    print("run test_cached_token_dataset.py for the unit test")
