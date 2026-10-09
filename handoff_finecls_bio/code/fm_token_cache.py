#!/usr/bin/env python3
"""Shared on-disk format for the frozen-FM token cache (`fmtok-v1`).

THIS FILE IS THE FORMAT.  Both the builder (GPU, model imports, h5py) and every reader
(training, eval, diagnostics) go through it, so the layout is defined exactly once.  It
is deliberately free of GPU code and of any model/h5py import: the whole thing must be
unit-testable on a login node, and it is (see the `__main__` self-test at the bottom).

WHY RAGGED/PACKED.  ATAC cells are collated to a FIXED 8192 window at training time, but
the mean valid length is 6,343 (post-QC).  Storing the pad would cost +1.53 TB, taking
the build from 8.10 TB to 9.63 TB -- above the 10 TB quota once the mandatory 1.15x
margin is applied. Packing is also EXACTLY faithful on the ATAC side: flash-attn's
BertEncoder `unpad_input`s the padding away before attention and `pad_input`s zeros
back, so padded positions carry zero information.  On the RNA side it is faithful to the
measured 1.19e-06 kernel-tiling noise, 400x below fp16 resolution.

WHY PACKING ALSO FIXES A BUG.  scFoundation's `gatherData` packs VALID tokens first
(ascending panel-column order) with padding at the end, so the two meta tokens (columns
19264 = target_resolution, 19265 = log10_total) sit at the LAST TWO VALID positions
(n-2, n-1) -- NOT at global [-2]/[-1], which are padding for every cell except the
batch-longest.  The live pooling at haoyun/multiomics_clip_finelip/model.py:215-218
reads the global positions and is therefore corrupt at batch_size > 1.  In a PACKED
slice the length IS the valid count, so there is no global last position to read by
mistake: the bug is unrepresentable here.

CONVENTIONS THAT MUST NOT DRIFT (all recorded in manifest.json, all asserted on open):
  * masks are True == PAD, everywhere, both modalities.  The ONLY inversion in the whole
    pipeline lives inside ATACRefinerFM.forward (`kpm = ~pad_mask`, flash-attn wants
    True == VALID).  A reader that "helpfully" inverts reintroduces the bug whose
    tombstone is model.py:229-243 (rna_pad=1.0, rna_valid=0.0, rna_proj_norm=0.0).
  * tokens are RAW encoder activations.  NOT L2-normalised.  FineCLS feeds raw
    activations to FixedSlotPooler and concatenates a UNIT-NORM cell block before a
    single Linear -- that scale mismatch is part of the trained function.  This class
    has no `l2` parameter on purpose (cf. the project's `l2_input_convention_bug`,
    where `load_set` defaulted l2=True).
  * validity is derived ONLY from the stored offsets.  Never from a token id
    (pad_token_id=103 IS the real gene ABLIM1) and never from non-zero-ness
    (flash-attn writes exact zeros at padded ATAC positions, and a refined token can
    legitimately be zero).
  * cCRE ids are stored RAW.  cCRE.bed 0-based row = id - 4; FixedSlotPooler subtracts
    the 4 itself (`token_offset=4`) and relies on out-of-range ids (SEP -> -2) failing
    its in-vocabulary test.  Pre-subtracting here would break it silently.

See CACHE_DESIGN.md in this directory for the full layout, sharding plan, manifest
schema, parity contract and landmine table.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # torch is optional for the writer; the reader's collate returns torch tensors.
    import torch
except ImportError:  # pragma: no cover - the builder always has torch
    torch = None


CACHE_FORMAT_VERSION = "fmtok-v1"

#: The two modalities, in canonical order.  Named rather than written as a literal in
#: eight places, because "which modalities does this cache hold" is now a REAL question:
#: the corpus is built in two passes and an RNA-only cache is a legitimate, readable,
#: verifiable intermediate state rather than a broken one.  Every function that used to
#: hardcode `("rna", "atac")` in a completeness test now takes the set as an argument.
MODALITIES: Tuple[str, ...] = ("rna", "atac")

#: Mask polarity, as a named constant so no call site has to guess.  fm_tokens() returns
#: True == PAD for both modalities; every array in this cache follows it.
MASK_TRUE_IS_PAD = True

#: Value gatherData writes into the padded tail of BOTH the value tensor and the gene-id
#: tensor.  103 is ALSO the real gene ABLIM1 (OS_scRNA_gene_index.19264.tsv line 105),
#: which is exactly why nothing in this module ever filters by id -- it is used only to
#: REPRODUCE the live pad values in `CacheReader.collate(pad_fill="live")`, never to
#: detect padding.
LIVE_RNA_PAD_FILL = 103.0

#: filip_align.NEG.  select_topk_by_score does `score.masked_fill(padding_mask, NEG)`,
#: so pad scores are moot; this is only used by `pad_fill="safe"`.
PAD_SCORE_SENTINEL = -1e4

#: Rows per block in verify_shard's NaN/Inf scan.  2**21 rows x 768 f32 = 6.4 GB would
#: still be too much, so this is sized on the WIDEST payload (768 cols): 2**19 x 768 x 4
#: = 1.6 GB, and 2**19 x 512 x 4 = 1.1 GB for ATAC.  NEVER scan the whole memmap in one
#: `np.asarray(..., np.float32)` -- see the comment in verify_shard.
NAN_SCAN_CHUNK_ROWS = 1 << 19

#: Name of the ownership stamp dropped inside every `<shard>.tmp` directory.  A .tmp is
#: ALWAYS a failed run; the stamp says WHICH run, so a later job can garbage-collect the
#: orphans of a dead array without deleting a peer's live work.
TMP_OWNER_FILE = "_OWNER.json"

_MAGIC = b"\x93NUMPY"
#: 12-byte v2.0 prefix + 116-byte header string = 128 bytes, a multiple of 64 as the
#: .npy spec requires for data alignment.  Fixed size so `finalize()` can rewrite the
#: true shape in place after streaming the payload.
_NPY_HEADER_BYTES = 128
_NPY_HEADER_LEN = _NPY_HEADER_BYTES - 12


# ------------------------------------------------------------------------------------ #
# Spec
# ------------------------------------------------------------------------------------ #


@dataclass(frozen=True)
class ArraySpec:
    """One file inside a shard directory."""

    name: str
    dtype: str
    ncols: Optional[int]  # None => 1-D
    kind: str  # "token" (one row per TOKEN) or "cell" (one row per CELL)
    optional: bool = False


@dataclass(frozen=True)
class CacheSpec:
    """Dims, dtypes, caps and file names of one cache.  Frozen: a spec is provenance.

    Every field here that can change the MEANING of the bytes is mirrored into
    manifest.json and re-asserted by `CacheReader`, which refuses to open a cache whose
    recorded spec differs from the one the consumer declares.  That is the load-time
    half of this project's standing "prove every arm-defining flag is live" rule; the
    build-time half is a liveness assert in the builder.
    """

    version: str = CACHE_FORMAT_VERSION

    # --- model dims (confirmed from the checkpoints, not from docs) ----------------- #
    rna_dim: int = 768  # scFoundation d_model (12 layers, FFN 3072, nhead 12)
    atac_dim: int = 512  # EpiAgent d_model (18-layer BERT)
    rna_cell_dim: int = 3072  # 4 * 768 = [emb1 | emb2 | max | mean]

    # --- vocabularies --------------------------------------------------------------- #
    n_genes: int = 19264  # rows of OS_scRNA_gene_index.19264.tsv
    rna_seq_len: int = 19266  # + [target_resolution, log10_total]
    # (resolution, log10_total), in THIS order
    rna_meta_ids: Tuple[int, int] = (19264, 19265)
    atac_vocab_size: int = 1355449  # 1,355,445 cCREs + 4 specials
    atac_pad_id: int = 0
    atac_cls_id: int = 1
    atac_sep_id: int = 2
    atac_id_offset: int = 4  # cCRE.bed 0-based row = raw token id - 4

    # --- caps / QC (hardcoded in multiomics_clip/dataset.py, so pinned here) -------- #
    max_atac_length: int = 8192  # EpiAgent rank_embedding has EXACTLY 8192 rows
    min_atac_length: int = 1000  # PairedMultiOmicsDataset default = what build_ds uses
    min_rna_nnz: int = 100
    max_rna_nnz: int = 8500
    target_resolution: float = 4.0
    # class default is True and calls np.random.choice (!)
    random_truncate: bool = False

    # --- conventions ---------------------------------------------------------------- #
    # meta tokens packed at (n-2, n-1); see module docstring
    include_summary: bool = True
    # [N,768] pad-column output, replays the live global slicing
    store_rna_pad: bool = True
    l2_normalized: bool = False  # raw activations -- do not change
    mask_true_is_pad: bool = MASK_TRUE_IS_PAD

    # --- dtypes --------------------------------------------------------------------- #
    token_dtype: str = "float16"
    gene_id_dtype: str = "uint16"  # max id 19265 < 65535; halves 7.4 GB to 3.7 GB
    ccre_id_dtype: str = "uint32"  # max observed 1,354,674 < 2^32
    cell_dtype: str = "float16"
    #: int64 EVERYWHERE.  Packed RNA is ~1.62e9 rows and packed ATAC ~4.52e9; both
    #: exceed 2^31.  A silent int32 wrap yields negative slice bounds that numpy reads
    #: as end-relative indexing -- every cell after the wrap gets another cell's tokens
    #: and NOTHING CRASHES.  Shard-local offsets stay small, but the type never narrows.
    offset_dtype: str = "int64"

    def __post_init__(self) -> None:
        assert self.version == CACHE_FORMAT_VERSION, f"unknown format {self.version!r}"
        # max_atac_length > 8192 is an IndexError inside EpiAgent (rank_embedding rows),
        # not a slowdown.  Catch it here rather than 3 hours into a build.
        assert 0 < self.max_atac_length <= 8192, "EpiAgent rank_embedding has 8192 rows"
        assert self.rna_cell_dim == 4 * self.rna_dim
        assert self.rna_meta_ids == (self.n_genes, self.n_genes + 1)
        assert self.rna_seq_len == self.n_genes + 2
        top_id = self.rna_meta_ids[1]
        assert np.dtype(self.gene_id_dtype).type(top_id) == top_id
        assert not self.l2_normalized, \
            "tokens are RAW activations; see module docstring"
        assert self.mask_true_is_pad, "True==PAD is the pipeline-wide convention"
        assert np.dtype(self.offset_dtype) == np.int64

    # --- derived -------------------------------------------------------------------- #

    @property
    def atac_trunc(self) -> int:
        """Sentence truncation used by PairedMultiOmicsDataset: max_atac_length - 2
        (CLS+SEP).
        """
        return self.max_atac_length - 2

    @property
    def max_rna_len(self) -> int:
        """Largest possible packed RNA length: max_rna_nnz + the 2 meta tokens."""
        return self.max_rna_nnz + 2

    def arrays(self, modality: str) -> List[ArraySpec]:
        """The files that make up one shard of `modality`, in write order."""
        if modality == "rna":
            out = [
                ArraySpec("rna_tokens.npy", self.token_dtype, self.rna_dim, "token"),
                ArraySpec("rna_gene_ids.npy", self.gene_id_dtype, None, "token"),
                ArraySpec("rna_values.npy", self.token_dtype, None, "token"),
                ArraySpec("rna_cell.npy", self.cell_dtype, self.rna_cell_dim, "cell"),
            ]
            if self.store_rna_pad:
                out.append(
                    ArraySpec("rna_pad.npy", self.cell_dtype, self.rna_dim, "cell"))
            return out
        if modality == "atac":
            return [
                ArraySpec("atac_tokens.npy", self.token_dtype, self.atac_dim, "token"),
                ArraySpec("atac_ccre_ids.npy", self.ccre_id_dtype, None, "token"),
                ArraySpec("atac_cls.npy", self.cell_dtype, self.atac_dim, "cell"),
            ]
        raise ValueError(f"modality must be 'rna' or 'atac', got {modality!r}")

    def offsets_name(self, modality: str) -> str:
        return f"{modality}_offsets.npy"

    def len_column(self, modality: str) -> str:
        return f"{modality}_len"

    def shard_dirname(self, shard_idx: int, num_shards: int) -> str:
        return f"shard{shard_idx:04d}of{num_shards:04d}"

    def shard_path(self, root: str, split: str, modality: str, shard_idx: int,
                   num_shards: int) -> str:
        return os.path.join(root, split, modality,
                            self.shard_dirname(shard_idx, num_shards))

    def to_json(self) -> Dict:
        d = asdict(self)
        d["rna_meta_ids"] = list(self.rna_meta_ids)
        return d

    @staticmethod
    def from_json(d: Dict) -> "CacheSpec":
        d = dict(d)
        d["rna_meta_ids"] = tuple(d["rna_meta_ids"])
        known = set(CacheSpec.__dataclass_fields__)
        unknown = set(d) - known
        assert not unknown, f"manifest carries unknown spec fields {sorted(unknown)}"
        return CacheSpec(**d)


#: Columns of the per-shard index.csv.  The global index.csv at the cache root is this
#: plus {split, orig_row, dataset_id, batch_id, label, <other>_shard, <other>_row, ...};
#: see CACHE_DESIGN.md section 1.4.  `barcode` is THE join key and is resolved through
#: the dataset's own two-level map bc_all[_rna_keep_orig[_valid_idx]] -- never
#: positionally.
SHARD_INDEX_COLUMNS = ("row", "barcode", "orig_row", "length")


# ------------------------------------------------------------------------------------ #
# .npy streaming primitives
# ------------------------------------------------------------------------------------ #


def _npy_header_bytes(dtype: np.dtype, shape: Tuple[int, ...]) -> bytes:
    """Build a FIXED-LENGTH (128 B) .npy v2.0 header.

    numpy's own writer sizes the header to the content, which would make an in-place
    rewrite at finalize() change the data offset.  Reserving a constant 128 bytes
    (12-byte prefix + 116-byte space-padded dict, a multiple of 64 as the spec requires)
    lets `finalize()` seek to 0 and stamp the true row count without moving a single
    payload byte.
    """
    d = f"{{'descr': {np.lib.format.dtype_to_descr(dtype)!r}, " \
        f"'fortran_order': False, 'shape': {shape!r}, }}"
    pad = _NPY_HEADER_LEN - len(d) - 1
    assert pad >= 0, f"header {len(d)} B does not fit in {_NPY_HEADER_LEN} B: {d}"
    header = (d + " " * pad + "\n").encode("latin1")
    assert len(header) == _NPY_HEADER_LEN
    return _MAGIC + bytes([2, 0]) + struct.pack("<I", _NPY_HEADER_LEN) + header


class _NpyAppendWriter:
    """Streaming .npy writer: buffered `file.write()` appends, never mmap stores.

    WHY NOT open_memmap.  An mmap store that hits ENOSPC on NFS delivers SIGBUS and
    kills the process with no Python traceback, leaving a sparse file that later reads
    as valid.  A `write()` returns a short count or raises OSError, which the caller can
    catch, truncate and re-raise.  On a filesystem that is 100% full and shared, that
    difference is the whole ball game.  Readers still get a perfectly ordinary .npy that
    np.load(mmap_mode="r") maps.
    """

    def __init__(self, path: str, dtype: str, ncols: Optional[int],
                 buffer_bytes: int = 1 << 22) -> None:
        self.path = path
        self.dtype = np.dtype(dtype)
        self.ncols = ncols
        self.rows = 0
        self._fp = open(path, "wb", buffering=buffer_bytes)
        # Placeholder header; rewritten with the true shape by close().
        self._fp.write(b"\x00" * _NPY_HEADER_BYTES)
        self._closed = False

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.rows,) if self.ncols is None else (self.rows, self.ncols)

    def append(self, arr: np.ndarray) -> None:
        """Append `arr` ([k] or [k, ncols]).  Cast is explicit and checked, never
        implicit.
        """
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        if self.ncols is None:
            assert arr.ndim == 1, f"{self.path}: expected 1-D, got {arr.shape}"
        else:
            assert arr.ndim == 2 and arr.shape[1] == self.ncols, \
                f"{self.path}: expected [k, {self.ncols}], got {arr.shape}"
        n = self._fp.write(arr.tobytes())
        # A short write on NFS is how ENOSPC shows up through a buffered handle.
        assert n == arr.nbytes, \
            f"{self.path}: short write {n} != {arr.nbytes} (ENOSPC?)"
        self.rows += int(arr.shape[0])

    def close(self) -> int:
        """Stamp the real shape into the reserved header.  Returns the file size in
        bytes.
        """
        if self._closed:
            return os.path.getsize(self.path)
        self._fp.flush()
        os.fsync(self._fp.fileno())
        self._fp.seek(0)
        self._fp.write(_npy_header_bytes(self.dtype, self.shape))
        self._fp.flush()
        os.fsync(self._fp.fileno())
        self._fp.close()
        self._closed = True
        return os.path.getsize(self.path)

    def abort(self) -> None:
        """Best-effort cleanup after a failed append; leaves no file that reads as
        valid.
        """
        try:
            self._fp.close()
        except Exception:  # pragma: no cover
            pass
        self._closed = True
        if os.path.exists(self.path):
            os.remove(self.path)


def write_npy(path: str, arr: np.ndarray) -> int:
    """Whole-array .npy write through the same fixed-header path (offsets, index
    mirrors).
    """
    w = _NpyAppendWriter(path, arr.dtype.str, None if arr.ndim == 1 else arr.shape[1])
    w.append(arr)
    return w.close()


def open_npy(path: str) -> np.ndarray:
    """Read-only memmap.  This is the ONLY way this module reads token payloads."""
    return np.load(path, mmap_mode="r")


def crc32_file(path: str, chunk: int = 1 << 23) -> str:
    """Streaming crc32.

    NOT sha256: a 48 GB shard sha256s in ~3.5 min (x224 shards = 13 h) whereas crc32
    streams at memory bandwidth.  The threat model here is truncation and bit-rot from
    ENOSPC or preemption, not adversarial substitution -- crc32 covers it.  The FM
    CHECKPOINTS are hashed with sha256 (measured 237 MiB/s => 23 s for the 5.4 GiB
    EpiAgent file), because those are provenance, not integrity.
    """
    c = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            c = zlib.crc32(b, c)
    return f"{c & 0xFFFFFFFF:08x}"


# ------------------------------------------------------------------------------------ #
# Writer
# ------------------------------------------------------------------------------------ #


class ShardWriter:
    """Streams ONE modality of ONE shard into a packed, offset-indexed directory.

    Usage (builder side)::

        w = ShardWriter(root, "train", "rna", 0, 96, spec, argv=sys.argv)
        for batch in loader:
            rc, rt, rm, ac, at, am, gs, cs, gid, aid = fm_tokens(model, batch, dev,
                                                                 include_summary=True,
                                                                 return_ids=True)
            w.append_batch(barcodes, tokens=rt, pad_mask=rm, ids=gid, values=gs,
                           cell=rc_valid_count, pad_emb=rna_pad, orig_rows=rows)
        w.finalize()

    The shard is built inside `<name>.tmp/` and os.replace()d into place only after
    `verify_shard()` and the caller's correspondence gate pass; `_DONE` is written LAST.
    A shard without `_DONE` must be recomputed in full, never partially trusted -- an 8
    TB job WILL be interrupted.
    """

    def __init__(self, root: str, split: str, modality: str, shard_idx: int,
                 num_shards: int, spec: Optional[CacheSpec] = None,
                 argv: Optional[Sequence[str]] = None,
                 extra_manifest: Optional[Dict] = None) -> None:
        assert modality in ("rna", "atac")
        assert 0 <= shard_idx < num_shards
        self.spec = spec or CacheSpec()
        self.root, self.split, self.modality = root, split, modality
        self.shard_idx, self.num_shards = shard_idx, num_shards
        self.final_dir = self.spec.shard_path(
            root, split, modality, shard_idx, num_shards)
        self.tmp_dir = self.final_dir + ".tmp"
        self.argv = list(argv or [])
        self.extra_manifest = dict(extra_manifest or {})

        if os.path.exists(self.tmp_dir):
            # a leftover .tmp is a failed run, never a resume
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=False)
        # OWNERSHIP STAMP.  A SIGKILL (cgroup OOM, preemption) leaves this .tmp behind
        # with the full payload in it and no handler can run, so the bytes are stranded
        # until THIS shard is retried.  Across a systematically-failing array that is
        # the whole build's worth of unpublished data on a shared volume.  The stamp
        # lets `sweep_stale_tmp()` distinguish "a dead job's orphan" from "a peer array
        # task's live work".
        with open(os.path.join(self.tmp_dir, TMP_OWNER_FILE), "w") as f:
            json.dump({"pid": os.getpid(), "host": os.uname().nodename,
                       "started": time.time(),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                       "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                       "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")},
                      f, indent=2, sort_keys=True)

        self._arrays = self.spec.arrays(modality)
        self._w: Dict[str, _NpyAppendWriter] = {
            a.name: _NpyAppendWriter(
                os.path.join(self.tmp_dir, a.name), a.dtype, a.ncols)
            for a in self._arrays
        }
        self._lengths: List[int] = []
        self._barcodes: List[str] = []
        self._orig_rows: List[int] = []
        self._finalized = False

    # --- append --------------------------------------------------------------------- #

    def append(self, barcode: str, tokens: np.ndarray, ids: np.ndarray,
               values: Optional[np.ndarray] = None, cell: Optional[np.ndarray] = None,
               pad_emb: Optional[np.ndarray] = None, orig_row: int = -1) -> None:
        """Append ONE cell's already-unpadded tokens.

        `tokens` [n, D], `ids` [n], `values` [n] (RNA only), `cell` [3072] or [512],
        `pad_emb` [768] (RNA only, when spec.store_rna_pad).  Everything is validated
        here rather than in verify_shard(), because a bad cell caught at append costs
        one cell and a bad cell caught at verify costs the whole shard.
        """
        tokens = np.asarray(tokens)
        ids = np.asarray(ids)
        n = int(tokens.shape[0])
        assert tokens.ndim == 2 and ids.shape == (n,), \
            f"tokens {tokens.shape} / ids {ids.shape} disagree"
        assert n > 0, f"{barcode}: empty cell"

        if self.modality == "rna":
            assert tokens.shape[1] == self.spec.rna_dim
            assert n <= self.spec.max_rna_len, \
                f"{barcode}: rna_len {n} > max_rna_nnz+2 = {self.spec.max_rna_len}"
            # gatherData emits STRICTLY ASCENDING panel columns for the valid prefix,
            # and the two meta columns are the highest indices -> they are the last two
            # entries.  This single assert catches any packing-order regression, any
            # off-by-one on the meta tokens, and any accidental inclusion of a pad
            # column (which would carry id 103 out of order).
            assert np.all(np.diff(ids.astype(np.int64)) > 0), \
                f"{barcode}: gene ids not strictly ascending (gatherData order broken)"
            assert int(ids[-2]) == self.spec.rna_meta_ids[0], \
                f"{barcode}: ids[n-2]={int(ids[-2])} != {self.spec.rna_meta_ids[0]}"
            assert int(ids[-1]) == self.spec.rna_meta_ids[1], \
                f"{barcode}: ids[n-1]={int(ids[-1])} != {self.spec.rna_meta_ids[1]}"
            assert int(ids[-3]) < self.spec.n_genes if n >= 3 else True
            assert values is not None and np.asarray(values).shape == (n,), \
                "RNA needs values"
            assert cell is not None
            assert np.asarray(cell).shape == (self.spec.rna_cell_dim,)
            self._w["rna_tokens.npy"].append(tokens)
            self._w["rna_gene_ids.npy"].append(ids)
            self._w["rna_values.npy"].append(np.asarray(values))
            self._w["rna_cell.npy"].append(np.asarray(cell)[None, :])
            if self.spec.store_rna_pad:
                assert pad_emb is not None, \
                    "spec.store_rna_pad=True but no pad_emb given"
                assert np.asarray(pad_emb).shape == (self.spec.rna_dim,)
                self._w["rna_pad.npy"].append(np.asarray(pad_emb)[None, :])
        else:
            assert tokens.shape[1] == self.spec.atac_dim
            assert n <= self.spec.max_atac_length, \
                f"{barcode}: atac_len {n} > cap {self.spec.max_atac_length}"
            i64 = ids.astype(np.int64)
            # include_summary=True keeps the FULL sequence: CLS at 0, SEP at n-1.
            # Getting this wrong by one token (the `ai` vs `ai[:, 1:]` trap) shifts
            # every cCRE's genome coordinate silently, because the ids stay valid --
            # they just belong to the neighbouring cCRE.
            assert i64[0] == self.spec.atac_cls_id, f"{barcode}: ids[0]={i64[0]} != CLS"
            assert i64[-1] == self.spec.atac_sep_id, \
                f"{barcode}: ids[-1]={i64[-1]} != SEP"
            if n > 2:
                mid = i64[1:-1]
                assert mid.min() >= self.spec.atac_id_offset and \
                    mid.max() < self.spec.atac_vocab_size, \
                    f"{barcode}: cCRE ids out of [{self.spec.atac_id_offset}, " \
                    f"{self.spec.atac_vocab_size})"
            assert cell is not None and np.asarray(cell).shape == (self.spec.atac_dim,)
            self._w["atac_tokens.npy"].append(tokens)
            self._w["atac_ccre_ids.npy"].append(ids)
            self._w["atac_cls.npy"].append(np.asarray(cell)[None, :])

        self._lengths.append(n)
        self._barcodes.append(str(barcode))
        self._orig_rows.append(int(orig_row))

    def append_batch(self, barcodes: Sequence[str], tokens, pad_mask, ids,
                     values=None, cell=None, pad_emb=None,
                     orig_rows: Optional[Sequence[int]] = None) -> None:
        """Append a padded [B, N, D] batch, slicing each cell by its own valid count.

        `pad_mask` is [B, N] with **True == PAD** (fm_tokens' polarity, unchanged).
        This is the ONE place in the pipeline where padded -> packed happens, so the
        polarity discipline lives here and nowhere else.
        """
        tokens = _to_numpy(tokens)
        pad_mask = _to_numpy(pad_mask).astype(bool)
        ids = _to_numpy(ids)
        values = None if values is None else _to_numpy(values)
        cell = None if cell is None else _to_numpy(cell)
        pad_emb = None if pad_emb is None else _to_numpy(pad_emb)

        b, n_pad = pad_mask.shape
        assert tokens.shape[:2] == (b, n_pad) and ids.shape[:2] == (b, n_pad), \
            f"tokens {tokens.shape} / ids {ids.shape} / mask {pad_mask.shape} disagree"
        assert len(barcodes) == b
        valid = ~pad_mask
        lens = valid.sum(1).astype(np.int64)
        # gatherData packs valid-FIRST and flash-attn pads at the END, so the valid
        # region is a strict PREFIX.  If it ever is not, slicing by count silently drops
        # real tokens and keeps padding -- verify that assumption instead of trusting
        # it.
        prefix = np.arange(n_pad)[None, :] < lens[:, None]
        assert np.array_equal(prefix, valid), \
            "valid tokens are not a contiguous prefix -- pad mask INVERTED (True " \
            "must mean PAD), or the padding convention changed; cf. the tombstone " \
            "at model.py:229-243"
        # A fully-masked row is the unmistakable signature of a wholesale polarity
        # inversion: rna_pad=1.0 / rna_valid=0.0 / rna_proj_norm=0.0 is exactly how it
        # presented last time.
        assert int(lens.min()) > 0, \
            "a cell has ZERO valid tokens -- pad mask is INVERTED (True must mean PAD)"
        for j in range(b):
            n = int(lens[j])
            self.append(
                barcodes[j], tokens[j, :n], ids[j, :n],
                values=None if values is None else values[j, :n],
                cell=None if cell is None else cell[j],
                pad_emb=None if pad_emb is None else pad_emb[j],
                orig_row=-1 if orig_rows is None else int(orig_rows[j]),
            )

    # --- finalize ------------------------------------------------------------------- #

    def finalize(self, verify: bool = True, extra: Optional[Dict] = None) -> Dict:
        """Close payloads, write offsets + index + shard_manifest, verify, publish,
        sentinel.
        """
        assert not self._finalized, "finalize() called twice"
        self._finalized = True
        n_rows = len(self._lengths)
        assert n_rows > 0, "refusing to publish an empty shard"

        sizes = {name: w.close() for name, w in self._w.items()}
        # The ownership stamp belongs to the .tmp, not to the published shard.
        stamp = os.path.join(self.tmp_dir, TMP_OWNER_FILE)
        if os.path.exists(stamp):
            os.remove(stamp)

        lengths = np.asarray(self._lengths, dtype=np.int64)
        offsets = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])  # int64 cumsum; see CacheSpec.offset_dtype
        off_name = self.spec.offsets_name(self.modality)
        sizes[off_name] = write_npy(os.path.join(self.tmp_dir, off_name), offsets)

        tok_name = self._arrays[0].name
        assert self._w[tok_name].rows == int(offsets[-1]), \
            f"token rows {self._w[tok_name].rows} != offsets[-1] {int(offsets[-1])}"

        idx_path = os.path.join(self.tmp_dir, "index.csv")
        with open(idx_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(SHARD_INDEX_COLUMNS)
            for i in range(n_rows):
                wcsv.writerow([i, self._barcodes[i], self._orig_rows[i],
                               int(lengths[i])])
        sizes["index.csv"] = os.path.getsize(idx_path)

        man = {
            "format_version": CACHE_FORMAT_VERSION,
            "split": self.split,
            "modality": self.modality,
            "shard_idx": self.shard_idx,
            "num_shards": self.num_shards,
            "n_rows": n_rows,
            "token_rows": int(offsets[-1]),
            "spec": self.spec.to_json(),
            "argv": self.argv,
            "length_stats": {
                "mean": float(lengths.mean()), "min": int(lengths.min()),
                "p50": float(np.median(lengths)), "max": int(lengths.max()),
                "frac_at_cap": float((lengths >= (self.spec.max_atac_length
                                                  if self.modality == "atac"
                                                  else self.spec.max_rna_len)).mean()),
            },
            "files": {n: {"bytes": int(sizes[n]),
                          "crc32": crc32_file(os.path.join(self.tmp_dir, n))}
                      for n in sorted(sizes)},
        }
        man.update(self.extra_manifest)
        if extra:
            man.update(extra)
        with open(os.path.join(self.tmp_dir, "shard_manifest.json"), "w") as f:
            json.dump(man, f, indent=2, sort_keys=True)

        if verify:
            # Verify in the .tmp dir, BEFORE publishing.  A shard that fails here never
            # becomes visible under its final name, so no consumer can pick it up.
            # The digests come from `man` rather than a second full read: they were
            # computed moments ago over these exact bytes, and re-reading 81 GB here is
            # ~15 min/shard the schedule never budgeted.  `verify_cache.py --full` is
            # the pass that recomputes from disk.
            verify_shard(self.tmp_dir, spec=self.spec, require_done=False, full=True,
                         known_crc={n: v["crc32"] for n, v in man["files"].items()})

        os.makedirs(os.path.dirname(self.final_dir), exist_ok=True)
        if os.path.exists(self.final_dir):
            shutil.rmtree(self.final_dir)
        os.replace(self.tmp_dir, self.final_dir)
        # _DONE is written LAST, after the atomic publish.  Its absence is the ONLY
        # signal that distinguishes a complete shard from a preempted one.
        with open(os.path.join(self.final_dir, "_DONE"), "w") as f:
            f.write(CACHE_FORMAT_VERSION + "\n")
        return man

    def abort(self) -> None:
        """Tear down after a failure.  Leaves nothing that could read as a finished
        shard.
        """
        for w in self._w.values():
            w.abort()
        if os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)


# ------------------------------------------------------------------------------------ #
# Orphaned .tmp bookkeeping  (a SIGKILL leaves one behind with the whole payload in it)
# ------------------------------------------------------------------------------------ #


def _dir_bytes(d: str) -> int:
    tot = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot


def iter_tmp_dirs(root: str, split: str) -> List[str]:
    """Every `<shard>.tmp` under this split, both modalities."""
    out: List[str] = []
    for modality in ("rna", "atac"):
        d = os.path.join(root, split, modality)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if name.endswith(".tmp") and os.path.isdir(p):
                out.append(p)
    return out


def tmp_bytes(root: str, split: str) -> int:
    """Bytes currently stranded in unpublished `.tmp` directories.

    Charged against free space by the builder's preflight: those bytes are already gone
    from the volume but are not yet part of any readable shard, so a gate that ignores
    them over-reports headroom on exactly the runs that need it most.
    """
    return sum(_dir_bytes(d) for d in iter_tmp_dirs(root, split))


def _owner_is_live(stamp: Dict) -> bool:
    """True if the SLURM job that created a .tmp is still running on this cluster.

    Conservative on every unknown: no stamp, no `squeue`, an unparsable answer -> treat
    the owner as LIVE and do not touch the directory.  Deleting a peer array task's
    in-flight shard would be far worse than leaving an orphan on disk.
    """
    jid = stamp.get("slurm_job_id")
    if not jid:
        return True
    try:
        import subprocess
        r = subprocess.run(["squeue", "-h", "-j", str(jid), "-o", "%T"],
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return True
    if r.returncode != 0:
        # squeue exits nonzero for an unknown job id -- that IS the "job is gone"
        # answer, but only when it says so; anything else is treated as unknown.
        return "Invalid job id" not in (r.stderr or "")
    return bool(r.stdout.strip())


def sweep_stale_tmp(root: str, split: str, min_age_s: float = 3600.0,
                    dry_run: bool = False, log_fn=None) -> Dict:
    """Garbage-collect `.tmp` shard dirs whose owning job is gone.

    A `.tmp` is ALWAYS a failed run (see ShardWriter), but it is reclaimed only when
    that exact shard is retried -- so a systematic mid-finalize kill across a 96-task
    array strands the whole build's worth of bytes on a shared filesystem.  This is the
    sweep the RUNBOOK points at after any failed array.

    Safety: a directory is removed ONLY when its owner stamp says the SLURM job is no
    longer queued/running AND it is older than `min_age_s`.  Anything unknown is kept.
    """
    out = {"removed": [], "kept": [], "bytes_freed": 0}
    now = time.time()
    for d in iter_tmp_dirs(root, split):
        stamp: Dict = {}
        sp = os.path.join(d, TMP_OWNER_FILE)
        if os.path.exists(sp):
            try:
                with open(sp) as f:
                    stamp = json.load(f)
            except Exception:
                stamp = {}
        age = now - float(stamp.get("started", os.path.getmtime(d)))
        why = None
        if age < min_age_s:
            why = f"only {age / 60:.0f} min old (< {min_age_s / 60:.0f})"
        elif not stamp:
            why = "no owner stamp (pre-stamp build, or a partial makedirs)"
        elif _owner_is_live(stamp):
            why = f"owning job {stamp.get('slurm_job_id')} still in squeue"
        if why:
            out["kept"].append({"dir": d, "why": why})
            continue
        n = _dir_bytes(d)
        if not dry_run:
            shutil.rmtree(d, ignore_errors=True)
        out["removed"].append({"dir": d, "bytes": n, "owner": stamp})
        out["bytes_freed"] += n
    if log_fn:
        for r in out["removed"]:
            log_fn(f"  GC {'(dry) ' if dry_run else ''}{r['dir']} "
                   f"({r['bytes'] / 1e9:.1f} GB)")
        for k in out["kept"]:
            log_fn(f"  keep {k['dir']}: {k['why']}")
    return out


# ------------------------------------------------------------------------------------ #
# Global index: assembled from the per-shard slices
# ------------------------------------------------------------------------------------ #


#: Filename pattern of a per-shard global-index slice, written by the builder.
INDEX_SLICE_GLOB = "index_shard"


def _parse_slice_name(name: str) -> Tuple[int, int]:
    """`index_shard0007of0096.npz` -> (7, 96)."""
    stem = name.split(".")[0]
    body = stem[len(INDEX_SLICE_GLOB):]
    a, b = body.split("of")
    return int(a), int(b)


def _fmt_shard_ids(ids: Iterable[int], max_parts: int = 12) -> str:
    """`[0,1,2,3,7]` -> `0-3,7`.  Compact enough to name 48 shards inside an assert."""
    xs = sorted({int(i) for i in ids})
    if not xs:
        return "(none)"
    runs: List[List[int]] = [[xs[0], xs[0]]]
    for x in xs[1:]:
        if x == runs[-1][1] + 1:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in runs]
    tail = "" if len(parts) <= max_parts else f",...(+{len(parts) - max_parts})"
    return ",".join(parts[:max_parts]) + tail


def _parse_shard_dirname(name: str) -> Tuple[int, int]:
    """`shard0007of0096` -> (7, 96).  ValueError on anything else.

    The SHARD DIRECTORY name carries the planned geometry the same way the index slice
    name does, which is what lets `CacheReader` answer "which shards are on disk" on a
    cache whose root manifest is missing -- without falling back to the index's own max,
    the one number that by construction cannot see a shard the index omits.
    """
    if not name.startswith("shard") or "of" not in name:
        raise ValueError(name)
    a, b = name[len("shard"):].split("of", 1)
    return int(a), int(b)


def assemble_index_from_slices(root: str, split: str,
                               expect_shards: Optional[int] = None,
                               allow_partial: Optional[Iterable[int]] = None
                               ) -> Dict[str, np.ndarray]:
    """Concatenate `<root>/<split>/index_shardNNNNofMMMM.npz` into the global index.

    THE COMPLETENESS ASSERT IS THE POINT.  A SLURM array element that never starts
    leaves no shard directory and no `.tmp`, so a checker that walks `os.listdir` sees
    a smaller cache and calls it healthy: every barcode it holds is real, every
    coverage ratio is 1.0, and ~8,200 cells are simply absent from an 8 TB corpus with
    nothing on disk recording it.  The slice filename already carries `ofMMMM`; this
    reads it and refuses a gap.

    `allow_partial` is the ONE way to express "short BY DESIGN" -- an incremental
    campaign that deliberately built only `ARRAY=0-47` of 96.  It is graded by EXACT
    SET EQUALITY, never as a subset: a cache short by exactly the declared range is a
    half-build, a cache short by anything else is a hole, and collapsing the two would
    give back the very blindness the gap assert exists to remove.  The default (None)
    keeps the full-range rule, so nothing that does not ask for a partial index can
    ever get one.
    """
    d = os.path.join(root, split)
    names = sorted(f for f in os.listdir(d)
                   if f.startswith(INDEX_SLICE_GLOB) and f.endswith(".npz")) \
        if os.path.isdir(d) else []
    assert names, (
        f"{root}: no index.npz/index.csv at the root and no "
        f"{INDEX_SLICE_GLOB}*.npz under {d} -- nothing maps a barcode to a "
        f"(shard, row)")
    parsed = [_parse_slice_name(n) for n in names]
    totals = sorted({t for _, t in parsed})
    assert len(totals) == 1, (
        f"{d}: index slices claim different shard counts {totals} -- the split was "
        "built under two different --num_shards and the rows cannot be one corpus")
    total = totals[0]
    if expect_shards is not None:
        assert total == expect_shards, (
            f"{d}: index slices say num_shards={total} but the root manifest says "
            f"{expect_shards}")
    have = sorted(i for i, _ in parsed)
    if allow_partial is None:
        want = set(range(total))
        missing = sorted(want - set(have))
        assert not missing, (
            f"{d}: index slices present for {len(have)} of {total} shards; MISSING "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}.  Those cells are "
            f"absent from the cache -- rebuild with ARRAY="
            f"{','.join(str(m) for m in missing[:20])}")
    else:
        want = {int(i) for i in allow_partial}
        assert want, "allow_partial is empty: an index of no shards is not a cache"
        assert want <= set(range(total)), (
            f"{d}: allow_partial names shard(s) outside 0..{total - 1}: "
            f"{sorted(want - set(range(total)))[:20]}")
        # EXACT equality, both directions.  A subset test would let a hole inside the
        # declared range pass as "partial by design"; a superset test would let an
        # unrelated shard smuggle itself in.
        short = sorted(want - set(have))
        extra = sorted(set(have) - want)
        assert not short and not extra, (
            f"{d}: partial index does not match the declared range.  MISSING from it "
            f"{short[:20]}{'...' if len(short) > 20 else ''}; UNEXPECTEDLY PRESENT "
            f"{extra[:20]}{'...' if len(extra) > 20 else ''}.  A cache may be short BY "
            f"DESIGN, never short by accident -- rebuild with ARRAY="
            f"{','.join(str(m) for m in short[:20])}")
    assert len(set(have)) == len(have), f"{d}: duplicate index slices for {have}"

    cols: Dict[str, List[np.ndarray]] = {}
    for name in names:
        z = np.load(os.path.join(d, name), allow_pickle=False)
        for k in z.files:
            # S64 -> str.  A bytes/str mismatch is the kind of quiet membership bug that
            # empties a filter without ever raising (`b"val" != "val"`).
            v = z[k].astype(str) if z[k].dtype.kind == "S" else z[k]
            cols.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(out["barcode"])
    assert all(len(v) == n for v in out.values()), "index slices have ragged columns"
    assert len(set(out["barcode"].tolist())) == n, \
        f"{d}: the assembled index has duplicate barcodes"
    return out


def write_global_index(root: str, split: str, expect_shards: Optional[int] = None,
                       allow_partial: Optional[Iterable[int]] = None) -> Dict:
    """Merge the per-shard slices into `<root>/index.npz` + `<root>/index.csv`.

    `CacheReader` (and therefore `CachedTokenDataset`) opens the ROOT index; without
    this step a finished 8 TB build cannot be opened at all.  Idempotent, and it appends
    to whatever other splits are already in the root index rather than replacing them.

    `allow_partial` forwards to `assemble_index_from_slices`, which grades it by EXACT
    set equality -- see that function.  Use it only for a deliberately incremental
    campaign (`--merge_index --shard_set 0-47`), and record the range in the manifest so
    the resulting index cannot be mistaken for a whole one.
    """
    idx = assemble_index_from_slices(root, split, expect_shards, allow_partial)
    npz_path = os.path.join(root, "index.npz")
    keep: Dict[str, np.ndarray] = {}
    if os.path.exists(npz_path):
        z = np.load(npz_path, allow_pickle=False)
        prev = {k: (z[k].astype(str) if z[k].dtype.kind == "S" else z[k])
                for k in z.files}
        if "split" in prev:
            other = prev["split"] != split
            if other.any():
                keep = {k: v[other] for k, v in prev.items()}
    if keep:
        assert set(keep) == set(idx), \
            "the existing index.npz has different columns than the new slices"
        idx = {k: np.concatenate([keep[k], idx[k]]) for k in idx}
    enc = {k: (np.asarray(v, dtype="S64") if v.dtype.kind in "UO" else v)
           for k, v in idx.items()}
    np.savez(npz_path + ".tmp.npz", **enc)
    os.replace(npz_path + ".tmp.npz", npz_path)
    csv_path = os.path.join(root, "index.csv")
    cols = list(idx)
    n = len(idx["barcode"])
    with open(csv_path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([idx[c][i] for c in cols])
    os.replace(csv_path + ".tmp", csv_path)
    return {"rows": n, "index_npz": npz_path, "index_csv": csv_path,
            "splits": sorted(set(idx["split"].tolist())) if "split" in idx else []}


def _to_numpy(x) -> np.ndarray:
    """torch -> numpy without importing torch at module scope for the writer's sake."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ------------------------------------------------------------------------------------ #
# Verifier
# ------------------------------------------------------------------------------------ #


def verify_shard(shard_dir: str, spec: Optional[CacheSpec] = None,
                 require_done: bool = True, full: bool = False,
                 nan_sample_rows: int = 1 << 20,
                 known_crc: Optional[Dict[str, str]] = None) -> Dict:
    """Structural check of one shard directory.  Raises AssertionError; returns stats.

    `full=True` also re-computes every crc32 and scans EVERY token row for NaN/Inf (used
    at finalize, when the pages are still hot).  `full=False` samples, for a cheap
    re-check on read.

    `known_crc` is `{filename: crc32}` for digests the CALLER has just computed over the
    same bytes.  `finalize()` passes the digests it wrote into the manifest, because
    re-reading an 81 GB shard a second time costs ~15 min/shard of pure NFS read that
    the wall-clock budget never accounted for (and evicts the page cache that the first
    read warmed).  The standalone acceptance pass -- `verify_cache.py --full` -- passes
    nothing, so it DOES recompute from disk, which is where a rot check belongs.
    """
    assert os.path.isdir(shard_dir), f"no such shard dir: {shard_dir}"
    man_path = os.path.join(shard_dir, "shard_manifest.json")
    assert os.path.exists(man_path), f"{shard_dir}: no shard_manifest.json"
    with open(man_path) as f:
        man = json.load(f)
    assert man["format_version"] == CACHE_FORMAT_VERSION, \
        f"{shard_dir}: format {man['format_version']} != {CACHE_FORMAT_VERSION}"
    spec = spec or CacheSpec.from_json(man["spec"])
    # A spec mismatch means the bytes mean something other than what the caller thinks.
    assert man["spec"] == spec.to_json(), \
        f"{shard_dir}: spec mismatch vs the caller's spec"
    if require_done:
        assert os.path.exists(os.path.join(shard_dir, "_DONE")), \
            f"{shard_dir}: no _DONE sentinel -- shard is INCOMPLETE, recompute it"

    modality = man["modality"]
    arrays = spec.arrays(modality)
    offsets = open_npy(os.path.join(shard_dir, spec.offsets_name(modality)))
    assert offsets.dtype == np.int64, \
        f"offsets dtype {offsets.dtype} != int64 (overflow risk)"
    n_rows = int(man["n_rows"])
    assert offsets.shape == (n_rows + 1,), f"offsets {offsets.shape} != ({n_rows + 1},)"
    assert int(offsets[0]) == 0, "offsets[0] != 0"
    d = np.diff(offsets.astype(np.int64))
    assert np.all(d >= 0), "offsets are not monotone non-decreasing"
    assert np.all(d > 0), "a cell has zero tokens"

    # index.csv must agree with the offsets, independently derived.
    with open(os.path.join(shard_dir, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_rows, \
        f"index.csv has {len(rows)} rows, manifest says {n_rows}"
    idx_len = np.asarray([int(r["length"]) for r in rows], dtype=np.int64)
    assert np.array_equal(idx_len, d), "index.csv 'length' disagrees with diff(offsets)"
    bcs = [r["barcode"] for r in rows]
    assert len(set(bcs)) == n_rows, f"{shard_dir}: duplicate barcodes in index.csv"

    cap = spec.max_atac_length if modality == "atac" else spec.max_rna_len
    assert int(d.max()) <= cap, f"length {int(d.max())} exceeds cap {cap}"

    stats = {"n_rows": n_rows, "token_rows": int(offsets[-1]), "modality": modality,
             "len_mean": float(d.mean()), "len_max": int(d.max())}

    for a in arrays:
        p = os.path.join(shard_dir, a.name)
        assert os.path.exists(p), f"{shard_dir}: missing {a.name}"
        arr = open_npy(p)
        assert arr.dtype == np.dtype(a.dtype), \
            f"{a.name}: dtype {arr.dtype} != {a.dtype}"
        want_rows = int(offsets[-1]) if a.kind == "token" else n_rows
        assert arr.shape[0] == want_rows, \
            f"{a.name}: {arr.shape[0]} rows, expected {want_rows}"
        if a.ncols is None:
            assert arr.ndim == 1, f"{a.name}: expected 1-D"
        else:
            assert arr.shape[1:] == (a.ncols,), \
                f"{a.name}: {arr.shape} != [*, {a.ncols}]"
        if full and "crc32" in man["files"].get(a.name, {}):
            want = man["files"][a.name]["crc32"]
            got = (known_crc or {}).get(a.name)
            if got is None:
                got = crc32_file(p)
            assert got == want, \
                f"{a.name}: crc32 {got} != {want} -- TRUNCATED/ROTTED"
            # Size is checked from disk even when the digest was supplied, so a file
            # truncated between the caller's read and now still cannot slip through.
            assert os.path.getsize(p) == man["files"][a.name]["bytes"], \
                f"{a.name}: size on disk != manifest -- TRUNCATED"

    tok = open_npy(os.path.join(shard_dir, arrays[0].name))
    ids = open_npy(os.path.join(shard_dir, arrays[1].name))

    # NaN/Inf.  fp16 overflow from an un-guarded activation would show up here and
    # nowhere else until it poisons a loss 6 hours into training.
    #
    # SCANNED IN BOUNDED CHUNKS, and that is not a micro-optimisation.  A train ATAC
    # shard is ~51e6 x 512 fp16 = 52 GB on disk; `np.asarray(whole_memmap,
    # dtype=np.float32)` COPIES it to 105 GB of anonymous RAM and `np.isfinite()` then
    # allocates another 26 GB of bool -- 131 GB against the launcher's --mem=96G, i.e.
    # a guaranteed cgroup OOM-kill inside finalize(), AFTER the whole shard has been
    # written and with SIGKILL uncatchable so `abort()` never runs.  Chunking holds
    # peak RAM at ~1.5 GB regardless of shard size.
    if full or tok.shape[0] <= nan_sample_rows:
        head = None
        for a in range(0, tok.shape[0], NAN_SCAN_CHUNK_ROWS):
            blk = np.asarray(tok[a:a + NAN_SCAN_CHUNK_ROWS], dtype=np.float32)
            assert np.isfinite(blk).all(), \
                f"{arrays[0].name}: NaN/Inf in the token payload (rows {a}..)"
            if head is None:
                head = blk[: min(4096, blk.shape[0])].copy()
            del blk
        slab_head = head if head is not None else np.zeros((0, 1), np.float32)
    else:
        rows_to_scan = np.linspace(0, tok.shape[0] - 1,
                                   nan_sample_rows).astype(np.int64)
        slab = np.asarray(tok[rows_to_scan], dtype=np.float32)
        assert np.isfinite(slab).all(), \
            f"{arrays[0].name}: NaN/Inf in the token payload"
        slab_head = slab[: min(4096, slab.shape[0])]

    # L2 tripwire: these MUST be raw activations.  An accidental normalisation anywhere
    # in the builder would put every row norm at 1.0 and silently change the trained
    # function.
    norms = np.linalg.norm(slab_head, axis=1) if slab_head.ndim == 2 \
        else np.zeros(0, np.float32)
    if norms.size:
        med = float(np.median(norms))
        assert not (0.99 <= med <= 1.01), (
            f"{arrays[0].name}: median row norm {med:.4f} ~ 1 -- "
            "tokens look L2-NORMALISED")
        stats["median_token_norm"] = med

    lo = offsets[:-1].astype(np.int64)
    hi = offsets[1:].astype(np.int64)
    if modality == "rna":
        i64 = np.asarray(ids, dtype=np.int64)
        assert np.array_equal(i64[hi - 2], np.full(n_rows, spec.rna_meta_ids[0])), \
            "some cell does not carry meta id 19264 at position n-2"
        assert np.array_equal(i64[hi - 1], np.full(n_rows, spec.rna_meta_ids[1])), \
            "some cell does not carry meta id 19265 at position n-1"
        assert np.all(i64[hi - 3] < spec.n_genes), \
            "a third-from-last token is not a gene"
        # Strictly ascending WITHIN each cell.  diff over the flat array is >0
        # everywhere except at cell boundaries, so mask those out rather than looping
        # over 800k cells.
        dif = np.diff(i64)
        boundary = np.zeros(dif.shape[0], dtype=bool)
        boundary[hi[:-1] - 1] = True
        assert np.all(dif[~boundary] > 0), \
            "gene ids are not strictly ascending within a cell"
        cell = open_npy(os.path.join(shard_dir, "rna_cell.npy"))
        head = np.asarray(cell[: min(4096, n_rows)], dtype=np.float32)
        assert np.isfinite(head).all(), "rna_cell.npy carries NaN/Inf"
    else:
        i64 = np.asarray(ids, dtype=np.int64)
        assert np.all(i64[lo] == spec.atac_cls_id), "some cell does not start with CLS"
        assert np.all(i64[hi - 1] == spec.atac_sep_id), \
            "some cell does not end with SEP"
        interior = np.ones(i64.shape[0], dtype=bool)
        interior[lo] = False
        interior[hi - 1] = False
        mid = i64[interior]
        if mid.size:
            in_range = (mid.min() >= spec.atac_id_offset
                        and mid.max() < spec.atac_vocab_size)
            assert in_range, \
                "cCRE ids outside [4, vocab) -- was +4 applied twice, or removed?"
        # atac_cls is a deliberate denormalisation of atac_tokens[offset]; prove it.
        cls = open_npy(os.path.join(shard_dir, "atac_cls.npy"))
        k = min(1024, n_rows)
        sel = np.linspace(0, n_rows - 1, k).astype(np.int64)
        assert np.array_equal(np.asarray(cls[sel]), np.asarray(tok[lo[sel]])), \
            "atac_cls != atac_tokens[offset] -- the CLS denormalisation drifted"

    return stats


# ------------------------------------------------------------------------------------ #
# Reader
# ------------------------------------------------------------------------------------ #


class CacheReader:
    """Memmap-backed reader.  `get(i)` returns one cell; `collate()` rebuilds
    fm_tokens().

    Shards are NEVER merged: each shard directory is a self-contained offset-indexed
    cache with offsets rebased to 0, and the global index carries (shard, row) per
    modality. Merging 8.1 TB would cost +4.0 h of pure I/O at the measured 563 MB/s
    write ceiling and would need 8.1 TB of extra free space we do not have -- for no
    benefit, since np.load(mmap_mode="r") over 224 files is the same page-cache
    behaviour as over one.

    The reader opens shard memmaps lazily and caches the handles, so a job that touches
    one shard pays for one shard.

    TWO ways a cache can be legitimately SHORT, and both are explicit here:
      * short by MODALITY -- `modalities`, resolved from what the cache RECORDS it
        holds; a modality it does not hold contributes no keys at all.
      * short by SHARD RANGE -- a root `index.npz` merged with `--merge_index
        --shard_set 0-47`.  `allow_partial_index=True` is the caller saying "yes, serve
        me the merged range"; without it a root index that is short RELATIVE TO THE
        SHARDS ON DISK is refused by name (see `_check_index_completeness`).  Either
        way the resolved state is recorded in `self.index_completeness` /
        `self.partial_index`, so no caller can be partial without being able to say so.
    """

    def __init__(self, root: str, split: str, spec: Optional[CacheSpec] = None,
                 index: Optional[Dict[str, np.ndarray]] = None,
                 require_done: bool = True,
                 modalities: Optional[Sequence[str]] = None,
                 allow_partial_index: bool = False) -> None:
        self.root, self.split = root, split
        self.require_done = require_done
        man_path = os.path.join(root, "manifest.json")
        self.manifest = {}
        if os.path.exists(man_path):
            with open(man_path) as f:
                self.manifest = json.load(f)
            disk = CacheSpec.from_json(self.manifest["spec"])
            if spec is not None:
                # REFUSE on mismatch.  A cache read under the wrong caps is exactly the
                # "consumer reads an 8192 cache assuming 4096" failure -- silent, and it
                # reintroduces the ATAC-truncation deficit this cache exists to remove.
                assert disk.to_json() == spec.to_json(), (
                    "cache spec != consumer spec; refusing to open.\n"
                    f"  disk:     {disk.to_json()}\n  consumer: {spec.to_json()}")
            spec = disk
        self.spec = spec or CacheSpec()

        # WHICH MODALITIES THIS CACHE ACTUALLY HOLDS.  The corpus is built in two
        # passes (`--modality rna` then `--modality atac`) and, on a near-full volume,
        # the RNA half can land days before the ATAC half.  Between those two moments
        # the ATAC shard directories DO NOT EXIST, so an unconditional ATAC read raised
        # a `_DONE` AssertionError on the first `get()` -- after `__init__` had already
        # printed a healthy cell count, i.e. a training run that looked fine through
        # setup and died at step 0.  Resolution order: the caller's explicit request,
        # then what the cache RECORDS about itself, then both (the pre-completeness
        # default).  A modality this reader does not hold is never zero-filled: a zeros
        # ATAC block is indistinguishable from a genuine flash-attn pad output (design
        # L10) and would train silently, so the keys are OMITTED and asking for them is
        # a KeyError that names the cache.
        declared = (self.manifest.get("splits", {}).get(split, {})
                    .get("modalities"))
        mods = modalities if modalities is not None else (
            declared if declared else MODALITIES)
        self.modalities: Tuple[str, ...] = tuple(str(m) for m in mods)
        assert self.modalities and set(self.modalities) <= set(MODALITIES), \
            f"modalities must be a non-empty subset of {MODALITIES}, got {mods!r}"
        if declared:
            missing = sorted(set(self.modalities) - set(declared))
            assert not missing, (
                f"{root}: this reader asks for {missing} but the cache RECORDS only "
                f"{sorted(declared)} as built for split {split!r}.  Build the missing "
                f"pass, or open the cache with modalities={tuple(declared)}.")

        self.index = index if index is not None else self._load_index()
        self.n = len(self.index["barcode"])
        # THE ROOT INDEX IS GRADED AGAINST THE SHARDS ON DISK, HERE, ON THE DEFAULT
        # PATH.  `allow_partial` (`--merge_index --shard_set 0-47`) made a
        # legitimately-short root index.npz possible, and nothing re-checked it once the
        # rest of the campaign landed: a COMPLETE both-modality cache whose stale
        # partial merge was never redone opened through THIS constructor at half its
        # cells, with every barcode real and no modality refusal able to see it (both
        # passes are built).  Not bypassable by the default: `allow_partial_index` has
        # to be asked for, and what it resolved to is recorded either way.
        self.index_completeness: Dict[str, object] = self._check_index_completeness(
            bool(allow_partial_index), caller_index=index is not None)
        self.partial_index: bool = \
            self.index_completeness["status"] == "PARTIAL_BY_DESIGN"
        self._mm: Dict[Tuple[str, int, str], np.ndarray] = {}
        self._checked: set = set()

    # --- index ---------------------------------------------------------------------- #

    def _load_index(self) -> Dict[str, np.ndarray]:
        npz = os.path.join(self.root, "index.npz")
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=False)
            # String columns are stored as fixed-width bytes (S64) because npz with
            # allow_pickle=False cannot hold an object array.  Decode them back to str
            # here so downstream comparisons ("split" == "val", barcode joins) are not
            # silently False against b"val" -- a bytes/str mismatch is exactly the kind
            # of quiet membership bug that empties a filter without raising.
            cols = {k: (z[k].astype(str) if z[k].dtype.kind == "S" else z[k])
                    for k in z.files}
            keep = (cols["split"] == self.split) if "split" in cols \
                else np.ones(len(cols["barcode"]), bool)
            return {k: v[keep] for k, v in cols.items()}
        csv_path = os.path.join(self.root, "index.csv")
        if not os.path.exists(csv_path):
            # The builder writes one index SLICE per shard and leaves the merge to
            # `build_fm_token_cache.py --merge_index` (RUNBOOK step 6).  Falling back to
            # assembling them here means a cache whose merge step was forgotten still
            # OPENS -- but `assemble_index_from_slices` refuses a gap in 0..MMMM-1, so
            # it cannot quietly serve a cache that is missing a shard.
            n_sh = int(self.manifest.get("splits", {}).get(self.split, {})
                       .get("rna_shards", 0)) or None
            return assemble_index_from_slices(self.root, self.split, n_sh)
        with open(csv_path) as f:
            rows = [r for r in csv.DictReader(f)
                    if r.get("split", self.split) == self.split]
        int_cols = ("orig_row", "label", "rna_shard", "rna_row", "rna_len", "rna_nnz",
                    "atac_shard", "atac_row", "atac_len", "atac_raw_len")
        out: Dict[str, np.ndarray] = {}
        for k in rows[0]:
            if k in int_cols:
                out[k] = np.asarray([int(r[k]) for r in rows], dtype=np.int64)
            else:
                out[k] = np.asarray([r[k] for r in rows])
        return out

    def __len__(self) -> int:
        return self.n

    # --- is the ROOT INDEX short relative to the payload? ---------------------------- #

    def _check_index_completeness(self, allow_partial_index: bool,
                                  caller_index: bool) -> Dict[str, object]:
        """REFUSE a root index that is SHORT relative to the shards published on disk.

        THE MISTAKE THIS CATCHES, EXACTLY.  `--merge_index --shard_set 0-47` is a
        required step of the incremental campaign, not a misuse: it writes a root
        `index.npz` covering 48 of 96 shards and records the declared range in
        `manifest["splits"][split]["completeness"]["index_shards"]`.  Shards 48-95 land
        days later.  If nobody re-runs the merge, the cache is COMPLETE on disk and its
        index is HALF -- and every barcode in that index is real, every length matches,
        every coverage ratio reads 1.0, so nothing downstream can see the ~383,000
        missing cells.  The modality refusal in `__init__` cannot catch it either: BOTH
        passes are built, so `modalities` is ["rna", "atac"] and the default cross-modal
        constructor opens happily at half the corpus.  It is the same class of silent
        shortness the `ofMMMM` gap assert removes, one level up: that assert grades the
        SLICES against their own filenames, this grades the MERGED INDEX against the
        payload.

        WHAT IS COMPARED, AND WHY IT IS O(1) IN CELLS.
          * `completeness["index_shards"]` -- written by `seal_completeness()` at every
            `--shard_set` merge and, until this check, READ BY NOBODY -- is the DECLARED
            range.  It is read here, and cross-checked: a record that OVER-claims (says
            0-47, index holds 0-23) is a stale seal hiding a hole and is refused; a
            record that under-claims is stale but harmless, and the index's own content
            wins.
          * the shard ids the index ACTUALLY holds come from one `np.unique` over an int
            column already in RAM -- no I/O, no join.
          * the payload side is the `_DONE` sentinels: one `os.path.exists` per shard,
            96 per modality, no shard opened and no shard `index.csv` read.
        The O(N) version -- assert every barcode and length in the root index against
        the shard row it points at -- deliberately does NOT run here.  It belongs at
        `--merge_index`: a single-writer, whole-split step that already reads every
        slice and runs once per campaign, not once per DataLoader worker.  Paying a
        766k-row join at every open, for a fact that cannot change while the process
        runs, is the wrong trade.  It is item 3 of the PRE-ATAC checklist in RUNBOOK.md.

        A GENUINELY PARTIAL CACHE STILL OPENS.  The RNA-first schedule merges `val` in
        full and `train` with `--shard_set 0-47` while only 0-47 exist, so nothing is
        short RELATIVE TO DISK and this check is silent.  It fires only once the payload
        overtakes the index; then `allow_partial_index=True` is the caller explicitly
        asking for the merged range.  Both outcomes are recorded in
        `self.index_completeness`.
        """
        state: Dict[str, object] = {
            "status": "COMPLETE", "split": self.split,
            "modalities": list(self.modalities), "n_index_rows": int(self.n),
            "index_shards_declared": None, "index_shards_covered": [],
            "published_on_disk": {}, "missing_from_index": {},
            "explicitly_allowed": bool(allow_partial_index)}
        if caller_index:
            # The caller handed us a VIEW -- a stratified parity sample, one shard's own
            # slice, a whitelist-filtered index.  Grading a deliberate subset against the
            # payload would refuse every one of them, and the staleness question belongs
            # to the ROOT index: the object this reader did not load.
            state["status"] = "NOT_GRADED_CALLER_INDEX"
            return state
        comp = (self.manifest.get("splits", {}).get(self.split, {})
                .get("completeness") or {})
        declared = comp.get("index_shards")
        if declared is not None:
            state["index_shards_declared"] = sorted(int(i) for i in declared)
        covered: Set[int] = set()
        for m in MODALITIES:
            col = self.index.get(f"{m}_shard")
            if col is not None and len(col):
                covered |= {int(v) for v in np.unique(np.asarray(col))}
        state["index_shards_covered"] = sorted(covered)
        if not covered:
            # No (shard, row) columns at all: this is not an index this reader can
            # dispatch with, and `get()` will say so with the missing key.  Nothing to
            # grade, and inventing a verdict here would be worse than saying so.
            state["status"] = "NOT_GRADED_NO_SHARD_COLUMNS"
            return state
        if declared is not None:
            over = sorted(set(state["index_shards_declared"]) - covered)
            assert not over, (
                f"{self.root}: manifest['splits'][{self.split!r}]['completeness']"
                f"['index_shards'] declares {_fmt_shard_ids(declared)} but the root "
                f"index holds only {_fmt_shard_ids(sorted(covered))} -- the seal "
                f"OVER-claims by {_fmt_shard_ids(over)}, i.e. it is recording a range "
                f"the index does not cover.  Re-run --merge_index for this split.")

        n_ref = 0
        missing_any = False
        for m in self.modalities:
            n_sh = self._n_shards(m)
            n_ref = max(n_ref, n_sh)
            pub = self._published_on_disk(m, n_sh)
            state["published_on_disk"][m] = sorted(pub)
            miss = sorted(pub - covered)
            state["missing_from_index"][m] = miss
            missing_any = missing_any or bool(miss)
        if not missing_any:
            return state
        if allow_partial_index:
            state["status"] = "PARTIAL_BY_DESIGN"
            return state
        detail = "; ".join(
            f"{m}: {len(covered & set(state['published_on_disk'][m]))} of "
            f"{len(state['published_on_disk'][m])} published shards are in the index, "
            f"MISSING {_fmt_shard_ids(state['missing_from_index'][m])}"
            for m in self.modalities if state["missing_from_index"][m])
        raise AssertionError(
            f"{self.root}: the ROOT INDEX for split {self.split!r} is SHORT relative to "
            f"the shards on disk -- {detail}.  It was merged over "
            f"{_fmt_shard_ids(sorted(covered))}"
            + (f" (declared: manifest['splits'][{self.split!r}]['completeness']"
               f"['index_shards'])" if declared is not None else "")
            + f" and NOT re-merged after the rest of the campaign landed, so opening it "
            f"would serve {self.n} cells and silently omit every cell of the shards "
            f"named above -- each barcode it does hold being real, nothing downstream "
            f"can see the hole.  Fix it; NO PAYLOAD IS REBUILT:\n"
            f"    python build_fm_token_cache.py --merge_index --split {self.split} "
            f"--num_shards {n_ref} --out_dir {self.root}\n"
            f"  If you genuinely want only the merged range, say so: "
            f"CacheReader(..., allow_partial_index=True) / "
            f"CachedTokenDataset(..., allow_partial_index=True).")

    def _published_on_disk(self, modality: str, n_shards: int) -> Set[int]:
        """Shard ids of `modality` that are PUBLISHED (`_DONE` present) right now.

        The same test `build_fm_token_cache.published_shards()` uses, duplicated rather
        than imported because the dependency runs the other way -- this module must stay
        free of the builder (and of GPU/model/h5py) so it self-tests on a login node.
        Cost: one `os.path.exists` per shard, once per reader.
        """
        return {i for i in range(int(n_shards))
                if os.path.exists(os.path.join(
                    self.spec.shard_path(self.root, self.split, modality, i,
                                         int(n_shards)), "_DONE"))}

    def _n_shards(self, modality: str) -> int:
        """Planned shard count for `modality` -- the shard-directory naming key.

        Manifest first (the PLANNED geometry, which stays at `num_shards` for both
        modalities whatever `--modality` built), then the shard directory NAMES, which
        carry `ofMMMM`, and only then the index.  The index's own max is the last resort
        ON PURPOSE: it is exactly the number that cannot see a shard the index omits,
        which is the hole `_check_index_completeness` exists to find.
        """
        n = int(self.manifest.get("splits", {}).get(self.split, {})
                .get(f"{modality}_shards", 0))
        if n:
            return n
        d = os.path.join(self.root, self.split, modality)
        totals: Set[int] = set()
        if os.path.isdir(d):
            for name in os.listdir(d):
                if name.endswith(".tmp"):
                    continue
                try:
                    totals.add(_parse_shard_dirname(name)[1])
                except ValueError:
                    continue
        if len(totals) == 1:
            return totals.pop()
        col = self.index.get(f"{modality}_shard")
        return int(np.asarray(col).max()) + 1 if col is not None and len(col) else 0

    # --- shard handles -------------------------------------------------------------- #

    def _shard_dir(self, modality: str, shard_idx: int) -> str:
        return self.spec.shard_path(
            self.root, self.split, modality, shard_idx, self._n_shards(modality))

    def _arr(self, modality: str, shard_idx: int, name: str) -> np.ndarray:
        key = (modality, shard_idx, name)
        mm = self._mm.get(key)
        if mm is None:
            d = self._shard_dir(modality, shard_idx)
            if (modality, shard_idx) not in self._checked:
                if self.require_done:
                    assert os.path.exists(os.path.join(d, "_DONE")), \
                        f"{d}: no _DONE sentinel -- INCOMPLETE shard, refusing to read"
                # PER-SHARD spec, not just the root manifest's.  `ensure_root_manifest`
                # OVERWRITES the root spec on every shard job, so rebuilding a handful
                # of shards under a different --min_atac_length / --max_atac_length
                # relabels the WHOLE cache while 94 shards still hold the old
                # population.  The root check would pass and the reader would serve two
                # corpora as one, with no signature in any log.  This is one small json
                # read per shard, once.
                mp = os.path.join(d, "shard_manifest.json")
                assert os.path.exists(mp), f"{d}: no shard_manifest.json"
                with open(mp) as f:
                    sm = json.load(f)
                assert sm["spec"] == self.spec.to_json(), (
                    f"{d}: this SHARD was built under a different CacheSpec than the "
                    f"cache is being read with -- a mixed-spec cache is two corpora, "
                    f"not one.  Rebuild it.\n  shard:  {sm['spec']}\n"
                    f"  reader: {self.spec.to_json()}")
                assert sm["split"] == self.split and sm["modality"] == modality, \
                    f"{d}: shard_manifest says {sm['split']}/{sm['modality']}"
                self._checked.add((modality, shard_idx))
            mm = open_npy(os.path.join(d, name))
            self._mm[key] = mm
        return mm

    def _slice(self, modality: str, i: int) -> Tuple[int, int, int]:
        sh = int(self.index[f"{modality}_shard"][i])
        row = int(self.index[f"{modality}_row"][i])
        off = self._arr(modality, sh, self.spec.offsets_name(modality))
        return sh, int(off[row]), int(off[row + 1])

    # --- per-cell access ------------------------------------------------------------ #

    def get(self, i: int) -> Dict:
        """One cell, upcast to fp32, with the fields fm_tokens() produces per cell.

        Keys use fm_tokens' own names, so a call site can be ported by deleting its
        forward pass and nothing else::

            rt      [n, 768]  raw scFoundation tokens, meta at (n-2, n-1)
            gene_id [n]       scFoundation panel column (19264/19265 = meta)
            gs      [n]       gathered log1p expression = the leak-free `bio` score
            rc      [3072]    valid-count-pooled cell vector -- NOT the encoder's
                              batch-dependent global-slice rc
            at      [m, 512]  raw EpiAgent tokens, CLS at 0, SEP at m-1
            ccre_id [m]       RAW token id (cCRE.bed row = id - 4; the pooler
                              subtracts the 4 itself)
            ac      [512]     EpiAgent CLS -- byte-identical to at[0]

        There is no `l2` argument, by design; see the module docstring.

        On a single-modality cache the OTHER modality's keys are ABSENT, not zero:
        `self.modalities` decides, and a consumer that reaches for `at` on an RNA-only
        cache gets a KeyError naming the cache rather than a plausible tensor of zeros.
        """
        out: Dict = {"barcode": str(self.index["barcode"][i])}
        if "rna" in self.modalities:
            sh_r, r0, r1 = self._slice("rna", i)
            # Per-CELL arrays are indexed by the shard-local row; per-TOKEN arrays by
            # the offset slice.  Mixing the two is the classic off-by-a-whole-array
            # mistake.
            rr = int(self.index["rna_row"][i])
            out.update({
                "n_rna": r1 - r0,
                "rt": np.asarray(
                    self._arr("rna", sh_r, "rna_tokens.npy")[r0:r1], np.float32),
                "gene_id": np.asarray(
                    self._arr("rna", sh_r, "rna_gene_ids.npy")[r0:r1], np.int64),
                "gs": np.asarray(
                    self._arr("rna", sh_r, "rna_values.npy")[r0:r1], np.float32),
                "rc": np.asarray(
                    self._arr("rna", sh_r, "rna_cell.npy")[rr], np.float32),
            })
            if self.spec.store_rna_pad:
                out["rna_pad"] = np.asarray(
                    self._arr("rna", sh_r, "rna_pad.npy")[rr], np.float32)
        if "atac" in self.modalities:
            sh_a, a0, a1 = self._slice("atac", i)
            ar = int(self.index["atac_row"][i])
            out.update({
                "n_atac": a1 - a0,
                "at": np.asarray(
                    self._arr("atac", sh_a, "atac_tokens.npy")[a0:a1], np.float32),
                "ccre_id": np.asarray(
                    self._arr("atac", sh_a, "atac_ccre_ids.npy")[a0:a1], np.int64),
                "ac": np.asarray(
                    self._arr("atac", sh_a, "atac_cls.npy")[ar], np.float32),
            })
        for k in ("dataset_id", "batch_id", "label", "orig_row"):
            if k in self.index:
                out[k] = self.index[k][i]
        return out

    # --- collate -------------------------------------------------------------------- #

    def collate(self, indices: Sequence[int], rc_mode: str = "valid_count",
                pad_fill: str = "live", atac_width: Optional[int] = None,
                as_torch: bool = True) -> Dict:
        """Pad a batch to the batch max and rebuild fm_tokens()' tensors.

        Returns {rc, rt, rm, ac, at, am, gs, cs, gene_id, ccre_id}, matching
        `fm_tokens(..., include_summary=True, return_ids=True)`'s 10-tuple element
        for element -- MINUS whatever modality this cache does not hold.  On an
        RNA-only cache the ATAC keys (`ac`, `at`, `am`, `cs`, `ccre_id`) are ABSENT.
        They are never zero-filled: flash-attn writes exact zeros at padded ATAC
        positions (design L10), so a zeros block is indistinguishable from real output
        and would train silently.  See `self.modalities`.

        rm / am are True == PAD.  We assert (~rm).sum(1) == rna_len before returning,
        which is the cheap version of the tombstone at model.py:229-243.

        rc_mode
            "valid_count" (default) -- the CORRECT pooling: emb1/emb2 at (n-2, n-1),
                max/mean over the cell's own gene tokens.  This is `fm_pool_rna`'s
                recipe and it is what `rna_cell.npy` stores.
            "live_global" -- reproduces model.py:215-218's batch-dependent slicing:
                emb1/emb2 at GLOBAL [-1]/[-2], max/mean over [:, :-2] including this
                cell's meta tokens AND its pad columns.  Requires spec.store_rna_pad,
                because the pad columns must carry their real encoder output for the
                max/mean to match.  Provided ONLY so the FineCLS fixed-slot path
                (fixed_slot_model_mixin.py:464-477, which reads the global positions)
                stays reproducible, and so parity tests T6/T8 have something to prove
                a difference against.

        pad_fill
            "live" (default) -- pad exactly as the live pipeline does: gene_id <- 103,
                gs <- 103.0, ATAC ids <- 0, and RNA pad tokens <- this cell's pad
                embedding when stored.  Those positions are masked out by rm/am and by
                select_topk_by_score's `masked_fill(padding_mask, NEG)`; they exist so
                a cached batch is byte-comparable with a live one.  103 is the real
                gene ABLIM1 -- it is a FILL value here, NEVER a pad DETECTOR.
            "safe" -- zeros / PAD_SCORE_SENTINEL, for consumers that would rather a
                bug surface as a nonsense score than as a plausible one.

        atac_width
            Force the ATAC pad width; pass spec.max_atac_length to match the live
            `collate_fn(fixed_atac_length=8192)` exactly.  Default = batch max, which
            is mathematically inert but changes `cs`'s length and, in the pathological
            case where every cell is shorter than kc, `k = min(kc, Na)`.
        """
        assert rc_mode in ("valid_count", "live_global")
        assert pad_fill in ("live", "safe")
        has_rna = "rna" in self.modalities
        has_atac = "atac" in self.modalities
        # Fail at the CALL, naming the cache -- not later, with a zeros tensor.
        assert has_rna or rc_mode == "valid_count", \
            "rc_mode='live_global' is an RNA quantity; this reader holds no RNA"
        idx = list(indices)
        b = len(idx)
        cells = [self.get(i) for i in idx]

        out: Dict = {"barcode": [c["barcode"] for c in cells]}
        if has_rna:
            nr = np.asarray([c["n_rna"] for c in cells], dtype=np.int64)
            n_r = int(nr.max())
            rt = np.zeros((b, n_r, self.spec.rna_dim), np.float32)
            gid_fill = int(LIVE_RNA_PAD_FILL) if pad_fill == "live" else 0
            gs_fill = LIVE_RNA_PAD_FILL if pad_fill == "live" else PAD_SCORE_SENTINEL
            gid = np.full((b, n_r), gid_fill, np.int64)
            gs = np.full((b, n_r), gs_fill, np.float32)
            rm = np.ones((b, n_r), bool)  # True == PAD
            for j, c in enumerate(cells):
                n = c["n_rna"]
                rt[j, :n] = c["rt"]
                gid[j, :n] = c["gene_id"]
                gs[j, :n] = c["gs"]
                rm[j, :n] = False
                if pad_fill == "live" and self.spec.store_rna_pad and n < n_r:
                    # The encoder output at EVERY pad column of a cell is bit-identical
                    # (same input token_emb(103.0)+pos_emb(103), same masked key set),
                    # so one vector broadcasts over the whole padded tail.
                    rt[j, n:] = c["rna_pad"]
            assert np.array_equal((~rm).sum(1), nr), \
                "rm polarity/length broken (True must be PAD)"
            out.update({"rt": rt, "rm": rm, "gs": gs, "gene_id": gid})
        if has_atac:
            na = np.asarray([c["n_atac"] for c in cells], dtype=np.int64)
            n_a = int(atac_width or na.max())
            assert n_a >= int(na.max()), f"atac_width {n_a} < batch max {int(na.max())}"
            at = np.zeros((b, n_a, self.spec.atac_dim), np.float32)
            cid = np.full((b, n_a), self.spec.atac_pad_id, np.int64)
            am = np.ones((b, n_a), bool)
            ac = np.stack([c["ac"] for c in cells])
            for j, c in enumerate(cells):
                m = c["n_atac"]
                at[j, :m] = c["at"]
                cid[j, :m] = c["ccre_id"]
                am[j, :m] = False
            assert np.array_equal((~am).sum(1), na), \
                "am polarity/length broken (True must be PAD)"
            # cs is EXACTLY -arange(Na): the negative token POSITION, i.e. TF-IDF
            # accessibility rank (the cell sentence is TF-IDF-descending).  Zero bits of
            # information -- always recomputed from the REPLAY width, never stored.
            cs = np.broadcast_to(-np.arange(n_a, dtype=np.float32), (b, n_a)).copy()
            out.update({"ac": ac, "at": at, "am": am, "cs": cs, "ccre_id": cid})
        elif atac_width is not None:
            # A width for a modality this cache does not hold is a silently-inert flag,
            # and this project has shipped four of those in one day.
            raise AssertionError(
                f"atac_width={atac_width} was passed to an RNA-only reader "
                f"({self.root}); the flag can reach nothing.  Drop it, or open the "
                f"cache with modalities=('rna', 'atac') once the ATAC pass has run.")

        if has_rna:
            if rc_mode == "valid_count":
                rc = np.stack([c["rc"] for c in cells])
            else:
                assert self.spec.store_rna_pad, "rc_mode='live_global' needs rna_pad"
                # model.py:215-218, verbatim, on the padded batch.
                emb1 = rt[:, -1, :]
                emb2 = rt[:, -2, :]
                emb3 = rt[:, :-2, :].max(axis=1)
                emb4 = rt[:, :-2, :].mean(axis=1)
                rc = np.concatenate([emb1, emb2, emb3, emb4], axis=1).astype(np.float32)
            out["rc"] = rc

        if as_torch:
            assert torch is not None, "as_torch=True needs torch"
            for k, v in list(out.items()):
                if isinstance(v, np.ndarray):
                    out[k] = torch.from_numpy(v)
        return out

    def close(self) -> None:
        self._mm.clear()


# ------------------------------------------------------------------------------------ #
# Self-test  --  runnable on a login node, no GPU, no model, no h5py
# ------------------------------------------------------------------------------------ #


def _synthetic(spec: CacheSpec, n_cells: int, seed: int = 0):
    """Synthetic cells with the REAL structural invariants: ascending gene ids ending in
    the
    two meta columns, ATAC sequences bracketed by CLS/SEP, ragged lengths including one
    cell
    at the ATAC cap and one at the QC floor."""
    rs = np.random.RandomState(seed)
    cells = []
    for i in range(n_cells):
        nnz = int(rs.randint(spec.min_rna_nnz, 400)) if i else spec.max_rna_nnz
        n = nnz + 2
        genes = np.sort(rs.choice(spec.n_genes, nnz, replace=False))
        ids = np.concatenate([genes, np.asarray(spec.rna_meta_ids)]).astype(np.int64)
        # Scale ~5 so the L2 tripwire in verify_shard() sees non-unit row norms, as real
        # scFoundation activations do.
        rt = (rs.randn(n, spec.rna_dim) * 5.0).astype(np.float32)
        gs = np.concatenate([rs.rand(nnz) * 5.0,
                             [spec.target_resolution, 3.7]]).astype(np.float32)
        rc = (rs.randn(spec.rna_cell_dim) * 5.0).astype(np.float32)
        rpad = (rs.randn(spec.rna_dim) * 5.0).astype(np.float32)

        raw = (spec.atac_trunc + 500 if i == 1
               else int(rs.randint(spec.min_atac_length, 3000)))
        m = min(raw, spec.atac_trunc) + 2
        body = rs.randint(spec.atac_id_offset, spec.atac_vocab_size, m - 2)
        ccre = np.concatenate([[spec.atac_cls_id], body,
                               [spec.atac_sep_id]]).astype(np.int64)
        at = (rs.randn(m, spec.atac_dim) * 3.0).astype(np.float32)
        cells.append(dict(bc=f"CELL{i:04d}-1", n=n, m=m, ids=ids, rt=rt, gs=gs, rc=rc,
                          rpad=rpad, ccre=ccre, at=at, raw=raw))
    return cells


def _self_test() -> None:  # pragma: no cover - exercised by `python fm_token_cache.py`
    import tempfile

    spec = CacheSpec()
    n_cells = 6
    cells = _synthetic(spec, n_cells)
    root = tempfile.mkdtemp(prefix="fmtok_selftest_")
    print(f"[self-test] format {CACHE_FORMAT_VERSION}  root={root}")

    # --- 1. write both modalities as one shard each --------------------------------- #
    wr = ShardWriter(root, "val", "rna", 0, 1, spec, argv=["self-test"])
    for k, c in enumerate(cells):
        wr.append(c["bc"], c["rt"], c["ids"], values=c["gs"], cell=c["rc"],
                  pad_emb=c["rpad"], orig_row=k * 3)
    mr = wr.finalize()

    wa = ShardWriter(root, "val", "atac", 0, 1, spec, argv=["self-test"])
    # exercise the padded->packed path, which is where the mask polarity discipline
    # lives
    n_pad = max(c["m"] for c in cells)
    at_b = np.zeros((n_cells, n_pad, spec.atac_dim), np.float32)
    cid_b = np.zeros((n_cells, n_pad), np.int64)
    am_b = np.ones((n_cells, n_pad), bool)  # True == PAD
    for j, c in enumerate(cells):
        at_b[j, :c["m"]] = c["at"]
        cid_b[j, :c["m"]] = c["ccre"]
        am_b[j, :c["m"]] = False
    wa.append_batch([c["bc"] for c in cells], at_b, am_b, cid_b,
                    cell=np.stack([c["at"][0] for c in cells]),
                    orig_rows=[k * 3 for k in range(n_cells)])
    ma = wa.finalize()
    print(f"[self-test] wrote rna shard: {mr['n_rows']} cells / "
          f"{mr['token_rows']} tokens; atac shard: {ma['n_rows']} cells / "
          f"{ma['token_rows']} tokens")

    # --- 2. verify ------------------------------------------------------------------ #
    sr = verify_shard(spec.shard_path(root, "val", "rna", 0, 1), spec, full=True)
    sa = verify_shard(spec.shard_path(root, "val", "atac", 0, 1), spec, full=True)
    print(f"[self-test] verify_shard rna  {sr}")
    print(f"[self-test] verify_shard atac {sa}")

    # --- 3. hand-build the global index + manifest (pass 0's job) ------------------- #
    index = {
        "barcode": np.asarray([c["bc"] for c in cells]),
        "split": np.asarray(["val"] * n_cells),
        "orig_row": np.arange(n_cells, dtype=np.int64) * 3,
        "dataset_id": np.asarray(["ds_a"] * n_cells),
        "batch_id": np.asarray(["b0"] * n_cells),
        "label": np.arange(n_cells, dtype=np.int64) % 3,
        "rna_shard": np.zeros(n_cells, np.int64),
        "rna_row": np.arange(n_cells, dtype=np.int64),
        "rna_len": np.asarray([c["n"] for c in cells], np.int64),
        "rna_nnz": np.asarray([c["n"] - 2 for c in cells], np.int64),
        "atac_shard": np.zeros(n_cells, np.int64),
        "atac_row": np.arange(n_cells, dtype=np.int64),
        "atac_len": np.asarray([c["m"] for c in cells], np.int64),
        "atac_raw_len": np.asarray([c["raw"] for c in cells], np.int64),
    }
    with open(os.path.join(root, "index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(index))
        for i in range(n_cells):
            w.writerow([index[k][i] for k in index])
    np.savez(os.path.join(root, "index.npz"),
             **{k: (np.asarray(v, dtype="S64") if v.dtype.kind in "UO" else v)
                for k, v in index.items()})
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"format_version": CACHE_FORMAT_VERSION, "spec": spec.to_json(),
                   "fm_mode": "eval_dropout_off",
                   "splits": {"val": {"rna_shards": 1, "atac_shards": 1,
                                      "n_cached": n_cells}}}, f, indent=2)

    # --- 4. read back and assert ROUND-TRIP equality -------------------------------- #
    rd = CacheReader(root, "val", spec)
    assert len(rd) == n_cells
    for i, c in enumerate(cells):
        g = rd.get(i)
        assert g["barcode"] == c["bc"], (g["barcode"], c["bc"])
        assert g["n_rna"] == c["n"] and g["n_atac"] == c["m"]
        # fp16 storage of fp32 activations: round-trip must equal the fp16 CAST exactly.
        assert np.array_equal(g["rt"], c["rt"].astype(np.float16).astype(np.float32))
        assert np.array_equal(g["at"], c["at"].astype(np.float16).astype(np.float32))
        assert np.array_equal(g["gene_id"], c["ids"])
        assert np.array_equal(g["ccre_id"], c["ccre"])
        assert np.array_equal(g["gs"], c["gs"].astype(np.float16).astype(np.float32))
        assert np.array_equal(g["rc"], c["rc"].astype(np.float16).astype(np.float32))
        assert np.array_equal(g["ac"], c["at"][0].astype(np.float16).astype(np.float32))
        assert np.array_equal(g["rna_pad"],
                              c["rpad"].astype(np.float16).astype(np.float32))
        # the invariants verify_shard() enforces, re-checked from the reader's own view
        assert tuple(g["gene_id"][-2:]) == spec.rna_meta_ids
        assert g["ccre_id"][0] == spec.atac_cls_id
        assert g["ccre_id"][-1] == spec.atac_sep_id
    max_err = max(float(np.abs(rd.get(i)["rt"] - cells[i]["rt"]).max()
                        / max(np.abs(cells[i]["rt"]).max(), 1e-6))
                  for i in range(n_cells))
    print(f"[self-test] round-trip EXACT vs the fp16 cast; fp16 rel err vs fp32 "
          f"= {max_err:.2e} (fp16 floor 2^-11 = {2 ** -11:.2e})")

    # --- 5. collate: shapes, mask polarity, cs, pad fills --------------------------- #
    bt = rd.collate(range(n_cells), as_torch=True)
    nr = index["rna_len"]
    na = index["atac_len"]
    assert tuple(bt["rt"].shape) == (n_cells, int(nr.max()), spec.rna_dim)
    assert tuple(bt["at"].shape) == (n_cells, int(na.max()), spec.atac_dim)
    assert bt["rm"].dtype == torch.bool and bt["am"].dtype == torch.bool
    assert torch.equal((~bt["rm"]).sum(1), torch.from_numpy(nr)), "rm polarity"
    assert torch.equal((~bt["am"]).sum(1), torch.from_numpy(na)), "am polarity"
    assert torch.equal(bt["cs"][0], -torch.arange(int(na.max()), dtype=torch.float32))
    j = int(np.argmin(nr))
    assert bool((bt["gene_id"][j, int(nr[j]):] == int(LIVE_RNA_PAD_FILL)).all()), \
        "live pad fill for gene_id must be 103 (a FILL value, never a pad DETECTOR)"
    assert bool((bt["ccre_id"][j, int(na[j]):] == spec.atac_pad_id).all())
    print(f"[self-test] collate rt{tuple(bt['rt'].shape)} at{tuple(bt['at'].shape)} "
          f"rm/am True==PAD OK, cs == -arange(Na) OK, live pad fills OK")

    # --- 6. forced ATAC width (parity mode) ----------------------------------------- #
    bw = rd.collate(range(n_cells), atac_width=spec.max_atac_length, as_torch=False)
    assert bw["at"].shape[1] == spec.max_atac_length
    assert bw["cs"].shape[1] == spec.max_atac_length
    print(f"[self-test] atac_width={spec.max_atac_length} forced OK "
          f"(matches collate_fn(fixed_atac_length=8192))")

    # --- 7. rc LIVENESS: the two rc conventions MUST differ (L1 / T8) ------------- #
    a = rd.collate(range(n_cells), rc_mode="valid_count", as_torch=False)["rc"]
    b = rd.collate(range(n_cells), rc_mode="live_global", as_torch=False)["rc"]
    rel = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-6))
    assert rel > 0.1, (
        f"rc_mode is INERT (rel {rel:.3e}) -- the batch-dependency probe is not "
        "exercising anything; this project has shipped bit-identical arms before")
    print(f"[self-test] rc liveness: valid_count vs live_global rel = {rel:.3f} "
          f"> 0.1 OK (global-slice path reproducible AND distinguishable)")

    # --- 8. negative tests: every guard must actually fire -------------------------- #
    fired = []
    bad = ShardWriter(root, "val", "rna", 0, 2, spec, argv=["neg"])
    try:
        c = cells[0]
        bad.append(c["bc"], c["rt"], c["ids"][::-1].copy(), values=c["gs"],
                   cell=c["rc"], pad_emb=c["rpad"])
    except AssertionError as e:
        fired.append(f"descending gene ids -> {str(e)[:52]}")
    finally:
        bad.abort()

    bad = ShardWriter(root, "val", "atac", 0, 2, spec, argv=["neg"])
    try:
        c = cells[0]
        bad.append(c["bc"], c["at"][1:], c["ccre"][1:], cell=c["at"][0])  # CLS dropped
    except AssertionError as e:
        fired.append(f"CLS dropped (ai vs ai[:,1:]) -> {str(e)[:52]}")
    finally:
        bad.abort()

    bad = ShardWriter(root, "val", "atac", 0, 2, spec, argv=["neg"])
    try:
        c = cells[0]
        # a real cell inside a padded window
        m, w = c["m"], c["m"] + 64
        tok = np.zeros((1, w, spec.atac_dim), np.float32); tok[0, :m] = c["at"]
        ids = np.zeros((1, w), np.int64); ids[0, :m] = c["ccre"]
        inv = np.zeros((1, w), bool); inv[0, :m] = True   # INVERTED: True == VALID
        bad.append_batch([c["bc"]], tok, inv, ids, cell=c["at"][0][None])
    except AssertionError as e:
        fired.append(f"inverted mask polarity -> {str(e)[:52]}")
    finally:
        bad.abort()

    try:
        CacheSpec(max_atac_length=16384)
    except AssertionError as e:
        fired.append(f"max_atac_length>8192 -> {str(e)[:52]}")
    try:
        CacheSpec(l2_normalized=True)
    except AssertionError as e:
        fired.append(f"l2_normalized=True -> {str(e)[:52]}")
    try:
        CacheReader(root, "val", CacheSpec(min_atac_length=100))
    except AssertionError as e:
        fired.append(f"QC spec mismatch on open -> {str(e)[:52]}")
    # --- 8b. index assembly: the merge step, and the MISSING-SHARD refusal ---------- #
    # Nothing in the builder writes <root>/index.{npz,csv}; it writes one slice per
    # shard.  A finished 8 TB cache that cannot be opened is not a cache, so the reader
    # falls back to assembling the slices -- and that assembly must REFUSE a gap.
    sl_dir = os.path.join(root, "val")
    np.savez(os.path.join(sl_dir, "index_shard0000of0001.npz"),
             **{k: (np.asarray(v, dtype="S64") if v.dtype.kind in "UO" else v)
                for k, v in index.items()})
    os.remove(os.path.join(root, "index.npz"))
    os.remove(os.path.join(root, "index.csv"))
    rd2 = CacheReader(root, "val", spec)
    assert len(rd2) == n_cells, f"assembled index has {len(rd2)} of {n_cells} rows"
    assert sorted(rd2.index["barcode"].tolist()) == sorted(index["barcode"].tolist())
    rd2.close()
    got = write_global_index(root, "val", expect_shards=1)
    assert got["rows"] == n_cells and os.path.exists(os.path.join(root, "index.npz"))
    # a GAP in 0..MMMM-1: array element 1 of 3 never ran, so its ~8k cells are simply
    # absent and every barcode still present is real.  Only the filename's `of0003`
    # knows.
    gap_root = os.path.join(root, "_gaptest")
    os.makedirs(os.path.join(gap_root, "val"), exist_ok=True)
    for sh in (0, 2):
        # distinct barcodes per slice: the assembler also refuses duplicates, and a
        # duplicate would mask the gap/partial verdict this leg is testing
        gi = dict(index, barcode=np.asarray([f"{b}#s{sh}" for b in index["barcode"]]),
                  rna_shard=np.full(n_cells, sh, np.int64),
                  atac_shard=np.full(n_cells, sh, np.int64))
        np.savez(os.path.join(gap_root, "val", f"index_shard{sh:04d}of0003.npz"),
                 **{k: (np.asarray(v, dtype="S64") if v.dtype.kind in "UO" else v)
                    for k, v in gi.items()})
    try:
        assemble_index_from_slices(gap_root, "val")
    except AssertionError as e:
        fired.append(f"index slice MISSING -> {str(e)[:52]}")
    # "SHORT BY DESIGN" is expressible, and it is graded by EXACT SET EQUALITY.  The
    # incremental campaign (ARRAY=0-47 of 96) needs a partial merge; what it must NOT
    # buy is a subset rule that would let a hole INSIDE the declared range pass.  So:
    # the declared {0, 2} is accepted, and {0, 1, 2} on the same slices is REFUSED.
    part = assemble_index_from_slices(gap_root, "val", allow_partial=(0, 2))
    assert len(part["barcode"]) == 2 * n_cells, part
    try:
        assemble_index_from_slices(gap_root, "val", allow_partial=(0, 1, 2))
    except AssertionError as e:
        fired.append(f"partial index != declared range -> {str(e)[:52]}")
    shutil.rmtree(gap_root)
    print(f"[self-test] index: reader assembled the slices, merge wrote "
          f"{got['rows']} rows to index.npz + index.csv; a MISSING slice is REFUSED, "
          f"a DECLARED partial range ({{0,2}} of 3) is accepted")

    # --- 8b3. a STALE PARTIAL MERGE: the index is short, the PAYLOAD is not -------- #
    # The constructible mistake: merge with `--shard_set 0` while only shard 0 exists
    # (legitimate -- it is the incremental campaign), then let shard 1 land and never
    # re-merge.  Both modalities are built, so the modality refusal above cannot see it,
    # and every barcode in the half index is real.  This is the ONE state that could
    # open the default cross-modal constructor at 50 % of its cells.
    sm_root = os.path.join(root, "_stalemerge")
    sm_cells = cells[1:4]
    for sh in (0, 1):
        sfx = "" if sh == 0 else "#s1"
        wsr = ShardWriter(sm_root, "val", "rna", sh, 2, spec, argv=["stale"])
        wsa = ShardWriter(sm_root, "val", "atac", sh, 2, spec, argv=["stale"])
        for k, c in enumerate(sm_cells):
            wsr.append(c["bc"] + sfx, c["rt"], c["ids"], values=c["gs"], cell=c["rc"],
                       pad_emb=c["rpad"], orig_row=sh * 10 + k)
            wsa.append(c["bc"] + sfx, c["at"], c["ccre"], cell=c["at"][0],
                       orig_row=sh * 10 + k)
        wsr.finalize()
        wsa.finalize()
        n_s = len(sm_cells)
        gi = {k: (v[1:4] if len(v) == n_cells else v) for k, v in index.items()}
        gi = dict(gi,
                  barcode=np.asarray([c["bc"] + sfx for c in sm_cells]),
                  rna_shard=np.full(n_s, sh, np.int64),
                  atac_shard=np.full(n_s, sh, np.int64),
                  rna_row=np.arange(n_s, dtype=np.int64),
                  atac_row=np.arange(n_s, dtype=np.int64))
        with open(os.path.join(sm_root, "manifest.json"), "w") as f:
            json.dump({"format_version": CACHE_FORMAT_VERSION, "spec": spec.to_json(),
                       "splits": {"val": {"rna_shards": 2, "atac_shards": 2,
                                          "modalities": ["rna", "atac"]}}}, f)
        if sh == 0:
            # the merge happens HERE, with slice 1 not yet in existence -- which is why
            # `allow_partial` accepts it and why nothing downstream ever revisits it
            np.savez(os.path.join(sm_root, "val", "index_shard0000of0002.npz"),
                     **{k: (np.asarray(v, dtype="S64") if v.dtype.kind in "UO" else v)
                        for k, v in gi.items()})
            write_global_index(sm_root, "val", expect_shards=2, allow_partial=(0,))
            with open(os.path.join(sm_root, "manifest.json")) as f:
                sm_man = json.load(f)
            sm_man["splits"]["val"]["completeness"] = {
                "status": "PARTIAL_BY_DESIGN", "declared_modalities": ["rna", "atac"],
                "index_shards": [0]}      # what seal_completeness() writes
            with open(os.path.join(sm_root, "manifest.json"), "w") as f:
                json.dump(sm_man, f)
        else:
            np.savez(os.path.join(sm_root, "val", "index_shard0001of0002.npz"),
                     **{k: (np.asarray(v, dtype="S64") if v.dtype.kind in "UO" else v)
                        for k, v in gi.items()})
    try:
        CacheReader(sm_root, "val", spec)
    except AssertionError as e:
        fired.append(f"ROOT INDEX short vs the payload -> {str(e)[:52]}")
    rdp = CacheReader(sm_root, "val", spec, allow_partial_index=True)
    assert rdp.partial_index and len(rdp) == len(sm_cells), rdp.index_completeness
    assert rdp.index_completeness["missing_from_index"] == {"rna": [1], "atac": [1]}, \
        rdp.index_completeness
    rdp.close()
    write_global_index(sm_root, "val", expect_shards=2)          # the remediation
    rdf = CacheReader(sm_root, "val", spec)
    assert len(rdf) == 2 * len(sm_cells) and not rdf.partial_index
    assert rdf.index_completeness["status"] == "COMPLETE", rdf.index_completeness
    rdf.close()
    shutil.rmtree(sm_root)
    print(f"[self-test] stale PARTIAL MERGE: a root index over shard {{0}} with shards "
          f"{{0,1}} PUBLISHED is REFUSED by the default reader, opens at "
          f"{len(sm_cells)} cells only when asked (allow_partial_index=True), and "
          f"re-merging restores all {2 * len(sm_cells)}")

    # --- 8b2. an RNA-ONLY cache is READABLE, and its ATAC keys are ABSENT ----------- #
    # The corpus is built in two passes and the RNA half can land days before the ATAC
    # half on a near-full volume.  Between those moments the ATAC shard dirs DO NOT
    # EXIST.  Before this, `get()` sliced ATAC unconditionally: __init__ succeeded and
    # printed a healthy cell count, then the first __getitem__ died on a missing _DONE
    # -- a run that looks fine through setup and fails at step 0.  The keys must be
    # OMITTED rather than zero-filled: flash-attn writes exact zeros at padded ATAC
    # positions (L10), so zeros are indistinguishable from real output and would train.
    ro_root = os.path.join(root, "_rnaonly")
    wro = ShardWriter(ro_root, "val", "rna", 0, 1, spec, argv=["self-test-rna-only"])
    for k, c in enumerate(cells):
        wro.append(c["bc"], c["rt"], c["ids"], values=c["gs"], cell=c["rc"],
                   pad_emb=c["rpad"], orig_row=k * 3)
    wro.finalize()
    with open(os.path.join(ro_root, "manifest.json"), "w") as f:
        json.dump({"format_version": CACHE_FORMAT_VERSION, "spec": spec.to_json(),
                   "splits": {"val": {"rna_shards": 1, "atac_shards": 1,
                                      "modalities": ["rna"]}}}, f)
    np.savez(os.path.join(ro_root, "val", "index_shard0000of0001.npz"),
             **{k: (np.asarray(v, dtype="S64") if v.dtype.kind in "UO" else v)
                for k, v in index.items()})
    rdo = CacheReader(ro_root, "val", spec)
    assert rdo.modalities == ("rna",), rdo.modalities
    c0 = rdo.get(0)
    assert "rt" in c0 and "rc" in c0, sorted(c0)
    for absent in ("at", "ac", "ccre_id", "n_atac"):
        assert absent not in c0, f"RNA-only get() leaked {absent!r}"
    bo = rdo.collate([0, 1], rc_mode="valid_count", as_torch=False)
    assert {"rt", "rm", "gs", "gene_id", "rc"} <= set(bo), sorted(bo)
    for absent in ("at", "am", "ac", "cs", "ccre_id"):
        assert absent not in bo, f"RNA-only collate() leaked {absent!r}"
    try:
        _ = bo["at"]
    except KeyError as e:
        fired.append(f"RNA-only cache has no 'at' -> KeyError {e}")
    try:
        rdo.collate([0, 1], atac_width=spec.max_atac_length, as_torch=False)
    except AssertionError as e:
        fired.append(f"atac_width on an RNA-only reader -> {str(e)[:52]}")
    try:
        CacheReader(ro_root, "val", spec, modalities=("rna", "atac"))
    except AssertionError as e:
        fired.append(f"asking for ATAC of an RNA-only cache -> {str(e)[:52]}")
    # The RNA-only reader is still a REAL reader: the rc liveness probe (design T8, the
    # `--store_rna_pad` arm) runs on it, which is the whole reason to make collate
    # modality-aware rather than just letting the ATAC read fail late.
    ra = rdo.collate([0, 1, 2, 3], rc_mode="valid_count", as_torch=False)["rc"]
    rb = rdo.collate([0, 1, 2, 3], rc_mode="live_global", as_torch=False)["rc"]
    rel_ro = float(np.abs(ra - rb).max() / max(np.abs(ra).max(), 1e-6))
    assert rel_ro > 0.1, (
        f"rc_mode is INERT on an RNA-only cache (rel {rel_ro:.3e}); T8 would go dark "
        f"for every shard of an RNA-first campaign")
    rdo.close()
    print(f"[self-test] RNA-ONLY cache: opens, modalities=('rna',), ATAC keys ABSENT "
          f"(not zeros), T8 rc_mode liveness rel = {rel_ro:.3f} > 0.1")

    # --- 8c. per-shard spec check: a MIXED-spec cache must not open ----------------- #
    # ensure_root_manifest() overwrites the root spec on every shard job, so rebuilding
    # a subset under a different QC would relabel the whole cache while the other shards
    # still hold the old population.  The reader checks each shard, not just the root.
    smp = os.path.join(spec.shard_path(root, "val", "rna", 0, 1), "shard_manifest.json")
    with open(smp) as f:
        sm_ok = json.load(f)
    sm_bad = dict(sm_ok)
    sm_bad["spec"] = dict(sm_ok["spec"], min_atac_length=100)
    with open(smp, "w") as f:
        json.dump(sm_bad, f)
    try:
        CacheReader(root, "val", spec).get(0)
    except AssertionError as e:
        fired.append(f"MIXED per-shard spec -> {str(e)[:52]}")
    with open(smp, "w") as f:
        json.dump(sm_ok, f)

    # --- 8d. orphaned .tmp sweep ---------------------------------------------------- #
    orphan = spec.shard_path(root, "val", "rna", 0, 7) + ".tmp"
    os.makedirs(orphan, exist_ok=True)
    with open(os.path.join(orphan, "payload.bin"), "wb") as f:
        f.write(b"x" * 4096)
    with open(os.path.join(orphan, TMP_OWNER_FILE), "w") as f:
        json.dump({"slurm_job_id": "99999999", "started": time.time() - 86400}, f)
    live_tmp = spec.shard_path(root, "val", "rna", 1, 7) + ".tmp"
    os.makedirs(live_tmp, exist_ok=True)
    with open(os.path.join(live_tmp, TMP_OWNER_FILE), "w") as f:
        json.dump({"slurm_job_id": "99999999", "started": time.time()}, f)
    assert tmp_bytes(root, "val") >= 4096, "tmp_bytes does not see the orphan"
    gc = sweep_stale_tmp(root, "val", min_age_s=3600.0)
    assert len(gc["removed"]) == 1 and gc["removed"][0]["dir"] == orphan, gc
    assert any(k["dir"] == live_tmp for k in gc["kept"]), gc
    assert not os.path.isdir(orphan) and os.path.isdir(live_tmp)
    shutil.rmtree(live_tmp)
    print(f"[self-test] .tmp sweep: reclaimed {gc['bytes_freed']} B from a dead "
          f"owner, KEPT the too-young one")

    # An EXACT count, not `>=`: a guard that stops firing has to break this test, and a
    # guard that is added without being counted has to break it too.  8 structural
    # guards + 5 modality/partial guards (partial-range mismatch, a ROOT INDEX short
    # relative to the payload, the RNA-only cache's absent `at`, `atac_width` on an
    # RNA-only reader, and asking an RNA-only cache for ATAC).
    assert len(fired) == 13, f"only {len(fired)}/13 guards fired: {fired}"
    for s in fired:
        print(f"[self-test]   GUARD FIRED: {s}")

    rd.close()
    shutil.rmtree(root)
    print("[self-test] ALL CHECKS PASSED")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
