#!/usr/bin/env python3
"""FineCLS on the REFINER track -- two-arm trainer over the frozen-FM token cache.

    ARM A  --arm cell_only   global (cell-embedding) branch ONLY
    ARM B  --arm finecls     global branch + FineCLS fixed slots

Design: experiments/finecls_refiner/EXPERIMENT_finecls_on_refiner.md.  Read it before
changing a default; every number below is traced there.

WHAT IS NEW HERE, AND WHY IT IS A SEPARATE FILE
-----------------------------------------------
FineCLS was measured on LIVE frozen FMs by `training_mpnce/fixed_slot_trainer.py`.  This
run puts the same fine branch on top of a TRAINED 2-layer refiner, fed from the 8.7 TB
`fmtok-v1` cache, and the composition is not a matter of wiring two existing files
together. Four traps sit between "it runs" and "it trains", and three of them are
silent:

  TRAP 1 -- `RNARefinerFM.native()` / `ATACRefinerFM.native()` are `@torch.no_grad()`.
      `native()` returns the refined PRE-projection tokens at native width (768 / 512),
      which is exactly and only what `FixedSlotPooler` can eat (FineCLS pools at native
      width and projects afterwards).  So the obvious wiring yields a refiner that gets
      ZERO gradient from the fine loss -- and it does not crash, because the fine term
      is summed with a grad-carrying global term.  FIX: `native_grad()`, added by
      `attn_refiner_native_grad.patch`; `native()` keeps its decorator and delegates.
      This file uses `native_grad` EXCLUSIVELY and asserts the patch is present at
      import.

  TRAP 2 -- `fixed_slot_model_mixin._fixed_encoder_context` (:430-435) IS
      `torch.no_grad()` whenever the FM is frozen, and `encode_{rna,atac}_fixed_slots`
      run their whole forward inside it.  Dropping the refiner in there re-breaks TRAP 1
      with no decorator to point at. FIX: this file does NOT call those two methods.  It
      re-implements their SHAPE CONTRACT (below) over cached tokens, and the string
      "no_grad" appears nowhere in the encode path.

  TRAP 3 -- the `pool_include_cls` concat can detach half the fine signal INVISIBLY. The
      cache ships frozen cell vectors `rc [3072]` / `ac [512]` that are right there and
      exactly the right shape, and `encode_atac_fixed_slots(return_frozen_cell=True)`
      returns `cell.detach()`. Using either makes the cell half of every slot a
      CONSTANT; gradient still reaches the refiner through the slot half, so a gradient
      assert stays GREEN while half the fine supervision is dead. FIX: the concatenated
      cell vector is recomputed from the REFINED tokens -- `fm_pool_rna(Gnat, rm)`
      (valid-count meta lookup, NOT the batch-dependent global `[:, -2:]` slice) and
      `l2(Cnat[:, 0, :])`.

  TRAP 4 -- `grad_cache_two_pass` (train_filip_combined.py:549) hard-codes a 2-tuple of
      pooled cell embeddings, so it cannot carry the `[b, M, 256]` slot tensors at all.
      FIX: `grad_cache_two_pass_n` below, an N-tuple generalisation that keeps the RNG
      replay AND extends the run-time `verify` to EVERY returned tensor.  Verifying only
      the globals would be a PROXY: it would print OK on a run whose slot branch
      replayed different dropout masks.

THE SHAPE CONTRACT WE REPRODUCE (fixed_slot_model_mixin.py:486-491 RNA, :544-549 ATAC)

    slots = pooler(native_tokens, token_ids, pad_mask)          # [B, M, native]
    slots = cat(slots, cell.unsqueeze(1).expand(-1, M, -1))      # RNA 768+3072 = 3840
    slots = fixed_projection(slots)                              # ATAC 512+ 512 = 1024
    slots = adapter(slots); slots = where(valid, slots, 0)

3840 and 1024 are the reference widths exactly, so the projection is the reference
projection.  The ONLY difference from the live-FM run is that the tokens and the cell
vector are refined and grad-carrying.

WHAT IS IMPORTED, NEVER COPIED
    CachedTokenDataset / cached_collate / make_fm_tokens_from_cache  (this directory)
    RNARefinerFM / ATACRefinerFM                                    (attn_refiner.py)
    FixedSlotPooler / load_fixed_prior_npz / fixed_slot_route_weights  (fixed_slot)
    ResidualTokenProjection / BlockLowRankAdapter          (fixed_slot_model_mixin)
    _center_by_group / _center_slots_by_group                 (fixed_slot_trainer)
    cell_infonce / supcon_xmodal / align_loss / fm_pool_rna / fm_pool_atac / head
    _rng_snapshot / _rng_restore                            (train_filip_combined)
    GatherWithGrad     (haoyun/multiomics_clip_finelip/training) -- the ALL-REDUCING one
    SameDatasetBlockedBatchSampler              (modules/blocked_sampler)

⛔ NOT imported, deliberately: the FineCLS splice loop at `fixed_slot_trainer.py:886`. It
recomputes the whole loss `accum` times and has NO RNG replay while `projection_dropout`
0.1 is live in the gradient path -- the equivalence test measures that failure mode at
231 % relative gradient error.  It would not crash and would make BOTH arms wrong in the
same direction, i.e. invisible to an A-vs-B contrast.

USAGE
    python train_finecls_refiner.py --arm cell_only --save_dir .../armA  [flags]
    python train_finecls_refiner.py --arm finecls   --save_dir .../armB \
        --prior_rna_path .../fixed64_rna_prior_sce2g_v1.npz \
        --prior_atac_path .../fixed64_atac_prior_sce2g_p3_v1.npz --num_slots 64
    torchrun --nproc_per_node=3 train_finecls_refiner.py --distributed ...   # 3-GPU DDP
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import zlib
from contextlib import ExitStack
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIXED_SLOT_ROOT = os.path.join(REPO, "experiments")
FINELIP = os.path.join(REPO, "haoyun", "multiomics_clip_finelip")


def _bootstrap_sys_path() -> None:
    """Put the four source trees this file composes on `sys.path`, in dependency order.

    `scFoundation/model` is required even though NO FM is loaded here: the fixed-slot
    mixin imports `gatherData` from it at module scope, and we import the mixin for its
    `ResidualTokenProjection` / `BlockLowRankAdapter` -- the reference projection and
    adapter, which must be the SAME objects or the arm is not comparable to theirs.
    """
    for p in (REPO, os.path.join(REPO, "haoyun"), FINELIP, FIXED_SLOT_ROOT,
              os.path.join(REPO, "scFoundation", "model"),
              os.path.dirname(os.path.abspath(__file__))):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_sys_path()

from cached_token_dataset import (                                    # noqa: E402
    CachedTokenDataset, cached_collate, make_fm_tokens_from_cache, move_batch_to_device,
)
from modules.attn_refiner import ATACRefinerFM, RNARefinerFM          # noqa: E402
from modules.blocked_sampler import SameDatasetBlockedBatchSampler    # noqa: E402
# ⛔ TWO `GatherWithGrad`s exist and they are NOT equivalent.  The
# `multiomics_clip/training` copy's backward returns `grad_output.chunk(W)[rank]` with
# all_reduce, so under DDP the reducer's subsequent MEAN over ranks delivers 1/W of the
# true gradient (measured 0.500 at W=2, 0.250 at W=4) -- a silent LR/W on every gathered
# term.  The finelip copy all-reduces first.  Loaded BY PATH, with the all_reduce
# asserted, because an `import` that silently resolves to the stale twin is exactly the
# class of defect this file exists to prevent. Same technique as `gradcache.py:164-186`.
_GA_PATH = os.path.join(FINELIP, "training", "gradient_accumulation.py")
assert "all_reduce" in open(_GA_PATH).read(), (
    f"{_GA_PATH} has no all_reduce in GatherWithGrad.backward -- that is the STALE "
    f"twin "
    f"(multiomics_clip/training/gradient_accumulation.py). Gathered negatives would "
    f"train at 1/world_size of the intended learning rate, silently.")
_ga_spec = importlib.util.spec_from_file_location("_finelip_grad_accum", _GA_PATH)
_ga = importlib.util.module_from_spec(_ga_spec)
_ga_spec.loader.exec_module(_ga)
GatherWithGrad = _ga.GatherWithGrad
from multiomics_clip_xinyu_June_fixed_slot_routing.modules.fixed_slot import (  # noqa
    FixedSlotPooler, fixed_slot_route_weights, fixed_slot_similarity,
    load_fixed_prior_npz,
)

#: The patch this whole track depends on.  Asserted at IMPORT, not at first backward:
#: without it the fine loss is summed with a grad-carrying global term, the run
#: completes, the loss curve looks healthy, and the refiner learns nothing from slots.
assert hasattr(RNARefinerFM, "native_grad") and hasattr(ATACRefinerFM, "native_grad"), (
    "attn_refiner.py has no native_grad(): apply "
    "experiments/finecls_refiner/attn_refiner_native_grad.patch, then re-run "
    "experiments/finecls_refiner/test_refiner_grad_through_slots.py (10/13 -> 17/17). "
    "Without it the refiner gets ZERO gradient from the fine loss and this arm is a "
    "lie that runs to completion.")

RNA_DIM, ATAC_DIM = 768, 512
N_GENES, N_CCRE = 19264, 1355445
ATAC_TOKEN_OFFSET = 4          # EpiAgent cCRE token id = cCRE.bed 0-based row + 4
RNA_META_IDS = (19264, 19265)  # (target_resolution, log10_total), appended by loader
#: A parameter INSIDE each refiner's attention stack.  Liveness asserts must NOT target
#: `proj.weight`: on the FineCLS wiring `proj` is applied after `native_grad` returns,
#: so a severed layer stack would still show a gradient there.
RNA_PROBE_PARAM = "rna_refiner.layers.0.self_attn.in_proj_weight"
ATAC_PROBE_PARAM = "atac_refiner.blocks.0.mixer.Wqkv.weight"

#: The production loss scale for `--refiner_precision fp16`.  MEASURED, not inherited
#: from torch's 2^16 default: across 4 real 512-cell production blocks, the refiner
#: gradient error against an all-fp32 reference converges at 2^18 (rna 1.6e-03 / atac
#: 1.0e-03, versus bf16's scale-invariant 5.3e-03 / 4.2e-03) and the FIRST non-finite
#: gradient appears at 2^24.  2^20 therefore sits 2 stops past convergence and 4 stops
#: below the ceiling.  At torch's default 2^16 the fp16 policy is a WASH with bf16
#: (4.4e-03 / 4.8e-03) and buys nothing for the cost of a scaler -- shipping there would
#: void the decision without anyone noticing.
GRAD_SCALER_INIT_DEFAULT = 2.0 ** 20


# ------------------------------------------------------------------------------------ #
# Lazily-imported pieces (heavy trees; kept out of module import so `--help` is instant)
# ------------------------------------------------------------------------------------ #

def _mixin_pieces():
    """`ResidualTokenProjection`, `BlockLowRankAdapter` and the two vocabulary readers.

    Imported from the reference mixin rather than re-declared: the projection depth,
    expansion, dropout placement and the adapter's zero-init `up` are part of what
    "FineCLS" MEANS, and a re-declaration would drift silently against their runs.
    """
    from multiomics_clip_xinyu_June_fixed_slot_routing.fixed_slot_model_mixin import (
        BlockLowRankAdapter, ResidualTokenProjection, _read_ccre_ids,
        _read_gene_symbols,
    )
    return (ResidualTokenProjection, BlockLowRankAdapter, _read_gene_symbols,
            _read_ccre_ids)


def _centering_helpers():
    """`_center_by_group` / `_center_slots_by_group`, from the reference trainer.

    They are `@staticmethod`s, so no trainer instance is constructed.  Importing them
    rather than copying matters because the two differ in a way memory says is decisive:
    `_center_by_group` ends with `F.normalize` and `_center_slots_by_group` does NOT --
    the re-normalisation is what carries the centring gain, and on the slot side it
    happens downstream inside `F.normalize(slots, dim=-1)` in the loss/similarity. A
    copy would eventually "fix" that asymmetry and quietly delete the effect.
    """
    from multiomics_clip_xinyu_June_fixed_slot_routing.training_mpnce \
        .fixed_slot_trainer import FixedSlotMPNCETrainer as T
    return T._center_by_group, T._center_slots_by_group


def _filip_pieces():
    """Loss terms + pooling recipes + the RNG-replay primitives, from the refiner track.

    `fm_pool_rna` is the load-bearing one: it locates the two RNA meta tokens by
    VALID-COUNT (`vc-2`, `vc-1`) instead of the global `[:, -2:]` slice the mixin uses.
    The global slice is the known `gatherData` batch-dependency bug -- at batch > 1 it
    reads padding for every cell that is not the batch-longest.
    """
    from scripts.train_filip_combined import (
        _rng_restore, _rng_snapshot, align_loss, cell_infonce, fm_pool_atac,
        fm_pool_rna, head, supcon_xmodal,
    )
    return dict(rng_snapshot=_rng_snapshot, rng_restore=_rng_restore,
                align_loss=align_loss, cell_infonce=cell_infonce,
                fm_pool_rna=fm_pool_rna, fm_pool_atac=fm_pool_atac, head=head,
                supcon_xmodal=supcon_xmodal)


def l2(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


# ------------------------------------------------------------------------------------ #
# Liveness bookkeeping
# ------------------------------------------------------------------------------------ #

class Liveness:
    """Counters + printed values for every arm-defining flag.

    THE RULE THIS IMPLEMENTS.  This project has produced two arms whose outputs differed
    by `max|delta| = 0.000e+00`, four silently-inert flags in one day, one liveness
    assert on a PROXY (a shard count) that killed 13 tasks, and one behind an exit that
    proved nothing because its output was SILENCE. So every entry here (a) asserts on
    the quantity the flag CONTROLS rather than on a proxy, and (b) PRINTS a value in
    BOTH arms -- an assert whose only evidence is the absence of a crash is not
    evidence.
    """

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self.values: Dict[str, object] = {}

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def set(self, key: str, value) -> None:
        self.values[key] = value

    def report(self, prefix: str = "[liveness]") -> str:
        rows = [f"{prefix} {k} = {v}" for k, v in sorted(self.values.items())]
        rows += [f"{prefix} count:{k} = {v}" for k, v in sorted(self.counts.items())]
        return "\n".join(rows)


def sha256_csr(path: str, matrix_name: str) -> str:
    """Content hash of a prior's CSR triple + shape.

    ⛔ THIS IS THE ONLY HONEST IDENTITY CHECK BETWEEN ARM B AND ARM C.  `a3_random64`
    shares `fixed64_sce2g_v1`'s `slot_ids` AND `slot_names` byte-for-byte (verified
    `np.array_equal == True` for both) and has the same `num_slots`, and the model's own
    consistency guard (`fixed_slot_model_mixin.py:310-313`) compares ONLY those fields.
    Every obvious liveness assert -- num_slots, slot ids, slot names, the catalogue
    printout -- therefore passes IDENTICALLY in the biology arm and the random arm.
    """
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for key in ("data", "indices", "indptr", "shape"):
            arr = np.ascontiguousarray(z[f"{matrix_name}_{key}"])
            h.update(str(arr.dtype).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def git_sha(default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


# ------------------------------------------------------------------------------------ #
# The model: ONE nn.Module holding refiners + global heads + (ARM B) the slot branch
# ------------------------------------------------------------------------------------ #

class FineCLSRefiner(nn.Module):
    """Refiners + global heads, plus the FineCLS fine branch when `arm == "finecls"`.

    ONE module on purpose.  `train_filip_combined.py:1409` DDP-wraps its trainables
    INDIVIDUALLY, and any trainable module left off that list all-reduces on every
    micro-batch instead of once per step (and, if it produces no grad in some
    micro-batch, raises "expected to have finished reduction"). Wrapping a single module
    makes `no_sync()` cover everything by construction.

    ARM A DOES NOT ALLOCATE THE SLOT BRANCH.  Constructing it and weighting its loss to
    zero would leave parameters with no gradient, which under DDP with
    `find_unused_parameters=False` CRASHES, and it buys nothing on gradient clipping --
    `clip_grad_norm_` is joint and a zero-gradient parameter contributes exactly zero to
    the norm.  The confound this leaves is stated in the design and is deliberate: ARM B
    has ~2-3 M more trainable parameters than ARM A, so B-A confounds "the fine branch"
    with "more capacity".  ARM C (`a3_random64`) is the parameter-matched answer to it.
    """

    def __init__(self, args, priors: Optional[Dict] = None) -> None:
        super().__init__()
        fp = _filip_pieces()
        self.arm = args.arm
        self.proj_dim = int(args.proj_dim)
        self._fm_pool_rna = fp["fm_pool_rna"]
        self._fm_pool_atac = fp["fm_pool_atac"]

        # --- shared trunk: 2 trainable layers on top of the FROZEN, CACHED FM tokens --
        # Built directly rather than through `make_refiners`, which needs a live
        # MultiOmicsCLIP for `model.rna_token_dim` / `init_from_fm`.  There is no FM on
        # this track; the dims are the cache spec's and are asserted against it.
        self.rna_refiner = RNARefinerFM(
            RNA_DIM, self.proj_dim, args.n_layers, dropout=args.dropout,
            grad_ckpt=bool(getattr(args, "rna_grad_ckpt", 1)),
            sub_batch=int(getattr(args, "rna_sub_batch", 0)),
            autocast=getattr(args, "rna_autocast", "off"))
        self.atac_refiner = ATACRefinerFM(
            ATAC_DIM, self.proj_dim, args.n_layers, dropout=args.dropout,
            use_sdpa=bool(args.atac_sdpa),
            use_varlen=bool(getattr(args, "atac_varlen", 0)),
            autocast=getattr(args, "atac_autocast", "fp16"))
        # --- global branch: the DECIDED `--cell_source fm_pool` recipe ----------------
        # fm_pool_rna -> [B, 4*proj_dim]; fm_pool_atac -> [B, 3*proj_dim].
        self.rna_head = fp["head"](4 * self.proj_dim, self.proj_dim)
        self.atac_head = fp["head"](3 * self.proj_dim, self.proj_dim)

        self.num_slots = 0
        self.rna_pooler = self.atac_pooler = None
        self.rna_slot_projection = self.atac_slot_projection = None
        self.rna_adapter = self.atac_adapter = None
        if self.arm == "finecls":
            assert priors is not None, "arm=finecls needs loaded priors"
            Projection, Adapter, _, _ = _mixin_pieces()
            self.num_slots = int(priors["rna"].matrix.shape[1])
            # attention_dim == the NATIVE width: `pool_native: true` in the reference.
            self.rna_pooler = FixedSlotPooler(
                priors["rna"].matrix, token_offset=0, attention_dim=RNA_DIM,
                attention_temperature=args.pooling_temperature,
                pool_topk=args.pool_topk)
            self.atac_pooler = FixedSlotPooler(
                priors["atac"].matrix, token_offset=ATAC_TOKEN_OFFSET,
                attention_dim=ATAC_DIM, attention_temperature=args.pooling_temperature,
                pool_topk=args.pool_topk)
            stat = 2 if args.pool_topk > 0 else 1
            # pool_include_cls: the cell vector is concatenated onto EVERY slot before
            # the projection.  RNA 768*stat + 3072 = 3840, ATAC 512*stat + 512 = 1024 --
            # the reference widths exactly, so this IS the reference projection.
            proj_kw = dict(output_dim=self.proj_dim, num_layers=args.projection_layers,
                           expansion=args.projection_expansion,
                           dropout=args.projection_dropout)
            self.rna_slot_projection = Projection(
                input_dim=RNA_DIM * stat + 4 * RNA_DIM, **proj_kw)
            self.atac_slot_projection = Projection(
                input_dim=ATAC_DIM * stat + ATAC_DIM, **proj_kw)
            if args.adapter_rank > 0:
                self.rna_adapter = Adapter(self.num_slots, self.proj_dim,
                                           args.adapter_rank)
                self.atac_adapter = Adapter(self.num_slots, self.proj_dim,
                                            args.adapter_rank)
            self.slot_ids = tuple(priors["rna"].slot_ids.tolist())
            self.slot_names = tuple(priors["rna"].slot_names.tolist())
            self.register_buffer(
                "rna_prior_mass", priors["rna"].matrix.sum(0).clamp_min(1e-8),
                persistent=False)
            self.register_buffer(
                "atac_prior_mass", priors["atac"].matrix.sum(0).clamp_min(1e-8),
                persistent=False)
        self.routing = dict(topk=int(args.routing_topk),
                            mass_power=float(args.routing_mass_power),
                            tail_weight=float(args.routing_tail_weight),
                            normalize_prior_mass=bool(
                                args.routing_normalize_prior_mass))

    # -- parameter groups ------------------------------------------------------------ #

    def refiner_parameters(self) -> List[nn.Parameter]:
        """The 2 FM-shaped layers ONLY -- `proj` rides with the heads.

        These get `--fm_layer_lr`.  ⛔ never 0: `--fm_layer_lr 0` is a BROKEN FREEZE in
        this repo (gradients flow, the parameter never moves, every gradient assert
        stays green and the arm is inert).  Use 1e-12 if a freeze is genuinely wanted.
        """
        return list(self.rna_refiner.layers.parameters()) + \
            list(self.atac_refiner.blocks.parameters())

    def head_parameters(self) -> List[nn.Parameter]:
        ref_ids = {id(p) for p in self.refiner_parameters()}
        return [p for p in self.parameters() if id(p) not in ref_ids]

    def slot_parameters(self) -> List[nn.Parameter]:
        """Parameters that exist ONLY because of the fine branch.  Empty in ARM A -- and
        that emptiness is asserted, in both directions, by `--arm`'s liveness check."""
        out: List[nn.Parameter] = []
        for m in (self.rna_pooler, self.atac_pooler, self.rna_slot_projection,
                  self.atac_slot_projection, self.rna_adapter, self.atac_adapter):
            if m is not None:
                out += list(m.parameters())
        return out

    # -- the encode path (TRAP 2: no torch.no_grad() anywhere below) --------------- #

    def encode(self, rt, rm, gene_id, at, am, ccre_id,
               liveness: Optional[Liveness] = None):
        """Cached FM tokens -> (global embeddings, slot embeddings, valid, mass).

        ONE refiner call per modality serves BOTH branches: `native_grad` returns the
        refined native-width tokens, and the FILIP-style projected tokens are then
        `l2(proj(native))`, which is `forward()` by construction.  Calling `forward()`
        separately would double the refiner's compute AND its activation memory -- the
        single most expensive thing in this model at S_atac = 8192.
        """
        gnat = self.rna_refiner.native_grad(rt, rm)                   # [B, N, 768]
        cnat = self.atac_refiner.native_grad(at, am)                  # [B, M, 512]
        if liveness is not None:
            liveness.bump("refiner_native_grad_calls", 2)

        # cell vectors, recomputed from the REFINED tokens (TRAP 3).  fm_pool_rna finds
        # the meta tokens at (vc-2, vc-1) per cell; fm_pool_atac takes CLS at position
        # 0.
        cell_r = self._fm_pool_rna(gnat, rm)                          # [B, 3072]
        cell_a = l2(cnat[:, 0, :])                                    # [B,  512]

        gproj = l2(self.rna_refiner.proj(gnat))                       # [B, N, proj_dim]
        cproj = l2(self.atac_refiner.proj(cnat))                      # [B, M, proj_dim]
        zr = l2(self.rna_head(self._fm_pool_rna(gproj, rm)))          # [B, proj_dim]
        za = l2(self.atac_head(self._fm_pool_atac(cproj, am)))        # [B, proj_dim]

        if self.arm != "finecls":
            return zr, za, None, None, None, None, None, None

        # RNA: the meta tokens are NOT sliced off.  Their ids 19264/19265 map to
        # row_ids >= prior.shape[0] = 19264, so `in_vocabulary` is False and they carry
        # zero slot mass.  That is asserted (not assumed) by `--arm`'s liveness check.
        rslots, rvalid, rmass = self.rna_pooler(gnat, gene_id, rm)
        # ATAC: drop CLS at position 0 -- it is the cell vector, not a cCRE.  Padding is
        # taken from the EXPLICIT mask, never from `ccre_id == 0`: under the cache's
        # `pad_stack` fill the pad id is -1, and `pad_token_id=103` on the RNA side is
        # the real gene ABLIM1.  Ids are never used to DETECT padding anywhere in this
        # file.
        aslots, avalid, amass = self.atac_pooler(cnat[:, 1:, :], ccre_id[:, 1:],
                                                 am[:, 1:])
        if liveness is not None:
            liveness.bump("slot_pooler_calls", 2)

        rslots = self.rna_slot_projection(
            torch.cat((rslots, cell_r.unsqueeze(1).expand(-1, self.num_slots, -1)), -1))
        aslots = self.atac_slot_projection(
            torch.cat((aslots, cell_a.unsqueeze(1).expand(-1, self.num_slots, -1)), -1))
        if self.routing["normalize_prior_mass"]:
            rmass = rmass / self.rna_prior_mass[None, :]
            amass = amass / self.atac_prior_mass[None, :]
        rslots = self._refine(rslots, rvalid, self.rna_adapter)
        aslots = self._refine(aslots, avalid, self.atac_adapter)
        return zr, za, rslots, aslots, rvalid, avalid, rmass, amass

    def forward(self, *a, **kw):
        """`forward` == `encode`, so DDP sees the whole model in one call.

        DDP only tracks parameters touched inside `forward`; calling `encode` directly
        on a DDP-wrapped module would bypass the reducer's autograd hooks entirely.
        """
        return self.encode(*a, **kw)

    @staticmethod
    def _refine(slots, valid, adapter):
        """`_refine_fixed_slots`, `residualize_common: false` (the reference)."""
        if adapter is not None:
            slots = adapter(slots)
        return torch.where(valid.unsqueeze(-1), slots, torch.zeros_like(slots))


# ------------------------------------------------------------------------------------ #
# The fine loss: the PER-SLOT analogue of the decided cell loss
# ------------------------------------------------------------------------------------ #

def per_slot_direction(query, candidate, q_valid, c_valid, q_mass, c_mass, labels, temp,
                       routing, infonce_frac):
    """One [B, B] contrastive problem PER SLOT, in one direction.

    WHY THIS LOSS AND NOT THE REFERENCE'S.  The reference config uses
    `module_loss_type: per_module_mpnce`, driven by `compute_soft_positive_mask` over an
    EMA teacher.  The standing decision for this project is `mpnce 0` -- which is a
    CELL-level flag, so it does not literally forbid the per-slot one, but running an
    mpnce fine branch under an infonce+supcon global branch introduces a LOSS MISMATCH
    between the two branches that would confound a null.  So the fine loss here is the
    per-slot analogue of the decided cell loss: InfoNCE (the true pair) * infonce_frac +
    cross-modal SupCon (same-cell-type off-diagonal) * (1 - infonce_frac).  Global and
    fine then optimise ONE objective at two granularities, and a null is attributable to
    the BRANCH rather than to the loss.  This is closer to the reference than it looks:
    with `label_positives` + `hierarchical_positives` + `multipositive_scheme: graded`
    their positive set IS same-cell-type, i.e. supcon.  The EMA teacher is a fourth
    moving part we decline to add on a track where the backbone is already changing.

    The routing, masking and weighting are the reference's, verbatim in structure
    (`fixed_slot_trainer._module_mpnce_direction`): route by evidence mass with
    `routing_topk` + `routing_tail_weight`, mask invalid pairs to -1e4, `log_softmax`
    over CANDIDATES within each slot, and weight each (query, slot) term by
    `sqrt(q_weight * c_weight[positive])`.  With `tail_weight > 0` EVERY valid slot is
    routed, not just the top-k, which is what makes slot count multiply supervision.
    """
    q_weight, q_route = fixed_slot_route_weights(
        q_mass, q_valid, routing["topk"], routing["mass_power"], routing["tail_weight"])
    c_weight, c_route = fixed_slot_route_weights(
        c_mass, c_valid, routing["topk"], routing["mass_power"], routing["tail_weight"])
    qn = F.normalize(query.float(), dim=-1)
    cn = F.normalize(candidate.float(), dim=-1)
    scores = torch.einsum("qmd,nmd->qnm", qn, cn)                     # [Q, B, M]
    pair_valid = q_route[:, None, :] & c_route[None, :, :]
    logits = (scores / temp).masked_fill(~pair_valid, -1e4)
    log_prob = F.log_softmax(logits, dim=1)                           # over candidates

    rows = torch.arange(query.shape[0], device=query.device)
    diag = -log_prob[rows, rows, :]                                   # [Q, M] InfoNCE
    per_module = infonce_frac * diag
    if labels is not None and (1.0 - infonce_frac) > 0:
        valid_lab = labels >= 0
        same = ((labels[:, None] == labels[None, :]) & valid_lab[:, None]
                & valid_lab[None, :]).float()
        same = same.clone()
        # Diagonal EXCLUDED: the true pair is owned by the InfoNCE term, exactly as
        # `supcon_xmodal` splits alpha*InfoNCE + (1-alpha)*SupCon_offdiag on the cell
        # side.
        same.fill_diagonal_(0.0)
        sup_f = same[:, :, None] * pair_valid.to(log_prob.dtype)
        n_pos = sup_f.sum(dim=1)
        sup = torch.where(n_pos > 0,
                          -(sup_f * log_prob).sum(dim=1) / n_pos.clamp_min(1.0),
                          torch.zeros_like(diag))
        per_module = per_module + (1.0 - infonce_frac) * sup
    weight = torch.sqrt(q_weight * c_weight[rows]) * pair_valid[rows, rows, :].float()
    loss = (per_module * weight).sum() / weight.sum().clamp_min(1e-8)
    with torch.no_grad():
        eff = pair_valid[rows, rows, :].float().sum(1).mean()
        acc = (logits.argmax(dim=1) == rows[:, None]).float().mean()
    return loss, {"effective_slots": float(eff), "slot_r1": float(acc)}


def fine_loss(rs, a_s, rv, av, rm_, am_, labels, temp, routing, infonce_frac):
    """Symmetrised per-slot loss. Both directions, never averaged silently elsewhere."""
    lr, mr = per_slot_direction(rs, a_s, rv, av, rm_, am_, labels, temp, routing,
                                infonce_frac)
    la, ma = per_slot_direction(a_s, rs, av, rv, am_, rm_, labels, temp, routing,
                                infonce_frac)
    stats = {"fine_r2a": float(lr), "fine_a2r": float(la),
             "effective_slots": ma["effective_slots"],
             "slot_r1_r2a": mr["slot_r1"], "slot_r1_a2r": ma["slot_r1"]}
    return 0.5 * (lr + la), stats


def fused_infonce(zr, za, rs, a_s, rv, av, rm_, am_, temp, routing,
                  alpha: float = 1.0, use_zscore: bool = True):
    """InfoNCE on the score the EVAL ACTUALLY RANKS WITH: zscore(global) + zscore(slot).

    ⛔ NOTHING IN THIS TRAINER OPTIMISED THIS BEFORE.  The global branch
    (cell_infonce + supcon + align) and the fine branch (per-MODULE mpnce) have entirely
    separate losses, and `zsum` is assembled only at eval time in `_fused_fn`.  So the
    model has been asked to make each channel good independently and NEVER to make their
    SUM rank correctly -- which is the quantity every headline number is computed from.

    Two details are load-bearing and both mirror `_fused_fn` exactly:
      * the z-score is INSIDE the loss.  Without it the sum is dominated by whichever
        channel happens to have the larger spread; measured sigma_S/sigma_g = 0.902, so
        the eval's implicit weighting is near-equal and this reproduces it.
      * the slot term is the AGGREGATED `fixed_slot_similarity`, not the per-module
        logits `fine_loss` uses.  Those train each slot in isolation; this trains the
        reduction over slots, which no existing term touches.

    ⚠️ The z-score is over the CONTRASTIVE BATCH here and over the 128-cell POOL at eval.
    Both are "the unit the ranking happens in", but they are not the same population.
    """
    G = zr @ za.t()
    # ⛔ MAP THE KEYS EXPLICITLY -- `**routing` DOES NOT WORK.  `mod.routing` is
    # dict(topk=, mass_power=, tail_weight=, normalize_prior_mass=) while
    # `fixed_slot_similarity` takes routing_topk / routing_mass_power /
    # routing_tail_weight and has no normalize_prior_mass at all, so the splat fails on
    # BOTH the names and the extra key.  `per_slot_direction` maps them by hand for the
    # same reason; this now matches it.
    S = fixed_slot_similarity(rs, a_s, rv, av, rm_, am_,
                              routing_topk=routing["topk"],
                              routing_mass_power=routing["mass_power"],
                              routing_tail_weight=routing["tail_weight"])
    gsd = G.std().clamp_min(1e-6)
    if use_zscore:
        # ⛔ RESTORE THE ORIGINAL SCALE AFTER Z-SCORING, OR `--temp` SILENTLY MEANS
        # SOMETHING ELSE HERE THAN IN `cell_infonce`.  Raw `zr @ za.T` on L2-normalised
        # D-dim vectors has std ~1/sqrt(D) = 0.0625 at D=256; z-scoring sets it to 1, a
        # 16x inflation, so the SAME temp 0.07 would run this term ~16x sharper than the
        # global one and dominate from step 0.  MEASURED on random tensors before this
        # line existed: fused 41.66 vs cell_infonce 5.60, in-batch acc exactly 0.000.
        # The eval does not hit this because argmax is scale-invariant -- in training the
        # scale IS the temperature.
        # `/ sqrt(1 + alpha^2)` keeps the SUM at that same std rather than sqrt(2) of it,
        # so the two channels' RELATIVE weighting is the eval's while the absolute scale
        # matches the objective this term sits next to.
        G = (G - G.mean()) / gsd
        S = (S - S.mean()) / S.std().clamp_min(1e-6)
        scale = gsd / float(np.sqrt(1.0 + alpha * alpha))
    else:
        scale = 1.0
    logits = scale * (G + alpha * S) / temp
    lab = torch.arange(zr.shape[0], device=zr.device)
    l = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    with torch.no_grad():
        acc = 0.5 * ((logits.argmax(1) == lab).float().mean().item()
                     + (logits.t().argmax(1) == lab).float().mean().item())
    return l, acc


def fine_logit_gib(batch: int, slots: int) -> float:
    """Bytes of ONE `[B, B, M]` fp32 tensor, in GiB. ~6 are live per direction."""
    return batch * batch * slots * 4 / 1024 ** 3


# ------------------------------------------------------------------------------------ #
# GradCache, generalised to an N-tuple (TRAP 4)
# ------------------------------------------------------------------------------------ #

def grad_cache_two_pass_n(micro_batches, forward_fn, loss_fn, device, ddp_mods=(),
                          rng_replay=True, verify=False, verify_tol=1e-2, log=print,
                          scaler=None, probe=None):
    """OpenCLIP/GradCache two-pass over an ARBITRARY tuple of cached feature tensors.

    `grad_cache_two_pass` (train_filip_combined.py:549) is the right algorithm and is
    stronger than OpenCLIP's own `--accum-freq` (the loss is taken ONCE on the full
    gathered batch instead of being recomputed per micro-batch, and it adds an RNG
    replay plus a run-time precondition check).  It only hard-codes `hr, ha =
    forward_fn(mb)`, two `[b, D]` tensors, so it cannot carry ARM B's `[b, M, 256]` slot
    embeddings.  The FILIP guard's rationale does not transfer: FILIP caches kg=1024
    tokens PER CELL, while FineCLS caches M=64 slot vectors per cell -- 268 MB of leaves
    at B=512, M=64.

        forward_fn(mb) -> (tensors: Tuple[Tensor, ...], aux: dict)
        loss_fn(tensors_full, aux_full) -> (total, extra)

    `aux` carries the NON-differentiable per-cell quantities (slot `valid`, slot `mass`,
    labels).  They get NO leaf and are NOT cached across the passes: `mass[b,m] =
    sum_n prior[id_n,m] * valid_n` is a pure function of token ids with no learnable
    weights, so pass 2 recomputes them identically -- and it MUST recompute them, or the
    pass-2 graph is not the graph pass 1 measured.

    ⛔ `verify` loops the FULL returned tuple. Checking only the globals would be a PROXY
    assert: it would print OK on a run whose slot branch replayed different dropout
    masks, i.e. whose fine gradients were wrong while the cell term verified clean.

    THE LOSS SCALE GOES HERE, AT THE PASS-1 LEAF BACKWARD, AND NOWHERE ELSE.
    `torch.autograd.backward(ts, gs)` in pass 2 is a vector-Jacobian product, i.e. LINEAR
    in `gs`.  So "scale the loss" and "scale the cached feature gradients at injection"
    are the SAME operation (measured bit-identical, relL2 0.000e+00, each equal to S x
    the single-pass gradient to float64 machine precision), and doing BOTH gives S^2.
    Scaling HERE is the placement that is also numerically right: the backward through
    the fp16 refiner stacks is seeded not by the loss (magnitude ~1) but by
    dL/dfeature ~ O(1/B_eff) = O(1/512), already three orders down before it reaches a
    single fp16 op.  Raising THAT seed is the whole mechanism -- measured, the fp16 ATAC
    refiner gradient sits at relL2 5.35e-01 from an all-fp32 reference at S=1 and
    9.9e-04 at S=2^20.

    ⛔ THE SCALE MUST NOT LIVE INSIDE `loss_fn`.  `scaler.scale()` returns a NEW tensor;
    if `loss_fn` returned it, the `total.detach()` below would hand the caller a loss S
    times too large while the per-term `stats` (taken as `float(l)` inside `loss_fn`)
    stayed in true units, and every logged number would silently change meaning.  The
    TRUE loss is what is returned.

    `probe`, when a dict is passed, receives `leaf_grad_absmax` -- the largest cached
    dL/dfeature that pass 2 injects.  That is the quantity a GradScaler CONTROLS, so it
    is what the liveness check asserts on; `scaler.get_scale()` is a configuration read,
    not evidence that the scale reached the graph.
    """
    fp = _filip_pieces()
    states, leaves, auxes = [], [], []
    for mb in micro_batches:                  # ---- pass 1: features, no graph ----
        states.append(fp["rng_snapshot"](device))
        with torch.no_grad():
            outs, aux = forward_fn(mb)
        # ⛔ THE LEAVES MUST BE FP32.  `scaler.unscale_` guards PARAMETER grads only (it
        # raises "Attempting to unscale FP16 gradients"); it never sees a leaf.  An fp16
        # leaf would accumulate the cached dL/dz in fp16 -- i.e. it would underflow
        # BEFORE the scale could help -- and nothing would say so.  `gradcache.py`'s
        # sibling has this guard; this one did not until the scaler landed.
        for t in outs:
            assert t.dtype == torch.float32, (
                f"grad_cache leaf came back {t.dtype}, not float32. The cached "
                f"dL/dfeature would be accumulated in half precision, underflowing "
                f"before the loss scale can reach the refiner stack. Both refiners' "
                f"`native_grad` end in `.float()`; something broke that contract.")
        leaves.append(tuple(t.detach().clone().requires_grad_(True) for t in outs))
        auxes.append(aux)
    full = tuple(torch.cat([lv[i] for lv in leaves], 0) for i in range(len(leaves[0])))
    aux_full = {}
    for k in auxes[0]:
        vals = [a[k] for a in auxes]
        aux_full[k] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    total, extra = loss_fn(full, aux_full)    # ---- loss on the FULL batch, ONCE ----
    # -> leaf.grad = S * dL/dfeature per micro-mb (S = 1 when no scaler is enabled)
    (scaler.scale(total) if scaler is not None else total).backward()
    if probe is not None:
        probe["leaf_grad_absmax"] = max(
            (float(lf.grad.detach().abs().max()) for lv in leaves for lf in lv
             if lf.grad is not None), default=0.0)
    last, vmax = len(micro_batches) - 1, 0.0
    for i, mb in enumerate(micro_batches):     # ---- pass 2: cached-grad backward ----
        if rng_replay:
            fp["rng_restore"](states[i], device)
        with ExitStack() as st:
            if i != last:                     # DDP: one all-reduce per opt step
                for m in ddp_mods:
                    st.enter_context(m.no_sync())
            outs, _ = forward_fn(mb)
            if verify:
                for t, leaf in zip(outs, leaves[i]):
                    d = (t.detach().float() - leaf.detach().float()).abs().max()
                    s = leaf.detach().float().abs().max().clamp(min=1e-6)
                    vmax = max(vmax, float(d / s))
            ts, gs = [], []
            for t, leaf in zip(outs, leaves[i]):
                if leaf.grad is not None:
                    ts.append(t)
                    gs.append(leaf.grad)
            if ts:
                # NO 1/accum rescale -- the loss was taken once over the full batch.
                torch.autograd.backward(ts, gs)
    if verify:
        if vmax > verify_tol:
            raise RuntimeError(
                f"grad_cache RNG replay FAILED: pass-2 features differ from the pass-1 "
                f"cached ones by {vmax:.3e} (relative, tol {verify_tol:g}). The cached "
                f"dL/dz therefore belongs to a DIFFERENT function than the graph it is "
                f"backpropagated through, i.e. EVERY gradient this run makes is wrong. "
                f"Most likely the dropout masks are not reproduced on this backend. "
                f"Re-run with --grad_cache 0 or --dropout 0.")
        log(f"  [grad_cache] pass1/pass2 replay check over {len(leaves[0])} cached "
            f"tensors: max rel diff {vmax:.3e} (tol {verify_tol:g}) OK")
    return total.detach(), extra


# ------------------------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # -- THE arm-defining flag ------------------------------------------------------- #
    p.add_argument("--arm", choices=("cell_only", "finecls"), required=True,
                   help="cell_only = ARM A (global branch only); finecls = ARM B/C "
                        "(global + FineCLS fixed slots). This is the ONE intended "
                        "difference between arms; all else must be byte-identical.")
    p.add_argument("--num_slots", type=int, default=64,
                   help="M. A free parameter, not fixed at 64 -- but 64-vs-80 already "
                        "ran matched (f80-f64 = +0.0012, smaller than f64's own "
                        "eval-to-eval jitter of 0.0013). Memory never binds: even "
                        "M=256 is ~5.7%% of a 140 GiB H200.")
    p.add_argument("--prior_rna_path", default=None)
    p.add_argument("--prior_atac_path", default=None)
    p.add_argument("--prior_rna_matrix", default="membership")
    p.add_argument("--prior_atac_matrix", default="normalized_membership")
    p.add_argument("--gene_list", default=os.path.join(
        REPO, "scFoundation/model/OS_scRNA_gene_index.19264.tsv"))
    p.add_argument("--ccre_bed", default=os.path.join(REPO, "EpiAgent/data/cCRE.bed"))
    p.add_argument("--skip_vocab_gate", type=int, default=0,
                   help="SMOKE ONLY: skip the feature-id/symbol order gate (reading "
                        "cCRE.bed costs ~20 s). Refused unless --smoke.")

    # -- data ------------------------------------------------------------------------ #
    p.add_argument("--cache_root", default=os.path.join(
        REPO, "fm_token_cache_expanded_8192"))
    p.add_argument("--train_split", default="train")
    p.add_argument("--val_split", default="val")

    # --- LIVE-FM CORPUS SCOPE (raw h5ads; the token cache is NOT used) -------------- #
    # 456 h5ad PAIRS == the projector track's train corpus.  Counted with h5py, not
    # estimated:
    #   163 base  preprocessed_data_expanded_20260305/  1,046,380 cells
    #   255 new   preprocessed_data_3m/paired/          3,254,285 cells
    #    38 new   preprocessed_data_newtissue/paired/     359,071 cells
    #   = 4,659,736 barcodes, ZERO duplicates, ZERO cross-cohort collision, ~897 GB.
    # ⛔ A GLOB IS NOT SAFE.  preprocessed_data_expanded_20260305/ also holds
    #    *_preprocessed.h5ad (the pre-pairing stage) AND a STALE all_rna_paired.h5ad /
    #    all_atac_paired.h5ad of 1,055,013 cells -- a DIFFERENT, larger unit set than
    #    the 163 the split was built from.  The file list is therefore DATA, in a
    #    manifest, whose row count and cell total are asserted at load.
    p.add_argument("--corpus_manifest", default=os.path.join(
        HERE, "corpus_livefm_456.tsv"),
        help="TSV, header unit/cohort/tissue/rna_h5ad/atac_h5ad/n_cells, 456 rows. The "
             "file list for --live_fm 1. Regenerate from "
             "experiments/newtissue/base_units.txt + "
             "preprocessed_data_3m/cache/MERGE_MANIFEST.json['libraries'].")
    p.add_argument("--corpus_cohorts", default="base,new",
                   help="comma list of manifest `cohort` values to TRAIN on. "
                        "'base,new' = the full 4.45M-cell corpus; 'base' alone = the "
                        "830,567-cell base-only control. The val split always "
                        "restricts to 'base' -- no val cell exists outside it.")
    # ⛔ THE SPLIT IS CELL-LEVEL, NOT LIBRARY-LEVEL.  All 163 base units contribute val
    #    cells, so reading the base h5ads WHOLE leaks 103,775 val + 103,882 test cells
    #    into train.  The 293 new units are 100% train and must NOT be whitelisted --
    #    none of their barcodes is in the list, so applying it to them would filter that
    #    entire 3.6M-cell cohort to ZERO cells and the run would silently train on
    #    830,567.  `resolve_live_corpus` therefore attaches the list PER UNIT.
    p.add_argument("--train_whitelist", default=os.path.join(
        HERE, "corpus_livefm_train_whitelist_base163.txt"),
        help="830,567 train barcodes, BASE cohort only. Applied to cohort=='base' units "
             "only. '' = off, which LEAKS val+test and is refused unless "
             "--corpus_cohorts excludes base.")
    p.add_argument("--val_whitelist", default=os.path.join(
        HERE, "corpus_livefm_val_whitelist.txt"),
        help="103,775 val barcodes == the base cache's own val index. Disjoint from "
             "train and from test.")
    # --- VAL LOSS AS THE MODEL-SELECTION CRITERION (user decision, 2026-08-23) ---------
    # ⛔ WHY NOT `best_model.pt`: this project's `best_model.pt` was mirrored from
    # `best_ood_model.pt` and selected on bmmc + fetal_heart -- HALF the OOD panel -- so
    # every absolute number taken from it inherits that leak.  Selecting on VAL, which is
    # held-out but IN-DISTRIBUTION, is the clean form of the same idea: it cannot leak the
    # OOD panel because it never touches it.
    # ⚠️ SELECT on val, REPORT on OOD, and do not expect them to agree: this project has
    # measured val_loss -0.15 moving OOD by +0.002 (indist_gains_do_not_transfer_ood).
    # ⚠️ val loss carries an `ln(pool)` offset, so it is comparable only at a FIXED pool
    # (a0 used the default 32, the cis arms 128: a raw 1.58-nat gap was really 0.19).
    # Here the pool is micro_batch*accum_freq, identical across arms and checkpoints.
    # ⚠️ ARM A and ARM B minimise DIFFERENT objectives (B adds the fine term), so val loss
    # is comparable WITHIN an arm -- which is all selection needs -- and NOT across arms.
    # The cross-arm claim is OOD retrieval, never this number.
    # --- RESUME, to train an arm to CONVERGENCE ---------------------------------------
    # WHY: the seed-0 val curves show ARM A flat from ~step 5000 (6000->8000 t=-0.31) but
    # ARM B STILL IMPROVING at 6000->8000 (delta -0.0238, t=-2.13).  B carries the slot
    # branch and converges later.  Comparing a converged A against a still-improving B at
    # a fixed step BIASES THE COMPARISON AGAINST B -- the arm the experiment exists to
    # evaluate -- so a null would be confounded with "B just needed more steps".
    # ⚠️ THE SAMPLER IS RESEEDED on resume.  `same_dataset_blocked` has NO epochs: it
    # draws a dataset uniformly per block and then cells WITH replacement, so it is
    # memoryless and a fresh seed is a valid continuation, NOT a replay.  Reusing
    # --seed would replay the identical 8000 blocks.  The derived seed is recorded in the
    # manifest and in liveness so the continuation is auditable.
    # --- IN-TRAINING VAL LOSS + BEST-CHECKPOINT TRACKING (user decision) --------------
    # The seed-0 wave had NO in-training eval at all, so convergence had to be
    # reconstructed post hoc from 12 numbered snapshots.  Computing it inline makes the
    # curve a first-class output and lets `best_by_valloss.pt` be maintained as the run
    # goes, instead of an argmin taken afterwards.
    # ⛔ RNG SAFETY: the eval must not perturb the training stream, or the continuation
    # stops being the run it would have been.  The torch/cuda RNG state is saved before
    # the eval and restored after, and the eval uses its OWN fixed-seed loader.
    p.add_argument("--val_every", type=int, default=0,
                   help="compute val loss every N optimizer steps (0 = off). The blocks "
                        "are FIXED across evaluations, so the curve is paired.")
    p.add_argument("--val_blocks", type=int, default=40)
    p.add_argument("--val_seed", type=int, default=12345)
    # ---- VAL RETRIEVAL R@1 (the second early-stopping criterion) ------------------ #
    # Scored INSIDE the --val_every hook, on the forward passes the val loss already
    # paid for, so its marginal cost is a batched [D,P,P] bmm per block (~13 MB,
    # milliseconds) against a live-FM forward of the whole block.  A separate retrieval
    # pass would double the most expensive part of the run.
    p.add_argument("--val_r1_pool", type=int, default=128,
                   help="gallery size for val R@1. 128 is the project standing pool; "
                        "THE POOL DEFINITION IS THE METRIC (the same cells score 0.1616 "
                        "at the dataset window vs 0.0329 within donor x cell_type), so a "
                        "run that changes it is not comparable to one that does not.")
    p.add_argument("--val_r1_draws", type=int, default=200,
                   help="pools drawn per val block, crc32-seeded on (--val_seed, block) "
                        "so every checkpoint draws the IDENTICAL pattern and the curve "
                        "is paired. 0 disables val R@1 entirely -- which also disables "
                        "early stopping, because the stop rule needs BOTH criteria.")
    p.add_argument("--val_r1_direction", choices=("mean", "r2a", "a2r"), default="mean",
                   help="which scalar the R@1 stall counter and best_by_valr1.pt are "
                        "selected on. BOTH directions are always computed and written "
                        "to val_loss_curve.jsonl; this only names the SELECTION scalar. "
                        "'mean' is a selection convenience and must NEVER be reported "
                        "as a retrieval number -- directions are never averaged here.")
    # ---- EARLY STOPPING ----------------------------------------------------------- #
    # ⛔ THE STANDING RULE IS *BOTH*: a run stops only when val_loss AND val R@1 have
    # each failed to improve for --early_stop_patience consecutive evaluations. Either
    # one alone has been wrong here before -- a val_loss argmin selector once discarded
    # a model whose val R@1 was 38% higher.
    # ⚠️ AND BOTH ARE IN-DISTRIBUTION. OOD retrieval saturates by ~20,000 steps while
    # in-dist keeps climbing to 96,000, so this rule will NOT stop at the OOD optimum
    # and must not be described as if it did. It is a compute guard; the OOD panel is
    # still scored offline from the numbered snapshots.
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="consecutive --val_every evaluations with no improvement in "
                        "BOTH val_loss and val R@1 before the run stops. 0 = OFF (the "
                        "default, so every previously shipped run is unchanged).")
    p.add_argument("--early_stop_min_delta_loss", type=float, default=1e-3,
                   help="val_loss must FALL by more than this to count as an "
                        "improvement. Must be > 0: at 0 the noisy criterion never "
                        "stalls and the AND-rule never fires.")
    p.add_argument("--early_stop_min_delta_r1", type=float, default=2e-3,
                   help="val R@1 must RISE by more than this to count as an "
                        "improvement. Set it from the run's own val_r1 `sem` (written "
                        "to val_loss_curve.jsonl, = sd across BLOCKS / sqrt(blocks)) "
                        "after the first few evals, not from a guess -- this metric has "
                        "never existed on this trainer, so 2e-3 is a placeholder.")
    p.add_argument("--early_stop_min_step", type=int, default=0,
                   help="no early stop before this step, whatever the counters say.")
    p.add_argument("--cooldown_start", type=int, default=0,
                   help="step the linear lr->0 decay is anchored at; with "
                        "--resume it should be the resume step so the LR is "
                        "continuous across the restart.")
    p.add_argument("--resume", default=None,
                   help="numbered snapshot to continue from: restores model, optimizer, "
                        "scheduler and grad_scaler, and runs from its step to --num_steps.")
    p.add_argument("--val_loss_ckpt", default=None,
                   help="EVAL-ONLY: load this snapshot, compute val loss on --val_split, "
                        "print it as JSON and exit. Trains nothing.")
    p.add_argument("--val_loss_blocks", type=int, default=40,
                   help="optimizer-steps' worth of val to average over. The SAME blocks "
                        "are used for every checkpoint (--val_loss_seed), so the sample "
                        "noise is COMMON and cancels in a checkpoint-to-checkpoint "
                        "comparison -- which is what selection needs.")
    p.add_argument("--val_loss_seed", type=int, default=12345,
                   help="FIXED, and deliberately NOT --seed: every arm and every "
                        "checkpoint must see the identical val cells in the identical "
                        "order, or the curve measures the sample rather than the model.")
    p.add_argument("--labels_csv", default=None,
                   help="master_labels.csv-style file. Without it every label-driven "
                        "term (supcon, per-slot supcon) is silently inert, so the "
                        "loader refuses an unlabeled cache unless --allow_unlabeled.")
    p.add_argument("--allow_unlabeled", type=int, default=0)
    p.add_argument("--dataset_id_column", default="dataset_id")
    p.add_argument("--max_atac_length", type=int, default=8192,
                   help="ATAC pad width. 8192 for every real run: the cache is built "
                        "at 8192 and 70-84%% of cells exceed 4096 on 5/6 sets. 0 means "
                        "'pad to the batch max', which is SMOKE ONLY -- it makes the "
                        "width batch-dependent.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=0,
                   help="DataLoader batches kept in flight PER WORKER. 0 = auto = the "
                        "smallest value for which num_workers * prefetch_factor >= "
                        "accum_freq. ⛔ THIS IS A THROUGHPUT FLAG WITH TEETH: the step "
                        "loop pulls accum_freq micro-batches SYNCHRONOUSLY at the top "
                        "of every step, so whatever the buffer cannot supply is paid on "
                        "the critical path, ON TOP of the GPU time -- torch's default "
                        "of 2 buffers 2*num_workers = 24 of the 64 a step needs, and "
                        "the other 40 were measured at ~2.3 s/step. It moves NO data "
                        "and changes NO order: a DataLoader with a batch_sampler "
                        "returns batches in sampler order regardless of how far ahead "
                        "the workers ran. Costs host RAM: ~115 MiB per buffered "
                        "micro-batch at micro 8 / S_atac 8192.")
    p.add_argument("--train_shards", default="all")

    # -- batch geometry -------------------------------------------------------------- #
    p.add_argument("--micro_batch", type=int, default=32)
    p.add_argument("--accum_freq", type=int, default=16,
                   help="contrastive batch = micro_batch * accum_freq * world_size")
    p.add_argument("--grad_cache", type=int, default=1)
    p.add_argument("--grad_cache_verify", type=int, default=1)
    p.add_argument("--gather_negatives", type=int, default=0)
    p.add_argument("--check_gradcache_equiv", type=int, default=0,
                   help="L8: at step 0, compare the two-pass gradients against the "
                        "single pass, AND against a replay-OFF negative control that "
                        "must FAIL. Costs 3 extra steps of compute, once.")
    p.add_argument("--gradcache_equiv_micro", type=int, default=3,
                   help="how many micro-batches the L8 probe uses. ⛔ NOT accum_freq: "
                        "the reference `single_pass_step` keeps EVERY micro-batch's "
                        "graph alive at once -- the exact footprint gradient caching "
                        "exists to avoid. MEASURED on an H200 at micro_batch 32, "
                        "S_atac 8192: 18.9 / 28.1 / 42.4 / 59.2 GiB at 1/2/3/4 "
                        "micro-batches, i.e. +13.4 GiB each, so accum_freq 16 would "
                        "need ~220 GiB of a 139.8 GiB card and OOM before step 0. "
                        "Equivalence is a property of the ALGORITHM, not of the accum "
                        "count -- test_gradcache_equiv.py's T6 proves it over the six "
                        "factorisations of B=12 -- so 3 is sufficient and 0 disables.")
    p.add_argument("--max_fine_logit_gib", type=float, default=4.0,
                   help="guard on ONE [B,B,M] fp32 tensor; ~6 are live per direction")

    # -- optimisation ---------------------------------------------------------------- #
    p.add_argument("--num_steps", type=int, default=8000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5, help="heads / slot branch")
    p.add_argument("--fm_layer_lr", type=float, default=1e-5,
                   help="the 2 FM-shaped refiner layers. NEVER 0 -- `--fm_layer_lr 0` "
                        "is a BROKEN freeze here (grads flow, params never move); use "
                        "1e-12.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_schedule", choices=("constant", "cosine", "cooldown"),
                   default="constant",
                   help="STANDING DECISION: constant, passed explicitly. The default "
                        "cosine elsewhere reaches 1/1300 of peak by 39.3k steps, so a "
                        "flat tail there is the SCHEDULE, not convergence.")
    p.add_argument("--grad_clip", type=float, default=4.0)

    # -- loss ------------------------------------------------------------------------ #
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--module_temperature", type=float, default=0.07)
    p.add_argument("--cell_infonce_weight", type=float, default=0.5)
    p.add_argument("--supcon_weight", type=float, default=0.5)
    p.add_argument("--align_weight", type=float, default=0.1)
    p.add_argument("--fine_weight", type=float, default=1.0)
    p.add_argument("--fine_infonce_frac", type=float, default=0.5,
                   help="per-slot InfoNCE vs per-slot cross-modal SupCon split; "
                        "0.5/0.5 mirrors the decided cell loss exactly")
    p.add_argument("--fine_warmup_steps", type=int, default=200)

    # -- fine-branch architecture (reference config, verbatim) ----------------------- #
    p.add_argument("--pooling_temperature", type=float, default=0.5)
    p.add_argument("--pool_topk", type=int, default=0)
    p.add_argument("--adapter_rank", type=int, default=8)
    p.add_argument("--projection_layers", type=int, default=3)
    p.add_argument("--projection_expansion", type=int, default=2)
    p.add_argument("--projection_dropout", type=float, default=0.1)
    p.add_argument("--routing_topk", type=int, default=16)
    p.add_argument("--routing_tail_weight", type=float, default=0.15)
    p.add_argument("--routing_mass_power", type=float, default=1.0)
    p.add_argument("--routing_normalize_prior_mass", type=int, default=1)

    # -- refiner --------------------------------------------------------------------- #
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--proj_dim", type=int, default=256,
                   help="256. 512 measurably HURTS on this project's own measurement.")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--atac_sdpa", type=int, default=1,
                   help="required for max_atac_length 8192; math-identical, and "
                        "without it the [B,H,S,S] score matrix OOMs")

    # -- SPEED (see RUNBOOK_training.md 'Why a step costs what it costs') ------------ #
    # Every flag below is a SPEED flag. The first two are MATH-IDENTICAL; the last two
    # are not, and are default-OFF for that reason.  All four are written into the
    # manifest and into every snapshot, because a later reader must be able to tell
    # which arithmetic produced a checkpoint.
    p.add_argument("--rna_grad_ckpt", type=int, default=0,
                   help="gradient checkpointing on the RNA refiner. MEASURED cost 4.7 "
                        "s of a 34.4 s step (13.8%%) to save memory nobody needs: the "
                        "whole-step peak is 13.3 GiB of 139.8. Math-identical either "
                        "way (the recompute forks the RNG, so dropout masks replay). "
                        "Default 0 = OFF; set 1 only if the peak actually binds.")
    p.add_argument("--rna_sub_batch", type=int, default=1,
                   help="run the RNA refiner on CONTIGUOUS groups of this many cells, "
                        "each trimmed to ITS OWN valid width. 0 = off (one call at the "
                        "micro-batch's padded width). MATH-IDENTICAL: the refiner is a "
                        "per-cell function and nothing is sorted or moved between "
                        "micro-batches, so batch composition is untouched. 1 pads "
                        "NOTHING. rna_len is mean 2316 / p50 1953 / max 8497, so the "
                        "shipped path spends ~2.3x the attention FLOPs it needs.")
    p.add_argument("--atac_varlen", type=int, default=0,
                   help="flash_attn varlen (unpad -> cu_seqlens -> pad_input) INSTEAD "
                        "of --atac_sdpa. NOT math-identical: MEASURED 3.6e-05 on "
                        "l2(proj(tokens)), the same class as the use_sdpa swap this "
                        "repo already accepted at 3.8e-05. It is the ONLY route off "
                        "ATAC padding -- atac_len's median IS the 8192 cap, so no "
                        "regrouping can shrink the batch max.")
    p.add_argument("--refiner_precision",
                   choices=("shipped", "atac_bf16", "bf16", "fp16", "fp32"),
                   default="shipped",
                   help="fp16 = BOTH refiners in fp16 WITH a GradScaler (the PRODUCTION "
                        "policy from 2026-08-22; see --grad_scaler_init_scale). shipped "
                        "= RNA fp32 + ATAC fp16 and NO scaler -- a PRE-EXISTING DEFECT, "
                        "not a design: measured 5.35e-01 relative gradient error at "
                        "group cosine 0.845 against an all-fp32 reference, i.e. roughly "
                        "half the ATAC gradient signal lost to fp16 underflow in every "
                        "run on this track before the scaler landed. atac_bf16 = fix "
                        "ONLY the ATAC side, by range rather than by scale, at zero "
                        "speed cost. bf16 = both refiners in bf16 (scale-invariant, no "
                        "scaler needed, but 4.2e-03 / 5.3e-03 relative and it never "
                        "gets closer). fp32 = NO autocast anywhere -- the arithmetic "
                        "REFERENCE the others are measured against; not a production "
                        "option and INCOMPATIBLE with --atac_varlen (flash_attn asserts "
                        "fp16/bf16 qkv). \u26d4 EVERY value here MOVES THE GRADIENT: "
                        "all 9 runs must share one policy.")

    # -- the GradScaler, which only exists because fp16 has 5 exponent bits ----------- #
    # These two are RECORDED IN THE MANIFEST because they are arithmetic, not logging: a
    # checkpoint whose loss scale cannot be recovered cannot be compared to another one.
    p.add_argument("--grad_scaler_init_scale", type=float,
                   default=float(GRAD_SCALER_INIT_DEFAULT),
                   help="initial (and, with growth off, the operating) loss scale. "
                        "MEASURED on real cached data against an all-fp32 reference: "
                        "the ATAC refiner gradient goes 5.35e-01 (S=1) -> 4.7e-03 "
                        "(2^16) -> 9.9e-04 (2^20) and the RNA refiner 3.89e-01 -> "
                        "4.4e-03 -> 1.6e-03, while bf16 sits at 4.2e-03 / 5.3e-03 "
                        "SCALE-INVARIANTLY. \u26d4 AT THE TORCH DEFAULT 2^16 THE fp16 "
                        "POLICY IS A WASH WITH bf16 and buys nothing for the cost of a "
                        "scaler; the whole case for fp16 lives at 2^19-2^20. Ignored "
                        "unless an fp16 policy is live.")
    p.add_argument("--grad_scaler_growth_interval", type=int, default=0,
                   help="0 = GROWTH OFF (interval set past --num_steps), which is the "
                        "default and is a SCIENCE choice, not a tuning one: a scaler "
                        "that grows makes the 9 arms execute different arithmetic, and "
                        "the arm matrix exists to differ in exactly one thing. Backoff "
                        "stays live either way, so a transient overflow costs one "
                        "skipped step instead of the run. Torch's own 2000 is 6.8 h of "
                        "wall clock per doubling at 12 s/step and was tuned for "
                        "millisecond steps.")
    p.add_argument("--grad_scaler_min_scale", type=float, default=float(2 ** 16),
                   help="hard floor. AT OR BELOW 2^16, MEASURED, fp16 is no better than "
                        "bf16 (4.4e-03 / 4.8e-03 vs bf16's 5.3e-03 / 4.2e-03) and at "
                        "2^14 it is 3-6x WORSE, so a scaler that has backed off that "
                        "far has VOIDED the policy choice: fail loudly rather than "
                        "quietly train a worse model. From the 2^20 default this is 4 "
                        "backoffs of room.")
    p.add_argument("--max_overflow_skip_frac", type=float, default=0.01,
                   help="abort if more than this fraction of steps were skipped for a "
                        "non-finite gradient. A scaler that skips 10%% of steps is not "
                        "a scaler, it is a broken operating point, and the run would "
                        "quietly take 10%% fewer updates than its sibling arms.")
    p.add_argument("--init_from_fm_state", default=None,
                   help="a .pt written by --dump_fm_layer_state: the FM's TOP n_layers "
                        "state dicts, so the refiner starts as a seamless continuation")
    p.add_argument("--dump_fm_layer_state", default=None,
                   help="write that .pt and exit (loads the FMs; needs "
                        "--rna_encoder_path/--atac_encoder_path)")
    p.add_argument("--rna_encoder_path", default=None)
    p.add_argument("--atac_encoder_path", default=None)

    # -- LIVE FM (raw h5ad corpus instead of the token cache) ------------------------ #
    p.add_argument("--live_fm", type=int, default=0,
                   help="read the RAW paired h5ads listed in --corpus_manifest (or "
                        "globbed from --live_paired_dir) and run scFoundation/EpiAgent "
                        "IN THIS PROCESS instead of replaying the token cache. 0 "
                        "(default) leaves --cache_root the source and every existing "
                        "run byte-identical. ⛔ REUSES --rna_encoder_path / "
                        "--atac_encoder_path, which today feed only "
                        "--dump_fm_layer_state. ⛔ THE FM IS THE DOMINANT COST: at the "
                        "measured 102 ms/cell (scFoundation fp32, H200, rna_bs=4, no "
                        "SDPA) and micro_batch x accum_freq cells per rank per step "
                        "this dwarfs the refiner's own ~6.8 s. TIME ONE SHORT JOB at "
                        "the production geometry before committing 4 GPUs for weeks.")
    p.add_argument("--live_paired_dir", default=None,
                   help="ESCAPE HATCH: a single directory of *_rna_paired.h5ad / "
                        "*_atac_paired.h5ad pairs, used INSTEAD of --corpus_manifest, "
                        "with the crc32 per-cell val holdout instead of the whitelists. "
                        "⛔ Do not point it at preprocessed_data_expanded_20260305 -- "
                        "that directory holds a stale all_*_paired.h5ad and the "
                        "pre-pairing *_preprocessed.h5ad files.")
    p.add_argument("--live_val_holdout_mod", type=int, default=100,
                   help="--live_paired_dir only: a cell is VAL iff "
                        "crc32('<dataset_id>|<barcode>') %% mod == "
                        "--live_val_holdout_rem. ⛔ crc32, never hash(): python's str "
                        "hash is PYTHONHASHSEED-salted, so hash() would put a different "
                        "set of cells in val on every rank and every restart -- an "
                        "unpaired val curve AND a train set that leaks val. Per-CELL and "
                        "not per-FILE because the blocked sampler draws a DATASET first "
                        "and then cells inside it.")
    p.add_argument("--live_val_holdout_rem", type=int, default=0)
    p.add_argument("--live_open_files", type=int, default=8,
                   help="per-worker cap on simultaneously open h5ad pairs, FIFO "
                        "eviction. The blocked sampler draws ONE dataset per optimizer "
                        "step and there is one dataset per file, so 8 evicts "
                        "approximately never while 456 x num_workers fds is avoided.")
    p.add_argument("--live_verify_pairing", type=int, default=1,
                   help="read BOTH obs/_index arrays of every pair at startup and "
                        "assert they are equal row for row. This is the "
                        "barcode-scramble backstop: a silent RNA<->ATAC row shift gives "
                        "every cell another cell's chromatin and is INVISIBLE in the "
                        "loss (measured rna_cos 1.0 / atac_cos 0.89). It costs one "
                        "extra index read per file per rank; 0 keeps the cheaper "
                        "row-count and sentence-count checks only and is for a REPEAT "
                        "run of an already-verified corpus.")
    p.add_argument("--fm_sdpa", type=int, default=1,
                   help="put the 12 scFoundation encoder layers in train() with every "
                        "dropout p=0 so attention routes through SDPA, O(B*N*D), "
                        "instead of BetterTransformer's DENSE [B,12,N,N] fp32 score "
                        "matrix. MATHEMATICALLY IDENTICAL (p=0 dropout is a no-op and "
                        "these layers have no other train/eval-dependent op). N reaches "
                        "~9.5k on this corpus, so this is the largest FM speed lever "
                        "that changes no arithmetic -- and without it micro_batch 32 "
                        "OOMs a 140 GiB H200 outright.")
    p.add_argument("--fm_autocast_rna", type=int, default=0,
                   help="run the RNA FM under autocast. ⛔ NOT ARITHMETIC-FREE: fp16 "
                        "COMPUTE error is 3.7e-03, 10.6x the 3.5e-04 of fp16 STORAGE, "
                        "and the cache was built fp32-compute on purpose. It is also "
                        "~4x faster (the widely-quoted 24 ms/cell constant is bf16; the "
                        "fp32 measurement is 102). The ATAC side already autocasts "
                        "unconditionally. Default OFF so a live-FM run is comparable to "
                        "the cached arms; turned on it is an ARM-DEFINING property and "
                        "all arms must share it.")
    p.add_argument("--fm_token_dtype", choices=("fp16", "fp32"), default="fp16",
                   help="dtype the hoisted FM tokens are HELD in between the FM and the "
                        "refiner. fp16 is the cache's own storage policy (3.5e-04, at "
                        "the fp16 floor) and halves the memory an accum_freq block of "
                        "tokens occupies; make_fm_tokens_from_cache widens with "
                        ".float() either way.")

    # -- centring -------------------------------------------------------------------- #
    p.add_argument("--fused_weight", type=float, default=0.0,
                   help="InfoNCE on zscore(global)+alpha*zscore(slot) -- the score the "
                        "eval ranks with. 0 reproduces every run before this flag existed.")
    p.add_argument("--fused_alpha", type=float, default=1.0,
                   help="slot weight inside the fused logits; 1.0 matches `_fused_fn`")
    p.add_argument("--fused_zscore", type=int, default=1,
                   help="z-score each channel before summing, as the eval does")
    p.add_argument("--center_global_by_dataset", type=int, default=1)
    p.add_argument("--center_slots_by_dataset", type=int, default=1,
                   help="meaningless without slots; asserted to execute 0 times in "
                        "ARM A and > 0 times in ARM B/C")

    # -- bookkeeping ----------------------------------------------------------------- #
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--save_steps", type=int, default=500,
                   help="numbered snapshots snapshots/step_%%06d.pt. ⛔ best_model.pt "
                        "LEAKS half the OOD panel; score arms at a FIXED step.")
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--ddp_timeout_min", type=int, default=120,
                   help="process-group / NCCL-watchdog timeout in MINUTES. ⛔ torch's "
                        "default is 30 min and it measures the wall clock between the "
                        "FIRST rank entering a collective and the LAST rank arriving -- "
                        "so a cold h5ad page-in over NFS, a live-FM warm-up or a "
                        "40-block val on one straggler is charged against it and a "
                        "healthy run is killed as if it had hung. This is NOT a hang "
                        "detector: the two real hangs in this file (a one-rank buffer "
                        "broadcast and a one-rank all_gather) are fixed structurally at "
                        "the DDP ctor and at the val hook.")
    p.add_argument("--device", default=None)

    # -- smoke ----------------------------------------------------------------------- #
    p.add_argument("--smoke", action="store_true",
                   help="unlocks the CPU-affordable knobs below and shrinks the run")
    p.add_argument("--subsample_kg", type=int, default=0)
    p.add_argument("--subsample_kc", type=int, default=0,
                   help="SMOKE ONLY per-cell top-k token subsample. This is NOT FILIP "
                        "selection and it CHANGES what the refiner is shown; refused "
                        "unless --smoke.")
    return p


def resolve_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    a = build_parser().parse_args(argv)
    a.batch_size = a.micro_batch * a.accum_freq
    if a.arm == "finecls":
        assert a.prior_rna_path and a.prior_atac_path, \
            "--arm finecls needs --prior_rna_path and --prior_atac_path"
        # ⛔ `--arm finecls --fine_weight 0` is a SILENTLY INERT ARM.  The loss gates the
        # fine term on `fine_weight > 0`, but `arm_liveness` builds `fine_loss` ITSELF,
        # so every liveness line stays green (slot modules allocated, fine-only gradient
        # reaches both refiners and the slot queries, slot centring executes) while the
        # slot branch receives exactly zero gradient for the whole run and
        # `compare_arm_checkpoints` still reports the arms as differing, because the
        # slot-projection dropout consumes RNG that ARM A does not.  Verified by running
        # it: 20 steps, exit 0, `slot_gnorm 0.0000` at every step, all 42 slot tensors
        # bit-identical to their init.  Refuse it at parse time.
        assert a.fine_weight > 0, (
            "--arm finecls with --fine_weight <= 0 allocates the slot branch, pays for "
            "it, and never trains it. Every liveness assert still passes. Use "
            "--arm cell_only for the no-fine-branch arm.")
    # ---- LIVE-FM GATES.  Every combination in which --live_fm would be silently
    # inert or silently wrong is refused at PARSE time, not three GPU-hours in.
    if a.live_fm:
        assert a.rna_encoder_path and a.atac_encoder_path, (
            "--live_fm 1 needs --rna_encoder_path and --atac_encoder_path; today those "
            "two flags feed ONLY --dump_fm_layer_state, so forgetting them would leave "
            "the run with no FM at all")
        assert a.max_atac_length > 0, (
            "--live_fm 1 needs an explicit --max_atac_length: `collate_fn`'s "
            "fixed_atac_length has no 'pad to the batch max' mode (dataset.py:588-598)")
        assert a.subsample_kg == 0 and a.subsample_kc == 0, \
            "--subsample_* is a CACHE-ONLY knob and cannot reach the live path"
        assert a.train_shards == "all", \
            "--train_shards is a CACHE-ONLY selector and cannot reach the live path"
        assert a.dataset_id_column == "dataset_id", (
            "--live_fm derives the dataset id from the FILE (one dataset_id per h5ad, "
            "asserted at load), so any other --dataset_id_column is silently inert")
        if a.live_paired_dir:
            assert os.path.isdir(a.live_paired_dir), \
                f"--live_paired_dir {a.live_paired_dir!r} is not a directory"
        else:
            assert a.corpus_manifest and os.path.exists(a.corpus_manifest), (
                f"--live_fm 1 needs --corpus_manifest (got {a.corpus_manifest!r}) or "
                f"--live_paired_dir")
    else:
        assert not a.live_paired_dir, (
            "--live_paired_dir without --live_fm 1 is a SILENTLY INERT flag: the "
            "loader would still read --cache_root")
    # ---- EARLY-STOP GATES.  The rule is an AND over two criteria; with R@1 switched
    # off there is only one, and a rule that silently degrades to val_loss-only is
    # exactly the single-criterion selection the standing rule exists to forbid.
    assert not (a.early_stop_patience > 0 and a.val_r1_draws <= 0), (
        "--early_stop_patience needs --val_r1_draws > 0: the standing rule is that a "
        "run stops only when BOTH val_loss and val R@1 have stalled, and with R@1 off "
        "this would quietly become val_loss-only early stopping.")
    assert not (a.early_stop_patience > 0 and a.val_every <= 0), (
        "--early_stop_patience needs --val_every > 0: the counters only advance in the "
        "validation hook.")
    if a.early_stop_patience > 0:
        assert a.early_stop_min_delta_loss > 0 and a.early_stop_min_delta_r1 > 0, (
            "--early_stop_min_delta_* must be > 0. At exactly 0 a criterion counts pure "
            "noise as an improvement, its counter resets forever, and the AND-rule "
            "never fires -- an early stop that is inert is worse than none, because the "
            "log says it is armed.")
    assert a.val_r1_draws == 0 or a.val_r1_pool >= 2, "--val_r1_pool must be >= 2"
    # ⛔ THE SILENT-nan GATE.  val R@1 is scored inside ONE val block, whose per-rank width is
    # micro_batch*accum_freq.  If that window is narrower than --val_r1_pool, NO block can fill a
    # pool, `blocks` is 0, R@1 is nan forever, `best_by_valr1.pt` is never written, and -- worst --
    # `stall_r1` never increments, so the "BOTH val_loss AND val R@1 stalled" early-stop rule is
    # vacuously false and early stopping silently never fires.  Observed on smoke job 15285707
    # (window 32 < pool 128).  Refuse at parse time instead of discovering it 100k steps later.
    _win = a.micro_batch * a.accum_freq
    assert a.val_r1_draws == 0 or _win >= a.val_r1_pool, (
        f"--val_r1_pool {a.val_r1_pool} exceeds the per-rank val window "
        f"micro_batch*accum_freq = {a.micro_batch}*{a.accum_freq} = {_win}: every val block "
        f"would be unscorable, val R@1 would be nan forever, best_by_valr1.pt would never be "
        f"written, and --early_stop_patience would never fire because stall_r1 cannot advance. "
        f"Raise micro_batch/accum_freq, or lower --val_r1_pool (>=2), or set --val_r1_draws 0 "
        f"to disable R@1 on purpose (which also forbids --early_stop_patience > 0).")
    assert a.grad_cache or not a.distributed, (
        "--grad_cache 0 is a single-process control path (accum forwards, ONE "
        "backward), which breaks DDP's reducer. Use --grad_cache 1 when distributed.")
    assert a.fm_layer_lr > 0, (
        "--fm_layer_lr 0 is a BROKEN FREEZE in this repo: gradients flow, the "
        "parameter never moves, every gradient assert stays green and the arm is "
        "inert. Use 1e-12.")
    assert a.proj_dim == 256 or a.smoke, \
        "proj dim 256 is a standing decision (512 measurably hurts)"
    assert not (a.atac_varlen and a.atac_sdpa), (
        "--atac_varlen and --atac_sdpa are two implementations of the SAME block and "
        "are mutually exclusive. --atac_varlen 1 needs --atac_sdpa 0 passed "
        "EXPLICITLY: silently clearing an arm-defining flag is how this project got "
        "four inert flags in one day.")
    assert a.atac_sdpa or a.atac_varlen or a.smoke, (
        "one of --atac_sdpa / --atac_varlen is REQUIRED at max_atac_length 8192: "
        "without either, the [B,H,8192,8192] score matrix is materialised and stored "
        "for backward, and OOMs above micro_batch 16 on a 139.8 GiB H200.")
    assert a.rna_sub_batch >= 0, "--rna_sub_batch must be >= 0 (0 = off)"
    # RNA and ATAC arithmetic move TOGETHER or not at all: a mixed policy makes the
    # step's error budget unattributable, and the ATAC side's fp16 is a DEFECT, not a
    # choice, so "fix ATAC only" is a real option and "fix RNA only" is not.
    a.rna_autocast = {"shipped": "off", "atac_bf16": "off", "bf16": "bf16",
                      "fp16": "fp16", "fp32": "off"}[a.refiner_precision]
    a.atac_autocast = {"shipped": "fp16", "atac_bf16": "bf16", "bf16": "bf16",
                       "fp16": "fp16", "fp32": "off"}[a.refiner_precision]
    # `fp16` is the ONLY policy that needs a loss scale, and it needs one on BOTH sides:
    # the RNA refiner underflows in fp16 just as badly as the ATAC one does (measured
    # 3.89e-01 vs 5.35e-01 relative at S=1), which was never visible because no fp16
    # policy for RNA existed.  Without the scaler this flag makes the model strictly
    # WORSE than `shipped`, so it is derived here rather than left to a second flag that
    # could be forgotten.
    # ⛔ GATED ON THE POLICY NAME, NOT ON "is fp16 anywhere".  `shipped` is fp16 on the
    # ATAC side and is DELIBERATELY left unscaled: it is the name of the arithmetic every
    # previous run on this track executed, and it is the BEFORE arm every numerics table
    # here is measured against.  Silently fixing it would destroy the only reproducible
    # reference for the defect.  It is the wrong thing to TRAIN with -- which is what the
    # warning in `build_grad_scaler` says, loudly, every time it is selected.
    a.grad_scaler = (a.refiner_precision == "fp16")
    assert a.grad_scaler_init_scale >= 1.0, "--grad_scaler_init_scale must be >= 1"
    assert a.grad_scaler_min_scale >= 1.0, "--grad_scaler_min_scale must be >= 1"
    assert a.grad_scaler_init_scale >= a.grad_scaler_min_scale, (
        f"--grad_scaler_init_scale {a.grad_scaler_init_scale:g} starts BELOW the floor "
        f"{a.grad_scaler_min_scale:g}: the run would abort on its first step")
    # ⛔ NOT a preference: `flash_attn.modules.mha.FlashSelfAttention.forward` opens with
    # `assert qkv.dtype in [torch.float16, torch.bfloat16]`, so the committed
    # `--atac_varlen 1` CANNOT RUN IN FP32 AT ALL.  Refusing it here turns a crash at the
    # first forward -- after the FMs, the cache index and the loader are up -- into a
    # parse-time message.  It also means no all-fp32 reference exists on the production
    # attention path; every fp32-referenced number on this track is measured on sdpa.
    assert not (a.refiner_precision == "fp32" and a.atac_varlen), (
        "--refiner_precision fp32 is incompatible with --atac_varlen 1: flash_attn's "
        "FlashSelfAttention asserts qkv.dtype in {float16, bfloat16} and has no fp32 "
        "kernel. Use --atac_sdpa 1 --atac_varlen 0 for the all-fp32 reference.")
    if not a.smoke:
        assert a.max_atac_length == 8192, \
            "the cache is built at 8192; a narrower or batch-dependent ATAC width " \
            "silently reintroduces the truncation deficit the cache exists to remove " \
            "(and on --live_fm 1 `collate_fn` would TRUNCATE SILENTLY rather than " \
            "refuse, unlike `cached_collate`)"
        assert a.subsample_kg == 0 and a.subsample_kc == 0, \
            "--subsample_* is SMOKE ONLY: it changes what the refiner is shown"
        assert not a.skip_vocab_gate, "--skip_vocab_gate is SMOKE ONLY"
    return a


# ------------------------------------------------------------------------------------ #
# Setup helpers
# ------------------------------------------------------------------------------------ #

#: A CPU-side process group whose ONLY job is `monitored_barrier`, which NCCL cannot do.
#: It turns "one rank did not reach a collective" from a silent multi-hour hang into an
#: exception that NAMES the missing ranks. Best-effort: None when gloo is unavailable.
_MONITOR_PG = None


def setup_distributed(args) -> Tuple[int, int, torch.device]:
    global _MONITOR_PG
    if args.distributed and not dist.is_initialized():
        # ⛔ THE DEFAULT 30-MINUTE TIMEOUT IS A STRAGGLER KILLER, NOT A HANG DETECTOR.
        # It measures the wall clock from the first rank entering a collective to the
        # last one arriving, so anything that delays ONE rank -- a live-FM warm-up, a
        # cold h5ad page-in from the ~897 GB paired tree over NFS, a 40-block val, a
        # 286 MB snapshot write on a 99%-full filesystem -- is charged against it and
        # aborts a healthy run.  Set it explicitly and RECORD it, because a timeout
        # inherited from a default is a number nobody can defend after the fact.
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(minutes=int(args.ddp_timeout_min)))
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    if dist.is_initialized() and world > 1 and _MONITOR_PG is None:
        try:
            _MONITOR_PG = dist.new_group(backend="gloo")
        except Exception as e:                                   # pragma: no cover
            _MONITOR_PG = None
            if rank == 0:
                print(f"  [ddp] WARNING no gloo monitor group ({e}); the collective "
                      f"tripwire in compute_val_loss is DISABLED", flush=True)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
    else:
        device = torch.device("cpu")
    return rank, world, device


def rank_of_default():
    return dist.get_rank() if dist.is_initialized() else 0


def _assert_buffers_identical_across_ranks(model, device, world, liveness, log) -> None:
    """PROOF that `broadcast_buffers=False` is lossless for THIS model.

    DDP's buffer sync runs off `module.buffers()`, which includes `persistent=False`
    ones, so the finecls arm's `rna_prior_mass` / `atac_prior_mass` and FixedSlotPooler's
    `prior` WERE being broadcast at the top of every forward.  Turning that off is only
    legal because all of them are pure functions of the prior .npz files every rank loads
    identically (`load_priors`) and NOTHING updates them at train time -- this model
    contains no BatchNorm and no running statistic.

    That last sentence is exactly the kind of claim this project has been wrong about, so
    it is CHECKED, once, against the other ranks: each buffer gets a position-weighted
    float64 checksum, the checksums are all_gathered, and a disagreement kills the run
    naming the buffer -- rather than silently freezing two ranks onto different priors
    for tens of thousands of steps.  Cost: one all_gather of `n_buffers` float64s.
    ⚠️ It catches a STARTUP disagreement, not one that develops during training; adding
    any running statistic to this model later would need this revisited.
    """
    mod = model_of(model)
    named = list(mod.named_buffers())
    # ⛔ PRINT BEFORE THE EARLY RETURN.  Putting it after the guard hid rank 0 entirely
    # (job 15285716: only rank 1 spoke), which made the mismatch unreadable.
    _r = dist.get_rank() if dist.is_initialized() else 0
    print(f"  [ddp-buffers rank{_r}] type={type(mod).__name__} n_named={len(named)} "
          f"names={[n for n, _ in named]}", flush=True)
    for n, b in named:
        print(f"  [ddp-buffers rank{_r}] {n}: shape={tuple(b.shape)} dtype={b.dtype} "
              f"dev={b.device} sum={float(b.detach().double().sum()):.6e} "
              f"min={float(b.detach().double().min()):.3e} "
              f"max={float(b.detach().double().max()):.3e}", flush=True)
    liveness.set("ddp_broadcast_buffers", False)
    liveness.set("ddp_buffers_checked", len(named))
    if not named or world <= 1 or not dist.is_initialized():
        return
    sums = []
    for _, b in named:
        if b is None or b.numel() == 0:
            sums.append(0.0)
            continue
        d = b.detach().to(device=device, dtype=torch.float64).reshape(-1)
        w = torch.arange(1, d.numel() + 1, device=device, dtype=torch.float64)
        sums.append(float((d * w).sum()))
    # ⛔ LOCAL-ONLY diagnostic: every rank prints its OWN buffer table to stdout.  No
    # collective -- an extra collective here desynchronises the ranks' NCCL sequence and
    # hangs the job (observed: rank0 SeqNum=1, rank1 SeqNum=2, job 15285711).
    # ⛔ SYNCHRONISE FIRST.  Without a barrier the ranks reach this all_gather at wildly
    # different times -- rank 0 is still reading the 347 MB `atac_pooler.prior` off NFS while
    # rank 1 is already here -- and a bare all_gather then matches against whatever collective
    # the other rank is in, returning GARBAGE rather than blocking.  Observed as
    # `rna_prior_mass DIFFERS (checksums [-2.14e-13, 218941])` where -2.14e-13 is not even a
    # possible value for this all-positive buffer (jobs 15285710/15285716/15285717).
    dist.barrier()
    # ⛔ FIXED WIDTH.  all_gather with different-length tensors per rank is UNDEFINED
    # BEHAVIOUR (garbage, not an error).  Pad to a constant so a buffer-count disagreement
    # shows up as a count mismatch we can name, not as corrupt checksums.
    _W = 64
    assert len(sums) <= _W, f"{len(sums)} buffers exceeds the fixed checksum width {_W}"
    _pad = [float(len(sums))] + list(sums) + [0.0] * (_W - len(sums) - 1)
    sig = torch.tensor(_pad, dtype=torch.float64, device=device)
    buf = [torch.zeros_like(sig) for _ in range(world)]
    dist.all_gather(buf, sig.contiguous())
    _counts = [int(g[0]) for g in buf]
    assert max(_counts) == min(_counts), (
        f"ranks registered DIFFERENT numbers of buffers {_counts} -- the models are not the "
        f"same module tree, which no amount of buffer broadcasting would fix")
    buf = [g[1:] for g in buf]
    for i, (n, _) in enumerate(named):
        vals = [float(g[i]) for g in buf]
        assert max(vals) == min(vals), (
            f"buffer {n!r} DIFFERS across ranks (checksums {vals}). With "
            f"broadcast_buffers=False that disagreement is never repaired and every "
            f"rank would train against a different prior.")
    liveness.set("ddp_buffer_checksum_rank0",
                 ",".join(f"{v:.6e}" for v in sums[:4]))
    log(f"  [ddp] broadcast_buffers=False | {len(named)} buffers verified BIT-EQUAL "
        f"across {world} ranks: {[n for n, _ in named]}")


def load_priors(args, liveness: Liveness) -> Optional[Dict]:
    """Load + gate the priors, and record the ONE identity separating B from C."""
    if args.arm != "finecls":
        liveness.set("prior_rna_content_sha256", "n/a (arm=cell_only)")
        liveness.set("prior_atac_content_sha256", "n/a (arm=cell_only)")
        return None
    _, _, read_genes, read_ccre = _mixin_pieces()
    gene_symbols = ccre_ids = None
    if not args.skip_vocab_gate:
        # L11: the prior's row order must BE the runtime vocabulary order. A silent
        # off-by-one here trains the fine branch on the wrong modules and looks healthy.
        gene_symbols = read_genes(args.gene_list)
        ccre_ids = read_ccre(args.ccre_bed)
        assert len(gene_symbols) == N_GENES, f"gene list has {len(gene_symbols)} rows"
        assert len(ccre_ids) == N_CCRE, f"cCRE bed has {len(ccre_ids)} rows"
    rna = load_fixed_prior_npz(args.prior_rna_path, matrix_name=args.prior_rna_matrix,
                               expected_rows=N_GENES, expected_num_slots=args.num_slots,
                               expected_feature_symbols=gene_symbols)
    atac = load_fixed_prior_npz(args.prior_atac_path,
                                matrix_name=args.prior_atac_matrix,
                                expected_rows=N_CCRE,
                                expected_num_slots=args.num_slots,
                                expected_feature_ids=ccre_ids)
    assert np.array_equal(rna.slot_ids, atac.slot_ids), "prior slot id order differs"
    assert np.array_equal(rna.slot_names, atac.slot_names), \
        "prior slot name order differs"
    liveness.set("prior_rna_content_sha256", sha256_csr(args.prior_rna_path,
                                                        args.prior_rna_matrix))
    liveness.set("prior_atac_content_sha256", sha256_csr(args.prior_atac_path,
                                                         args.prior_atac_matrix))
    liveness.set("prior_rna_nnz", int((rna.matrix != 0).sum()))
    liveness.set("prior_atac_nnz", int((atac.matrix != 0).sum()))
    names_blob = "\n".join(rna.slot_names.tolist()).encode()
    liveness.set("slot_names_sha256", hashlib.sha256(names_blob).hexdigest()[:16])
    liveness.set("num_slots_loaded", int(rna.matrix.shape[1]))
    assert int(rna.matrix.shape[1]) == args.num_slots
    return {"rna": rna, "atac": atac}


# ------------------------------------------------------------------------------------ #
# LIVE-FM DATA PATH  (--live_fm 1)
#
# The cached-token path replays a pre-computed `fmtok-v1` cache.  This path instead
# reads the RAW paired h5ads and runs the FROZEN scFoundation / EpiAgent forwards in
# this process, so the corpus is not limited to what has been cached.
#
# VERIFIED ON DISK 2026-09-04 (h5py, this session):
#   456 *_rna_paired.h5ad + *_atac_paired.h5ad pairs over THREE directories --
#     163 base  preprocessed_data_expanded_20260305/   1,046,380 cells
#     255 new   preprocessed_data_3m/paired/           3,254,285 cells
#      38 new   preprocessed_data_newtissue/paired/      359,071 cells
#     = 4,659,736 barcodes / ~897 GB, enumerated in `corpus_livefm_456.tsv`.
#   RNA X: DENSE float64 [n, 19264], ALREADY log1p-normalised (max ~5.04, non-integer)
#   ATAC obs/cell_sentences: PLAIN vlen bytes holding a JSON list (NOT categorical)
#   obs/dataset_id: categorical, EXACTLY ONE dataset_id per file
#   RNA and ATAC obs/_index equal ROW FOR ROW -> pairing is POSITIONAL (asserted below)
#   obs/n_genes is NOT the 19264-panel nnz (measured corr 0.454 on a 500-cell sample) --
#     see LIVE_RNA_NNZ_FILTER_NOTE.
# ------------------------------------------------------------------------------------ #
import ast                                                            # noqa: E402
import glob as _glob                                                  # noqa: E402
from torch.utils.data import Dataset                                  # noqa: E402

LIVE_TARGET_RESOLUTION = 4.0
LIVE_N_GENES = 19264
LIVE_RNA_IN_DIM = 19266
LIVE_ATAC_CLS, LIVE_ATAC_SEP = 1, 2

#: ⛔ THE RNA nnz FILTER (100..8500, multiomics_clip/dataset.py:257) IS **NOT** APPLIED
#: HERE, AND THAT IS A DELIBERATE, RECORDED DIVERGENCE.  Reproducing it requires a
#: per-cell non-zero count over the whole X -- ~897 GB of dense float64 read at startup,
#: on every rank -- and `obs/n_genes` is NOT a usable proxy for it (it is the
#: pre-alignment gene count; measured corr 0.454 against the true 19264-panel nnz on a
#: 500-cell sample, so a filter on it would drop the WRONG cells).  The projector track
#: reports 4,418,109 cells after the filter against 4,452,079 before it: a 0.76%
#: difference.  This run therefore trains on the UNFILTERED whitelisted set, the fact is
#: written into the manifest and liveness as `live_rna_nnz_filter`, and any
#: cell-count-matched comparison with the projector track must subtract it explicitly
#: rather than assume the corpora are identical.
LIVE_RNA_NNZ_FILTER_NOTE = ("off: the 100..8500 nnz filter needs a full-X scan (~897 GB) "
                            "and obs/n_genes is not a proxy (corr 0.454); corpus is the "
                            "raw whitelisted set, ~0.76% larger than the projector's")

#: Totals verified with h5py this session; asserted against `--corpus_manifest` so a
#: stale manifest aborts at parse time and not four GPU-hours in.
LIVE_CORPUS_TOTALS = {"pairs": 456, "base_pairs": 163, "new_pairs": 293,
                      "raw_cells": 4659736, "base_cells": 1046380,
                      "val_cells": 103775, "test_cells": 103882,
                      "train_wl_cells": 830567,
                      "raw_train_cells": 4452079, "filtered_train_cells": 4418109}


def resolve_live_corpus(args, split: str, verbose: bool = False) -> List[Dict]:
    """`--corpus_manifest` -> the list of h5ad units this split reads, each with its
    OWN whitelist (a `set`, or `None` meaning "take every cell in this file").

    ⛔ THE WHITELIST IS PER UNIT, NOT GLOBAL.  The 830,567-barcode train list covers the
    163 BASE units only (the split is CELL-level and lives entirely inside them); the
    293 new units are 100% train.  Handing the list to a new unit would filter that
    whole 3.6M-cell cohort to ZERO cells; NOT handing it to a base unit leaks the
    103,775 val + 103,882 test cells into train.  Both failures are silent.

    ⛔ A GLOB OVER THE BASE DIRECTORY IS NOT SAFE: it also holds `*_preprocessed.h5ad`
    (the pre-pairing stage) and a STALE `all_{rna,atac}_paired.h5ad` of 1,055,013 cells
    -- a DIFFERENT unit set from the 163 the split was built on.  The file list is data,
    in a manifest.
    """
    import csv as _csv
    man = args.corpus_manifest
    assert man and os.path.exists(man), f"--corpus_manifest {man!r} does not exist"
    if split == args.train_split:
        want = {c.strip() for c in str(args.corpus_cohorts).split(",") if c.strip()}
        wl_path, wl_exp = args.train_whitelist, LIVE_CORPUS_TOTALS["train_wl_cells"]
    else:
        # No val cell exists outside the base cohort, so a val loader never opens `new`.
        want, wl_path, wl_exp = {"base"}, args.val_whitelist, LIVE_CORPUS_TOTALS["val_cells"]
    assert want <= {"base", "new"}, f"--corpus_cohorts {args.corpus_cohorts!r}"
    rows = []
    with open(man) as f:
        for r in _csv.DictReader(f, delimiter="\t"):
            r["n_cells"] = int(r["n_cells"])
            rows.append(r)
    n_base = sum(1 for r in rows if r["cohort"] == "base")
    assert (len(rows) == LIVE_CORPUS_TOTALS["pairs"]
            and n_base == LIVE_CORPUS_TOTALS["base_pairs"]), (
        f"{man}: {len(rows)} rows / {n_base} base -- expected "
        f"{LIVE_CORPUS_TOTALS['pairs']} / {LIVE_CORPUS_TOTALS['base_pairs']}")
    assert sum(r["n_cells"] for r in rows) == LIVE_CORPUS_TOTALS["raw_cells"], (
        f"{man}: n_cells sums to {sum(r['n_cells'] for r in rows)}, expected "
        f"{LIVE_CORPUS_TOTALS['raw_cells']} -- the manifest is STALE, rebuild it")
    wl = None
    if wl_path:
        wl = {l.strip() for l in open(wl_path) if l.strip()}
        assert len(wl) == wl_exp, f"{wl_path}: {len(wl)} barcodes, expected {wl_exp}"
    if wl is None and "base" in want:
        raise AssertionError(
            "the BASE cohort is selected with an EMPTY whitelist: all 103,775 val and "
            "103,882 test cells live inside those 163 files and would enter train. "
            "Pass --train_whitelist, or --corpus_cohorts new.")
    units = []
    for r in rows:
        if r["cohort"] not in want:
            continue
        for k in ("rna_h5ad", "atac_h5ad"):
            assert os.path.exists(r[k]), f"{man}: missing {r[k]}"
        units.append({"unit": r["unit"], "cohort": r["cohort"],
                      "rna_h5ad": r["rna_h5ad"], "atac_h5ad": r["atac_h5ad"],
                      "n_cells": r["n_cells"],
                      "whitelist": wl if r["cohort"] == "base" else None})
    assert units, f"{man} selected 0 units for cohorts {sorted(want)}"
    if verbose:
        print(f"live corpus [{split}]: {len(units)} h5ad pairs / "
              f"{sum(u['n_cells'] for u in units):,} raw cells | cohorts "
              f"{sorted(want)} | whitelist {0 if wl is None else len(wl):,} barcodes "
              f"({os.path.basename(wl_path) if wl_path else 'none'}) applied to the "
              f"{sum(1 for u in units if u['whitelist'] is not None)} base units only",
              flush=True)
    return units


def live_units_from_dir(paired_dir: str) -> List[Dict]:
    """`--live_paired_dir` -> units with NO whitelist (the crc32 holdout is used instead).

    The single-directory escape hatch, for a smoke run or a cohort that has no manifest
    yet.  ⛔ It cannot be pointed at `preprocessed_data_expanded_20260305`: that
    directory holds the stale `all_*_paired.h5ad` and the pre-pairing `*_preprocessed`
    files, and a whole-directory glob there is a different corpus from the manifest's.
    """
    rna_paths = sorted(_glob.glob(os.path.join(paired_dir, "*_rna_paired.h5ad")))
    assert rna_paths, f"no *_rna_paired.h5ad under {paired_dir}"
    units = []
    for rp in rna_paths:
        ap = rp.replace("_rna_paired.h5ad", "_atac_paired.h5ad")
        assert os.path.exists(ap), f"{rp} has no _atac_paired sibling"
        units.append({"unit": os.path.basename(rp)[:-len("_rna_paired.h5ad")],
                      "cohort": "dir", "rna_h5ad": rp, "atac_h5ad": ap,
                      "n_cells": -1, "whitelist": None})
    return units


class PairedH5adCorpus(Dataset):
    """The paired h5ad UNITS as ONE lazy dataset, emitting `cached_collate`'s inputs.

    ⛔ WHY NOT `ConcatDataset` OF N `PairedMultiOmicsDataset`.  Three reasons, each
    fatal on its own.  (1) FORK SAFETY: that constructor keeps a live backed AnnData --
    an OPEN h5py handle -- built in the parent (multiomics_clip/dataset.py:277-283),
    and `num_workers > 0` forks it into every worker; 456 files x 4 workers of shared
    HDF5 handles is the classic silent-corruption / hang.  Here NOTHING is left open by
    `__init__` and handles are opened lazily PER PID, exactly as
    `CachedTokenDataset._reader` does (cached_token_dataset.py:414-437).  (2) MEMORY:
    its non-`skip_rna_filter` path chunk-reads the WHOLE X to count non-zeros
    (dataset.py:236-249) -- ~897 GB here -- and its ATAC obs would pull `cell_sentences`
    into RAM unless wrapped in `LazyATACData`.  (3) ALIGNMENT: its `self.cell_labels` is
    indexed off `cell_barcodes`, which the class's own `get_labels` docstring
    (dataset.py:420-431) says is misaligned in backed mode.

    ⛔ AND WHY THE WHITELIST CANNOT GO THROUGH THAT CLASS AT ALL: a plain backed-AnnData
    view has no `_h5_row_indices`, so the whitelist view->original-row remap gives every
    cell ANOTHER cell's ATAC (rna_cos 1.0 / atac_cos 0.89, measured).  This class never
    builds a view: it resolves the whitelist to ORIGINAL h5 row numbers here, at load,
    and `__getitem__` reads that row directly from both files.

    ✅ WHAT IS COPIED VERBATIM: `__getitem__`'s arithmetic is
    `PairedMultiOmicsDataset.__getitem__` (dataset.py:453-548) term for term --
    `expm1(x).sum()`, `log10(max(tc, 1e-6) + 1e-6)`, the `[x, resolution, log10_total]`
    concat to 19266, `arange(19266)` position ids, the head-of-TF-IDF truncation to
    `max_atac_length - 2` with `random_truncate=False`, and `[1] + sentence + [2]`.
    """

    def __init__(self, units, split, want_val=None, holdout_mod=100, holdout_rem=0,
                 max_atac_length=8192, labels_csv=None,
                 label_barcode_column="cell_barcode",
                 label_celltype_column="cell_type", allow_unlabeled=False,
                 open_files=8, verify_pairing=True,
                 target_resolution=LIVE_TARGET_RESOLUTION, verbose=False):
        import h5py
        assert units, "PairedH5adCorpus needs at least one unit"
        assert max_atac_length >= 3, "max_atac_length must leave room for CLS+SEP"
        self.split = split
        self.max_atac_length = int(max_atac_length)
        self.target_resolution = float(target_resolution)
        self.open_files = max(1, int(open_files))
        self.holdout_mod = int(holdout_mod)
        self.holdout_rem = int(holdout_rem)
        self.want_val = bool(want_val)
        self.label_coverage = 0.0
        self.n_classes = 0
        self.n_pairing_checked = 0
        self.files, fi_l, row_l, bc_l = [], [], [], []
        for u in units:
            rp, ap = u["rna_h5ad"], u["atac_h5ad"]
            # ⛔ EVERY handle opened here is CLOSED before __init__ returns.  A handle
            # that survives into the DataLoader's fork is the bug this class avoids.
            with h5py.File(rp, "r") as fr, h5py.File(ap, "r") as fa:
                n_r = int(fr["X"].shape[0])
                assert int(fr["X"].shape[1]) == LIVE_N_GENES, (
                    f"{rp}: RNA must be exactly {LIVE_N_GENES} scFoundation genes, "
                    f"got {fr['X'].shape[1]}")
                rbc = fr["obs/_index"][:].astype(str)
                assert len(rbc) == n_r, f"{rp}: obs/_index {len(rbc)} vs X {n_r}"
                assert "cell_sentences" in fa["obs"], f"{ap}: obs has no cell_sentences"
                assert int(fa["obs/cell_sentences"].shape[0]) == n_r, (
                    f"{ap}: {fa['obs/cell_sentences'].shape[0]} sentences vs {n_r} "
                    f"RNA rows -- the pair is not row-aligned")
                if verify_pairing:
                    # THE BARCODE-SCRAMBLE BACKSTOP.  A silent RNA<->ATAC row shift gives
                    # every cell another cell's chromatin and is INVISIBLE in the loss
                    # (measured rna_cos 1.0 / atac_cos 0.89 the last time it happened).
                    # Pairing here is POSITIONAL, so this is the assumption that has to
                    # be paid for, once, at startup.
                    abc = fa["obs/_index"][:].astype(str)
                    assert np.array_equal(rbc, abc), (
                        f"{os.path.basename(rp)}: RNA and ATAC obs/_index differ -- "
                        f"pairing is POSITIONAL here and this file breaks it")
                    self.n_pairing_checked += 1
                dg = fr["obs/dataset_id"]
                if isinstance(dg, h5py.Group):
                    names = np.unique(dg["categories"][:].astype(str)[dg["codes"][:]])
                else:
                    names = np.unique(dg[:].astype(str))
                assert len(names) == 1, (
                    f"{rp}: {len(names)} dataset_ids in one file; the blocked sampler's "
                    f"'one block == one dataset' contract assumes exactly one")
                dname = str(names[0])
            wl = u.get("whitelist")
            if wl is not None:
                keep = np.fromiter((b in wl for b in rbc), dtype=bool, count=n_r)
            elif want_val is None:
                keep = np.ones(n_r, dtype=bool)
            else:
                # DETERMINISTIC PER-CELL SPLIT for the --live_paired_dir escape hatch.
                # ⛔ crc32, never `hash()`: python's str hash is PYTHONHASHSEED-salted,
                # so `hash()` would put a DIFFERENT set of cells in val on every rank
                # and every restart -- an unpaired val curve AND a train set that leaks
                # val.  Per-CELL and not per-FILE because the blocked sampler draws a
                # DATASET first and then cells inside it: a whole-file holdout would
                # leave train unable to draw those datasets at all.
                keep = np.fromiter(
                    ((zlib.crc32(f"{dname}|{b}".encode()) % self.holdout_mod
                      == self.holdout_rem) == self.want_val for b in rbc),
                    dtype=bool, count=n_r)
            k = np.where(keep)[0]
            if k.size == 0:
                continue
            self.files.append((rp, ap, dname))
            fi_l.append(np.full(k.size, len(self.files) - 1, dtype=np.int32))
            row_l.append(k.astype(np.int64))
            bc_l.append(rbc[k])
        assert self.files, f"the {split!r} selection kept 0 cells"
        self._fi = np.concatenate(fi_l)
        self._row = np.concatenate(row_l)
        self.dataset_names = [d for _, _, d in self.files]
        self.n = int(self._fi.size)
        barcodes = np.concatenate(bc_l)
        self.labels = self._resolve_labels(
            labels_csv, barcodes, label_barcode_column, label_celltype_column,
            bool(allow_unlabeled))
        # ~300 MB of unicode for 4.66M cells, forked into every worker.  The only
        # downstream consumer is the val-R@1 dedupe, and what THAT needs is a unique
        # CELL IDENTITY, not the barcode text -- so `__getitem__` emits "<file>:<row>",
        # which is exactly as unique (one dataset_id per file, one row per cell) and
        # costs nothing to carry.
        del barcodes, bc_l
        self._open = {}
        self._pid = -1
        if verbose:
            print(f"[PairedH5adCorpus] split={split} files={len(self.files)} "
                  f"cells={self.n:,} | pairing verified on "
                  f"{self.n_pairing_checked}/{len(self.files)} pairs | "
                  f"max_atac_length={self.max_atac_length} "
                  f"label_coverage={self.label_coverage:.3f} "
                  f"classes={self.n_classes} | rna_nnz_filter={LIVE_RNA_NNZ_FILTER_NOTE}",
                  flush=True)

    # -- labels (CachedTokenDataset._resolve_labels' recipe) ------------------------- #

    def _resolve_labels(self, labels_csv, barcodes, bc_col, ct_col, allow_unlabeled):
        if labels_csv is None:
            assert allow_unlabeled, (
                "--live_fm with no --labels_csv: every label-conditioned term "
                "(supcon_xmodal at weight 0.5, and the per-slot SupCon half of the "
                "fine loss) would be SILENTLY INERT.  Pass --labels_csv, or "
                "--allow_unlabeled 1 on purpose.")
            return np.full(self.n, -1, dtype=np.int64)
        import csv as _csv
        d, cts = {}, set()
        with open(labels_csv, newline="") as f:
            rd = _csv.reader(f)
            hdr = next(rd)
            assert bc_col in hdr and ct_col in hdr, \
                f"{labels_csv}: need columns {bc_col!r} and {ct_col!r}, got {hdr}"
            ib, ic = hdr.index(bc_col), hdr.index(ct_col)
            for row in rd:
                if not row:
                    continue
                d[row[ib]] = row[ic]
                cts.add(row[ic])
        # ⛔ THE CLASS TABLE IS BUILT FROM THE CSV, SORTED -- so every rank, every
        # resume and every arm maps the same cell type to the same integer.  A dict
        # insertion order would be file-order-dependent and would silently re-label the
        # corpus on a rebuilt csv.
        ct2id = {c: i for i, c in enumerate(sorted(cts))}
        # ⛔ ANY barcode absent from the csv falls to -1 = UNLABELLED, which the loss
        # ignores.  That is the whole point of shipping a csv with the pseudo-labels
        # OMITTED rather than one that spells them "Unknown": `build_label_dict` does
        # NOT gate "Unknown" to -1, and 26% of this corpus carrying that string would
        # make 1.15M cells ONE giant supcon positive class.
        lab = np.fromiter((ct2id.get(d.get(b, None), -1) for b in barcodes),
                          dtype=np.int64, count=self.n)
        cov = float((lab >= 0).mean()) if self.n else 0.0
        assert cov > 0.0 or allow_unlabeled, (
            f"{labels_csv}: 0/{self.n} corpus barcodes carry a label -- barcode "
            f"namespace mismatch. corpus e.g. {barcodes[0]!r}; csv e.g. "
            f"{next(iter(d))!r}")
        self.label_coverage = cov
        self.n_classes = len(ct2id)
        return lab

    # -- per-worker h5py handles ----------------------------------------------------- #

    def _h(self, fi):
        """(rna, atac) h5py handles for file `fi`, opened in THIS process.

        Keyed on the PID so a handle is never used across a fork, and capped at
        `open_files` with FIFO eviction: the blocked sampler draws ONE dataset per
        optimizer step and there is exactly one dataset per file, so a cap of 8 evicts
        approximately never while 456 x num_workers open fds is avoided.
        """
        import h5py
        pid = os.getpid()
        if pid != self._pid:
            self._open = {}
            self._pid = pid
        h = self._open.get(fi)
        if h is None:
            while len(self._open) >= self.open_files:
                old = self._open.pop(next(iter(self._open)))
                old[0].close()
                old[1].close()
            rp, ap, _ = self.files[fi]
            h = (h5py.File(rp, "r"), h5py.File(ap, "r"))
            self._open[fi] = h
        return h

    def close(self):
        for h in self._open.values():
            h[0].close()
            h[1].close()
        self._open = {}
        self._pid = -1

    def __getstate__(self):
        # Whatever the start method, a pickled dataset must not carry live handles.
        st = dict(self.__dict__)
        st["_open"] = {}
        st["_pid"] = -1
        return st

    # -- access ---------------------------------------------------------------------- #

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        """`PairedMultiOmicsDataset.__getitem__` (dataset.py:453-548), arithmetic-identical.

        Returns a SIX-tuple: the reference 4-tuple, then the PRE-truncation sentence
        length (which `live_collate` turns into the truncation liveness counter) and the
        cell identity `"<file>:<row>"`.  `collate_fn` itself only ever sees the first four.
        """
        i = int(i)
        fi = int(self._fi[i])
        fr, fa = self._h(fi)
        r = int(self._row[i])
        x = np.asarray(fr["X"][r:r + 1]).reshape(-1).astype(np.float32)
        tc = float(np.expm1(x).sum())          # total count BEFORE log1p
        rna = np.empty(LIVE_RNA_IN_DIM, dtype=np.float32)
        rna[:LIVE_N_GENES] = x
        rna[LIVE_N_GENES] = self.target_resolution
        rna[LIVE_N_GENES + 1] = float(np.log10(max(tc, 1e-6) + 1e-6))
        raw = fa["obs/cell_sentences"][r]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            sent = json.loads(raw)
        except json.JSONDecodeError:
            sent = ast.literal_eval(raw)
        raw_len = len(sent)
        cap = self.max_atac_length - 2
        if raw_len > cap:
            # HEAD of the TF-IDF-DESCENDING sentence, i.e. random_truncate=False -- the
            # same rule the token cache was built under.  ~36% of cells hit this at
            # 8190; the count is reported by LiveFMPipe, never silent.
            sent = sent[:cap]
        ids = torch.as_tensor([LIVE_ATAC_CLS] + list(sent) + [LIVE_ATAC_SEP],
                              dtype=torch.long)
        return (torch.from_numpy(rna),
                torch.arange(LIVE_RNA_IN_DIM, dtype=torch.long),
                ids,
                torch.tensor(int(self.labels[i]), dtype=torch.long),
                raw_len,
                f"{fi}:{r}")

    # -- the CachedTokenDataset API the trainer already calls ------------------------ #

    def get_dataset_ids(self, column="dataset_id"):
        """Per-cell dataset code, ALIGNED to `__getitem__` order.

        Returned as an INT array, not strings: `SameDatasetBlockedBatchSampler` only
        does `np.unique(ids)` and `ids == d` (blocked_sampler.py:61-63), and a
        4.66M-element unicode array would cost ~300 MB for nothing.  One dataset_id per
        file is ASSERTED at load, so file index IS dataset identity.
        """
        assert column == "dataset_id", (
            f"--dataset_id_column {column!r}: the live corpus derives the id from the "
            f"FILE (exactly one dataset_id per h5ad, asserted at load), so any other "
            f"column would be a SILENTLY INERT flag")
        return self._fi

    def get_labels(self):
        return self.labels.copy()

    @property
    def spec(self):                                          # pragma: no cover
        raise AttributeError("the live corpus has no CacheSpec (--live_fm 1)")

    @property
    def manifest(self):
        """`write_manifest` reads `ds.manifest` (train_finecls_refiner.py:1317)."""
        return {"splits": {self.split: {"completeness": {
            "status": f"live_fm (no cache); rna_nnz_filter {LIVE_RNA_NNZ_FILTER_NOTE}",
            "rna": {"complete": True, "expected": len(self.files)},
            "atac": {"complete": True, "expected": len(self.files)}}}}}


def live_collate(batch, fixed_atac_length):
    """The reference `collate_fn` (multiomics_clip/dataset.py:555), re-keyed as a DICT.

    A DICT and not the reference 4-tuple because everything downstream of the loader in
    this trainer is dict-shaped: `move_batch_to_device` iterates `batch.items()`
    (cached_token_dataset.py:762-777) and `forward_fn` reads `mb["labels"]` (:1408).
    """
    from multiomics_clip.dataset import collate_fn as _paired_collate
    raw_len = torch.as_tensor([int(b[4]) for b in batch], dtype=torch.long)
    bcs = [b[5] for b in batch]
    rd, gi, ai, labels = _paired_collate([tuple(b[:4]) for b in batch],
                                         fixed_atac_length=int(fixed_atac_length))
    return {"rd": rd, "gi": gi, "ai": ai, "labels": labels,
            "atac_len_raw": raw_len, "barcode": bcs}


def build_live_fm(args, device):
    """A FROZEN, eval-mode MultiOmicsCLIP_FineLIP -- the FM pair and nothing else.

    `train_filip_combined.py:247-273` verbatim, minus `make_refiners`: this trainer owns
    its refiners (`--init_from_fm_state`), and building the colleague's too would
    allocate two unused modules AND consume RNG the cached arms never consumed.
    """
    from multiomics_clip_finelip import MultiOmicsCLIP_FineLIP, FineLIPConfig
    assert args.rna_encoder_path and args.atac_encoder_path, \
        "--live_fm 1 needs --rna_encoder_path and --atac_encoder_path"
    cfg = FineLIPConfig(rna_encoder_path=args.rna_encoder_path,
                        atac_encoder_path=args.atac_encoder_path,
                        projection_dim=args.proj_dim, temperature=0.07,
                        freeze_rna_encoder=True, freeze_atac_encoder=True,
                        token_dim=args.proj_dim,
                        max_atac_length=args.max_atac_length,
                        target_resolution=LIVE_TARGET_RESOLUTION)
    fm = MultiOmicsCLIP_FineLIP(cfg).to(device)
    fm.rna_encoder.eval()
    fm.atac_encoder.eval()
    for p in fm.rna_encoder.parameters():
        p.requires_grad_(False)
    for p in fm.atac_encoder.parameters():
        p.requires_grad_(False)
    if int(getattr(args, "fm_sdpa", 1)):
        # train_filip_combined.py:257-273, verbatim.  MATHEMATICALLY IDENTICAL --
        # dropout p=0 is a no-op and these layers have no other train/eval-dependent op
        # -- but it routes the 12 scFoundation layers through
        # F.multi_head_attention_forward -> SDPA, O(B*N*D) instead of the
        # BetterTransformer path's DENSE [B, 12, N, N] fp32 score matrix.  N reaches
        # ~9.5k on this corpus, so this is the largest FM speed lever that changes no
        # arithmetic -- and without it micro_batch 32 OOMs a 140 GiB H200 outright.
        n_l = 0
        for _l in fm.rna_encoder.model.encoder.transformer_encoder:
            for _a in ("dropout", "dropout1", "dropout2"):
                if hasattr(_l, _a):
                    getattr(_l, _a).p = 0.0
            if hasattr(_l.self_attn, "dropout"):
                _l.self_attn.dropout = 0.0
            _l.train()
            n_l += 1
        assert n_l > 0, "--fm_sdpa found no scFoundation encoder layers to switch"
        print(f"  [fm_sdpa] {n_l} scFoundation layers -> train()+dropout0 (SDPA path)",
              flush=True)
    # LIVENESS, not documentation: a live FM whose parameters still require grad would
    # be trained by whatever optimizer later saw them, silently -- and would also put
    # the frozen forward into the GradScaler's graph, which the 2^20 init scale was
    # never characterised on.
    # ⛔ freeze_{rna,atac}_encoder does NOT cover the FineLIP heads that
    # MultiOmicsCLIP_FineLIP also builds -- rna_projection / atac_projection /
    # log_temperature / cross_net (model.py:105-131), 35 parameters.  Without this the
    # assert below fires on every --live_fm 1 start (reproduced, job 15285693).  They are
    # never touched by fm_tokens(), which runs under @torch.no_grad(), so freezing them
    # cannot change a single token -- it only makes the liveness assert true.
    fm.requires_grad_(False)
    n_grad = sum(1 for p in fm.parameters() if p.requires_grad)
    assert n_grad == 0, (
        f"{n_grad} FM parameters still require grad; the live FM must be a pure frozen "
        f"feature extractor (the refiner track's whole premise)")
    return fm


class LiveFMPipe:
    """raw h5ad micro-batch -> the SAME dict `cached_collate` produces.

    ⛔ WHERE THIS SITS RELATIVE TO GRADCACHE, AND WHY IT IS NOT IN `forward_fn`.
    `grad_cache_two_pass_n` calls `forward_fn(mb)` TWICE per micro-batch per step --
    once under `torch.no_grad()` to build the leaves and once with grad after the RNG
    replay.  Putting `fm_tokens` inside `forward_fn` (:1404) would therefore run
    scFoundation + EpiAgent TWICE for every cell of every step, and would re-pay the FM
    for all six-plus step-0 probe forwards as well.  So the FM runs ONCE, here, and
    `forward_fn` is left completely untouched -- the dict handed to it satisfies
    `make_fm_tokens_from_cache` exactly.
    """

    def __init__(self, fm, device, args, liveness=None):
        from scripts.train_filip_combined import fm_tokens as _fm_tokens
        self._fm_tokens = _fm_tokens
        self.fm = fm
        self.device = device
        self.liveness = liveness
        self.autocast_rna = bool(int(getattr(args, "fm_autocast_rna", 0)))
        self.token_dtype = (torch.float16
                            if getattr(args, "fm_token_dtype", "fp16") == "fp16"
                            else torch.float32)
        self.max_atac_length = int(args.max_atac_length)
        self._checked = False
        self.n_cells = 0
        self.n_truncated = 0

    def micro(self, raw):
        raw = move_batch_to_device(raw, self.device)
        # `fm_tokens` is `@torch.no_grad()` (train_filip_combined.py:283) and indexes
        # batch[0..2], so a 3-tuple is its native input.  include_summary=True is NOT a
        # choice: `cached_collate` documents its output as
        # `fm_tokens(..., include_summary=True, return_ids=True)`
        # (cached_token_dataset.py:607-609) and this trainer has NO --include_summary
        # flag, so False would silently drop the two RNA meta tokens that `_fm_pool_rna`
        # locates at (valid_count-2, valid_count-1) and shift the ATAC CLS that `encode`
        # reads as the cell vector.
        rc, rt, rm, ac, at, am, rv, cs, gid, cid = self._fm_tokens(
            self.fm, (raw["rd"], raw["gi"], raw["ai"]), self.device,
            include_summary=True, autocast_rna=self.autocast_rna, return_ids=True)
        if not self._checked:
            self._check(rt, rm, gid, at, am, cid)
        td = self.token_dtype
        b = int(rt.shape[0])
        self.n_cells += b
        self.n_truncated += int((raw["atac_len_raw"] > self.max_atac_length - 2).sum())
        # fp16 STORAGE, fp32 COMPUTE -- the cache's own policy (3.5e-04 storage error vs
        # 3.7e-03 for fp16 compute, 10.6x).  `make_fm_tokens_from_cache` widens with
        # `.float()` (cast_float=True), so the arithmetic the refiner sees is identical
        # to every cached run.
        # ⛔ `rc`/`ac` are carried only to keep the dict shape; `forward_fn` deletes them
        # (:1406).  Live `rc` uses the encoder's GLOBAL [-1]/[-2] slice
        # (model.py:215-218), which is PADDING for every cell but the batch-longest
        # (0.590 relative error vs valid-count pooling).  It must never escape.
        return {"rc": rc.to(td), "rt": rt.to(td), "rm": rm,
                "ac": ac.to(td), "at": at.to(td), "am": am,
                "gs": rv.to(td), "cs": cs,
                "gene_id": gid, "ccre_id": cid,
                "labels": raw["labels"],
                "rna_len": (~rm).sum(1), "atac_len": (~am).sum(1),
                "barcode": raw["barcode"]}

    def _check(self, rt, rm, gid, at, am, cid):
        """The invariants separating a live batch from a silently wrong one. ONCE."""
        assert rt.shape[-1] == RNA_DIM and at.shape[-1] == ATAC_DIM, \
            f"FM widths {tuple(rt.shape)} / {tuple(at.shape)}; expected 768 / 512"
        assert rm.dtype == torch.bool and am.dtype == torch.bool, "masks must be bool"
        assert rm.shape == rt.shape[:2] and am.shape == at.shape[:2]
        # PER-TOKEN EMBEDDINGS **AND** IDS -- what FineCLS routing needs.  An id array
        # one token wider or narrower shifts every cCRE's genome coordinate SILENTLY,
        # because the ids stay valid, they just belong to the neighbouring cCRE.
        assert gid.shape == rt.shape[:2] and cid.shape == at.shape[:2], \
            "token ids are not token-aligned; FineCLS routing would read shifted ids"
        # TRUE == PAD, pipeline-wide.  If this inverts, the refiner attends to padding
        # and the pooler drops every real token, and nothing else notices.
        assert int((~rm).sum(1).min()) > 0 and int((~am).sum(1).min()) > 0, \
            "a cell has ZERO valid tokens: mask polarity inverted (True must be PAD)"
        assert int(gid.max()) >= LIVE_N_GENES, (
            "no gene_id >= 19264 in the first batch: the RNA meta tokens are missing, "
            "i.e. include_summary did not take effect")
        assert not bool(am[:, 0].any()), (
            "ATAC CLS at position 0 is masked as padding; `encode` reads it as the "
            "cell vector")
        if self.liveness is not None:
            self.liveness.set("live_fm_rna_token_width", int(rt.shape[1]))
            self.liveness.set("live_fm_atac_token_width", int(at.shape[1]))
            self.liveness.set("live_fm_autocast_rna", bool(self.autocast_rna))
            self.liveness.set("live_fm_token_dtype", str(self.token_dtype))
            self.liveness.set("live_fm_gene_id_max", int(gid.max()))
        self._checked = True


class LiveFMLoader:
    """A DataLoader whose batches have already been through the frozen FMs.

    The FM forward MUST happen in the main process (workers have no GPU), so this is a
    thin generator over the real loader rather than a `collate_fn`.  Consequence, stated
    plainly: the FM is on the CRITICAL PATH of every step and is NOT overlapped with the
    refiner's compute the way `prefetch_factor` overlaps the h5ad read.
    """

    def __init__(self, loader, pipe):
        self._loader = loader
        self._pipe = pipe
        self._buffer_depth = getattr(loader, "_buffer_depth", 0)

    def __iter__(self):
        for raw in self._loader:
            yield self._pipe.micro(raw)

    def __len__(self):
        return len(self._loader)


_LIVE_PIPE = [None]


def get_live_pipe(args, liveness=None):
    """ONE frozen FM pair per process, built lazily on the first `make_loader` call.

    `make_loader` runs up to twice per run (train, val) and the FM pair is several GB of
    frozen weights, so it is a process singleton.  The device comes from
    `torch.cuda.current_device()`, which `setup_distributed` has already pinned to
    LOCAL_RANK.  ⛔ A second live loader built with a DIFFERENT --max_atac_length or
    --fm_autocast_rna would silently reuse the first pipe's settings; the two call sites
    in `make_loader` pass the same `args`, so that cannot happen today.
    """
    if _LIVE_PIPE[0] is None:
        if args.device:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")
        _LIVE_PIPE[0] = LiveFMPipe(build_live_fm(args, device), device, args, liveness)
    elif liveness is not None and _LIVE_PIPE[0].liveness is None:
        _LIVE_PIPE[0].liveness = liveness
    return _LIVE_PIPE[0]



def resolve_prefetch_factor(args) -> int:
    """`--prefetch_factor 0` -> the smallest depth that covers one whole step.

    WHY A WHOLE STEP.  `train()` does
    `micro = [move_batch_to_device(next(it), device) for _ in range(accum_freq)]`
    before it touches the GPU.  The DataLoader can only hand over what its workers have
    already produced -- `num_workers * prefetch_factor` batches -- and the remainder is
    produced while the main process blocks in `next()`.  Covering accum_freq means the
    workers, which ran for the whole previous GPU step, have the entire next step ready.
    """
    if args.prefetch_factor > 0:
        return int(args.prefetch_factor)
    nw = max(1, int(args.num_workers))
    return max(2, -(-int(args.accum_freq) // nw))          # ceil, floor of 2 = torch's


def make_loader(args, split: str, world: int, rank: int, num_steps: int, seed: int):
    """Cache -> (dataset, DataLoader) with the same_dataset_blocked batch sampler.

    ⛔ THE SAMPLER MUST BE FED THE CACHE'S OWN dataset ids.  The cache's row order is
    SHARD order, not h5ad order, so `PairedMultiOmicsDataset.get_dataset_ids()` would
    hand back a same-looking id array attached to the WRONG rows and every "blocked"
    batch would silently mix datasets. `CachedTokenDataset.get_dataset_ids()` is aligned
    to `__getitem__` order by construction.
    """
    sub = None
    if args.subsample_kg or args.subsample_kc:
        sub = {"kg": args.subsample_kg or None, "kc": args.subsample_kc or None}
    shards = args.train_shards
    if shards != "all":
        shards = [int(s) for s in str(shards).split(",")]
    if args.live_fm:
        # ⛔ NO CACHE.  --live_fm reads the raw paired h5ads and runs the FMs in this
        # process.  --train_shards / --subsample_* are cache-only concepts and are
        # refused at parse time rather than silently ignored here.
        if args.live_paired_dir:
            units = live_units_from_dir(args.live_paired_dir)
            want_val = (split == args.val_split)
        else:
            units = resolve_live_corpus(args, split, verbose=(rank == 0))
            want_val = None            # the whitelists already decide the split
        ds = PairedH5adCorpus(
            units, split, want_val=want_val,
            holdout_mod=args.live_val_holdout_mod,
            holdout_rem=args.live_val_holdout_rem,
            max_atac_length=args.max_atac_length,
            labels_csv=args.labels_csv,
            allow_unlabeled=bool(args.allow_unlabeled),
            open_files=args.live_open_files,
            verify_pairing=bool(args.live_verify_pairing), verbose=(rank == 0))
    else:
        ds = CachedTokenDataset(
            args.cache_root, split, shards=shards if split == args.train_split else "all",
            labels_csv=args.labels_csv, allow_unlabeled=bool(args.allow_unlabeled),
            subsample=sub, verbose=(rank == 0))
    sampler = SameDatasetBlockedBatchSampler(
        ds.get_dataset_ids(args.dataset_id_column), micro_batch=args.micro_batch,
        accum=args.accum_freq, num_steps=num_steps, seed=seed, rank=rank,
        world_size=world)
    from functools import partial
    # max_atac_length 0 -> None -> pad to the batch max. Real runs pin 8192; a
    # batch-dependent width is not inert for anything that takes a top-k over tokens.
    if args.live_fm:
        # ⚠️ `collate_fn`'s `fixed_atac_length` has NO "pad to batch max" mode and it
        # TRUNCATES SILENTLY (multiomics_clip/dataset.py:588-592), where
        # `cached_collate` REFUSES to.  The truncation actually happens one level
        # earlier, in `PairedH5adCorpus.__getitem__` at `max_atac_length - 2` -- head of
        # the TF-IDF-descending sentence, the rule the cache was built under -- and is
        # COUNTED there, so it is reported, not silent.  ~36% of this corpus' cells
        # exceed 8190 cCREs.
        coll = partial(live_collate, fixed_atac_length=args.max_atac_length)
    else:
        coll = partial(cached_collate, spec=ds.spec,
                       max_atac_length=args.max_atac_length or None)
    # --prefetch_factor: how deep the worker pool runs ahead.  See the argparse help --
    # the step loop's pull is SERIAL, so a buffer shallower than accum_freq is paid in
    # wall clock.  `prefetch_factor` is only legal with num_workers > 0 (torch raises
    # otherwise), and torch's own default is 2, so keep None in the 0-worker case.
    kw = {}
    if args.num_workers > 0:
        kw["prefetch_factor"] = resolve_prefetch_factor(args)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers,
                        collate_fn=coll, pin_memory=torch.cuda.is_available(),
                        persistent_workers=bool(args.num_workers), **kw)
    # The liveness statistic is the BUFFER DEPTH against accum_freq, not the flag: a
    # prefetch_factor that argparse accepted but that does not cover a step is exactly
    # the silently-inert flag this project keeps producing.
    loader._buffer_depth = args.num_workers * kw.get("prefetch_factor", 0)
    if args.live_fm:
        # ⛔ THIS -- NOT `forward_fn` -- IS THE `make_fm_tokens_from_cache` SWAP.
        # `LiveFMPipe.micro` calls `fm_tokens(fm, (rd, gi, ai), device,
        # include_summary=True, return_ids=True)` and re-keys its 10-tuple into the
        # dict `cached_collate` produces, element for element, so `forward_fn`,
        # `move_batch_to_device`, `compute_val_loss`, the step-0 probes and every
        # gradcache path stay untouched.
        # Putting the FM in `forward_fn` instead would run it TWICE per micro-batch per
        # step (gradcache pass-1 under no_grad, pass-2 with grad after the RNG replay)
        # and would re-pay it for all six-plus step-0 probe forwards as well.
        loader = LiveFMLoader(loader, get_live_pipe(args))
    return ds, loader, sampler


def build_scheduler(opt, args):
    """warmup -> flat (constant) or warmup -> cosine.

    L4 asserts on the LR the scheduler ACTUALLY RETURNS, not on the flag's presence in
    argparse: this project has had four silently-inert flags in one day.
    """
    def lam(step: int) -> float:
        warm = min(1.0, (step + 1) / max(1, args.warmup_steps))
        if args.lr_schedule == "constant":
            return warm
        if args.lr_schedule == "cooldown":
            # TERMINAL COOL-DOWN: linear lr -> 0 across [cooldown_start, num_steps].
            #
            # ⛔ WHY THIS IS NOT JUST "cosine".  `cosine` computes progress from step 0
            # over num_steps, so on a RESUMED run it would jump to whatever the cosine
            # happens to be at the resume step -- a discontinuity, not a cool-down.
            # This anchors the decay at the resume point so the LR is continuous.
            #
            # ⚠️ AND WHY IT NEEDS A CONTROL.  A cool-down forces the loss flat whether or
            # not a minimum was found, which is exactly the ambiguity constant-LR was
            # adopted to avoid ("a flat tail is the SCHEDULE not convergence").  The
            # evidence is recovered by cooling from TWO different starting checkpoints:
            # if both land on the same loss, the steps between them bought nothing and
            # the run WAS converged; if the later one lands lower, they did.
            lo = int(getattr(args, "cooldown_start", 0) or 0)
            span = max(1, int(args.num_steps) - lo)
            prog = min(1.0, max(0.0, (step - lo) / span))
            return warm * (1.0 - prog)
        prog = min(1.0, step / max(1, args.num_steps))
        return warm * 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, lam)


def write_manifest(args, model, liveness: Liveness, ds, world: int, path: str) -> Dict:
    """Run manifest -- written BEFORE step 0, so a crashed run stays auditable."""
    comp = {}
    try:
        comp = ds.manifest.get("splits", {}).get(args.train_split, {}).get(
            "completeness", {})
        comp = {"status": comp.get("status"),
                "rna_complete": comp.get("rna", {}).get("complete"),
                "atac_complete": comp.get("atac", {}).get("complete"),
                "rna_shards": comp.get("rna", {}).get("expected"),
                "atac_shards": comp.get("atac", {}).get("expected")}
    except Exception as exc:                                   # pragma: no cover
        comp = {"error": repr(exc)}
    man = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "argv": sys.argv,
        "arm": args.arm,
        "num_slots": int(model.num_slots),
        "seed": int(args.seed),
        "world_size": int(world),
        # ⛔ THE POOL DEFINITION IS THE METRIC, and `batch_size * world` is the pool
        # ONLY when the ranks actually see each other.  `--gather_negatives` is read at
        # exactly one place (in `make_loss_fn`) and was written to NEITHER the manifest
        # NOR liveness, so a 4-GPU run's manifest claimed 512 while `--gather_negatives
        # 0` would have left each rank contrasting against its own 128 -- a 4x lie in
        # the one number the val loss's ln(pool) offset is measured against.  Record the
        # flag AND the true pool; the old key is kept so nothing downstream breaks.
        "gather_negatives": int(args.gather_negatives),
        "contrastive_pool": (int(args.batch_size * world)
                             if (args.gather_negatives and world > 1)
                             else int(args.batch_size)),
        "effective_contrastive_batch": int(args.batch_size * world),
        "micro_batch": int(args.micro_batch),
        "accum_freq": int(args.accum_freq),
        "num_workers": int(args.num_workers),
        # Recorded because it is a WALL-CLOCK term, not a science term: a run whose
        # buffer did not cover a step is ~20 % slower than one whose did, and that
        # difference is otherwise invisible after the fact.
        "prefetch_factor": (int(resolve_prefetch_factor(args))
                            if args.num_workers > 0 else 0),
        "loader_buffer_depth": (int(args.num_workers)
                                * int(resolve_prefetch_factor(args))
                                if args.num_workers > 0 else 0),
        "lr_schedule": args.lr_schedule,
        "lr": args.lr, "fm_layer_lr": args.fm_layer_lr,
        "max_atac_length": int(args.max_atac_length),
        # The four SPEED flags. In the manifest AND in every snapshot: two of them
        # change the arithmetic, and a checkpoint whose precision policy cannot be
        # recovered cannot be compared to anything.
        "rna_grad_ckpt": int(args.rna_grad_ckpt),
        "rna_sub_batch": int(args.rna_sub_batch),
        "atac_attn": ("varlen" if args.atac_varlen else
                      ("sdpa" if args.atac_sdpa else "dense")),
        "refiner_precision": args.refiner_precision,
        "rna_autocast": args.rna_autocast,
        "atac_autocast": args.atac_autocast,
        # The loss scale is part of the ARITHMETIC. Two arms that ended at different
        # scales, or took different numbers of applied updates, are not the clean
        # one-factor comparison the arm matrix is built to be -- so the settings AND the
        # outcome (final scale, minimum scale, applied/skipped counts, all in
        # `liveness`) are both recorded here.
        "grad_scaler": bool(args.grad_scaler),
        "grad_scaler_init_scale": float(args.grad_scaler_init_scale),
        "grad_scaler_growth_interval": (int(args.grad_scaler_growth_interval)
                                        if args.grad_scaler_growth_interval > 0
                                        else int(args.num_steps) + 1),
        "grad_scaler_growth_on": bool(args.grad_scaler_growth_interval > 0),
        "grad_scaler_min_scale": float(args.grad_scaler_min_scale),
        "cache_root": (args.cache_root if not args.live_fm else
                       f"n/a (--live_fm 1: {args.live_paired_dir or args.corpus_manifest})"),
        "live_fm": bool(args.live_fm),
        "live_corpus": ({"manifest": args.corpus_manifest,
                         "paired_dir": args.live_paired_dir,
                         "cohorts": args.corpus_cohorts,
                         "train_whitelist": args.train_whitelist,
                         "val_whitelist": args.val_whitelist,
                         "files": len(getattr(ds, "files", []) or []),
                         "label_coverage": round(float(getattr(ds, "label_coverage",
                                                               0.0)), 4),
                         "n_classes": int(getattr(ds, "n_classes", 0)),
                         "rna_nnz_filter": LIVE_RNA_NNZ_FILTER_NOTE,
                         "fm_sdpa": bool(args.fm_sdpa),
                         "fm_autocast_rna": bool(args.fm_autocast_rna),
                         "fm_token_dtype": args.fm_token_dtype,
                         "rna_encoder_path": args.rna_encoder_path,
                         "atac_encoder_path": args.atac_encoder_path}
                        if args.live_fm else None),
        "cache_cells": int(len(ds)),
        "cache_completeness": comp,
        "params_total": int(sum(p.numel() for p in model.parameters())),
        "params_trainable": int(sum(p.numel() for p in model.parameters()
                                    if p.requires_grad)),
        "params_slot_branch": int(sum(p.numel() for p in model.slot_parameters())),
        "params_refiner": int(sum(p.numel() for p in model.refiner_parameters())),
        "liveness": {**liveness.values, **{f"count:{k}": v for k, v in
                                           liveness.counts.items()}},
        "registered_prediction": (
            "H1 PRIMARY: B - A on the GLOBAL scorer, 4-OOD mean within-(dataset x "
            "cell_type) R@1, pool 128, >=500 draws, PER DIRECTION, at a FIXED step, "
            "scored raw AND test-centred. PREDICTED NULL, delta in [-0.001, +0.002], "
            "possibly negative (the colleague reports global R@1 -2.1% as FineCLS's "
            "price, and this project has 6/6 that in-dist gains do not reach OOD)."),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(man, fh, indent=2)
    return man


# ------------------------------------------------------------------------------------ #
# One optimizer step
# ------------------------------------------------------------------------------------ #

def make_forward_fn(model, args, liveness: Liveness):
    """mb (a collated cache batch, already on device) -> (cached tensors, aux)."""
    def forward_fn(mb):
        # `fm_tokens()`'s 10-tuple, element for element. `rc` / `ac` (the FROZEN FM
        # cell vectors) are deliberately UNUSED: feeding them into the include_cls
        # concat is TRAP 3, and it would leave the cell half of every slot a constant
        # while every gradient assert stayed green.
        # ⛔ THE LIVE-FM SWAP IS **NOT** HERE.  `fm_tokens(model, batch, device,
        # include_summary=True, return_ids=True)` (train_filip_combined.py:284-317)
        # returns this exact 10-tuple, so the literal swap would type-check -- and would
        # run scFoundation + EpiAgent TWICE PER MICRO-BATCH PER STEP, because
        # `grad_cache_two_pass_n` calls `forward_fn` once under no_grad to build the
        # leaves and again with grad after the RNG replay.  So `--live_fm` runs the FMs
        # ONCE, in `LiveFMLoader.__iter__`, and hands this call the identical dict.
        # `mb` is therefore EITHER a cached batch OR a live-FM batch; both carry
        # {rc, rt, rm, ac, at, am, gs, cs, gene_id, ccre_id, labels, barcode}.
        rc, rt, rm, ac, at, am, rv, cs, gid, cid = make_fm_tokens_from_cache(
            mb, return_ids=True)
        del rc, ac, rv, cs
        zr, za, rs, a_s, rvd, avd, rms, ams = model(rt, rm, gid, at, am, cid, liveness)
        labels = mb["labels"]
        if model_of(model).arm == "finecls":
            return (zr, za, rs, a_s), {"labels": labels, "rv": rvd, "av": avd,
                                       "rm": rms, "am": ams}
        return (zr, za), {"labels": labels}
    return forward_fn


def model_of(m):
    """The wrapped module, DDP or not."""
    return m.module if hasattr(m, "module") else m


def make_loss_fn(model, args, world: int, step_ref: List[int], liveness: Liveness):
    """The loss on the FULL (gathered) contrastive batch.

    Centring lives HERE, not in the forward, because the mean must be taken over the
    whole contrastive window -- which under gradient caching is the concatenation of
    every micro-batch, and under DDP the concatenation across ranks as well.
    `groups=None` is correct and not a shortcut: `SameDatasetBlockedBatchSampler` draws
    ONE dataset per optimizer step, so the window mean IS the dataset mean.

    ⛔ CENTRING IS A TRAIN x TEST INTERACTION. Train-time alone is worth ~0; test-time is
    the headline; together they are 1.45x/1.40x. `_dcenter`-style gating on
    `self.training` is why every checkpoint must be scored BOTH ways at eval time.
    Nothing here can substitute for that.
    """
    fp = _filip_pieces()
    center_by_group, center_slots_by_group = _centering_helpers()
    mod = model_of(model)

    def loss_fn(full, aux):
        labels = aux["labels"]
        if args.arm == "finecls":
            zr, za, rs, a_s = full
            rvd, avd, rms, ams = aux["rv"], aux["av"], aux["rm"], aux["am"]
        else:
            zr, za = full
            rs = a_s = rvd = avd = rms = ams = None
        if args.gather_negatives and dist.is_initialized() and world > 1:
            zr, za = GatherWithGrad.apply(zr), GatherWithGrad.apply(za)
            if rs is not None:
                # shape-agnostic: works unchanged on [b, M, 256]
                rs, a_s = GatherWithGrad.apply(rs), GatherWithGrad.apply(a_s)
            labels = _gather_plain(labels, world)
            if rvd is not None:
                rvd, avd = _gather_plain(rvd, world), _gather_plain(avd, world)
                rms, ams = _gather_plain(rms, world), _gather_plain(ams, world)
        if args.center_global_by_dataset:
            zr, za = center_by_group(zr, None), center_by_group(za, None)
            liveness.bump("center_global_calls", 2)
        if rs is not None and args.center_slots_by_dataset:
            rs = center_slots_by_group(rs, rvd, None)
            a_s = center_slots_by_group(a_s, avd, None)
            liveness.bump("center_slots_calls", 2)

        stats: Dict[str, float] = {}
        total = zr.new_zeros(())
        if args.cell_infonce_weight > 0:
            l = fp["cell_infonce"](zr, za, args.temp)
            total = total + args.cell_infonce_weight * l
            stats["cell_infonce"] = float(l)
        if args.align_weight > 0:
            l = fp["align_loss"](zr, za)
            total = total + args.align_weight * l
            stats["align"] = float(l)
        if args.supcon_weight > 0 and labels is not None:
            l = fp["supcon_xmodal"](zr, za, labels, args.temp)
            total = total + args.supcon_weight * l
            stats["supcon_xmodal"] = float(l)
        stats["global_total"] = float(total)
        if rs is not None and args.fine_weight > 0:
            gib = fine_logit_gib(rs.shape[0], rs.shape[1])
            assert gib <= args.max_fine_logit_gib, (
                f"per-slot logits are [{rs.shape[0]},{rs.shape[0]},{rs.shape[1]}] "
                f"fp32 = "
                f"{gib:.2f} GiB and ~6 are live per direction; raise "
                f"--max_fine_logit_gib deliberately or lower the contrastive batch")
            l, fstats = fine_loss(rs, a_s, rvd, avd, rms, ams, labels,
                                  args.module_temperature, mod.routing,
                                  args.fine_infonce_frac)
            # Ramp, never to EXACTLY zero: a zero-weight branch has no gradient, which
            # under DDP(find_unused_parameters=False) crashes rather than warns.
            ramp = min(1.0, (step_ref[0] + 1) / max(1, args.fine_warmup_steps))
            total = total + args.fine_weight * ramp * l
            stats.update(fstats)
            stats["fine"] = float(l)
            stats["fine_ramp"] = float(ramp)
        if rs is not None and args.fused_weight > 0:
            lf, facc = fused_infonce(zr, za, rs, a_s, rvd, avd, rms, ams,
                                     args.temp, mod.routing,
                                     alpha=args.fused_alpha,
                                     use_zscore=bool(args.fused_zscore))
            # Same ramp as the fine branch: the slot projection is random at step 0, so an
            # un-ramped fused term would drag the global branch toward noise.
            ramp = min(1.0, (step_ref[0] + 1) / max(1, args.fine_warmup_steps))
            total = total + args.fused_weight * ramp * lf
            stats["fused_infonce"] = float(lf)
            stats["fused_acc"] = float(facc)
        stats["total"] = float(total)
        return total, stats
    return loss_fn


def _gather_plain(t: torch.Tensor, world: int) -> torch.Tensor:
    """all_gather WITHOUT grad -- for labels / slot `valid` / slot `mass`.

    These carry no learnable weights (`mass[b,m] = sum_n prior[id_n,m]*valid_n` is a
    pure function of token ids), so gathering them with grad would build a graph over
    constants.
    """
    buf = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(buf, t.contiguous())
    return torch.cat(buf, 0)


def single_pass_step(micro_batches, forward_fn, loss_fn, ddp_mods=(), scaler=None):
    # DDP expects exactly one forward per backward. This path does `accum` forwards and
    # ONE backward, which trips the reducer -- so it is refused rather than silently
    # producing wrong (or hanging) gradients. It is a control, not a training regime.
    assert not ddp_mods, ("--grad_cache 0 is a single-process control path; under DDP "
                          "its accum-forwards/one-backward shape breaks the reducer")
    """The control path (`--grad_cache 0`): forward EVERY micro-batch WITH grad.

    Mathematically identical to the two-pass cache -- same contrastive batch, same loss,
    same gradients -- and it differs ONLY in peak activation memory (all micro-batches
    alive at once instead of one). Kept because it is the reference the equivalence
    check compares against; it is not an alternative training regime.
    """
    outs_all, aux_all = [], []
    for mb in micro_batches:
        outs, aux = forward_fn(mb)
        outs_all.append(outs)
        aux_all.append(aux)
    full = tuple(torch.cat([o[i] for o in outs_all], 0)
                 for i in range(len(outs_all[0])))
    aux_full = {}
    for k in aux_all[0]:
        vals = [a[k] for a in aux_all]
        aux_full[k] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    total, extra = loss_fn(full, aux_full)
    # Same composition as the two-pass: scale the loss, return the TRUE loss.  The
    # equivalence probe compares the two paths, so a scale applied to only one of them
    # would make the comparison meaningless (and `rel()` normalises by max|a|, so a
    # COMMON scale cancels exactly and the verdict is unchanged).
    (scaler.scale(total) if scaler is not None else total).backward()
    return total.detach(), extra


# ------------------------------------------------------------------------------------ #
# LIVENESS: the --arm flag, asserted on the quantity it CONTROLS
# ------------------------------------------------------------------------------------ #

def grad_norm(params) -> float:
    """L2 norm over a parameter GROUP, ignoring parameters with no grad.

    Logged every `--log_steps` for the refiner specifically, because the aggregate
    `gnorm` is dominated by the heads and would stay healthy-looking while the refiner
    received nothing. A refiner grad norm that is exactly 0.0 for a whole run is the
    TRAP-1 signature and must be visible in the log, not only in the step-0 probe.
    """
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    return sq ** 0.5


def _grad_max(model, name: str) -> Optional[float]:
    p = dict(model.named_parameters()).get(name)
    if p is None or p.grad is None:
        return None
    return float(p.grad.abs().max())


def arm_liveness(model, micro_batches, forward_fn, args, liveness: Liveness, log=print):
    """Prove `--arm` is LIVE, with a POSITIVE printed value in BOTH directions.

    What `--arm` controls is not "a config string" but "does the fine (slot) loss
    deliver gradient into the shared refiner".  So that is what is measured, on a real
    batch:

      arm=finecls  : backward the FINE loss ALONE -> max|grad| > 0 on a parameter INSIDE
                     each refiner's attention stack AND on `slot_queries`; then backward
                     the GLOBAL loss alone -> `slot_queries.grad` must be EXACTLY 0.0
                     while the shared refiner still receives gradient.
      arm=cell_only: the slot branch must not exist (0 modules, 0 parameters) AND the
                     global loss must still reach both refiners -- printed, not implied.

    ⛔ A non-None gradient is NOT proof an arm trains. `--fm_layer_lr 0` is a broken
    freeze here: gradients flow, the parameter never moves, every gradient assert stays
    green. The optimizer-step check in `main` (max|param delta| > 0 after one step) is
    the other half and is not optional.
    """
    mod = model_of(model)
    mb = micro_batches[0]
    fp = _filip_pieces()
    n_slot_params = sum(p.numel() for p in mod.slot_parameters())
    n_slot_modules = sum(1 for m in (mod.rna_pooler, mod.atac_pooler,
                                     mod.rna_slot_projection, mod.atac_slot_projection,
                                     mod.rna_adapter, mod.atac_adapter)
                          if m is not None)
    liveness.set("arm", args.arm)
    liveness.set("slot_modules_allocated", n_slot_modules)
    liveness.set("slot_params", n_slot_params)

    outs, aux = forward_fn(mb)
    if args.arm == "cell_only":
        assert n_slot_modules == 0 and n_slot_params == 0, \
            "arm=cell_only allocated slot machinery -- the arms are not what they claim"
        assert len(outs) == 2, f"arm=cell_only returned {len(outs)} tensors, expected 2"
        zr, za = outs
        glob = fp["cell_infonce"](zr, za, args.temp)
        mod.zero_grad(set_to_none=True)
        glob.backward()
        gr, ga = _grad_max(mod, RNA_PROBE_PARAM), _grad_max(mod, ATAC_PROBE_PARAM)
        assert gr and gr > 0 and ga and ga > 0, (
            f"arm=cell_only: the GLOBAL loss does not reach the refiners "
            f"(rna {gr}, atac {ga}) -- the trunk is severed")
        liveness.set("global_only_grad_rna_refiner", f"{gr:.3e}")
        liveness.set("global_only_grad_atac_refiner", f"{ga:.3e}")
        liveness.set("fine_only_grad_rna_refiner",
                     "n/a (no slot branch by construction)")
    else:
        assert n_slot_modules >= 4 and n_slot_params > 0, \
            "arm=finecls allocated no slot machinery"
        assert len(outs) == 4, f"arm=finecls returned {len(outs)} tensors, expected 4"
        zr, za, rs, a_s = outs
        assert torch.isfinite(rs).all() and torch.isfinite(a_s).all(), \
            "slot embeddings are not finite (FixedSlotPooler nan_to_num under autocast)"
        # L11, asserted rather than relied on: the two RNA meta tokens (ids 19264/19265)
        # map to rows outside the prior, so they carry ZERO slot mass.
        gid = make_fm_tokens_from_cache(mb, return_ids=True)[8].long()
        meta = (gid == RNA_META_IDS[0]) | (gid == RNA_META_IDS[1])
        row = gid - 0                                   # RNA token_offset is 0
        in_vocab = (row >= 0) & (row < N_GENES)
        n_meta_in_vocab = int((meta & in_vocab).sum())
        assert n_meta_in_vocab == 0, (
            f"{n_meta_in_vocab} RNA meta tokens landed INSIDE the prior vocabulary: "
            f"(resolution, log10_total) would be pooled as if they were genes")
        liveness.set("rna_meta_tokens_in_batch", int(meta.sum()))
        liveness.set("rna_meta_tokens_reaching_a_slot", n_meta_in_vocab)

        fine, _ = fine_loss(rs, a_s, aux["rv"], aux["av"], aux["rm"], aux["am"],
                            aux["labels"], args.module_temperature, mod.routing,
                            args.fine_infonce_frac)
        mod.zero_grad(set_to_none=True)
        fine.backward(retain_graph=False)
        gr, ga = _grad_max(mod, RNA_PROBE_PARAM), _grad_max(mod, ATAC_PROBE_PARAM)
        gq = _grad_max(mod, "rna_pooler.slot_queries")
        assert gr and gr > 0, (
            f"arm=finecls: the FINE loss alone does NOT reach {RNA_PROBE_PARAM} "
            f"(grad={gr}). That is the TRAP-1 failure: the refiner is fed through "
            f"a no-grad path and would train on the global term only.")
        assert ga and ga > 0, f"fine loss does not reach {ATAC_PROBE_PARAM} (grad={ga})"
        assert gq and gq > 0, f"fine loss does not reach the slot queries (grad={gq})"
        liveness.set("fine_only_grad_rna_refiner", f"{gr:.3e}")
        liveness.set("fine_only_grad_atac_refiner", f"{ga:.3e}")
        liveness.set("fine_only_grad_slot_queries", f"{gq:.3e}")

        # CONVERSE: global-only must leave the slot-only parameters at exactly zero.
        outs2, _ = forward_fn(mb)
        glob = fp["cell_infonce"](outs2[0], outs2[1], args.temp)
        mod.zero_grad(set_to_none=True)
        glob.backward()
        gq2 = _grad_max(mod, "rna_pooler.slot_queries")
        gr2 = _grad_max(mod, RNA_PROBE_PARAM)
        assert gq2 is None or gq2 == 0.0, (
            f"the GLOBAL loss reaches the slot queries (grad={gq2}): the two terms are "
            f"not separable and no A-vs-B attribution would be possible")
        assert gr2 and gr2 > 0, "global loss does not reach the (shared) refiner"
        liveness.set("global_only_grad_slot_queries",
                     "None" if gq2 is None else f"{gq2:.3e}")
        liveness.set("global_only_grad_rna_refiner", f"{gr2:.3e}")
    mod.zero_grad(set_to_none=True)


def optimizer_liveness(opt, model, liveness: Liveness) -> None:
    """Every refiner parameter must sit in a group with lr > 0 (L6)."""
    mod = model_of(model)
    ref_ids = {id(p) for p in mod.refiner_parameters()}
    seen, lrs = 0, set()
    for g in opt.param_groups:
        for p in g["params"]:
            if id(p) in ref_ids:
                seen += 1
                lrs.add(g["lr"])
    assert seen == len(ref_ids), \
        f"{len(ref_ids) - seen} refiner parameters are in NO optimizer group"
    assert lrs and min(lrs) > 0, \
        f"refiner lr is {lrs}: --fm_layer_lr 0 is a broken freeze here"
    liveness.set("refiner_params_in_optimizer", seen)
    liveness.set("refiner_group_lr", sorted(lrs))


def sdpa_liveness(model, args, liveness: Liveness) -> None:
    """`--atac_sdpa` took: a BOUND method has `__self__`, the closure does not."""
    mod = model_of(model)
    patched = [not hasattr(b.mixer.inner_attn.forward, "__self__")
               for b in mod.atac_refiner.blocks]
    liveness.set("atac_sdpa_flag", int(args.atac_sdpa))
    liveness.set("atac_sdpa_blocks_patched", patched)
    assert all(patched) == bool(args.atac_sdpa), (
        f"--atac_sdpa={args.atac_sdpa} but per-block patched={patched}; without SDPA "
        f"the [B,H,S,S] score matrix is materialised and OOMs at S=8192")


def refiner_speed_liveness(model, micro_batches, args, liveness: Liveness,
                           log=print) -> None:
    """PROVE the four SPEED flags are live -- on the quantity each one CONTROLS.

    THE RULE THIS IMPLEMENTS.  This project produced four silently-inert flags in one
    day and two arms that came out bit-identical (max|delta| 0.000e+00).  argparse
    accepting a flag, the config carrying it and the log printing it ALL pass in that
    failure mode.  So each flag below is asserted on a MEASURED behavioural consequence,
    and the measurement is PRINTED in both settings:

      --rna_grad_ckpt   counts nn.TransformerEncoderLayer.forward CALLS across one
                        fwd+bwd.  Checkpointing recomputes the stack in backward, so
                        the count is 2 x n_layers with it and n_layers without.  A flag
                        that failed to take shows up as the wrong count, not as silence.
      --rna_sub_batch   counts the calls AND sums rows x width over them -- the PADDED
                        token count the attention actually pays for.  Compared against
                        the true (valid) token count and against the shipped path's
                        B x N.  Asserted to be STRICTLY smaller whenever the micro-batch
                        holds cells of differing length, which is the only case where
                        the flag can do anything.
      --atac_varlen     asserts the block is genuinely on flash_attn's varlen path
                        (inner_attn is FlashSelfAttention, mixer.use_flash_attn True)
                        AND that its attention dropout is 0.0.  ⛔ THE SECOND HALF IS
                        THE LOAD-BEARING ONE: create_block wires
                        FlashSelfAttention(attention_dropout=
                        attention_probs_dropout_prob),
                        which is --dropout (0.2 in production), while the shipped SDPA
                        path hardwires 0.0.  Un-asserted, --atac_varlen would ship a
                        REGULARIZATION change wearing a speed change's clothes.
      --refiner_precision
                        records the dtype the stack actually computed in.  `torch.
                        autocast` is silently inert on CPU and silently inert if the
                        context does not wrap the op, so the policy string is not
                        evidence -- the observed dtype is.

    Runs on ONE micro-batch, on the UNWRAPPED model, before DDP wrapping, and zeroes the
    gradients it produced.
    """
    mod = model_of(model)
    mb = micro_batches[0]
    rt, rm = mb["rt"].float(), mb["rm"]
    at, am = mb["at"].float(), mb["am"]
    B, N = int(rt.shape[0]), int(rt.shape[1])
    lens = (~rm).sum(1).tolist()
    n_layers = len(mod.rna_refiner.layers)

    calls: List[Tuple[int, int, str]] = []

    # ⛔ NOT a forward HOOK.  torch 2.2 SUPPRESSES module forward hooks during
    # `checkpoint(use_reentrant=False)`'s backward recompute -- measured: a hook counts
    # 2 calls with grad_ckpt on and 2 with it off, while patching the bound `forward`
    # counts 4 and 2.  A hook-based liveness check would therefore have reported
    # "--rna_grad_ckpt is inert" for a flag that is working perfectly, which is exactly
    # the class of false negative this file exists to avoid.  An instance attribute
    # shadows the class method and `Module._call_impl` dispatches through
    # `self.forward`,
    # so this sees the recompute.
    import types as _types

    def wrap(lay):
        base = type(lay).forward

        def f(self, x, *a, **k):
            calls.append((int(x.shape[0]), int(x.shape[1]),
                          str(x.dtype).split(".")[-1]))
            return base(self, x, *a, **k)
        return _types.MethodType(f, lay)

    # The GEMM dtype, which is what a precision policy actually controls.  ⛔ NOT the
    # layer's INPUT dtype: `torch.autocast` casts per OP, and `nn.LayerNorm` is on its
    # fp32 list, so a post-norm TransformerEncoderLayer hands the next layer fp32 even
    # under bf16 -- an input-dtype assert reports "the flag is inert" for a flag that is
    # accelerating every matmul in the stack.  Measured that false negative here before
    # this comment existed.
    gemm: Dict[str, str] = {}

    def dt_hook(_m, _a, o):
        gemm.setdefault("rna", str(o.dtype).split(".")[-1])

    dth = mod.rna_refiner.layers[0].linear1.register_forward_hook(dt_hook)
    layers = list(mod.rna_refiner.layers)
    for lay in layers:
        lay.forward = wrap(lay)
    try:
        was = mod.training
        mod.train()
        out = mod.rna_refiner.native_grad(rt, rm)
        out.float().sum().backward()
        mod.zero_grad(set_to_none=True)
        if not was:
            mod.eval()
    finally:
        for lay in layers:
            del lay.forward
        dth.remove()

    # -- --rna_grad_ckpt ------------------------------------------------------------- #
    groups = 1 if args.rna_sub_batch <= 0 or args.rna_sub_batch >= B else \
        -(-B // args.rna_sub_batch)
    want = n_layers * groups * (2 if args.rna_grad_ckpt else 1)
    liveness.set("rna_grad_ckpt_flag", int(args.rna_grad_ckpt))
    liveness.set("rna_layer_forward_calls", len(calls))
    liveness.set("rna_layer_forward_calls_expected", want)
    assert len(calls) == want, (
        f"--rna_grad_ckpt={args.rna_grad_ckpt} --rna_sub_batch={args.rna_sub_batch} "
        f"but the RNA layer stack ran {len(calls)} forwards over one fwd+bwd, not "
        f"{want} ({n_layers} layers x {groups} groups x "
        f"{2 if args.rna_grad_ckpt else 1} passes). One of the two flags is INERT.")

    # -- --rna_sub_batch: the padded-token count it exists to cut -------------------- #
    padded = sum(r * n for r, n, _ in calls) // (2 if args.rna_grad_ckpt else 1)
    shipped = B * N * n_layers
    true_tok = sum(lens) * n_layers
    liveness.set("rna_sub_batch_flag", int(args.rna_sub_batch))
    liveness.set("rna_padded_tokens_per_layerpass", int(padded // n_layers))
    liveness.set("rna_padded_tokens_shipped_path", int(shipped // n_layers))
    liveness.set("rna_true_tokens", int(true_tok // n_layers))
    log(f"[liveness] RNA micro-batch B={B} pad width N={N} | padded tokens/layer "
        f"{padded // n_layers:,} vs shipped {shipped // n_layers:,} vs true "
        f"{true_tok // n_layers:,} | flag --rna_sub_batch {args.rna_sub_batch}")
    if args.rna_sub_batch > 0 and min(lens) != max(lens) and args.rna_sub_batch < B:
        assert padded < shipped, (
            f"--rna_sub_batch={args.rna_sub_batch} changed NOTHING: the stack still "
            f"paid for {padded:,} padded tokens against the shipped path's "
            f"{shipped:,}, on a micro-batch whose lengths run "
            f"{min(lens)}..{max(lens)}. The flag is INERT.")
        if args.rna_sub_batch == 1:
            assert padded == true_tok, (
                f"--rna_sub_batch=1 must pad NOTHING, but paid {padded:,} for "
                f"{true_tok:,} real tokens")
    elif args.rna_sub_batch <= 0:
        assert padded == shipped, "sub-batching is off but the widths do not match B*N"

    # -- --refiner_precision, on the OBSERVED dtype --------------------------------- #
    # ⛔ ON CPU EVERY AUTOCAST ARM IS SILENTLY INERT.  `torch.autocast("cuda", ...)`
    # prints "CUDA is not available. Disabling" and carries on, so on CPU the observed
    # dtype is fp32 in EVERY policy -- which is exactly why the four CPU-only guard
    # tests cannot see a precision change, and why `test_gpu_refiner_numerics.py`
    # exists.  Record what was observed either way; assert the equality only where the
    # question is answerable.
    on_cuda = next(mod.parameters()).is_cuda
    want_rna = {"off": "float32", "bf16": "bfloat16", "fp16": "float16"}[
        args.rna_autocast]
    liveness.set("refiner_precision_flag", args.refiner_precision)
    liveness.set("rna_stack_input_dtypes", sorted({c[2] for c in calls}))
    liveness.set("rna_gemm_dtype", gemm.get("rna"))
    liveness.set("autocast_observable", bool(on_cuda))
    if on_cuda:
        assert gemm.get("rna") == want_rna, (
            f"--refiner_precision={args.refiner_precision} implies rna_autocast="
            f"{args.rna_autocast} ({want_rna}) but the RNA stack's FFN GEMM ran in "
            f"{gemm.get('rna')}. torch.autocast is silently inert when it does not "
            f"wrap the op.")
    else:
        log(f"[liveness] ⚠️ CPU: torch.autocast('cuda') is INERT here, so "
            f"--refiner_precision={args.refiner_precision} is UNVERIFIABLE on this "
            f"device (observed rna GEMM {gemm.get('rna')}). This flag can only be "
            f"proven on a GPU.")

    adt = {}

    def ahook(_m, _a, o):
        adt["dtype"] = str((o[0] if isinstance(o, tuple) else o).dtype).split(".")[-1]

    h = mod.atac_refiner.blocks[0].mixer.Wqkv.register_forward_hook(ahook)
    try:
        with torch.no_grad():
            mod.atac_refiner.native_grad(at[:2], am[:2])
    finally:
        h.remove()
    want_atac = {"off": "float32", "bf16": "bfloat16", "fp16": "float16"}[
        args.atac_autocast]
    liveness.set("atac_stack_dtype", adt.get("dtype"))
    if on_cuda:
        assert adt.get("dtype") == want_atac, (
            f"--refiner_precision={args.refiner_precision} implies atac_autocast="
            f"{args.atac_autocast} ({want_atac}) but the ATAC stack computed in "
            f"{adt.get('dtype')}")

    # -- --atac_varlen: the path AND the dropout trap -------------------------------- #
    liveness.set("atac_varlen_flag", int(args.atac_varlen))
    if args.atac_varlen:
        assert on_cuda, (
            "--atac_varlen 1 needs a GPU: flash_attn's varlen kernel has no CPU "
            "implementation, so this would fail at the first forward rather than here. "
            "Use --atac_sdpa 1 --atac_varlen 0 for a CPU smoke.")
        from flash_attn.modules.mha import FlashSelfAttention
        kinds = [type(b.mixer.inner_attn).__name__ for b in mod.atac_refiner.blocks]
        flags = [bool(b.mixer.use_flash_attn) for b in mod.atac_refiner.blocks]
        drops = [float(b.mixer.inner_attn.drop.p) for b in mod.atac_refiner.blocks]
        liveness.set("atac_varlen_inner_attn", kinds)
        liveness.set("atac_varlen_attn_dropout", drops)
        assert all(isinstance(b.mixer.inner_attn, FlashSelfAttention)
                   for b in mod.atac_refiner.blocks) and all(flags), (
            f"--atac_varlen 1 but inner_attn={kinds} use_flash_attn={flags}: the flag "
            f"never reached create_block and the block is still on the padded path")
        assert all(p == 0.0 for p in drops), (
            f"--atac_varlen 1 left attention dropout at {drops}. create_block wires "
            f"FlashSelfAttention(attention_dropout=--dropout), while the shipped SDPA "
            f"path is hardwired to 0.0 -- this would be a REGULARIZATION change "
            f"disguised as a speed change.")
        log(f"[liveness] ATAC varlen LIVE: inner_attn={kinds} attn_dropout={drops}")
    else:
        assert all(not b.mixer.use_flash_attn for b in mod.atac_refiner.blocks), \
            "--atac_varlen 0 but a block is on flash_attn's varlen path"


# ------------------------------------------------------------------------------------ #
# The GradScaler: construction, and the liveness assert on the quantity it CONTROLS
# ------------------------------------------------------------------------------------ #

def build_grad_scaler(args, device, liveness: Liveness, log=print, world: int = 1):
    """The loss scaler for `--refiner_precision fp16`.

    WHY IT EXISTS AT ALL.  `ATACRefinerFM` has autocast="fp16" as its CLASS DEFAULT and
    has run that way in every job on this track, while NO GradScaler existed anywhere in
    this repository.  Measured against an all-fp32 reference on a real 512-cell
    production block: the ATAC refiner's gradient sits at relative L2 5.35e-01 at group
    cosine 0.845 -- roughly half the signal gone -- and 97.7-98.3 % of the attention
    input-projection's ACTIVATION gradient stream was being flushed to exactly zero.
    That is a PRE-EXISTING DEFECT this file now fixes, not a cost the fp16 decision
    introduces.  The attribution is three-way independent: the error falls monotonically
    with the loss scale (underflow) while bf16's does not move at all (rounding), and a
    fp32-forward/fp16-backward arm reproduces the WHOLE error.

    GROWTH IS OFF BY DEFAULT and that is a science choice.  A dynamic scale makes the 9
    arms execute different arithmetic, and `configs/diff_arms.py` asserts the arms differ
    in an exact number of keys.  Backoff stays live so a transient overflow costs one
    skipped step rather than the run; the scale is therefore monotone non-increasing and
    lands on a value that is recorded in the manifest.

    ⛔ DISABLED IS NOT A FALLBACK.  `torch.cuda.amp.GradScaler` self-disables without
    CUDA, so an fp16 policy on CPU would run UNSCALED -- which is precisely the defect
    above, wearing the fixed version's flag.  That combination is refused.
    """
    on_cuda = torch.device(device).type == "cuda"
    if args.grad_scaler and not on_cuda:
        raise AssertionError(
            f"--refiner_precision {args.refiner_precision} needs CUDA. On CPU "
            f"torch.autocast('cuda') is inert AND GradScaler self-disables, so this "
            f"would run the SHIPPED unscaled arithmetic while every log line claimed "
            f"fp16 -- the silently-inert-flag failure this project has shipped four of. "
            f"Use --refiner_precision fp32 (or shipped) for a CPU smoke.")
    # growth_interval must be a positive int even when growth is meant to be off, so
    # "off" is expressed as an interval the run cannot reach.
    gi = (int(args.grad_scaler_growth_interval)
          if args.grad_scaler_growth_interval > 0 else int(args.num_steps) + 1)
    scaler = torch.cuda.amp.GradScaler(
        init_scale=float(args.grad_scaler_init_scale), growth_factor=2.0,
        backoff_factor=0.5, growth_interval=gi, enabled=bool(args.grad_scaler))
    # ⛔ THE SAFE SCALE IS COUPLED TO THE EFFECTIVE BATCH, AND THE COUPLING IS MEASURED.
    # The loss is a MEAN, so at B_eff cells the per-micro-batch dL/dz -- and with it the
    # fp16 tensor an autocast `nn.Linear` materialises for `grad_weight` before it is
    # accumulated into the fp32 `p.grad` -- scales as 1/B_eff.  MEASURED: at the
    # production B_eff 512 the ceiling sits between 2^22 (clean on 4 real blocks) and
    # 2^24 (overflows); the 6-step smoke at B_eff 4 overflows at 2^20, i.e. 128x smaller
    # batch, ~7 stops lower ceiling, exactly as the 1/B_eff law predicts.  So the default
    # 2^20 is a number for B_eff 512 and nothing else.  A warning rather than an assert:
    # the runtime backoff handles a modest mismatch and the skip-fraction abort catches a
    # hopeless one, but a silent mismatch is what this project keeps shipping.
    b_eff = int(args.micro_batch) * int(args.accum_freq) * max(int(world), 1)
    if args.grad_scaler and b_eff != 512:
        implied = args.grad_scaler_init_scale * b_eff / 512.0
        log(f"[precision] \u26a0\ufe0f --grad_scaler_init_scale "
            f"{args.grad_scaler_init_scale:g} was MEASURED at B_eff 512; this run is "
            f"B_eff {b_eff} ({args.micro_batch} x {args.accum_freq} x world "
            f"{max(int(world), 1)}). The fp16 grad_weight ceiling scales as 1/B_eff, so "
            f"the equivalent scale here is ~{implied:g}. The scaler will back off if it "
            f"overflows (each backoff costs one step and is counted), and the run aborts "
            f"if it lands below --grad_scaler_min_scale "
            f"{args.grad_scaler_min_scale:g}.")
    if not args.grad_scaler and "fp16" in (args.rna_autocast, args.atac_autocast):
        log(f"[precision] \u26d4 WARNING --refiner_precision {args.refiner_precision} "
            f"runs fp16 with NO loss scale. That is the PRE-EXISTING DEFECT, kept "
            f"reproducible on purpose: measured 5.35e-01 relative gradient error at "
            f"group cosine 0.845 on the ATAC refiner against an all-fp32 reference, "
            f"i.e. ~half the ATAC gradient signal lost to underflow. Use "
            f"--refiner_precision fp16 to train; this arm is a REFERENCE.")
    liveness.set("grad_scaler_enabled", bool(args.grad_scaler))
    liveness.set("grad_scaler_init_scale", float(args.grad_scaler_init_scale))
    liveness.set("grad_scaler_growth_interval", gi)
    liveness.set("grad_scaler_growth_on", bool(args.grad_scaler_growth_interval > 0))
    liveness.set("grad_scaler_min_scale", float(args.grad_scaler_min_scale))
    log(f"[precision] --refiner_precision {args.refiner_precision} -> rna_autocast "
        f"{args.rna_autocast} / atac_autocast {args.atac_autocast} | GradScaler "
        f"{'ENABLED' if args.grad_scaler else 'disabled'} init_scale "
        f"{args.grad_scaler_init_scale:g} growth_interval {gi} "
        f"({'ON' if args.grad_scaler_growth_interval > 0 else 'OFF'}) floor "
        f"{args.grad_scaler_min_scale:g}")
    return scaler


def gradscaler_liveness(model, micro_batches, args, scaler, forward_fn, loss_fn, device,
                        liveness: Liveness, log=print) -> None:
    """PROVE the loss scale REACHES the graph -- on the injected feature gradient.

    ⛔ ASSERTED ON THE CONTROLLED QUANTITY, NOT A PROXY.  `scaler.get_scale()` is a
    configuration read; `scaler.is_enabled()` is a constructor argument echoed back.
    Neither would fail if `scaler.scale()` were never called, if the scaled tensor were
    dropped, or if the scale were applied somewhere that does not feed the fp16 stacks.
    What a GradScaler CONTROLS in a two-pass gradient cache is exactly one number: the
    magnitude of the cached dL/dfeature that pass 2 injects with
    `torch.autograd.backward(ts, gs)`.  So this runs the SAME micro-batch twice, once
    scaled and once not, and requires the ratio of that magnitude to be the scale
    EXACTLY (it is a single multiply into a linear operator, so an approximate match
    would itself be a finding).

    Two micro-batches, not `accum_freq` -- the property is per-leaf, and the cost is a
    real forward+backward at production widths.
    """
    if not scaler.is_enabled():
        liveness.set("grad_scaler_leaf_ratio", "n/a (scaler disabled)")
        return
    mod = model_of(model)
    mbs = micro_batches[:2]
    S = float(scaler.get_scale())
    assert S > 1.0, (
        f"the GradScaler is enabled but its scale is {S}: a scale of 1 is the unscaled "
        f"arithmetic this flag exists to replace")

    def leaf_absmax(sc):
        mod.zero_grad(set_to_none=True)
        torch.manual_seed(4321)
        probe: Dict[str, float] = {}
        grad_cache_two_pass_n(mbs, forward_fn, loss_fn, device, scaler=sc,
                              probe=probe, log=lambda *x: None)
        mod.zero_grad(set_to_none=True)
        return float(probe["leaf_grad_absmax"])

    a1 = leaf_absmax(None)
    a2 = leaf_absmax(scaler)
    ratio = a2 / max(a1, 1e-30)
    liveness.set("grad_scaler_scale_at_step0", S)
    liveness.set("grad_scaler_leaf_absmax_unscaled", f"{a1:.3e}")
    liveness.set("grad_scaler_leaf_absmax_scaled", f"{a2:.3e}")
    liveness.set("grad_scaler_leaf_ratio", f"{ratio:.6e}")
    log(f"[liveness] GradScaler reaches the CACHED FEATURE GRADIENT: ||dL/dz||_inf "
        f"{a1:.3e} -> {a2:.3e}, ratio {ratio:.6e} (scale {S:g})")
    assert abs(ratio / S - 1.0) < 1e-6, (
        f"the loss scale did NOT reach the cached feature gradient: the pass-1 leaf "
        f"gradient grew {ratio:.6e}x, not {S:g}x. Pass 2 injects that tensor into the "
        f"fp16 refiner stacks, so this is the ONLY place a scaler can act on a two-pass "
        f"cache -- an fp16 run with this broken is the UNSCALED arithmetic with a "
        f"scaler-shaped log line.")
    # The converse half: the scale must be big enough to matter.  At 2^16 fp16 is a WASH
    # with bf16 (measured 4.75e-03 vs 4.20e-03 on the ATAC refiner gradient); below 2^14
    # it is strictly worse.  A run that silently operates there is a run whose precision
    # policy has been voided.
    assert S >= args.grad_scaler_min_scale, (
        f"loss scale {S:g} is below the floor {args.grad_scaler_min_scale:g}")



def gradcache_equivalence(model, micro_batches, forward_fn, loss_fn, device, log=print,
                          tol=None, min_ratio=100.0, scaler=None):
    """L8, in BOTH directions: the two-pass must match the single pass, and must FAIL
    without the RNG replay.

    The negative control is the load-bearing half.  With `projection_dropout 0.1` and
    `--dropout 0.2` live in the gradient-carrying path, a two-pass that does not replay
    the RNG backpropagates a cached dL/dz that belongs to a DIFFERENT function than the
    graph it flows through.  The reference test measures that failure at 231 % relative
    gradient error; a run with it would not crash and would not show in the loss curve.

    TOLERANCE.  On CPU (fp32 throughout) the two paths agree to 0.000e+00 and 1e-4 is
    generous.  On CUDA they cannot agree exactly and it is not a bug: the ATAC refiner
    runs under fp16 autocast and the two paths reduce in a different ORDER (one graph
    over the whole batch versus `accum` cached-gradient backwards), so ~1e-4 relative is
    floating-point, not a defect -- MEASURED 2.5e-04 here while the pass-1/pass-2
    feature replay was still exactly 0.000e+00.  That is why the verdict is the RATIO to
    the negative control (measured 2.9e-01, i.e. >1000x) rather than an absolute cut:
    an absolute threshold tight enough to be meaningful on CPU is a false alarm on GPU,
    and one loose enough for GPU proves nothing on its own.
    """
    if tol is None:
        tol = 5e-3 if torch.device(device).type == "cuda" else 1e-4
    mod = model_of(model)

    def grads_after(fn):
        mod.zero_grad(set_to_none=True)
        torch.manual_seed(1234)
        fn()
        return {n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in mod.named_parameters()}

    # ⛔ THE SAME SCALE ON ALL THREE PATHS.  `rel()` normalises by max|a|, so a COMMON
    # loss scale cancels EXACTLY and the verdict is unchanged -- while the code path
    # under test becomes the one that ships, scale included.  Scaling only one side would
    # make the comparison meaningless.
    #
    # ⛔ BUT THE PROBE'S SAFE SCALE IS NOT THE STEP'S, AND THAT IS A REAL PROPERTY, NOT A
    # WORKAROUND.  This probe runs on `--gradcache_equiv_micro` micro-batches (3 of 64 in
    # production), because `single_pass_step` keeps every micro-batch's graph alive and
    # the full step would OOM.  The loss is a MEAN, so at 24 cells instead of 512 each
    # per-micro-batch dL/dz -- and therefore the fp16 tensor an autocast `nn.Linear`
    # materialises for `grad_weight` before it is accumulated into the fp32 `p.grad` --
    # is ~21x LARGER than in the real step.  MEASURED: the ceiling at the production
    # B_eff 512 is between 2^22 (clean, 4 blocks) and 2^24 (overflows), and the 6-step
    # GPU smoke at B_eff 4 overflows at 2^20.  So the scale is chosen by BACKING OFF
    # until every path is finite -- exactly what a GradScaler does at run time -- and the
    # scale actually used is PRINTED, never silently substituted.
    def run_at(S):
        sc = (None if S <= 1.0 else
              torch.cuda.amp.GradScaler(init_scale=float(S), growth_interval=10 ** 9,
                                        enabled=(torch.device(device).type == "cuda")))
        gs = grads_after(lambda: single_pass_step(micro_batches, forward_fn, loss_fn,
                                                  scaler=sc))
        gc = grads_after(lambda: grad_cache_two_pass_n(
            micro_batches, forward_fn, loss_fn, device, verify=True, log=log, scaler=sc))
        bad = sum(1 for d in (gs, gc) for v in d.values()
                  if v is not None and not bool(torch.isfinite(v).all()))
        return gs, gc, bad

    S = float(scaler.get_scale()) if (scaler is not None and scaler.is_enabled()) else 1.0
    S0, tries = S, 0
    while True:
        g_single, g_cache, bad = run_at(S)
        if not bad or S <= 1.0 or tries >= 24:
            break
        S, tries = S / 2.0, tries + 1
    assert not bad, (
        f"the equivalence probe produced NON-FINITE gradients at EVERY loss scale down "
        f"to {S:g}. That is not an overflow, it is a defect in the model or the data.")
    if S != S0:
        log(f"  [gradcache] loss scale for the probe BACKED OFF {S0:g} -> {S:g} "
            f"({tries} halvings). NOT A DEFECT: this probe takes the loss over only "
            f"{len(micro_batches)} micro-batches (`--gradcache_equiv_micro`), because "
            f"the single-pass reference it compares against keeps every micro-batch's "
            f"graph alive. The loss is a MEAN, so at this batch its dL/dz -- and with it "
            f"the fp16 grad_weight tensor that overflows -- is larger than the real "
            f"step's in proportion. THE TRAINING STEP STILL RUNS AT {S0:g}; only this "
            f"probe was rescaled, and its verdict is scale-invariant because a COMMON "
            f"scale cancels exactly in `rel()`.")
    elif S > 1.0:
        log(f"  [gradcache] probe run at the production loss scale {S:g}, 0 non-finite")
    g_noreplay = grads_after(lambda: grad_cache_two_pass_n(
        micro_batches, forward_fn, loss_fn, device, rng_replay=False, log=log,
        scaler=(None if S <= 1.0 else torch.cuda.amp.GradScaler(
            init_scale=float(S), growth_interval=10 ** 9,
            enabled=(torch.device(device).type == "cuda")))))
    liveness_scale = S

    def rel(a, b):
        worst, scale = 0.0, 0.0
        for n in a:
            if a[n] is None or b.get(n) is None:
                continue
            worst = max(worst, float((a[n] - b[n]).abs().max()))
            scale = max(scale, float(a[n].abs().max()))
        return worst / max(scale, 1e-12)

    ok = rel(g_single, g_cache)
    bad = rel(g_single, g_noreplay)
    assert math.isfinite(ok) and math.isfinite(bad), (
        f"the equivalence comparison is non-finite (replay-on {ok}, replay-off {bad}) "
        f"at loss scale {S:g}")
    log(f"  [gradcache] single vs two-pass (replay ON) : max rel grad diff {ok:.3e}")
    log(f"  [gradcache] single vs two-pass (replay OFF): max rel grad diff {bad:.3e}"
        f"   <- NEGATIVE CONTROL, must be MUCH larger")
    assert ok < tol, (
        f"grad cache is NOT equivalent to the single pass ({ok:.3e} > tol {tol:g}); "
        f"every gradient this run would produce is wrong")
    assert bad > min_ratio * max(ok, 1e-12), (
        f"the RNG replay is not load-bearing (replay-off diff {bad:.3e} vs replay-on "
        f"{ok:.3e}, ratio {bad / max(ok, 1e-12):.1f}x < {min_ratio:g}x): the "
        f"equivalence check is VACUOUS and proves nothing")
    log(f"  [gradcache] replay-off / replay-on ratio = {bad / max(ok, 1e-12):.0f}x "
        f"(need >= {min_ratio:g}x)")
    mod.zero_grad(set_to_none=True)
    return ok, bad, liveness_scale


# ------------------------------------------------------------------------------------ #
# Checkpoints
# ------------------------------------------------------------------------------------ #

def save_snapshot(model, opt, sched, args, step: int, liveness: Liveness,
                  scaler=None, path: Optional[str] = None,
                  selection: Optional[Dict] = None) -> str:
    """`snapshots/step_%06d.pt`.

    ⛔ NO `best_model.pt` IS EVER WRITTEN BY THIS FILE.  `best_model.pt` on the projector
    track is mirrored from `best_ood_model.pt`, which is selected on bmmc + fetal_heart
    -- HALF the four-set OOD panel. Every absolute number taken from it inherits that
    leak. The arms are compared at a FIXED step, so the snapshot cadence IS the
    checkpoint policy; the step is in the filename AND recorded inside the file so a
    renamed file cannot pass silently.
    """
    d = os.path.join(args.save_dir, "snapshots")
    os.makedirs(d, exist_ok=True)
    path = path or os.path.join(d, f"step_{step:06d}.pt")
    torch.save({"step": int(step), "arm": args.arm,
                "num_slots": int(model_of(model).num_slots), "seed": int(args.seed),
                "args": vars(args), "git_sha": git_sha(),
                "liveness": {**liveness.values,
                             **{f"count:{k}": v for k, v in liveness.counts.items()}},
                "model": model_of(model).state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                # The loss scale is ARITHMETIC, not logging: a checkpoint whose scale
                # cannot be recovered cannot be resumed onto the same operating point,
                # and a reader cannot tell whether it was written under a backed-off
                # scaler.  None when no scaler was constructed (older callers).
                "grad_scaler": (scaler.state_dict() if scaler is not None else None),
                # ⛔ THE SELECTION STATE IS PART OF THE RUN, NOT OF THE LOGGING.  Without
                # it a --resume restarts `best_loss` at +inf and `best_r1` at -inf, so
                # the FIRST post-resume evaluation "improves" unconditionally and
                # OVERWRITES both best checkpoints with a model that is very probably
                # worse than the one it replaced -- silently, because the log line says
                # UPDATED either way.  It also resets both no-improve counters, so a run
                # resumed every few hours could never early stop.
                "selection": (dict(selection) if selection is not None else None)},
               path)
    return path


# ------------------------------------------------------------------------------------ #
# FM-layer initialisation (`--init_from_fm`, off the cache)
# ------------------------------------------------------------------------------------ #

def dump_fm_layer_state(args) -> None:
    """Write the FMs' TOP `n_layers` encoder-layer state dicts and exit.

    `make_refiners(..., init_from_fm=True)` needs a LIVE MultiOmicsCLIP, which does not
    exist on the cached-token track (that is the point of the cache).  Loading ~100 M +
    ~30 M parameters once, offline, to extract 2+2 layers keeps the canonical
    `--init_from_fm` recipe available without dragging the FMs into every training job.
    """
    assert args.rna_encoder_path and args.atac_encoder_path, \
        "--dump_fm_layer_state needs --rna_encoder_path and --atac_encoder_path"
    from scripts.train_filip_combined import build
    # `build()` reads refiner_type / n_layers / dropout / init_from_fm UNGUARDED
    # (train_filip_combined.py:273-274) and returns a THREE-tuple
    # (model, rna_ref, atac_ref) -- binding it to a single name and reading
    # `.rna_encoder` raises AttributeError on a tuple.  init_from_fm=False on purpose:
    # this function's whole job is to EXTRACT the FM's top layers so training never has
    # to load a 7 GB FM again; a refiner initialised from them is not what we want here.
    ns = argparse.Namespace(rna_encoder_path=args.rna_encoder_path,
                            atac_encoder_path=args.atac_encoder_path,
                            proj_dim=args.proj_dim,
                            max_atac_length=args.max_atac_length, fm_sdpa=0,
                            refiner_type="fm_layer", n_layers=args.n_layers,
                            dropout=args.dropout, init_from_fm=False, atac_sdpa=False)
    # ⛔ NOT torch.device("cpu"): scFoundation/model/load.py:150 ends
    # `return model.cuda(), config`, so this needs a GPU node whatever device is asked
    # for. Measured on the login node: RuntimeError: Found no NVIDIA driver.
    assert torch.cuda.is_available(), (
        "--dump_fm_layer_state needs a GPU node: scFoundation/model/load.py:150 ends "
        "`return model.cuda(), config`, unconditionally. Run it under srun/sbatch on "
        "drjieliu-h200 with --gres=gpu:1; it is a ~2 minute extraction, not training.")
    model, _rr, _ar = build(ns, torch.device("cuda"))
    rna_src = model.rna_encoder.model.encoder.transformer_encoder
    atac_src = model.atac_encoder.model.EpiAgent_transformer.layers
    n = args.n_layers
    def _cpu(sd):
        # The artifact is loaded with map_location="cpu" by init_from_fm_state, but a
        # CUDA-resident state dict would still pickle CUDA storages and refuse to load
        # on a CPU-only node (the self-test path). Move once, here.
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    torch.save({"n_layers": n,
                "rna": [_cpu(rna_src[len(rna_src) - n + i].state_dict())
                        for i in range(n)],
                "atac": [_cpu(atac_src[len(atac_src) - n + i].state_dict())
                         for i in range(n)]},
               args.dump_fm_layer_state)
    print(f"wrote {args.dump_fm_layer_state}")


def init_from_fm_state(model, path: str, liveness: Liveness) -> None:
    """Load those layers into the refiners, and PROVE the load changed the weights.

    A state-dict load that silently matches nothing is the classic silently-inert flag:
    `load_state_dict(strict=False)` returns happily, the run trains from random init,
    and nothing in the log says so. So a hash of the affected parameters is taken
    before and after, and asserted to DIFFER.
    """
    mod = model_of(model)
    blob = torch.load(path, map_location="cpu")
    assert int(blob["n_layers"]) == len(mod.rna_refiner.layers), \
        f"state has {blob['n_layers']} layers, refiner has " \
        f"{len(mod.rna_refiner.layers)}"

    def digest():
        h = hashlib.sha256()
        for p in mod.refiner_parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    before = digest()
    for i, sd in enumerate(blob["rna"]):
        mod.rna_refiner.layers[i].load_state_dict(sd)
    for i, sd in enumerate(blob["atac"]):
        mod.atac_refiner.blocks[i].load_state_dict(sd)
    after = digest()
    assert before != after, "init_from_fm_state changed nothing -- the load was a no-op"
    liveness.set("init_from_fm_state", os.path.basename(path))
    liveness.set("refiner_weights_sha256_before", before[:16])
    liveness.set("refiner_weights_sha256_after", after[:16])


# ------------------------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------------------------ #

def _r1_seed(*parts) -> int:
    """crc32 of the joined parts -- this project's standing pool-draw seed recipe."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


@torch.no_grad()
def val_retrieval_r1(zr, za, barcodes, args, center_by_group, block_idx: int):
    """Pool-`--val_r1_pool` R@1 on ONE val block's GLOBAL z vectors, BOTH directions.

    Scored on `zr`/`za` -- the two 256-d cell vectors `cell_infonce` is computed on --
    and on nothing else.  The slot tensors are deliberately not scored: slot routing is
    INERT (top1 ties top64 ties a uniform average) and after the refiner a cell's 64
    slots have pairwise cos 0.9891, so a slot scorer would be a noisier copy of this
    one, at 64x the arithmetic, for a selection decision.

    THE POOL DEFINITION IS THE METRIC.  One block is ONE dataset by construction
    (`SameDatasetBlockedBatchSampler.__iter__` draws one dataset per block), so a pool
    drawn inside a block is a DATASET-WINDOW pool -- the level the user requires
    retrieval to be reported at.  It is NOT the within-(dataset x cell_type) level of
    the registered OOD prediction, where the same cells score ~5x lower.  This number
    selects checkpoints; it is never quotable beside a panel number.

    Returns (r2a, a2r, dup_frac) or None when the block cannot fill one pool.
    """
    # DEDUPE FIRST.  The blocked sampler draws WITH replacement whenever the block's
    # dataset holds fewer than global_eff cells (the run prints that fraction as
    # `sampler_frac_replace_cells`), so a block can carry the same cell twice -- and two
    # identical rows make the "correct" column of the similarity matrix ambiguous, which
    # silently biases R@1.  The identity key is the loader's own per-cell id, never row
    # position.
    seen, keep = set(), []
    for i, b in enumerate(barcodes):
        if b not in seen:
            seen.add(b)
            keep.append(i)
    dup_frac = 1.0 - len(keep) / max(1, len(barcodes))
    P, D = int(args.val_r1_pool), int(args.val_r1_draws)
    if len(keep) < P:
        return None
    sel = torch.as_tensor(keep, device=zr.device, dtype=torch.long)
    ZR = zr.index_select(0, sel).float()
    ZA = za.index_select(0, sel).float()
    # IN-REGIME, AND CENTRE BEFORE NORMALISE.  A `--center_global_by_dataset 1` model
    # emits a near-constant vector plus a tiny variation; L2-normalising that WITHOUT
    # centring first collapses every cell onto the same point (measured
    # ||z_i - z_j|| = 0.000e+00).  `_center_by_group(x, None)` is the SAME staticmethod
    # `loss_fn` calls, with the same `groups=None` (one block == one dataset, so the
    # window mean IS the dataset mean), and it already ends in F.normalize; the
    # normalize below is idempotent there and is what covers the un-centred arm.
    if args.center_global_by_dataset:
        ZR, ZA = center_by_group(ZR, None), center_by_group(ZA, None)
    ZR, ZA = F.normalize(ZR, dim=-1), F.normalize(ZA, dim=-1)
    n = ZR.shape[0]
    # SEEDED ON THE BLOCK, NOT ON THE STEP OR THE RANK.  Every evaluation of every
    # checkpoint draws the IDENTICAL index pattern from the identical fixed val blocks
    # (--val_seed), so the step-to-step curve is PAIRED and the draw noise cancels --
    # the same reason --val_loss_seed is fixed and is not --seed.
    rng = np.random.RandomState(_r1_seed(args.val_seed, block_idx, P, D))
    J = torch.as_tensor(np.stack([rng.choice(n, P, replace=False) for _ in range(D)]),
                        device=ZR.device, dtype=torch.long)
    S = torch.bmm(ZR[J], ZA[J].transpose(1, 2))          # [D, P, P]
    lab = torch.arange(P, device=S.device).expand(D, P)
    # BOTH DIRECTIONS, KEPT SEPARATE.  Averaging them is a standing prohibition in this
    # project; the average that appears downstream is a SELECTION SCALAR under an
    # explicit flag, and both directions are written to the curve.
    r2a = float((S.argmax(2) == lab).float().mean())
    a2r = float((S.argmax(1) == lab).float().mean())
    return r2a, a2r, dup_frac


def compute_val_loss(model, args, world, liveness, device, ramp_step, loader, n_blocks):
    """Val loss under the arm's OWN objective, on FIXED blocks.  Used both by the
    eval-only `--val_loss_ckpt` mode and by the in-training `--val_every` hook.

    ⛔ RNG-SAFE.  Training and evaluation share the global torch RNG, so an eval that
    consumes from it shifts every subsequent dropout mask and the continuation stops
    being the run it would have been.  The CPU and CUDA states are saved and restored
    around the whole evaluation.
    ⛔ `ramp_step` MUST be the step whose weights are being scored, not 0: `make_loss_fn`
    reads it to compute `fine_ramp = min(1, step/fine_warmup_steps)`, and at 0 the fine
    term enters at 1/200 of its weight -- which silently turns ARM B's "val loss" into
    its global loss.  Measured once, the hard way.
    """
    import copy
    # ⛔ THIS FUNCTION IS COLLECTIVE.  `make_loss_fn` reaches GatherWithGrad /
    # `_gather_plain` whenever --gather_negatives is on and world > 1, and the tail
    # below all-reduces the scalars.  It MUST be entered by every rank.  A gloo
    # `monitored_barrier` is the tripwire: if a future edit re-guards the caller with
    # `rank == 0`, the ranks that DID arrive raise here in 120 s naming the absent ones,
    # instead of deadlocking until --ddp_timeout_min.  NCCL cannot do this -- hence the
    # separate gloo group built in `setup_distributed`.
    if _MONITOR_PG is not None:
        dist.monitored_barrier(group=_MONITOR_PG, timeout=timedelta(seconds=120),
                               wait_all_ranks=True)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    model.eval()
    fwd = make_forward_fn(model, args, liveness)
    lfn = make_loss_fn(model, args, world, [int(ramp_step)], liveness)
    # The SAME staticmethod `loss_fn` uses -- not a copy.  `_center_by_group` ends in
    # `F.normalize` and `_center_slots_by_group` does NOT, and a copy would eventually
    # "fix" that asymmetry and quietly delete the effect it carries.
    center_by_group, _center_slots_unused = _centering_helpers()
    per_block, comp = [], {}
    r1_r2a, r1_a2r, r1_ok, r1_dup = [], [], [], []
    try:
        it = iter(loader)
        with torch.no_grad():
            for _ in range(n_blocks):
                micro = [move_batch_to_device(next(it), device)
                         for _ in range(args.accum_freq)]
                outs, auxs = [], []
                for mb in micro:
                    o, a_ = fwd(mb); outs.append(o); auxs.append(a_)
                full = tuple(torch.cat([o[i] for o in outs], 0)
                             for i in range(len(outs[0])))
                aux = {}
                for k in auxs[0]:
                    vv = [a_[k] for a_ in auxs]
                    aux[k] = torch.cat(vv, 0) if torch.is_tensor(vv[0]) else vv[0]
                total, extra = lfn(full, aux)
                v = float(total.detach())
                assert math.isfinite(v), "non-finite val loss"
                per_block.append(v)
                # ⛔ SCORED ON `full`, THE PRE-GATHER, PRE-CENTRING OUTPUT, and centred
                # inside the scorer.  `lfn` centres (and, under --gather_negatives,
                # gathers) its own private copies; reading the metric off `full` keeps
                # the R@1 window exactly this rank's own block whether or not negatives
                # are gathered, so the number does not silently change meaning with a
                # loss flag.  It DOES depend on micro_batch x accum_freq, recorded below
                # as `window_per_rank`.
                if int(args.val_r1_draws) > 0:
                    bcs = [b for mb in micro for b in mb["barcode"]]
                    assert len(bcs) == int(full[0].shape[0]), (
                        f"val R@1 barcode/embedding misalignment: {len(bcs)} ids vs "
                        f"{int(full[0].shape[0])} rows. `full` is the concatenation of "
                        f"the micro-batch outputs in `micro` order; if that stops "
                        f"holding, every R@1 below is scored against the wrong cells.")
                    got = val_retrieval_r1(full[0], full[1], bcs, args,
                                           center_by_group, len(per_block) - 1)
                    # ⛔ FIXED-LENGTH ARRAYS WITH A VALIDITY MASK, never a conditional
                    # append.  `val_retrieval_r1` returns None when a block cannot fill
                    # one pool, and whether it does depends on THIS rank's own slice, so
                    # a conditional append would leave the ranks with DIFFERENT list
                    # lengths -- and the all_reduce below would then be called with
                    # mismatched shapes on different ranks, which does not raise: it
                    # hangs until the watchdog kills the job.
                    r1_r2a.append(got[0] if got is not None else 0.0)
                    r1_a2r.append(got[1] if got is not None else 0.0)
                    r1_ok.append(1.0 if got is not None else 0.0)
                    if got is not None:
                        r1_dup.append(got[2])
                for k, val in extra.items():
                    if isinstance(val, (int, float)):
                        comp.setdefault(k, []).append(float(val))
    finally:
        if was_training:
            model.train()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    # ⛔ REDUCE ACROSS RANKS, PER BLOCK, BEFORE ANY SELECTION ARITHMETIC.  Two reasons,
    # and the second is a hang:
    #   1. every rank holds a DISJOINT slice of the same global block, so the cross-rank
    #      mean of block b is that block's full micro_batch x accum_freq x world
    #      estimate, and the BLOCK stays the independent unit for the standard error.
    #   2. early stopping BREAKS the training loop.  If rank 3 computed a different
    #      val_loss from rank 0 -- which it does whenever --gather_negatives is 0,
    #      because then nothing in `loss_fn` is collective -- the ranks could disagree
    #      about whether to stop, and the ranks that kept going would sit in the next
    #      collective until the watchdog killed the job.  After this all_reduce every
    #      rank holds bit-identical numbers, so the stop decision is unanimous BY
    #      CONSTRUCTION and needs no separate broadcast.
    # ⚠️ At world > 1 this makes val_loss the mean OVER RANKS of the same fixed blocks;
    # at world == 1 it is a no-op and the number is byte-identical to today's.
    # `comp` keys are the same on every rank (same arm, same branches, same shapes), so
    # sorting them gives a rank-invariant packing order.
    if dist.is_initialized() and world > 1:
        nb, n_r1 = len(per_block), len(r1_r2a)
        keys = sorted(comp)
        flat = (list(per_block) + list(r1_r2a) + list(r1_a2r) + list(r1_ok)
                + [x for k in keys for x in comp[k]])
        t = torch.tensor(flat, device=device, dtype=torch.float64)
        dist.all_reduce(t)                       # SUM; the divisors differ per section
        vals = t.tolist()
        per_block = [x / world for x in vals[:nb]]
        r1_r2a = vals[nb:nb + n_r1]              # SUMS -- divided by r1_ok below
        r1_a2r = vals[nb + n_r1:nb + 2 * n_r1]
        r1_ok = vals[nb + 2 * n_r1:nb + 3 * n_r1]
        off = nb + 3 * n_r1
        for k in keys:
            comp[k] = [x / world for x in vals[off:off + nb]]
            off += nb
    # A block counts once it was scorable on at least one rank; its value is the mean
    # over the ranks that could score it, so a small dataset that only some ranks could
    # fill a pool from contributes an unbiased block estimate instead of a zero.
    ok_blocks = [(a, b, o) for a, b, o in zip(r1_r2a, r1_a2r, r1_ok) if o > 0]
    r1_r2a = [a / o for a, _, o in ok_blocks]
    r1_a2r = [b / o for _, b, o in ok_blocks]
    mean = sum(per_block) / max(1, len(per_block))
    r1: Dict = {"blocks": len(r1_r2a), "pool": int(args.val_r1_pool),
                "draws": int(args.val_r1_draws),
                "window_per_rank": int(args.micro_batch) * int(args.accum_freq),
                "level": "dataset_window",
                "centred": bool(args.center_global_by_dataset)}
    if r1_r2a:
        m_r2a = sum(r1_r2a) / len(r1_r2a)
        m_a2r = sum(r1_a2r) / len(r1_a2r)
        # ⛔ THE STANDARD ERROR IS ACROSS BLOCKS, NEVER ACROSS DRAWS.  The 200 draws
        # inside one block resample the SAME <= (micro x accum) cells, so a per-draw sem
        # understates the real uncertainty by roughly sqrt(draws) and would make every
        # noise fluctuation look significant to the early-stop rule.
        pick = {"mean": lambda a, b: 0.5 * (a + b),
                "r2a": lambda a, b: a, "a2r": lambda a, b: b}[args.val_r1_direction]
        per_block_sel = [pick(a, b) for a, b in zip(r1_r2a, r1_a2r)]
        sel_r1 = sum(per_block_sel) / len(per_block_sel)
        sd = ((sum((x - sel_r1) ** 2 for x in per_block_sel)
               / max(1, len(per_block_sel) - 1)) ** 0.5)
        r1.update({"r2a": round(m_r2a, 6), "a2r": round(m_a2r, 6),
                   "sel": sel_r1, "direction": args.val_r1_direction,
                   "per_block_r2a": [round(x, 6) for x in r1_r2a],
                   "per_block_a2r": [round(x, 6) for x in r1_a2r],
                   "sd_across_blocks": round(sd, 6),
                   "sem": round(sd / max(1, len(r1_r2a)) ** 0.5, 6),
                   # this rank's own value; the duplicate fraction is a per-rank-slice
                   # property and is reported as a tripwire, not as a statistic.
                   "dup_frac_max": round(max(r1_dup), 4) if r1_dup else 0.0,
                   "blocks_unscorable": int(len(r1_ok) - len(ok_blocks))})
    return mean, per_block, {k: sum(v) / len(v) for k, v in comp.items()}, r1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = resolve_args(argv)
    if args.dump_fm_layer_state:
        dump_fm_layer_state(args)
        return 0

    rank, world, device = setup_distributed(args)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    liveness = Liveness()
    liveness.set("seed", args.seed)
    liveness.set("world_size", world)
    liveness.set("device", str(device))
    liveness.set("lr_schedule_flag", args.lr_schedule)

    priors = load_priors(args, liveness)
    model = FineCLSRefiner(args, priors).to(device)
    if args.init_from_fm_state:
        init_from_fm_state(model, args.init_from_fm_state, liveness)
    sdpa_liveness(model, args, liveness)

    # ---------------------------------------------------------------------------- #
    # VAL LOSS (eval-only).  Selection criterion, per the user's 2026-08-23 decision.
    #
    # It reuses `make_forward_fn` / `make_loss_fn` -- the SAME factories the training
    # step uses -- so the number is the objective being minimised, not a second
    # implementation of it that could drift.  Everything below is under `no_grad`, so
    # nothing here can touch a weight.
    #
    # THE POOL IS FIXED at micro_batch * accum_freq, identical to training's B_eff.  The
    # InfoNCE floor is ln(pool), so a val loss is only comparable at a fixed pool -- in
    # this project a raw 1.58-nat gap between two runs turned out to be 0.19 once
    # ln(32) vs ln(128) was subtracted.
    #
    # THE VAL BLOCKS ARE FIXED (--val_loss_seed, NOT --seed): every checkpoint of every
    # arm sees the identical cells in the identical order.  That makes the comparison
    # PAIRED -- the sample noise is common and cancels -- which is what lets 40 blocks
    # resolve differences that would need far more if each checkpoint drew its own.
    # ---------------------------------------------------------------------------- #
    if args.val_loss_ckpt:
        assert "best" not in os.path.basename(args.val_loss_ckpt), (
            f"refusing {args.val_loss_ckpt}: a `best_*` checkpoint in THIS project was "
            "mirrored from best_ood_model.pt and selected on bmmc + fetal_heart, i.e. "
            "half the OOD panel. Point this at a numbered snapshot.")
        blob = torch.load(args.val_loss_ckpt, map_location="cpu")
        assert blob["arm"] == args.arm, (
            f"checkpoint arm {blob['arm']!r} != --arm {args.arm!r}: the val loss would "
            "be computed under the wrong objective")
        model_of(model).load_state_dict(blob["model"])
        model.eval()
        vds, vloader, vsampler = make_loader(
            args, args.val_split, world, rank,
            args.val_loss_blocks, args.val_loss_seed)
        # ⛔ THE RAMP MUST BE AT THE CHECKPOINT'S OWN STEP, NOT 0.  `make_loss_fn` reads
        # `step_ref[0]` to compute `fine_ramp = min(1, step/fine_warmup_steps)`.  With
        # step_ref=[0] the ramp is 1/200 = 0.005, so the fine term enters at 1/200 of its
        # weight and the "val loss" for ARM B/C is `global_total` in all but name --
        # MEASURED at step 8000: total 5.413417 = global_total 5.387075 + 0.005*fine.
        # That would make the selection criterion the WRONG objective for exactly the two
        # arms the experiment is about.
        step_ref_v = [int(blob["step"])]
        fwd = make_forward_fn(model, args, liveness)
        lfn = make_loss_fn(model, args, world, step_ref_v, liveness)
        vit = iter(vloader)
        tot, per_block, n_seen, comp = 0.0, [], 0, {}
        with torch.no_grad():
            for b in range(args.val_loss_blocks):
                micro = [move_batch_to_device(next(vit), device)
                         for _ in range(args.accum_freq)]
                outs_all, aux_all = [], []
                for mb in micro:
                    o, a_ = fwd(mb)
                    outs_all.append(o); aux_all.append(a_)
                full = tuple(torch.cat([o[i] for o in outs_all], 0)
                             for i in range(len(outs_all[0])))
                aux_full = {}
                for k in aux_all[0]:
                    vv = [a_[k] for a_ in aux_all]
                    aux_full[k] = (torch.cat(vv, 0) if torch.is_tensor(vv[0]) else vv[0])
                total, extra = lfn(full, aux_full)
                v = float(total.detach())
                assert math.isfinite(v), f"non-finite val loss at block {b}"
                # ⛔ `extra` IS THE POINT, not a debug extra -- discarding it was a real
                # defect.  `total` is each arm's OWN objective (ARM A: cell_infonce +
                # supcon + align; ARM B/C: that PLUS the fine term), so `total` is
                # comparable WITHIN an arm and NOT across arms.  `extra["global_total"]`
                # is the SAME three terms in every arm, so it IS cross-arm comparable --
                # it is the only val number that can be put in a table with A, B and C
                # side by side.  And `extra["fine"]` is comparable between B and C, which
                # is the biology-vs-random contrast.  Keeping only `total` threw both away.
                per_block.append(v); tot += v
                for k, val in extra.items():
                    if isinstance(val, (int, float)):
                        comp.setdefault(k, []).append(float(val))
                n_seen += int(full[0].shape[0])
        mean = tot / max(1, len(per_block))
        sd = (sum((x - mean) ** 2 for x in per_block) / max(1, len(per_block) - 1)) ** 0.5
        pool = args.micro_batch * args.accum_freq * world
        out = {"ckpt": os.path.abspath(args.val_loss_ckpt), "step": int(blob["step"]),
               "arm": blob["arm"], "seed": int(blob.get("seed", -1)),
               "val_split": args.val_split, "val_loss": round(mean, 6),
               "val_loss_sd_across_blocks": round(sd, 6),
               "val_loss_sem": round(sd / max(1, len(per_block)) ** 0.5, 6),
               # PER-BLOCK VALUES ARE THE POINT, not a debug extra.  The 40 blocks are
               # IDENTICAL across checkpoints (--val_loss_seed), so checkpoint-to-
               # checkpoint selection is a PAIRED comparison and its standard error is
               # sd(per-block DIFFERENCES)/sqrt(40) -- which is far smaller than the
               # sem of either mean.  Measured on ARM A: the last five checkpoints span
               # 0.0137 against a per-mean sem of 0.090, so WITHOUT the per-block values
               # an argmin over them is indistinguishable from noise and the selection
               # is not defensible.  Keeping only mean+sd threw that away.
               "per_block": [round(x, 6) for x in per_block],
               # per-component means + their per-block series, so a PAIRED test can be
               # run on `global_total` (cross-arm) and on `fine` (B vs C) independently
               # of the arm's own `total`.
               "components": {k: round(sum(v) / len(v), 6) for k, v in comp.items()},
               "components_per_block": {k: [round(x, 6) for x in v]
                                        for k, v in comp.items()},
               "blocks": len(per_block), "cells_scored": n_seen,
               "pool": pool, "ln_pool": round(math.log(pool), 6),
               "val_loss_minus_ln_pool": round(mean - math.log(pool), 6),
               "val_loss_seed": args.val_loss_seed}
        print("VAL_LOSS_JSON " + json.dumps(out, sort_keys=True))
        return 0

    start_step, resume_blob = 0, None
    if args.resume:
        assert "best" not in os.path.basename(args.resume), (
            f"refusing {args.resume}: `best_*` in this project was selected on half the "
            "OOD panel. Resume from a numbered snapshot.")
        resume_blob = torch.load(args.resume, map_location="cpu")
        assert resume_blob["arm"] == args.arm, (
            f"resume arm {resume_blob['arm']!r} != --arm {args.arm!r}")
        assert int(resume_blob["seed"]) == int(args.seed), (
            f"resume seed {resume_blob['seed']} != --seed {args.seed}")
        start_step = int(resume_blob["step"])
        assert start_step < args.num_steps, (
            f"--resume is at step {start_step} but --num_steps is {args.num_steps}: "
            "nothing to do")
        liveness.set("resume_from", os.path.abspath(args.resume))
        liveness.set("resume_start_step", start_step)
    # See the --resume docstring: a fresh seed is a CONTINUATION for a memoryless
    # sampler; reusing args.seed would replay the identical blocks.
    loader_seed = args.seed if not args.resume else (args.seed * 1000003 + start_step)
    liveness.set("loader_seed", int(loader_seed))
    ds, loader, sampler = make_loader(args, args.train_split, world, rank,
                                      args.num_steps - start_step, loader_seed)
    liveness.set("sampler_frac_replace_cells", round(float(sampler.frac_replace), 4))
    if args.live_fm:
        # The pipe is built inside make_loader (before `liveness` exists), so attach the
        # recorder now -- `LiveFMPipe._check` runs on the FIRST micro-batch, which the
        # step-0 probe pulls further down.
        _pipe = get_live_pipe(args, liveness)
        _pipe.liveness = liveness
        liveness.set("live_fm", True)
        liveness.set("live_corpus_source",
                     os.path.abspath(args.live_paired_dir or args.corpus_manifest))
        liveness.set("live_cohorts", args.corpus_cohorts)
        liveness.set("live_files", int(len(ds.files)))
        liveness.set("live_train_cells", int(len(ds)))
        liveness.set("live_label_coverage", round(float(ds.label_coverage), 4))
        liveness.set("live_label_classes", int(ds.n_classes))
        liveness.set("live_pairing_verified",
                     f"{ds.n_pairing_checked}/{len(ds.files)}")
        liveness.set("live_rna_nnz_filter", LIVE_RNA_NNZ_FILTER_NOTE)
        liveness.set("fm_sdpa", bool(args.fm_sdpa))
        liveness.set("fm_autocast_rna", bool(args.fm_autocast_rna))
    val_loader = None
    if args.val_every > 0:
        # FIXED blocks, own seed: every evaluation scores the identical val cells in the
        # identical order, so the curve is PAIRED -- checkpoint-to-checkpoint differences
        # are the model, not the sample.  That is what lets 40 blocks resolve a gap the
        # per-mean sem (0.09) never could.
        _, val_loader, _ = make_loader(args, args.val_split, world, rank,
                                       args.val_blocks, args.val_seed)
        for k, v in (("val_every", args.val_every), ("val_blocks", args.val_blocks),
                     ("val_seed", args.val_seed)):
            liveness.set(k, int(v))
    liveness.set("sampler_global_eff", int(sampler.global_eff))
    liveness.set("sampler_n_datasets", int(len(sampler.dataset_names)))
    # L-PREFETCH. Asserted on the BUFFER DEPTH, not on the flag: the failure mode is a
    # flag argparse accepted that still leaves the pull serial. Depth >= accum_freq is
    # the condition under which the reader is fully hidden behind the previous step.
    depth = int(getattr(loader, "_buffer_depth", 0))
    liveness.set("loader_prefetch_factor", int(resolve_prefetch_factor(args))
                 if args.num_workers > 0 else 0)
    liveness.set("loader_buffer_depth", depth)
    liveness.set("loader_buffer_covers_step", bool(depth >= args.accum_freq))
    if args.num_workers > 0 and depth < args.accum_freq:
        log(f"  [loader] WARNING buffer depth {depth} < accum_freq "
            f"{args.accum_freq}: {args.accum_freq - depth} micro-batches per step are "
            f"read ON THE CRITICAL PATH (measured ~2.3 s/step at depth 24). Raise "
            f"--prefetch_factor or --num_workers.")

    mod = model_of(model)
    opt = torch.optim.AdamW(
        [{"params": mod.head_parameters(), "lr": args.lr},
         {"params": mod.refiner_parameters(), "lr": args.fm_layer_lr}],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    sched = build_scheduler(opt, args)
    optimizer_liveness(opt, model, liveness)
    scaler = build_grad_scaler(args, device, liveness, log, world=world)
    if resume_blob is not None:
        model_of(model).load_state_dict(resume_blob["model"])
        opt.load_state_dict(resume_blob["optimizer"])
        sched.load_state_dict(resume_blob["scheduler"])
        if resume_blob.get("grad_scaler") is not None:
            scaler.load_state_dict(resume_blob["grad_scaler"])
        # The selection state is restored where `sel` is built, below, from this same
        # blob; it is reported HERE so a resume that inherited nothing is visible in the
        # log at the moment it happens rather than three hours later.
        _rs = resume_blob.get("selection")
        if _rs is None:
            log("  [resume] ⚠️ this snapshot carries NO selection state (written before "
                "the field existed): best_val_loss/best_val_r1 restart from scratch and "
                "the first evaluation WILL overwrite both best_* checkpoints.")
        else:
            log(f"  [resume] selection restored: best_loss {_rs['best_loss']:.6f} @ "
                f"{_rs['best_loss_step']} | best_r1 {_rs['best_r1']:.6f} @ "
                f"{_rs['best_r1_step']} | stalls loss={_rs['stall_loss']} "
                f"r1={_rs['stall_r1']} over {_rs['n_evals']} evals")
        log(f"RESUMED from {args.resume} at step {start_step}; "
            f"lr={sched.get_last_lr()[0]:.3e} scale={scaler.get_scale():.0f}")

    os.makedirs(args.save_dir, exist_ok=True)
    man_path = os.path.join(args.save_dir, "run_manifest.json")
    step_ref = [0]
    model.train()

    # ---- everything that must be PROVEN before the first optimizer step -------------
    # Run the probes on the UNWRAPPED model, BEFORE DDP wrapping. The probes do extra
    # forward/backward passes on rank 0 only; once the reducer's autograd hooks are
    # installed those would enqueue all-reduces the other ranks never make, and the job
    # would hang -- a liveness check that deadlocks the run it is protecting.
    it = iter(loader)
    probe_micro = [move_batch_to_device(next(it), device)
                   for _ in range(args.accum_freq)]
    probe_forward = make_forward_fn(model, args, liveness)
    probe_loss = make_loss_fn(model, args, world, step_ref, liveness)
    # ⛔ THESE PROBES ARE COLLECTIVE.  `gradscaler_liveness` and `gradcache_equivalence` are
    # handed `probe_loss = make_loss_fn(...)`, whose body reaches GatherWithGrad / _gather_plain
    # -> dist.all_gather whenever `--gather_negatives 1 and world > 1`.  Running them under
    # `if rank == 0:` made rank 0 enter an all_gather no other rank ever joins, and the job
    # deadlocked on its FIRST NCCL op -- rank0 ALLGATHER(4096) against rank1's barrier
    # ALLREDUCE(1) -- burning the full watchdog timeout (jobs 15285710/17/18/24, up to 2 h each).
    # This is the SAME FAMILY as the val-hook bug: a collective inside a rank-0 guard.
    # Every rank runs the probes; only rank 0 writes the manifest and logs.
    refiner_speed_liveness(model, probe_micro, args, liveness, log)
    gradscaler_liveness(model, probe_micro, args, scaler, probe_forward, probe_loss,
                        device, liveness, log)
    arm_liveness(model, probe_micro, probe_forward, args, liveness, log)
    if True:
        if args.check_gradcache_equiv:
            # ⛔ NOT `probe_micro` whole.  `single_pass_step` -- the reference the probe
            # compares against -- keeps EVERY micro-batch's graph alive simultaneously,
            # which is precisely the footprint gradient caching exists to avoid.
            # MEASURED on an H200 (ARM B, real priors, real cache lengths, S_atac 8192,
            # micro_batch 32): PEAK 18.92 / 28.07 / 42.41 / 59.16 GiB at 1/2/3/4
            # micro-batches -> +13.41 GiB each -> accum_freq 16 extrapolates to 220 GiB
            # on a 139.8 GiB card, i.e. CUDA-OOM before step 0, in every arm.
            k = max(1, min(int(args.gradcache_equiv_micro), len(probe_micro)))
            # ⛔ AND at ramp 1.0.  `loss_fn` scales the fine term by
            # (step+1)/fine_warmup_steps, so probing at step 0 with the committed
            # warmup 200 enters the fine term at 0.005 -- and `rel()` normalises by the
            # max |grad| over ALL parameters, which the (unramped) heads dominate. A
            # slot-branch-only replay defect is therefore attenuated 200x before the
            # comparison: MEASURED 5.008e-04 at warmup 200 against the CUDA tolerance
            # 5e-3, i.e. the assert would PASS on a run whose entire slot replay was
            # broken. The probe is a property check, not a training step, so it is run
            # at the weight the fine term will actually carry.
            saved_step = step_ref[0]
            step_ref[0] = max(0, int(args.fine_warmup_steps) - 1)
            try:
                ok, bad, eq_scale = gradcache_equivalence(
                    model, probe_micro[:k], probe_forward, probe_loss, device, log,
                    scaler=scaler)
                liveness.set("gradcache_equiv_loss_scale", float(eq_scale))
            finally:
                step_ref[0] = saved_step
            liveness.set("gradcache_equiv_micro_batches", k)
            liveness.set("gradcache_equiv_fine_ramp", 1.0 if args.arm == "finecls"
                         else "n/a (no fine term)")
            liveness.set("gradcache_rel_grad_diff_replay_on", f"{ok:.3e}")
            liveness.set("gradcache_rel_grad_diff_replay_off", f"{bad:.3e}")
    if rank == 0:
        man = write_manifest(args, mod, liveness, ds, world, man_path)
        log(liveness.report())
        log(f"[manifest] {man_path}")
        log(f"[params] total {man['params_total']:,} | trainable "
            f"{man['params_trainable']:,} | refiner {man['params_refiner']:,} | "
            f"slot branch {man['params_slot_branch']:,}")
    param_before = dict(mod.named_parameters())[RNA_PROBE_PARAM].detach().clone()
    center_slots_before = liveness.counts.get("center_slots_calls", 0)

    ddp_mods: Tuple = ()
    if args.distributed and world > 1:
        # ⛔ HANG (a): `broadcast_buffers` DEFAULTS TO TRUE and DDP's sync iterates
        # `module.buffers()`, which does NOT skip `persistent=False`.  The finecls arm
        # registers `rna_prior_mass` / `atac_prior_mass` and FixedSlotPooler registers
        # `prior`, so every DDP forward opened with a rank-0 -> all broadcast.  The val
        # hook then ran a DDP forward on rank 0 ALONE, rank 0 entered that broadcast,
        # the other ranks never arrived, and the job sat there until the NCCL watchdog
        # fired.  Note this is ARM-SPECIFIC: arm `cell_only` registers no buffers and
        # would never have shown it.  Dropping the sync is lossless here and that is
        # PROVEN, not asserted:
        _assert_buffers_identical_across_ranks(model, device, world, liveness, log)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False, broadcast_buffers=False)
        ddp_mods = (model,)
    forward_fn = make_forward_fn(model, args, liveness)
    loss_fn = make_loss_fn(model, args, world, step_ref, liveness)

    # FINDING 7 -- joint clipping COUPLES the arms. `clip_grad_norm_(model.parameters)`
    # rescales EVERY gradient by grad_clip/||g||, so whenever the clip is active the
    # global branch of an arm with a larger total norm takes a proportionally SMALLER
    # step than the same branch in an arm with a smaller one -- a third confound on B-A
    # (after capacity and the second loss term), acting in the SAME direction as the
    # registered negative prediction.  The design proposed "the fraction of steps where
    # the clip is active" as the diagnostic; measured, that fraction is 100 % in BOTH
    # arms, so it cannot see this. The statistic that can is the RATIO of pre-clip norm
    # between arms, which requires the mean to be RECORDED per arm, not merely printed
    # every --log_steps.  B - C is immune (both arms carry the fine branch).
    gnorm_sum, gnorm_n, clip_hits, slot_gnorm_sum = 0.0, 0, 0, 0.0
    # A skipped step is an APPLIED-UPDATE COUNT difference between arms, and the arms are
    # compared at a FIXED step -- so "step 8000" would mean different numbers of updates.
    # Counted, recorded and reported rather than absorbed.  `scale_min` is the audit
    # trail for the backoff ratchet: growth is off, so the scale is monotone
    # non-increasing and its minimum IS its final value.
    n_skipped, n_applied, scale_min = 0, 0, float(scaler.get_scale())
    # ⛔ THE BEST CHECKPOINT IS DECIDED BY VAL LOSS, never by the OOD panel.  This
    # project's `best_model.pt` was mirrored from `best_ood_model.pt` and selected on
    # bmmc + fetal_heart -- HALF the OOD panel -- so every absolute number off it
    # inherits that leak.  Val is held-out but IN-DISTRIBUTION and cannot leak the panel.
    # TWO CHECKPOINTS, TWO CRITERIA, TWO COUNTERS -- and all of it RESUME-CARRYING.
    # `best_by_valloss.pt` is the val_loss argmin and `best_by_valr1.pt` the val R@1
    # max; they are different models whenever the two criteria disagree, which is the
    # case the user asked to stop losing.  Neither is `best_model.pt`: that name on this
    # project was mirrored from an OOD-selected checkpoint and is still refused by
    # --resume / --val_loss_ckpt.
    sel = {"best_loss": float("inf"), "best_loss_step": -1,
           "best_r1": float("-inf"), "best_r1_step": -1,
           "stall_loss": 0, "stall_r1": 0, "n_evals": 0}
    if resume_blob is not None and resume_blob.get("selection"):
        sel.update({k: resume_blob["selection"][k] for k in list(sel)
                    if k in resume_blob["selection"]})
    stop_reason = None
    first_applied_checked = False
    t0 = time.time()
    for step in range(start_step, args.num_steps):
        step_ref[0] = step
        # step 0 reuses the block the probes already read, so the sampler's blocking
        # contract is not disturbed by having probed.
        if step == 0:
            micro = probe_micro
            # Drop the extra reference: at micro_batch 32 / S_atac 8192 one block is
            # ~5.2 GiB of resident fp16 tokens, and holding it for the whole run would
            # be a permanent tax paid for a step-0 probe.
            probe_micro = None
        else:
            micro = [move_batch_to_device(next(it), device)
                     for _ in range(args.accum_freq)]

        opt.zero_grad(set_to_none=True)
        verify = bool(args.grad_cache_verify) and step == 0
        if args.grad_cache:
            total, stats = grad_cache_two_pass_n(micro, forward_fn, loss_fn, device,
                                                 ddp_mods=ddp_mods, verify=verify,
                                                 log=log, scaler=scaler)
        else:
            total, stats = single_pass_step(micro, forward_fn, loss_fn, ddp_mods,
                                            scaler=scaler)
        # ⛔ UNSCALE FIRST, AND BEFORE THE NORM READS -- NOT ONLY BEFORE THE CLIP.
        # `clip_grad_norm_` is scale-COVARIANT, so with the clip SATURATED (FINDING 7:
        # active on 100 % of steps in BOTH arms) it silently ABSORBS a missing unscale --
        # (S g) C / ||S g|| == g C / ||g||.  What it does NOT absorb is an unscale that
        # happens AFTER it, which is exactly what `scaler.step()` does if the clip runs
        # first: the applied gradient norm is then pinned to C/S = 6.1e-05 instead of
        # 4.0 with the DIRECTION perfectly preserved (measured cos 1.0000000000), and
        # AdamW renormalises that away to a 0.25 % change in the applied update.  Every
        # guard below stays green through it.  The safety net covers the harmless bug and
        # misses the dangerous one, and the dangerous one is the natural code order.
        # And the norms themselves must be read unscaled: FINDING 7's diagnostic is the
        # CROSS-ARM ratio of the pre-clip norm, and two arms' scalers need not hold the
        # same S.
        scaler.unscale_(opt)
        # Read the refiner's own grad norm BEFORE clipping: clip_grad_norm_ rescales in
        # place, so afterwards every group reports a throttled number.
        ref_norm = grad_norm(mod.refiner_parameters())
        slot_norm = grad_norm(mod.slot_parameters()) if mod.num_slots else 0.0
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        args.grad_clip))
        # A PUBLIC, EXACT overflow detector: `clip_grad_norm_` returns inf/nan iff some
        # gradient is non-finite, so no one has to read `scaler._per_optimizer_states`.
        # ⚠️ It then multiplies every gradient by C/inf, leaving a MIX of 0.0 and nan --
        # so `grad_norm()` returns nan and `nan > 0` is False.  The asserts below are
        # therefore gated on this flag; ungated, the FIRST recoverable overflow would
        # raise AssertionError and kill a 27-hour run.
        overflow = not math.isfinite(pre_clip)
        if overflow:
            n_skipped += 1
            log(f"step {step:6d} | NON-FINITE gradient (clip return {pre_clip}) -> "
                f"optimizer step SKIPPED, loss scale {scaler.get_scale():g} -> "
                f"{scaler.get_scale() * 0.5:g} | skipped {n_skipped}/{step + 1}")
            # With no scaler there is nothing to back off and nothing to recover: a
            # non-finite gradient in fp32/bf16 is a BUG, not a transient, and must not be
            # converted into a silently skipped step.
            assert scaler.is_enabled(), (
                f"step {step}: a non-finite gradient with NO loss scaler enabled "
                f"(--refiner_precision {args.refiner_precision}). Nothing here can "
                f"back off, so this is a defect in the model or the data, not an "
                f"overflow to be absorbed.")
            assert n_skipped <= max(1, int(args.max_overflow_skip_frac
                                           * args.num_steps)), (
                f"{n_skipped} of {step + 1} steps skipped for a non-finite gradient "
                f"(> --max_overflow_skip_frac {args.max_overflow_skip_frac:g}). The "
                f"loss scale is at a broken operating point; this arm would take "
                f"materially fewer updates than its siblings.")
        else:
            gnorm_sum += pre_clip
            gnorm_n += 1
            clip_hits += int(pre_clip > args.grad_clip)
            slot_gnorm_sum += slot_norm
            # Written into EVERY snapshot so `compare_arm_checkpoints.py` can tell a
            # trained fine branch from an allocated-but-never-updated one. A differing
            # trunk cannot: ARM B's slot-projection dropout consumes RNG ARM A does not,
            # so the trunks diverge at the same magnitude whether or not the fine term is
            # in the loss.
            liveness.set("slot_gnorm_last", round(slot_norm, 6))
            liveness.set("slot_gnorm_mean", round(slot_gnorm_sum / gnorm_n, 6))
            assert ref_norm > 0, (
                f"step {step}: the refiner grad norm is EXACTLY 0 -- the trunk receives "
                f"no gradient at all. This is the TRAP-1 signature (a no-grad path "
                f"feeding the branch); do not let this run continue.")
            # The converse guard, on the OPTIMIZED objective rather than on a probe.
            # `arm_liveness` proves the fine loss CAN reach the branch; this proves it IS
            # in the loss that was just backwarded. They are different claims: a run with
            # --fine_weight 0 passes the first and fails this one at step 0.
            if mod.num_slots and args.fine_weight > 0:
                assert slot_norm > 0, (
                    f"step {step}: arm=finecls but the slot-branch grad norm is EXACTLY "
                    f"0. The fine term is allocated and paid for but is not in the "
                    f"objective; every liveness probe would still be green.")
        # `scaler.step` re-checks found_inf itself and skips the WHOLE optimizer on an
        # overflow -- necessarily whole-step, because the accum_freq pass-2 backwards
        # accumulate into the SAME p.grad buffers, so one inf has already contaminated
        # the sum.  Measured: max|dp| = 0.000e+00 over all 56 tensors, scale halved, and
        # the next clean step applies normally.  With the scaler disabled it is
        # `opt.step()` verbatim.
        scaler.step(opt)
        scaler.update()
        scale_now = float(scaler.get_scale())
        scale_min = min(scale_min, scale_now)
        n_applied += int(not overflow)
        sched.step()
        assert scale_now >= args.grad_scaler_min_scale or not scaler.is_enabled(), (
            f"step {step}: the loss scale has backed off to {scale_now:g}, below the "
            f"floor {args.grad_scaler_min_scale:g}. MEASURED, fp16 below 2^14 is no "
            f"better than bf16 and the precision policy is void -- fail loudly rather "
            f"than quietly train a worse model.")

        # ⛔ THE FIRST *APPLIED* STEP, NOT STEP 0.  `scaler.step` skips the whole
        # optimizer on an overflow, so on a skipped step 0 the parameter is UNCHANGED for
        # a legitimate reason and this assert would kill the run for doing the right
        # thing.  The claim being tested -- "one applied optimizer step moves the trunk"
        # -- is unaffected by which step it is measured on.
        if (not first_applied_checked) and rank == 0 and not overflow:
            first_applied_checked = True
            liveness.set("param_delta_measured_at_step", step)
            # L1's second half: gradients flowing is NOT proof the arm trains.
            after = dict(mod.named_parameters())[RNA_PROBE_PARAM]
            delta = float((after - param_before).abs().max())
            assert delta > 0, (
                f"one opt.step() left {RNA_PROBE_PARAM} UNCHANGED (max|delta| "
                f"{delta}). Gradients flow but the parameter never moves -- the "
                f"`--fm_layer_lr 0` "
                f"broken-freeze signature. This arm would be inert.")
            liveness.set("param_delta_after_one_step", f"{delta:.3e}")
            log(f"[liveness] param_delta_after_one_step = {delta:.3e}")
            # L9, on the COUNTER rather than on the flag: `center_slots_by_dataset` is
            # meaningless without slots, so it must have executed in ARM B/C and must
            # NOT have executed in ARM A -- a positive number in one arm and a hard zero
            # in the other, both printed.
            n_cs = liveness.counts.get("center_slots_calls", 0) - center_slots_before
            if args.arm == "finecls" and args.center_slots_by_dataset:
                assert n_cs > 0, "center_slots_by_dataset never executed in ARM B/C"
            else:
                assert n_cs == 0, \
                    f"slot centring executed {n_cs} times with no slot branch"
            liveness.set("center_slots_calls_in_step0", n_cs)
            log(f"[liveness] center_slots_calls_in_step0 = {n_cs}")

        if rank == 0 and (step % args.log_steps == 0 or step == args.num_steps - 1):
            lr_now = sched.get_last_lr()[0]
            if step > args.warmup_steps and args.lr_schedule == "constant":
                assert abs(lr_now - args.lr) < 1e-12, (
                    f"--lr_schedule constant but the scheduler gave {lr_now} at step "
                    f"{step} (expected {args.lr}); the flag is not controlling the LR")
            terms = " ".join(f"{k}={v:.4f}" for k, v in sorted(stats.items()))
            log(f"step {step:6d} | loss {float(total):.4f} | lr {lr_now:.3e} | "
                f"gnorm {pre_clip:.3f} | refiner_gnorm {ref_norm:.4f} | "
                f"slot_gnorm {slot_norm:.4f} | clip "
                f"{'Y' if pre_clip > args.grad_clip else 'n'} "
                f"| scale {scale_now:g} skipped {n_skipped} "
                f"| {terms} | {time.time() - t0:.1f}s")

        # ⛔ HANG (b), AND WHY THE FIX IS *COLLECTIVE* RATHER THAN *GATHER-OFF*.
        # As shipped this block was guarded by `rank == 0`, so rank 0 alone walked into
        # compute_val_loss -> make_loss_fn -> GatherWithGrad / `_gather_plain`, i.e. a
        # one-rank all_gather: deadlock.  The other repair -- forcing --gather_negatives
        # off inside val -- was REJECTED because it silently changes the metric: the
        # InfoNCE floor is ln(pool), and the val pool would fall from
        # micro_batch*accum_freq*world to micro_batch*accum_freq, i.e. 512 -> 128 at
        # world 4, moving val_loss by ln(4) = 1.386 nats.  All 20 shipped runs are world
        # 1 with micro_batch 8 x accum_freq 64, i.e. pool 512 -- so running val
        # COLLECTIVELY at 4 x 128 keeps the identical floor and the curve stays
        # comparable to every one of them.  The pool is written into the jsonl so that
        # can be audited instead of trusted.
        # EVERY RANK COMPUTES; ONLY RANK 0 WRITES.
        if (val_loader is not None
                and ((step + 1) % args.val_every == 0 or step == args.num_steps - 1)):
            vmean, vblocks, vcomp, vr1 = compute_val_loss(
                model, args, world, liveness, device, step + 1, val_loader,
                args.val_blocks)
            val_pool = int(args.batch_size * world) if (args.gather_negatives
                                                        and world > 1) \
                else int(args.batch_size)
            sel["n_evals"] += 1
            r1_now = vr1.get("sel")
            # STRICTLY BEYOND THE MIN DELTA, both criteria.  An improvement smaller than
            # the delta is a NO-improvement and ADVANCES the counter; that is what makes
            # a plateau detectable at all.  Both booleans derive from all-reduced
            # scalars, so every rank computes the same ones and the counters stay
            # replicated -- which is what lets the `break` below be unanimous.
            imp_loss = vmean < sel["best_loss"] - args.early_stop_min_delta_loss
            imp_r1 = (r1_now is not None
                      and r1_now > sel["best_r1"] + args.early_stop_min_delta_r1)
            if imp_loss:
                sel["best_loss"], sel["best_loss_step"] = vmean, step + 1
                sel["stall_loss"] = 0
            else:
                sel["stall_loss"] += 1
            if r1_now is not None:
                if imp_r1:
                    sel["best_r1"], sel["best_r1_step"] = r1_now, step + 1
                    sel["stall_r1"] = 0
                else:
                    sel["stall_r1"] += 1
            if rank == 0:
                # Same file, same name as before; the new keys are ADDITIVE so an old
                # reader still parses every line it used to.
                with open(os.path.join(args.save_dir,
                                       "val_loss_curve.jsonl"), "a") as fh:
                    fh.write(json.dumps(
                        {"step": step + 1, "val_loss": round(vmean, 6),
                         # ln(pool) is the InfoNCE floor; a curve without its pool is
                         # not comparable to anything, and this project has already lost
                         # 1.58 nats to exactly that omission.
                         "pool": val_pool, "world": int(world),
                         "per_block": [round(x, 6) for x in vblocks],
                         "components": {k: round(v, 6) for k, v in vcomp.items()},
                         "val_r1": vr1,
                         "stall_loss": sel["stall_loss"],
                         "stall_r1": sel["stall_r1"]},
                        sort_keys=True) + "\n")
                if imp_loss:
                    save_snapshot(model, opt, sched, args, step + 1, liveness, scaler,
                                  path=os.path.join(args.save_dir,
                                                    "best_by_valloss.pt"),
                                  selection=sel)
                if imp_r1:
                    save_snapshot(model, opt, sched, args, step + 1, liveness, scaler,
                                  path=os.path.join(args.save_dir, "best_by_valr1.pt"),
                                  selection=sel)
            liveness.set("best_val_loss", round(sel["best_loss"], 6))
            liveness.set("best_val_step", int(sel["best_loss_step"]))
            liveness.set("best_val_r1", round(sel["best_r1"], 6)
                         if sel["best_r1"] > float("-inf") else "n/a")
            liveness.set("best_val_r1_step", int(sel["best_r1_step"]))
            liveness.set("val_r1_direction_used", args.val_r1_direction)
            liveness.set("val_pool", val_pool)
            liveness.set("val_ln_pool", round(math.log(val_pool), 6))
            liveness.set("val_hook_collective", True)
            log(f"[val] step {step+1} val {vmean:.6f} "
                f"global {vcomp.get('global_total', float('nan')):.6f} "
                f"fine {vcomp.get('fine', float('nan')):.6f} | pool {val_pool} | "
                f"r1 r2a {vr1.get('r2a', float('nan')):.4f} "
                f"a2r {vr1.get('a2r', float('nan')):.4f} "
                f"(sem {vr1.get('sem', float('nan')):.4f}, "
                f"{vr1.get('blocks', 0)} blocks) | "
                f"best loss {sel['best_loss']:.6f} @ {sel['best_loss_step']} "
                + ("UPDATED" if imp_loss else f"stall x{sel['stall_loss']}"))
            log(f"[val] step {step+1} best r1 {sel['best_r1']:.6f} @ "
                f"{sel['best_r1_step']} "
                + ("UPDATED" if imp_r1 else f"stall x{sel['stall_r1']}"))
            # ⛔ BOTH, NOT EITHER.  The run stops only when val_loss AND val R@1 have
            # each been flat for --early_stop_patience consecutive evaluations.
            if (args.early_stop_patience > 0
                    and (step + 1) >= args.early_stop_min_step
                    and sel["stall_loss"] >= args.early_stop_patience
                    and sel["stall_r1"] >= args.early_stop_patience):
                stop_reason = (f"early stop at step {step + 1}: val_loss flat for "
                               f"{sel['stall_loss']} evals (best {sel['best_loss']:.6f} "
                               f"@ {sel['best_loss_step']}) AND val R@1 flat for "
                               f"{sel['stall_r1']} evals (best {sel['best_r1']:.6f} @ "
                               f"{sel['best_r1_step']}), patience "
                               f"{args.early_stop_patience}")

        if rank == 0 and args.save_steps > 0 and (
                (step + 1) % args.save_steps == 0 or step == args.num_steps - 1):
            # ⛔ `selection=sel` ON THE NUMBERED SNAPSHOTS TOO -- these are the ONLY
            # checkpoints --resume accepts (both best_* files are refused by the "best"
            # guard), so a numbered snapshot without the selection state is exactly the
            # resume that restarts best_val at +inf and overwrites the best checkpoint.
            p = save_snapshot(model, opt, sched, args, step + 1, liveness, scaler,
                              selection=sel)
            log(f"[snapshot] {p}")
        if stop_reason is not None:
            # A numbered snapshot AT THE STOP STEP, unconditionally: the two best_* files
            # are refused by --resume and by --val_loss_ckpt, so without this a stopped
            # run may have no resumable artifact at its final step.  Every rank reaches
            # this line with the SAME `stop_reason` (it is derived from all-reduced
            # scalars), so all ranks leave the loop together and meet at the existing
            # dist.barrier() below.
            if rank == 0:
                p = save_snapshot(model, opt, sched, args, step + 1, liveness, scaler,
                                  selection=sel)
                log(f"[snapshot] {p}")
                log(f"[early-stop] {stop_reason}")
            break

    if rank == 0:
        liveness.set("pre_clip_gnorm_mean", round(gnorm_sum / max(gnorm_n, 1), 4))
        liveness.set("grad_clip_active_frac", round(clip_hits / max(gnorm_n, 1), 4))
        # The three numbers that make two arms comparable under a dynamic scale. Arms
        # that ended at DIFFERENT scales ran different arithmetic, and arms with
        # different `optimizer_steps_applied` did not take the same number of updates.
        # Both are reportable confounds, so both are recorded rather than assumed away.
        liveness.set("grad_scaler_scale_last", float(scaler.get_scale()))
        liveness.set("grad_scaler_scale_min", float(scale_min))
        liveness.set("optimizer_steps_applied", int(n_applied))
        # An early-stopped run took FEWER optimizer steps than --num_steps, which is the
        # same reportable confound as steps_skipped_overflow beside it: without this a
        # later reader compares two arms "at step 88000" when one of them stopped at
        # 26000.
        liveness.set("early_stop_patience", int(args.early_stop_patience))
        liveness.set("early_stopped", bool(stop_reason is not None))
        liveness.set("early_stop_reason", stop_reason or "n/a (ran to --num_steps)")
        liveness.set("val_evals", int(sel["n_evals"]))
        liveness.set("steps_skipped_overflow", int(n_skipped))
        write_manifest(args, mod, liveness, ds, world, man_path)
        log(liveness.report("[liveness-final]"))
        log(f"FINECLS_TRAIN_DONE arm={args.arm} steps={args.num_steps} "
            f"slots={mod.num_slots} seed={args.seed}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
