"""OpenCLIP-style gradient caching (Gao et al. 2021 "GradCache") for the FineCLS
refiner track -- generalized from TWO cached cell vectors to an ARBITRARY tuple of
cached feature tensors, so the fine branch's ``[B, M, 256]`` slot embeddings ride along
with the ``[B, 256]`` globals.

WHY A SEPARATE FILE. The production two-pass lives at
``haoyun/multiomics_clip_finelip/scripts/train_filip_combined.py:549``
(``grad_cache_two_pass``). It is correct and RNG-replay-guarded, but it hard-codes a
2-tuple: ``hr, ha = forward_fn(mb)`` at :573, ``leaves.append((hr..., ha...))`` at :576,
and BOTH the verify loop (:588) and the backward loop (:591) iterate the literal ``((hr,
leaves[i][0]), (ha, leaves[i][1]))``. ARM B returns FOUR grad-carrying tensors plus
non-grad ``valid``/``mass``/``labels``, so the tuple has to become an N-tuple. This
module is that generalization, kept out of the colleague's tree, and it is the version
the FineCLS trainer imports. ``grad_cache_two_pass`` here is a strict superset: handed a
2-tuple forward it reproduces the upstream behaviour, which ``test_gradcache_equiv.py``
checks directly against the upstream function (T0).

WHAT THE TWO PASSES ARE. Let ``f_theta`` be the per-micro-batch forward and
``L(F_1..F_n)`` the contrastive loss on the concatenated batch.

  pass 1   F_i = f_theta(x_i)  under ``torch.no_grad()``;  keep ONLY F_i, detached,
           as an autograd LEAF with ``requires_grad_(True)``.  No activation graph.
  loss     L is taken ONCE on ``cat(F_1..F_n)``; ``L.backward()`` fills ``leaf.grad``
           = dL/dF_i.  (OpenCLIP's ``--accum-freq`` instead RE-computes L once per
           micro-batch with the others spliced in detached; that is ``accum`` times the
           loss work and, without an RNG replay, ``accum`` different dropout draws.)
  pass 2   re-forward each micro-batch WITH grad and
           ``torch.autograd.backward(F'_i, leaf.grad)`` -> the chain rule finishes the
           job and ``theta.grad`` ends up exactly where a single full-batch
           forward+backward would have left it.

  Peak activation memory = ONE micro-batch. NO ``1/accum`` rescale anywhere: the loss
  was taken once, so dividing would silently scale the effective LR by 1/accum.

THE PRECONDITION, AND WHY ``verify`` MUST COVER THE SLOTS. ``leaf.grad`` is the gradient
of L *at the pass-1 value* F_i. Pass 2 must therefore recompute the SAME F_i. With
``projection_dropout 0.1`` and ``--dropout 0.2`` live in the gradient path that requires
replaying the RNG (``_rng_snapshot`` / ``_rng_restore``, GradCache's ``RandContext``).
Generalizing the tuple while leaving ``verify`` on the two globals turns the check into
a PROXY: it prints OK on a run whose SLOT branch is replaying different dropout masks,
i.e. every fine-branch gradient is wrong and nothing says so. ``verify_features`` exists
so the test can demonstrate exactly that failure in both directions (T5); production
must leave it at ``"all"``.

⛔ DO NOT copy the FineCLS splice loop (``training_mpnce/fixed_slot_trainer.py:886``)
instead. It is the literal OpenCLIP splice -- recompute the whole loss ``accum`` times
via ``cat([pools[:lo].detach(), current, pools[hi:].detach()])`` -- with NO RNG replay
(``grep -n 'rng\\|RandContext\\|get_rng_state'`` on that file returns nothing) while
``projection_dropout: 0.1`` is live.  T2/T4 here measure that failure mode.

⚠️ ``multiomics_clip/training/gradient_accumulation.py`` does NOT implement gradient
accumulation despite its docstring: 57 lines containing only ``GatherWithGrad``.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

# The fine branch's routing is the colleague's, unmodified: importing it (rather than
# re-deriving it) is what makes the loss below the SAME object the reference FineCLS
# runs optimize, so a null here is attributable to the track and not to a
# re-implementation.
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXED_SLOT_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "multiomics_clip_xinyu_June_fixed_slot_routing")
)
if _FIXED_SLOT_DIR not in sys.path:
    sys.path.insert(0, _FIXED_SLOT_DIR)
from modules.fixed_slot import fixed_slot_route_weights  # noqa: E402


# ------------------------------------------------------------------------------------ #
# What a forward pass hands back
# ------------------------------------------------------------------------------------ #


class GCOut(tuple):
    """``(features, aux)`` -- what ``forward_fn`` returns when it also has non-grad
    side data.

    ``features``: grad-carrying tensors that BECOME LEAVES.  Each is cached in pass 1,
                  receives dL/dF from the full-batch loss, and is re-forwarded and
                  backwarded in pass 2.  For ARM B:
                  ``(rna_global[b,256], atac_global[b,256],
                     rna_slots[b,M,256], atac_slots[b,M,256])``.
    ``aux``     : tensors the loss needs but that carry NO learnable dependence --
                  ``valid[b,M]`` (bool), ``mass[b,M]``, ``labels[b]``.  They are
                  concatenated for the loss and are NOT leaves.

    WHY aux IS NOT A LEAF, AND MUST NOT BE. ``mass[b,m] = sum_n prior[id_n, m] *
    valid_n`` is a pure function of TOKEN IDS with no learnable weights. Caching it as a
    leaf would invent a gradient path that does not exist; recomputing it in pass 2 is
    free and is what keeps the pass-2 graph identical to the pass-1 one. It is still
    VERIFIED (a silent drift in ``mass`` would invalidate the cached dL/dF just as
    surely as a dropout mismatch would).

    A bare ``tuple``/``list`` of tensors is also accepted by ``grad_cache_two_pass`` and
    is read as "all features, no aux" -- which is the upstream 2-tuple signature.
    """

    __slots__ = ()

    def __new__(cls, features: Sequence[torch.Tensor],
                aux: Sequence[torch.Tensor] = ()):
        return super().__new__(cls, (tuple(features), tuple(aux)))

    @property
    def features(self) -> Tuple[torch.Tensor, ...]:
        return self[0]

    @property
    def aux(self) -> Tuple[torch.Tensor, ...]:
        return self[1]


def _split(out) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    """Normalize whatever ``forward_fn`` returned into ``(features, aux)``."""
    if isinstance(out, GCOut):
        return out.features, out.aux
    if torch.is_tensor(out):
        return (out,), ()
    feats = tuple(out)
    assert all(torch.is_tensor(t) for t in feats), (
        "forward_fn must return a tensor, a tuple of tensors, or GCOut(features, aux); "
        f"got a {type(out).__name__} containing "
        f"{[type(t).__name__ for t in feats]}")
    return feats, ()


# ------------------------------------------------------------------------------------ #
# RNG replay (GradCache's RandContext)
# ------------------------------------------------------------------------------------ #


def _rng_snapshot(device):
    """``(cpu_state, cuda_state)`` captured BEFORE a micro-batch forward so pass 2
    replays the EXACT dropout masks. Without it the pass-2 graph is a DIFFERENT function
    than the
    one whose output the cached dL/dF was computed for -> silently wrong gradients."""
    dev = torch.device(device)
    cuda = (torch.cuda.get_rng_state(dev)
            if (dev.type == "cuda" and torch.cuda.is_available()) else None)
    return torch.get_rng_state(), cuda


def _rng_restore(state, device):
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, torch.device(device))


# ------------------------------------------------------------------------------------ #
# Cross-rank gathering -- shape-agnostic, so [b, M, 256] works unchanged
# ------------------------------------------------------------------------------------ #

# The FIXED GatherWithGrad (the one that all-reduces the incoming grad before slicing).
# ⛔ multiomics_clip/training/gradient_accumulation.py holds an OLDER copy WITHOUT that
# all_reduce; composed with DDP's mean-reduce it delivers 1/world_size of the true
# gradient -- a silent LR/W. Import the finelip one, by path, so no sys.path ordering
# accident can pick the wrong twin. Loaded BY FILE PATH, not as
# `multiomics_clip_finelip.training.gradient_accumulation`: that package's __init__
# imports the whole FineLIP trainer (and through it scanpy and
# multiomics_clip.evaluation), which is 20 s of import and a hard dependency this file
# does not have. The path also removes any chance of sys.path order picking the stale
# twin.
import importlib.util as _ilu  # noqa: E402

_GA_PATH = os.path.abspath(os.path.join(
    _HERE, "..", "..", "haoyun", "multiomics_clip_finelip", "training",
    "gradient_accumulation.py"))
_ga_spec = _ilu.spec_from_file_location("_finelip_gradient_accumulation", _GA_PATH)
_ga = _ilu.module_from_spec(_ga_spec)
_ga_spec.loader.exec_module(_ga)
GatherWithGrad = _ga.GatherWithGrad
assert "all_reduce" in open(_GA_PATH).read(), (
    f"{_GA_PATH} has no all_reduce in GatherWithGrad.backward -- that is the STALE "
    f"twin (multiomics_clip/training/gradient_accumulation.py).  Composed with DDP's "
    f"mean-reduce it delivers 1/world_size of the true gradient: a silent LR/W.")


def dist_on() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def gather_features(t: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
    """All-gather along dim 0.  ``with_grad=False`` for ``valid``/``mass``/``labels``:
    they carry no gradient, and routing them through ``GatherWithGrad`` would add an
    autograd node (and an all_reduce in its backward) for nothing."""
    if not dist_on():
        return t
    if with_grad:
        return GatherWithGrad.apply(t)
    parts = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, t.contiguous())
    return torch.cat(parts, 0)


# ------------------------------------------------------------------------------------ #
# The two-pass
# ------------------------------------------------------------------------------------ #


def chunk_bounds(n: int, n_chunks: int) -> List[Tuple[int, int]]:
    """Split ``range(n)`` into ``n_chunks`` contiguous halves-open intervals."""
    n_chunks = max(1, min(int(n_chunks), int(n)))
    size = int(math.ceil(n / n_chunks))
    return [(lo, min(lo + size, n)) for lo in range(0, n, size)]


def length_sorted_microbatches(indices, lengths, micro_batch: int):
    """Regroup ONE optimizer step's cells into micro-batches by TOKEN LENGTH.

    WHY THIS IS PART OF THE EFFECTIVE-BATCH MACHINERY, AND NOT A MICRO-OPTIMISATION.
    Under the grad cache every micro-batch is padded to ITS OWN max, and ALL `accum`
    payloads stay resident because pass 2 re-forwards them.  The RNA length distribution
    on this cache is heavily skewed -- mean 2,315, p50 1,953, p99 7,025, max 8,497 -- so
    a RANDOM micro-batch of 32 pads to ~6,500 and carries 2.8x more RNA tokens than it
    has.  That waste is paid TWICE: in resident bytes and in refiner FLOPs.  Measured on
    an H200 at B_eff 512, M=64, ARM B: resident 7.68 -> 9.24 GiB and 46.1 -> 51.8
    s/step as micro_batch goes 8 -> 64, purely because the padded RNA length goes
    4,962 -> 7,058.

    WHY IT IS FREE, AND CANNOT BREAK `same_dataset_blocked`.  This is a PERMUTATION
    WITHIN ONE OPTIMIZER STEP.  The set of cells in the step -- which IS the contrastive
    batch, because the loss is taken once on the concatenation -- is bit-identical; only
    which cells share a padding bucket changes.  The sampler already drew the whole step
    from a SINGLE dataset, so no cross-dataset negative can appear.  ⛔ Do NOT sort
    across steps: that would order the training stream by sequencing depth.

    Args:
      indices     the step's cell indices, in sampler order (len == micro_batch * accum)
      lengths     per-cell token count aligned to `indices` (use the LONGER modality, or
                  `rna_len + atac_len`; `CachedTokenDataset.lengths("rna")` gives it
                  straight from the index, no reads)
      micro_batch size of each returned list
    Returns a list of `accum` index lists whose CONCATENATION is a permutation of
    `indices`.
    """
    import numpy as _np

    idx = _np.asarray(indices)
    ln = _np.asarray(lengths)
    assert idx.shape == ln.shape, f"indices {idx.shape} vs lengths {ln.shape}"
    assert len(idx) % micro_batch == 0, (
        f"{len(idx)} cells is not a whole number of micro-batches of {micro_batch}")
    order = _np.argsort(ln, kind="stable")
    srt = idx[order]
    return [srt[i:i + micro_batch].tolist()
            for i in range(0, len(srt), micro_batch)]


def grad_cache_two_pass(
    micro_batches: Sequence[Any],
    forward_fn: Callable[[Any], Any],
    loss_fn: Callable[..., Tuple[torch.Tensor, Any]],
    device,
    ddp_mods: Sequence[torch.nn.Module] = (),
    rng_replay: bool = True,
    verify: bool = False,
    verify_tol: float = 1e-2,
    verify_features: Any = "all",
    loss_chunks: int = 1,
    feature_names: Optional[Sequence[str]] = None,
    aux_names: Optional[Sequence[str]] = None,
    require_fp32_leaves: bool = True,
    report: Optional[Dict[str, Any]] = None,
):
    """Two-pass gradient cache over an ARBITRARY tuple of cached feature tensors.

    Args:
      micro_batches   opaque items, one per micro-batch, handed to ``forward_fn``.
      forward_fn(mb)  -> tensor | tuple of tensors | ``GCOut(features, aux)``.  Called
                      TWICE per step per micro-batch (pass 1 under ``no_grad``, pass 2
                      with grad and the replayed RNG), so it must be a pure function of
                      ``mb`` and the RNG.
      loss_fn         ``loss_fn(*FULL_FEATURES, *FULL_AUX) -> (total, extra)``, where
                      each FULL_* is the dim-0 concatenation over micro-batches.  Any
                      cross-rank gather lives INSIDE ``loss_fn`` so its gradient reaches
                      the leaves.  With ``loss_chunks > 1`` it must additionally expose
                      ``prepare`` / ``denominators`` / ``chunk`` (see ``ChunkedLoss``).
      ddp_mods        DDP-wrapped modules -> all but the LAST pass-2 micro-batch runs
                      under ``no_sync()``, so the step still costs ONE all-reduce.
      rng_replay      ``False`` ONLY for the equivalence test's negative control.
      verify          recompute-drift check, THE precondition of the method.  Run
                      it on the first grad-cache step of a run, and on every step of a
                      smoke test.
      verify_features ``"all"`` in production.  A tuple of feature indices NARROWS the
                      check -- which exists only so the test can DEMONSTRATE that a
                      globals-only verify is a proxy: it passes while the slot branch is
                      silently mismatched (T5).
      loss_chunks     split the loss's QUERY dim into this many chunks, backward each.
                      Exactly equivalent -- the total is a plain sum over query chunks
                      and the normalizer is a pure function of the non-grad aux -- and
                      it keeps the per-slot ``[Q, B, M]`` intermediates bounded.
      require_fp32_leaves
                      a leaf in fp16 accumulates dL/dF in fp16.  Refined tokens leave
                      ``ATACRefinerFM.native_grad`` through a ``.float()`` and every
                      projection output is fp32, so a half leaf means something upstream
                      changed: fail loudly, do not lose decimal digits of gradient.
      report          optional dict, filled with diagnostics (verify max, which features
                      received no gradient, chunk count).  Use it for the run's liveness
                      line rather than re-deriving these numbers.

    Returns ``(total.detach(), extra)``.
    """
    from contextlib import ExitStack

    n_mb = len(micro_batches)
    assert n_mb > 0, "grad_cache_two_pass needs at least one micro-batch"
    rep = report if report is not None else {}

    # ---- pass 1: features only, no graph kept ----------------------------------------
    states: List[Any] = []
    leaves: List[Tuple[torch.Tensor, ...]] = []
    auxes: List[Tuple[torch.Tensor, ...]] = []
    for mb in micro_batches:
        states.append(_rng_snapshot(device))
        with torch.no_grad():
            feats, aux = _split(forward_fn(mb))
        if require_fp32_leaves:
            bad = [i for i, t in enumerate(feats) if t.dtype != torch.float32]
            assert not bad, (
                f"grad_cache leaves must be fp32; feature(s) {bad} are "
                f"{[feats[i].dtype for i in bad]}.  A half leaf accumulates dL/dF in "
                f"fp16.  If this is deliberate pass require_fp32_leaves=False.")
        leaves.append(tuple(t.detach().clone().requires_grad_(True) for t in feats))
        auxes.append(tuple(aux))

    n_feat = len(leaves[0])
    n_aux = len(auxes[0])
    assert all(len(l) == n_feat for l in leaves), (
        "forward_fn returned a different number of FEATURES on different micro-batches")
    assert all(len(a) == n_aux for a in auxes), (
        "forward_fn returned a different number of AUX tensors across micro-batches")
    fnames = (list(feature_names) if feature_names
              else [f"feat{i}" for i in range(n_feat)])
    anames = list(aux_names) if aux_names else [f"aux{i}" for i in range(n_aux)]

    FULL_F = [torch.cat([l[i] for l in leaves], 0) for i in range(n_feat)]
    FULL_A = [torch.cat([a[i] for a in auxes], 0) for i in range(n_aux)]

    # ---- loss on the FULL batch, ONCE (optionally query-chunked) ---------------
    chunks = max(1, int(loss_chunks))
    if chunks > 1:
        assert hasattr(loss_fn, "chunk"), (
            "loss_chunks > 1 requires a ChunkedLoss (prepare/denominators/chunk); "
            f"{type(loss_fn).__name__} has no .chunk")
        prep = (loss_fn.prepare(*FULL_F, *FULL_A) if hasattr(loss_fn, "prepare")
                else tuple(FULL_F) + tuple(FULL_A))
        with torch.no_grad():
            denom = loss_fn.denominators(*prep)
        # ⛔ chunk the LOSS's query dim, not the local one. With --gather_negatives the
        # prepare() step all-gathers, so the loss's query axis is world_size x larger
        # than FULL_F's; chunking on the local length would leave (W-1)/W of the queries
        # out of the loss entirely and the run would look merely "a bit worse". A loss
        # whose first prepared tensor is not the query axis (CombinedLoss, whose globals
        # may be ungathered while its slots are gathered) says so via query_size().
        B = int(loss_fn.query_size(*prep) if hasattr(loss_fn, "query_size")
                else prep[0].shape[0])
        bounds = chunk_bounds(B, chunks)
        total_val = 0.0
        extras: List[Any] = []
        for j, (lo, hi) in enumerate(bounds):
            part, ex = loss_fn.chunk(*prep, q=slice(lo, hi), denom=denom)
            # retain_graph on all but the last: `prepare` may hold a SHARED prefix graph
            # (the cross-rank gather is the case that matters) which every chunk reuses.
            part.backward(retain_graph=(j < len(bounds) - 1))
            total_val += float(part.detach())
            extras.append(ex)
        extra = (loss_fn.merge_extra(extras) if hasattr(loss_fn, "merge_extra")
                 else _merge_extra_default(extras))
        total = torch.as_tensor(total_val)
        rep["loss_chunks"] = len(bounds)
    else:
        total, extra = loss_fn(*FULL_F, *FULL_A)
        total.backward()
        rep["loss_chunks"] = 1

    no_grad_feats = [fnames[i] for i in range(n_feat)
                     if all(l[i].grad is None for l in leaves)]
    rep["features_without_gradient"] = no_grad_feats

    # ---- pass 2: re-forward + backward the cached dL/dF -------------------------
    last = n_mb - 1
    vmax = 0.0
    vworst = ""
    which = (range(n_feat) if verify_features == "all"
             else tuple(int(i) for i in verify_features))
    for i, mb in enumerate(micro_batches):
        if rng_replay:
            _rng_restore(states[i], device)
        with ExitStack() as st:
            if i != last:
                for m in ddp_mods:
                    st.enter_context(m.no_sync())
            feats, aux = _split(forward_fn(mb))
            if verify:
                for k in which:
                    d, s = _drift(feats[k], leaves[i][k])
                    if d / s > vmax:
                        vmax, vworst = d / s, f"{fnames[k]}[mb{i}]"
                for k in range(n_aux):
                    # aux carries no gradient but the cached dL/dF was computed WITH it,
                    # so a drift here invalidates the cache exactly as a dropout
                    # mismatch would. bool aux must match EXACTLY -- a routing mask that
                    # flips is not a rounding error.
                    a1, a0 = aux[k], auxes[i][k]
                    if a0.dtype == torch.bool:
                        if not torch.equal(a1, a0):
                            raise RuntimeError(
                                f"--grad_cache: AUX '{anames[k]}' (bool) differs "
                                f"between pass 1 and pass 2 on micro-batch {i} "
                                f"({int((a1 != a0).sum())} of {a0.numel()}).  The "
                                f"routing/validity mask is not a pure function of the "
                                f"token ids, so every cached dL/dF belongs to a "
                                f"different problem than pass 2 backpropagates.")
                    else:
                        d, s = _drift(a1, a0)
                        if d / s > vmax:
                            vmax, vworst = d / s, f"{anames[k]}[mb{i}](aux)"
            ts, gs = [], []
            for k in range(n_feat):
                g = leaves[i][k].grad
                if g is not None:
                    ts.append(feats[k])
                    gs.append(g)
            if ts:
                torch.autograd.backward(ts, gs)   # NO 1/accum: the loss was taken once

    if verify:
        rep["verify_max_rel"] = vmax
        rep["verify_worst"] = vworst
        rep["verify_checked"] = ([fnames[k] for k in which]
                                 + [f"{a}(aux)" for a in anames])
        if vmax > verify_tol:
            raise RuntimeError(
                f"--grad_cache RNG replay FAILED: the pass-2 re-forward differs from "
                f"the pass-1 cached tensors by {vmax:.3e} (rel, tol {verify_tol:g}); "
                f"worst offender {vworst}.  The cached dL/dF therefore belongs to a "
                f"DIFFERENT function than the graph it is backpropagated through, i.e. "
                f"every gradient this step produced is WRONG.  Most likely the dropout "
                f"masks are not being reproduced (the forward consumes RNG differently "
                f"under torch.no_grad() than with grad enabled here).  Re-run "
                f"with --grad_cache 0, or with --dropout 0 --projection_dropout 0.")
        checked = ", ".join(rep["verify_checked"])
        print(f"  [grad_cache] pass1/pass2 replay check over {{{checked}}}: "
              f"max rel diff {vmax:.3e} at {vworst or 'n/a'} (tol {verify_tol:g}) OK",
              flush=True)
        if verify_features != "all":
            print(f"  [grad_cache] ⚠️ verify_features={verify_features} -- this is a "
                  f"PROXY check; features "
                  f"{[fnames[k] for k in range(n_feat) if k not in which]}"
                  f" were NOT verified", flush=True)
    if no_grad_feats:
        # Not an error: ARM A legitimately produces no slot features at all.  But a
        # feature that IS produced and gets no gradient means a loss term is off (or
        # weighted 0), which under DDP with find_unused_parameters=False CRASHES on the
        # module that produced it.  Say so loudly, once, with the names.
        print(f"  [grad_cache] NOTE: cached feature(s) {no_grad_feats} received NO "
              f"gradient from the loss -- the module(s) producing them are "
              f"untrained this "
              f"step (DDP find_unused_parameters=False raises on them).", flush=True)
    return total.detach(), extra


def _drift(new: torch.Tensor, old: torch.Tensor) -> Tuple[float, float]:
    """(max abs difference, scale) between a pass-2 tensor and its pass-1 cache."""
    d = float((new.detach().float() - old.detach().float()).abs().max())
    s = float(old.detach().float().abs().max().clamp(min=1e-6))
    return d, s


def _merge_extra_default(extras: List[Any]):
    """Sum the numeric fields of per-chunk ``extra`` dicts; pass strings through."""
    if not extras:
        return {}
    if not isinstance(extras[0], dict):
        return extras[-1]
    out: Dict[str, Any] = {}
    for k in extras[0]:
        vals = [e[k] for e in extras if k in e]
        out[k] = sum(vals) if isinstance(vals[0], (int, float)) else vals[-1]
    return out


# ------------------------------------------------------------------------------------ #
# The fine loss the cache has to carry: per-slot InfoNCE + per-slot cross-modal SupCon
# ------------------------------------------------------------------------------------ #


class ChunkedLoss:
    """Protocol for a loss whose QUERY dimension can be split.

    ``sum_over_chunks(chunk(...)) == __call__(...)`` must hold EXACTLY (up to fp), which
    it does iff the loss's normalizer is a pure function of the NON-GRAD aux -- true
    here: ``loss_weight = sqrt(q_weight * c_weight) * valid_module`` comes from
    ``mass``, ``valid`` and the labels, none of which touch the features.

      prepare(*tensors)      -> tensors'  (optional; the cross-rank gather lives here
                                          so it happens ONCE, not once per chunk)
      denominators(*t')      -> denom      (computed under no_grad, from aux only)
      chunk(*t', q, denom)   -> (partial, extra)
    """


class SlotContrastiveLoss(ChunkedLoss):
    """Per-slot analogue of the DECIDED cell loss: ``w_infonce`` x InfoNCE (diagonal
    positive) + ``w_supcon`` x cross-modal SupCon (same-label off-diagonal positives),
    symmetrized over the two retrieval directions.

    WHY THIS AND NOT ``per_module_mpnce``.  The reference FineCLS config drives the fine
    branch with ``module_loss_type: per_module_mpnce`` over an EMA teacher.  Running an
    MP-NCE fine branch under an InfoNCE+SupCon global branch introduces a LOSS MISMATCH
    between the two branches that would confound a null: we could not tell "the fine
    branch does not transfer" from "the two branches optimize different things".  So the
    fine loss is the per-slot analogue of the cell loss, at ``module_temperature 0.07``.
    This is closer to the reference than it looks -- with ``label_positives`` +
    ``hierarchical_positives`` + ``multipositive_scheme: graded``, their positive set IS
    same-cell-type, i.e. SupCon.  The EMA teacher is a fourth moving part we decline to
    add on a track whose backbone is already changing.

    Shapes: ``rna_slots``/``atac_slots`` ``[B, M, D]``; ``valid``/``mass`` ``[B, M]``;
    ``labels`` ``[B]`` with -1 = unknown.  Routing (``fixed_slot_route_weights``) is the
    colleague's function, imported not copied.
    """

    def __init__(self, temperature: float = 0.07, w_infonce: float = 0.5,
                 w_supcon: float = 0.5, routing_topk: int = 16,
                 routing_mass_power: float = 1.0, routing_tail_weight: float = 0.15,
                 gather: bool = False, eps: float = 1e-8):
        self.temperature = float(temperature)
        self.w_infonce = float(w_infonce)
        self.w_supcon = float(w_supcon)
        self.routing_topk = int(routing_topk)
        self.routing_mass_power = float(routing_mass_power)
        self.routing_tail_weight = float(routing_tail_weight)
        self.gather = bool(gather)
        self.eps = float(eps)

    # -- gather once, outside the chunk loop ------------------------------------------
    def prepare(self, rna_slots, atac_slots, rna_valid, atac_valid, rna_mass, atac_mass,
                labels):
        if self.gather and dist_on():
            # ⛔ valid/mass/labels with_grad=False: they carry no gradient, and a
            # GatherWithGrad node on them would add an all_reduce in its backward for
            # nothing.  The features MUST use the grad-carrying gather so the cross-rank
            # negatives' dL/dF comes back to this rank's leaves.
            rna_slots = gather_features(rna_slots, with_grad=True)
            atac_slots = gather_features(atac_slots, with_grad=True)
            rna_valid = gather_features(rna_valid, with_grad=False)
            atac_valid = gather_features(atac_valid, with_grad=False)
            rna_mass = gather_features(rna_mass, with_grad=False)
            atac_mass = gather_features(atac_mass, with_grad=False)
            labels = gather_features(labels, with_grad=False)
        return (rna_slots, atac_slots, rna_valid, atac_valid, rna_mass, atac_mass,
                labels)

    # -- routing weights + the normalizer, all from AUX only ---------------------------
    def _routes(self, rna_valid, atac_valid, rna_mass, atac_mass):
        q_w, q_r = fixed_slot_route_weights(
            rna_mass, rna_valid, self.routing_topk, self.routing_mass_power,
            self.routing_tail_weight)
        c_w, c_r = fixed_slot_route_weights(
            atac_mass, atac_valid, self.routing_topk, self.routing_mass_power,
            self.routing_tail_weight)
        return q_w, q_r, c_w, c_r

    def denominators(self, rna_slots, atac_slots, rna_valid, atac_valid, rna_mass,
                     atac_mass, labels):
        """``loss_weight.sum()`` for both directions.  Pure aux -> constant w.r.t. the
        features, which is exactly why chunking the query dim is EXACT and not an
        approximation."""
        q_w, q_r, c_w, c_r = self._routes(rna_valid, atac_valid, rna_mass, atac_mass)
        pv_diag = q_r & c_r                                    # [B,M] at the TRUE pair
        w_ra = (torch.sqrt(q_w * c_w) * pv_diag.float()).sum()
        w_ar = (torch.sqrt(c_w * q_w) * pv_diag.float()).sum()
        return {"w_ra": w_ra.clamp_min(self.eps), "w_ar": w_ar.clamp_min(self.eps)}

    # -- one direction over a slice of queries ----------------------------------------
    def _direction(self, query_slots, cand_slots, q_w, q_r, c_w, c_r, labels, q,
                   denom_sum):
        """Queries ``q`` against ALL candidates.  Returns the PARTIAL
        ``(per_module * loss_weight).sum() / denom_sum``, whose sum over query chunks
        reproduces the full ``(...).sum() / (...).sum()`` exactly -- because denom_sum
        is a constant of the aux, not of the features."""
        Q = F.normalize(query_slots[q].float(), dim=-1)        # [q,M,D]
        C = F.normalize(cand_slots.float(), dim=-1)            # [B,M,D]
        scores = torch.einsum("qmd,nmd->qnm", Q, C)            # [q,B,M]
        pair_valid = q_r[q][:, None, :] & c_r[None, :, :]      # [q,B,M]
        logits = (scores / self.temperature).masked_fill(~pair_valid, -1e4)
        log_prob = F.log_softmax(logits, dim=1)                # over CANDIDATES

        rows = torch.arange(q.start, q.stop, device=scores.device)
        loc = torch.arange(rows.numel(), device=scores.device)
        diag_loss = -log_prob[loc, rows, :]                    # [q,M]  the true pair
        diag_valid = pair_valid[loc, rows, :]                  # [q,M]

        lab_q, lab_c = labels[q], labels
        known = (lab_q >= 0)[:, None] & (lab_c >= 0)[None, :]
        same = ((lab_q[:, None] == lab_c[None, :]) & known).float()
        same[loc, rows] = 0.0                            # diagonal owned by InfoNCE
        sup_f = same[:, :, None] * pair_valid.float()          # [q,B,M]
        n_pos = sup_f.sum(dim=1)                               # [q,M]
        sup_loss = torch.where(n_pos > 0,
                               -(sup_f * log_prob).sum(dim=1) / n_pos.clamp(min=1.0),
                               torch.zeros_like(diag_loss))

        per_module = self.w_infonce * diag_loss + self.w_supcon * sup_loss
        # slot activity weighting: sqrt of the two routed masses at the TRUE pair, which
        # is what makes a module that this cell barely observes contribute barely at
        # all.
        w = torch.sqrt(q_w[q] * c_w[rows]) * diag_valid.float()
        return (per_module * w).sum() / denom_sum, {
            "fine_diag": float((diag_loss * w).sum().detach()),
            "fine_supcon": float((sup_loss * w).sum().detach()),
            "fine_w": float(w.sum().detach()),
        }

    def chunk(self, rna_slots, atac_slots, rna_valid, atac_valid, rna_mass, atac_mass,
              labels, q: slice, denom):
        q_w, q_r, c_w, c_r = self._routes(rna_valid, atac_valid, rna_mass, atac_mass)
        ra, e1 = self._direction(rna_slots, atac_slots, q_w, q_r, c_w, c_r, labels, q,
                                 denom["w_ra"])
        ar, e2 = self._direction(atac_slots, rna_slots, c_w, c_r, q_w, q_r, labels, q,
                                 denom["w_ar"])
        total = 0.5 * (ra + ar)
        return total, {k: e1.get(k, 0.0) + e2.get(k, 0.0) for k in set(e1) | set(e2)}

    def __call__(self, rna_slots, atac_slots, rna_valid, atac_valid, rna_mass,
                 atac_mass, labels):
        prep = self.prepare(rna_slots, atac_slots, rna_valid, atac_valid, rna_mass,
                            atac_mass, labels)
        with torch.no_grad():
            denom = self.denominators(*prep)
        B = int(prep[0].shape[0])
        return self.chunk(*prep, q=slice(0, B), denom=denom)


class CombinedLoss(ChunkedLoss):
    """ARM B's objective: the DECIDED cell loss on the globals + the per-slot loss.

    Feature order (must match ``forward_fn``'s):
        ``rna_global[B,256], atac_global[B,256], rna_slots[B,M,256],
        atac_slots[B,M,256]``
    Aux order:
        ``rna_valid[B,M], atac_valid[B,M], rna_mass[B,M], atac_mass[B,M], labels[B]``

    ``global_fn(rna_global, atac_global, labels) -> (total, extra_dict)`` is supplied by
    the caller so the cell loss stays the trainer's own, not a copy that can drift.

    ⛔ ONE GATHER POLICY.  When ``gather=True`` this class all-gathers EVERYTHING in
    ``prepare`` -- globals, slots, valid, mass, labels -- exactly once per step, and
    ``global_fn`` MUST NOT gather internally.  The upstream ``gc_cell_losses``
    (train_filip_combined.py:615) DOES gather internally, so on this track it is wrapped
    with its gathering removed.  A double gather is not a crash: it silently squares the
    batch the cell loss sees while the slot loss sees ``W*B``, and the two branches then
    optimize different-sized problems.

    ARM A does NOT use this class: it passes ``global_fn`` to ``grad_cache_two_pass``
    directly with ``features = (rna_global, atac_global)`` and ``aux = (labels,)``. That
    is the structural arm difference, and it is why ARM A must not ALLOCATE the slot
    machinery -- a constructed-but-unused slot branch gets no gradient, which under DDP
    with ``find_unused_parameters=False`` raises.

    ⚠️ GRADIENT CLIPPING IS JOINT (``clip_grad_norm_(model.parameters(), 4.0)``). A wide
    fine branch throttles the global branch's effective step for reasons unrelated to
    slots (their U1 saw the same-run global drop 0.3720 -> 0.2678).  Log the pre-clip
    total grad norm and the clip-active fraction PER ARM, or an A-vs-B global contrast
    is uninterpretable.
    """

    def __init__(self, global_fn, slot_loss: "SlotContrastiveLoss",
                 global_weight: float = 1.0, fine_weight: float = 1.0):
        self.global_fn = global_fn
        self.slot = slot_loss
        self.global_weight = float(global_weight)
        self.fine_weight = float(fine_weight)
        # ⛔ fine_weight 0 is NOT "arm A". `0.0 * fine_loss` still builds the whole
        # [B,B,M] graph, still back-propagates, and hands every slot parameter a tensor
        # of ZEROS -- so it costs full compute, trains nothing, and is INVISIBLE to a
        # "did this parameter get a gradient" check (the grad is not None, it is 0). ARM
        # A does not construct the slot machinery at all and its loss does not mention
        # it.
        assert self.fine_weight != 0.0, (
            "CombinedLoss with fine_weight=0 pays for the fine branch and trains "
            "nothing.  For ARM A pass global_fn to grad_cache_two_pass directly with "
            "features=(rna_global, atac_global), aux=(labels,) and do not allocate the "
            "slot modules.")

    @property
    def gather(self) -> bool:
        return self.slot.gather

    def prepare(self, rg, ag, rs, ats, rv, av, rmass, amass, labels):
        if self.gather and dist_on():
            rg = gather_features(rg, with_grad=True)
            ag = gather_features(ag, with_grad=True)
        rs, ats, rv, av, rmass, amass, labels = self.slot.prepare(
            rs, ats, rv, av, rmass, amass, labels)
        return (rg, ag, rs, ats, rv, av, rmass, amass, labels)

    def denominators(self, rg, ag, rs, ats, rv, av, rmass, amass, labels):
        return self.slot.denominators(rs, ats, rv, av, rmass, amass, labels)

    def query_size(self, rg, ag, rs, ats, rv, av, rmass, amass, labels) -> int:
        """The SLOT tensors define the chunked query axis (the globals are never
        chunked -- the O(B^2) cell term has no M axis and is 0.001 GiB at B=512)."""
        return int(rs.shape[0])

    def chunk(self, rg, ag, rs, ats, rv, av, rmass, amass, labels, q: slice, denom):
        total, extra = self.slot.chunk(rs, ats, rv, av, rmass, amass, labels, q=q,
                                       denom=denom)
        total = self.fine_weight * total
        extra = dict(extra)
        if q.start == 0:
            # Add the global term to exactly ONE chunk: sum(chunks) then equals the
            # unchunked total EXACTLY rather than approximately.
            g, gx = self.global_fn(rg, ag, labels)
            total = total + self.global_weight * g
            extra.update(gx if isinstance(gx, dict) else {"global": float(gx)})
            extra["global_loss"] = float(g.detach())
        return total, extra

    def __call__(self, rg, ag, rs, ats, rv, av, rmass, amass, labels):
        prep = self.prepare(rg, ag, rs, ats, rv, av, rmass, amass, labels)
        with torch.no_grad():
            denom = self.denominators(*prep)
        return self.chunk(*prep, q=slice(0, self.query_size(*prep)), denom=denom)
