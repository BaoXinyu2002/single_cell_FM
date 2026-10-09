#!/usr/bin/env python3
"""Score the FineCLS-on-refiner arms: within-(dataset x cell_type) R@1, true OOD.

Design: experiments/finecls_refiner/EXPERIMENT_finecls_on_refiner.md, section 6.  Read
it before changing a default.  Every constant below is traced there or to a memory note.

WHAT THIS SCRIPT IS FOR
-----------------------
ARM A (`cell_only`), ARM B (`finecls`, biology prior) and ARM C (`finecls`, a3_random64)
are trained by `train_finecls_refiner.py` and must be contrasted on ONE metric, computed
ONE way, at ONE fixed step.  No existing script does that:

    eval_ckpt_stratified.py  has the literal "within ds x cell_type" stratum, but
                             resolves labels only through master_labels.csv (the OOD
                             sets carry theirs in `obs`, so it drops all four), and its
                             `r1_draws` is documented "Symmetric R@1" -- the direction
                             averaging that retired the +0.0372 centring headline.
    eval_within_type.py      has the right strata AND the right OOD label plumbing, but
                             scores with FILIP token similarity, not the cell embedding.
    refiner_ood_center.py    has the correct test-time centring and the pool machinery,
                              but only donor x cell_type and only the cell embedding.

So this file composes them: label/donor resolution in the shape `eval_within_type.py`
proved correct, pools and per-draw R@1 from `diag_b_ceiling`, centring imported VERBATIM
from `refiner_ood_center.per_group_center`.  Nothing about the metric is reimplemented
except the parts that had to be split per direction.

FOUR THINGS IT REFUSES TO DO
----------------------------
1.  Score a checkpoint whose path contains "best".  `best_model.pt` is selected on val
    loss and, on the projector track, mirrored from `best_ood_model.pt`, which was
    selected on bmmc + fetal_heart -- HALF this panel.  Every absolute number taken from
    it inherits that leak.  Arms are compared at a FIXED step, passed explicitly.
2.  Report a panel that is not exactly {breast, liver, fetal_heart, bmmc}.  islet
    and pln are IN the 163-dataset training corpus (islet_concat's donors
    HPAP-079/093/095/096/129/130 are all training donors); they may be scored, but they
    are labelled `in_training: true` and never enter a panel mean.
3.  Average the two retrieval directions.  r2a and a2r are reported separately and a
    verdict must hold in BOTH.
4.  Compare R@1 across sets whose label granularity is unmatched without saying so
    loudly -- coarse labels make within-type retrieval 47-66% easier, so a cross-set
    difference can be pure annotation depth.

THE POOL DEFINITION *IS* THE METRIC
-----------------------------------
On the same cells and the same model this project measured dataset-window R@1 0.1616,
donor 0.1323, donor x cell_type 0.0329 -- a 4.9x spread.  So all four strata are always
computed and printed side by side; the PRIMARY is `dataset x cell_type` and the others
exist so a reader can see which pool a number lives on.

CENTRING IS A TRAIN x TEST INTERACTION
--------------------------------------
Train-time centring is fixed by the checkpoint (`center_global_by_dataset` /
`center_slots_by_dataset` in its saved args) and is read, not chosen, here.  Test-time
centring is a post-hoc transform on the saved embeddings and every checkpoint is scored
BOTH ways -- non-negotiable, because `_dcenter` is gated on `self.training`, so a model
trained with centring is not automatically evaluated with it.

    metrics.json is emitted per (arm, step, test-centring).

⛔ THE GALLERY-ONLY NO-OP.  Memory (`centering_mechanism_is_renormalization`) says the
RE-NORMALISATION carries the gain and that a gallery-only shift is a no-op.  That is a
rank identity, not folklore: for a fixed query row, subtracting a constant from every
gallery vector shifts the whole row by -q.mu, which cannot reorder it.  So this script
runs it as a two-directional CONTROL on every set: a gallery-only shift WITHOUT
re-normalisation must move R@1 by EXACTLY 0.0, and the real both-sides + re-normalised
centring must move it by something non-zero.  If the first is non-zero the scorer is not
a plain inner-product ranking; if the second is zero the centring switch is inert and
every centred number in the run is a silent lie.

USAGE
-----
    # 1. embeddings, once per (arm, checkpoint) -- LIVE FMs, GPU, ~4 h/checkpoint
    python eval_finecls_refiner.py extract --ckpt .../snapshots/step_008000.pt \\
        --expect_step 8000 --out_dir results/armB --sets bmmc breast fetal_heart liver

    # 2. scoring -- CPU, seconds, every centring variant free
    python eval_finecls_refiner.py score --emb results/armA --emb results/armB \\
        --emb results/armC --expect_step 8000 --out_dir results/score

    # 3. the harness's own tests (CPU, no data, no GPU)
    python eval_finecls_refiner.py selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = "/nfs/turbo/umms-drjieliu1/usr/xinyubao/sclip"
HERE = os.path.dirname(os.path.abspath(__file__))


def _bootstrap_sys_path() -> None:
    """Put the four trees this file borrows from on sys.path, in a fixed order.

    Order matters: `experiments/results/ood_diag` must precede the finelip scripts dir
    so `diag_common` resolves to the diagnosis tree's copy, which is the one every
    published OOD number in this project was computed with.
    """
    for p in (REPO,
              f"{REPO}/haoyun",
              f"{REPO}/experiments/results/ood_diag",
              f"{REPO}/experiments/scripts",
              f"{REPO}/haoyun/multiomics_clip_finelip/scripts",
              HERE):
        if p not in sys.path:
            sys.path.insert(0, p)


def _bootstrap_model_path() -> None:
    """Extra trees needed only by the SLOT scorer and by `extract` (heavy imports).

    Kept out of the module-level bootstrap so that `score`, the gates and the self-test
    never pay for scanpy / scFoundation / the FineLIP package.  `REPO/experiments` is
    what makes `multiomics_clip_xinyu_June_fixed_slot_routing.modules.fixed_slot`
    importable as a package, exactly as `train_finecls_refiner.py` resolves it, so the
    eval imports the SAME `fixed_slot_similarity` the training loss uses.
    """
    for p in (f"{REPO}/experiments",
              f"{REPO}/haoyun/multiomics_clip_finelip",
              f"{REPO}/scFoundation/model"):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_sys_path()


# ------------------------------------------------------------------------------------ #
# The panel.  These four names are the whole reason this file exists.
# ------------------------------------------------------------------------------------ #

# ⛔ ONLY FOUR SETS ARE TRUE OOD.  islet and pln are IN the training corpus: the 163
# dataset_ids include HPAP donors 079/093/095/096/129/130, which is exactly what
# `islet_concat` is made of.  A panel that quietly includes them reports a training
# number as a generalisation number.
OOD_TRUE: Tuple[str, ...] = ("bmmc", "breast", "fetal_heart", "liver")

# Names that are legal to score but must be LABELLED in-training and excluded from every
# panel aggregate.
IN_TRAINING: Tuple[str, ...] = ("islet", "islet_concat", "pln", "curated_val", "val",
                                "val_indist")

# Cell counts are the cheapest correctness check there is: `islet_concat` (9,976 cells)
# and the fig2 islet file (26,892) differ by 2.7x, and only one of them is the file the
# rest of the project scored.  A wrong h5ad path is caught here, before any GPU time.
# ⛔ MEASURED against the h5ads these paths actually point at, not copied from a memory
# note.  `liver` is the trap: the eval-path memory records 30,135, which is the CACHED
# FEATURE count -- `sc.read_h5ad(ood_work/liver/liver_rna.h5ad, backed="r").n_obs` is
# 30,144, and `cache_refined_tokens.py:66-68` already documents the discrepancy. A gate
# that fires on the largest OOD set AFTER the training spend pushes the operator toward
# `--skip_cell_count_gate` (disabling the wrong-file check) or `--allow_partial_panel`
# (silently scoring 3 sets), which is worse than no gate at all.
EXPECTED_N: Dict[str, int] = {
    "breast": 9739, "liver": 30144, "fetal_heart": 8668, "bmmc": 20766,
    "islet": 26892, "pln": 26486,
}

# name -> (rna h5ad, atac h5ad, label source).  Taken from
# eval_parity_metricsjson.py:72, the only DS dict in the repo that carries all six sets
# WITH per-set cell counts. ⛔ do NOT import eval_summary_ood.DS: its islet entry points
# at islet_concat, i.e. at training donors.  For bmmc this is the `_aligned` pair, whose
# ATAC carries `cell_sentences`; `bmmc_test_atac_paired.h5ad` does not and the live-FM
# path asserts.
DS: Dict[str, Tuple[str, str, str]] = {
    "breast":      (f"{REPO}/ood_work/breast/breast_rna.h5ad",
                    f"{REPO}/ood_work/breast/breast_atac.h5ad", "obs"),
    "liver":       (f"{REPO}/ood_work/liver/liver_rna.h5ad",
                    f"{REPO}/ood_work/liver/liver_atac.h5ad", "obs"),
    "fetal_heart": (f"{REPO}/preprocessed_data_ood/fetal_heart_rna.h5ad",
                    f"{REPO}/preprocessed_data_ood/fetal_heart_atac.h5ad", "obs"),
    "bmmc":        (f"{REPO}/preprocessed_data_ood/bmmc_aligned_rna.h5ad",
                    f"{REPO}/preprocessed_data_ood/bmmc_aligned_atac.h5ad", "obs"),
    "islet":       (
        f"{REPO}/experiments/fig2_integration/data/test_islet_hpap_rna.h5ad",
        f"{REPO}/experiments/fig2_integration/data/test_islet_hpap_atac.h5ad", "obs"),
    "pln":         (f"{REPO}/experiments/fig2_integration/data/test_pln_hpap_rna.h5ad",
                    f"{REPO}/experiments/fig2_integration/data/test_pln_hpap_atac.h5ad",
                    "obs"),
}

# All four OOD sets carry a SINGLE dataset_id while holding 4-19 donors, so `dataset` is
# not the donor analogue there and the donor column has to come from obs.  Getting this
# wrong collapses "within donor" to "within set" silently.
DONOR_COL: Dict[str, str] = {"fetal_heart": "batch", "bmmc": "batch", "breast": "batch",
                             "liver": "batch", "islet": "batch", "pln": "batch"}

# ------------------------------------------------------------------------------------ #
# The 11 EXPANDED-SOURCE islet donors, registered for `extract` ONLY (Fig 5 arm-B rerun)
# ------------------------------------------------------------------------------------ #
# WHY.  emb_cooled/islet.npz covers `DS["islet"]` = the fig2 26,892-cell file, whose
# donors are HPAP-147..160.  The Fig 5 T1D islet cohort
# (results/perturbation_score_modules_hpap/cohort/beta_meta.csv, 4,256 Beta cells,
# 14 donors) draws 3,477 of its cells from ELEVEN OTHER donors that live only as
# per-donor h5ads under preprocessed_data_expanded_20260305/.  Matching the cohort
# against islet.npz returns 779 / 4,256 -- three donors -- which would leave the
# T1D-vs-Unaffected contrast with a single control donor.  Registering these files here
# lets `extract` produce arm-B vectors for them with the SAME code path, the same
# checkpoint gates and the same barcode recovery as every other published arm-B number.
#
# ⛔⛔ IN-TRAINING, AND MORE SO THAN `islet` ITSELF.  All eleven appear in the token
# cache index (fm_token_cache_expanded_8192/index.npz): these are outright TRAINING
# donors.  They are added to IN_TRAINING below so `resolve_panel` keeps them out of
# every OOD panel mean, exactly as `islet` is kept out.  Nothing extracted from them is
# a generalisation result.
#
# ⛔ THE TWO OBS TRAPS, MEASURED (h5py read of all 22 files, 2026-09-01):
#   RNA obs  = ['batch_id', 'dataset_id', 'n_genes']       -- NO 'batch', NO 'cell_type'
#   ATAC obs = ['batch_id', 'cell_sentences', 'dataset_id']
#   (1) DONOR: the DONOR_COL default 'batch' is ABSENT, and cmd_extract's fallback for a
#       missing donor column is a silent np.array(["_"] * N).  So the donor column is
#       registered EXPLICITLY as 'dataset_id', whose value is the donor-identifying
#       string 'islet_HPAP-104' ('batch_id' is the constant 'batch_islet' in every file
#       and is useless as a donor key).
#   (2) LABEL: there is no cell-type annotation in these files at all, so
#       `get_dataset_ids("cell_type")` would raise KeyError and stop the extract.  The
#       registered label column is None, which fills cell_type with UNLABELED_CT.  That
#       value is INSIDE diag_common.UNKNOWN, so `labeled_mask` DROPS every such cell at
#       score time -- an accidental `score` on these sets yields zero scored cells
#       (loud) instead of collapsing `dataset x cell_type` into `dataset` (silent).
#       The Fig 5 consumer does not need it: donor / stage / cell_type all come from
#       beta_meta.csv, joined on barcode, which is what _build_islet_beta_hpap.py built
#       from Donor_Summary_190.xlsx and the per-donor celltype.tsv files.
#
# Cell counts are the h5ad n_obs, MEASURED, not the cohort's Beta subset -- `extract`
# embeds the whole file.  Sum = 19,214 cells; ~1 GPU-h on one h200.
ISLET_EXPANDED_N: Dict[str, int] = {
    "islet_HPAP-095": 1179, "islet_HPAP-104": 3028, "islet_HPAP-129": 1908,
    "islet_HPAP-130": 2299, "islet_HPAP-131":  765, "islet_HPAP-135": 1189,
    "islet_HPAP-136": 1706, "islet_HPAP-137":  681, "islet_HPAP-139": 2517,
    "islet_HPAP-141": 1508, "islet_HPAP-146": 2434,
}
_EXPANDED_ROOT = f"{REPO}/preprocessed_data_expanded_20260305"

# The sentinel written into `cell_type` for a set with no annotation.  It MUST stay a
# member of diag_common.UNKNOWN ({"Unknown", "unknown", "Unassigned", "", "nan", "NaN",
# "None", "unk"}); changing it to anything outside that set turns the loud failure above
# into a silent stratum collapse.
UNLABELED_CT = "unknown"

# obs column each set's cell_type comes from.  None == the set carries no annotation.
# Only the 11 donor files use None; every registered set keeps "cell_type".
LABEL_COL: Dict[str, Optional[str]] = {n: None for n in ISLET_EXPANDED_N}

# Extend the four tables IN PLACE, immediately after their definitions and before any
# use, so the original four stay byte-identical and the diff is one reviewable block.
IN_TRAINING = IN_TRAINING + tuple(sorted(ISLET_EXPANDED_N))          # noqa: F811
EXPECTED_N.update(ISLET_EXPANDED_N)
DS.update({n: (f"{_EXPANDED_ROOT}/{n}_rna_paired.h5ad",
               f"{_EXPANDED_ROOT}/{n}_atac_paired.h5ad", "obs")
           for n in ISLET_EXPANDED_N})
DONOR_COL.update({n: "dataset_id" for n in ISLET_EXPANDED_N})
assert not (set(ISLET_EXPANDED_N) & set(OOD_TRUE)), \
    "an expanded islet donor collided with the OOD panel"
assert UNLABELED_CT in ("Unknown", "unknown", "Unassigned", "", "nan", "NaN", "None",
                        "unk"), "UNLABELED_CT must be inside diag_common.UNKNOWN"


# The four strata, coarsest first.  `dataset x cell_type` is the PRIMARY; the other
# three are printed beside it because the pool definition IS the metric (0.1616 / 0.1323
# / 0.0329 for the same cells and the same model -- a 4.9x spread).
STRATA: Dict[str, Tuple[str, ...]] = {
    "dataset":                     ("dataset",),
    "dataset x donor":             ("dataset", "donor"),
    "dataset x cell_type":         ("dataset", "cell_type"),
    "dataset x donor x cell_type": ("dataset", "donor", "cell_type"),
}
PRIMARY_STRATUM = "dataset x cell_type"

# The stratum whose pools are single-donor BY CONSTRUCTION.  The gallery-only control
# needs the centring group to be constant inside a pool, otherwise the shift is not a
# per-row constant and the rank identity does not apply.
SINGLE_DONOR_STRATUM = "dataset x donor x cell_type"

CENTER_ARMS: Tuple[str, ...] = ("off", "donor_self", "donor_ext", "set_ext")

# `refiner_ood_center.py` splits the cells 50/50 with RandomState(0): the `stat` half is
# the external reference for ctr_donor_ext / ctr_set_ext, the `ev` half is scored.
# Every centring arm -- including `off` -- is scored on the SAME `ev` cells with the
# SAME pool draws, which is what makes the ladder paired.  Reproduced here exactly.
CENTER_SPLIT_SEED = 0


class PanelError(RuntimeError):
    """Raised when the resolved evaluation panel is not the registered one."""


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is not scoreable (leaky, or the wrong step)."""


# ------------------------------------------------------------------------------------ #
# Gates
# ------------------------------------------------------------------------------------ #

def resolve_panel(names: Sequence[str], require_full: bool = True
                  ) -> Tuple[List[str], List[str]]:
    """-> (ood, in_training).  REFUSES a panel that is not exactly the registered four.

    The assert the design registers is `len(OOD) == 4`, but a bare length check passes
    on {bmmc, breast, fetal_heart, islet}, which silently swaps a training set into the
    panel.  So identity is checked too, and unknown names are refused outright rather
    than being dropped -- a typo'd set name must not shrink the panel in silence.
    """
    names = [str(n) for n in names]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise PanelError(f"duplicate set names in the panel: {dup}")
    unknown = [n for n in names if n not in OOD_TRUE and n not in IN_TRAINING]
    if unknown:
        raise PanelError(
            f"unknown evaluation set(s) {unknown}; known OOD {list(OOD_TRUE)}, known "
            f"in-training {list(IN_TRAINING)}. Add it to DS/EXPECTED_N deliberately, "
            f"with its cell count, rather than letting an unrecognised name through.")
    ood = [n for n in names if n in OOD_TRUE]
    intr = [n for n in names if n in IN_TRAINING]
    if require_full and set(ood) != set(OOD_TRUE):
        missing = sorted(set(OOD_TRUE) - set(ood))
        raise PanelError(
            f"len(OOD)=={len(ood)}, expected 4. Resolved OOD panel "
            f"{sorted(ood)} != the registered {sorted(OOD_TRUE)} (missing "
            f"{missing}). ONLY breast, liver, fetal_heart and bmmc are true OOD; "
            f"islet and pln are IN the 163-dataset "
            f"training corpus and cannot stand in for a missing set. Pass "
            f"--allow_partial_panel to score a subset -- it is then NOT the registered "
            f"panel and no panel mean is emitted.")
    assert len(ood) == 4 or not require_full          # the registered assert, literally
    return ood, intr


def refuse_best_checkpoint(path: str) -> None:
    """⛔ Any path containing 'best' is refused, at extract AND at score time.

    `best_model.pt` is min-val-loss selected and on the projector track was mirrored
    from `best_ood_model.pt`, chosen on bmmc + fetal_heart -- half of this panel.  The
    refusal is on the whole path, not just the basename, because a `best_ood/` directory
    leaks exactly as much as a `best_ood.pt` file.
    """
    if "best" in str(path).lower():
        raise CheckpointError(
            f"REFUSED: {path}\n"
            f"  a checkpoint path containing 'best' is selected on a quantity "
            f"that leaks the OOD panel (best_model.pt is mirrored from "
            f"best_ood_model.pt, selected on bmmc + fetal_heart = HALF this panel). "
            f"Score a FIXED-step snapshot, "
            f"e.g. .../snapshots/step_008000.pt, identical across arms.")


def assert_checkpoint_step(path: str, expect_step: int,
                           recorded_step: Optional[int] = None) -> int:
    """Filename step == requested step == the step recorded INSIDE the file.

    All three, because each pair alone has a hole: a renamed file passes the filename
    check, a resumed run can write a file whose recorded step disagrees with its name,
    and comparing arms at different steps is the single easiest way to manufacture an
    effect.  `recorded_step` may be supplied by a caller that has already loaded the
    file (or by a metadata sidecar); otherwise the file is opened here.
    """
    refuse_best_checkpoint(path)
    m = re.search(r"step_(\d+)", os.path.basename(str(path)))
    if not m:
        raise CheckpointError(
            f"REFUSED: {path} has no step_<N> in its basename. The arms are "
            f"compared at a FIXED step and the step must be visible in the path.")
    fname_step = int(m.group(1))
    if fname_step != int(expect_step):
        raise CheckpointError(
            f"REFUSED: {path} is step {fname_step}, --expect_step is {expect_step}.")
    if recorded_step is None and os.path.exists(path):
        import torch                        # local: score-only runs need no torch
        recorded_step = int(torch.load(path, map_location="cpu",
                                       weights_only=False)["step"])
    if recorded_step is not None and int(recorded_step) != fname_step:
        raise CheckpointError(
            f"REFUSED: {path} is named step {fname_step} but records step "
            f"{recorded_step} inside. A renamed file must not pass silently.")
    return fname_step


# ------------------------------------------------------------------------------------ #
# Label granularity -- the cross-set comparability gate
# ------------------------------------------------------------------------------------ #

def label_granularity(cell_type: np.ndarray) -> Dict[str, float]:
    """How fine is this set's annotation?

    Within-type retrieval removes cross-type confusion BY CONSTRUCTION, so chance is
    exactly 1/pool whatever the labels are -- but the DIFFICULTY is not label-invariant.
    A coarse label leaves real sub-type structure inside the pool for the model to use,
    which this project measured as making within-type R@1 47-66% easier.  So a
    difference between two SETS is confounded with their annotation depth, and the
    honest unit of comparison is arm-vs-arm WITHIN a set.

    `n_eff` (exp of the label entropy) is the number to compare, not `n_types`: a set
    with 40 types of which 2 hold 95% of the cells behaves like a 3-type set.
    """
    ct = np.asarray(cell_type, dtype=str)
    _, cnt = np.unique(ct, return_counts=True)
    p = cnt / cnt.sum()
    ent = float(-(p * np.log(p)).sum())
    return {"n_cells": int(cnt.sum()), "n_types": int(len(cnt)),
            "entropy_nats": ent, "n_eff_types": float(math.exp(ent)),
            "median_cells_per_type": float(np.median(cnt)),
            "largest_type_frac": float(cnt.max() / cnt.sum())}


def check_matched_granularity(stats: Dict[str, Dict[str, float]],
                              max_ratio: float = 2.0,
                              strict: bool = False) -> Dict[str, object]:
    """Compare n_eff_types across sets; flag (or refuse) an unmatched cross-set panel.

    This does NOT invalidate an arm contrast -- arms share the labels exactly, so B - A
    within a set is granularity-free.  What it gates is reading a per-SET R@1 against
    another set's, and the unweighted panel MEAN, which silently weights whichever set
    is easiest.
    """
    if not stats:
        return {"applicable": False}
    ne = {k: v["n_eff_types"] for k, v in stats.items()}
    lo_k, hi_k = min(ne, key=ne.get), max(ne, key=ne.get)
    ratio = ne[hi_k] / max(ne[lo_k], 1e-9)
    matched = bool(ratio <= max_ratio)
    out = {"applicable": True, "n_eff_types": ne, "ratio": float(ratio),
           "max_ratio": float(max_ratio), "matched": matched,
           "coarsest": lo_k, "finest": hi_k}
    if not matched:
        msg = (f"LABEL GRANULARITY NOT MATCHED across sets: n_eff_types spans "
               f"{ne[lo_k]:.1f} ({lo_k}) to {ne[hi_k]:.1f} ({hi_k}), ratio "
               f"{ratio:.2f} > {max_ratio}. Coarse labels make within-type retrieval "
               f"47-66% EASIER, so a per-set R@1 read against another set's is "
               f"confounded with annotation depth. Arm-vs-arm WITHIN a set is "
               f"unaffected; the unweighted panel mean is reported but must be "
               f"quoted with this caveat.")
        if strict:
            raise PanelError(msg)
        out["warning"] = msg
    return out


# ------------------------------------------------------------------------------------ #
# Borrowed machinery.  Imported, never reimplemented -- and the source file is printed
# so the log records WHICH copy produced the numbers.
# ------------------------------------------------------------------------------------ #

_REUSED: Dict[str, object] = {}


def reused() -> Dict[str, object]:
    """Resolve the shared metric + centring functions, once, and report where from.

    `refiner_ood_center` is a ~17 s import (it drags in scanpy and the FineLIP package)
    so it is deferred to first use rather than paid at module import: the self-test and
    every gate above run without it.

    ⛔ `per_group_center` is the ONLY centring implementation this file may use.  It is
    the correct variant -- subtract the per-group mean from BOTH modalities and
    re-normalise (`l2np`) -- and memory says the re-normalisation is what carries the
    gain.  Re-typing four lines here would be how the two implementations quietly drift.
    """
    if _REUSED:
        return _REUSED
    from diag_common import l2np, labeled_mask, UNKNOWN            # noqa: E402
    from diag_b_ceiling import make_group_pools, per_draw_r1       # noqa: E402
    import refiner_ood_center as _roc                              # noqa: E402
    _REUSED.update(
        l2np=l2np, labeled_mask=labeled_mask, UNKNOWN=UNKNOWN,
        make_group_pools=make_group_pools, per_draw_r1=per_draw_r1,
        per_group_center=_roc.per_group_center,
        _src=dict(center=os.path.abspath(_roc.__file__),
                  pools=os.path.abspath(sys.modules["diag_b_ceiling"].__file__),
                  common=os.path.abspath(sys.modules["diag_common"].__file__)))
    return _REUSED


def print_provenance(log=print) -> None:
    r = reused()["_src"]
    log("  reused code:")
    log(f"    per_group_center            <- {r['center']}")
    log(f"    make_group_pools/per_draw_r1<- {r['pools']}")
    log(f"    l2np/labeled_mask           <- {r['common']}")


# ------------------------------------------------------------------------------------ #
# Strata, pools, and per-direction R@1
# ------------------------------------------------------------------------------------ #

def stratum_key(cols: Dict[str, np.ndarray], fields: Sequence[str]) -> np.ndarray:
    """Composite group key, built in PYTHON.

    ⛔ NOT `np.char.add`: `obs[col].astype(str).values` is OBJECT dtype and np.char.*
    raises on it.  This project has hit that trap at least four times, and the failure
    mode when it is worked around badly (silently casting) is a key that collides.
    """
    parts = [np.asarray(cols[f], dtype=str) for f in fields]
    return np.array(["|".join(t) for t in zip(*parts)], dtype=str)


def pool_seed(dataset: str, stratum: str, extra: str = "") -> int:
    """A STABLE seed for the pool draws.

    Python's `hash()` of a str is salted per process, so seeding from it would hand
    every arm a different pool set and void the paired comparison without any visible
    symptom.
    """
    blob = f"{dataset}|{stratum}|{extra}".encode()
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


def _r1_both_from_S(S) -> Tuple[float, float]:
    """(rna->atac, atac->rna) R@1 for one pool, aligned by construction.

    Tie handling is `argsort(descending)` + `argmax` of the match indicator -- byte for
    byte what `diag_b_ceiling.per_draw_r1` does, so the generic path below and the
    borrowed one cannot disagree (asserted in the self-test).
    """
    import torch
    lab = np.arange(S.shape[0])
    out = []
    for M in (S, S.t()):
        order = M.argsort(1, descending=True)
        rk = (order == torch.as_tensor(lab)[:, None]).float().argmax(1)
        out.append(float((rk == 0).float().mean().item()))
    return out[0], out[1]


def per_draw_r1_generic(sim_fn, idxp) -> Tuple[np.ndarray, np.ndarray]:
    """Per-direction R@1 per draw for an ARBITRARY similarity, e.g. the slot scorer.

    `diag_b_ceiling.per_draw_r1` is hard-wired to `zr[i] @ za[i].t()` and is used
    unchanged for the GLOBAL scorer (it is the reference implementation and the arm
    contrast rides on it).  Everything that is not an inner product of two [N,D]
    matrices -- slots, the fused score, the depth null -- comes through here instead.
    """
    a, b = [], []
    for idx in idxp:
        r, s = _r1_both_from_S(sim_fn(np.asarray(idx)))
        a.append(r)
        b.append(s)
    return np.asarray(a), np.asarray(b)


def cluster_se(d: np.ndarray, gname: Sequence[str]) -> Tuple[float, int]:
    """Cluster-robust SE of mean(d), clustering draws by the GROUP they were drawn from.

    Draws are not independent: `make_group_pools` picks a group at random and then a
    random subset of it, so two draws from the same group re-use most of the same cells.
    The naive std/sqrt(n) therefore understates.  Clustering on the group name is the
    same correction `axis1_fine_summary.csv`'s `mde` column uses.
    """
    d = np.asarray(d, dtype=np.float64)
    g = np.asarray(gname, dtype=str)
    n = len(d)
    if n < 2:
        return float("nan"), 0
    dbar = d.mean()
    groups = np.unique(g)
    ssq = 0.0
    for gg in groups:
        ssq += float(((d[g == gg] - dbar).sum()) ** 2)
    if len(groups) < 2:
        return float(d.std(ddof=1) / math.sqrt(n)), 1
    var = ssq * len(groups) / max(len(groups) - 1, 1) / (n ** 2)
    return float(math.sqrt(max(var, 0.0))), int(len(groups))


def two_sided_p(t: float, df: int) -> float:
    """Two-sided p from a t statistic, without scipy (not guaranteed in this env)."""
    if not np.isfinite(t) or df < 1:
        return float("nan")
    x = df / (df + t * t)
    return float(_betainc_half(0.5 * df, 0.5, x))


def _betainc_half(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) by continued fraction (Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _bcf(a, b, x) / a
    return 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) * \
        _bcf(b, a, 1 - x) / b


def _bcf(a: float, b: float, x: float, itmax: int = 300, eps: float = 1e-12) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


# ------------------------------------------------------------------------------------ #
# Test-time centring
# ------------------------------------------------------------------------------------ #

def apply_test_centering(arm: str, zr, za, donor_ev, zr_ref=None, za_ref=None,
                         donor_ref=None):
    """One rung of the centring ladder, all through the REUSED `per_group_center`.

    off          nothing
    donor_self   per-donor mean of the EVAL cells themselves (transductive)
    donor_ext    per-donor mean from a DISJOINT half of the same donor  <- deployable
    set_ext      one mean for the whole set, from the disjoint half

    `donor_ext` needs m >= 128 cells per donor in the reference half to reach 95.9% of
    the full-donor gain; only m = 32 loses measurably.  That is checked and reported by
    the caller, not silently accepted.
    """
    R = reused()
    pgc, l2np = R["per_group_center"], R["l2np"]
    if arm == "off":
        return zr, za
    if arm == "donor_self":
        return pgc(zr, donor_ev), pgc(za, donor_ev)
    if arm == "donor_ext":
        if zr_ref is None:
            raise ValueError("donor_ext needs a disjoint reference half")
        return (pgc(zr, donor_ev, zr_ref, donor_ref),
                pgc(za, donor_ev, za_ref, donor_ref))
    if arm == "set_ext":
        if zr_ref is None:
            raise ValueError("set_ext needs a disjoint reference half")
        return (l2np(zr - zr_ref.mean(0, keepdims=True)),
                l2np(za - za_ref.mean(0, keepdims=True)))
    raise ValueError(f"unknown centring arm {arm!r}")


def centering_controls(zr, za, group, idxp, gname, pool: int) -> Dict[str, object]:
    """⛔ IS THE CENTRING SWITCH A NO-OP?  Two-directional, on the same pools.

    C1 (must be EXACTLY 0).  Shift the GALLERY of one direction by a constant, without
        re-normalising.  For a fixed query row the whole row moves by -q.mu, so nothing
        can reorder: dR@1 must be 0.0 bit-exactly.  If it is not, the scorer is not a
        plain inner-product ranking and every rank identity this project relies on is
        void.  This is the "gallery-only centring is a bit-identical NO-OP" claim, run
        as a test rather than quoted.
    C2 (must be NON-zero).  The real `per_group_center`: both modalities, re-normalised.
        If C2 is 0.0 while C1 is 0.0, the centring flag is inert and every "centred"
        number in the run is the raw number under a different name.

    The pools must be single-group for C1's premise to hold (the shift has to be
    constant inside a pool), so the caller passes the single-donor stratum's draws.
    """
    R = reused()
    pgc, per_draw_r1 = R["per_group_center"], R["per_draw_r1"]
    grp = np.asarray(group, dtype=str)
    means = {g: zr[grp == g].mean(0) for g in np.unique(grp)}
    ameans = {g: za[grp == g].mean(0) for g in np.unique(grp)}
    shift_a = np.stack([ameans[g] for g in grp])       # gallery for rna->atac
    shift_r = np.stack([means[g] for g in grp])        # gallery for atac->rna

    base_r2a, base_a2r = per_draw_r1(zr, za, idxp)
    # C1a: shift ONLY the atac side, no renorm -> rna->atac must not move at all.
    c1a_r2a, _ = per_draw_r1(zr, (za - shift_a).astype(np.float32), idxp)
    # C1b: shift ONLY the rna side, no renorm -> atac->rna must not move at all.
    _, c1b_a2r = per_draw_r1((zr - shift_r).astype(np.float32), za, idxp)
    cz, ca = pgc(zr, grp), pgc(za, grp)
    c2_r2a, c2_a2r = per_draw_r1(cz, ca, idxp)

    d1a = float(np.abs(c1a_r2a - base_r2a).max())
    d1b = float(np.abs(c1b_a2r - base_a2r).max())
    d2 = max(float(np.abs(c2_r2a - base_r2a).mean()),
             float(np.abs(c2_a2r - base_a2r).mean()))
    # A single pair flipping in the worst draw moves that draw's R@1 by exactly 1/pool,
    # so `<= 1/pool` is the widest a NEAR-TIE explanation can be. Anything above that is
    # a scorer that is not ranking by inner product, which would void the identity the
    # whole centring story rests on. On synthetic data with well-separated scores the
    # value is 0.0 exactly.
    tol = 1.0 / float(pool)
    exact1 = (d1a == 0.0) and (d1b == 0.0)
    ok1 = max(d1a, d1b) <= tol
    ok2 = d2 > 0.0
    if not ok1:
        verdict = ("C1 FAILED: a gallery-only constant shift moved R@1 by more than "
                   f"1/pool ({max(d1a, d1b):.4f} > {tol:.4f}); the scorer is not a "
                   "plain inner-product ranking and no rank identity here holds")
    elif not ok2:
        verdict = ("C2 FAILED: the real both-sides re-normalised centring changed "
                   "NOTHING. The centring switch is INERT and every 'centred' "
                   "number in this run is the raw number under a different name")
    elif not exact1:
        verdict = f"OK (C1 non-zero at {max(d1a, d1b):.4f} <= 1/pool: near-tie flips)"
    else:
        verdict = "OK"
    return {"n_draws": int(len(idxp)), "pool": int(pool),
            "C1_gallery_only_no_renorm_max_abs_delta_r2a": d1a,
            "C1_gallery_only_no_renorm_max_abs_delta_a2r": d1b,
            "C1_exactly_zero": bool(exact1), "C1_pass": bool(ok1),
            "C2_both_sides_renorm_mean_abs_delta": d2,
            "C2_pass_is_nonzero": bool(ok2),
            "verdict": verdict}


# ------------------------------------------------------------------------------------ #
# The embedding container -- the seam between the GPU half and the CPU half
# ------------------------------------------------------------------------------------ #
#
# `extract` runs the live FMs once per checkpoint (~4 GPU-h at max_atac_length 8192; the
# OOD sets are NOT in the 8.7 TB token cache, which holds only train/ and val/) and
# writes one .npz per set.  `score` then reads those and every stratum, every centring
# rung and every null is free on CPU.  Keeping the seam explicit is also what makes the
# harness testable: the self-test below feeds it synthetic embeddings with a KNOWN
# retrieval ordering and checks the reported R@1 against arithmetic, not against
# another script.
#
# npz contract (one file per evaluation set, named <set>.npz):
#     rna        [N, D] float32, L2-normalised   the cell embedding the loss saw
#     atac       [N, D] float32, L2-normalised
#     barcode    [N] str      dataset [N] str      donor [N] str     cell_type [N] str
#     rna_depth  [N] float32  optional -- log valid-token count, for the depth-only null
#     atac_depth [N] float32  optional
#     rna_slots  [N, M, D] float32   ARM B / ARM C only
#     atac_slots [N, M, D] float32
#     rna_valid  [N, M] bool      atac_valid [N, M] bool
#     rna_mass   [N, M] float32   atac_mass  [N, M] float32
# meta.json sits beside them and carries arm / step / checkpoint / train-time centring.

EMB_REQUIRED = ("rna", "atac", "barcode", "dataset", "donor", "cell_type")


def write_emb_npz(path: str, **arrays) -> None:
    missing = [k for k in EMB_REQUIRED if k not in arrays]
    assert not missing, f"embedding npz is missing {missing}"
    n = len(arrays["rna"])
    for k, v in arrays.items():
        # 0-d entries are per-RUN metadata, not per-cell arrays: the finecls path stores
        # routing_topk / routing_mass_power / routing_tail_weight as np.array(scalar) so
        # they travel with the embeddings.  `len()` on those raises
        # `TypeError: len() of unsized object`, which is what this assert did on every
        # ARM B/C extract -- ARM A never hit it because it writes no slot arrays.
        # Exempt TRUE scalars only; everything array-like still has to match rna's rows,
        # because a silently short column is the bug this assert exists to catch.
        if getattr(v, "ndim", None) == 0:
            continue
        assert len(v) == n, f"{k} has {len(v)} rows, rna has {n}"
    # ⛔ OBJECT ARRAYS CANNOT BE READ BACK.  `barcode` and `donor` arrive as dtype=object
    # (string columns out of h5py/pandas); np.savez writes them happily, and np.load then
    # refuses with "Object arrays cannot be loaded when allow_pickle=False".  MEASURED:
    # every interim npz had object `barcode` and `donor` while `dataset`/`cell_type` were
    # already <U, so `extract` succeeded and `score` died on the first file it opened.
    # Coerce to fixed-width unicode HERE, so what is written is readable by construction
    # rather than by a pickle flag at the far end.
    clean = {}
    for k, v in arrays.items():
        if getattr(v, "dtype", None) == object:
            bad = [x for x in v[:64] if not isinstance(x, (str, bytes, bytearray))]
            assert not bad, f"{k}: object array holding non-strings, e.g. {bad[0]!r}"
            v = np.asarray([x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                            for x in v])
        clean[k] = v
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **clean)


def load_emb_dir(d: str) -> Tuple[Dict[str, object], Dict[str, Dict[str, np.ndarray]]]:
    meta_p = os.path.join(d, "meta.json")
    assert os.path.exists(meta_p), f"{d}: no meta.json (was this written by `extract`?)"
    with open(meta_p) as fh:
        meta = json.load(fh)
    sets = {}
    for name in sorted(meta["sets"]):
        f = os.path.join(d, f"{name}.npz")
        assert os.path.exists(f), f"meta.json lists {name} but {f} is missing"
        # allow_pickle=True reads the npz files written BEFORE the coercion above,
        # whose `barcode`/`donor` are dtype=object.  These are our own artifacts on
        # our own filesystem, so the pickle path is not a trust boundary -- and it
        # saves re-running ~3 h of GPU extraction to change a dtype.  New files are
        # written as <U by write_emb_npz and do not need it.
        z = np.load(f, allow_pickle=True)
        sets[name] = {k: z[k] for k in z.files}
    return meta, sets


# ------------------------------------------------------------------------------------ #
# Scoring one set
# ------------------------------------------------------------------------------------ #

def _slot_wmean(slots, valid, rows=None):
    """Validity-weighted per-slot mean, exactly as `_center_slots_by_group` computes it.

    Only cells where a slot is ACTIVE contribute to that slot's mean.  This is not a
    refinement -- it is required for agreement with training: one real RNA slot is active
    in just 20.7% of bmmc cells, so an unweighted mean over all cells is dominated by
    rows that never enter the similarity.
    """
    s = slots if rows is None else slots[rows]
    w = (valid if rows is None else valid[rows]).astype(np.float32)[..., None]
    denom = w.sum(0, keepdims=True)
    np.maximum(denom, 1.0, out=denom)
    return (s * w).sum(0, keepdims=True) / denom


def _center_slots(arm: str, rs, as_, rv, av, donor_ev, rs_ref=None, as_ref=None,
                  rv_ref=None, av_ref=None, donor_ref=None):
    """The centring rung applied to EVERY slot, matching the TRAINING helper.

    ⛔ WHY THIS EXISTS.  `slot_fn_all` used to be built ONCE, outside the centring loop,
    so the slot similarity was byte-identical across all four rungs (measured: slot r2a
    0.0905 at off / donor_self / donor_ext / set_ext) while the global scorer moved
    0.0660 -> 0.1418.  `fused` therefore mixed a CENTRED global with an UNCENTRED slot,
    which is not a fusion of comparable scores and made FineCLS look worse than it is.

    ⛔ WHY NOT `per_group_center`.  That is the ONLY permitted implementation for the
    GLOBAL branch, but the slot branch has its own in training -- `_center_slots_by_group`
    -- which differs in two ways that matter:
      1. the mean is VALIDITY-WEIGHTED per slot (see `_slot_wmean`);
      2. it does NOT re-normalise, because `fixed_slot_similarity` applies
         `F.normalize(..., dim=-1)` internally.  Re-normalising here would be harmless
         (normalize is idempotent) but the validity weighting is NOT.
      3. inactive slots are left UNTOUCHED, never shifted by another slot's mean.
    """
    if arm == "off":
        return rs, as_
    if arm in ("donor_ext", "set_ext") and rs_ref is None:
        raise ValueError(f"{arm} needs a disjoint reference half")

    def one(z, v, z_ref, v_ref):
        out = z.copy()
        if arm == "donor_self":
            for d in np.unique(donor_ev):
                m = donor_ev == d
                if int(m.sum()) <= 1:
                    continue
                mu = _slot_wmean(z, v, m)
                out[m] = np.where(v[m][..., None], z[m] - mu, z[m])
        elif arm == "donor_ext":
            for d in np.unique(donor_ev):
                m = donor_ev == d
                mr = donor_ref == d
                if not mr.any():
                    continue
                mu = _slot_wmean(z_ref, v_ref, mr)
                out[m] = np.where(v[m][..., None], z[m] - mu, z[m])
        elif arm == "set_ext":
            mu = _slot_wmean(z_ref, v_ref)
            out = np.where(v[..., None], z - mu, z)
        else:
            raise ValueError(f"unknown centring arm {arm!r}")
        return out

    return one(rs, rv, rs_ref, rv_ref), one(as_, av, as_ref, av_ref)


def _slot_sim_fn(arrays, idx_all, rs_over=None, as_over=None):
    """Return a callable idx -> [P,P] slot similarity, or None when there are no slots.

    `fixed_slot_similarity` is imported from the model package, not reimplemented: it is
    the same routing (`topk` + `tail_weight`) and the same internal `F.normalize` the
    training loss uses, so the eval number and the train objective are the same object.
    """
    if "rna_slots" not in arrays:
        return None
    import torch
    _bootstrap_model_path()
    from multiomics_clip_xinyu_June_fixed_slot_routing.modules.fixed_slot import (
        fixed_slot_similarity)
    rs = torch.as_tensor(arrays["rna_slots"][idx_all] if rs_over is None else rs_over)
    as_ = torch.as_tensor(arrays["atac_slots"][idx_all] if as_over is None else as_over)
    rv = torch.as_tensor(arrays["rna_valid"][idx_all])
    av = torch.as_tensor(arrays["atac_valid"][idx_all])
    # Both masses or neither: `fixed_slot_similarity` falls back to uniform slot
    # weights when either is None, so a half-populated npz would silently switch the
    # eval to a routing the training loss never used.
    has_mass = "rna_mass" in arrays and "atac_mass" in arrays
    rm = torch.as_tensor(arrays["rna_mass"][idx_all]) if has_mass else None
    am = torch.as_tensor(arrays["atac_mass"][idx_all]) if has_mass else None
    routing = dict(routing_topk=int(arrays.get("routing_topk", np.array(0))),
                   routing_mass_power=float(arrays.get("routing_mass_power",
                                                       np.array(1.0))),
                   routing_tail_weight=float(arrays.get("routing_tail_weight",
                                                        np.array(0.0))))

    def fn(idx):
        i = torch.as_tensor(np.asarray(idx))
        return fixed_slot_similarity(rs[i], as_[i], rv[i], av[i],
                                     None if rm is None else rm[i],
                                     None if am is None else am[i], **routing)
    return fn


def score_one_set(name: str, arrays: Dict[str, np.ndarray], pool: int, draws: int,
                  center_arms: Sequence[str], center_split: bool, slot_draws: int,
                  log=print) -> Dict[str, object]:
    """All strata x all centring rungs x all scorers, on ONE set, with SHARED pools."""
    import torch
    R = reused()
    l2np, labeled_mask = R["l2np"], R["labeled_mask"]
    make_group_pools, per_draw_r1 = R["make_group_pools"], R["per_draw_r1"]

    import pandas as pd
    df = pd.DataFrame({"cell_barcode": arrays["barcode"].astype(str),
                       "dataset": arrays["dataset"].astype(str),
                       "donor": arrays["donor"].astype(str),
                       "cell_type": arrays["cell_type"].astype(str)})
    # Drop unlabeled ONCE, up front, so all four strata are computed on the IDENTICAL
    # cells.  If the coarse stratum kept cells the fine one cannot use, the 4.9x pool
    # spread would be confounded with a different denominator.
    keep = labeled_mask(df)
    idx_all = np.where(keep)[0]
    df = df.loc[keep].reset_index(drop=True)
    zr_all = np.ascontiguousarray(arrays["rna"][idx_all], dtype=np.float32)
    za_all = np.ascontiguousarray(arrays["atac"][idx_all], dtype=np.float32)
    n_lab = len(idx_all)

    # The 50/50 split of refiner_ood_center.py: `stat` is the external reference half
    # for donor_ext / set_ext, `ev` is scored.  Every rung -- INCLUDING `off` -- is
    # scored on the same `ev` cells with the same draws, which is what makes the ladder
    # paired.
    if center_split:
        rng = np.random.RandomState(CENTER_SPLIT_SEED)
        stat = rng.rand(n_lab) < 0.5
    else:
        stat = np.zeros(n_lab, bool)
    ev = ~stat
    ev_idx = np.where(ev)[0]
    cols = {k: df[k].values for k in ("dataset", "donor", "cell_type")}
    cols_ev = {k: v[ev_idx] for k, v in cols.items()}
    zr, za = zr_all[ev_idx], za_all[ev_idx]
    donor_ev = np.asarray(cols_ev["donor"], dtype=str)
    zr_ref = za_ref = donor_ref = None
    if center_split:
        st_idx = np.where(stat)[0]
        zr_ref, za_ref = zr_all[st_idx], za_all[st_idx]
        donor_ref = np.asarray(cols["donor"][st_idx], dtype=str)

    gran = label_granularity(cols_ev["cell_type"])
    nd = {d: int((donor_ev == d).sum()) for d in np.unique(donor_ev)}
    small_ref = {}
    if center_split:
        small_ref = {d: int((donor_ref == d).sum()) for d in np.unique(donor_ref)
                     if (donor_ref == d).sum() < 128}
    log(f"\n[{name}] {len(arrays['rna']):,} cells -> {n_lab:,} labelled -> "
        f"{ev.sum():,} scored ({'50/50 split' if center_split else 'no split'}) | "
        f"{len(nd)} donors | {gran['n_types']} types (n_eff {gran['n_eff_types']:.1f})")
    if small_ref:
        log(f"    ⚠ donors with < 128 reference cells (donor_ext loses power there): "
            f"{sorted(small_ref.items())[:6]}{' ...' if len(small_ref) > 6 else ''}")

    # ---- pools: built ONCE per stratum, reused by every arm and every rung ----------
    pools: Dict[str, Tuple[List[np.ndarray], List[str]]] = {}
    for stratum, fields in STRATA.items():
        keys = stratum_key(cols_ev, fields)
        idxp, gname = make_group_pools(keys, pool, draws,
                                       seed=pool_seed(name, stratum))
        pools[stratum] = (idxp, gname)
        uk, cnt = np.unique(keys, return_counts=True)
        big = int((cnt >= pool).sum())
        log(f"    {stratum:30s} {big:4d}/{len(uk):4d} groups >= {pool} cells  "
            f"(median {int(np.median(cnt))})"
            + ("   <- PRIMARY" if stratum == PRIMARY_STRATUM else "")
            + ("   [NO USABLE POOL]" if not idxp else ""))

    # Per-CENTRING-RUNG slot scorer.  Built lazily and cached: each rung needs its own
    # centred copy of the [N, M, D] slots, and all four strata reuse the same rung.
    _ev_abs = idx_all[ev_idx]
    _ref_abs = idx_all[np.where(stat)[0]] if center_split else None
    _slot_cache: Dict[str, object] = {}

    def slot_fn_for(arm: str):
        if arm in _slot_cache:
            return _slot_cache[arm]
        if "rna_slots" not in arrays:
            _slot_cache[arm] = None
            return None
        rs = np.asarray(arrays["rna_slots"][_ev_abs], np.float32)
        as_ = np.asarray(arrays["atac_slots"][_ev_abs], np.float32)
        rv = np.asarray(arrays["rna_valid"][_ev_abs], bool)
        av = np.asarray(arrays["atac_valid"][_ev_abs], bool)
        if arm != "off":
            rs_ref = as_ref = rv_ref = av_ref = None
            if _ref_abs is not None:
                rs_ref = np.asarray(arrays["rna_slots"][_ref_abs], np.float32)
                as_ref = np.asarray(arrays["atac_slots"][_ref_abs], np.float32)
                rv_ref = np.asarray(arrays["rna_valid"][_ref_abs], bool)
                av_ref = np.asarray(arrays["atac_valid"][_ref_abs], bool)
            rs, as_ = _center_slots(arm, rs, as_, rv, av, donor_ev,
                                    rs_ref, as_ref, rv_ref, av_ref, donor_ref)
        _slot_cache[arm] = _slot_sim_fn(arrays, _ev_abs, rs_over=rs, as_over=as_)
        return _slot_cache[arm]

    slot_fn_all = slot_fn_for("off")   # kept for the has-slots test below
    depth = None
    if "rna_depth" in arrays and "atac_depth" in arrays:
        rd = np.asarray(arrays["rna_depth"], np.float32)[idx_all][ev_idx]
        ad = np.asarray(arrays["atac_depth"], np.float32)[idx_all][ev_idx]
        rz = (rd - rd.mean()) / max(rd.std(), 1e-6)
        az = (ad - ad.mean()) / max(ad.std(), 1e-6)
        depth = (rz, az)

    out: Dict[str, object] = {
        "set": name, "in_training": name in IN_TRAINING,
        "n_cells_total": int(len(arrays["rna"])), "n_labelled": int(n_lab),
        "n_scored": int(ev.sum()), "n_donors": len(nd), "pool": int(pool),
        "draws": int(draws), "chance": 1.0 / pool, "granularity": gran,
        "center_split": bool(center_split), "strata": {}}
    draw_arrays: Dict[str, np.ndarray] = {}

    for stratum, (idxp, gname) in pools.items():
        if not idxp:
            out["strata"][stratum] = {"skipped": "no group >= pool"}
            continue
        srec: Dict[str, object] = {"n_draws": len(idxp),
                                   "n_groups": int(len(set(gname))),
                                   "groups": sorted(set(gname))[:40]}
        # The group each draw came from, kept so the PAIRED arm contrast can cluster its
        # SE on it.  Draws from one group re-use most of the same cells, so the naive
        # std/sqrt(n) understates the paired delta's error.
        draw_arrays[f"{stratum}|__gname__"] = np.asarray(gname, dtype=str)
        for arm in center_arms:
            if arm in ("donor_ext", "set_ext") and not center_split:
                continue
            czr, cza = apply_test_centering(arm, zr, za, donor_ev,
                                            zr_ref, za_ref, donor_ref)
            rec: Dict[str, object] = {}
            # GLOBAL -- the arm-contrast scorer, defined in every arm.  Borrowed
            # implementation, used verbatim.
            r2a, a2r = per_draw_r1(czr, cza, idxp)
            rec["global"] = _summarise(r2a, a2r, gname)
            draw_arrays[f"{stratum}|{arm}|global|r2a"] = r2a
            draw_arrays[f"{stratum}|{arm}|global|a2r"] = a2r
            # PERM twin: the empirical floor for THIS score matrix.  It must land at
            # 1/pool; if it does not, the pool draw itself is confounded.
            prng = np.random.RandomState(pool_seed(name, stratum, "perm"))
            perm_fn = _perm_fn(czr, cza, prng)
            p2a, pa2 = per_draw_r1_generic(perm_fn, idxp)
            rec["perm"] = _summarise(p2a, pa2, gname)
            if depth is not None:
                rec["depth"] = _summarise(*per_draw_r1_generic(
                    _depth_fn(*depth), idxp), gname)
            if slot_fn_all is not None:
                sub = idxp[:min(slot_draws, len(idxp))]
                sg = gname[:len(sub)]
                # ⛔ the slot scorer gets the SAME centring rung as the global one.
                sfn = slot_fn_for(arm)
                s2a, sa2 = per_draw_r1_generic(sfn, sub)
                rec["slot"] = _summarise(s2a, sa2, sg)
                draw_arrays[f"{stratum}|{arm}|slot|r2a"] = s2a
                draw_arrays[f"{stratum}|{arm}|slot|a2r"] = sa2
                f2a, fa2 = per_draw_r1_generic(_fused_fn(czr, cza, sfn), sub)
                rec["fused"] = _summarise(f2a, fa2, sg)
                draw_arrays[f"{stratum}|{arm}|fused|r2a"] = f2a
                draw_arrays[f"{stratum}|{arm}|fused|a2r"] = fa2
            srec[arm] = rec
        out["strata"][stratum] = srec

    # ---- the gallery-only / renormalisation control, on single-donor pools ----------
    ctrl_stratum = SINGLE_DONOR_STRATUM if pools.get(SINGLE_DONOR_STRATUM, ([],))[0] \
        else None
    if ctrl_stratum is None:
        out["centering_controls"] = {
            "applicable": False,
            "why": f"no usable pool in {SINGLE_DONOR_STRATUM}; the gallery-only rank "
                   f"identity needs the centring group constant inside a pool"}
    else:
        idxp, gname = pools[ctrl_stratum]
        sub = idxp[:min(200, len(idxp))]
        out["centering_controls"] = {
            "applicable": True, "stratum": ctrl_stratum,
            **centering_controls(zr, za, donor_ev, sub, gname[:len(sub)], pool)}

    # ⛔ LIVENESS: centring MUST reach the slots.  The bug this replaces was
    # invisible for exactly one reason -- nothing ever compared the slot score
    # across rungs.  Assert on the quantity the fix controls, not on a proxy.
    if slot_fn_all is not None:
        for _st in STRATA:
            k0, k1 = f"{_st}|off|slot|r2a", f"{_st}|donor_self|slot|r2a"
            if k0 in draw_arrays and k1 in draw_arrays:
                d = float(abs(draw_arrays[k0].mean() - draw_arrays[k1].mean()))
                out["slot_centering_live"] = {"stratum": _st,
                                              "off_vs_donor_self": d}
                assert d > 0, (
                    f"[{name}/{_st}] slot r2a is IDENTICAL under `off` and "
                    f"`donor_self` -- centring is NOT reaching the slots. That is "
                    f"the defect `_center_slots` exists to fix.")
                break
    return out, draw_arrays


def _perm_fn(zr, za, rng):
    def fn(idx):
        import torch
        i = np.asarray(idx)
        S = torch.as_tensor(zr[i]) @ torch.as_tensor(za[i]).t()
        return S[:, torch.as_tensor(rng.permutation(len(i)))]
    return fn


def _depth_fn(rz, az):
    """The trivial explanation the PERM twin cannot rule out.

    Inside one cell type RNA depth and ATAC depth correlate because it is the same
    nucleus, so a scorer that matches only "how deep is this cell" beats the perm floor
    with zero regulatory content.  Permuting destroys the depth match along with
    everything else, which is exactly why perm cannot separate them.
    """
    def fn(idx):
        import torch
        i = np.asarray(idx)
        r = torch.as_tensor(rz[i])[:, None]
        return -(r - torch.as_tensor(az[i])[None, :]).abs()
    return fn


def _fused_fn(zr, za, slot_fn):
    """z-scored sum of the global and slot similarities, per pool.

    Per-pool z-scoring, not a global one: the two scores live on different scales and
    the pool is the unit the ranking happens in, so standardising anywhere else would
    let one pool's spread set another's weighting.
    """
    def fn(idx):
        import torch
        i = np.asarray(idx)
        g = torch.as_tensor(zr[i]) @ torch.as_tensor(za[i]).t()
        s = slot_fn(i)
        gz = (g - g.mean()) / g.std().clamp(min=1e-6)
        sz = (s - s.mean()) / s.std().clamp(min=1e-6)
        return gz + sz
    return fn


def _summarise(r2a: np.ndarray, a2r: np.ndarray, gname) -> Dict[str, float]:
    """⛔ PER DIRECTION, NEVER AVERAGED.

    `eval_ckpt_stratified.py:48` is documented "Symmetric R@1" and
    `diag_b_ceiling.score` returns (r2a + a2r) / 2.  Memory retired the +0.0372 centring
    headline precisely because it was direction-averaged.  Both directions are kept
    separate here and a verdict has to hold in both.
    """
    o: Dict[str, float] = {}
    for tag, v in (("rna2atac", r2a), ("atac2rna", a2r)):
        v = np.asarray(v, np.float64)
        cse, ncl = cluster_se(v, gname)
        o[f"r1_{tag}"] = float(v.mean())
        o[f"se_{tag}"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else \
            float("nan")
        o[f"cluster_se_{tag}"] = cse
        o[f"n_clusters_{tag}"] = ncl
    o["n_draws"] = int(len(r2a))
    return o


# ------------------------------------------------------------------------------------ #
# metrics.json, one per (arm, step, test-centring)
# ------------------------------------------------------------------------------------ #

SCHEMA = "finecls_refiner_ood_eval/v1"


def build_metrics(meta: Dict[str, object], per_set: Dict[str, Dict[str, object]],
                  center_arm: str, ood: Sequence[str], intr: Sequence[str],
                  gran_check: Dict[str, object], full_panel: bool) -> Dict[str, object]:
    """Slice the per-set records to ONE test-centring rung; add panel aggregates.

    The panel mean covers the four TRUE OOD sets only.  islet / pln are carried in the
    same file, with `in_training: true`, because a reader needs to see them -- but they
    are in the 163-dataset training corpus and cannot enter a generalisation number.
    """
    sets_out: Dict[str, object] = {}
    for name, rec in per_set.items():
        strata = {}
        for stratum, srec in rec["strata"].items():
            if "skipped" in srec:
                strata[stratum] = dict(srec)
                continue
            body = {k: srec[k] for k in ("n_draws", "n_groups", "groups")}
            if center_arm in srec:
                body["scorers"] = srec[center_arm]
            strata[stratum] = body
        sets_out[name] = {
            "in_training": rec["in_training"], "n_cells_total": rec["n_cells_total"],
            "n_labelled": rec["n_labelled"], "n_scored": rec["n_scored"],
            "n_donors": rec["n_donors"], "granularity": rec["granularity"],
            "centering_controls": rec["centering_controls"], "strata": strata}

    panel: Dict[str, object] = {}
    if full_panel and ood:
        for stratum in STRATA:
            vals = {"rna2atac": [], "atac2rna": []}
            for name in ood:
                sc = sets_out[name]["strata"].get(stratum, {}).get("scorers")
                if not sc or "global" not in sc:
                    vals = None
                    break
                for d in ("rna2atac", "atac2rna"):
                    vals[d].append(sc["global"][f"r1_{d}"])
            if not vals:
                continue
            panel[stratum] = {
                "n_sets": len(ood),
                "mean_r1_rna2atac": float(np.mean(vals["rna2atac"])),
                "mean_r1_atac2rna": float(np.mean(vals["atac2rna"])),
                "per_set_rna2atac": dict(zip(ood, map(float, vals["rna2atac"]))),
                "per_set_atac2rna": dict(zip(ood, map(float, vals["atac2rna"])))}

    return {
        "schema": SCHEMA,
        "arm": meta.get("arm"), "step": meta.get("step"),
        "checkpoint": meta.get("checkpoint"), "seed": meta.get("seed"),
        "git_sha": meta.get("git_sha"), "num_slots": meta.get("num_slots"),
        # ⛔ TRAIN-time centring is a property of the CHECKPOINT, read not chosen. The
        # 2x2 is train (fixed) x test (free); `_dcenter` is gated on `self.training`, so
        # a model trained with centring is NOT automatically evaluated with it.
        "train_centering": meta.get("train_centering"),
        "test_centering": center_arm,
        "primary_stratum": PRIMARY_STRATUM,
        "panel": {"ood": list(ood), "in_training": list(intr),
                  "len_ood": len(ood), "registered_panel": list(OOD_TRUE),
                  "is_registered_panel": bool(full_panel)},
        "label_granularity_check": gran_check,
        "sets": sets_out,
        "panel_mean_true_ood_only": panel,
        "notes": [
            "R@1 is reported PER DIRECTION and must never be averaged: "
            "eval_ckpt_stratified.py:48 and diag_b_ceiling.score both average, and the "
            "+0.0372 centring headline was retired for exactly that.",
            "The pool definition IS the metric: the same cells give ~0.16 at the "
            "dataset window, ~0.13 within donor and ~0.03 within donor x cell_type.",
            "chance = 1/pool exactly; `perm` is the empirical floor for this score "
            "matrix; `depth` is the depth-only scorer the perm twin cannot rule out.",
        ]}


def write_metrics(out_dir: str, meta, per_set, center_arm, ood, intr, gran_check,
                  full_panel, draws_by_set) -> str:
    d = os.path.join(out_dir, str(meta.get("arm")),
                     f"step_{int(meta['step']):06d}", f"center_{center_arm}")
    os.makedirs(d, exist_ok=True)
    m = build_metrics(meta, per_set, center_arm, ood, intr, gran_check, full_panel)
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2, sort_keys=False)
    # Per-draw arrays are the ONLY thing not recomputable after exit, and they are what
    # the paired arm contrast needs: the draws are shared across arms by construction
    # (the pool seed is a sha256 of set|stratum), so B - A is a paired difference.
    pack = {}
    for name, arrs in draws_by_set.items():
        for k, v in arrs.items():
            # `__gname__` carries no centring token and must ride along regardless: it
            # is what a re-analysis needs to cluster the paired SE on the drawn group.
            if f"|{center_arm}|" in k or k.endswith("|__gname__"):
                pack[f"{name}|{k}"] = v
    if pack:
        np.savez_compressed(os.path.join(d, "draws.npz"), **pack)
    return os.path.join(d, "metrics.json")


# ------------------------------------------------------------------------------------ #
# The arm contrast
# ------------------------------------------------------------------------------------ #

def contrast_arms(hi: str, lo: str,
                  draws: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                  ood: Sequence[str],
                  scorer: str = "global",
                  full_panel: bool = True) -> List[Dict[str, object]]:
    """Paired, per-direction delta on the shared draws, plus a Stouffer panel combine.

    Paired because both arms saw the SAME pool index sets: `make_group_pools` is seeded
    from sha256(set|stratum), which is stable across processes (Python's str hash is
    salted and would have silently unpaired them).

    ⛔ A 4-set sign test cannot reach p < 0.05: 4/4 same-sign is p = 0.125 two-sided. So
    the panel claim is a Stouffer combine of the per-set paired-by-draw z's, and the
    number of same-sign sets is reported as descriptive only.
    """
    rows: List[Dict[str, object]] = []
    for stratum in STRATA:
        for arm in CENTER_ARMS:
            for direction in ("r2a", "a2r"):
                key = f"{stratum}|{arm}|{scorer}|{direction}"
                zs, per_set = [], {}
                for name in ood:
                    dh = draws.get(hi, {}).get(name, {}).get(key)
                    dl = draws.get(lo, {}).get(name, {}).get(key)
                    if dh is None or dl is None or len(dh) != len(dl):
                        continue
                    gk = f"{stratum}|__gname__"
                    gname = draws[hi][name].get(gk, np.array(["g"] * len(dh)))
                    d = np.asarray(dh, np.float64) - np.asarray(dl, np.float64)
                    cse, ncl = cluster_se(d, gname[:len(d)])
                    t = d.mean() / cse if cse and np.isfinite(cse) and cse > 0 else \
                        float("nan")
                    p = two_sided_p(t, max(ncl - 1, 1))
                    per_set[name] = dict(delta=float(d.mean()), cluster_se=cse,
                                         n_clusters=ncl, t=float(t), p=float(p),
                                         r1_hi=float(np.mean(dh)),
                                         r1_lo=float(np.mean(dl)),
                                         n_draws=int(len(d)))
                    if np.isfinite(t):
                        # ⛔ Stouffer combines STANDARD NORMAL deviates. Appending the
                        # raw t would be anti-conservative: Var(t_df) = df/(df-2), and
                        # on this panel's real cluster counts (bmmc 18, fetal_heart 8,
                        # breast 6, liver 6 usable groups -> df 17/7/5/5) that inflates
                        # Var(Z) to ~1.47, taking the true type-I error at alpha=0.05 to
                        # ~0.10 and at 0.01 to ~0.037. Convert through the p at the
                        # CORRECT df first; the sign is carried separately because
                        # `two_sided_p` is sign-blind.
                        zs.append(_z_from_t(t, max(ncl - 1, 1)))
                if not per_set:
                    continue
                deltas = [v["delta"] for v in per_set.values()]
                stouffer = (float(np.sum(zs) / math.sqrt(len(zs))) if zs
                            else float("nan"))
                # ⛔ On a partial panel the mean is NOT the registered statistic and must
                # not appear anywhere a reader could mistake it for one -- not in the
                # printed table, not in comparison.csv, not in comparison.json.
                # `build_metrics` already gated `panel_mean_true_ood_only` on
                # `full_panel`; these three consumers never saw it, so a 3-set mean
                # was printed under a header reading "4-set TRUE-OOD panel".
                rows.append({
                    "contrast": f"{hi}-{lo}", "scorer": scorer, "stratum": stratum,
                    "test_centering": arm, "direction": direction,
                    "n_sets": len(per_set),
                    "is_registered_panel": bool(full_panel),
                    "panel_mean_delta": (float(np.mean(deltas)) if full_panel
                                         else float("nan")),
                    "n_sets_positive": int(sum(1 for x in deltas if x > 0)),
                    "stouffer_z": stouffer if full_panel else float("nan"),
                    "stouffer_p": (float(2 * (1 - _norm_cdf(abs(stouffer))))
                                   if np.isfinite(stouffer) and full_panel
                                   else float("nan")),
                    "per_set": per_set})
    return rows


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(q: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, |err| < 1.15e-9).

    Written out rather than pulled from scipy because this module must import and
    self-test on a login node with no scipy guarantee, and because a silently different
    ppf would move every panel p-value.
    """
    q = min(max(float(q), 1e-16), 1.0 - 1e-16)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plo = 0.02425
    if q < plo:
        r = math.sqrt(-2 * math.log(q))
        return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
               ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    if q > 1 - plo:
        return -_norm_ppf(1 - q)
    r = q - 0.5
    t = r * r
    return (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * r / \
           (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1)


def _z_from_t(t: float, df: int) -> float:
    """The signed standard-normal deviate carrying the same two-sided p as t on df."""
    if not np.isfinite(t):
        return float("nan")
    p = two_sided_p(t, df)
    return math.copysign(_norm_ppf(1.0 - p / 2.0), t)


def print_comparison(rows: Sequence[Dict[str, object]], ood: Sequence[str],
                     mde_3seed: float = 0.0023, log=print,
                     full_panel: bool = True) -> None:
    """The one table.  PRIMARY stratum first, then the pool-definition ladder."""
    log("\n" + "=" * 96)
    if full_panel:
        log("ARM CONTRAST -- global (cell-embedding) scorer, 4-set TRUE-OOD panel")
    else:
        log(f"ARM CONTRAST -- global (cell-embedding) scorer, PARTIAL panel "
            f"{sorted(ood)} ({len(ood)} of 4)")
        log("  ⛔ NOT the registered panel. No panel mean and no Stouffer combine are "
            "emitted; the per-set columns below are the only readable numbers.")
    log("  ⛔ the arm contrast MUST be on the global scorer: it is the only one defined "
        "in every arm.")
    log("  ⛔ per direction, never averaged. A verdict has to hold in BOTH.")
    log("=" * 96)
    order = [PRIMARY_STRATUM] + [s for s in STRATA if s != PRIMARY_STRATUM]
    hdr = (f"{'contrast':>9s} {'stratum':30s} {'ctr':11s} {'dir':4s} "
           f"{'panelΔ':>9s} {'+sets':>5s} {'Z':>7s} {'p':>8s}   per-set Δ")
    for stratum in order:
        sel = [r for r in rows if r["stratum"] == stratum]
        if not sel:
            continue
        log(f"\n--- {stratum}"
            + ("   <<< PRIMARY (registered)" if stratum == PRIMARY_STRATUM else ""))
        log(hdr)
        for r in sel:
            ps = "  ".join(f"{k[:4]} {r['per_set'][k]['delta']:+.4f}"
                           for k in ood if k in r["per_set"])
            pmd = r["panel_mean_delta"]
            flag = ("  *" if np.isfinite(pmd) and abs(pmd) >= mde_3seed else "")
            pm = f"{pmd:+9.4f}" if np.isfinite(pmd) else f"{'--':>9s}"
            zz = (f"{r['stouffer_z']:+7.2f}" if np.isfinite(r["stouffer_z"])
                  else f"{'--':>7s}")
            pp = (f"{r['stouffer_p']:8.4f}" if np.isfinite(r["stouffer_p"])
                  else f"{'--':>8s}")
            log(f"{r['contrast']:>9s} {r['stratum']:30s} {r['test_centering']:11s} "
                f"{r['direction']:4s} {pm} "
                f"{r['n_sets_positive']:d}/{r['n_sets']:d}   {zz} "
                f"{pp}   {ps}{flag}")
    log(f"\n  * = |panel mean delta| >= the registered 3-seed MDE {mde_3seed:.4f}.")
    log("  n_sets_positive is DESCRIPTIVE ONLY: 4/4 same-sign is p = 0.125 two-sided "
        "and can never clear 0.05 on its own. The panel claim is the Stouffer combine.")


def write_comparison_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    import pandas as pd
    flat = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != "per_set"}
        flat.append({**base, "set": "__PANEL__"})
        for name, v in r["per_set"].items():
            cols = {f"set_{k}": vv for k, vv in v.items()}
            flat.append({**base, "set": name, **cols})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(flat).to_csv(path, index=False)


# ------------------------------------------------------------------------------------ #
# combine -- the SEED is the replication unit, and nothing else in this file uses it
# ------------------------------------------------------------------------------------ #

def _t_ppf(q: float, df: int) -> float:
    """Inverse t CDF by bisection on `two_sided_p` -- adequate and dependency-free."""
    if df < 1:
        return float("nan")
    lo, hi = 0.0, 200.0
    target = 2.0 * (1.0 - q)                       # the two-sided p at the |t| we want
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def combine_seeds(per_seed: Dict[int, Sequence[Dict[str, object]]],
                  ood: Sequence[str], mde: float,
                  primary_only: bool = False) -> List[Dict[str, object]]:
    """Aggregate per-seed arm contrasts with the SEED as the unit of replication.

    ⛔ WHY THIS EXISTS, AND WHY `contrast_arms` IS NOT ENOUGH.  `contrast_arms` clusters
    its SE on the drawn retrieval group, i.e. it quantifies *pool-draw* noise with ONE
    pair of trained models held fixed.  The quantity the pre-registration is about is a
    property of the TRAINING RUN, and the run-to-run term is COMMON to all four OOD sets
    -- so it cannot be estimated from the four sets, no matter how many draws each gets.
    Concretely: if the arms are truly equal and seed 0 happens to land 2 SD apart
    (~5 % of seed pairs), every set shifts together by ~+0.0028, all four per-set
    cluster-robust t's read ~2.5, the Stouffer combine reads Z ~ 5, and the registered
    WIN rule fires on a single lucky pair of runs.  Three seeds are TRAINED for exactly
    this reason (design section 8.1: 2 seeds is WORSE than 1 because t(1) = 12.7), and
    before this function nothing in the harness consumed them: RUNBOOK step 9 scored
    `results/seed0` alone and seeds 1 and 2 (~90 GPU-h) never entered a statistic.

    The panel statistic is therefore a ONE-SAMPLE t over seeds of the per-seed 4-set
    delta, on df = n_seeds - 1.  It also reports the OBSERVED sigma_seed, which is the
    first measurement of that quantity on the refiner track -- design section 8.4 says
    that the registered 0.0010 is inherited from the projector track and is not measured
    here.
    """
    if not per_seed:
        return []
    seeds = sorted(per_seed)
    keyf = ("contrast", "scorer", "stratum", "test_centering", "direction")
    index: Dict[Tuple, Dict[int, Dict[str, object]]] = {}
    for sd in seeds:
        for r in per_seed[sd]:
            if primary_only and r["stratum"] != PRIMARY_STRATUM:
                continue
            index.setdefault(tuple(r[k] for k in keyf), {})[sd] = r

    out: List[Dict[str, object]] = []
    for key, by_seed in sorted(index.items(), key=lambda kv: str(kv[0])):
        have = [sd for sd in seeds if sd in by_seed]
        # The per-seed panel mean: one number per TRAINING RUN pair.
        pm = [by_seed[sd]["panel_mean_delta"] for sd in have]
        pm = [float(x) for x in pm if x is not None and np.isfinite(float(x))]
        row: Dict[str, object] = dict(zip(keyf, key))
        row["seeds"] = have
        row["n_seeds"] = len(pm)
        if len(pm) == 0:
            continue
        mean = float(np.mean(pm))
        row["per_seed_panel_delta"] = pm
        row["mean_panel_delta"] = mean
        if len(pm) >= 2:
            # ddof=1: sigma_seed is being ESTIMATED here, not assumed.
            sd_seed = float(np.std(pm, ddof=1))
            se = sd_seed / math.sqrt(len(pm))
            df = len(pm) - 1
            t = mean / se if se > 0 else float("nan")
            crit = _t_ppf(0.975, df)
            row.update(sigma_seed_observed=sd_seed, se_seed=se, df=df, t=float(t),
                       p=float(two_sided_p(t, df)),
                       ci95=[mean - crit * se, mean + crit * se],
                       n_seeds_positive=int(sum(1 for x in pm if x > 0)))
        else:
            # n = 1 CANNOT separate "ARM B is better" from "this seed of ARM B is
            # better". Say so in the row rather than emitting a p-value that reads as
            # if it could.
            row.update(sigma_seed_observed=float("nan"), se_seed=float("nan"), df=0,
                       t=float("nan"), p=float("nan"),
                       ci95=[float("nan")] * 2, n_seeds_positive=int(mean > 0),
                       caveat="n_seeds=1: no seed-level inference is possible")
        # Per-set means across seeds, descriptive.
        sets: Dict[str, Dict[str, float]] = {}
        for name in ood:
            vals = [float(by_seed[sd]["per_set"][name]["delta"]) for sd in have
                    if name in by_seed[sd].get("per_set", {})]
            if vals:
                sets[name] = {"mean_delta": float(np.mean(vals)),
                              "n_seeds": len(vals),
                              "n_seeds_positive": int(sum(1 for v in vals if v > 0))}
        row["per_set_across_seeds"] = sets
        row["verdict"] = seed_verdict(row, mde)
        out.append(row)
    return out


def seed_verdict(row: Dict[str, object], mde: float) -> str:
    """The registered decision rule, applied to the SEED-level statistic.

    Registered before the data (design section 8.3), reproduced here verbatim so the
    code and the pre-registration cannot drift apart:
      WIN      mean delta >= +mde AND the seed-level 95 % CI excludes 0
      NEGATIVE mean delta <= -mde AND the seed-level 95 % CI excludes 0
      NULL     |mean delta| <  mde  and the CI contains 0
    Anything else is AMBIGUOUS -- a large point estimate whose CI still spans zero is
    NOT a win, and this project's standing prior (6/6 in-dist gains that never carried
    to OOD) says it usually will not become one.
    """
    m = float(row.get("mean_panel_delta", float("nan")))
    lo, hi = row.get("ci95", [float("nan")] * 2)
    if not np.isfinite(m):
        return "INVALID (no panel mean; partial panel?)"
    if row.get("n_seeds", 0) < 2:
        return "UNDETERMINED (n_seeds < 2: seed noise is not estimable)"
    excl = np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
    if m >= mde and excl:
        return "WIN"
    if m <= -mde and excl:
        return "NEGATIVE (replicates the reported global cost)"
    if abs(m) < mde and not excl:
        return "NULL"
    return "AMBIGUOUS"


def cmd_combine(args) -> int:
    """Read one `comparison.json` per seed and emit the seed-level verdict."""
    per_seed: Dict[int, Sequence[Dict[str, object]]] = {}
    ood: List[str] = []
    for spec in args.seed:
        if "=" not in spec:
            raise SystemExit(f"--seed wants SEED=path/to/comparison.json, got {spec!r}")
        sd, path = spec.split("=", 1)
        with open(path) as fh:
            j = json.load(fh)
        assert j.get("is_registered_panel", True), (
            f"{path} was scored on a PARTIAL panel; its rows carry no panel mean and "
            f"cannot be combined across seeds.")
        if int(j["expect_step"]) != int(args.expect_step):
            raise SystemExit(
                f"{path} is step {j['expect_step']}, --expect_step is "
                f"{args.expect_step}. Every seed must be scored at the SAME fixed step "
                f"or the combine mixes checkpoints.")
        per_seed[int(sd)] = j["rows"]
        ood = j["panel"]["ood"] or ood
    rows = combine_seeds(per_seed, ood, args.mde, primary_only=args.primary_only)
    print("\n" + "=" * 96)
    print(f"SEED-LEVEL COMBINE -- {len(per_seed)} seed(s): {sorted(per_seed)}")
    print("  The SEED is the unit of replication. The per-draw cluster-robust p's in "
          "each")
    print("  comparison.json quantify POOL noise with the models held fixed; they "
          "cannot")
    print("  see the run-to-run term, which is COMMON to all four sets.")
    print("=" * 96)
    hdr = (f"{'contrast':>16s} {'stratum':30s} {'ctr':11s} {'dir':4s} {'meanD':>9s} "
           f"{'sigSeed':>8s} {'t':>7s} {'p':>8s} {'+seeds':>7s}  verdict")
    order = [PRIMARY_STRATUM] + [s for s in STRATA if s != PRIMARY_STRATUM]
    for stratum in order:
        sel = [r for r in rows if r["stratum"] == stratum]
        if not sel:
            continue
        print(f"\n--- {stratum}"
              + ("   <<< PRIMARY (registered)" if stratum == PRIMARY_STRATUM else ""))
        print(hdr)
        for r in sel:
            sg, tt, pp = r["sigma_seed_observed"], r["t"], r["p"]
            c_sg = f"{sg:8.4f}" if np.isfinite(sg) else f"{'--':>8s}"
            c_t = f"{tt:7.2f}" if np.isfinite(tt) else f"{'--':>7s}"
            c_p = f"{pp:8.4f}" if np.isfinite(pp) else f"{'--':>8s}"
            print(f"{str(r['contrast']):>16s} {r['stratum']:30s} "
                  f"{r['test_centering']:11s} {r['direction']:4s} "
                  f"{r['mean_panel_delta']:+9.4f} {c_sg} {c_t} {c_p} "
                  f"{r['n_seeds_positive']:d}/{r['n_seeds']:d}      {r['verdict']}")
    print(f"\n  registered MDE {args.mde:.4f}. sigma_seed shown is OBSERVED on this "
          f"track, not the")
    print("  inherited 0.0010 -- design section 8.4 records that the registered value "
          "came from the")
    print("  projector track and had never been measured here.")
    if any(r["n_seeds"] < 3 for r in rows):
        print("  ⚠ fewer than the registered 3 seeds: 2 seeds is WORSE than 1 "
              "(t(1) = 12.7).")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"schema": SCHEMA, "expect_step": args.expect_step,
                       # per-arm steps, so a val-selected comparison can never be
                       # mistaken for a step-matched one when the JSON is read later
                       "val_selected_steps": bool(getattr(args, "val_selected_steps", False)),
                       "step_by_arm": dict(step_by_arm),
                       "registered_mde": args.mde, "panel": ood,
                       "seeds": sorted(per_seed), "rows": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")
    return 0


# ------------------------------------------------------------------------------------ #
# extract -- checkpoint -> embeddings (LIVE FMs: the OOD sets are not in the cache)
# ------------------------------------------------------------------------------------ #

def cmd_extract(args) -> int:
    """One checkpoint -> one .npz per set + meta.json.

    ⚠️ The 8.7 TB `fm_token_cache_expanded_8192` holds ONLY `train/` and `val/`, so
    every OOD score is a live-FM forward: ~69,300 cells over the four sets, ~4 GPU-h per
    checkpoint at max_atac_length 8192.  Extract ONCE; every stratum, every centring
    rung and every null is then free on CPU.
    """
    _bootstrap_model_path()
    import torch
    import scanpy as sc
    from functools import partial

    # ⛔ the two gates, before a single GPU-second is spent.
    step = assert_checkpoint_step(args.ckpt, args.expect_step)
    ood, intr = resolve_panel(args.sets, require_full=not args.allow_partial_panel)
    print(f"panel: OOD {ood}  |  in-training (reported, never in a panel mean) {intr}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    assert int(ck["step"]) == step
    tr = argparse.Namespace(**ck["args"])
    # `train_finecls_refiner` asserts at IMPORT that attn_refiner.py carries
    # native_grad. Letting that assert fire here is deliberate: an eval built against
    # the unpatched module would silently score a differently-wired model than the one
    # that trained.
    import train_finecls_refiner as T
    live = T.Liveness()
    priors = T.load_priors(tr, live)
    model = T.FineCLSRefiner(tr, priors)
    incompat = model.load_state_dict(ck["model"], strict=True)
    assert not incompat.missing_keys and not incompat.unexpected_keys, incompat
    dev = torch.device(args.device)
    model = model.to(dev).eval()
    n_slots = int(getattr(model, "num_slots", 0))
    print(f"ckpt step {step} arm={ck['arm']} slots={n_slots} seed={ck['seed']} "
          f"git={str(ck.get('git_sha'))[:12]}")
    print(f"  train-time centring: global={bool(tr.center_global_by_dataset)} "
          f"slots={bool(tr.center_slots_by_dataset)}   (a CHECKPOINT property, not a "
          f"choice made here)")
    # The SPEED/ARITHMETIC policy the checkpoint was trained under, printed for the same
    # reason the centring is: it is a property OF THE CHECKPOINT.  `FineCLSRefiner`
    # above rebuilds the refiners from these very fields, so eval runs the same
    # arithmetic the training run did -- and a checkpoint written before these flags
    # existed falls back to the shipped defaults (grad_ckpt on, no sub-batching, sdpa,
    # RNA fp32 + ATAC fp16), which is exactly what it was trained with.
    _attn = ("varlen" if getattr(tr, "atac_varlen", 0)
             else ("sdpa" if tr.atac_sdpa else "dense"))
    print(f"  refiner policy: attn={_attn}"
          f" precision={getattr(tr, 'refiner_precision', 'shipped')}"
          f" (rna {getattr(tr, 'rna_autocast', 'off')} / atac "
          f"{getattr(tr, 'atac_autocast', 'fp16')})"
          f" rna_sub_batch={getattr(tr, 'rna_sub_batch', 0)}"
          f" rna_grad_ckpt={getattr(tr, 'rna_grad_ckpt', 1)}")
    if args.dry_run:
        print("  --dry_run: model rebuilt and state loaded; stopping before the FMs.")
        return 0

    from multiomics_clip_finelip import MultiOmicsCLIP_FineLIP, FineLIPConfig
    from multiomics_clip.dataset import PairedMultiOmicsDataset, collate_fn
    from multiomics_clip.preprocessing import load_gene_list
    from scripts.train_filip_combined import fm_tokens
    from torch.utils.data import DataLoader

    fmcfg = FineLIPConfig(rna_encoder_path=args.rna_encoder_path,
                          atac_encoder_path=args.atac_encoder_path,
                          projection_dim=int(tr.proj_dim), temperature=0.07,
                          freeze_rna_encoder=True, freeze_atac_encoder=True,
                          token_dim=int(tr.proj_dim),
                          max_atac_length=int(tr.max_atac_length),
                          target_resolution=4.0)
    fm = MultiOmicsCLIP_FineLIP(fmcfg).to(dev).eval()
    gl = load_gene_list(args.gene_list_path)
    coll = partial(collate_fn, fixed_atac_length=int(tr.max_atac_length))

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for name in ood + intr:
        rp, ap, lsrc = DS[name]
        rna_backed = sc.read_h5ad(rp, backed="r")
        # THE CHEAPEST CORRECTNESS CHECK THERE IS.  islet_concat (9,976 training cells)
        # and the fig2 islet file (26,892) differ by 2.7x; a wrong path is caught here.
        if name in EXPECTED_N and not args.skip_cell_count_gate:
            assert rna_backed.n_obs == EXPECTED_N[name], (
                f"{name}: {rp} has {rna_backed.n_obs} cells, expected "
                f"{EXPECTED_N[name]}. This is the wrong file.")
        ds = PairedMultiOmicsDataset(
            rna_adata=rna_backed, atac_adata=sc.read_h5ad(ap, backed="r"), gene_list=gl,
            target_resolution=4.0, max_atac_length=int(tr.max_atac_length),
            random_truncate=False, backed=True)
        # Backed mode drops cells lazily (RNA nnz filter + a scattered min-ATAC filter),
        # so row i is _rna_keep_orig[_valid_idx][i].  Resolving positionally gave 0.0000
        # barcode agreement on breast; this project has hit that bug class twice.
        vidx = np.asarray(ds._valid_idx, dtype=np.int64)
        orig = (np.asarray(ds._rna_keep_orig, np.int64)[vidx]
                if getattr(ds, "_rna_keep_orig", None) is not None else vidx)
        bc_all = np.asarray(ds._rna_ref.obs.index.astype(str).values)[orig]
        # LABEL_COL is consulted rather than hardcoding "cell_type": the 11
        # islet_HPAP-* donor files carry NO annotation, and get_dataset_ids would raise
        # KeyError and kill the whole extract.  None fills UNLABELED_CT, which
        # diag_common.labeled_mask DROPS at score time -- so an accidental `score` on an
        # unannotated set reports zero scored cells instead of silently collapsing
        # `dataset x cell_type` into `dataset`.
        lcol = LABEL_COL.get(name, "cell_type")
        if lcol is None:
            ct_all = np.array([UNLABELED_CT] * len(bc_all), dtype=str)
            print(f"  ⚠ {name}: no cell-type annotation registered (LABEL_COL=None); "
                  f"cell_type filled with '{UNLABELED_CT}', which labeled_mask DROPS. "
                  f"These vectors are for barcode-joined downstream use, NOT retrieval "
                  f"scoring.")
        else:
            ct_all = np.asarray(ds.get_dataset_ids(lcol), dtype=str)
        dcol = DONOR_COL.get(name, "batch")
        # ⛔ the silent ["_"] fallback below is why every set that needs a non-default
        # donor column must be registered in DONOR_COL.  Refuse it for a set that IS
        # registered: a typo'd column name there would collapse "within donor" to
        # "within set" without a single line of output.
        assert not (name in DONOR_COL and dcol not in rna_backed.obs.columns), (
            f"{name}: DONOR_COL says '{dcol}' but {rp} obs has "
            f"{list(rna_backed.obs.columns)}")
        don_all = (np.asarray(rna_backed.obs[dcol].astype(str).values)[orig]
                   if dcol in rna_backed.obs.columns else np.array(["_"] * len(bc_all)))
        print(f"  {name}: donor<-obs['{dcol}'] labels<-"
              f"{('obs[' + repr(lcol) + ']') if lcol else 'NONE (' + UNLABELED_CT + ')'}"
              f"  in_training={name in IN_TRAINING}")
        assert len(ct_all) == len(bc_all) == len(don_all), \
            "label / donor arrays are not dataset-aligned"

        loader = DataLoader(ds, batch_size=args.micro_batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=coll)
        ZR, ZA, RS, AS, RV, AV, RM, AM, RD, AD = ([] for _ in range(10))
        n = 0
        with torch.no_grad():
            for batch in loader:
                out = fm_tokens(fm, batch, dev, include_summary=True,
                                autocast_rna=bool(getattr(tr, "fm_autocast", 1)),
                                return_ids=True)
                rc, rt, rm, ac, at, am, rv, cs, gid, cid = out
                zr, za, rs, a_s, rvd, avd, rms, ams = model.encode(
                    rt, rm, gid.long(), at, am, cid.long())
                ZR.append(zr.float().cpu()); ZA.append(za.float().cpu())
                RD.append((~rm).sum(1).clamp(min=1).float().log().cpu())
                AD.append((~am).sum(1).clamp(min=1).float().log().cpu())
                if rs is not None:
                    RS.append(rs.float().cpu()); AS.append(a_s.float().cpu())
                    RV.append(rvd.cpu()); AV.append(avd.cpu())
                    RM.append(rms.float().cpu()); AM.append(ams.float().cpu())
                n += zr.shape[0]
                if n >= args.max_cells:
                    break
        N = n
        arrays = dict(rna=torch.cat(ZR).numpy(), atac=torch.cat(ZA).numpy(),
                      barcode=bc_all[:N], dataset=np.array([name] * N),
                      donor=don_all[:N], cell_type=ct_all[:N],
                      rna_depth=torch.cat(RD).numpy(),
                      atac_depth=torch.cat(AD).numpy())
        if RS:
            arrays.update(rna_slots=torch.cat(RS).numpy(),
                          atac_slots=torch.cat(AS).numpy(),
                          rna_valid=torch.cat(RV).numpy(),
                          atac_valid=torch.cat(AV).numpy(),
                          rna_mass=torch.cat(RM).numpy(),
                          atac_mass=torch.cat(AM).numpy(),
                          routing_topk=np.array(int(tr.routing_topk)),
                          routing_mass_power=np.array(float(tr.routing_mass_power)),
                          routing_tail_weight=np.array(float(tr.routing_tail_weight)))
        write_emb_npz(os.path.join(args.out_dir, f"{name}.npz"), **arrays)
        written.append(name)
        print(f"  {name}: {N:,} cells embedded -> {name}.npz")

    meta = {"schema": SCHEMA, "arm": args.label or ck["arm"], "trainer_arm": ck["arm"],
            "step": step, "checkpoint": os.path.abspath(args.ckpt),
            "seed": int(ck["seed"]), "git_sha": ck.get("git_sha"),
            "num_slots": n_slots, "sets": written,
            "train_centering": {"global": bool(tr.center_global_by_dataset),
                                "slots": bool(tr.center_slots_by_dataset)},
            "max_atac_length": int(tr.max_atac_length), "proj_dim": int(tr.proj_dim)}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {args.out_dir}/meta.json")
    return 0


# ------------------------------------------------------------------------------------ #
# score
# ------------------------------------------------------------------------------------ #

def _parse_emb_args(entries: Sequence[str]) -> List[Tuple[Optional[str], str]]:
    out = []
    for e in entries:
        if "=" in e and not os.path.isdir(e):
            lab, d = e.split("=", 1)
            out.append((lab, d))
        else:
            out.append((None, e))
    return out


def cmd_score(args) -> int:
    assert args.draws >= 200, (
        f"--draws {args.draws}: retrieval needs >= 200 draws (single-draw "
        f"retrieval is a documented artefact in this project). The registered "
        f"protocol is >= 500.")
    print_provenance()
    arms: Dict[str, Dict[str, object]] = {}
    step_by_arm: Dict[str, int] = {}
    for label, d in _parse_emb_args(args.emb):
        meta, sets = load_emb_dir(d)
        lab = label or str(meta.get("arm"))
        if lab in arms:
            raise SystemExit(
                f"two embedding dirs resolve to the arm label {lab!r} (ARM B and ARM C "
                f"are both `finecls` by construction). Pass LABEL=DIR explicitly.")
        # ⛔ the leak gate again, at SCORING time -- this is where numbers are made.
        refuse_best_checkpoint(str(meta.get("checkpoint", "")))
        if getattr(args, "val_selected_steps", False):
            # ⛔ THE GUARD IS RELAXED ONLY UNDER A CONVERGENCE CLAIM, AND ONLY VISIBLY.
            # Step-matching exists because comparing arms at different steps is the
            # easiest way to manufacture an effect.  That danger is real when the arms
            # are still moving.  It is NOT when every arm sits on its own val plateau:
            # once converged, an extra 1000 steps changes nothing, and FORCING one step
            # instead penalises whichever arm converges later (measured: ARM B is still
            # improving at 6000->8000, t=-2.13, while ARM A is flat from ~5000).  So the
            # arms are selected independently by VAL loss -- held-out and in-distribution,
            # so it cannot leak the OOD panel -- and every per-arm step is RECORDED in
            # the output, never silently averaged away.
            step_by_arm[lab] = int(meta["step"])
        else:
            assert int(meta["step"]) == args.expect_step, (
                f"{d}: embeddings are from step {meta['step']}, --expect_step is "
                f"{args.expect_step}. Arms must be compared at ONE fixed step.")
        arms[lab] = {"meta": meta, "sets": sets, "dir": d}
        print(f"  arm {lab:16s} step {meta['step']:>6}  train_centering="
              f"{meta.get('train_centering')}  <- {d}")
    assert arms, "no --emb given"

    names = sorted({n for a in arms.values() for n in a["sets"]})
    for lab, a in arms.items():
        miss = sorted(set(names) - set(a["sets"]))
        assert not miss, (
            f"arm {lab} is missing {miss}; the arms must score the same sets")
    ood, intr = resolve_panel(names, require_full=not args.allow_partial_panel)
    full_panel = set(ood) == set(OOD_TRUE)
    print(f"\npanel: OOD {ood} (len {len(ood)})"
          + (f"  |  in-training, EXCLUDED from panel means: {intr}" if intr else ""))
    if not full_panel:
        print("  ⚠ --allow_partial_panel: this is NOT the registered panel and "
              "no panel mean is emitted.")

    # Same cells, in the same order, in every arm -- otherwise the pools are not shared
    # and the contrast is not paired.
    ref = arms[list(arms)[0]]
    for lab, a in arms.items():
        for n in names:
            b0 = np.asarray(ref["sets"][n]["barcode"], dtype=str)
            b1 = np.asarray(a["sets"][n]["barcode"], dtype=str)
            assert b0.shape == b1.shape and (b0 == b1).all(), (
                f"arm {lab}, set {n}: barcodes differ from arm {list(arms)[0]}. The "
                f"pools are seeded per (set, stratum) and shared across arms; "
                f"different cells means the contrast is NOT paired.")

    center_arms = list(args.test_centering)
    results: Dict[str, Dict[str, Dict[str, object]]] = {}
    draws_all: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for lab, a in arms.items():
        print(f"\n{'=' * 96}\nARM {lab}\n{'=' * 96}")
        per_set, per_draw = {}, {}
        for n in names:
            rec, dr = score_one_set(n, a["sets"][n], args.pool, args.draws, center_arms,
                                    bool(args.center_split), args.slot_draws)
            per_set[n] = rec
            per_draw[n] = dr
            cc = rec["centering_controls"]
            if cc.get("applicable"):
                c1r = cc["C1_gallery_only_no_renorm_max_abs_delta_r2a"]
                c1a = cc["C1_gallery_only_no_renorm_max_abs_delta_a2r"]
                c2 = cc["C2_both_sides_renorm_mean_abs_delta"]
                print(f"    centring control [{cc['stratum']}]: C1 gallery-only "
                      f"|Δ| r2a {c1r:.3e} a2r {c1a:.3e} (must be 0) | C2 real "
                      f"centring mean|Δ| {c2:.4f} (must be > 0) -> {cc['verdict']}")
                if args.strict_controls and not cc["verdict"].startswith("OK"):
                    raise SystemExit(f"centring control FAILED on {n}: {cc['verdict']}")
        results[lab] = per_set
        draws_all[lab] = per_draw

    gstats = {n: results[list(arms)[0]][n]["granularity"] for n in ood}
    gchk = check_matched_granularity(gstats, args.granularity_max_ratio,
                                     strict=args.strict_granularity)
    print("\n--- label granularity (identical across arms; the risk is CROSS-SET) ---")
    for n in ood + intr:
        g = results[list(arms)[0]][n]["granularity"]
        tag = "  [IN-TRAINING]" if n in intr else ""
        print(f"    {n:14s} n_types {g['n_types']:4d}  n_eff {g['n_eff_types']:6.2f}  "
              f"median/type {g['median_cells_per_type']:7.1f}  largest "
              f"{g['largest_type_frac']:.2f}{tag}")
    if gchk.get("warning"):
        print(f"    ⚠ {gchk['warning']}")
    else:
        print(f"    matched: ratio {gchk.get('ratio', float('nan')):.2f} <= "
              f"{args.granularity_max_ratio}")

    for lab, a in arms.items():
        for ca in center_arms:
            if ca in ("donor_ext", "set_ext") and not args.center_split:
                continue
            p = write_metrics(args.out_dir, {**a["meta"], "arm": lab}, results[lab], ca,
                              ood, intr, gchk, full_panel, draws_all[lab])
            print(f"  wrote {p}")

    labels = list(arms)
    contrasts = [tuple(c.split(":", 1)) for c in args.contrast] if args.contrast else \
        [(h, labels[0]) for h in labels[1:]]
    rows: List[Dict[str, object]] = []
    for hi, lo in contrasts:
        assert hi in arms and lo in arms, f"--contrast {hi}:{lo}: unknown arm label"
        for _sc in ("global", "slot", "fused"):
            # `cell_only` has no slots, so slot/fused rows appear only for B vs C.
            rows += contrast_arms(hi, lo, draws_all, ood, scorer=_sc,
                                  full_panel=full_panel)
    if rows:
        print_comparison(rows, ood, args.mde, full_panel=full_panel)
        csv = os.path.join(args.out_dir, "comparison.csv")
        write_comparison_csv(rows, csv)
        with open(os.path.join(args.out_dir, "comparison.json"), "w") as fh:
            json.dump({"schema": SCHEMA, "expect_step": args.expect_step,
                       "is_registered_panel": bool(full_panel),
                       "pool": args.pool, "draws": args.draws,
                       "registered_mde_3seed": args.mde,
                       "primary_stratum": PRIMARY_STRATUM,
                       "panel": {"ood": ood, "in_training": intr},
                       "label_granularity_check": gchk, "rows": rows}, fh, indent=2)
        print(f"\n  wrote {csv}")
    return 0


# ------------------------------------------------------------------------------------ #
# selftest -- the harness is checked against ARITHMETIC, not against another script
# ------------------------------------------------------------------------------------ #

def synth_embeddings(n_groups: int, per_group: int, k_correct: int
                     ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Embeddings whose R@1 is EXACTLY known in BOTH directions, by construction.

    ATAC is the identity basis (unit rows).  RNA row i puts its mass on two columns, its
    own and its cyclic successor t(i) = i+1 inside the group:

        i is "correct":   0.80 at column i,  0.60 at column t(i)     -> argmax = i
        i is "wrong":     0.28 at column i,  0.96 at column t(i)     -> argmax = t(i)

    0.80^2 + 0.60^2 = 0.28^2 + 0.96^2 = 1, so every row is EXACTLY unit norm -- which
    matters, because the whole project's L2-input convention would otherwise be violated
    by the test fixture itself.  The four magnitudes are pairwise distinct, so no argmax
    is ever decided by a tie-break.

    Then, with the first k rows of each group correct and t(i) = i+1 (mod per_group):
        RNA->ATAC  correct  <=>  i in C                       ->  k / per_group
        ATAC->RNA  column j sees 0.80/0.28 from row j and 0.60/0.96 from row j-1, so
                   correct  <=>  j in C AND j-1 in C          -> (k-1) / per_group
    Different values in the two directions on purpose: a harness that averaged them (the
    documented bug in eval_ckpt_stratified.py:48 and diag_b_ceiling.score) would return
    the mean of the two and fail this test.
    """
    assert 0 < k_correct < per_group
    N = n_groups * per_group
    za = np.eye(N, dtype=np.float32)
    zr = np.zeros((N, N), dtype=np.float32)
    for g in range(n_groups):
        off = g * per_group
        for a in range(per_group):
            i, t = off + a, off + (a + 1) % per_group
            if a < k_correct:
                zr[i, i], zr[i, t] = 0.80, 0.60
            else:
                zr[i, i], zr[i, t] = 0.28, 0.96
    return zr, za, k_correct / per_group, (k_correct - 1) / per_group


def _synth_arrays(name: str, n_groups: int, per_group: int, k_correct: int,
                  jitter: float = 0.0, seed: int = 0):
    zr, za, exp_r2a, exp_a2r = synth_embeddings(n_groups, per_group, k_correct)
    if jitter:
        # ARM-B stand-in: the same structure, slightly worse, so a CONTRAST has a known
        # sign.  The jitter is applied to RNA only and re-normalised, so unit norm
        # holds.
        rng = np.random.RandomState(seed)
        zr = zr + jitter * rng.randn(*zr.shape).astype(np.float32)
        zr /= np.linalg.norm(zr, axis=1, keepdims=True)
    N = n_groups * per_group
    g = np.arange(N) // per_group
    return dict(rna=zr.astype(np.float32), atac=za.astype(np.float32),
                barcode=np.array([f"{name}_c{i:05d}" for i in range(N)]),
                dataset=np.array([name] * N),
                donor=np.array([f"D{x}" for x in g]),
                cell_type=np.array([f"T{x}" for x in g])), exp_r2a, exp_a2r


def synth_domain_shift(n_groups: int, per_group: int, dim: int = 128, pair_noise=0.2,
                       donor_offset=4.0, seed: int = 0):
    """A fixture where per-donor centring has a KNOWN, LARGE, POSITIVE effect.

    WHY A SECOND FIXTURE.  `synth_embeddings` is deliberately symmetric -- ATAC is the
    identity basis and the donor mean is the same vector for every cell -- and on it
    per-donor centring is provably INERT (measured: R@1 identical to 1e-16 at donor
    offsets up to 2.0).  That is a fine property to demonstrate, and the self-test does
    demonstrate it, because a control that never fires proves nothing.  But it cannot
    show that the centring switch WORKS, so this fixture injects the thing centring
    exists to remove:

        rna_i  = normalize( s_i + b * n1_i + g * u_donor(i) )
        atac_i = normalize( s_i + b * n2_i + g * u_donor(i) )

    `s_i` is the shared per-cell latent (the pairing signal), `n1/n2` independent noise,
    `u_d` a per-donor unit offset.  The offset does NOT hurt through its constant part
    -- that is rank-invariant, which is exactly what control C1 asserts -- but through
    the cross term g*<u_d, b_j>, which varies with the gallery cell.  Subtracting the
    per-donor mean removes the u component from both modalities and the pairing comes
    back.  Measured on the defaults (dim 128, noise 0.2, offset 4.0, pool 128, seed 3):

        no offset at all   r2a 0.2324   a2r 0.2535
        offset 4.0, raw    r2a 0.1609   a2r 0.1383     <- the shift costs ~40% of it
        offset 4.0, centred r2a 0.2367  a2r 0.2746     <- back to the no-shift level

    So the assertion has a SIGN and a target, not merely "the number moved".
    """
    rng = np.random.RandomState(seed)
    N = n_groups * per_group
    g = np.arange(N) // per_group
    s = rng.randn(N, dim).astype(np.float32)
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    u = rng.randn(n_groups, dim).astype(np.float32)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    zr = s + pair_noise * rng.randn(N, dim).astype(np.float32) + donor_offset * u[g]
    za = s + pair_noise * rng.randn(N, dim).astype(np.float32) + donor_offset * u[g]
    zr /= np.linalg.norm(zr, axis=1, keepdims=True)
    za /= np.linalg.norm(za, axis=1, keepdims=True)
    return dict(rna=zr.astype(np.float32), atac=za.astype(np.float32),
                barcode=np.array([f"shift_c{i:05d}" for i in range(N)]),
                dataset=np.array(["bmmc"] * N),
                donor=np.array([f"D{x}" for x in g]),
                cell_type=np.array([f"T{x}" for x in g]))


class _T:
    def __init__(self):
        self.n = self.ok = 0
        self.fail: List[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        self.n += 1
        if cond:
            self.ok += 1
            print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
        else:
            self.fail.append(name)
            print(f"  FAIL  {name}" + (f"   [{detail}]" if detail else ""))

    def raises(self, name: str, exc, fn, want: str = "") -> None:
        self.n += 1
        try:
            fn()
        except exc as e:
            hit = (want.lower() in str(e).lower()) if want else True
            if hit:
                self.ok += 1
                first = str(e).strip().splitlines()[0][:110]
                print(f"  PASS  {name}\n            raised {exc.__name__}: {first}")
                return
            self.fail.append(name)
            print(f"  FAIL  {name}   [raised but message lacks {want!r}: {e}]")
            return
        except Exception as e:                                   # noqa: BLE001
            self.fail.append(name)
            print(f"  FAIL  {name}   [raised {type(e).__name__}, expected "
                  f"{exc.__name__}: {e}]")
            return
        self.fail.append(name)
        print(f"  FAIL  {name}   [did NOT raise {exc.__name__}]")


def cmd_selftest(args) -> int:
    import tempfile
    t = _T()
    pool, draws = 128, 250
    print("=" * 96)
    print("SELFTEST -- eval_finecls_refiner.py")
    print("=" * 96)

    # ---------------------------------------------------------------- T1 known ordering
    print("\n[T1] synthetic embeddings with a KNOWN retrieval ordering")
    arrays, exp_r2a, exp_a2r = _synth_arrays("bmmc", 2, pool, 96)
    rec, dr = score_one_set("bmmc", arrays, pool, draws,
                            center_arms=("off", "donor_self"), center_split=False,
                            slot_draws=0, log=lambda *a, **k: None)
    got = rec["strata"][PRIMARY_STRATUM]["off"]["global"]
    t.check("T1a within-(dataset x cell_type) RNA->ATAC R@1 == k/pool exactly",
            got["r1_rna2atac"] == exp_r2a,
            f"got {got['r1_rna2atac']:.10f} want {exp_r2a:.10f}")
    t.check("T1b within-(dataset x cell_type) ATAC->RNA R@1 == (k-1)/pool exactly",
            got["r1_atac2rna"] == exp_a2r,
            f"got {got['r1_atac2rna']:.10f} want {exp_a2r:.10f}")
    t.check("T1c the two directions are NOT averaged (they differ by exactly 1/pool)",
            abs((got["r1_rna2atac"] - got["r1_atac2rna"]) - 1.0 / pool) < 1e-12,
            f"delta {got['r1_rna2atac'] - got['r1_atac2rna']:.10f} = 1/{pool}")
    perm = rec["strata"][PRIMARY_STRATUM]["off"]["perm"]
    t.check("T1d the PERM twin lands at chance 1/pool",
            abs(perm["r1_rna2atac"] - 1.0 / pool) < 3.0 / pool,
            f"perm {perm['r1_rna2atac']:.4f} vs chance {1.0 / pool:.4f}")
    t.check("T1e all four strata were computed (the pool definition IS the metric)",
            all(k in rec["strata"] for k in STRATA),
            f"{[k for k in rec['strata']]}")
    win = rec["strata"]["dataset"]["off"]["global"]["r1_rna2atac"]
    t.check("T1f the dataset-window number differs from the within-type one",
            win != got["r1_rna2atac"],
            f"dataset-window {win:.4f} vs primary {got['r1_rna2atac']:.4f}")

    # generic path == borrowed path, so the slot/fused/null scorers cannot drift from
    # the reference implementation the arm contrast rides on.
    R = reused()
    keys = stratum_key({k: arrays[k] for k in ("dataset", "donor", "cell_type")},
                       STRATA[PRIMARY_STRATUM])
    idxp, gname = R["make_group_pools"](keys, pool, 20,
                                        seed=pool_seed("bmmc", PRIMARY_STRATUM))
    a1, b1 = R["per_draw_r1"](arrays["rna"], arrays["atac"], idxp)
    import torch as _torch

    tr_, ta_ = _torch.as_tensor(arrays["rna"]), _torch.as_tensor(arrays["atac"])

    def _gfn(idx):
        i = _torch.as_tensor(np.asarray(idx))
        return tr_[i] @ ta_[i].t()
    a2, b2 = per_draw_r1_generic(_gfn, idxp)
    t.check("T1g generic scorer path reproduces diag_b_ceiling.per_draw_r1 bit-for-bit",
            float(np.abs(a1 - a2).max()) == 0.0 and float(np.abs(b1 - b2).max()) == 0.0,
            f"max|delta| r2a {np.abs(a1 - a2).max():.3e} "
            f"a2r {np.abs(b1 - b2).max():.3e}")

    # ---------------------------------------------------------------- T2 panel refusal
    print("\n[T2] the len(OOD)==4 panel assert")
    ok_ood, ok_intr = resolve_panel(["bmmc", "breast", "fetal_heart", "liver"])
    t.check("T2a the registered panel resolves to exactly 4 OOD sets",
            len(ok_ood) == 4 and set(ok_ood) == set(OOD_TRUE), f"{sorted(ok_ood)}")
    ok_ood2, ok_intr2 = resolve_panel(["bmmc", "breast", "fetal_heart", "liver",
                                       "islet", "pln"])
    t.check("T2b islet/pln are accepted but LABELLED in-training, panel still 4",
            len(ok_ood2) == 4 and sorted(ok_intr2) == ["islet", "pln"],
            f"ood {sorted(ok_ood2)} in_training {sorted(ok_intr2)}")
    t.raises("T2c a 3-set panel is REFUSED", PanelError,
             lambda: resolve_panel(["bmmc", "breast", "fetal_heart"]), "len(OOD)==3")
    t.raises("T2d swapping a training set in for liver is REFUSED", PanelError,
             lambda: resolve_panel(["bmmc", "breast", "fetal_heart", "islet"]),
             "missing ['liver']")
    t.raises("T2e an unknown set name is REFUSED, not silently dropped", PanelError,
             lambda: resolve_panel(["bmmc", "breast", "fetal_heart", "liver", "lung"]),
             "unknown evaluation set")
    t.check("T2f --allow_partial_panel lets a subset through, without the 4-assert",
            resolve_panel(["bmmc", "breast"], require_full=False)[0] == ["bmmc",
                                                                        "breast"])

    # ---------------------------------------------------------------- T3 best_ refusal
    print("\n[T3] the best_* checkpoint refusal and the fixed-step gate")
    import torch
    tmp = tempfile.mkdtemp(prefix="finecls_eval_selftest_")
    snap = os.path.join(tmp, "snapshots")
    os.makedirs(snap, exist_ok=True)
    good = os.path.join(snap, "step_008000.pt")
    torch.save({"step": 8000}, good)
    bad_best = os.path.join(tmp, "best_model.pt")
    torch.save({"step": 8000}, bad_best)
    wrong = os.path.join(snap, "step_007500.pt")
    torch.save({"step": 7500}, wrong)
    renamed = os.path.join(snap, "step_008000_renamed", "step_008000.pt")
    os.makedirs(os.path.dirname(renamed), exist_ok=True)
    torch.save({"step": 7500}, renamed)
    t.check("T3a a fixed-step snapshot passes",
            assert_checkpoint_step(good, 8000) == 8000, os.path.basename(good))
    t.raises("T3b best_model.pt is REFUSED", CheckpointError,
             lambda: assert_checkpoint_step(bad_best, 8000), "REFUSED")
    t.raises("T3c a path containing 'best' anywhere is REFUSED", CheckpointError,
             lambda: refuse_best_checkpoint(f"{tmp}/best_ood/snapshots/step_008000.pt"),
             "REFUSED")
    t.raises("T3d the WRONG step is REFUSED", CheckpointError,
             lambda: assert_checkpoint_step(wrong, 8000), "is step 7500")
    t.raises("T3e a RENAMED file (name says 8000, file says 7500) is REFUSED",
             CheckpointError, lambda: assert_checkpoint_step(renamed, 8000),
             "records step 7500")

    # ---------------------------------------------------------------- T4 centring live
    print("\n[T4] the test-time centring switch, in BOTH directions")
    shifted = synth_domain_shift(2, pool, dim=128, pair_noise=0.2, donor_offset=4.0,
                                 seed=3)
    unshifted = synth_domain_shift(2, pool, dim=128, pair_noise=0.2, donor_offset=0.0,
                                   seed=3)
    urec, _ = score_one_set("bmmc", unshifted, pool, draws, center_arms=("off",),
                            center_split=False, slot_draws=0, log=lambda *a, **k: None)
    ubase = urec["strata"][PRIMARY_STRATUM]["off"]["global"]
    srec, _ = score_one_set("bmmc", shifted, pool, draws,
                            center_arms=("off", "donor_self"), center_split=False,
                            slot_draws=0, log=lambda *a, **k: None)
    cc = srec["centering_controls"]
    t.check("T4a the control ran on a single-donor stratum (the rank identity "
            "needs the centring group constant inside a pool)",
            cc.get("applicable") and cc["stratum"] == SINGLE_DONOR_STRATUM,
            str(cc.get("stratum")))
    t.check("T4b C1: a GALLERY-ONLY shift WITHOUT re-normalisation moves R@1 by "
            "EXACTLY 0 -- i.e. this is not the degenerate gallery-only variant",
            cc["C1_exactly_zero"],
            f"r2a {cc['C1_gallery_only_no_renorm_max_abs_delta_r2a']:.3e} "
            f"a2r {cc['C1_gallery_only_no_renorm_max_abs_delta_a2r']:.3e}")
    t.check("T4c C2: the real both-sides RE-NORMALISED centring DOES move R@1 "
            "(a no-op here would be a silent failure)",
            cc["C2_pass_is_nonzero"],
            f"mean|delta| {cc['C2_both_sides_renorm_mean_abs_delta']:.4f}")
    raw = srec["strata"][PRIMARY_STRATUM]["off"]["global"]
    ctr = srec["strata"][PRIMARY_STRATUM]["donor_self"]["global"]
    t.check("T4d centring RECOVERS the pairing an injected per-donor offset destroyed, "
            "in BOTH directions -- the effect has the right SIGN, not just a size",
            ctr["r1_rna2atac"] > raw["r1_rna2atac"] + 0.05 and
            ctr["r1_atac2rna"] > raw["r1_atac2rna"] + 0.05,
            f"r2a {raw['r1_rna2atac']:.4f} -> {ctr['r1_rna2atac']:.4f} "
            f"(no-offset reference {ubase['r1_rna2atac']:.4f}) | "
            f"a2r {raw['r1_atac2rna']:.4f} -> {ctr['r1_atac2rna']:.4f} "
            f"(reference {ubase['r1_atac2rna']:.4f}) | chance {1.0 / pool:.4f}")
    t.check("T4e centring is the REUSED per_group_center, not a local copy",
            os.path.basename(reused()["_src"]["center"]) == "refiner_ood_center.py",
            reused()["_src"]["center"])
    # The other direction of the control itself: on the SYMMETRIC T1 fixture (ATAC = the
    # identity basis, one donor mean shared by every cell) centring is provably inert,
    # and the control must SAY SO rather than pass quietly. A control that can only ever
    # print OK proves nothing -- this is the same standard the arm-liveness asserts are
    # held to.
    cc0 = rec["centering_controls"]
    t.check("T4f the control FIRES when centring really is inert (symmetric fixture)",
            cc0["C1_exactly_zero"] and not cc0["C2_pass_is_nonzero"]
            and cc0["verdict"].startswith("C2 FAILED"),
            f"C2 mean|delta| {cc0['C2_both_sides_renorm_mean_abs_delta']:.1e} -> "
            f"{cc0['verdict'][:60]}")

    # ---------------------------------------------------------------- T5 granularity
    print("\n[T5] the matched-label-granularity gate for cross-set comparison")
    fine = {f"t{i}": 1 for i in range(20)}
    st_matched = {"a": label_granularity(np.array(sum([[k] * 100 for k in fine], []))),
                  "b": label_granularity(np.array(sum([[k] * 100 for k in fine], [])))}
    st_coarse = {"a": st_matched["a"],
                 "b": label_granularity(np.array(["x"] * 1000 + ["y"] * 1000))}
    t.check("T5a matched granularity passes",
            check_matched_granularity(st_matched)["matched"] is True,
            f"ratio {check_matched_granularity(st_matched)['ratio']:.2f}")
    r = check_matched_granularity(st_coarse)
    t.check("T5b unmatched granularity is FLAGGED (20 eff types vs 2)",
            r["matched"] is False and "47-66%" in r["warning"],
            f"ratio {r['ratio']:.2f}")
    t.raises("T5c --strict_granularity turns the flag into a refusal", PanelError,
             lambda: check_matched_granularity(st_coarse, strict=True),
             "NOT MATCHED")

    # ---------------------------------------------------------------- T6 end-to-end
    print("\n[T6] end-to-end: 2 arms x the full 4-set panel -> metrics.json + table")
    print("      NOTE: the cell_only arm's centring control below reports `C2 "
          "FAILED`. The e2e\n      fixture is the SYMMETRIC T1 one, on which centring "
          "is provably inert (see T4f);\n      the control firing there is it doing "
          "its job, not a harness defect.")
    root = os.path.join(tmp, "e2e")
    exp = {}
    for lab, jit in (("cell_only", 0.0), ("finecls", 0.10)):
        d = os.path.join(root, lab)
        os.makedirs(d, exist_ok=True)
        for name in OOD_TRUE:
            arr, e1, e2 = _synth_arrays(name, 2, pool, 96, jitter=jit, seed=7)
            write_emb_npz(os.path.join(d, f"{name}.npz"), **arr)
            exp[name] = (e1, e2)
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump({"schema": SCHEMA, "arm": lab, "step": 8000,
                       "checkpoint": f"/fake/{lab}/snapshots/step_008000.pt",
                       "seed": 0, "git_sha": "deadbeef", "num_slots": 0,
                       "sets": list(OOD_TRUE),
                       "train_centering": {"global": True, "slots": lab == "finecls"}},
                      fh)
    out = os.path.join(root, "score")
    ns = argparse.Namespace(
        emb=[os.path.join(root, "cell_only"), os.path.join(root, "finecls")],
        expect_step=8000, out_dir=out, pool=pool, draws=draws, slot_draws=0,
        test_centering=("off", "donor_self"), center_split=0, contrast=[],
        allow_partial_panel=False, strict_controls=False, strict_granularity=False,
        granularity_max_ratio=2.0, mde=0.0023)
    rc = cmd_score(ns)
    t.check("T6a cmd_score returned 0", rc == 0)
    mp = os.path.join(out, "cell_only", "step_008000", "center_off", "metrics.json")
    t.check("T6b metrics.json exists per (arm, step, centring)", os.path.exists(mp), mp)
    with open(mp) as fh:
        mj = json.load(fh)
    t.check("T6c metrics.json records len(OOD)==4 and the registered panel",
            mj["panel"]["len_ood"] == 4 and mj["panel"]["is_registered_panel"])
    t.check("T6d metrics.json records TRAIN-time centring from the checkpoint",
            mj["train_centering"] == {"global": True, "slots": False},
            str(mj["train_centering"]))
    pm = mj["panel_mean_true_ood_only"][PRIMARY_STRATUM]
    t.check("T6e the 4-set panel mean equals the known per-set value",
            abs(pm["mean_r1_rna2atac"] - exp["bmmc"][0]) < 1e-12 and
            abs(pm["mean_r1_atac2rna"] - exp["bmmc"][1]) < 1e-12,
            f"r2a {pm['mean_r1_rna2atac']:.10f} a2r {pm['mean_r1_atac2rna']:.10f}")
    t.check("T6f a comparison table was written",
            os.path.exists(os.path.join(out, "comparison.csv")) and
            os.path.exists(os.path.join(out, "comparison.json")))
    with open(os.path.join(out, "comparison.json")) as fh:
        cj = json.load(fh)
    prim = [r for r in cj["rows"] if r["stratum"] == PRIMARY_STRATUM
            and r["test_centering"] == "off" and r["direction"] == "r2a"]
    t.check("T6g the contrast is NEGATIVE for the deliberately-degraded arm",
            len(prim) == 1 and prim[0]["panel_mean_delta"] < 0,
            f"delta {prim[0]['panel_mean_delta']:+.4f} over {prim[0]['n_sets']} sets"
            if prim else "no row")
    t.raises("T6h scoring a `best_` checkpoint is REFUSED at SCORE time too",
             CheckpointError,
             lambda: refuse_best_checkpoint(
                 json.load(open(os.path.join(root, "cell_only",
                                             "meta.json")))["checkpoint"]
                 .replace("step_008000.pt", "best_model.pt")), "REFUSED")

    # ---------------------------------------------- T7 the Stouffer z-vs-t correction
    print("\n[T7] Stouffer combines Z's, not t's (the panel p is the registered claim)")
    # A t on few df has heavier tails than a normal, so the SAME t carries a LARGER p
    # therefore a SMALLER |z|. Appending the raw t inflates the combine.
    t.check("T7a a t on small df maps to a strictly SMALLER |z|",
            abs(_z_from_t(3.0, 5)) < 3.0 - 1e-6,
            f"t=3.0 df=5 -> z={_z_from_t(3.0, 5):.4f} (p={two_sided_p(3.0, 5):.4f})")
    # The correction must SHRINK monotonically with df -- the property that makes it a
    # df effect and not an arbitrary rescale. It does not vanish at any finite df.
    gaps = [3.0 - abs(_z_from_t(3.0, d)) for d in (5, 20, 200, 5000)]
    t.check("T7b the correction shrinks monotonically with df",
            all(gaps[i] > gaps[i + 1] > 0 for i in range(len(gaps) - 1)),
            "gap at df 5/20/200/5000 = " + " / ".join(f"{g:.4f}" for g in gaps))
    t.check("T7c the sign is carried (two_sided_p is sign-blind)",
            _z_from_t(-3.0, 5) < 0 < _z_from_t(3.0, 5),
            f"{_z_from_t(-3.0, 5):+.4f} / {_z_from_t(3.0, 5):+.4f}")
    # The real panel's cluster counts, from the OOD obs after the 50/50 centring split.
    dfs = [17, 7, 5, 5]
    z_t = sum(3.0 for _ in dfs) / math.sqrt(len(dfs))
    z_z = sum(_z_from_t(3.0, d) for d in dfs) / math.sqrt(len(dfs))
    t.check("T7d on THIS panel's df the correction is material, not cosmetic",
            z_z < z_t - 0.15,
            f"raw-t combine Z {z_t:.3f} vs corrected {z_z:.3f} (df {dfs})")

    # ---------------------------------------- T8 the partial panel emits NO panel mean
    print("\n[T8] --allow_partial_panel: no panel mean, no Stouffer, in every artifact")
    fake_draws = {"hi": {}, "lo": {}}
    key = f"{PRIMARY_STRATUM}|off|global|r2a"
    gk = f"{PRIMARY_STRATUM}|__gname__"
    rng8 = np.random.RandomState(0)
    for nm in ("bmmc", "breast", "fetal_heart"):
        base = rng8.rand(200)
        fake_draws["hi"][nm] = {key: base + 0.05, gk: np.array(["g%d" % (i % 8)
                                                               for i in range(200)])}
        fake_draws["lo"][nm] = {key: base, gk: np.array(["g%d" % (i % 8)
                                                         for i in range(200)])}
    part = contrast_arms("hi", "lo", fake_draws, ["bmmc", "breast", "fetal_heart"],
                         full_panel=False)
    full = contrast_arms("hi", "lo", fake_draws, ["bmmc", "breast", "fetal_heart"],
                         full_panel=True)
    t.check("T8a a PARTIAL panel emits NaN for panel_mean_delta and stouffer",
            part and all(not np.isfinite(r["panel_mean_delta"])
                         and not np.isfinite(r["stouffer_z"]) for r in part),
            f"{len(part)} rows, panel_mean_delta "
            f"{part[0]['panel_mean_delta'] if part else 'n/a'}")
    t.check("T8b the SAME draws on a full panel DO emit both (both directions)",
            full and all(np.isfinite(r["panel_mean_delta"])
                         and np.isfinite(r["stouffer_z"]) for r in full),
            f"panel_mean_delta {full[0]['panel_mean_delta']:+.4f} "
            f"Z {full[0]['stouffer_z']:+.2f}")
    t.check("T8c every row records whether it is the registered panel",
            all(r["is_registered_panel"] is False for r in part)
            and all(r["is_registered_panel"] is True for r in full))

    # -------------------------------------------- T9 the SEED is the replication unit
    print("\n[T9] the seed-level combine -- what a per-draw p CANNOT see")
    def _row(delta, sets=("bmmc", "breast", "fetal_heart", "liver")):
        return {"contrast": "B-A", "scorer": "global", "stratum": PRIMARY_STRATUM,
                "test_centering": "off", "direction": "r2a",
                "panel_mean_delta": delta, "n_sets": len(sets),
                "is_registered_panel": True, "n_sets_positive": 4 if delta > 0 else 0,
                "stouffer_z": 5.0, "stouffer_p": 1e-6,
                "per_set": {k: {"delta": delta} for k in sets}}
    ood9 = ["bmmc", "breast", "fetal_heart", "liver"]
    # (a) THE FAILURE MODE. One seed lands 2 SD out; every set moves TOGETHER because
    # run-to-run term is a property of the trained models, not of the eval sets.  The
    # per-draw Stouffer in that single comparison.json reads Z = 5.0, p = 1e-6.
    one = combine_seeds({0: [_row(+0.0028)]}, ood9, 0.0023)
    t.check("T9a ONE seed cannot support a verdict, whatever its per-draw p says",
            len(one) == 1 and one[0]["verdict"].startswith("UNDETERMINED"),
            f"per-draw Z was 5.0 (p 1e-6); seed-level verdict {one[0]['verdict']!r}")
    # (b) three seeds that DISAGREE -> the honest answer is NULL, and the mean is
    # near zero even though each seed's own per-draw p was 1e-6.
    dis = combine_seeds({0: [_row(+0.0028)], 1: [_row(-0.0031)], 2: [_row(+0.0004)]},
                        ood9, 0.0023)
    t.check("T9b three DISAGREEING seeds -> NULL, not a win",
            dis[0]["verdict"] == "NULL" and dis[0]["n_seeds_positive"] == 2,
            f"mean {dis[0]['mean_panel_delta']:+.5f} "
            f"sigma_seed {dis[0]['sigma_seed_observed']:.5f} "
            f"p {dis[0]['p']:.3f} -> {dis[0]['verdict']}")
    # (c) the CONVERSE: three seeds that AGREE and clear the registered MDE -> WIN.
    agr = combine_seeds({0: [_row(+0.0060)], 1: [_row(+0.0052)], 2: [_row(+0.0058)]},
                        ood9, 0.0023)
    t.check("T9c three AGREEING seeds above the MDE -> WIN (positive both ways)",
            agr[0]["verdict"] == "WIN" and agr[0]["ci95"][0] > 0,
            f"mean {agr[0]['mean_panel_delta']:+.5f} "
            f"CI95 [{agr[0]['ci95'][0]:+.5f}, {agr[0]['ci95'][1]:+.5f}] "
            f"-> {agr[0]['verdict']}")
    neg = combine_seeds({0: [_row(-0.0060)], 1: [_row(-0.0052)], 2: [_row(-0.0058)]},
                        ood9, 0.0023)
    t.check("T9d a consistent NEGATIVE is reported as a replication, not a bug",
            neg[0]["verdict"].startswith("NEGATIVE"),
            f"mean {neg[0]['mean_panel_delta']:+.5f} -> {neg[0]['verdict']}")
    t.check("T9e sigma_seed is MEASURED here, not inherited from the projector track",
            np.isfinite(dis[0]["sigma_seed_observed"])
            and dis[0]["sigma_seed_observed"] > 0,
            f"observed sigma_seed {dis[0]['sigma_seed_observed']:.5f} "
            f"(registered, inherited value: 0.0010)")

    print("\n" + "=" * 96)
    print(f"{t.ok}/{t.n} passed" + ("" if not t.fail else f"   FAILED: {t.fail}"))
    print(f"artifacts: {tmp}")
    print("=" * 96)
    return 0 if not t.fail else 1


# ------------------------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="checkpoint -> embeddings npz (LIVE FMs, GPU)")
    e.add_argument("--ckpt", required=True,
                   help="a FIXED-STEP snapshot. ⛔ any path containing 'best' "
                        "is refused")
    e.add_argument("--expect_step", type=int, required=True,
                   help="the step every arm is scored at; must match name AND file")
    e.add_argument("--out_dir", required=True)
    e.add_argument("--label", default=None,
                   help="arm label; REQUIRED to tell ARM B from ARM C (both are "
                        "`finecls` in the trainer)")
    e.add_argument("--sets", nargs="+", default=list(OOD_TRUE))
    e.add_argument("--allow_partial_panel", action="store_true")
    e.add_argument("--skip_cell_count_gate", action="store_true",
                   help="⛔ only for a deliberately subsetted h5ad")
    e.add_argument("--rna_encoder_path",
                   default="/nfs/turbo/umms-drjieliu1/usr/xinyubao/scFoundation/model/"
                           "models/models1.ckpt")
    e.add_argument("--atac_encoder_path",
                   default=f"{REPO}/EpiAgent/model/pretrained_EpiAgent.pth")
    e.add_argument("--gene_list_path",
                   default=f"{REPO}/scFoundation/model/OS_scRNA_gene_index.19264.tsv")
    e.add_argument("--micro_batch", type=int, default=8)
    e.add_argument("--num_workers", type=int, default=4)
    e.add_argument("--max_cells", type=int, default=10 ** 9)
    e.add_argument("--device", default="cuda")
    e.add_argument("--dry_run", action="store_true",
                   help="rebuild the model from the checkpoint and stop before the FMs")
    e.set_defaults(fn=cmd_extract)

    s = sub.add_parser("score",
                       help="embeddings -> metrics.json + the comparison table")
    s.add_argument("--emb", action="append", required=True, metavar="[LABEL=]DIR",
                   help="an `extract` output dir; repeat per arm")
    s.add_argument("--expect_step", type=int, required=True)
    s.add_argument("--val_selected_steps", action="store_true",
                   help="allow the arms to sit at DIFFERENT steps because each was "
                        "selected independently by val loss. Legitimate ONLY when every "
                        "arm has been shown to be on its val plateau; the per-arm steps "
                        "are recorded in the output.")
    s.add_argument("--out_dir", required=True)
    s.add_argument("--pool", type=int, default=128)
    s.add_argument("--draws", type=int, default=500,
                   help=">= 200 enforced; single-draw retrieval is a known "
                        "artefact here")
    s.add_argument("--slot_draws", type=int, default=200,
                   help="draws for the ARM-B-internal slot/fused scorers (O(P^2 M D))")
    s.add_argument("--test_centering", nargs="+", default=list(CENTER_ARMS),
                   choices=list(CENTER_ARMS))
    s.add_argument("--center_split", type=int, default=1,
                   help="1 = the refiner_ood_center 50/50 split, so donor_ext/set_ext "
                        "have a DISJOINT reference and every rung shares the pools")
    s.add_argument("--contrast", action="append", default=[], metavar="HI:LO",
                   help="e.g. --contrast finecls_bio:cell_only "
                        "--contrast finecls_bio:finecls_rnd")
    s.add_argument("--allow_partial_panel", action="store_true")
    s.add_argument("--strict_controls", action="store_true",
                   help="turn a failed centring control into a non-zero exit")
    s.add_argument("--strict_granularity", action="store_true")
    s.add_argument("--granularity_max_ratio", type=float, default=2.0)
    s.add_argument("--mde", type=float, default=0.0023,
                   help="registered 3-seed MDE; only marks the table")
    s.set_defaults(fn=cmd_score)

    c = sub.add_parser("combine",
                       help="per-seed comparison.json -> the SEED-level verdict")
    c.add_argument("--seed", action="append", required=True, metavar="SEED=PATH",
                   help="e.g. --seed 0=results/seed0/comparison.json "
                        "--seed 1=... --seed 2=...")
    c.add_argument("--expect_step", type=int, required=True)
    c.add_argument("--mde", type=float, default=0.0023)
    c.add_argument("--primary_only", action="store_true")
    c.add_argument("--out", default=None, help="write the combined JSON here")
    c.set_defaults(fn=cmd_combine)

    t = sub.add_parser("selftest", help="synthetic + gate tests; CPU, no data")
    t.set_defaults(fn=cmd_selftest)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
