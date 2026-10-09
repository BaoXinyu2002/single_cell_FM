# FineCLS refiner (`finecls_bio`) — handoff bundle

Everything needed to read, re-score, or retrain the `B_finecls_bio` arm: the checkpoint's
location, the model definition, the evaluation and its fusion rule, and the exact training
command. **Code is copied for reading; it is not a runnable package** — see *Running it* below.

Repo root referred to below as `$REPO = /nfs/turbo/umms-drjieliu1/usr/xinyubao/sclip`.
Git SHA of the run: `b7fcc9087794a3aa48be7def8e8501f8818cc9c6`.

---

## 1. Checkpoints and artifacts (paths, not copies — the ckpt is 340 MB)

| what | path |
|---|---|
| **the scored checkpoint** (cool-down endpoint, step 30000) | `$REPO/experiments/finecls_refiner/runs/seed0_conv/finecls_bio/cooled_step_030000.pt` |
| val-loss-selected checkpoint | `.../finecls_bio/best_by_valloss.pt` |
| per-step snapshots | `.../finecls_bio/snapshots/step_*.pt` |
| val loss curve | `.../finecls_bio/val_loss_curve.jsonl` |
| **extracted embeddings** used by the comparison | `.../finecls_bio/emb_cooled/{bmmc,breast,fetal_heart,liver,islet,pln}.npz` + `meta.json` |
| FM layer init | `$REPO/experiments/finecls_refiner/fm_layer_state_n2.pt` |
| biology prior (RNA) | `$REPO/experiments/multiomics_clip_xinyu_June_fixed_slot_routing/biology_prior/fixed64_sce2g_v1/fixed64_rna_prior_sce2g_v1.npz` |
| biology prior (ATAC) | `.../fixed64_sce2g_v1/fixed64_atac_prior_sce2g_p3_v1.npz` |
| **training data** (see section 1b) | `$REPO/fm_token_cache_expanded_8192` |

⛔ **Do not use any checkpoint whose name contains `best`** for scoring: `eval_finecls_refiner.py`
refuses those paths at both extract and score time. The comparison below is on
`cooled_step_030000.pt`.

Sizes: 24,897,536 trainable params — 20,480,512 in the refiners, 2,907,648 in the slot branch.

## 1b. Training data

**`$REPO/fm_token_cache_expanded_8192`** -- 9.5 TB, passed as `--cache_root`. It caches
**per-token frozen-FM embeddings**, not raw counts: the two FMs are run once, offline, and the
refiner reads their token outputs. This is a *different* cache from the projector track's, which
stores POOLED features (`$REPO/preprocessed_data_3m/cache_annot`, 3072-d RNA / 512-d ATAC per
cell). Do not mix them.

```
fm_token_cache_expanded_8192/
  manifest.json      spec + per-split completeness
  index.npz          per-cell row directory (index.csv is the same content)
  train/  val/       shard files
```

| | |
|---|---|
| cells | **858,604** -- train **763,279** / val **95,325** |
| `dataset_id`s | **163** |
| RNA | 768-d per token, 3072-d cell vector, `rna_seq_len` 19266 (19264 genes + 2 meta), median 1953 valid tokens |
| ATAC | 512-d per token, `max_atac_length` **8192**, `atac_vocab_size` 1,355,449 |
| dtype | `float16`, `l2_normalized: false`, `include_summary: true` |
| build | `fm_mode: eval_dropout_off`, `rna_kernel: sdpa_no_bettertransformer` |

`index.npz` carries per cell: `barcode, split, orig_row, dataset_id, batch_id, label,
rna_shard/rna_row/rna_len/rna_nnz, atac_shard/atac_row/atac_len/atac_raw_len` -- tokens are
addressed by (shard, row), and the true pre-truncation length stays recoverable.

⛔ **53.1% of cells (456,229 / 858,604) are truncated at 8192.** Raw ATAC length has median 8787
and max 631,757, so the cap bites on more than half the corpus; `atac_raw_len` records what was
cut. Earlier runs on this track capped at 4096 -- a known deficit; this run is the 8192 one.

⛔ **This is the OLD corpus:** 163 `dataset_id`s / 763k training cells. The projector track has
since moved to **456 `dataset_id`s / 4,418,109 training cells**
(`preprocessed_data_3m/cache_annot`, which is the old 830k as a strict row-prefix plus 3.59 M new
cells). The refiner has never been trained there, which is exactly why the architecture question
in section 5 is unresolved on current data.

## 2. Model

`code/train_finecls_refiner.py` → **`class FineCLSRefiner(nn.Module)`** (line ~310). One module
on purpose (DDP wraps it whole). It composes:

- `code/modules/attn_refiner.py` → `RNARefinerFM`, `ATACRefinerFM` — 2-layer refiners initialised
  from FM layers (RNA scFoundation L10-11, ATAC EpiAgent L16-17), fed **cached FM token
  embeddings**, not the FMs themselves. Both end in `F.normalize(..., dim=-1)`.
- `code/modules/fixed_slot.py` → `FixedSlotPooler`, `fixed_slot_route_weights`,
  `fixed_slot_similarity`, `load_fixed_prior_npz` — the 64 biology-prior slots.
- `code/modules/blocked_sampler.py` → `SameDatasetBlockedBatchSampler` (a batch is one
  `dataset_id`, which is what makes the `--center_*_by_dataset` terms domain means).
- `code/cached_token_dataset.py`, `code/fm_token_cache.py` — the token-cache reader.
- `code/gradcache.py` — gradient caching (effective contrastive batch >> micro batch).

Arm A (`cell_only`) does **not** allocate the slot branch, so B−A confounds "the fine branch"
with ~2-3 M more parameters; arm C (`random64`) is the parameter-matched control.

## 3. Evaluation and the fusion rule

`code/eval_finecls_refiner.py` scores **three** matrices per pool:

| scorer | what it is |
|---|---|
| `global` | cosine between the two pooled cell vectors |
| `slot` | `fixed_slot_similarity` over the 64 slots |
| `fused` | **per-pool z-scored sum of the two** |

The fusion is `_fused_fn` (line ~1131), and it is deliberately per-pool:

```python
g  = zr[i] @ za[i].T          # global similarity
s  = slot_fn(i)               # slot similarity
gz = (g - g.mean()) / g.std().clamp(min=1e-6)
sz = (s - s.mean()) / s.std().clamp(min=1e-6)
return gz + sz                # z-sum
```

Per-pool rather than global z-scoring because the two scores live on different scales and the
pool is the unit the ranking happens in.

⛔ `_summarise` keeps **r2a and a2r separate and never averages them** — a verdict has to hold in
both directions.

`code/cross_track_table.py` is the script that produced the refiner-vs-projector comparison. Its
scorer is plain single-vector cosine (`S = r[p] @ a[p].T`), dataset-window pools of 128 x 200
draws, crc32-seeded per dataset so every arm sees the same pools, with **barcode intersection on
breast and liver** (the refiner's cells are a strict subset there: 8859/9739 and 29922/30135;
not intersecting moves breast a2r@1 by +0.0039, larger than the MDE of 0.0023).

## 3b. ⛔ TWO INFERENCE-TIME LEVERS YOU MUST APPLY — they are worth 2x

Neither is in the checkpoint. Both are applied **at scoring time**, cost nothing to train, and
together they nearly double OOD R@1. Measured on the projector track, 4 true-OOD sets
(bmmc/breast/fetal_heart/liver), R@1, pool 128 x 200 draws, dataset window:

| test-side treatment | R@1 | vs bare |
|---|---|---|
| bare cosine | 0.1040 | — |
| + CSLS(k=10) | 0.1314 | **x1.26** |
| + per-donor centring | 0.1799 | **x1.73** |
| **+ both** | **0.2065** | **x1.98** |

**1. Per-donor centring.** Subtract each donor's mean from the embeddings, then re-normalise,
*before* drawing pools. It is the single largest training-free OOD gain measured in this
project. It is genuine domain adaptation, not a generic representation fix: estimating the mean
on a DISJOINT half of the donor scores the same (+0.0063 vs +0.0062), while a mean taken from
TRAINING cells is worth +0.0001 (p=0.71).
⛔ Centre **per donor**, not per dataset — the per-donor operation is +0.0372 (6/6, p=0.024)
versus +0.0101 (4/6, p=0.175) for a whole-set mean. 10 of 163 `dataset_id`s pool several
samples (21.2% of cells), and on those a dataset mean is a study mean, not a donor mean.
⛔ A model trained with `--batch_center` **collapses if you L2-normalise without centring** —
centre first, then normalise.

**2. CSLS(k=10).** Cross-domain similarity local scaling, applied to the pool's score matrix.
`csls(S).T == csls(S.T)` exactly (both are `2S − rowmean − colmean`), which the scorer asserts —
if that fails, the a2r direction is wrong.

⚠️ **The table in §4 below is donor-centred but has NO CSLS**, so it understates every arm by
roughly +0.027. Any re-score, and any comparison against numbers from elsewhere, has to state
which rung it is on — the rung moves R@1 by more than any architecture contrast in this project.

## 4. The headline number

4-OOD means (bmmc / breast / fetal_heart / liver), donor test-centring, **no CSLS**, each arm
scored in the regime it trained in:

| arm | r2a@1 | a2r@1 | r2a@10 |
|---|---|---|---|
| **B `finecls_bio`** | **0.1484** | **0.1353** | 0.5426 |
| A `cell_only` | 0.1446 | 0.1352 | **0.5447** |
| C `random64` | 0.1482 | 0.1351 | 0.5389 |
| projector `P_c1` | 0.1332 | 0.1289 | 0.5054 |

**refiner best − projector best = +0.0152 r2a@1 / +0.0064 a2r@1 / +0.0372 r2a@10, 4/4 sets,
all above MDE 0.0023.** That contrast is the only real one in the panel.

## 5. What is NOT claimed

- **The bio prior is inert.** B − C (biology prior vs random 64 slots) is +0.0002 r2a@1, 3/4 —
  a 4th confirmation. The slot branch helps; *which* slots it uses does not.
- **FineCLS over plain refiner is not established.** B − A is +0.0038 (3/4) on r2a@1, +0.0000
  (1/4) on a2r@1, and **−0.0021 on r2a@10**.
- **The fused objective costs in-distribution** and returns nothing OOD (arms D/E).
- **Corpus.** These numbers are on the OLD corpus. The projector has since been retrained on a
  4.42 M-cell corpus and scores 0.1792/0.1806 at this same rung — ahead of this refiner, but the
  corpus contrast alone is worth +0.0435, so **the architecture question is unresolved on the
  current corpus**. The refiner has never been trained or scored there.
- **This track is NOT affected by the 2026-09-02 FILIP retraction.** That leak was in
  `filip_relevance_select` (token selection using the paired cell's other modality), used by the
  `ours_anchored*` evals. `cross_track_table.py` and the `global` scorer here are plain cosine on
  pooled vectors and never call it.
- Known open item: the refiner's ATAC input was truncated at 4096 in earlier runs; this run uses
  `--max_atac_length 8192`.

## 6. Running it

The copied files import each other through a `sys.path` bootstrap that expects the original
repo layout (`REPO`, `REPO/haoyun`, `REPO/haoyun/multiomics_clip_finelip`, `REPO/experiments`,
`REPO/scFoundation/model`) — see `_bootstrap_sys_path()` at the top of
`train_finecls_refiner.py`. **Run them from their original locations**, not from this folder;
this bundle is for reading the code and knowing where everything lives.

`TRAIN_COMMAND.txt` holds the exact argv of the run, and `run_manifest.json` its full recorded
configuration (including the liveness asserts the run passed).
