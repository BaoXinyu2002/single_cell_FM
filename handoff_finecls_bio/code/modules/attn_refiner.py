"""Trainable transformer layers ON TOP of the frozen FM, in the FM's NATIVE
token dim (RNA 768 / ATAC 512) — i.e. "2 more FM layers, but trainable".

Unlike the earlier attn_encoder (which projected to 256 + used a [CLS] bottleneck
+ top-256 token selection), this keeps ALL tokens at native dim, refines them
with N_LAYERS self-attention layers, pools by MAX-MEAN, optionally concatenates
the ORIGINAL frozen FM cell embedding (an OOD-generalizing anchor), then projects
to the shared contrastive space. SDPA (flash) attention + gradient checkpointing
keep all-token / native-dim training within memory.
"""
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

NEG = -1e4

#: The three arithmetic policies a refiner may run its stack under.  "off" means the
#: module's own dtype (fp32 on this track); the other two enter `torch.autocast`.
_AUTOCAST = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _autocast_ctx(name):
    """`torch.autocast` for "bf16"/"fp16", a NO-OP context for "off".

    Kept as one function so a precision policy is a single string that can be printed
    into a manifest and asserted on, rather than a `with` statement duplicated in four
    places that can drift.  On CPU `torch.autocast("cuda", ...)` is inert (torch warns
    and disables it), which is exactly why the CPU-only guard tests cannot see a
    precision change -- see the GPU numerics gate in experiments/finecls_refiner/.
    """
    dt = _AUTOCAST[name]
    if dt is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dt)


@contextlib.contextmanager
def no_bettertransformer(module):
    """Keep every nn.TransformerEncoderLayer inside `module` OFF PyTorch's BetterTransformer
    "sparsity fast path".

    nn.TransformerEncoderLayer.forward dispatches to aten::_transformer_encoder_layer_fwd ->
    native_multi_head_attention whenever ALL of these hold (torch/nn/modules/transformer.py:635-683):
        eval() mode  AND  grad disabled  AND  autocast off  AND  even n_heads  AND  batch_first.
    That kernel materializes a DENSE [B, n_heads, N, N] attention matrix instead of using SDPA's
    O(B*N*D) kernel: at B=32, 12 heads, N=5817 genes that is a single 48.41 GiB fp32 allocation
    (and 2-3 are live at once), which OOMs an H200. Training never hits it (the refiner is in
    train() mode), but evaluate() calls rna_ref.eval() under @torch.no_grad() and does.

    Running the layers in train() with EVERY dropout p forced to 0 is MATHEMATICALLY IDENTICAL to
    eval() -- dropout is the only train/eval-dependent op in nn.TransformerEncoderLayer -- but
    routes through F.multi_head_attention_forward -> scaled_dot_product_attention. Same trick as
    --fm_sdpa applies to the frozen scFoundation encoder in train_filip_combined.build().
    """
    saved = []
    for m in module.modules():
        if isinstance(m, nn.TransformerEncoderLayer) and not m.training:
            saved.append((m, m.dropout.p, m.dropout1.p, m.dropout2.p, m.self_attn.dropout))
            m.dropout.p = m.dropout1.p = m.dropout2.p = 0.0
            m.self_attn.dropout = 0.0
            m.train()
    try:
        yield
    finally:
        for m, p, p1, p2, pa in saved:
            m.eval()
            m.dropout.p, m.dropout1.p, m.dropout2.p, m.self_attn.dropout = p, p1, p2, pa


class TokenRefiner(nn.Module):
    """N trainable transformer layers on top of the frozen FM, in NATIVE token dim
    (all tokens), then a per-token projection to a shared space. Outputs refined,
    L2-normalized PER-TOKEN embeddings [B, N, proj_dim] for FILIP two-modality
    late interaction (NO pooling here — the reduction happens in the max-mean sim)."""

    def __init__(self, in_dim, proj_dim=256, n_layers=2, n_heads=8, dropout=0.1, grad_ckpt=True):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=in_dim, nhead=n_heads, dim_feedforward=in_dim * 2,
                                       dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(n_layers)])
        self.grad_ckpt = grad_ckpt
        self.proj = nn.Linear(in_dim, proj_dim)

    def forward(self, tokens, pad_mask):
        x = tokens
        with no_bettertransformer(self):                         # eval(): stay on SDPA, not [B,H,N,N]
            for layer in self.layers:
                if self.grad_ckpt and self.training:
                    x = checkpoint(lambda t, l=layer, m=pad_mask: l(t, src_key_padding_mask=m), x, use_reentrant=False)
                else:
                    x = layer(x, src_key_padding_mask=pad_mask)
        return F.normalize(self.proj(x), dim=-1)                 # [B, N, proj_dim], per-token L2-norm


class RNARefinerFM(nn.Module):
    """2 layers ARCHITECTURALLY IDENTICAL to scFoundation's encoder layer
    (nn.TransformerEncoderLayer d=768, 12 heads, FFN=4*d=3072, POST-norm, RELU) --
    i.e. "2 more scFoundation layers on top". Mask True=pad matches the FM output
    (no inversion). Then a Linear proj to the shared FILIP dim. Optionally init from
    the FM's top layers so it starts as a seamless continuation of the FM.

    THE TWO SPEED KNOBS ARE OFF BY DEFAULT, deliberately.  36 files outside
    experiments/finecls_refiner/ import this module, five of them through the
    `native()` contract that test_native_unchanged.py pins at max|delta| 0.000e+00.
    An unconditional edit here would move every one of them silently, so both knobs
    are constructor flags that the caller must ask for and record in its checkpoint
    config -- the precedent `ATACRefinerFM._patch_sdpa` already sets for `use_sdpa`.

    sub_batch  > 0: run the layer stack on CONTIGUOUS groups of `sub_batch` cells,
               each group trimmed to ITS OWN valid width.  0 = off (one call at the
               caller's padded width, i.e. the shipped behaviour).  MATH-IDENTICAL on
               the valid tokens -- see `_sub_batched`.
    autocast   "off" = fp32, the CLASS DEFAULT | "bf16" | "fp16".  NOT math-identical;
               measure it, do not assume it.  ⛔ "fp16" UNDERFLOWS HERE TOO -- measured
               3.89e-01 relative gradient error unscaled, 1.43e-03 with a loss scale of
               2^20 -- so a caller selecting it MUST also supply a GradScaler.  This was
               invisible until 2026-08-22 because no fp16 policy for the RNA side
               existed.  See ATACRefinerFM's `autocast` note and RUNBOOK_training.md 7b.
    """

    def __init__(self, in_dim=768, proj_dim=256, n_layers=2, n_heads=12, dropout=0.1,
                 grad_ckpt=True, sub_batch=0, autocast="off"):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=in_dim, nhead=n_heads, dim_feedforward=in_dim * 4,
                                       dropout=dropout, activation="relu", batch_first=True, norm_first=False)
            for _ in range(n_layers)])
        self.grad_ckpt = grad_ckpt
        self.proj = nn.Linear(in_dim, proj_dim)
        self.sub_batch = int(sub_batch)
        self.autocast = str(autocast)
        assert self.autocast in _AUTOCAST, \
            f"autocast must be one of {sorted(_AUTOCAST)}, got {self.autocast!r}"

    def init_from_fm(self, src_layers):
        n = len(self.layers); L = len(src_layers)
        for i in range(n):
            self.layers[i].load_state_dict(src_layers[L - n + i].state_dict())

    # -- the layer stack, factored out so `sub_batch` can call it per group --------- #

    def _layers(self, x, pad_mask):
        """The op sequence that WAS inline in `native_grad`, unchanged.

        `no_bettertransformer` is applied by the CALLER (once per `native_grad`), not
        here, so a sub-batched call does not flip every layer's train/eval flag
        `ceil(B/k)` times per micro-batch for the same guarantee.
        """
        for layer in self.layers:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    lambda t, l=layer, m=pad_mask: l(t, src_key_padding_mask=m),
                    x, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=pad_mask)
        return x

    def _sub_batched(self, tokens, pad_mask):
        """Refine CONTIGUOUS groups of `sub_batch` cells, each at its own valid width.

        WHY THIS IS EXACT, NOT AN APPROXIMATION.  The refiner is a PER-CELL function:
        self-attention is masked by `src_key_padding_mask` so it never crosses a cell,
        and there is no batch-norm and no cross-cell reduction anywhere in
        nn.TransformerEncoderLayer.  So the batch a cell is evaluated in is a free
        choice, and trimming a group to `max(valid length)` only removes columns that
        were masked out of every softmax anyway.  Measured on real micro-batches:
        max|delta| on the VALID tokens is 0.000e+00 on 2 of 3 and 2.06e-06 on the third
        (fp32 GEMM tile / split-k reduction order, which changes with N).

        WHY IT IS NOT THE BUCKETING THAT IS RULED OUT.  Nothing is sorted and nothing
        moves between micro-batches: groups are CONTIGUOUS SLICES in the sampler's own
        order, so batch composition stays exactly what the sampler drew and cannot
        become correlated with sequencing depth.  `sub_batch=1` pads NOTHING at all,
        which is strictly better than any regrouping could be, at zero data movement.

        ⛔ THE PAD REGION OF THE RETURN IS ZERO, not the shipped path's garbage.
        In the shipped path a pad ROW is still a query (only KEYS are masked), so it
        carries a batch-dependent junk activation; here it carries 0.  That is safe
        ONLY because every consumer multiplies the pad region by exactly zero:
        `fm_pool_rna` masks with `~rm` for the mean and `masked_fill(-1e4)` for the max
        and reads the meta pair at VALID positions (vc-2, vc-1); `FixedSlotPooler` sets
        `weights = membership * valid_tokens` and then `einsum(weights, tokens)`, so a
        pad column is multiplied by 0 in both the value and the gradient.  Do not
        "optimise" a consumer into reading `[:, -k:]` without revisiting this.
        """
        B, N, _ = tokens.shape
        k = self.sub_batch
        if k <= 0 or k >= B:
            return self._layers(tokens, pad_mask)
        # ⛔ THE PRECONDITION, ASSERTED RATHER THAN ASSUMED: the valid region of every
        # cell must be a PREFIX.  Trimming to `max(valid count)` is only the same
        # function if there is no padding BEFORE a valid token; with interior padding it
        # would silently DROP valid tokens and the loss would simply be a bit different.
        # `cached_collate` guarantees the prefix (it starts fully-padded and clears
        # `[:n]`, and asserts `(~rm).sum(1) == n_rna`), and the FMs pack valid-first, so
        # this is a cheap guard on a contract someone else owns -- exactly the kind that
        # is worth one reduction.  A prefix mask is equivalent to `pad_mask` being
        # monotone non-decreasing along the token axis.
        assert bool((pad_mask[:, 1:] >= pad_mask[:, :-1]).all()), (
            "sub_batch requires the valid region of every cell to be a PREFIX "
            "(pad_mask monotone along dim 1); this batch has INTERIOR padding, and "
            "trimming would drop valid tokens")
        # ONE device sync for the whole call: `int(t.max())` inside the loop would cost
        # ceil(B/k) syncs per micro-batch per pass, which at k=1 is 1,024 per step.
        lens = (~pad_mask).sum(1).tolist()
        out = []
        for lo in range(0, B, k):
            hi = min(lo + k, B)
            n = max(lens[lo:hi])
            assert n > 0, f"cells [{lo}:{hi}] are entirely padding"
            y = self._layers(tokens[lo:hi, :n], pad_mask[lo:hi, :n])
            if n < N:
                y = F.pad(y, (0, 0, 0, N - n))                 # right-pad the token dim
            out.append(y)
        return torch.cat(out, 0)

    def forward(self, tokens, pad_mask):
        if self.sub_batch > 0 or self.autocast != "off":
            # Keep the identity test_gradcache_equiv.py's T-id relies on --
            # forward() == normalize(proj(native_grad())) -- true UNDER the knobs too.
            return F.normalize(self.proj(self.native_grad(tokens, pad_mask)), dim=-1)
        x = tokens
        with no_bettertransformer(self):            # eval(): stay on SDPA, not a dense [B,12,N,N]
            for layer in self.layers:
                if self.grad_ckpt and self.training:
                    x = checkpoint(lambda t, l=layer, m=pad_mask: l(t, src_key_padding_mask=m), x, use_reentrant=False)
                else:
                    x = layer(x, src_key_padding_mask=pad_mask)
        return F.normalize(self.proj(x), dim=-1)

    def native_grad(self, tokens, pad_mask):
        """Refined PRE-proj tokens (native 768-d), GRAD-CARRYING. The FineCLS entry point.

        Identical arithmetic to `native()`; the ONLY difference is the absent
        @torch.no_grad(), so `loss.backward()` actually reaches self.layers. FineCLS pools
        at NATIVE width and projects afterwards, so its fine branch MUST consume this
        tensor -- and consuming `native()`'s output trains nothing: that result carries
        requires_grad=False / grad_fn=None (measured), so the fine term either raises
        "element 0 of tensors does not require grad" or, when it is summed with a global
        term that does carry grad, contributes SILENTLY NOTHING to the refiner.

        `no_bettertransformer` is KEPT and is load-bearing here, not vestigial. Torch
        disables its dense fast path when ANY of {self.training, autocast on,
        is_grad_enabled() and some tensor arg requires_grad} holds
        (torch/nn/modules/transformer.py:635-683), so during ordinary training it is a
        no-op -- its own guard is `not m.training`, so it also leaves dropout alone. It
        bites in the one grad-ENABLED configuration that is not training: a Stage-2-style
        frozen refiner, which `train_stage2_merge.load_frozen_refiner` builds with .eval()
        AND requires_grad_(False). There every fast-path condition passes and a dense
        [B, 12, N, N] materialises.  It is applied ONCE around the whole call, so
        `sub_batch` does not weaken it.

        The checkpoint gate additionally tests torch.is_grad_enabled(), so `native()`'s
        delegation below executes exactly the op sequence it executed before this method
        existed.

        `.float()` is applied ONLY under autocast (where the stack returns bf16/fp16 and
        every downstream `nn.Linear` still holds fp32 weights).  With autocast "off" the
        tensor is returned untouched, so the fp32 path is byte-for-byte what it was.
        """
        with no_bettertransformer(self):            # eval()+grad+frozen: dense-attention trap
            with _autocast_ctx(self.autocast):
                if self.sub_batch > 0:
                    x = self._sub_batched(tokens, pad_mask)
                else:
                    x = self._layers(tokens, pad_mask)
        return x if self.autocast == "off" else x.float()     # [B, N, 768]

    @torch.no_grad()
    def native(self, tokens, pad_mask):
        """Refined PRE-proj tokens (native 768-d) for a rich cell-level pooling.

        CONTRACT UNCHANGED: still @torch.no_grad(). All five existing callers
        (train_merge_cached.py:100/103, eval_summary_ood.py:148, eval_fuse_refined.py:67,
        eval_ood_metricsjson.py:157, diag_refiner_rollback.py:53) already sit inside an
        outer no_grad, so the decorator was redundant at each of them and is retained here
        purely as the guarantee attached to THIS name.  Those callers construct with the
        default `sub_batch=0, autocast="off"`, so what they get is unchanged -- which is
        what test_native_unchanged.py measures.
        """
        return self.native_grad(tokens, pad_mask)


class ATACRefinerFM(nn.Module):
    """2 layers IDENTICAL to EpiAgent's BERT block (flash_attn Block: d=512, 8 heads,
    FFN=4*d=2048, GELU, POST-norm). flash_attn key_padding_mask is True=VALID, so we
    INVERT the incoming pad_mask (True=pad). use_flash=True runs under fp16 autocast
    (matches the FM, memory-efficient); Linear proj to the shared FILIP dim.

    THREE ATTENTION IMPLEMENTATIONS, all on the SAME parameters (the state_dict is
    identical in keys, shapes and values across all three -- 26 tensors -- because
    flash_attn only swaps `mixer.inner_attn`, which has no parameters):
        default        pure-PyTorch SelfAttention, materialises [B, H, S, S]
        use_sdpa       F.scaled_dot_product_attention, never materialises it
        use_varlen     flash_attn varlen: unpad -> cu_seqlens -> pad_input.  Never
                       materialises the score matrix AND never computes the pad rows.
    They are mutually exclusive and the choice MUST be recorded in the checkpoint
    config; otherwise a later reader cannot tell which arithmetic produced a checkpoint.

    autocast  "fp16" (the CLASS DEFAULT, and the shipped arithmetic) | "bf16" | "off".
              ⛔ fp16 IS ONLY SAFE WITH A LOSS SCALE.  Unscaled it costs a MEASURED
              5.59e-01 relative gradient error at group cosine 0.845 on real production
              blocks -- pure UNDERFLOW: 97.7-98.3 % of the attention input projection's
              activation-gradient stream is flushed to exactly zero, and the fine-term
              gradient norm comes out 11-20 % SHORT, i.e. the objective is silently
              reweighted.  With `torch.cuda.amp.GradScaler(init_scale=2**20)` the same
              number is 9.42e-04 (bf16, which has fp32's exponent range and needs no
              scaler, sits at 3.87e-03).  ⛔ EVERY CALLER THAT BACKWARDS THROUGH THIS
              MODULE UNDER fp16 MUST SUPPLY THAT SCALER; the five `.native()` callers do
              not, but they are all forward-only under `no_grad`, so the requirement is
              on the TRAINERS.  `experiments/finecls_refiner/train_finecls_refiner.py`
              does it via `--refiner_precision fp16`; see RUNBOOK_training.md 7b for the
              composition with the two-pass gradient cache, which is where it is easy to
              get silently wrong.
    """

    def __init__(self, in_dim=512, proj_dim=256, n_layers=2, n_heads=8, dropout=0.1,
                 use_sdpa=False, use_varlen=False, autocast="fp16"):
        super().__init__()
        from transformers import BertConfig
        from flash_attn.models.bert import create_block
        cfg = BertConfig(num_hidden_layers=n_layers, hidden_size=in_dim, num_attention_heads=n_heads,
                         intermediate_size=in_dim * 4, max_position_embeddings=8192,
                         hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)
        # use_flash_attn=False: pure-PyTorch SelfAttention softmax path (fp32-safe, no
        # BertEncoder unpad needed). Calling Block directly can't use the flash kernel;
        # we get memory efficiency instead from fp16 autocast in forward().
        assert not (use_sdpa and use_varlen), (
            "use_sdpa and use_varlen are two different attention implementations of "
            "the same block; pick one and record it")
        cfg.use_flash_attn = bool(use_varlen)
        self.blocks = nn.ModuleList([create_block(cfg, layer_idx=i) for i in range(n_layers)])
        self.proj = nn.Linear(in_dim, proj_dim)
        self.use_sdpa = bool(use_sdpa)
        self.use_varlen = bool(use_varlen)
        self.autocast = str(autocast)
        assert self.autocast in _AUTOCAST, \
            f"autocast must be one of {sorted(_AUTOCAST)}, got {self.autocast!r}"
        if self.use_sdpa:
            self._patch_sdpa()
        if self.use_varlen:
            self._arm_varlen(dropout)

    def _patch_sdpa(self):
        """Route each block's inner attention through F.scaled_dot_product_attention.

        WHY. The pure-PyTorch path above MATERIALISES the [B, H, S, S] score matrix and stores the
        softmax output for backward, so at S=8192, H=8 it costs B x 1.07 GiB per layer under fp16
        and OOMs a 140 GiB H200 above batch 16. SDPA computes the SAME quantity -- same weights,
        same softmax_scale, same key padding -- without ever holding that matrix.

        MEASURED (probe_atac_refiner_mem.py, H200, S=8192, dropout=0, eval):
            peak GiB   B=64   B=128
            current     OOM     OOM
            sdpa       25.2    50.7
        and against the unpatched path on identical weights and input, max|delta| = 3.8e-05 on the
        L2-normalised output -- fp16 kernel noise. The chunked-recompute arm, which is identical by
        construction, returned exactly 0.0 in the same test, which is what validates the harness.

        DEFAULT OFF. This class exists to be weight- and math-identical to EpiAgent's BERT block,
        so the kernel swap is opt-in and is recorded in the checkpoint config -- otherwise a later
        reader cannot tell which attention implementation produced a given checkpoint. Attention
        dropout is forced to 0 on this path (SDPA's dropout_p is not wired), so use_sdpa must not
        be combined with a nonzero attention dropout expectation.
        """
        for blk in self.blocks:
            inner = blk.mixer.inner_attn

            def fwd(qkv, causal=None, key_padding_mask=None):
                q, k, v = qkv.unbind(dim=2)                        # each [B, S, H, D]
                q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # [B, H, S, D]
                am = key_padding_mask[:, None, None, :] if key_padding_mask is not None else None
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=0.0)
                return o.transpose(1, 2)                           # [B, S, H, D]

            inner.forward = fwd

    def _arm_varlen(self, dropout):
        """⛔ THE TRAP: `use_flash_attn=True` SILENTLY TURNS ATTENTION DROPOUT BACK ON.

        `create_block -> create_mixer_cls` builds `MHA(..., dropout=
        config.attention_probs_dropout_prob)`, which becomes
        `FlashSelfAttention(attention_dropout=0.2)` at the production `--dropout 0.2`.
        The SHIPPED `_patch_sdpa` path hardwires `dropout_p=0.0`.  So a bare
        `cfg.use_flash_attn=True` would ship a REGULARIZATION change disguised as a
        speed change, and it would surface as a worse-generalising arm and be
        misattributed to varlen.  Force it to 0 and ASSERT it, in the constructor, in
        the same place the flag is set.
        """
        from flash_attn.modules.mha import FlashSelfAttention
        for blk in self.blocks:
            inner = blk.mixer.inner_attn
            assert isinstance(inner, FlashSelfAttention), (
                f"use_varlen=True but inner_attn is {type(inner).__name__}; "
                f"cfg.use_flash_attn did not reach create_block")
            inner.drop.p = 0.0
        assert all(b.mixer.inner_attn.drop.p == 0.0 for b in self.blocks), \
            "attention dropout survived the varlen arming"
        self._varlen_attn_dropout_was = float(dropout)

    def init_from_fm(self, src_layers):
        n = len(self.blocks); L = len(src_layers)
        for i in range(n):
            self.blocks[i].load_state_dict(src_layers[L - n + i].state_dict())

    def _run_blocks(self, tokens, kpm):
        """The block stack.  `use_varlen` PACKS the batch first, so no pad token ever
        enters a GEMM or an attention -- this is the ATAC answer to padding, and the
        only one available: `atac_len`'s MEDIAN is 8192 and 53.2 % of cells sit at the
        cap, so the batch max is 8192 in essentially every micro-batch and no
        regrouping or sub-batching can shrink it (measured: SDPA padded to the batch
        max is BIT-IDENTICAL, 0.000e+00, to SDPA at 8192).

        This is verbatim `flash_attn.models.bert.BertEncoder.forward`, i.e. EpiAgent's
        own recipe -- the same `unpad_input`/`pad_input` pair the FM itself runs.
        """
        if not self.use_varlen:
            y = tokens
            for blk in self.blocks:
                y = blk(y, mixer_kwargs={"key_padding_mask": kpm})
            return y
        from flash_attn.bert_padding import pad_input, unpad_input
        B, N, _ = tokens.shape
        yu, idx, cu, ms = unpad_input(tokens, kpm)
        mk = {"cu_seqlens": cu, "max_seqlen": ms}
        for blk in self.blocks:
            yu = blk(yu, mixer_kwargs=mk)
        # `pad_input` scatters the valid rows back and leaves ZEROS in the pad region.
        # Safe for the same reason as RNARefinerFM._sub_batched: `fm_pool_atac` masks
        # with `~am`, and `FixedSlotPooler` multiplies every pad column by weight 0.
        return pad_input(yu, idx, B, N)

    def forward(self, tokens, pad_mask):
        kpm = ~pad_mask                                          # flash_attn convention: True=valid
        with _autocast_ctx(self.autocast):
            y = self._run_blocks(tokens, kpm)
        return F.normalize(self.proj(y.float()), dim=-1)

    def native_grad(self, tokens, pad_mask):
        """Refined PRE-proj tokens (native 512-d), GRAD-CARRYING. The FineCLS entry point.

        Same body as `native()` minus @torch.no_grad(). There is no BetterTransformer trap
        on this side (the blocks are flash_attn Blocks, not nn.TransformerEncoderLayer);
        the equivalent escape is `use_sdpa=True` OR `use_varlen=True`, which keeps the
        [B, H, S, S] score matrix from being materialised AND STORED FOR BACKWARD at
        S=8192. That is already the measured grad-enabled configuration:
        probe_atac_refiner_mem.py calls out.sum().backward() before it reads
        max_memory_allocated, and reports peak 25.2 GiB at B=64 / 50.7 GiB at B=128 for
        sdpa versus OOM for the unpatched path. So the constructor MUST be given one of
        the two on this track; assert it, do not assume it.  `use_varlen` gives the
        guarantee MORE strongly than `use_sdpa`: flash never forms the score matrix at
        all, and never even computes the padded rows.

        The autocast is kept verbatim at its "fp16" default, and the `.float()` at the
        end is a differentiable cast -- it does not break the graph.
        """
        kpm = ~pad_mask                                          # flash_attn: True=valid
        with _autocast_ctx(self.autocast):
            y = self._run_blocks(tokens, kpm)
        return y.float()                                         # [B, N, 512]

    @torch.no_grad()
    def native(self, tokens, pad_mask):
        """Refined PRE-proj tokens (native 512-d) for a rich cell-level pooling.

        CONTRACT UNCHANGED: still @torch.no_grad(), for the same five callers listed on
        RNARefinerFM.native.
        """
        return self.native_grad(tokens, pad_mask)


def make_refiners(refiner_type, model, proj_dim, n_layers, dropout, init_from_fm=False,
                  atac_sdpa=False, atac_varlen=False, rna_sub_batch=0,
                  rna_autocast="off", atac_autocast="fp16"):
    """Factory: returns (rna_ref, atac_ref). refiner_type 'generic' = TokenRefiner
    (nn.TransformerEncoderLayer, gelu/pre-norm/8-head/FFNx2); 'fm_layer' = FM-identical
    layers, optionally weight-initialized from the FM's top n_layers.

    atac_sdpa: route ATACRefinerFM's attention through SDPA. Weight- and math-identical
    (max|delta| 3.8e-05 measured), but it never materialises [B, H, S, S], which is the only
    way max_atac_length=8192 fits a usable batch. See ATACRefinerFM._patch_sdpa.

    atac_varlen: the flash_attn unpad path INSTEAD of atac_sdpa (mutually exclusive).
    rna_sub_batch / rna_autocast / atac_autocast: the speed knobs.  All default to the
    shipped behaviour; every one of them must be recorded in the checkpoint config."""
    if refiner_type == "generic":
        return (TokenRefiner(model.rna_token_dim, proj_dim, n_layers, dropout=dropout),
                TokenRefiner(model.atac_token_dim, proj_dim, n_layers, dropout=dropout))
    rna_ref = RNARefinerFM(model.rna_token_dim, proj_dim, n_layers, dropout=dropout,
                           sub_batch=rna_sub_batch, autocast=rna_autocast)
    atac_ref = ATACRefinerFM(model.atac_token_dim, proj_dim, n_layers, dropout=dropout,
                             use_sdpa=atac_sdpa, use_varlen=atac_varlen,
                             autocast=atac_autocast)
    if init_from_fm:
        rna_ref.init_from_fm(model.rna_encoder.model.encoder.transformer_encoder)
        atac_ref.init_from_fm(model.atac_encoder.model.EpiAgent_transformer.layers)
    return rna_ref, atac_ref


class TokenAttnRefiner(nn.Module):
    def __init__(self, in_dim, proj_dim=512, n_layers=2, n_heads=8, dropout=0.1,
                 cell_dim=0, grad_ckpt=True):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=in_dim, nhead=n_heads, dim_feedforward=in_dim * 2,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(n_layers)])
        self.grad_ckpt = grad_ckpt
        pooled_dim = 2 * in_dim + cell_dim          # [max || mean] (|| cell_emb)
        self.proj = nn.Sequential(nn.Linear(pooled_dim, proj_dim), nn.LayerNorm(proj_dim))

    def forward(self, tokens, pad_mask, cell_emb=None):
        """tokens [B, N, in_dim]; pad_mask [B, N] True=pad; cell_emb [B, cell_dim] or None.
        Returns L2-normalized cell embedding [B, proj_dim]."""
        x = tokens
        with no_bettertransformer(self):            # eval(): stay on SDPA, not a dense [B,H,N,N]
            for layer in self.layers:
                if self.grad_ckpt and self.training:
                    x = checkpoint(lambda t, l=layer, m=pad_mask: l(t, src_key_padding_mask=m),
                                   x, use_reentrant=False)
                else:
                    x = layer(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()               # [B, N, 1]
        mean = (x * valid).sum(1) / valid.sum(1).clamp(min=1.0)  # mean over valid tokens
        mx = x.masked_fill(pad_mask.unsqueeze(-1), NEG).max(1).values  # max over valid tokens
        pooled = torch.cat([mx, mean], dim=-1)                   # [B, 2*in_dim]
        if cell_emb is not None:
            pooled = torch.cat([pooled, cell_emb], dim=-1)
        return F.normalize(self.proj(pooled), dim=-1)
