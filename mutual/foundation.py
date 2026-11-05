"""Frozen foundation models used by MUTUAL: scFoundation (RNA) and EpiAgent (ATAC).

Plain-PyTorch re-implementations of the two encoders, loaded straight from the original checkpoints. No flash-attn,
no scFoundation / EpiAgent package imports. The arithmetic matches the feature exports MUTUAL was trained on:

  scFoundation  token_emb + pos_emb in fp32, 12 post-norm transformer layers (bf16 autocast on GPUs that support
                bf16, fp32 otherwise), final LayerNorm in fp32. Cell feature (3,072-d) = [final token of the log10-total
                meta token, final token of the resolution meta token, max over gene tokens, mean over gene tokens].
  EpiAgent      cCRE_embedding + rank_embedding in fp32, 18 post-norm BERT blocks (fp16 autocast on GPU, fp32 on CPU).
                Cell feature (512-d) = final [CLS] output.

Cells are batched by EXACT input length, so no padding mask is ever needed (results do not depend on batch
composition beyond kernel-level rounding).

ATAC token policy ("T*", the one MUTUAL was trained with): a cell sentence with L <= 8,190 tokens is fed as
[CLS] + sentence + [SEP]. A deeper cell is split into m = ceil(L / 8,190) interleaved subsets (positions j, j+m, ...),
each subset is forwarded on its own, and the cell's feature is the mean of the m [CLS] outputs.
"""
from __future__ import annotations

import contextlib
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_GENES = 19264
RNA_META_IDS = (19264, 19265)      # position ids of the two meta tokens (resolution, log10 total)
TARGET_RES = 4.0
PAD_ID = 103                       # scFoundation pad token id (never produced here: batches are unpadded)
MASK_ID = 102
ATAC_MAX = 8192                    # EpiAgent rank_embedding rows: [CLS] + 8,190 tokens + [SEP]
ATAC_KEEP = ATAC_MAX - 2
CLS_ID, SEP_ID, TOKEN_OFFSET = 1, 2, 4   # EpiAgent token id = cCRE.bed row + 4
N_CCRE = 1355445


# ----------------------------------------------------------------------------------------------------------------------
# scFoundation
# ----------------------------------------------------------------------------------------------------------------------
class AutoDiscretizationEmbedding2(nn.Module):
    """scFoundation/model/pretrainmodels/mae_autobin.py:AutoDiscretizationEmbedding2 (forward statement for statement)."""

    def __init__(self, dim=768, bin_num=100, bin_alpha=1.0, mask_token_id=MASK_ID, pad_token_id=PAD_ID):
        super().__init__()
        self.dim, self.bin_num, self.bin_alpha = dim, bin_num, bin_alpha
        self.mlp = nn.Linear(1, bin_num)
        self.mlp2 = nn.Linear(bin_num, bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(bin_num, dim)
        self.emb_mask = nn.Embedding(1, dim)
        self.emb_pad = nn.Embedding(1, dim)
        self.mask_token_id, self.pad_token_id = mask_token_id, pad_token_id

    def forward(self, x):                                   # x [B, N, 1]
        x_mask_idx = (x == self.mask_token_id).nonzero()
        x_pad_idx = (x == self.pad_token_id).nonzero()
        x = self.mlp(x)
        x = self.LeakyReLU(x)
        x_crosslayer = self.mlp2(x)
        x = self.bin_alpha * x + x_crosslayer
        weight = self.Softmax(x)
        token_emb = self.emb(torch.arange(self.bin_num, device=x.device))
        x = torch.matmul(weight, token_emb)
        t0 = torch.zeros((), dtype=torch.long, device=x.device)
        x[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = self.emb_mask(t0).type(x.dtype).repeat(x_mask_idx.shape[0], 1)
        x[x_pad_idx[:, 0], x_pad_idx[:, 1], :] = self.emb_pad(t0).type(x.dtype).repeat(x_pad_idx.shape[0], 1)
        return x


class ScFoundationEncoder(nn.Module):
    """token_emb + pos_emb + 12 post-norm nn.TransformerEncoderLayer + final LayerNorm (the 'gene' model's encoder)."""

    def __init__(self, dim=768, depth=12, heads=12, ff=3072, bin_num=100, bin_alpha=1.0, n_pos=19267):
        super().__init__()
        self.token_emb = AutoDiscretizationEmbedding2(dim, bin_num, bin_alpha)
        self.pos_emb = nn.Embedding(n_pos, dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=ff,
                                                                batch_first=True, norm_first=False, dropout=0.0)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def train(self, mode=True):
        # Every dropout is 0, so train() computes the same function as eval(); train() keeps torch off the fused
        # "fast path" kernel, which is what the training-time feature export used.
        super().train(mode)
        for layer in self.layers:
            layer.train(True)
        return self


def load_scfoundation(path, device="cpu"):
    """scFoundation encoder from the released checkpoint (models1.ckpt, key 'gene'), frozen."""
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["gene"]
    cfg = ck["config"]["model_config"]["mae_autobin"]
    enc = cfg["encoder"]
    assert (enc["hidden_dim"], enc["depth"], enc["heads"], enc["norm_first"]) == (768, 12, 12, False), enc
    assert (cfg["pad_token_id"], cfg["mask_token_id"], cfg["bin_num"], cfg["bin_alpha"]) == (PAD_ID, MASK_ID, 100, 1.0)
    m = ScFoundationEncoder(bin_num=cfg["bin_num"], bin_alpha=cfg["bin_alpha"])
    new = {}
    for k, v in ck["state_dict"].items():
        k = k[len("model."):]
        if k.startswith("token_emb.") or k.startswith("pos_emb."):
            new[k] = v
        elif k.startswith("encoder.transformer_encoder."):
            new["layers." + k[len("encoder.transformer_encoder."):]] = v
        elif k.startswith("encoder.norm."):
            new["norm." + k[len("encoder.norm."):]] = v
    m.load_state_dict(new, strict=True)
    m = m.to(device).train()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def rna_tokens(x_row):
    """One cell's scFoundation input from its log1p(CP10k) row over the 19,264 genes: the expressed genes in ascending
    column order, then the two meta tokens [4.0 (resolution), log10 total]. The total is recomputed from the row
    (sum of expm1(x) over the panel, float32), as in the training-time export. -> (values f32 [m], ids int64 [m])."""
    x = np.asarray(x_row, dtype=np.float32).ravel()
    assert x.shape[0] == N_GENES, x.shape
    total = np.expm1(x).sum()
    tok = np.log10(max(total, 1e-6) + 1e-6)
    nz = np.nonzero(x > 0)[0]
    vals = np.concatenate([x[nz], np.array([TARGET_RES, tok], dtype=np.float32)]).astype(np.float32)
    ids = np.concatenate([nz, np.array(RNA_META_IDS)]).astype(np.int64)
    return vals, ids


# ----------------------------------------------------------------------------------------------------------------------
# EpiAgent
# ----------------------------------------------------------------------------------------------------------------------
class EpiBlock(nn.Module):
    """flash_attn Block with prenorm=False, as EpiAgent builds it: h = LN1(MHA(h) + h); h = LN2(MLP(h) + h)."""

    def __init__(self, d=512, heads=8, ff=2048, eps=1e-12):
        super().__init__()
        self.heads = heads
        self.Wqkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d, eps=eps)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)
        self.norm2 = nn.LayerNorm(d, eps=eps)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.Wqkv(x).view(B, L, 3, self.heads, D // self.heads)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        o = F.scaled_dot_product_attention(q, k, v)
        o = self.out_proj(o.transpose(1, 2).reshape(B, L, D))
        x = self.norm1((o + x).to(self.norm1.weight.dtype))
        y = self.fc2(F.gelu(self.fc1(x), approximate="none"))
        return self.norm2((y + x).to(self.norm2.weight.dtype))


class EpiAgentEncoder(nn.Module):
    def __init__(self, vocab=N_CCRE + TOKEN_OFFSET, d=512, depth=18, heads=8, max_rank=ATAC_MAX):
        super().__init__()
        self.cCRE_embedding = nn.Embedding(vocab, d)
        self.rank_embedding = nn.Embedding(max_rank, d)
        self.layers = nn.ModuleList([EpiBlock(d, heads, 4 * d) for _ in range(depth)])


def load_epiagent(path, device="cpu"):
    """EpiAgent encoder from the released checkpoint (pretrained_EpiAgent.pth), frozen. The signal decoder is unused."""
    sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    m = EpiAgentEncoder()
    new = {"cCRE_embedding.weight": sd["cCRE_embedding.weight"], "rank_embedding.weight": sd["rank_embedding.weight"]}
    for i in range(18):
        p = f"EpiAgent_transformer.layers.{i}."
        for a, b in (("mixer.Wqkv", "Wqkv"), ("mixer.out_proj", "out_proj"), ("norm1", "norm1"), ("mlp.fc1", "fc1"),
                     ("mlp.fc2", "fc2"), ("norm2", "norm2")):
            for w in ("weight", "bias"):
                new[f"layers.{i}.{b}.{w}"] = sd[p + f"{a}.{w}"]
    m.load_state_dict(new, strict=True)
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def tstar_subsets(sentence):
    """T* policy: the whole sentence if it has <= 8,190 tokens, else m = ceil(L / 8,190) interleaved subsets."""
    s = np.asarray(sentence, dtype=np.int64).ravel()
    L = len(s)
    if L <= ATAC_KEEP:
        return [s]
    m = int(math.ceil(L / ATAC_KEEP))
    return [s[j::m] for j in range(m)]


def atac_wrap(selected):
    s = np.asarray(selected, dtype=np.int64).ravel()
    assert 1 <= len(s) <= ATAC_KEEP, len(s)
    assert s.min() >= TOKEN_OFFSET and s.max() < N_CCRE + TOKEN_OFFSET, "token ids must be cCRE.bed row + 4"
    return np.concatenate([[CLS_ID], s, [SEP_ID]]).astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# batched forward
# ----------------------------------------------------------------------------------------------------------------------
def _batches(lengths, budget, max_cells):
    """Equal-length batches: indices grouped by exact length, each <= budget tokens and <= max_cells cells."""
    lengths = np.asarray(lengths, np.int64)
    order = np.argsort(-lengths, kind="stable")
    out, i = [], 0
    while i < len(order):
        L = int(lengths[order[i]])
        j = i
        while j < len(order) and lengths[order[j]] == L:
            j += 1
        per = max(1, min(max_cells, budget // max(L, 1)))
        for a in range(i, j, per):
            out.append(order[a:min(j, a + per)])
        i = j
    return out


@contextlib.contextmanager
def _no_tf32():
    a, b = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = a, b


class FoundationModels:
    """The two frozen encoders. rna_features(log1p rows) -> [n, 3072]; atac_features(cell sentences) -> [n, 512].

    device: "cuda" or "cpu". On GPU, scFoundation runs under bf16 autocast (fp32 if the GPU has no bf16, e.g. V100)
    and EpiAgent under fp16 autocast; on CPU both run in fp32. token_budget / max_cells bound one forward batch.
    """

    def __init__(self, scf_ckpt=None, epiagent_ckpt=None, device="cuda", token_budget=262144, max_cells=512,
                 verbose=True):
        self.dev = torch.device(device)
        self.scf = load_scfoundation(scf_ckpt, self.dev) if scf_ckpt else None
        self.epi = load_epiagent(epiagent_ckpt, self.dev) if epiagent_ckpt else None
        self.budget, self.max_cells, self.verbose = token_budget, max_cells, verbose
        cuda = self.dev.type == "cuda"
        self.rna_amp = cuda and torch.cuda.is_bf16_supported()
        self.atac_amp = cuda

    def _log(self, *a):
        if self.verbose:
            print("[foundation]", *a, flush=True)

    def _run(self, fn, b):
        """fn(batch) -> array; on CUDA OOM the batch is split in halves (cells are independent)."""
        try:
            return fn(b)
        except torch.cuda.OutOfMemoryError:
            if len(b) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(b) // 2
            return np.concatenate([self._run(fn, b[:h]), self._run(fn, b[h:])])

    # ---------------- RNA ----------------
    @torch.no_grad()
    def _rna_batch(self, vals, ids):
        scf = self.scf
        with _no_tf32():
            x = scf.token_emb(vals.unsqueeze(2).float()) + scf.pos_emb(ids)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.rna_amp):
                for layer in scf.layers:
                    x = layer(x, src_key_padding_mask=None)
            f = scf.norm(x.float())
        m = f.shape[1]
        g = f[:, :m - 2]
        return torch.cat([f[:, m - 1], f[:, m - 2], g.max(1).values, g.sum(1) / float(m - 2)], 1).float()

    def rna_features(self, X):
        """X: log1p(CP10k) rows over the 19,264 scFoundation genes (dense or scipy.sparse), every row with >= 1
        expressed gene. -> float32 [n, 3072]."""
        assert self.scf is not None, "scFoundation checkpoint not loaded"
        X = X.tocsr() if hasattr(X, "tocsr") else np.asarray(X)
        n = X.shape[0]
        out = np.zeros((n, 3072), np.float32)
        t0 = time.time()
        for c0 in range(0, n, 4096):
            rows = range(c0, min(n, c0 + 4096))
            cells = [rna_tokens(X[i].toarray() if hasattr(X, "tocsr") else X[i]) for i in rows]
            lens = [len(v) for v, _ in cells]
            assert min(lens) >= 3, "every cell needs >= 1 expressed scFoundation gene"

            def run(b):
                V = torch.from_numpy(np.stack([cells[k][0] for k in b])).to(self.dev)
                I = torch.from_numpy(np.stack([cells[k][1] for k in b])).to(self.dev)
                return self._rna_batch(V, I).cpu().numpy()

            for b in _batches(lens, self.budget, self.max_cells):
                out[c0 + b] = self._run(run, b)
            self._log(f"scFoundation {min(n, c0 + 4096)}/{n} cells, {time.time() - t0:.0f}s")
        return out

    # ---------------- ATAC ----------------
    @torch.no_grad()
    def _atac_batch(self, ids):
        epi = self.epi
        L = ids.shape[1]
        with _no_tf32():
            y = epi.cCRE_embedding(ids) + epi.rank_embedding(torch.arange(L, device=ids.device))[None]
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.atac_amp):
                for layer in epi.layers:
                    y = layer(y)
        return y[:, 0].float()

    def atac_features(self, sentences):
        """sentences: one int array of EpiAgent token ids per cell, in cell-sentence order (highest TF-IDF first).
        T* policy (module docstring). -> float32 [n, 512]."""
        assert self.epi is not None, "EpiAgent checkpoint not loaded"
        n = len(sentences)
        out = np.zeros((n, 512), np.float64)
        cnt = np.zeros(n, np.int64)
        t0 = time.time()
        for c0 in range(0, n, 4096):
            seqs, owner = [], []
            for i in range(c0, min(n, c0 + 4096)):
                for s in tstar_subsets(sentences[i]):
                    seqs.append(atac_wrap(s))
                    owner.append(i)
            owner = np.asarray(owner, np.int64)

            def run(b):
                return self._atac_batch(torch.from_numpy(np.stack([seqs[k] for k in b])).to(self.dev)).cpu().numpy()

            for b in _batches([len(s) for s in seqs], self.budget, self.max_cells):
                r = self._run(run, b)
                np.add.at(out, owner[b], r.astype(np.float64))
                np.add.at(cnt, owner[b], 1)
            self._log(f"EpiAgent {min(n, c0 + 4096)}/{n} cells, {time.time() - t0:.0f}s")
        assert (cnt > 0).all(), "a cell has an empty sentence"
        return (out / cnt[:, None]).astype(np.float32)
