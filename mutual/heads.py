"""MUTUAL trainable heads (the part learned during MUTUAL training; 6,226,099 parameters at inference).

Per modality m in {RNA, ATAC}:
  FM view   x_fm = [l2(foundation feature) ; l2(linear summary)] / sqrt(2)          RNA 3,072 + 256, ATAC 512 + 256
            z_fm = l2(MLP(x_fm))                                                     MLP: Linear-GELU-LayerNorm-(Dropout)-Linear, 512 hidden, 128 out
  raw view  w    = detected raw features (RNA: log1p value / the cell's mean over detected genes; ATAC: accessible = 1)
            pooled = sum_j w_j exp(b_j) E_j / sum_j w_j exp(b_j)                  learned table E [features, 256], log-weights b
            z_raw = l2(MLP(LayerNorm(pooled)))                                       MLP as above, 128 out
  output    z = [sqrt(0.5) z_fm ; sqrt(0.5) z_raw]  (unit norm, 256-d); a cell with no detected raw feature gets [z_fm ; 0]
RNA and ATAC embeddings live in one space: cosine similarity compares cells across modalities.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECTED_KNOBS = {"HIDDEN": 512, "FM_VIEW_DIM": 128, "RAW_VIEW_DIM": 128, "PROJ_DIM": 256, "VIEWS": "both",
                  "FUSION_MODE": "late", "HEAD_LINEAR": False, "N_RES_BLOCKS": 0, "RNA_DIM": 3328, "ATAC_DIM": 768,
                  "TABLE_SRC_ATAC": "random", "TABLE_LEARN_ATAC": "learn", "ADAPTER_ATAC": "none"}
FUSE_ALPHA = 0.5
BAG_DIM = 256


def l2(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class ProjectionHead(nn.Module):
    """Linear -> GELU -> LayerNorm -> Dropout -> Linear."""

    def __init__(self, d_in, d_hidden=512, d_out=128, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.drop(self.norm(F.gelu(self.fc1(x)))))


class RawBagView(nn.Module):
    def __init__(self, n_feat, kind):
        super().__init__()
        self.kind = kind
        self.E = nn.Parameter(torch.zeros(n_feat, BAG_DIM))
        self.b = nn.Parameter(torch.zeros(n_feat))
        self.mlp = ProjectionHead(BAG_DIM)

    def forward(self, x_raw):
        x = x_raw.float().clamp_min(0)
        w = (x > 0).float() if self.kind == "atac" else x
        nnz = (w > 0).sum(1)
        if self.kind == "rna":
            w = w / (w.sum(1, keepdim=True) / nnz.clamp_min(1)[:, None]).clamp_min(1e-6)
        ww = w * self.b.clamp(-5.0, 5.0).exp()[None]
        pooled = (ww @ self.E) / ww.sum(1, keepdim=True).clamp_min(1e-6)
        return self.mlp(F.layer_norm(pooled, (BAG_DIM,))), nnz


class MUTUALHeads(nn.Module):
    def __init__(self, rna_dim=3328, atac_dim=768, n_hvg=4000, n_ccre=10000):
        super().__init__()
        self.rna_heads = nn.ModuleList([ProjectionHead(rna_dim)])
        self.atac_heads = nn.ModuleList([ProjectionHead(atac_dim)])
        self.rna_bag = RawBagView(n_hvg, "rna")
        self.atac_bag = RawBagView(n_ccre, "atac")
        # learned contrastive temperatures (training only; kept so the checkpoint loads strictly)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.log_temp_fm = nn.Parameter(torch.zeros(()))
        self.log_temp_raw = nn.Parameter(torch.zeros(()))

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Load a MUTUAL snapshot (seed*/step_400000.pt: {config, knobs, state, step}); fp16 weights -> fp32."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        kn = blob["knobs"]
        bad = {k: (kn.get(k), v) for k, v in EXPECTED_KNOBS.items() if kn.get(k) != v}
        assert not bad, f"checkpoint is not the MUTUAL (XCAT) configuration: {bad}"
        m = cls()
        sd = m.state_dict()
        st = blob["state"]
        missing, extra = set(sd) - set(st), set(st) - set(sd)
        assert not missing and not extra, (sorted(missing), sorted(extra))
        m.load_state_dict({k: st[k].to(sd[k].dtype) for k in sd}, strict=True)
        m.step = int(blob["step"])
        return m.to(device).eval()

    @staticmethod
    def fm_input(fm_feature, linear_summary):
        """[l2(FM feature) ; l2(linear summary)] / sqrt(2), both taken as float16 -> float32 (the training dtype)."""
        a = l2(torch.as_tensor(np.asarray(fm_feature, np.float16)).float())
        b = l2(torch.as_tensor(np.asarray(linear_summary, np.float16)).float())
        return torch.cat([a, b], 1) / math.sqrt(2.0)

    def _encode(self, head, bag, x, x_raw):
        z_fm = l2(head(x))
        z_raw_pre, nnz = bag(x_raw)
        z_raw = l2(z_raw_pre)
        a = torch.where(nnz > 0, torch.full(nnz.shape, FUSE_ALPHA, device=x.device), torch.ones(nnz.shape, device=x.device))[:, None]
        return torch.cat([a.sqrt() * z_fm, (1.0 - a).clamp_min(0).sqrt() * z_raw], 1)

    @torch.no_grad()
    def encode(self, modality, fm_feature, linear_summary, raw, batch_size=8192):
        """modality 'rna' | 'atac'. Inputs are numpy arrays [n, .]. -> float32 numpy [n, 256], unit norm."""
        head, bag = ((self.rna_heads[0], self.rna_bag) if modality == "rna" else (self.atac_heads[0], self.atac_bag))
        dev = next(self.parameters()).device
        X = self.fm_input(fm_feature, linear_summary)
        R = torch.as_tensor(np.asarray(raw, np.float16))
        out = [self._encode(head, bag, X[i:i + batch_size].to(dev), R[i:i + batch_size].to(dev).float()).float().cpu()
               for i in range(0, X.shape[0], batch_size)]
        return torch.cat(out, 0).numpy() if out else np.zeros((0, 256), np.float32)
