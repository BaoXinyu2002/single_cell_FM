"""Deterministic biology-prior fixed-slot pooling and alignment.

This module is intentionally independent of the FineLIP aggregation, routing,
similarity, and loss implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class LoadedFixedPrior:
    matrix: torch.Tensor
    feature_ids: np.ndarray
    feature_symbols: Optional[np.ndarray]
    slot_ids: np.ndarray
    slot_names: np.ndarray
    matrix_name: str
    path: str


def _read_csr(z: np.lib.npyio.NpzFile, name: str) -> csr_matrix:
    required = [
        f"{name}_data",
        f"{name}_indices",
        f"{name}_indptr",
        f"{name}_shape",
    ]
    missing = [key for key in required if key not in z.files]
    if missing:
        raise ValueError(f"Prior is missing CSR arrays for '{name}': {missing}")
    shape = tuple(int(x) for x in z[f"{name}_shape"])
    return csr_matrix(
        (z[f"{name}_data"], z[f"{name}_indices"], z[f"{name}_indptr"]),
        shape=shape,
    )


def load_fixed_prior_npz(
    path: str,
    matrix_name: str,
    expected_rows: int,
    expected_num_slots: int,
    expected_feature_ids: Optional[Sequence[str]] = None,
    expected_feature_symbols: Optional[Sequence[str]] = None,
) -> LoadedFixedPrior:
    """Load a sparse prior, validate identifiers/order, and materialize a buffer.

    Priors are small in column count (16 for the MVP). A dense row-indexable
    tensor makes per-batch token lookup deterministic and avoids constructing a
    huge `[B, vocabulary, slots]` tensor during pooling.
    """

    prior_path = Path(path).expanduser().resolve()
    if not prior_path.is_file():
        raise FileNotFoundError(f"Fixed-slot prior does not exist: {prior_path}")

    with np.load(prior_path, allow_pickle=False) as z:
        matrix = _read_csr(z, matrix_name)
        if matrix.shape != (expected_rows, expected_num_slots):
            raise ValueError(
                f"{prior_path}: expected shape {(expected_rows, expected_num_slots)}, "
                f"got {matrix.shape}"
            )
        for key in ("feature_ids", "slot_ids", "slot_names"):
            if key not in z.files:
                raise ValueError(f"{prior_path}: missing required identifier array '{key}'")

        feature_ids = z["feature_ids"].astype(str)
        feature_symbols = (
            z["feature_symbols"].astype(str) if "feature_symbols" in z.files else None
        )
        slot_ids = z["slot_ids"].astype(str)
        slot_names = z["slot_names"].astype(str)

    if len(feature_ids) != expected_rows:
        raise ValueError(
            f"{prior_path}: feature_ids length {len(feature_ids)} != {expected_rows}"
        )
    if len(slot_ids) != expected_num_slots or len(slot_names) != expected_num_slots:
        raise ValueError(f"{prior_path}: slot identifier lengths do not match matrix")
    if len(set(slot_ids.tolist())) != expected_num_slots:
        raise ValueError(f"{prior_path}: slot IDs are not unique")

    if expected_feature_ids is not None:
        expected = np.asarray(expected_feature_ids, dtype=str)
        if not np.array_equal(feature_ids, expected):
            mismatch = int(np.flatnonzero(feature_ids != expected)[0])
            raise ValueError(
                f"{prior_path}: feature ID order mismatch at row {mismatch}: "
                f"prior={feature_ids[mismatch]!r}, runtime={expected[mismatch]!r}"
            )
    if expected_feature_symbols is not None:
        if feature_symbols is None:
            raise ValueError(f"{prior_path}: expected feature_symbols but none are embedded")
        expected = np.asarray(expected_feature_symbols, dtype=str)
        if not np.array_equal(feature_symbols, expected):
            mismatch = int(np.flatnonzero(feature_symbols != expected)[0])
            raise ValueError(
                f"{prior_path}: feature symbol order mismatch at row {mismatch}: "
                f"prior={feature_symbols[mismatch]!r}, runtime={expected[mismatch]!r}"
            )

    dense = torch.from_numpy(matrix.toarray().astype(np.float32, copy=False))
    if not torch.isfinite(dense).all() or (dense < 0).any():
        raise ValueError(f"{prior_path}: memberships must be finite and non-negative")
    return LoadedFixedPrior(
        matrix=dense,
        feature_ids=feature_ids,
        feature_symbols=feature_symbols,
        slot_ids=slot_ids,
        slot_names=slot_names,
        matrix_name=matrix_name,
        path=str(prior_path),
    )


class FixedSlotPooler(nn.Module):
    """Pool token embeddings with a fixed `[vocabulary, slots]` prior."""

    def __init__(
        self,
        prior: torch.Tensor,
        token_offset: int = 0,
        eps: float = 1e-8,
        attention_dim: Optional[int] = None,
        attention_temperature: float = 0.5,
        pool_topk: int = 0,
    ):
        super().__init__()
        if prior.ndim != 2:
            raise ValueError(f"prior must be rank 2, got {tuple(prior.shape)}")
        if token_offset < 0:
            raise ValueError("token_offset must be non-negative")
        # Recreated from the configured artifact on checkpoint load; omitting it
        # from state_dict keeps every checkpoint from growing by ~87 MB.
        self.register_buffer("prior", prior.float(), persistent=False)
        self.token_offset = int(token_offset)
        self.eps = float(eps)
        self.attention_temperature = float(attention_temperature)
        # Attention pooling is a convex combination and therefore cannot express
        # an extreme-value statistic.  When pool_topk > 0 the pooled vector is
        # [attention_mean, masked_top_k_mean], so "some member of this module is
        # strongly on" survives aggregation.
        self.pool_topk = int(pool_topk)
        if self.pool_topk < 0:
            raise ValueError("pool_topk must be non-negative")
        self.slot_queries = None
        if attention_dim is not None:
            if attention_dim <= 0 or attention_temperature <= 0:
                raise ValueError("attention_dim and attention_temperature must be positive")
            self.slot_queries = nn.Parameter(torch.empty(self.num_slots, attention_dim))
            nn.init.normal_(self.slot_queries, std=attention_dim ** -0.5)

    @property
    def num_slots(self) -> int:
        return int(self.prior.shape[1])

    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must have shape [B,N,D]")
        if token_ids.shape != token_embeddings.shape[:2]:
            raise ValueError("token_ids shape must match token_embeddings[:2]")
        if padding_mask.shape != token_ids.shape:
            raise ValueError("padding_mask shape must match token_ids")

        row_ids = token_ids.long() - self.token_offset
        in_vocabulary = (row_ids >= 0) & (row_ids < self.prior.shape[0])
        valid_tokens = (~padding_mask.bool()) & in_vocabulary
        safe_rows = row_ids.clamp(min=0, max=self.prior.shape[0] - 1)
        membership = self.prior[safe_rows]
        weights = membership * valid_tokens.unsqueeze(-1).to(membership.dtype)
        # Preserve biological evidence before attention normalizes every active
        # slot to unit mass. This drives cell-specific sparse routing.
        mass = weights.sum(dim=1)
        active = mass > self.eps
        if self.slot_queries is not None:
            if token_embeddings.shape[-1] != self.slot_queries.shape[-1]:
                raise ValueError("token embedding dimension does not match slot queries")
            logits = torch.einsum(
                "bnd,md->bnm", token_embeddings.float(), self.slot_queries.float()
            ) / self.attention_temperature
            logits = logits.masked_fill(weights <= 0, -torch.inf)
            max_logits = logits.amax(dim=1, keepdim=True)
            max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
            attention = torch.exp(logits - max_logits) * weights.float()
            weights = attention / attention.sum(dim=1, keepdim=True).clamp_min(self.eps)

        denom = weights.sum(dim=1)  # [B,M]
        # Accumulate potentially thousands of token contributions in fp32 even
        # under AMP. This prevents fp16 sum overflow while preserving gradients
        # through the projected token embeddings.
        numerator = torch.einsum(
            "bnm,bnd->bmd", weights.float(), token_embeddings.float()
        )
        slots = numerator / denom.clamp_min(self.eps).unsqueeze(-1)
        slots = torch.where(active.unsqueeze(-1), slots, torch.zeros_like(slots))
        slots = torch.nan_to_num(slots, nan=0.0, posinf=0.0, neginf=0.0)

        if self.pool_topk > 0:
            slots = torch.cat((slots, self._topk_mean(token_embeddings, membership,
                                                      valid_tokens, active)), dim=-1)
        return slots, active, mass

    def _topk_mean(self, token_embeddings: torch.Tensor, membership: torch.Tensor,
                   valid_tokens: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Mean of each module's k largest member values, per embedding dimension.

        k = 1 is a plain max.  Larger k keeps the extreme-value semantics while
        being far less sensitive to how many tokens the cell happened to sample,
        which matters because sequencing depth is a strong non-monotonic
        confounder in this data.
        """
        member = (membership > 0) & valid_tokens.unsqueeze(-1)     # [B,N,M]
        x = token_embeddings.float()                               # [B,N,D]
        neg_inf = torch.finfo(x.dtype).min
        outputs = []
        # A [B,M,N,D] intermediate would be prohibitive, so reduce one slot at a
        # time: M is 2-64 here while N reaches 8k.
        for slot in range(member.shape[-1]):
            mask = member[..., slot].unsqueeze(-1)                  # [B,N,1]
            masked = x.masked_fill(~mask, neg_inf)
            k = min(self.pool_topk, masked.shape[1])
            top = masked.topk(k, dim=1).values                      # [B,k,D]
            counts = mask.squeeze(-1).sum(dim=1).clamp_min(1)       # [B]
            keep = torch.arange(k, device=x.device)[None, :] < counts[:, None]
            top = torch.where(keep.unsqueeze(-1), top, torch.zeros_like(top))
            outputs.append(top.sum(dim=1) / keep.sum(dim=1).clamp_min(1).unsqueeze(-1))
        out = torch.stack(outputs, dim=1)                           # [B,M,D]
        out = torch.where(active.unsqueeze(-1), out, torch.zeros_like(out))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fixed_slot_route_weights(
    mass: torch.Tensor,
    valid: torch.Tensor,
    topk: int = 0,
    mass_power: float = 1.0,
    tail_weight: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert observed module evidence into sparse, per-cell route weights."""
    if mass.shape != valid.shape:
        raise ValueError("mass and valid must have the same [B,M] shape")
    if mass_power <= 0:
        raise ValueError("mass_power must be positive")
    evidence = torch.log1p(mass.float().clamp_min(0)).pow(mass_power)
    evidence = evidence * valid.to(evidence.dtype)
    if not 0 <= tail_weight < 1:
        raise ValueError("tail_weight must be in [0, 1)")
    routed = valid.bool().clone()
    if 0 < topk < evidence.shape[1]:
        indices = evidence.topk(topk, dim=1).indices
        head = torch.zeros_like(valid, dtype=torch.bool)
        head.scatter_(1, indices, True)
        head &= valid.bool()
        head_evidence = evidence * head.to(evidence.dtype)
        if tail_weight > 0:
            tail = valid.bool() & ~head
            tail_evidence = evidence * tail.to(evidence.dtype)
            head_weights = head_evidence / head_evidence.sum(dim=1, keepdim=True).clamp_min(eps)
            tail_weights = tail_evidence / tail_evidence.sum(dim=1, keepdim=True).clamp_min(eps)
            has_tail = tail_evidence.sum(dim=1, keepdim=True) > 0
            weights = ((1.0 - tail_weight) * head_weights
                       + tail_weight * tail_weights * has_tail.to(evidence.dtype))
            routed = valid.bool()
        else:
            routed = head
            weights = head_evidence / head_evidence.sum(dim=1, keepdim=True).clamp_min(eps)
    else:
        weights = evidence / evidence.sum(dim=1, keepdim=True).clamp_min(eps)
    return weights, routed


def fixed_slot_similarity(
    rna_slots: torch.Tensor,
    atac_slots: torch.Tensor,
    rna_valid: torch.Tensor,
    atac_valid: torch.Tensor,
    rna_mass: Optional[torch.Tensor] = None,
    atac_mass: Optional[torch.Tensor] = None,
    routing_topk: int = 0,
    routing_mass_power: float = 1.0,
    routing_tail_weight: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare equal slot indices across every RNA/ATAC cell pair."""

    if rna_slots.ndim != 3 or atac_slots.ndim != 3:
        raise ValueError("slot tensors must have shape [B,M,D]")
    if rna_slots.shape[1:] != atac_slots.shape[1:]:
        raise ValueError(
            f"RNA/ATAC slot shapes must agree, got {rna_slots.shape[1:]} and "
            f"{atac_slots.shape[1:]}"
        )
    if rna_valid.shape != rna_slots.shape[:2] or atac_valid.shape != atac_slots.shape[:2]:
        raise ValueError("valid masks must have shape [B,M]")

    rna_norm = F.normalize(rna_slots.float(), dim=-1, eps=eps)
    atac_norm = F.normalize(atac_slots.float(), dim=-1, eps=eps)
    per_slot = torch.einsum("imd,jmd->ijm", rna_norm, atac_norm)
    if rna_mass is None or atac_mass is None:
        rna_weight = rna_valid.float() / rna_valid.float().sum(1, keepdim=True).clamp_min(eps)
        atac_weight = atac_valid.float() / atac_valid.float().sum(1, keepdim=True).clamp_min(eps)
        rna_route, atac_route = rna_valid.bool(), atac_valid.bool()
    else:
        rna_weight, rna_route = fixed_slot_route_weights(
            rna_mass, rna_valid, routing_topk, routing_mass_power, routing_tail_weight, eps
        )
        atac_weight, atac_route = fixed_slot_route_weights(
            atac_mass, atac_valid, routing_topk, routing_mass_power, routing_tail_weight, eps
        )
    pair_valid = rna_route[:, None, :] & atac_route[None, :, :]
    weights = torch.sqrt(
        rna_weight[:, None, :].clamp_min(0) * atac_weight[None, :, :].clamp_min(0)
    ) * pair_valid.to(per_slot.dtype)
    denom = weights.sum(dim=-1)
    scores = (per_slot * weights).sum(dim=-1) / denom.clamp_min(eps)
    return torch.where(denom > 0, scores, torch.zeros_like(scores))


def paired_block_identity_loss(
    rna_slots: torch.Tensor,
    atac_slots: torch.Tensor,
    rna_valid: torch.Tensor,
    atac_valid: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, dict]:
    """Identify the matching ATAC block among blocks from the same paired cell."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    rna = F.normalize(rna_slots.float(), dim=-1)
    atac = F.normalize(atac_slots.float(), dim=-1)
    scores = torch.einsum("bmd,bnd->bmn", rna, atac)
    pair_valid = rna_valid.bool()[:, :, None] & atac_valid.bool()[:, None, :]
    labels = torch.arange(scores.shape[1], device=scores.device)[None, :].expand(scores.shape[0], -1)
    diagonal_valid = rna_valid.bool() & atac_valid.bool()

    forward_logits = (scores / temperature).masked_fill(~pair_valid, -1e4)
    reverse_logits = (scores.transpose(1, 2) / temperature).masked_fill(
        ~pair_valid.transpose(1, 2), -1e4
    )
    if not diagonal_valid.any():
        zero = scores.sum() * 0.0
        return zero, {"block_identity_loss": 0.0, "block_identity_accuracy": 0.0,
                      "block_matched_cosine": 0.0, "block_wrong_cosine": 0.0,
                      "block_identity_margin": 0.0}
    loss_forward = F.cross_entropy(
        forward_logits[diagonal_valid], labels[diagonal_valid]
    )
    loss_reverse = F.cross_entropy(
        reverse_logits[diagonal_valid], labels[diagonal_valid]
    )
    loss = 0.5 * (loss_forward + loss_reverse)
    diagonal = scores.diagonal(dim1=1, dim2=2)
    wrong_mask = pair_valid & ~torch.eye(scores.shape[1], device=scores.device, dtype=torch.bool)[None]
    wrong = scores[wrong_mask]
    matched = diagonal[diagonal_valid]
    accuracy = 0.5 * (
        (forward_logits.argmax(-1)[diagonal_valid] == labels[diagonal_valid]).float().mean()
        + (reverse_logits.argmax(-1)[diagonal_valid] == labels[diagonal_valid]).float().mean()
    )
    wrong_mean = wrong.mean() if wrong.numel() else scores.new_zeros(())
    return loss, {
        "block_identity_loss": float(loss.detach().item()),
        "block_identity_accuracy": float(accuracy.detach().item()),
        "block_matched_cosine": float(matched.detach().mean().item()),
        "block_wrong_cosine": float(wrong_mean.detach().item()),
        "block_identity_margin": float((matched.mean() - wrong_mean).detach().item()),
    }


def symmetric_fixed_slot_loss(
    scores_rna_to_atac: torch.Tensor,
    scores_atac_to_rna: torch.Tensor,
    positive_indices: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, dict]:
    """Standard symmetric diagonal contrastive loss for fixed-slot scores."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    loss_rna = F.cross_entropy(scores_rna_to_atac / temperature, positive_indices)
    loss_atac = F.cross_entropy(scores_atac_to_rna / temperature, positive_indices)
    loss = 0.5 * (loss_rna + loss_atac)

    rows = torch.arange(positive_indices.numel(), device=positive_indices.device)
    positive = scores_rna_to_atac[rows, positive_indices]
    negative_mask = torch.ones_like(scores_rna_to_atac, dtype=torch.bool)
    negative_mask[rows, positive_indices] = False
    negatives = scores_rna_to_atac.masked_select(negative_mask)
    return loss, {
        "slot_loss": float(loss.detach().item()),
        "slot_rna_loss": float(loss_rna.detach().item()),
        "slot_atac_loss": float(loss_atac.detach().item()),
        "slot_matched_cosine": float(positive.detach().mean().item()),
        "slot_mismatched_cosine": (
            float(negatives.detach().mean().item()) if negatives.numel() else 0.0
        ),
    }
