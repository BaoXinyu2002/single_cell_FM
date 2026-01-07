"""Retrieval metrics for cross-modal evaluation"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_recall_at_k(
    similarity_matrix: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    """
    Compute Recall@K for cross-modal retrieval

    Args:
        similarity_matrix: Similarity matrix [n_samples, n_samples]
                          similarity_matrix[i, j] is similarity between query i and candidate j
        k_values: List of K values for Recall@K

    Returns:
        Dictionary with Recall@K for each k
    """
    n_samples = similarity_matrix.shape[0]

    # Get top-k indices for each query
    _, top_k_indices = torch.topk(similarity_matrix, max(k_values), dim=1)

    # Ground truth: diagonal (i-th query matches i-th candidate)
    ground_truth = torch.arange(n_samples, device=similarity_matrix.device)

    recalls = {}
    for k in k_values:
        # Check if ground truth is in top-k
        top_k = top_k_indices[:, :k]
        matches = (top_k == ground_truth.unsqueeze(1)).any(dim=1)
        recall = matches.float().mean().item()
        recalls[f'recall@{k}'] = recall

    return recalls


def compute_mean_rank(similarity_matrix: torch.Tensor) -> float:
    """
    Compute Mean Rank (MR) for retrieval

    Args:
        similarity_matrix: Similarity matrix [n_samples, n_samples]

    Returns:
        Mean rank
    """
    n_samples = similarity_matrix.shape[0]

    # Get ranking for each query
    _, ranking = torch.sort(similarity_matrix, dim=1, descending=True)

    # Ground truth positions
    ground_truth = torch.arange(n_samples, device=similarity_matrix.device)

    # Find rank of ground truth for each query
    ranks = []
    for i in range(n_samples):
        rank = (ranking[i] == ground_truth[i]).nonzero(as_tuple=True)[0].item()
        ranks.append(rank + 1)  # 1-indexed rank

    mean_rank = np.mean(ranks)
    return mean_rank


def compute_mean_reciprocal_rank(similarity_matrix: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for retrieval

    Args:
        similarity_matrix: Similarity matrix [n_samples, n_samples]

    Returns:
        Mean reciprocal rank
    """
    n_samples = similarity_matrix.shape[0]

    # Get ranking for each query
    _, ranking = torch.sort(similarity_matrix, dim=1, descending=True)

    # Ground truth positions
    ground_truth = torch.arange(n_samples, device=similarity_matrix.device)

    # Find reciprocal rank of ground truth for each query
    reciprocal_ranks = []
    for i in range(n_samples):
        rank = (ranking[i] == ground_truth[i]).nonzero(as_tuple=True)[0].item()
        reciprocal_ranks.append(1.0 / (rank + 1))  # 1-indexed rank

    mrr = np.mean(reciprocal_ranks)
    return mrr


def compute_retrieval_metrics(
    rna_features: torch.Tensor,
    atac_features: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    """
    Compute all retrieval metrics for both RNA->ATAC and ATAC->RNA

    Args:
        rna_features: RNA embeddings [n_samples, dim]
        atac_features: ATAC embeddings [n_samples, dim]
        k_values: List of K values for Recall@K

    Returns:
        Dictionary with all retrieval metrics
    """
    # Normalize features
    rna_features = torch.nn.functional.normalize(rna_features, dim=-1)
    atac_features = torch.nn.functional.normalize(atac_features, dim=-1)

    # Compute similarity matrices
    # RNA -> ATAC: use RNA as query, ATAC as candidates
    sim_rna_to_atac = torch.matmul(rna_features, atac_features.T)

    # ATAC -> RNA: use ATAC as query, RNA as candidates
    sim_atac_to_rna = torch.matmul(atac_features, rna_features.T)

    # Compute metrics for RNA -> ATAC
    metrics_rna_to_atac = compute_recall_at_k(sim_rna_to_atac, k_values)
    metrics_rna_to_atac['mean_rank'] = compute_mean_rank(sim_rna_to_atac)
    metrics_rna_to_atac['mrr'] = compute_mean_reciprocal_rank(sim_rna_to_atac)

    # Compute metrics for ATAC -> RNA
    metrics_atac_to_rna = compute_recall_at_k(sim_atac_to_rna, k_values)
    metrics_atac_to_rna['mean_rank'] = compute_mean_rank(sim_atac_to_rna)
    metrics_atac_to_rna['mrr'] = compute_mean_reciprocal_rank(sim_atac_to_rna)

    # Combine metrics with prefixes
    all_metrics = {}
    for key, value in metrics_rna_to_atac.items():
        all_metrics[f'rna_to_atac_{key}'] = value
    for key, value in metrics_atac_to_rna.items():
        all_metrics[f'atac_to_rna_{key}'] = value

    # Compute average metrics
    for k in k_values:
        all_metrics[f'avg_recall@{k}'] = (
            metrics_rna_to_atac[f'recall@{k}'] +
            metrics_atac_to_rna[f'recall@{k}']
        ) / 2

    all_metrics['avg_mean_rank'] = (
        metrics_rna_to_atac['mean_rank'] +
        metrics_atac_to_rna['mean_rank']
    ) / 2

    all_metrics['avg_mrr'] = (
        metrics_rna_to_atac['mrr'] +
        metrics_atac_to_rna['mrr']
    ) / 2

    return all_metrics


def compute_top_k_accuracy(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    k: int = 1
) -> float:
    """
    Compute top-K accuracy for retrieval

    Args:
        query_features: Query embeddings [n_samples, dim]
        candidate_features: Candidate embeddings [n_samples, dim]
        k: K for top-K accuracy

    Returns:
        Top-K accuracy
    """
    # Normalize
    query_features = torch.nn.functional.normalize(query_features, dim=-1)
    candidate_features = torch.nn.functional.normalize(candidate_features, dim=-1)

    # Compute similarity
    similarity = torch.matmul(query_features, candidate_features.T)

    # Get top-k
    _, top_k_indices = torch.topk(similarity, k, dim=1)

    # Ground truth
    n_samples = query_features.shape[0]
    ground_truth = torch.arange(n_samples, device=query_features.device)

    # Check if ground truth in top-k
    matches = (top_k_indices == ground_truth.unsqueeze(1)).any(dim=1)
    accuracy = matches.float().mean().item()

    return accuracy
