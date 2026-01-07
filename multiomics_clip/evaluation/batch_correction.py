"""Batch correction metrics for evaluating batch mixing"""

import torch
import numpy as np
from typing import Dict, Optional
from scipy.spatial.distance import cdist


def compute_silhouette_score_batch(
    features: torch.Tensor,
    batch_labels: np.ndarray
) -> float:
    """
    Compute silhouette score for batch mixing (lower is better)

    A low silhouette score indicates good batch mixing.

    Args:
        features: Feature embeddings [n_samples, dim]
        batch_labels: Batch labels [n_samples]

    Returns:
        Silhouette score
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("Warning: scikit-learn not available. Silhouette score not computed.")
        return 0.0

    if torch.is_tensor(features):
        features = features.cpu().numpy()

    # Compute silhouette score
    # Metric='cosine' for embedding spaces
    score = silhouette_score(features, batch_labels, metric='cosine')

    return score


def compute_kbet(
    features: torch.Tensor,
    batch_labels: np.ndarray,
    k: int = 25,
    n_subsamples: int = 100,
    subsample_size: int = 100
) -> Dict[str, float]:
    """
    Compute k-nearest neighbor Batch Effect Test (kBET)

    kBET quantifies batch mixing by testing if the batch composition
    in k-nearest neighbors matches the global batch composition.

    Args:
        features: Feature embeddings [n_samples, dim]
        batch_labels: Batch labels [n_samples]
        k: Number of neighbors
        n_subsamples: Number of subsamples to test
        subsample_size: Size of each subsample

    Returns:
        Dictionary with kBET metrics
    """
    if torch.is_tensor(features):
        features = features.cpu().numpy()

    n_samples = features.shape[0]
    batches = np.unique(batch_labels)
    n_batches = len(batches)

    # Global batch composition
    global_composition = np.array([
        (batch_labels == batch).sum() / n_samples
        for batch in batches
    ])

    # Compute pairwise distances
    distances = cdist(features, features, metric='cosine')

    # Test on subsamples
    rejection_rates = []

    for _ in range(n_subsamples):
        # Sample random cells
        sample_idx = np.random.choice(n_samples, subsample_size, replace=False)

        for idx in sample_idx:
            # Get k-nearest neighbors
            neighbor_distances = distances[idx]
            neighbor_idx = np.argsort(neighbor_distances)[1:k+1]  # Exclude self

            # Batch composition in neighborhood
            neighbor_batches = batch_labels[neighbor_idx]
            observed_composition = np.array([
                (neighbor_batches == batch).sum() / k
                for batch in batches
            ])

            # Chi-square test for batch composition
            # Expected counts vs observed counts
            expected = global_composition * k
            observed = observed_composition * k

            # Chi-square statistic
            chi_square = np.sum((observed - expected) ** 2 / (expected + 1e-8))

            # Degrees of freedom
            df = n_batches - 1

            # Simple threshold test (chi-square > critical value)
            # For df=1, critical value at p=0.05 is ~3.84
            # For df=2, critical value at p=0.05 is ~5.99
            critical_value = 3.84 + (df - 1) * 2.15  # Approximate

            if chi_square > critical_value:
                rejection_rates.append(1)
            else:
                rejection_rates.append(0)

    # Acceptance rate (good batch mixing)
    acceptance_rate = 1.0 - np.mean(rejection_rates)

    metrics = {
        'kbet_acceptance_rate': acceptance_rate,
        'kbet_rejection_rate': np.mean(rejection_rates)
    }

    return metrics


def compute_ilisi(
    features: torch.Tensor,
    batch_labels: np.ndarray,
    k: int = 90
) -> float:
    """
    Compute integration Local Inverse Simpson's Index (iLISI)

    iLISI measures batch mixing. Higher values indicate better mixing.
    Maximum value is the number of batches.

    Args:
        features: Feature embeddings [n_samples, dim]
        batch_labels: Batch labels [n_samples]
        k: Number of neighbors

    Returns:
        iLISI score
    """
    if torch.is_tensor(features):
        features = features.cpu().numpy()

    n_samples = features.shape[0]
    batches = np.unique(batch_labels)

    # Compute pairwise distances
    distances = cdist(features, features, metric='cosine')

    ilisi_scores = []

    for i in range(n_samples):
        # Get k-nearest neighbors
        neighbor_distances = distances[i]
        neighbor_idx = np.argsort(neighbor_distances)[1:k+1]  # Exclude self

        # Batch composition in neighborhood
        neighbor_batches = batch_labels[neighbor_idx]

        # Compute Simpson's index
        simpson = 0.0
        for batch in batches:
            p = (neighbor_batches == batch).sum() / k
            simpson += p ** 2

        # Inverse Simpson's index
        if simpson > 0:
            ilisi = 1.0 / simpson
        else:
            ilisi = 1.0

        ilisi_scores.append(ilisi)

    # Average iLISI
    mean_ilisi = np.mean(ilisi_scores)

    return mean_ilisi


def compute_batch_metrics(
    features: torch.Tensor,
    batch_labels: np.ndarray,
    k: int = 25
) -> Dict[str, float]:
    """
    Compute all batch correction metrics

    Args:
        features: Feature embeddings [n_samples, dim]
        batch_labels: Batch labels [n_samples]
        k: Number of neighbors for kBET and iLISI

    Returns:
        Dictionary with all batch metrics
    """
    metrics = {}

    # Silhouette score (lower is better for batch mixing)
    metrics['silhouette_batch'] = compute_silhouette_score_batch(features, batch_labels)

    # kBET (higher acceptance rate is better)
    kbet_metrics = compute_kbet(features, batch_labels, k=k)
    metrics.update(kbet_metrics)

    # iLISI (higher is better, max = number of batches)
    metrics['ilisi'] = compute_ilisi(features, batch_labels, k=k)

    return metrics


def compute_cross_modal_batch_metrics(
    rna_features: torch.Tensor,
    atac_features: torch.Tensor,
    batch_labels: np.ndarray,
    k: int = 25
) -> Dict[str, float]:
    """
    Compute batch correction metrics for both modalities

    Args:
        rna_features: RNA embeddings [n_samples, dim]
        atac_features: ATAC embeddings [n_samples, dim]
        batch_labels: Batch labels [n_samples]
        k: Number of neighbors

    Returns:
        Dictionary with batch metrics for both modalities
    """
    # RNA metrics
    rna_metrics = compute_batch_metrics(rna_features, batch_labels, k=k)

    # ATAC metrics
    atac_metrics = compute_batch_metrics(atac_features, batch_labels, k=k)

    # Joint metrics
    joint_features = torch.cat([rna_features, atac_features], dim=1)
    joint_metrics = compute_batch_metrics(joint_features, batch_labels, k=k)

    # Combine with prefixes
    all_metrics = {}
    for key, value in rna_metrics.items():
        all_metrics[f'rna_{key}'] = value
    for key, value in atac_metrics.items():
        all_metrics[f'atac_{key}'] = value
    for key, value in joint_metrics.items():
        all_metrics[f'joint_{key}'] = value

    return all_metrics
