"""Clustering and label transfer metrics"""

import torch
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def knn_label_transfer(
    train_features: torch.Tensor,
    train_labels: np.ndarray,
    test_features: torch.Tensor,
    k: int = 10
) -> np.ndarray:
    """
    Transfer labels using k-nearest neighbors

    Args:
        train_features: Training features [n_train, dim]
        train_labels: Training labels [n_train]
        test_features: Test features [n_test, dim]
        k: Number of neighbors

    Returns:
        Predicted labels [n_test]
    """
    # Convert to numpy
    if torch.is_tensor(train_features):
        train_features = train_features.cpu().numpy()
    if torch.is_tensor(test_features):
        test_features = test_features.cpu().numpy()

    # Train kNN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_features, train_labels)

    # Predict
    pred_labels = knn.predict(test_features)

    return pred_labels


def compute_clustering_metrics(
    features: torch.Tensor,
    true_labels: np.ndarray,
    pred_labels: Optional[np.ndarray] = None,
    k: int = 10
) -> Dict[str, float]:
    """
    Compute clustering metrics (ARI, NMI, F1)

    Args:
        features: Feature embeddings [n_samples, dim]
        true_labels: Ground truth labels [n_samples]
        pred_labels: Predicted labels [n_samples] (if None, use kNN on features)
        k: Number of neighbors for kNN (if pred_labels is None)

    Returns:
        Dictionary with clustering metrics
    """
    if pred_labels is None:
        # Use kNN with self as reference (leave-one-out style)
        pred_labels = knn_label_transfer(features, true_labels, features, k=k)

    # Compute metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    # F1 score (macro-averaged for multi-class)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')

    metrics = {
        'ari': ari,
        'nmi': nmi,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

    return metrics


def compute_cross_modal_label_transfer(
    rna_features: torch.Tensor,
    atac_features: torch.Tensor,
    cell_type_labels: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate cross-modal label transfer using kNN in joint embedding space

    Args:
        rna_features: RNA embeddings [n_samples, dim]
        atac_features: ATAC embeddings [n_samples, dim]
        cell_type_labels: Cell type labels [n_samples]
        k: Number of neighbors for kNN

    Returns:
        Dictionary with label transfer metrics
    """
    # Split data into train (RNA) and test (ATAC)
    # In practice, you might want to do proper cross-validation
    n_samples = len(cell_type_labels)
    split_idx = n_samples // 2

    # RNA to ATAC transfer
    rna_to_atac_pred = knn_label_transfer(
        rna_features[:split_idx],
        cell_type_labels[:split_idx],
        atac_features[split_idx:],
        k=k
    )

    rna_to_atac_metrics = compute_clustering_metrics(
        atac_features[split_idx:],
        cell_type_labels[split_idx:],
        rna_to_atac_pred
    )

    # ATAC to RNA transfer
    atac_to_rna_pred = knn_label_transfer(
        atac_features[:split_idx],
        cell_type_labels[:split_idx],
        rna_features[split_idx:],
        k=k
    )

    atac_to_rna_metrics = compute_clustering_metrics(
        rna_features[split_idx:],
        cell_type_labels[split_idx:],
        atac_to_rna_pred
    )

    # Combine metrics with prefixes
    all_metrics = {}
    for key, value in rna_to_atac_metrics.items():
        all_metrics[f'rna_to_atac_{key}'] = value
    for key, value in atac_to_rna_metrics.items():
        all_metrics[f'atac_to_rna_{key}'] = value

    # Average metrics
    for key in ['ari', 'nmi', 'f1_macro', 'f1_weighted']:
        all_metrics[f'avg_{key}'] = (
            rna_to_atac_metrics[key] + atac_to_rna_metrics[key]
        ) / 2

    return all_metrics


def compute_joint_clustering_metrics(
    rna_features: torch.Tensor,
    atac_features: torch.Tensor,
    cell_type_labels: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate clustering in joint RNA-ATAC embedding space

    Args:
        rna_features: RNA embeddings [n_samples, dim]
        atac_features: ATAC embeddings [n_samples, dim]
        cell_type_labels: Cell type labels [n_samples]
        k: Number of neighbors for kNN

    Returns:
        Dictionary with clustering metrics
    """
    # Concatenate RNA and ATAC features
    joint_features = torch.cat([rna_features, atac_features], dim=1)

    # Compute clustering metrics
    metrics = compute_clustering_metrics(joint_features, cell_type_labels, k=k)

    # Add prefix
    joint_metrics = {f'joint_{key}': value for key, value in metrics.items()}

    return joint_metrics
