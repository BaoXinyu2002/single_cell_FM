#!/usr/bin/env python3
"""
Evaluation script for Multi-Omics CLIP

This script evaluates a trained Multi-Omics CLIP model on test data.
"""

import argparse
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import scanpy as sc
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from multiomics_clip import MultiOmicsCLIP, CLIPConfig, PairedMultiOmicsDataset, collate_fn
from multiomics_clip.preprocessing import load_gene_list
from multiomics_clip.evaluation import (
    compute_retrieval_metrics,
    compute_clustering_metrics,
    compute_batch_metrics,
    compute_cross_modal_label_transfer
)
from multiomics_clip.utils.checkpoint import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-Omics CLIP model"
    )

    # Model checkpoint
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    # Data paths
    parser.add_argument(
        '--test_rna',
        type=str,
        required=True,
        help='Path to test RNA h5ad file'
    )
    parser.add_argument(
        '--test_atac',
        type=str,
        required=True,
        help='Path to test ATAC h5ad file'
    )
    parser.add_argument(
        '--gene_list_path',
        type=str,
        default='scFoundation/model/OS_scRNA_gene_index.19264.tsv',
        help='Path to scFoundation gene list'
    )

    # Evaluation parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )

    # Optional metadata for evaluation
    parser.add_argument(
        '--celltype_column',
        type=str,
        help='Column name for cell type labels (for clustering metrics)'
    )
    parser.add_argument(
        '--batch_column',
        type=str,
        help='Column name for batch labels (for batch correction metrics)'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Omics CLIP: Evaluation")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config from checkpoint
    print(f"\nLoading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config_dict = checkpoint.get('config', {})
    config = CLIPConfig(**config_dict)

    # Load model
    model = MultiOmicsCLIP(config)
    model = load_model_from_checkpoint(model, args.checkpoint, device=args.device)
    model.eval()

    print("Model loaded successfully!")

    # Load gene list
    print("\nLoading gene list...")
    gene_list = load_gene_list(args.gene_list_path)

    # Load test data
    print(f"\nLoading test data...")
    print(f"  RNA:  {args.test_rna}")
    print(f"  ATAC: {args.test_atac}")

    test_rna = sc.read_h5ad(args.test_rna)
    test_atac = sc.read_h5ad(args.test_atac)

    print(f"  Test samples: {len(test_rna)}")

    # Create test dataset
    test_dataset = PairedMultiOmicsDataset(
        rna_adata=test_rna,
        atac_adata=test_atac,
        gene_list=gene_list,
        target_resolution=config.target_resolution,
        max_atac_length=config.max_atac_length,
        random_truncate=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )

    # Extract features
    print("\nExtracting features...")
    rna_features_list = []
    atac_features_list = []

    with torch.no_grad():
        for batch in test_dataloader:
            rna_data, rna_gene_ids, atac_input_ids = [x.to(args.device) for x in batch]

            # Encode
            rna_features = model.encode_rna(rna_data, rna_gene_ids)
            atac_features = model.encode_atac(atac_input_ids)

            rna_features_list.append(rna_features.cpu())
            atac_features_list.append(atac_features.cpu())

    rna_features = torch.cat(rna_features_list, dim=0)
    atac_features = torch.cat(atac_features_list, dim=0)

    print(f"Extracted features: RNA {rna_features.shape}, ATAC {atac_features.shape}")

    # Compute retrieval metrics
    print("\n" + "=" * 80)
    print("Computing Retrieval Metrics...")
    print("=" * 80)

    retrieval_metrics = compute_retrieval_metrics(
        rna_features,
        atac_features,
        k_values=config.retrieval_k_values
    )

    for key, value in retrieval_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Compute clustering metrics if cell type labels available
    if args.celltype_column and args.celltype_column in test_rna.obs.columns:
        print("\n" + "=" * 80)
        print("Computing Clustering Metrics...")
        print("=" * 80)

        celltype_labels = test_rna.obs[args.celltype_column].values

        # Label transfer
        transfer_metrics = compute_cross_modal_label_transfer(
            rna_features,
            atac_features,
            celltype_labels,
            k=10
        )

        for key, value in transfer_metrics.items():
            print(f"  {key}: {value:.4f}")

    else:
        print("\nSkipping clustering metrics (no cell type labels provided)")
        transfer_metrics = {}

    # Compute batch correction metrics if batch labels available
    if args.batch_column and args.batch_column in test_rna.obs.columns:
        print("\n" + "=" * 80)
        print("Computing Batch Correction Metrics...")
        print("=" * 80)

        batch_labels = test_rna.obs[args.batch_column].values

        batch_metrics = compute_batch_metrics(
            rna_features,
            batch_labels,
            k=25
        )

        for key, value in batch_metrics.items():
            print(f"  {key}: {value:.4f}")

    else:
        print("\nSkipping batch correction metrics (no batch labels provided)")
        batch_metrics = {}

    # Save results
    all_metrics = {
        **retrieval_metrics,
        **transfer_metrics,
        **batch_metrics
    }

    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved results to: {results_path}")

    # Save features
    features_path = os.path.join(args.output_dir, 'features.npz')
    np.savez(
        features_path,
        rna_features=rna_features.numpy(),
        atac_features=atac_features.numpy()
    )

    print(f"Saved features to: {features_path}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
