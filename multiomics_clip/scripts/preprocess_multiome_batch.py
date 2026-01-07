#!/usr/bin/env python3
"""
Multi-Dataset Preprocessing Script for 10x Multiome Data

This script processes multiple 10x Multiome datasets with proper barcode prefixing
and consistent TF-IDF tokenization across datasets.

Usage:
    python multiomics_clip/scripts/preprocess_multiome_batch.py \
        --config configs/example_multiome_datasets.yaml \
        --gene_list scFoundation/model/OS_scRNA_gene_index.19264.tsv \
        --ccre_bed EpiAgent/data/cCRE.bed \
        --epiagent_df EpiAgent/data/cCRE_document_frequency.npy \
        --output_dir ./preprocessed_data_batch

Workflow:
    1. Load dataset configuration from YAML
    2. Preprocess each dataset separately (with barcode prefixes)
    3. Pair RNA-ATAC within each dataset
    4. Apply consistent TF-IDF using precomputed document frequencies
    5. Concatenate all datasets
    6. Split into train/val/test
"""

import argparse
import sys
import os
import yaml
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import scanpy as sc
from multiomics_clip.preprocessing import (
    preprocess_rna_multiome,
    preprocess_atac_multiome,
    pair_rna_atac_by_barcode,
    split_train_val_test
)


def load_config(config_path: str) -> Dict:
    """Load dataset configuration from YAML file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Found {len(config['datasets'])} datasets:")
    for ds in config['datasets']:
        print(f"  - {ds['name']}")

    return config


def preprocess_single_dataset(
    dataset_config: Dict,
    gene_list_path: str,
    ccre_bed_path: str,
    epiagent_df_path: str,
    output_dir: str,
    rna_params: Dict,
    atac_params: Dict
) -> Tuple[sc.AnnData, sc.AnnData, Dict]:
    """
    Preprocess a single dataset with barcode prefixing.

    Returns:
        (rna_paired, atac_paired, stats)
    """
    dataset_name = dataset_config['name']
    print("\n" + "=" * 80)
    print(f"Processing dataset: {dataset_name}")
    print("=" * 80)

    stats = {'name': dataset_name, 'timestamp': str(datetime.now())}

    # ==========================
    # Step 1: Preprocess RNA
    # ==========================
    print("\n[1/5] Preprocessing RNA...")
    rna_adata = preprocess_rna_multiome(
        h5_path=dataset_config['rna_h5'],
        gene_list_path=gene_list_path,
        min_genes=rna_params['min_genes'],
        max_genes=rna_params.get('max_genes'),
        min_cells=rna_params['min_cells_gene']
    )

    stats['rna_cells_after_qc'] = rna_adata.shape[0]
    stats['rna_genes'] = rna_adata.shape[1]

    # Add dataset prefix to barcodes
    print(f"  Adding dataset prefix: {dataset_name}_")
    original_barcodes = rna_adata.obs_names.tolist()
    rna_adata.obs_names = [f"{dataset_name}_{bc}" for bc in original_barcodes]

    # Add dataset metadata
    rna_adata.obs['dataset_id'] = dataset_name
    if 'batch_id' in dataset_config:
        rna_adata.obs['batch_id'] = dataset_config['batch_id']
    if 'description' in dataset_config:
        rna_adata.obs['description'] = dataset_config['description']

    # Save intermediate RNA
    rna_preprocessed_path = os.path.join(output_dir, f"{dataset_name}_rna_preprocessed.h5ad")
    rna_adata.write_h5ad(rna_preprocessed_path)
    print(f"  Saved to: {rna_preprocessed_path}")

    # ==========================
    # Step 2: Preprocess ATAC
    # ==========================
    print("\n[2/5] Preprocessing ATAC...")

    # Create dataset-specific ATAC directory
    atac_work_dir = os.path.join(output_dir, f"{dataset_name}_atac_preprocessing")
    os.makedirs(atac_work_dir, exist_ok=True)

    atac_adata = preprocess_atac_multiome(
        fragments_path=dataset_config['atac_fragments'],
        ccre_bed_path=ccre_bed_path,
        output_dir=atac_work_dir,
        min_fragments=atac_params['min_fragments'],
        min_cells_per_ccre=atac_params['min_cells_per_ccre'],
        ccre_df_path=epiagent_df_path
    )

    stats['atac_cells_after_qc'] = atac_adata.shape[0]
    stats['atac_ccres'] = atac_adata.shape[1]

    # Add dataset prefix to barcodes
    print(f"  Adding dataset prefix: {dataset_name}_")
    original_barcodes_atac = atac_adata.obs_names.tolist()
    atac_adata.obs_names = [f"{dataset_name}_{bc}" for bc in original_barcodes_atac]

    # Add dataset metadata
    atac_adata.obs['dataset_id'] = dataset_name
    if 'batch_id' in dataset_config:
        atac_adata.obs['batch_id'] = dataset_config['batch_id']
    if 'description' in dataset_config:
        atac_adata.obs['description'] = dataset_config['description']

    # Save intermediate ATAC
    atac_preprocessed_path = os.path.join(output_dir, f"{dataset_name}_atac_preprocessed.h5ad")
    atac_adata.write_h5ad(atac_preprocessed_path)
    print(f"  Saved to: {atac_preprocessed_path}")

    # ==========================
    # Step 3: Pair modalities
    # ==========================
    print("\n[3/5] Pairing RNA and ATAC by barcodes...")
    rna_paired, atac_paired, common_barcodes = pair_rna_atac_by_barcode(rna_adata, atac_adata)

    stats['paired_cells'] = len(common_barcodes)
    stats['pairing_efficiency'] = stats['paired_cells'] / min(stats['rna_cells_after_qc'], stats['atac_cells_after_qc'])

    print(f"  Paired cells: {stats['paired_cells']}")
    print(f"  Pairing efficiency: {stats['pairing_efficiency']:.2%}")

    # Save paired data
    rna_paired_path = os.path.join(output_dir, f"{dataset_name}_rna_paired.h5ad")
    atac_paired_path = os.path.join(output_dir, f"{dataset_name}_atac_paired.h5ad")

    rna_paired.write_h5ad(rna_paired_path)
    atac_paired.write_h5ad(atac_paired_path)

    print(f"  Saved RNA to: {rna_paired_path}")
    print(f"  Saved ATAC to: {atac_paired_path}")

    return rna_paired, atac_paired, stats


def concatenate_datasets(dataset_list: List[Tuple[sc.AnnData, sc.AnnData]]) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Concatenate multiple paired datasets.

    Args:
        dataset_list: List of (rna_paired, atac_paired) tuples

    Returns:
        (all_rna, all_atac)
    """
    print("\n" + "=" * 80)
    print("Concatenating all datasets...")
    print("=" * 80)

    rna_list = [ds[0] for ds in dataset_list]
    atac_list = [ds[1] for ds in dataset_list]

    print(f"  Concatenating {len(rna_list)} RNA datasets...")
    all_rna = sc.concat(rna_list, join='outer', index_unique=None)

    print(f"  Concatenating {len(atac_list)} ATAC datasets...")
    all_atac = sc.concat(atac_list, join='outer', index_unique=None)

    print(f"\n  Total cells: {all_rna.shape[0]}")
    print(f"  RNA genes: {all_rna.shape[1]}")
    print(f"  ATAC cCREs: {all_atac.shape[1]}")

    # Verify barcode matching
    assert all_rna.obs_names.tolist() == all_atac.obs_names.tolist(), \
        "Barcode mismatch after concatenation!"

    # Print dataset distribution
    print("\n  Dataset distribution:")
    dataset_counts = all_rna.obs['dataset_id'].value_counts()
    for dataset_id, count in dataset_counts.items():
        print(f"    {dataset_id}: {count} cells ({count/all_rna.shape[0]:.1%})")

    return all_rna, all_atac


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess multiple 10x Multiome datasets with consistent TF-IDF"
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file specifying datasets'
    )

    # Reference files
    parser.add_argument(
        '--gene_list',
        type=str,
        default='scFoundation/model/OS_scRNA_gene_index.19264.tsv',
        help='Path to scFoundation gene list'
    )
    parser.add_argument(
        '--ccre_bed',
        type=str,
        default='EpiAgent/data/cCRE.bed',
        help='Path to cCRE BED file'
    )
    parser.add_argument(
        '--epiagent_df',
        type=str,
        default='EpiAgent/data/cCRE_document_frequency.npy',
        help='Path to precomputed EpiAgent document frequency file'
    )

    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for preprocessed data'
    )

    # RNA QC parameters
    parser.add_argument('--min_genes', type=int, default=500,
                       help='Minimum genes per cell')
    parser.add_argument('--max_genes', type=int, default=None,
                       help='Maximum genes per cell (None = no limit)')
    parser.add_argument('--min_cells_gene', type=int, default=3,
                       help='Minimum cells per gene')

    # ATAC QC parameters
    parser.add_argument('--min_fragments', type=int, default=200,
                       help='Minimum fragments per cell')
    parser.add_argument('--min_cells_per_ccre', type=int, default=3,
                       help='Minimum cells per cCRE')

    # Splitting
    parser.add_argument('--train_frac', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--val_frac', type=float, default=0.1,
                       help='Fraction of data for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Options
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove intermediate ATAC preprocessing files')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Multi-Dataset Preprocessing for Multi-Omics CLIP")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Gene list: {args.gene_list}")
    print(f"cCRE BED: {args.ccre_bed}")
    print(f"EpiAgent DF: {args.epiagent_df}")

    # Load configuration
    config = load_config(args.config)

    # Copy config to output directory
    config_output_path = os.path.join(args.output_dir, 'datasets_config.yaml')
    with open(config_output_path, 'w') as f:
        yaml.dump(config, f)
    print(f"\nSaved config to: {config_output_path}")

    # Prepare parameter dicts
    rna_params = {
        'min_genes': args.min_genes,
        'max_genes': args.max_genes,
        'min_cells_gene': args.min_cells_gene
    }

    atac_params = {
        'min_fragments': args.min_fragments,
        'min_cells_per_ccre': args.min_cells_per_ccre
    }

    # Process each dataset
    processed_datasets = []
    all_stats = []

    for dataset_config in config['datasets']:
        try:
            rna_paired, atac_paired, stats = preprocess_single_dataset(
                dataset_config=dataset_config,
                gene_list_path=args.gene_list,
                ccre_bed_path=args.ccre_bed,
                epiagent_df_path=args.epiagent_df,
                output_dir=args.output_dir,
                rna_params=rna_params,
                atac_params=atac_params
            )

            processed_datasets.append((rna_paired, atac_paired))
            all_stats.append(stats)

        except Exception as e:
            print(f"\nERROR processing dataset {dataset_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not processed_datasets:
        print("\nERROR: No datasets were successfully processed!")
        return

    # Concatenate all datasets
    all_rna, all_atac = concatenate_datasets(processed_datasets)

    # Save concatenated data
    print("\nSaving concatenated paired data...")
    all_rna.write_h5ad(os.path.join(args.output_dir, 'all_rna_paired.h5ad'))
    all_atac.write_h5ad(os.path.join(args.output_dir, 'all_atac_paired.h5ad'))

    # Split into train/val/test
    print("\n" + "=" * 80)
    print("Splitting into train/val/test...")
    print("=" * 80)

    train_rna, train_atac, val_rna, val_atac, test_rna, test_atac = split_train_val_test(
        rna_adata=all_rna,
        atac_adata=all_atac,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        random_seed=args.random_seed
    )

    print(f"  Training: {train_rna.shape[0]} cells")
    print(f"  Validation: {val_rna.shape[0]} cells")
    print(f"  Test: {test_rna.shape[0]} cells")

    # Print per-dataset distribution in each split
    print("\n  Dataset distribution per split:")
    for split_name, split_rna in [('train', train_rna), ('val', val_rna), ('test', test_rna)]:
        print(f"\n    {split_name.capitalize()}:")
        dataset_counts = split_rna.obs['dataset_id'].value_counts()
        for dataset_id, count in dataset_counts.items():
            print(f"      {dataset_id}: {count} cells ({count/split_rna.shape[0]:.1%})")

    # Save splits
    print("\nSaving train/val/test splits...")
    train_rna.write_h5ad(os.path.join(args.output_dir, 'train_rna.h5ad'))
    train_atac.write_h5ad(os.path.join(args.output_dir, 'train_atac.h5ad'))
    val_rna.write_h5ad(os.path.join(args.output_dir, 'val_rna.h5ad'))
    val_atac.write_h5ad(os.path.join(args.output_dir, 'val_atac.h5ad'))
    test_rna.write_h5ad(os.path.join(args.output_dir, 'test_rna.h5ad'))
    test_atac.write_h5ad(os.path.join(args.output_dir, 'test_atac.h5ad'))

    # Save preprocessing statistics
    print("\nSaving preprocessing statistics...")
    stats_summary = {
        'datasets': all_stats,
        'total_cells': all_rna.shape[0],
        'train_cells': train_rna.shape[0],
        'val_cells': val_rna.shape[0],
        'test_cells': test_rna.shape[0],
        'rna_genes': all_rna.shape[1],
        'atac_ccres': all_atac.shape[1],
        'parameters': {
            'rna_qc': rna_params,
            'atac_qc': atac_params,
            'train_frac': args.train_frac,
            'val_frac': args.val_frac,
            'random_seed': args.random_seed
        }
    }

    with open(os.path.join(args.output_dir, 'preprocessing_stats.json'), 'w') as f:
        json.dump(stats_summary, f, indent=2)

    # Cleanup intermediate files if requested
    if args.cleanup:
        print("\nCleaning up intermediate ATAC preprocessing files...")
        for dataset_config in config['datasets']:
            atac_work_dir = os.path.join(args.output_dir, f"{dataset_config['name']}_atac_preprocessing")
            if os.path.exists(atac_work_dir):
                import shutil
                shutil.rmtree(atac_work_dir)
                print(f"  Removed: {atac_work_dir}")

    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nGenerated files:")
    print("  Per-dataset files:")
    for dataset_config in config['datasets']:
        dataset_name = dataset_config['name']
        print(f"    - {dataset_name}_rna_preprocessed.h5ad")
        print(f"    - {dataset_name}_atac_preprocessed.h5ad")
        print(f"    - {dataset_name}_rna_paired.h5ad")
        print(f"    - {dataset_name}_atac_paired.h5ad")
    print("\n  Combined files:")
    print("    - all_rna_paired.h5ad")
    print("    - all_atac_paired.h5ad")
    print("\n  Train/Val/Test splits:")
    print("    - train_rna.h5ad, train_atac.h5ad")
    print("    - val_rna.h5ad, val_atac.h5ad")
    print("    - test_rna.h5ad, test_atac.h5ad")
    print("\n  Metadata:")
    print("    - datasets_config.yaml")
    print("    - preprocessing_stats.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
