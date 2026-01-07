#!/usr/bin/env python3
"""
Visualization Script for Preprocessed Multi-Omics Data

This script generates comprehensive visualizations of preprocessed RNA and ATAC data:
- QC metrics (cell counts, distributions)
- Pairing statistics (barcode overlap, efficiency)
- UMAP/t-SNE embeddings
- TF-IDF analysis (top cCREs, score distributions)

Usage:
    python multiomics_clip/scripts/visualize_preprocessed.py \
        --input_dir ./preprocessed_data \
        --output_dir ./visualizations \
        --splits train,val,test
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize preprocessed multi-omics data")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing preprocessed h5ad files')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--splits', type=str, default='train,val,test',
                       help='Comma-separated list of splits to visualize')
    parser.add_argument('--umap', action='store_true', default=True,
                       help='Compute UMAP embeddings')
    parser.add_argument('--n_neighbors', type=int, default=30,
                       help='Number of neighbors for UMAP')
    parser.add_argument('--min_dist', type=float, default=0.3,
                       help='Min distance for UMAP')
    parser.add_argument('--format', type=str, default='png,html',
                       help='Output formats (png, html, pdf)')
    return parser.parse_args()


def load_data(input_dir: str, splits: List[str]) -> Dict[str, Dict[str, sc.AnnData]]:
    """Load preprocessed RNA and ATAC data for specified splits."""
    print("Loading preprocessed data...")
    data = {}

    for split in splits:
        rna_path = os.path.join(input_dir, f'{split}_rna.h5ad')
        atac_path = os.path.join(input_dir, f'{split}_atac.h5ad')

        if not os.path.exists(rna_path) or not os.path.exists(atac_path):
            print(f"Warning: {split} split not found, skipping...")
            continue

        print(f"  Loading {split} split...")
        data[split] = {
            'rna': sc.read_h5ad(rna_path),
            'atac': sc.read_h5ad(atac_path)
        }
        print(f"    RNA: {data[split]['rna'].shape}")
        print(f"    ATAC: {data[split]['atac'].shape}")

    # Also load full paired data if available
    full_rna_path = os.path.join(input_dir, 'rna_paired.h5ad')
    full_atac_path = os.path.join(input_dir, 'atac_paired.h5ad')

    if os.path.exists(full_rna_path) and os.path.exists(full_atac_path):
        print("  Loading full paired data...")
        data['full'] = {
            'rna': sc.read_h5ad(full_rna_path),
            'atac': sc.read_h5ad(full_atac_path)
        }
        print(f"    RNA: {data['full']['rna'].shape}")
        print(f"    ATAC: {data['full']['atac'].shape}")

    return data


def compute_qc_metrics(data: Dict[str, Dict[str, sc.AnnData]]) -> Dict:
    """Compute QC metrics for RNA and ATAC data."""
    print("\nComputing QC metrics...")
    stats = {}

    for split, modalities in data.items():
        stats[split] = {'rna': {}, 'atac': {}}

        # RNA metrics
        rna = modalities['rna']
        stats[split]['rna']['n_cells'] = rna.shape[0]
        stats[split]['rna']['n_genes'] = rna.shape[1]
        stats[split]['rna']['n_genes_detected'] = (rna.X > 0).sum(axis=1)
        stats[split]['rna']['total_counts'] = rna.X.sum(axis=1)

        if 'is_padded' in rna.var.columns:
            n_padded = rna.var['is_padded'].sum()
            stats[split]['rna']['n_padded_genes'] = n_padded
            stats[split]['rna']['frac_padded'] = n_padded / rna.shape[1]

        # ATAC metrics
        atac = modalities['atac']
        stats[split]['atac']['n_cells'] = atac.shape[0]
        stats[split]['atac']['n_ccres'] = atac.shape[1]
        stats[split]['atac']['n_ccres_accessible'] = (atac.X > 0).sum(axis=1)
        stats[split]['atac']['total_accessibility'] = atac.X.sum(axis=1)

        # Cell sentence lengths
        if 'cell_sentences' in atac.obs.columns:
            sentence_lengths = atac.obs['cell_sentences'].apply(
                lambda x: len(json.loads(x)) if isinstance(x, str) else 0
            )
            stats[split]['atac']['cell_sentence_length'] = sentence_lengths

        # Dataset ID distribution (if available)
        if 'dataset_id' in rna.obs.columns:
            stats[split]['dataset_distribution'] = rna.obs['dataset_id'].value_counts().to_dict()

    return stats


def plot_qc_flowchart(stats: Dict, output_dir: str):
    """Plot cell count flowchart showing QC filtering steps."""
    print("Plotting QC flowchart...")

    splits = [s for s in ['train', 'val', 'test'] if s in stats]
    cell_counts = {split: stats[split]['rna']['n_cells'] for split in splits}

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(splits, [cell_counts[s] for s in splits], color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_ylabel('Number of Cells', fontsize=12)
    ax.set_xlabel('Split', fontsize=12)
    ax.set_title('Cell Count Distribution Across Splits', fontsize=14, fontweight='bold')

    # Add count labels on bars
    for bar, split in zip(bars, splits):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11)

    # Add total
    total = sum(cell_counts.values())
    ax.text(0.5, 0.95, f'Total: {total:,} cells',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qc_cell_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_qc_distributions(stats: Dict, data: Dict, output_dir: str):
    """Plot distributions of QC metrics."""
    print("Plotting QC metric distributions...")

    splits = [s for s in ['train', 'val', 'test'] if s in data]

    # RNA metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RNA QC Metrics', fontsize=16, fontweight='bold')

    for split in splits:
        rna = data[split]['rna']

        # n_genes_detected
        n_genes = np.asarray((rna.X > 0).sum(axis=1)).flatten()
        axes[0, 0].hist(n_genes, bins=50, alpha=0.6, label=split)

        # total_counts
        total_counts = np.asarray(rna.X.sum(axis=1)).flatten()
        axes[0, 1].hist(np.log10(total_counts + 1), bins=50, alpha=0.6, label=split)

        # Sparsity
        sparsity = 1 - (n_genes / rna.shape[1])
        axes[1, 0].hist(sparsity, bins=50, alpha=0.6, label=split)

        # Gene expression distribution (mean across cells)
        mean_expr = np.asarray(rna.X.mean(axis=0)).flatten()
        axes[1, 1].hist(mean_expr[mean_expr > 0], bins=50, alpha=0.6, label=split, log=True)

    axes[0, 0].set_xlabel('Genes Detected per Cell')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()

    axes[0, 1].set_xlabel('log10(Total Counts + 1)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    axes[1, 0].set_xlabel('Sparsity (Fraction Zeros)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    axes[1, 1].set_xlabel('Mean Expression (non-zero genes)')
    axes[1, 1].set_ylabel('Frequency (log scale)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qc_rna_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ATAC metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ATAC QC Metrics', fontsize=16, fontweight='bold')

    for split in splits:
        atac = data[split]['atac']

        # n_ccres_accessible
        n_ccres = np.asarray((atac.X > 0).sum(axis=1)).flatten()
        axes[0, 0].hist(n_ccres, bins=50, alpha=0.6, label=split)

        # total_accessibility (TF-IDF sum)
        total_acc = np.asarray(atac.X.sum(axis=1)).flatten()
        axes[0, 1].hist(total_acc, bins=50, alpha=0.6, label=split)

        # Cell sentence length
        if 'cell_sentences' in atac.obs.columns:
            lengths = atac.obs['cell_sentences'].apply(
                lambda x: len(json.loads(x)) if isinstance(x, str) else 0
            )
            axes[1, 0].hist(lengths, bins=50, alpha=0.6, label=split)

        # cCRE accessibility frequency
        cCRE_freq = np.asarray((atac.X > 0).sum(axis=0)).flatten()
        axes[1, 1].hist(np.log10(cCRE_freq + 1), bins=50, alpha=0.6, label=split, log=True)

    axes[0, 0].set_xlabel('Accessible cCREs per Cell')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()

    axes[0, 1].set_xlabel('Total TF-IDF Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    axes[1, 0].set_xlabel('Cell Sentence Length')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    axes[1, 1].set_xlabel('log10(Cells with cCRE + 1)')
    axes[1, 1].set_ylabel('Frequency (log scale)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qc_atac_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pairing_stats(input_dir: str, output_dir: str):
    """Plot pairing statistics if unpaired data is available."""
    print("Plotting pairing statistics...")

    # Try to load unpaired data
    rna_unpaired_path = os.path.join(input_dir, 'rna_preprocessed.h5ad')
    atac_unpaired_path = os.path.join(input_dir, 'atac_preprocessed.h5ad')
    rna_paired_path = os.path.join(input_dir, 'rna_paired.h5ad')
    atac_paired_path = os.path.join(input_dir, 'atac_paired.h5ad')

    if not all(os.path.exists(p) for p in [rna_unpaired_path, atac_unpaired_path, rna_paired_path]):
        print("  Unpaired data not found, skipping pairing statistics...")
        return

    rna_unpaired = sc.read_h5ad(rna_unpaired_path)
    atac_unpaired = sc.read_h5ad(atac_unpaired_path)
    rna_paired = sc.read_h5ad(rna_paired_path)

    # Barcode overlap
    rna_barcodes = set(rna_unpaired.obs_names)
    atac_barcodes = set(atac_unpaired.obs_names)
    paired_barcodes = set(rna_paired.obs_names)

    # Venn diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Barcode overlap
    venn2([rna_barcodes, atac_barcodes],
          set_labels=('RNA', 'ATAC'),
          ax=axes[0])
    axes[0].set_title('Barcode Overlap (Before Pairing)', fontsize=14, fontweight='bold')

    # Pairing efficiency
    pairing_stats = {
        'RNA cells': len(rna_barcodes),
        'ATAC cells': len(atac_barcodes),
        'Paired cells': len(paired_barcodes),
        'RNA lost': len(rna_barcodes) - len(paired_barcodes),
        'ATAC lost': len(atac_barcodes) - len(paired_barcodes)
    }

    axes[1].bar(range(len(pairing_stats)), list(pairing_stats.values()),
                color=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12'])
    axes[1].set_xticks(range(len(pairing_stats)))
    axes[1].set_xticklabels(list(pairing_stats.keys()), rotation=45, ha='right')
    axes[1].set_ylabel('Number of Cells')
    axes[1].set_title('Pairing Statistics', fontsize=14, fontweight='bold')

    for i, (k, v) in enumerate(pairing_stats.items()):
        axes[1].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

    # Add efficiency text
    efficiency = len(paired_barcodes) / min(len(rna_barcodes), len(atac_barcodes)) * 100
    axes[1].text(0.5, 0.95, f'Pairing Efficiency: {efficiency:.1f}%',
                transform=axes[1].transAxes, ha='center', va='top',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairing_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def compute_umap(data: Dict, n_neighbors: int = 30, min_dist: float = 0.3):
    """Compute UMAP embeddings for RNA and ATAC data."""
    print("\nComputing UMAP embeddings...")

    for split, modalities in data.items():
        print(f"  {split} split...")

        # RNA UMAP
        rna = modalities['rna']
        print("    Computing RNA UMAP...")
        sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor='seurat_v3')
        sc.pp.pca(rna, n_comps=50, use_highly_variable=True)
        sc.pp.neighbors(rna, n_neighbors=n_neighbors, n_pcs=50)
        sc.tl.umap(rna, min_dist=min_dist)

        # ATAC UMAP (on top variable cCREs)
        atac = modalities['atac']
        print("    Computing ATAC UMAP...")

        # Find variable cCREs (handle sparse matrix)
        from scipy.sparse import issparse
        if issparse(atac.X):
            # For sparse matrix, compute variance manually
            cCRE_mean = np.asarray(atac.X.mean(axis=0)).flatten()
            cCRE_var = np.asarray(atac.X.power(2).mean(axis=0)).flatten() - cCRE_mean ** 2
        else:
            cCRE_var = np.asarray(atac.X.var(axis=0)).flatten()
        top_ccres_idx = np.argsort(cCRE_var)[-5000:]  # Top 5000 variable cCREs

        # Subset for PCA
        atac_subset = atac[:, top_ccres_idx].copy()
        sc.pp.pca(atac_subset, n_comps=50)
        sc.pp.neighbors(atac_subset, n_neighbors=n_neighbors, n_pcs=50)
        sc.tl.umap(atac_subset, min_dist=min_dist)

        # Copy UMAP back to original
        atac.obsm['X_umap'] = atac_subset.obsm['X_umap']
        atac.obsm['X_pca'] = atac_subset.obsm['X_pca']


def plot_umap(data: Dict, output_dir: str):
    """Plot UMAP embeddings."""
    print("\nPlotting UMAP embeddings...")

    splits = [s for s in ['train', 'val', 'test'] if s in data and s != 'full']

    # RNA UMAP
    fig, axes = plt.subplots(1, len(splits), figsize=(6*len(splits), 5))
    if len(splits) == 1:
        axes = [axes]

    fig.suptitle('RNA UMAP Embeddings', fontsize=16, fontweight='bold')

    for ax, split in zip(axes, splits):
        rna = data[split]['rna']
        # Color by dataset_id if available
        if 'dataset_id' in rna.obs.columns:
            sc.pl.umap(rna, color='dataset_id', ax=ax, show=False,
                       title=f'{split.capitalize()} ({rna.shape[0]} cells)',
                       legend_loc='right margin')
        else:
            sc.pl.umap(rna, ax=ax, show=False, title=f'{split.capitalize()} ({rna.shape[0]} cells)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_rna.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ATAC UMAP
    fig, axes = plt.subplots(1, len(splits), figsize=(6*len(splits), 5))
    if len(splits) == 1:
        axes = [axes]

    fig.suptitle('ATAC UMAP Embeddings', fontsize=16, fontweight='bold')

    for ax, split in zip(axes, splits):
        atac = data[split]['atac']
        # Color by dataset_id if available
        if 'dataset_id' in atac.obs.columns:
            sc.pl.umap(atac, color='dataset_id', ax=ax, show=False,
                       title=f'{split.capitalize()} ({atac.shape[0]} cells)',
                       legend_loc='right margin')
        else:
            sc.pl.umap(atac, ax=ax, show=False, title=f'{split.capitalize()} ({atac.shape[0]} cells)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_atac.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Combined plot (if dataset_id available)
    if 'full' in data and 'dataset_id' in data['full']['rna'].obs.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('UMAP Colored by Dataset ID', fontsize=16, fontweight='bold')

        sc.pl.umap(data['full']['rna'], color='dataset_id', ax=axes[0], show=False, title='RNA')
        sc.pl.umap(data['full']['atac'], color='dataset_id', ax=axes[1], show=False, title='ATAC')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_by_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_tfidf_analysis(data: Dict, output_dir: str):
    """Plot TF-IDF score analysis."""
    print("\nPlotting TF-IDF analysis...")

    # Use full paired data if available, otherwise train split
    if 'full' in data:
        atac = data['full']['atac']
        title_suffix = '(All Paired Cells)'
    else:
        atac = data['train']['atac']
        title_suffix = '(Training Set)'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'ATAC TF-IDF Analysis {title_suffix}', fontsize=16, fontweight='bold')

    # Top cCREs by document frequency
    cCRE_df = np.asarray((atac.X > 0).sum(axis=0)).flatten()
    top_100_idx = np.argsort(cCRE_df)[-100:]

    axes[0, 0].bar(range(100), cCRE_df[top_100_idx], color='#3498db')
    axes[0, 0].set_xlabel('cCRE Rank')
    axes[0, 0].set_ylabel('Document Frequency (# Cells)')
    axes[0, 0].set_title('Top 100 cCREs by Accessibility')

    # TF-IDF score distribution
    from scipy.sparse import issparse
    if issparse(atac.X):
        # For sparse matrix, get non-zero values directly
        tfidf_scores = atac.X.data
    else:
        # For dense matrix, flatten and get non-zero values
        tfidf_scores = atac.X[atac.X > 0].flatten()
    axes[0, 1].hist(tfidf_scores, bins=100, color='#2ecc71', edgecolor='black')
    axes[0, 1].set_xlabel('TF-IDF Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('TF-IDF Score Distribution')
    axes[0, 1].set_yscale('log')

    # Document frequency distribution
    axes[1, 0].hist(np.log10(cCRE_df + 1), bins=100, color='#9b59b6', edgecolor='black')
    axes[1, 0].set_xlabel('log10(Document Frequency + 1)')
    axes[1, 0].set_ylabel('Number of cCREs')
    axes[1, 0].set_title('cCRE Document Frequency Distribution')

    # Cell sentence statistics
    if 'cell_sentences' in atac.obs.columns:
        sentence_lengths = atac.obs['cell_sentences'].apply(
            lambda x: len(json.loads(x)) if isinstance(x, str) else 0
        )
        axes[1, 1].hist(sentence_lengths, bins=50, color='#e74c3c', edgecolor='black')
        axes[1, 1].set_xlabel('Cell Sentence Length')
        axes[1, 1].set_ylabel('Number of Cells')
        axes[1, 1].set_title('Tokenized Cell Sentence Lengths')

        # Add statistics text
        stats_text = f'Mean: {sentence_lengths.mean():.1f}\nMedian: {sentence_lengths.median():.1f}\nMax: {sentence_lengths.max()}'
        axes[1, 1].text(0.95, 0.95, stats_text,
                       transform=axes[1, 1].transAxes, ha='right', va='top',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tfidf_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_summary_stats(stats: Dict, output_dir: str):
    """Save summary statistics to JSON."""
    print("\nSaving summary statistics...")

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    stats_serializable = convert_to_serializable(stats)

    with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"  Saved to {os.path.join(output_dir, 'summary_statistics.json')}")


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Multi-Omics Data Visualization")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    # Parse splits
    splits = [s.strip() for s in args.splits.split(',')]
    print(f"Splits to visualize: {', '.join(splits)}")

    # Load data
    data = load_data(args.input_dir, splits)

    if not data:
        print("ERROR: No data loaded. Check input directory and split names.")
        return

    # Compute QC metrics
    stats = compute_qc_metrics(data)

    # Generate visualizations
    plot_qc_flowchart(stats, args.output_dir)
    plot_qc_distributions(stats, data, args.output_dir)
    plot_pairing_stats(args.input_dir, args.output_dir)

    if args.umap:
        compute_umap(data, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
        plot_umap(data, args.output_dir)

    plot_tfidf_analysis(data, args.output_dir)

    # Save statistics
    save_summary_stats(stats, args.output_dir)

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print(f"Output saved to: {args.output_dir}")
    print("\nGenerated files:")
    for file in os.listdir(args.output_dir):
        print(f"  - {file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
