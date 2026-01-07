#!/usr/bin/env python3
"""
Preprocessing script for 10x Multiome data

python3 multiomics_clip/scripts/preprocess_multiome.py \
    --rna_h5 /nfs/turbo/umms-drjieliu/usr/xinyubao/sclip/data/pbmc/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5 \
    --atac_fragments /nfs/turbo/umms-drjieliu/usr/xinyubao/sclip/data/pbmc/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz \
    --gene_list scFoundation/model/OS_scRNA_gene_index.19264.tsv \
    --ccre_bed EpiAgent/data/cCRE.bed \
    --output_dir ./preprocessed_data
    
This script processes 10x Multiome RNA and ATAC data for Multi-Omics CLIP training.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import scanpy as sc
from multiomics_clip.preprocessing import (
    preprocess_rna_multiome,
    preprocess_atac_multiome,
    pair_rna_atac_by_barcode,
    split_train_val_test
)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess 10x Multiome data for Multi-Omics CLIP"
    )

    # Input paths
    parser.add_argument(
        '--rna_h5',
        type=str,
        required=True,
        help='Path to 10x filtered_feature_bc_matrix.h5 (RNA)'
    )
    parser.add_argument(
        '--atac_fragments',
        type=str,
        required=True,
        help='Path to fragments.tsv.gz (ATAC)'
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
        '--ccre_df',
        type=str,
        default='EpiAgent/data/cCRE_document_frequency.npy',
        help='Path to precomputed cCRE document frequency file (.npy or .csv)'
    )

    # Output paths
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for preprocessed data'
    )

    # RNA preprocessing parameters
    parser.add_argument(
        '--min_genes',
        type=int,
        default=500,
        help='Minimum genes per cell for QC'
    )
    parser.add_argument(
        '--max_genes',
        type=int,
        default=None,
        help='Maximum genes per cell for QC'
    )
    parser.add_argument(
        '--min_cells_gene',
        type=int,
        default=3,
        help='Minimum cells per gene for QC'
    )

    # ATAC preprocessing parameters
    parser.add_argument(
        '--min_fragments',
        type=int,
        default=200,
        help='Minimum fragments per cell for QC'
    )
    parser.add_argument(
        '--max_fragments',
        type=int,
        default=None,
        help='Maximum fragments per cell for QC'
    )
    parser.add_argument(
        '--min_cells_ccre',
        type=int,
        default=3,
        help='Minimum cells per cCRE for QC'
    )

    # Pairing and splitting
    parser.add_argument(
        '--train_frac',
        type=float,
        default=0.8,
        help='Fraction for training set'
    )
    parser.add_argument(
        '--val_frac',
        type=float,
        default=0.1,
        help='Fraction for validation set'
    )
    parser.add_argument(
        '--test_frac',
        type=float,
        default=0.1,
        help='Fraction for test set'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Multi-Omics CLIP: 10x Multiome Preprocessing")
    print("=" * 80)

    # Step 1: Preprocess RNA
    print("\n[1/4] Preprocessing RNA data...")
    rna_adata = preprocess_rna_multiome(
        h5_path=args.rna_h5,
        gene_list_path=args.gene_list,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        min_cells=args.min_cells_gene
    )

    # Save RNA data
    rna_output = os.path.join(args.output_dir, 'rna_preprocessed.h5ad')
    rna_adata.write_h5ad(rna_output)
    print(f"Saved RNA data to: {rna_output}")

    # Step 2: Preprocess ATAC
    print("\n[2/4] Preprocessing ATAC data...")
    atac_output_dir = os.path.join(args.output_dir, 'atac_preprocessing')
    atac_adata = preprocess_atac_multiome(
        fragments_path=args.atac_fragments,
        ccre_bed_path=args.ccre_bed,
        output_dir=atac_output_dir,
        min_fragments=args.min_fragments,
        max_fragments=args.max_fragments,
        min_cells_per_ccre=args.min_cells_ccre,
        ccre_df_path=args.ccre_df
    )

    # Save ATAC data
    atac_output = os.path.join(args.output_dir, 'atac_preprocessed.h5ad')
    atac_adata.write_h5ad(atac_output)
    print(f"Saved ATAC data to: {atac_output}")

    # Step 3: Pair modalities
    print("\n[3/4] Pairing RNA and ATAC by cell barcodes...")
    rna_paired, atac_paired, common_barcodes = pair_rna_atac_by_barcode(
        rna_adata,
        atac_adata
    )

    # Save paired data
    rna_paired_output = os.path.join(args.output_dir, 'rna_paired.h5ad')
    atac_paired_output = os.path.join(args.output_dir, 'atac_paired.h5ad')

    rna_paired.write_h5ad(rna_paired_output)
    atac_paired.write_h5ad(atac_paired_output)

    print(f"Saved paired RNA data to: {rna_paired_output}")
    print(f"Saved paired ATAC data to: {atac_paired_output}")

    # Step 4: Split train/val/test
    print("\n[4/4] Splitting into train/val/test sets...")
    train_rna, train_atac, val_rna, val_atac, test_rna, test_atac = split_train_val_test(
        rna_paired,
        atac_paired,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_seed=args.random_seed
    )

    # Save splits
    train_rna.write_h5ad(os.path.join(args.output_dir, 'train_rna.h5ad'))
    train_atac.write_h5ad(os.path.join(args.output_dir, 'train_atac.h5ad'))

    val_rna.write_h5ad(os.path.join(args.output_dir, 'val_rna.h5ad'))
    val_atac.write_h5ad(os.path.join(args.output_dir, 'val_atac.h5ad'))

    test_rna.write_h5ad(os.path.join(args.output_dir, 'test_rna.h5ad'))
    test_atac.write_h5ad(os.path.join(args.output_dir, 'test_atac.h5ad'))

    print(f"\nSaved train/val/test splits to: {args.output_dir}")

    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - RNA (all):    {rna_output}")
    print(f"  - ATAC (all):   {atac_output}")
    print(f"  - RNA (paired): {rna_paired_output}")
    print(f"  - ATAC (paired): {atac_paired_output}")
    print(f"  - Train/Val/Test splits in: {args.output_dir}")


if __name__ == '__main__':
    main()
