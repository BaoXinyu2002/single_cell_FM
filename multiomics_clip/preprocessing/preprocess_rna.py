"""Preprocessing utilities for 10x Multiome RNA-seq data"""

import scanpy as sc
import pandas as pd
import numpy as np
from typing import List, Optional
from scipy.sparse import issparse


def load_gene_list(gene_list_path: str) -> List[str]:
    """
    Load scFoundation gene list (19264 genes)

    Args:
        gene_list_path: Path to OS_scRNA_gene_index.19264.tsv

    Returns:
        List of gene names
    """
    gene_list_df = pd.read_csv(gene_list_path, header=0, delimiter='\t')
    return list(gene_list_df['gene_name'])


def align_genes_to_scfoundation(
    adata,
    gene_list: List[str],
    var_gene_names_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Align gene expression data to scFoundation's 19264 genes

    Args:
        adata: AnnData object with gene expression
        gene_list: List of 19264 scFoundation genes
        var_gene_names_col: Column name in adata.var with gene names
                           (if None, uses adata.var_names)

    Returns:
        DataFrame with aligned genes [n_cells, 19264]
    """
    # Validate gene_list has no duplicates
    if len(gene_list) != len(set(gene_list)):
        raise ValueError(f"gene_list contains duplicates: {len(gene_list)} total, {len(set(gene_list))} unique")

    # Extract expression matrix
    if issparse(adata.X):
        expr_df = pd.DataFrame(
            adata.X.toarray(),
            index=adata.obs_names,
            columns=adata.var_names if var_gene_names_col is None else adata.var[var_gene_names_col]
        )
    else:
        expr_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names if var_gene_names_col is None else adata.var[var_gene_names_col]
        )

    # Check for duplicate column names in expr_df
    if expr_df.columns.duplicated().any():
        print(f"Warning: Found {expr_df.columns.duplicated().sum()} duplicate gene names in data")
        # Keep only first occurrence of each gene
        expr_df = expr_df.loc[:, ~expr_df.columns.duplicated(keep='first')]
        print(f"After removing duplicates: {expr_df.shape[1]} genes")

    # Identify genes that are present in data and in gene_list
    common_genes = list(set(expr_df.columns) & set(gene_list))

    # Identify missing genes (need zero-padding)
    to_fill_columns = list(set(gene_list) - set(expr_df.columns))

    print(f"Genes in common: {len(common_genes)}")
    print(f"Genes to zero-pad: {len(to_fill_columns)}")

    # Select only common genes from expr_df
    expr_df_subset = expr_df[common_genes]

    # Create zero-padding dataframe for missing genes
    padding_df = pd.DataFrame(
        np.zeros((expr_df.shape[0], len(to_fill_columns))),
        columns=to_fill_columns,
        index=expr_df.index
    )

    # Concatenate and reorder to match gene_list
    expr_df = pd.concat([expr_df_subset, padding_df], axis=1)
    expr_df = expr_df[gene_list]

    # Final validation
    assert expr_df.shape[1] == len(gene_list), f"Shape mismatch: {expr_df.shape[1]} != {len(gene_list)}"

    return expr_df


def preprocess_rna_multiome(
    h5_path: str,
    gene_list_path: str,
    min_genes: int = 200,
    min_cells: int = 3,
    target_sum: float = 1e4,
    max_genes: Optional[int] = None,
    var_gene_names_col: Optional[str] = None
):
    """
    Preprocess 10x Multiome RNA data for scFoundation

    Pipeline:
    1. Load 10x H5 file
    2. QC filtering (min_genes per cell, min_cells per gene)
    3. Normalize to target_sum counts per cell
    4. Log1p transform
    5. Align genes to scFoundation's 19264 genes

    Args:
        h5_path: Path to 10x filtered_feature_bc_matrix.h5
        gene_list_path: Path to OS_scRNA_gene_index.19264.tsv
        min_genes: Minimum genes expressed per cell for QC
        min_cells: Minimum cells expressing each gene for QC
        target_sum: Normalize counts to this value (default 1e4)
        max_genes: Maximum genes per cell for QC (optional)
        var_gene_names_col: Column name for gene names in var

    Returns:
        AnnData object with preprocessed RNA data [n_cells, 19264]
    """
    print(f"Loading RNA data from {h5_path}...")
    adata = sc.read_10x_h5(h5_path, gex_only=True)

    print(f"Initial data shape: {adata.shape}")

    # QC filtering
    print("Performing QC filtering...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    print(f"After QC: {adata.shape}")

    # Normalization
    print(f"Normalizing to {target_sum} counts per cell...")
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # Log1p transformation
    print("Applying log1p transformation...")
    sc.pp.log1p(adata)

    # Load gene list
    print(f"Loading gene list from {gene_list_path}...")
    gene_list = load_gene_list(gene_list_path)

    # Align genes
    print("Aligning genes to scFoundation's 19264 genes...")
    expr_aligned = align_genes_to_scfoundation(
        adata,
        gene_list,
        var_gene_names_col=var_gene_names_col
    )

    # Create new AnnData with aligned genes
    adata_aligned = sc.AnnData(
        X=expr_aligned.values,
        obs=adata.obs,
        var=pd.DataFrame(index=gene_list)
    )

    # Mark zero-padded genes
    existing_genes = set(adata.var_names if var_gene_names_col is None else adata.var[var_gene_names_col])
    adata_aligned.var['is_padded'] = [
        0 if gene in existing_genes else 1 for gene in gene_list
    ]

    print(f"Final shape: {adata_aligned.shape}")
    print(f"Zero-padded genes: {adata_aligned.var['is_padded'].sum()}")

    return adata_aligned


def preprocess_rna_from_anndata(
    adata,
    gene_list_path: str,
    min_genes: int = 200,
    min_cells: int = 3,
    target_sum: float = 1e4,
    max_genes: Optional[int] = None,
    var_gene_names_col: Optional[str] = None,
    skip_normalization: bool = False
):
    """
    Preprocess RNA data from existing AnnData object

    Args:
        adata: Input AnnData object
        gene_list_path: Path to OS_scRNA_gene_index.19264.tsv
        min_genes: Minimum genes expressed per cell for QC
        min_cells: Minimum cells expressing each gene for QC
        target_sum: Normalize counts to this value
        max_genes: Maximum genes per cell for QC (optional)
        var_gene_names_col: Column name for gene names in var
        skip_normalization: Skip normalization if data is already normalized

    Returns:
        AnnData object with preprocessed RNA data [n_cells, 19264]
    """
    print(f"Initial data shape: {adata.shape}")

    # Make a copy to avoid modifying original
    adata = adata.copy()

    # QC filtering
    print("Performing QC filtering...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    print(f"After QC: {adata.shape}")

    if not skip_normalization:
        # Normalization
        print(f"Normalizing to {target_sum} counts per cell...")
        sc.pp.normalize_total(adata, target_sum=target_sum)

        # Log1p transformation
        print("Applying log1p transformation...")
        sc.pp.log1p(adata)

    # Load gene list
    print(f"Loading gene list from {gene_list_path}...")
    gene_list = load_gene_list(gene_list_path)

    # Align genes
    print("Aligning genes to scFoundation's 19264 genes...")
    expr_aligned = align_genes_to_scfoundation(
        adata,
        gene_list,
        var_gene_names_col=var_gene_names_col
    )

    # Create new AnnData with aligned genes
    adata_aligned = sc.AnnData(
        X=expr_aligned.values,
        obs=adata.obs,
        var=pd.DataFrame(index=gene_list)
    )

    # Mark zero-padded genes
    existing_genes = set(adata.var_names if var_gene_names_col is None else adata.var[var_gene_names_col])
    adata_aligned.var['is_padded'] = [
        0 if gene in existing_genes else 1 for gene in gene_list
    ]

    print(f"Final shape: {adata_aligned.shape}")
    print(f"Zero-padded genes: {adata_aligned.var['is_padded'].sum()}")

    return adata_aligned
