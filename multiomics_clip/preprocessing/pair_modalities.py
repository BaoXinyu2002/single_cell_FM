"""Utilities for pairing RNA and ATAC modalities by cell barcodes"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List


def pair_rna_atac_by_barcode(
    rna_adata,
    atac_adata,
    barcode_suffix_rna: Optional[str] = None,
    barcode_suffix_atac: Optional[str] = None,
    min_common_cells: int = 100
) -> Tuple:
    """
    Pair RNA and ATAC data by matching cell barcodes

    For 10x Multiome data, RNA and ATAC measurements come from the same cells.
    This function aligns the two modalities by their cell barcodes.

    Args:
        rna_adata: AnnData object with RNA data
        atac_adata: AnnData object with ATAC data
        barcode_suffix_rna: Suffix to remove from RNA barcodes (e.g., '-1')
        barcode_suffix_atac: Suffix to remove from ATAC barcodes (e.g., '-1')
        min_common_cells: Minimum number of common cells required

    Returns:
        Tuple of (paired_rna_adata, paired_atac_adata, common_barcodes)

    Raises:
        ValueError: If fewer than min_common_cells common barcodes are found
    """
    print("Pairing RNA and ATAC data by cell barcodes...")

    # Get barcodes
    rna_barcodes = rna_adata.obs_names.tolist()
    atac_barcodes = atac_adata.obs_names.tolist()

    print(f"RNA cells: {len(rna_barcodes)}")
    print(f"ATAC cells: {len(atac_barcodes)}")

    # Remove suffixes if specified
    if barcode_suffix_rna:
        rna_barcodes_clean = [bc.replace(barcode_suffix_rna, '') for bc in rna_barcodes]
        rna_barcode_map = dict(zip(rna_barcodes_clean, rna_barcodes))
    else:
        rna_barcodes_clean = rna_barcodes
        rna_barcode_map = dict(zip(rna_barcodes, rna_barcodes))

    if barcode_suffix_atac:
        atac_barcodes_clean = [bc.replace(barcode_suffix_atac, '') for bc in atac_barcodes]
        atac_barcode_map = dict(zip(atac_barcodes_clean, atac_barcodes))
    else:
        atac_barcodes_clean = atac_barcodes
        atac_barcode_map = dict(zip(atac_barcodes, atac_barcodes))

    # Find common barcodes
    rna_set = set(rna_barcodes_clean)
    atac_set = set(atac_barcodes_clean)
    common_barcodes = sorted(list(rna_set.intersection(atac_set)))

    print(f"Common cells: {len(common_barcodes)}")

    if len(common_barcodes) < min_common_cells:
        raise ValueError(
            f"Only {len(common_barcodes)} common cells found, "
            f"less than minimum required {min_common_cells}. "
            f"Check if barcode formats match between RNA and ATAC data."
        )

    # Map back to original barcodes
    rna_barcodes_paired = [rna_barcode_map[bc] for bc in common_barcodes]
    atac_barcodes_paired = [atac_barcode_map[bc] for bc in common_barcodes]

    # Subset and reorder both datasets
    rna_adata_paired = rna_adata[rna_barcodes_paired, :].copy()
    atac_adata_paired = atac_adata[atac_barcodes_paired, :].copy()

    # Verify order matches
    assert len(rna_adata_paired) == len(atac_adata_paired)

    print(f"Paired datasets created with {len(common_barcodes)} cells")

    return rna_adata_paired, atac_adata_paired, common_barcodes


def split_train_val_test(
    rna_adata,
    atac_adata,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 42
) -> Tuple:
    """
    Split paired RNA-ATAC data into train/val/test sets

    Args:
        rna_adata: Paired RNA AnnData
        atac_adata: Paired ATAC AnnData
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_rna, train_atac, val_rna, val_atac, test_rna, test_atac)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Train/val/test fractions must sum to 1.0"

    assert len(rna_adata) == len(atac_adata), \
        "RNA and ATAC must have same number of cells"

    n_cells = len(rna_adata)

    # Generate random indices
    np.random.seed(random_seed)
    indices = np.random.permutation(n_cells)

    # Calculate split points
    n_train = int(n_cells * train_frac)
    n_val = int(n_cells * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"Splitting data:")
    print(f"  Train: {len(train_idx)} cells ({len(train_idx) / n_cells * 100:.1f}%)")
    print(f"  Val:   {len(val_idx)} cells ({len(val_idx) / n_cells * 100:.1f}%)")
    print(f"  Test:  {len(test_idx)} cells ({len(test_idx) / n_cells * 100:.1f}%)")

    # Split datasets
    train_rna = rna_adata[train_idx, :].copy()
    train_atac = atac_adata[train_idx, :].copy()

    val_rna = rna_adata[val_idx, :].copy()
    val_atac = atac_adata[val_idx, :].copy()

    test_rna = rna_adata[test_idx, :].copy()
    test_atac = atac_adata[test_idx, :].copy()

    return train_rna, train_atac, val_rna, val_atac, test_rna, test_atac


def stratified_split_by_celltype(
    rna_adata,
    atac_adata,
    celltype_column: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 42
) -> Tuple:
    """
    Stratified split by cell type to maintain cell type proportions

    Args:
        rna_adata: Paired RNA AnnData (must have celltype_column in obs)
        atac_adata: Paired ATAC AnnData
        celltype_column: Column name in rna_adata.obs with cell type labels
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_rna, train_atac, val_rna, val_atac, test_rna, test_atac)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    assert len(rna_adata) == len(atac_adata)

    assert celltype_column in rna_adata.obs.columns, \
        f"Column '{celltype_column}' not found in rna_adata.obs"

    np.random.seed(random_seed)

    # Get cell types
    celltypes = rna_adata.obs[celltype_column].values
    unique_celltypes = np.unique(celltypes)

    train_indices = []
    val_indices = []
    test_indices = []

    # Stratify by cell type
    for celltype in unique_celltypes:
        celltype_mask = celltypes == celltype
        celltype_indices = np.where(celltype_mask)[0]

        n_celltype = len(celltype_indices)

        # Shuffle indices for this cell type
        np.random.shuffle(celltype_indices)

        # Split
        n_train = int(n_celltype * train_frac)
        n_val = int(n_celltype * val_frac)

        train_indices.extend(celltype_indices[:n_train])
        val_indices.extend(celltype_indices[n_train:n_train + n_val])
        test_indices.extend(celltype_indices[n_train + n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    print(f"Stratified split by cell type:")
    print(f"  Train: {len(train_indices)} cells")
    print(f"  Val:   {len(val_indices)} cells")
    print(f"  Test:  {len(test_indices)} cells")

    # Split datasets
    train_rna = rna_adata[train_indices, :].copy()
    train_atac = atac_adata[train_indices, :].copy()

    val_rna = rna_adata[val_indices, :].copy()
    val_atac = atac_adata[val_indices, :].copy()

    test_rna = rna_adata[test_indices, :].copy()
    test_atac = atac_adata[test_indices, :].copy()

    # Print cell type distributions
    print("\nCell type distribution:")
    for split_name, split_rna in [("Train", train_rna), ("Val", val_rna), ("Test", test_rna)]:
        ct_counts = split_rna.obs[celltype_column].value_counts()
        print(f"  {split_name}:")
        for ct, count in ct_counts.items():
            print(f"    {ct}: {count}")

    return train_rna, train_atac, val_rna, val_atac, test_rna, test_atac
