"""Dataset classes for paired RNA-seq and ATAC-seq data"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import json
import ast
import pandas as pd
import warnings
from scipy.sparse import issparse
import os


def _should_log():
    """Check if current process should log (rank 0 only in distributed mode)"""
    rank = int(os.environ.get('RANK', 0))
    return rank == 0


class PairedMultiOmicsDataset(Dataset):
    """
    Dataset for paired RNA-seq and ATAC-seq data from the same cells

    This dataset handles 10x Multiome data where RNA and ATAC measurements
    are from the same cells (matched by barcodes).

    Args:
        rna_adata: AnnData object with normalized log1p gene expression
                   Shape: [n_cells, 19264 genes] (scFoundation standard gene set)
                   Note: Dataset will append 2 metadata values to create 19266-dim vectors
        atac_adata: AnnData object with tokenized cell_sentences in obs
                    obs['cell_sentences'] contains JSON-formatted cCRE indices
        gene_list: List of 19264 genes for scFoundation (in order)
        target_resolution: scFoundation target resolution parameter (default: 4.0)
        max_atac_length: Maximum ATAC sequence length (tokens will be truncated)
        random_truncate: If True, randomly sample cCREs when truncating
        min_atac_length: Minimum ATAC sequence length for quality filtering (default: 2000)
                        Cells with fewer cCREs are filtered out as low quality
    """

    def __init__(
        self,
        rna_adata,
        atac_adata,
        gene_list: List[str],
        target_resolution: float = 4.0,
        max_atac_length: int = 8192,
        random_truncate: bool = True,
        min_atac_length: int = 1000
    ):
        # Validate inputs
        assert len(rna_adata) == len(atac_adata), \
            f"RNA and ATAC must have same number of cells: {len(rna_adata)} vs {len(atac_adata)}"

        # scFoundation requires exactly 19264 genes (standard gene set)
        # Two metadata values (target_resolution, log10_total_count) will be appended later
        assert rna_adata.shape[1] == 19264, \
            f"RNA data must have exactly 19264 genes (scFoundation standard), got {rna_adata.shape[1]}"

        assert 'cell_sentences' in atac_adata.obs.columns, \
            "ATAC data must have 'cell_sentences' column in obs"

        # ===== CRITICAL FIX: Filter low-quality cells by ATAC length =====
        # Short ATAC sequences indicate poor quality (damaged cells, low coverage)
        # This also prevents memory fragmentation by ensuring consistent tensor sizes
        if _should_log():
            print(f"\nFiltering cells by ATAC sequence length (min={min_atac_length} cCREs)...")
        atac_sentences_all = atac_adata.obs['cell_sentences'].tolist()

        valid_indices = []
        filtered_count = 0

        for idx, sentence_str in enumerate(atac_sentences_all):
            # Parse cell sentence to get length
            if isinstance(sentence_str, str):
                try:
                    sentence = json.loads(sentence_str)
                except json.JSONDecodeError:
                    sentence = ast.literal_eval(sentence_str)
            elif isinstance(sentence_str, list):
                sentence = sentence_str
            else:
                warnings.warn(f"Unexpected cell_sentence format at index {idx}: {type(sentence_str)}")
                filtered_count += 1
                continue

            length = len(sentence)

            if length >= min_atac_length:
                valid_indices.append(idx)
            else:
                filtered_count += 1

        if _should_log():
            print(f"  Filtered: {filtered_count}/{len(atac_sentences_all)} low-quality cells")
            print(f"  Remaining: {len(valid_indices)} high-quality cells")

        if len(valid_indices) == 0:
            raise ValueError(
                f"All cells filtered out with min_atac_length={min_atac_length}. "
                "Consider lowering the threshold."
            )

        # ===== RNA-side filtering: Remove cells with too many non-zero genes =====
        # High gene count cells cause memory spikes in scFoundation's gatherData() function
        if _should_log():
            print(f"\nFiltering cells by RNA non-zero gene count (max=9000)...")
        rna_temp = rna_adata[valid_indices]

        # Count non-zero genes per cell
        if issparse(rna_temp.X):
            nnz_per_cell = np.array((rna_temp.X != 0).sum(axis=1)).flatten()
        else:
            nnz_per_cell = np.count_nonzero(rna_temp.X, axis=1)

        # Filter cells with >9000 non-zero genes
        rna_valid_mask = nnz_per_cell <= 8500
        rna_valid_indices_local = np.where(rna_valid_mask)[0]

        # Map back to original indices
        final_valid_indices = [valid_indices[i] for i in rna_valid_indices_local]

        rna_filtered_count = len(valid_indices) - len(final_valid_indices)
        if _should_log():
            print(f"  Filtered: {rna_filtered_count}/{len(valid_indices)} high-gene-count cells")
            print(f"  Remaining: {len(final_valid_indices)} cells after both filters")
            print(f"  RNA nnz stats - min: {nnz_per_cell[rna_valid_indices_local].min()}, "
                  f"max: {nnz_per_cell[rna_valid_indices_local].max()}, "
                  f"mean: {nnz_per_cell[rna_valid_indices_local].mean():.1f}")

        if len(final_valid_indices) == 0:
            raise ValueError(
                "All cells filtered out after ATAC and RNA quality filtering. "
                "Consider adjusting thresholds."
            )

        # Filter both RNA and ATAC data to keep only valid cells
        self.rna_adata = rna_adata[final_valid_indices].copy()
        self.atac_sentences = [atac_sentences_all[i] for i in final_valid_indices]
        self.cell_barcodes = rna_adata.obs_names[final_valid_indices].tolist()

        self.gene_list = gene_list
        self.target_resolution = target_resolution
        self.max_atac_length = max_atac_length
        self.random_truncate = random_truncate

    def __len__(self) -> int:
        return len(self.rna_adata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get paired RNA and ATAC data for a single cell

        Args:
            idx: Cell index

        Returns:
            Tuple of (rna_data, rna_gene_ids, atac_input_ids)
                - rna_data: [19266] = 19264 genes + 2 metadata (target_resolution, log10_total_count)
                - rna_gene_ids: [19266] position IDs for all elements
                - atac_input_ids: [seq_len] cCRE indices
        """
        # ===== RNA Data =====
        # Extract gene expression for this cell
        if issparse(self.rna_adata.X):
            rna_expr = self.rna_adata[idx].X.toarray().flatten()
        else:
            rna_expr = self.rna_adata[idx].X.flatten()

        # Calculate total count (before log1p, so we need to exp)
        # Assuming data is already log1p transformed
        total_count = np.expm1(rna_expr).sum()

        # Validate that total_count is reasonable (should be > 100 for typical scRNA-seq)
        # If too low, data may be pre-normalized (divided by total count)
        if total_count < 10:
            warnings.warn(
                f"Cell {idx} has suspiciously low total count ({total_count:.2f}). "
                "If data is pre-normalized (CPM/TPM), the log10_total_count feature will be inaccurate. "
                "scFoundation expects unnormalized log1p counts.",
                UserWarning
            )

        # Use larger epsilon (1e-6) for numerical stability
        # Prevents log10(0) when total_count is very small or zero
        log10_total_count = np.log10(max(total_count, 1e-6) + 1e-6)

        # Prepare scFoundation input format
        # Concatenate: 19264 genes + 2 metadata values = 19266 total elements
        # Last 2 values: [target_resolution, log10(total_count)]
        rna_data = np.concatenate([
            rna_expr,  # 19264 genes (already log1p normalized)
            [self.target_resolution, log10_total_count]  # 2 metadata values
        ])
        rna_data = torch.tensor(rna_data, dtype=torch.float32)  # Final shape: [19266]

        # Position IDs for all 19266 elements (0 to 19265)
        rna_gene_ids = torch.arange(19266, dtype=torch.long)

        # ===== ATAC Data =====
        # Parse cell sentence (JSON-formatted list of cCRE indices)
        cell_sentence_str = self.atac_sentences[idx]

        # Handle different formats
        if isinstance(cell_sentence_str, str):
            try:
                cell_sentence = json.loads(cell_sentence_str)
            except json.JSONDecodeError:
                # If it's a string representation of a list, parse safely
                cell_sentence = ast.literal_eval(cell_sentence_str)
        elif isinstance(cell_sentence_str, list):
            cell_sentence = cell_sentence_str
        else:
            raise ValueError(f"Unexpected cell_sentence format: {type(cell_sentence_str)}")

        # Truncate if needed (keep max_atac_length - 2 for CLS and SEP tokens)
        max_ccres = self.max_atac_length - 2
        if len(cell_sentence) > max_ccres:
            if self.random_truncate:
                # Randomly sample cCREs
                indices = np.sort(np.random.choice(
                    len(cell_sentence),
                    max_ccres,
                    replace=False
                ))
                cell_sentence = [cell_sentence[i] for i in indices]
            else:
                # Keep top-k by TF-IDF (already sorted in preprocessing)
                cell_sentence = cell_sentence[:max_ccres]

        # Add special tokens: CLS=1, SEP=2
        atac_input = [1] + cell_sentence + [2]
        atac_input_ids = torch.tensor(atac_input, dtype=torch.long)

        return rna_data, rna_gene_ids, atac_input_ids


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], fixed_atac_length: int = 8192) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for batching paired multi-omics data

    CRITICAL FIX: Uses fixed-length padding for ATAC sequences to prevent memory fragmentation
    during gradient accumulation. Variable-length padding causes different tensor sizes across
    micro-batches, leading to severe GPU memory fragmentation and OOM errors.

    Args:
        batch: List of (rna_data, rna_gene_ids, atac_input_ids) tuples
        fixed_atac_length: Fixed length to pad all ATAC sequences (default: 8192)
                          All sequences are padded/truncated to this exact length

    Returns:
        Tuple of batched tensors:
            - rna_data: [batch_size, 19266]
            - rna_gene_ids: [batch_size, 19266]
            - atac_input_ids: [batch_size, fixed_atac_length] (padded with 0s)
    """
    rna_data_list, rna_gene_ids_list, atac_input_list = zip(*batch)

    # Stack RNA data (fixed length)
    rna_data = torch.stack(rna_data_list, dim=0)
    rna_gene_ids = torch.stack(rna_gene_ids_list, dim=0)

    # CRITICAL FIX: Pad all ATAC inputs to FIXED length (not batch max)
    # This ensures identical tensor sizes across all micro-batches, eliminating fragmentation
    atac_padded = []
    for x in atac_input_list:
        # Truncate if longer than fixed length
        if x.shape[0] > fixed_atac_length:
            x = x[:fixed_atac_length]

        # Pad to fixed length with 0 (PAD token)
        padded = torch.nn.functional.pad(
            x, (0, fixed_atac_length - x.shape[0]), value=0
        )
        atac_padded.append(padded)

    atac_input_ids = torch.stack(atac_padded, dim=0)

    return rna_data, rna_gene_ids, atac_input_ids


def create_dataloader(
    rna_adata,
    atac_adata,
    gene_list: List[str],
    batch_size: int = 32,
    target_resolution: float = 4.0,
    max_atac_length: int = 8192,
    min_atac_length: int = 2000,
    shuffle: bool = True,
    num_workers: int = 4,
    random_truncate: bool = True
):
    """
    Create a DataLoader for paired multi-omics data

    Args:
        rna_adata: AnnData object with RNA data
        atac_adata: AnnData object with ATAC data
        gene_list: List of 19264 genes
        batch_size: Batch size
        target_resolution: scFoundation target resolution
        max_atac_length: Maximum ATAC sequence length (used for fixed padding)
        min_atac_length: Minimum ATAC sequence length for quality filtering (default: 2000)
        shuffle: Whether to shuffle data
        num_workers: Number of dataloader workers
        random_truncate: Randomly truncate ATAC sequences

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = PairedMultiOmicsDataset(
        rna_adata=rna_adata,
        atac_adata=atac_adata,
        gene_list=gene_list,
        target_resolution=target_resolution,
        max_atac_length=max_atac_length,
        min_atac_length=min_atac_length,
        random_truncate=random_truncate
    )

    # Create collate function with fixed ATAC length to prevent memory fragmentation
    from functools import partial
    collate_fn_fixed = partial(collate_fn, fixed_atac_length=max_atac_length)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_fixed,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for training
    )

    return dataloader
