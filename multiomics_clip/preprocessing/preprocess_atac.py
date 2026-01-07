"""Preprocessing utilities for 10x Multiome ATAC-seq data"""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional
import subprocess
import os
import shutil

try:
    from epiagent.preprocessing import construct_cell_by_ccre_matrix, global_TFIDF
    from epiagent.tokenization import tokenization
except ImportError as e:
    raise ImportError(
        "EpiAgent package not found. Please install it using:\n"
        "  conda activate EpiAgent\n"
        "  pip install epiagent\n"
        f"Original error: {e}"
    )


def preprocess_fragments_to_ccre(
    fragments_path: str,
    ccre_bed_path: str,
    output_intersect_path: str,
    genome_build: str = "hg38"
) -> str:
    """
    Intersect fragment file with cCRE bed file using bedtools

    Args:
        fragments_path: Path to fragments.tsv.gz file
        ccre_bed_path: Path to cCRE.bed file (hg38)
        output_intersect_path: Output path for intersection results
        genome_build: Genome build of fragments (hg38 or hg19)

    Returns:
        Path to intersection file
    """
    # Check if liftOver is needed
    if genome_build == "hg19":
        print("Warning: Fragments are in hg19. Consider using liftOver to convert to hg38.")
        print("You can use: liftOver fragments.bed hg19ToHg38.over.chain.gz output.hg38.bed unmapped.bed")
        raise NotImplementedError(
            "hg19 to hg38 conversion not implemented. Please convert fragments to hg38 first."
        )

    # Check if bedtools is installed
    if not shutil.which("bedtools"):
        raise RuntimeError(
            "bedtools not found. Please install it using:\n"
            "  conda install -c bioconda bedtools"
        )

    # Run bedtools intersect (using list args to prevent shell injection)
    print(f"Intersecting fragments with cCREs using bedtools...")

    try:
        with open(output_intersect_path, 'w') as outfile:
            subprocess.run(
                ["bedtools", "intersect", "-a", fragments_path, "-b", ccre_bed_path, "-wa", "-wb"],
                stdout=outfile,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
        print(f"Intersection complete: {output_intersect_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running bedtools: {e}")
        print(f"stderr: {e.stderr}")
        raise

    return output_intersect_path


def preprocess_atac_multiome(
    fragments_path: str,
    ccre_bed_path: str,
    output_dir: str,
    min_fragments: int = 200,
    max_fragments: Optional[int] = None,
    min_cells_per_ccre: int = 3,
    num_ccres: int = 1355445,
    ccre_df_path: Optional[str] = None
):
    """
    Preprocess 10x Multiome ATAC data for EpiAgent

    Pipeline:
    1. Intersect fragments with cCREs using bedtools
    2. Construct cell-by-cCRE matrix
    3. QC filtering (min/max fragments per cell, min cells per cCRE)
    4. Apply global TF-IDF transformation
    5. Tokenize cCREs by TF-IDF scores to create cell sentences

    Args:
        fragments_path: Path to fragments.tsv.gz
        ccre_bed_path: Path to cCRE.bed file
        output_dir: Directory to save intermediate and final outputs
        min_fragments: Minimum fragments per cell for QC
        max_fragments: Maximum fragments per cell for QC (optional)
        min_cells_per_ccre: Minimum cells expressing each cCRE for QC
        num_ccres: Total number of cCREs
        ccre_df_path: Path to cCRE document frequency file (for TF-IDF)

    Returns:
        AnnData object with tokenized cell sentences in obs['cell_sentences']
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Intersect fragments with cCREs
    intersect_path = os.path.join(output_dir, "fragments_ccre_intersect.txt")
    if not os.path.exists(intersect_path):
        print("Step 1: Intersecting fragments with cCREs...")
        preprocess_fragments_to_ccre(
            fragments_path,
            ccre_bed_path,
            intersect_path,
            genome_build="hg38"
        )
    else:
        print(f"Using existing intersection file: {intersect_path}")

    # Step 2: Construct cell-by-cCRE matrix
    print("Step 2: Constructing cell-by-cCRE matrix...")
    adata = construct_cell_by_ccre_matrix(intersect_path, ccre_bed_path)

    print(f"Initial data shape: {adata.shape}")

    # Step 3: QC filtering
    print("Step 3: Performing QC filtering...")

    # Calculate fragments per cell (sum of cCRE accessibility)
    if hasattr(adata.X, 'toarray'):
        fragments_per_cell = np.array(adata.X.sum(axis=1)).flatten()
    else:
        fragments_per_cell = adata.X.sum(axis=1)

    # Filter cells by fragment count
    cell_mask = fragments_per_cell >= min_fragments
    if max_fragments is not None:
        cell_mask &= fragments_per_cell <= max_fragments

    adata = adata[cell_mask, :].copy()
    print(f"After cell filtering: {adata.shape}")

    # NOTE: We do NOT filter cCREs here because EpiAgent tokenization requires
    # exactly 1,355,445 cCREs (the full vocabulary). Low-count cCREs will have
    # low TF-IDF scores and won't be selected during tokenization anyway.
    print(f"Keeping all {adata.shape[1]} cCREs for EpiAgent compatibility")

    # Step 4: Apply global TF-IDF
    print("Step 4: Applying global TF-IDF transformation...")

    if ccre_df_path is not None and os.path.exists(ccre_df_path):
        # Load precomputed document frequencies
        print(f"Loading precomputed document frequencies from: {ccre_df_path}")
        if ccre_df_path.endswith('.npy'):
            ccre_doc_freq = np.load(ccre_df_path)
        elif ccre_df_path.endswith('.csv'):
            ccre_df = pd.read_csv(ccre_df_path, index_col=0)
            ccre_doc_freq = ccre_df.values.flatten()
        else:
            raise ValueError(f"Unsupported file format: {ccre_df_path}. Use .npy or .csv")

        print(f"  Loaded document frequencies: shape={ccre_doc_freq.shape}, mean={ccre_doc_freq.mean():.2f}")
        adata = global_TFIDF(adata, ccre_doc_freq)
    else:
        # Compute document frequencies from current data
        print("WARNING: No ccre_df_path provided. Computing document frequencies from current data.")
        print("This may lead to poor TF-IDF scores. Use precomputed frequencies from EpiAgent corpus instead.")
        # Document frequency = number of cells in which each cCRE appears (> 0)
        ccre_doc_freq = np.asarray((adata.X > 0).sum(axis=0)).flatten()
        print(f"  Document frequency shape: {ccre_doc_freq.shape}")
        print(f"  Min: {ccre_doc_freq.min()}, Max: {ccre_doc_freq.max()}, Mean: {ccre_doc_freq.mean():.2f}")
        adata = global_TFIDF(adata, ccre_doc_freq)

    # Step 5: Tokenization
    print("Step 5: Tokenizing cCREs to create cell sentences...")
    tokenization(adata)

    print(f"Final shape: {adata.shape}")
    print(f"Cell sentences created for {len(adata)} cells")

    # Save processed data
    output_path = os.path.join(output_dir, "atac_preprocessed.h5ad")
    adata.write_h5ad(output_path)
    print(f"Saved preprocessed ATAC data to: {output_path}")

    return adata


def preprocess_atac_from_anndata(
    adata,
    ccre_bed_path: str,
    min_fragments: int = 200,
    max_fragments: Optional[int] = None,
    min_cells_per_ccre: int = 3,
    num_ccres: int = 1355445,
    ccre_df_path: Optional[str] = None
):
    """
    Preprocess ATAC data from existing AnnData with cell-by-cCRE matrix

    Args:
        adata: Input AnnData with cell-by-cCRE counts
        ccre_bed_path: Path to cCRE.bed file
        min_fragments: Minimum fragments per cell for QC
        max_fragments: Maximum fragments per cell for QC
        min_cells_per_ccre: Minimum cells expressing each cCRE
        num_ccres: Total number of cCREs
        ccre_df_path: Path to cCRE document frequency file

    Returns:
        AnnData object with tokenized cell sentences
    """
    print(f"Initial data shape: {adata.shape}")

    # Make a copy
    adata = adata.copy()

    # QC filtering
    print("Performing QC filtering...")

    # Calculate fragments per cell
    if hasattr(adata.X, 'toarray'):
        fragments_per_cell = np.array(adata.X.sum(axis=1)).flatten()
    else:
        fragments_per_cell = adata.X.sum(axis=1)

    # Filter cells
    cell_mask = fragments_per_cell >= min_fragments
    if max_fragments is not None:
        cell_mask &= fragments_per_cell <= max_fragments

    adata = adata[cell_mask, :].copy()
    print(f"After cell filtering: {adata.shape}")

    # NOTE: We do NOT filter cCREs here because EpiAgent tokenization requires
    # exactly 1,355,445 cCREs (the full vocabulary).
    print(f"Keeping all {adata.shape[1]} cCREs for EpiAgent compatibility")

    # Apply global TF-IDF
    print("Applying global TF-IDF transformation...")
    if ccre_df_path is not None and os.path.exists(ccre_df_path):
        print(f"Loading precomputed document frequencies from: {ccre_df_path}")
        if ccre_df_path.endswith('.npy'):
            ccre_doc_freq = np.load(ccre_df_path)
        elif ccre_df_path.endswith('.csv'):
            ccre_df = pd.read_csv(ccre_df_path, index_col=0)
            ccre_doc_freq = ccre_df.values.flatten()
        else:
            raise ValueError(f"Unsupported file format: {ccre_df_path}. Use .npy or .csv")

        print(f"  Loaded document frequencies: shape={ccre_doc_freq.shape}, mean={ccre_doc_freq.mean():.2f}")
        adata = global_TFIDF(adata, ccre_doc_freq)
    else:
        # Compute document frequencies from current data
        print("WARNING: No ccre_df_path provided. Computing document frequencies from current data.")
        print("This may lead to poor TF-IDF scores. Use precomputed frequencies from EpiAgent corpus instead.")
        ccre_doc_freq = np.asarray((adata.X > 0).sum(axis=0)).flatten()
        print(f"  Document frequency: min={ccre_doc_freq.min()}, max={ccre_doc_freq.max()}, mean={ccre_doc_freq.mean():.2f}")
        adata = global_TFIDF(adata, ccre_doc_freq)

    # Tokenization
    print("Tokenizing cCREs to create cell sentences...")
    tokenization(adata, num_cCREs=num_ccres)

    print(f"Final shape: {adata.shape}")
    print(f"Cell sentences created for {len(adata)} cells")

    return adata
