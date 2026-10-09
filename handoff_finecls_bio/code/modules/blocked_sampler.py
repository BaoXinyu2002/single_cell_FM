"""Same-dataset-blocked batch sampler for the LIVE-FM FILIP refiner trainer.

WHY: the plain ``DataLoader(shuffle=True)`` yields batches whose cells come from
DIFFERENT curated datasets. In a contrastive loss the in-batch negatives are then
mostly cross-dataset -> trivially separable by dataset/batch signature, so the
model learns a DATASET SHORTCUT instead of RNA<->ATAC cell correspondence. It
looks great in-distribution (mixed-pool val R@1 rides the same shortcut) and
collapses once the easy negatives stop carrying it. This mirrors the fix proven
in the pooled trainer (``multiomics_clip_mpnce`` ``same_dataset_blocked``).

The FILIP refiner trainer decouples the FM-forward MICRO-batch (size ``micro_batch``,
run under no_grad) from the CONTRASTIVE EFFECTIVE batch (``micro_batch * accum``,
gathered by concatenating ``accum`` micro-batches). To make the *gathered*
contrastive batch entirely within one dataset -- so every in-batch negative is a
HARD, same-dataset negative -- this sampler yields micro-batch-sized index lists
where each consecutive block of ``accum`` micro-batches is drawn from a SINGLE
randomly chosen ``dataset_id``.

Use as a ``DataLoader(batch_sampler=...)`` (mutually exclusive with
``batch_size`` / ``shuffle`` / ``drop_last``). The training loop's
``for _ in range(accum): next(iter)`` then naturally assembles a same-dataset
effective batch. The sampler yields ``num_blocks`` blocks so the loop never
exhausts it mid-block (which would otherwise splice two datasets together on
re-iteration).
"""
from __future__ import annotations

from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class SameDatasetBlockedBatchSampler(Sampler[List[int]]):
    """Yield micro-batch index lists in same-dataset blocks of ``accum``.

    Args:
        dataset_ids: per-cell dataset label, ALIGNED to ``dataset.__getitem__``
                     order (see ``PairedMultiOmicsDataset.get_dataset_ids``).
        micro_batch: size of each yielded index list (FM-forward micro-batch).
        accum:       number of consecutive micro-batches per same-dataset block;
                     == gradient-accumulation factor == effective_batch/micro_batch.
        num_steps:   number of optimizer steps (blocks) the trainer will take.
        seed:        RNG seed for reproducible dataset/cell selection.
        buffer_blocks: extra blocks beyond ``num_steps`` so the DataLoader iterator
                     never raises StopIteration mid-block during training.
    """

    def __init__(self, dataset_ids: Sequence, micro_batch: int, accum: int,
                 num_steps: int, seed: int = 0, buffer_blocks: int = 16,
                 rank: int = 0, world_size: int = 1):
        """rank/world_size: when world_size > 1 ALL ranks must pass the SAME `seed`. Each block then
        draws ONE dataset (identical on every rank, from the shared RNG) and samples
        `world_size * eff` cells from it, handing rank r the DISJOINT slice [r*eff:(r+1)*eff].
        With cross-rank negative gathering this makes the GLOBAL contrastive batch
        (world_size * eff) entirely same-dataset -- i.e. every gathered negative is HARD.
        Seeding per rank instead (the old behaviour) puts a DIFFERENT dataset on each rank, so
        after gathering half the negatives are cross-dataset = trivially separable = the exact
        dataset shortcut this sampler exists to prevent. world_size=1 is byte-identical to before."""
        ids = np.asarray(dataset_ids)
        unique = np.unique(ids)
        self.indices_by_dataset = {d: np.where(ids == d)[0] for d in unique}
        self.dataset_names = list(unique)
        self.micro_batch = int(micro_batch)
        self.accum = int(accum)
        self.eff = self.micro_batch * self.accum
        self.num_blocks = int(num_steps) + int(buffer_blocks)
        self.seed = seed
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.global_eff = self.eff * self.world_size
        self.n = len(ids)
        n_small = sum(1 for d in unique if len(self.indices_by_dataset[d]) < self.global_eff)
        cells_small = sum(len(self.indices_by_dataset[d]) for d in unique
                          if len(self.indices_by_dataset[d]) < self.global_eff)
        self.frac_replace = cells_small / max(1, len(ids))
        if self.rank == 0:
            print(f"  [blocked_sampler] global_eff={self.global_eff} (eff {self.eff} x world {self.world_size}) | "
                  f"{n_small}/{len(unique)} datasets smaller than that -> {100*self.frac_replace:.1f}% of cells "
                  f"drawn WITH replacement (duplicates act as false negatives)", flush=True)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed)          # SHARED across ranks -> same dataset everywhere
        for _ in range(self.num_blocks):
            d = self.dataset_names[rng.randint(len(self.dataset_names))]
            pool = self.indices_by_dataset[d]
            replace = len(pool) < self.global_eff
            gidx = rng.choice(pool, size=self.global_eff, replace=replace)
            idx = gidx[self.rank * self.eff:(self.rank + 1) * self.eff]   # DISJOINT slice for this rank
            # split this rank's same-dataset effective batch into `accum` micro-batches
            for j in range(self.accum):
                yield idx[j * self.micro_batch:(j + 1) * self.micro_batch].tolist()

    def __len__(self) -> int:
        # total number of micro-batches (index lists) yielded
        return self.num_blocks * self.accum
