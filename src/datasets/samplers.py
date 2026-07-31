"""Batch samplers for variable-size multi-condition point sets."""

from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class ConditionBatchSampler(Sampler[List[int]]):
    """Keep each batch within one condition so point counts may differ."""

    def __init__(self, dataset, batch_size: int, shuffle: bool):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.groups = {}
        for global_idx, (condition_id, _local_idx) in enumerate(dataset._index_map):
            self.groups.setdefault(condition_id, []).append(global_idx)

    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        for indices in self.groups.values():
            if self.shuffle:
                order = torch.randperm(len(indices)).tolist()
                ordered = [indices[i] for i in order]
            else:
                ordered = indices
            batches.extend(
                ordered[start:start + self.batch_size]
                for start in range(0, len(ordered), self.batch_size)
            )

        if self.shuffle and batches:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        )
