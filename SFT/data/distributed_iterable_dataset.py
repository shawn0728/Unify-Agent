# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import random
from .debug_trace import agent_log


# region agent log
agent_log("H8", "data/distributed_iterable_dataset.py:imports", "before import torch")
# endregion
import torch
# region agent log
agent_log("H8", "data/distributed_iterable_dataset.py:imports", "after import torch")
# endregion


class DistributedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, local_rank=0, world_size=1, num_workers=8):
        self.dataset_name = dataset_name
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.rng = random.Random()
        self.data_paths = None
        # Some datasets may already shard data by global rank before set_epoch().
        # In that case we should only shuffle locally, and skip rank-level split.
        self.data_already_sharded = False

    def get_data_paths(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _to_sortable(value):
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, dict):
            if "ip_index" in value:
                return value["ip_index"]
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        if isinstance(value, (list, tuple)):
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        return str(value)

    def set_epoch(self, seed=42):
        if self.data_paths is None:
            return
        if len(self.data_paths) == 0:
            return

        if isinstance(self.data_paths[0], tuple):
            data_paths = sorted(
                self.data_paths,
                key=lambda x: (
                    self._to_sortable(x[0]) if len(x) > 0 else "",
                    self._to_sortable(x[1]) if len(x) > 1 else "",
                ),
            )
        elif isinstance(self.data_paths[0], str):
            data_paths = sorted(self.data_paths)
        else:
            raise ValueError(f"Unknown data_paths type: {type(self.data_paths[0])}")

        self.rng.seed(seed)
        self.rng.shuffle(data_paths)

        split_world_size = 1 if self.data_already_sharded else self.world_size
        split_rank = 0 if self.data_already_sharded else self.local_rank

        num_files_per_rank = len(data_paths) // split_world_size
        local_start = split_rank * num_files_per_rank
        local_end = (split_rank + 1) * num_files_per_rank
        self.num_files_per_rank = num_files_per_rank
        self.data_paths_per_rank = data_paths[local_start:local_end]

    def get_data_paths_per_worker(self):
        if self.data_paths is None:
            return None

        info = torch.utils.data.get_worker_info()
        if info is None:
            # Single worker: Use all files assigned to the rank
            return self.data_paths_per_rank, 0

        worker_id = info.id
        num_files_per_worker = self.num_files_per_rank // info.num_workers
        start = num_files_per_worker * worker_id
        end = num_files_per_worker * (worker_id + 1)
        data_paths_per_worker = self.data_paths_per_rank[start:end]

        return data_paths_per_worker[::-1], worker_id

    def __iter__(self):
        raise NotImplementedError
