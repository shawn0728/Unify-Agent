# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from .debug_trace import agent_log


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
    'umm_sft': {},
}

DATASET_CLASS_PATHS = {
    "t2i_pretrain": ("data.t2i_dataset", "T2IIterableDataset"),
    "vlm_sft": ("data.vlm_dataset", "SftJSONLIterableDataset"),
    "unified_edit": ("data.interleave_datasets", "UnifiedEditIterableDataset"),
    "umm_sft": ("data.umm_sft_dataset", "SftAgenticIterableDataset"),
}


def get_dataset_class(dataset_name):
    if dataset_name not in DATASET_CLASS_PATHS:
        raise KeyError(f"Unknown dataset name: {dataset_name}")
    module_name, class_name = DATASET_CLASS_PATHS[dataset_name]
    # region agent log
    agent_log(
        "H10",
        "data/dataset_info.py:get_dataset_class",
        "lazy import dataset class",
        {"dataset_name": dataset_name, "module": module_name, "class": class_name},
    )
    # endregion
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _build_umm_sft_info(output_root):
    """
    Build umm_sft metainfo from output_root/category/{traj,intermediate,images}.
    """
    root = Path(output_root)
    # region agent log
    agent_log("H1", "data/dataset_info.py:_build_umm_sft_info", "start build umm_sft info", {"output_root": output_root})
    # endregion
    if not root.exists():
        return {}

    dataset_meta = {}
    for category_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        # region agent log
        agent_log("H1", "data/dataset_info.py:_build_umm_sft_info", "scan category", {"category": category_dir.name})
        # endregion
        traj_dir = category_dir / 'traj'
        reference_dir = category_dir / 'intermediate'
        generation_dir = category_dir / 'images'
        if not (traj_dir.is_dir() and reference_dir.is_dir() and generation_dir.is_dir()):
            continue

        # Avoid expensive full-directory counting during module import.
        # We only need to know whether this category has at least one sample.
        has_sample = next(traj_dir.glob('*_trajectory.json'), None) is not None
        if not has_sample:
            continue

        dataset_meta[category_dir.name] = {
            # Required by dataset_base.build_datasets.
            'data_dir': str(traj_dir),
            'reference_dir': str(reference_dir),
            'generation_dir': str(generation_dir),
            # For umm_sft, this can be a directory of *_trajectory.json files.
            'json_path': str(traj_dir),
            # Keep a placeholder to preserve schema; exact size is not required
            # for umm_sft loading in this project path.
            'num_total_samples': 1,
        }

    # region agent log
    agent_log("H1", "data/dataset_info.py:_build_umm_sft_info", "finish build umm_sft info", {"category_count": len(dataset_meta)})
    # endregion
    return dataset_meta


_UMM_SFT_OUTPUT_ROOT = os.environ.get("UMM_SFT_OUTPUT_ROOT", "")
if _UMM_SFT_OUTPUT_ROOT:
    DATASET_INFO['umm_sft'].update(_build_umm_sft_info(_UMM_SFT_OUTPUT_ROOT))