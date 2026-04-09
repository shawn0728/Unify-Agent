# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import importlib
from ..debug_trace import agent_log


# region agent log
agent_log("H6", "data/interleave_datasets/__init__.py", "before import edit_dataset")
# endregion
_edit = importlib.import_module("data.interleave_datasets.edit_dataset")
UnifiedEditIterableDataset = _edit.UnifiedEditIterableDataset
# region agent log
agent_log("H6", "data/interleave_datasets/__init__.py", "after import edit_dataset")
# endregion

