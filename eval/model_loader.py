# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Thin shim that proxies :mod:`model_loader` to the canonical implementation
shipped under ``infer/``.

The eval scripts (``bagel_infer.py``, ``bagel_infer_batch.py``) prepend their
own directory to ``sys.path`` and then ``from model_loader import
load_model_and_inferencer``. To avoid duplicating ~200 lines of loader logic,
this module delegates to ``infer/model_loader.py`` after exposing the
``infer/`` directory on ``sys.path``.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_INFER_DIR = os.path.join(_REPO_ROOT, "infer")

if not os.path.isdir(_INFER_DIR):
    raise ImportError(
        f"Expected sibling 'infer/' directory at {_INFER_DIR}; cannot locate "
        "the canonical model_loader implementation."
    )

if _INFER_DIR not in sys.path:
    sys.path.insert(0, _INFER_DIR)

# Load the real loader by absolute path under a private module name so it does
# not collide with this shim in ``sys.modules``.
_real_path = os.path.join(_INFER_DIR, "model_loader.py")
_spec = importlib.util.spec_from_file_location("_unify_agent_model_loader", _real_path)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    raise ImportError(f"Failed to load {_real_path}")
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)

load_model_and_inferencer = _real.load_model_and_inferencer

__all__ = ["load_model_and_inferencer"]
