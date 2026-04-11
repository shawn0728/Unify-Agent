# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight structured logging for the data pipeline.

Provides a ``get_logger`` factory that returns a stdlib ``logging.Logger``
with a consistent format across all pipeline stages.  Usage::

    from data_pipeline._logging import get_logger
    log = get_logger("stage1")
    log.info("Loaded %d IPs", n)
"""

import logging
import sys

_FMT = "[%(asctime)s] [%(name)s] %(levelname)s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root():
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the pipeline-wide format."""
    _configure_root()
    return logging.getLogger(name)
