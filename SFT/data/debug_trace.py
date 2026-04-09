# Copyright 2026

import importlib
import json
import os
import time


DEBUG_LOG_PATH = os.environ.get("SFT_DEBUG_LOG", "")


def agent_log(hypothesis_id: str, location: str, message: str, data=None):
    payload = {
        "id": f"log_{int(time.time() * 1000)}_{os.getpid()}",
        "timestamp": int(time.time() * 1000),
        "runId": "import-trace",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
    }
    if not DEBUG_LOG_PATH:
        return
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def import_with_log(
    module_name: str,
    hypothesis_id: str,
    location: str,
    before_message: str,
    after_message: str | None = None,
):
    # region agent log
    agent_log(hypothesis_id, location, before_message)
    # endregion
    module = importlib.import_module(module_name)
    if after_message is not None:
        # region agent log
        agent_log(hypothesis_id, location, after_message)
        # endregion
    return module

