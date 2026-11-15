"""Append-only token usage log, written from the main pipeline environment.

Deliberately has no litellm/pricing dependency -- this module only records
what each call actually consumed (model, stage, token counts). Turning that
into a dollar figure happens later, out of process, in the isolated
.venv-costcalc environment (see scripts/compute_pipeline_cost.py), because
litellm's own dependency on openai>=2.0 conflicts with marker-pdf's
openai<2.0 pin used elsewhere in this same environment.
"""

import json
import os
import time
from pathlib import Path

DEFAULT_USAGE_LOG_PATH = Path("data/usage_log.jsonl")


def usage_log_path() -> Path:
    return Path(os.environ.get("USAGE_LOG_PATH", DEFAULT_USAGE_LOG_PATH))


def log_usage(model: str, stage: str, prompt_tokens: int, completion_tokens: int) -> None:
    path = usage_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "model": model,
        "stage": stage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
