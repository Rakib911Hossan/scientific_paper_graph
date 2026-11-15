#!/usr/bin/env python
"""Computes real dollar cost from the pipeline's token usage log.

Must be run with the ISOLATED cost-calc environment's interpreter, not the
main pipeline's -- litellm (used here for its pricing table) depends on
openai>=2.0, which conflicts with marker-pdf's openai<2.0 pin used elsewhere
in this codebase. See CLAUDE.md / .venv-costcalc/ for why these are kept in
separate environments. This script itself has no dependency on the
claim_pipeline package (no google-genai, no marker-pdf) so it can run
standalone in that isolated venv.

Usage:

    .venv-costcalc/bin/python scripts/compute_pipeline_cost.py \
        --usage-log data/usage_log.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from litellm import cost_per_token


def _litellm_model_name(model: str) -> str:
    """The usage log stores the bare model string passed to google-genai
    (e.g. "gemini-3.5-flash"); litellm's pricing table keys Gemini entries
    under the "gemini/" provider prefix."""
    return model if "/" in model else f"gemini/{model}"


def load_usage(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_cost(records: list[dict]) -> dict:
    per_stage = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "cost": 0.0, "first_ts": None, "last_ts": None}
    )
    unpriced_models = set()
    total_cost = 0.0
    all_ts = []

    for r in records:
        litellm_model = _litellm_model_name(r["model"])
        try:
            in_cost, out_cost = cost_per_token(
                model=litellm_model,
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
            )
        except Exception:
            unpriced_models.add(litellm_model)
            in_cost = out_cost = 0.0

        call_cost = in_cost + out_cost
        stage = per_stage[r["stage"]]
        stage["prompt_tokens"] += r["prompt_tokens"]
        stage["completion_tokens"] += r["completion_tokens"]
        stage["calls"] += 1
        stage["cost"] += call_cost
        total_cost += call_cost

        ts = r.get("ts")
        if ts is not None:
            all_ts.append(ts)
            if stage["first_ts"] is None or ts < stage["first_ts"]:
                stage["first_ts"] = ts
            if stage["last_ts"] is None or ts > stage["last_ts"]:
                stage["last_ts"] = ts

    for stage in per_stage.values():
        if stage["first_ts"] is not None and stage["last_ts"] is not None:
            stage["duration_seconds"] = round(stage["last_ts"] - stage["first_ts"], 1)
        else:
            stage["duration_seconds"] = None

    total_duration_seconds = round(max(all_ts) - min(all_ts), 1) if all_ts else None

    return {
        "total_cost_usd": total_cost,
        "total_duration_seconds": total_duration_seconds,
        "by_stage": dict(per_stage),
        "unpriced_models": sorted(unpriced_models),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usage-log", default="data/usage_log.jsonl", help="Path to the jsonl usage log written by generate_with_retry()")
    parser.add_argument("--output", default=None, help="Optional path to also write the cost report as JSON")
    args = parser.parse_args()

    usage_log_path = Path(args.usage_log)
    if not usage_log_path.exists():
        raise SystemExit(f"No usage log found at {usage_log_path} -- run the pipeline first.")

    records = load_usage(usage_log_path)
    report = compute_cost(records)

    total_duration = report["total_duration_seconds"]
    duration_str = f"{total_duration:.1f}s" if total_duration is not None else "n/a"
    print(f"Total cost: ${report['total_cost_usd']:.4f}   Total wall-clock span: {duration_str}\n")
    print(f"{'stage':<28} {'calls':>6} {'prompt_tok':>12} {'completion_tok':>15} {'cost':>10} {'duration':>10}")
    for stage, s in sorted(report["by_stage"].items(), key=lambda kv: -kv[1]["cost"]):
        dur = f"{s['duration_seconds']:.1f}s" if s["duration_seconds"] is not None else "n/a"
        print(f"{stage:<28} {s['calls']:>6} {s['prompt_tokens']:>12} {s['completion_tokens']:>15} ${s['cost']:>9.4f} {dur:>10}")

    if report["unpriced_models"]:
        print(f"\nWARNING: no litellm pricing found for: {', '.join(report['unpriced_models'])} -- those calls are costed as $0 above.")

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
