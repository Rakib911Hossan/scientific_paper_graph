#!/usr/bin/env python
"""Stage 4: keep only the atomic claims worth independently verifying --
drops opinions, hedges/expectations, personal-experience asides, and bare
figure/table references with no checkable content of their own. Usage:

    python scripts/run_final_filter.py --input data/checkpoints/atomic_claims_checkpoint.jsonl
"""
import argparse
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_pipeline.config import load_config
from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.resolution.final_filter import filter_atomic_claims
from claim_pipeline.utils.io import save_json


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Path to the JSONL checkpoint from run_decontextualize.py"
    )
    parser.add_argument(
        "--arxiv-id",
        default=None,
        help="arXiv paper id, used to place default outputs in a per-paper subfolder (data/processed/<arxiv_id>/) so results from different papers don't overwrite each other. Ignored if --output is given.",
    )
    parser.add_argument(
        "--output", default=None, help="Path to write the filtered flat claim list (default: configs/paths.data_processed[/--arxiv-id]/final_claims.json)"
    )
    args = parser.parse_args()

    config = load_config()
    os.environ["CLAIM_PIPELINE_BACKEND"] = config.backend
    if config.backend == "local":
        os.environ.setdefault("CLAIM_PIPELINE_LOCAL_MODEL", config.local_model_id)
        os.environ.setdefault("CLAIM_PIPELINE_LOCAL_DTYPE", config.local_dtype)
        pool = None
    else:
        pool = KeyPool(config.api_keys, config.min_seconds_between_calls)

    source_records = load_jsonl(args.input)
    all_claims = [
        {"text": c["text"], "question": c["question"], "answer": c["answer"], "source_id": r["id"]}
        for r in source_records
        for c in r["atomic_claims"]
    ]

    print(f"Filtering {len(all_claims)} atomic claims...")
    kept = filter_atomic_claims(all_claims, pool, config.model)
    dropped_count = len(all_claims) - len(kept)

    print(f"Kept {len(kept)}, dropped {dropped_count}.\n")
    for c in kept:
        print(f"- {c['text']}")

    default_dir = config.data_processed / args.arxiv_id if args.arxiv_id else config.data_processed
    output_path = Path(args.output) if args.output else default_dir / "final_claims.json"
    save_json(kept, output_path)
    print(f"\nSaved {len(kept)} filtered claims (with provenance) to {output_path}")

    flat_path = output_path.with_name(output_path.stem + "_flat.json")
    save_json([c["text"] for c in kept], flat_path)
    print(f"Saved flat claim list to {flat_path}")


if __name__ == "__main__":
    main()
