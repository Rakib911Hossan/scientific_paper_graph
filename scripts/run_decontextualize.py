#!/usr/bin/env python
"""Stage 3: rewrite every evidence (question, answer) pair from Stage 2 into
standalone, decontextualized atomic claims, with JSONL checkpointing so
reruns skip completed source claims. Usage:

    python scripts/run_decontextualize.py --input data/checkpoints/claim_evidence_checkpoint.jsonl
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_pipeline.config import load_config
from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.resolution.pipeline import run_decontextualization
from claim_pipeline.utils.io import save_json


def load_jsonl(path: str) -> list[dict]:
    import json

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
        "--input", required=True, help="Path to the JSONL evidence checkpoint from run_generate_evidence.py"
    )
    parser.add_argument(
        "--arxiv-id",
        default=None,
        help="arXiv paper id -- the output json goes in a per-paper subfolder (data/processed/<arxiv_id>/atomic_claims.json), the checkpoint jsonl stays flat but namespaced by filename. Ignored if --output/--checkpoint are given.",
    )
    parser.add_argument(
        "--output", default=None, help="Path to write the flat atomic_claims.json list (default: configs/paths.data_processed[/--arxiv-id]/atomic_claims.json)"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the JSONL checkpoint (default: configs/paths.data_checkpoints/atomic_claims_checkpoint.jsonl, namespaced by --arxiv-id if given)",
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

    evidence_records = load_jsonl(args.input)
    checkpoint_stem = f"{args.arxiv_id}_atomic_claims_checkpoint" if args.arxiv_id else "atomic_claims_checkpoint"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else config.data_checkpoints / f"{checkpoint_stem}.jsonl"

    results = run_decontextualization(evidence_records, checkpoint_path, pool, config.model)

    flat_claims = [c["text"] for r in results for c in r["atomic_claims"]]

    for r in results:
        print(f"{r['id']}: {r['source_claim'][:80]}")
        for c in r["atomic_claims"]:
            print(f"  - {c['text']}")
        print()

    print(f"Total atomic claims: {len(flat_claims)}")

    default_dir = config.data_processed / args.arxiv_id if args.arxiv_id else config.data_processed
    output_path = Path(args.output) if args.output else default_dir / "atomic_claims.json"
    save_json(flat_claims, output_path)
    print(f"Saved flat claim list to {output_path}")
    print(f"Checkpoint (with provenance): {checkpoint_path}")


if __name__ == "__main__":
    main()
