#!/usr/bin/env python
"""Drops claims that restate the same underlying fact as another claim in
the set -- run this on final_claims.json before Stage 2.1 graph
construction, since claims decontextualized from different QA pairs can
independently restate the same fact. Usage:

    python scripts/run_dedup.py --input data/processed/final_claims.json
"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_pipeline.config import load_config
from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.resolution.dedup import dedup_atomic_claims
from claim_pipeline.utils.io import load_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Path to final_claims.json (Stage 4's output, with provenance)"
    )
    parser.add_argument(
        "--arxiv-id",
        default=None,
        help="arXiv paper id, used to place default outputs in a per-paper subfolder (data/processed/<arxiv_id>/) so results from different papers don't overwrite each other. Ignored if --output is given.",
    )
    parser.add_argument(
        "--output", default=None, help="Path to write the deduped claim list (default: configs/paths.data_processed[/--arxiv-id]/final_claims_deduped.json)"
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

    claims = load_json(args.input)

    print(f"Deduping {len(claims)} claims...")
    kept, duplicate_groups = dedup_atomic_claims(claims, pool, config.model)
    dropped_count = len(claims) - len(kept)

    print(f"Kept {len(kept)}, dropped {dropped_count} duplicates.\n")
    if duplicate_groups:
        print("Duplicate groups found:")
        for group in duplicate_groups:
            print(f"  indices={group['indices']} keep={group['keep']}: {group.get('reason', '(no reason given)')}")
        print()
    for c in kept:
        print(f"- {c['text']}")

    default_dir = config.data_processed / args.arxiv_id if args.arxiv_id else config.data_processed
    output_path = Path(args.output) if args.output else default_dir / "final_claims_deduped.json"
    save_json(kept, output_path)
    print(f"\nSaved {len(kept)} deduped claims (with provenance) to {output_path}")

    flat_path = output_path.with_name(output_path.stem + "_flat.json")
    save_json([c["text"] for c in kept], flat_path)
    print(f"Saved flat claim list to {flat_path}")


if __name__ == "__main__":
    main()
