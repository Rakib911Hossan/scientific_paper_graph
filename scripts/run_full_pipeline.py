#!/usr/bin/env python
"""Chains all five stages: arXiv id -> final, filtered, checkable claims. Usage:

    python scripts/run_full_pipeline.py --arxiv-id 2503.13507
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claim_pipeline.config import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv-id", required=True, help="arXiv paper id, e.g. 2503.13507")
    args = parser.parse_args()

    config = load_config()
    python = sys.executable

    subprocess.run([python, str(ROOT / "scripts" / "run_ingest.py"), "--arxiv-id", args.arxiv_id], check=True)

    sentences_path = config.data_interim / args.arxiv_id / "sentences.json"
    theses_path = config.data_processed / args.arxiv_id / "theses.json"
    subprocess.run(
        [
            python,
            str(ROOT / "scripts" / "run_generate_theses.py"),
            "--input",
            str(sentences_path),
            "--output",
            str(theses_path),
        ],
        check=True,
    )

    evidence_checkpoint = config.data_checkpoints / f"{args.arxiv_id}_claim_evidence_checkpoint.jsonl"
    subprocess.run(
        [
            python,
            str(ROOT / "scripts" / "run_generate_evidence.py"),
            "--input",
            str(theses_path),
            "--full-sentences",
            str(sentences_path),
            "--checkpoint",
            str(evidence_checkpoint),
        ],
        check=True,
    )

    atomic_claims_checkpoint = config.data_checkpoints / f"{args.arxiv_id}_atomic_claims_checkpoint.jsonl"
    subprocess.run(
        [
            python,
            str(ROOT / "scripts" / "run_decontextualize.py"),
            "--input",
            str(evidence_checkpoint),
            "--checkpoint",
            str(atomic_claims_checkpoint),
            "--output",
            str(config.data_processed / args.arxiv_id / "atomic_claims.json"),
        ],
        check=True,
    )

    subprocess.run(
        [
            python,
            str(ROOT / "scripts" / "run_final_filter.py"),
            "--input",
            str(atomic_claims_checkpoint),
            "--output",
            str(config.data_processed / args.arxiv_id / "final_claims.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
