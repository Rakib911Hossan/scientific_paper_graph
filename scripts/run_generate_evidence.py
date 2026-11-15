#!/usr/bin/env python
"""Stage 2: generate supporting evidence Q&A for each thesis claim (replaces
the old Perceptor classify -> window/BM25 Q&A -> rewrite chain), with JSONL
checkpointing so reruns skip completed claims. Usage:

    python scripts/run_generate_evidence.py \\
        --input data/processed/theses.json \\
        --full-sentences data/interim/2503.13507_sentences.json
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_pipeline.config import load_config
from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.resolution.pipeline import run_pipeline
from claim_pipeline.utils.io import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to theses.json from run_generate_theses.py")
    parser.add_argument(
        "--full-sentences",
        required=True,
        help="Path to the full sentence pool (*_sentences.json from run_ingest.py), used as the BM25 corpus.",
    )
    parser.add_argument(
        "--arxiv-id",
        default=None,
        help="arXiv paper id, used to namespace the default checkpoint filename so results from different papers don't overwrite each other. Ignored if --checkpoint is given.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the JSONL checkpoint (default: configs/paths.data_checkpoints/claim_evidence_checkpoint.jsonl, namespaced by --arxiv-id if given)",
    )
    parser.add_argument(
        "--doc-summary",
        default=None,
        help=(
            "Path to a text file with Stage 1's whole-document summary. Prepended to every "
            "answering call. Defaults to <input-stem>_doc_summary.txt next to --input (what "
            "run_generate_theses.py writes); proceeds without a summary if that file doesn't exist."
        ),
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

    theses = load_json(args.input)
    full_sentences = load_json(args.full_sentences)
    checkpoint_stem = f"{args.arxiv_id}_claim_evidence_checkpoint" if args.arxiv_id else "claim_evidence_checkpoint"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else config.data_checkpoints / f"{checkpoint_stem}.jsonl"

    input_path = Path(args.input)
    doc_summary_path = Path(args.doc_summary) if args.doc_summary else input_path.with_name(input_path.stem + "_doc_summary.txt")
    if doc_summary_path.exists():
        doc_summary = doc_summary_path.read_text(encoding="utf-8").strip()
        print(f"Using document summary from {doc_summary_path}")
    else:
        doc_summary = ""
        print(f"No document summary found at {doc_summary_path}, proceeding without one.")

    results = run_pipeline(
        theses,
        full_sentences,
        checkpoint_path,
        pool,
        config.model,
        doc_summary,
        n_questions=config.evidence_max_questions_per_claim,
        candidate_top_k=config.evidence_candidate_top_k,
        bm25_top_k=config.bm25_top_k,
    )

    for r in results:
        print(f"{r['id']}: {r['claim']}")
        for i, qa in enumerate(r["qa_pairs"], 1):
            print(f"  Q{i}: {qa['question']}")
            print(f"  A{i}: {qa['answer']}")
        print()

    print(f"Processed {len(results)} claims. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
