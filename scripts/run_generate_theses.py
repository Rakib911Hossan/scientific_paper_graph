#!/usr/bin/env python
"""Stage 1: synthesize up to N distinct, high-level thesis claims the paper
supports (replaces the old salience-ranking + dedup stages). Usage:

    python scripts/run_generate_theses.py --input data/interim/2503.13507_sentences.json \\
        --output data/processed/theses.json
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_pipeline.config import load_config
from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.llm.gemini import call_with_retry
from claim_pipeline.synthesis.summarize import generate_document_summary
from claim_pipeline.synthesis.thesis import generate_thesis_claims
from claim_pipeline.utils.io import load_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to *_sentences.json from run_ingest.py")
    parser.add_argument(
        "--arxiv-id",
        default=None,
        help="arXiv paper id, used to place the default output in a per-paper subfolder (e.g. data/processed/2503.13507/theses.json) so results from different papers don't overwrite each other. Ignored if --output is given.",
    )
    parser.add_argument("--output", default=None, help="Path to write theses.json (default: configs/paths.data_processed[/--arxiv-id]/theses.json)")
    args = parser.parse_args()

    config = load_config()
    os.environ["CLAIM_PIPELINE_BACKEND"] = config.backend
    if config.backend == "local":
        os.environ.setdefault("CLAIM_PIPELINE_LOCAL_MODEL", config.local_model_id)
        os.environ.setdefault("CLAIM_PIPELINE_LOCAL_DTYPE", config.local_dtype)
        pool = None
    else:
        pool = KeyPool(config.api_keys, config.min_seconds_between_calls)

    sentences = load_json(args.input)

    print("Generating document summary...")
    doc_summary = call_with_retry(
        generate_document_summary,
        sentences,
        pool,
        config.model,
        chunk_size=config.chunk_size_sentences,
        max_retries=config.max_retries,
        base_delay=config.base_delay,
    )
    print("Document summary:\n", doc_summary, "\n")

    print(f"Synthesizing up to {config.max_claims} thesis claims...")
    claims = call_with_retry(
        generate_thesis_claims,
        sentences,
        doc_summary,
        pool,
        config.model,
        max_claims=config.max_claims,
        max_retries=config.max_retries,
        base_delay=config.base_delay,
        chunk_size=config.chunk_size_sentences,
    )

    print(f"Generated {len(claims)} claims:\n")
    for i, c in enumerate(claims, 1):
        print(f"{i}. {c}\n")

    default_dir = config.data_processed / args.arxiv_id if args.arxiv_id else config.data_processed
    output_path = Path(args.output) if args.output else default_dir / "theses.json"
    save_json(claims, output_path)
    print(f"Saved to {output_path}")

    summary_path = output_path.with_name(output_path.stem + "_doc_summary.txt")
    summary_path.write_text(doc_summary, encoding="utf-8")
    print(f"Saved document summary to {summary_path}")


if __name__ == "__main__":
    main()
