#!/usr/bin/env python
"""Stage 0: fetch an arXiv paper, convert to Markdown, clean, and split into
sentences. Usage:

    python scripts/run_ingest.py --arxiv-id 2503.13507
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_pipeline.config import load_config
from claim_pipeline.ingest.fetch_arxiv import fetch_arxiv_pdf
from claim_pipeline.ingest.pdf_to_markdown import convert_pdf_to_markdown
from claim_pipeline.preprocess.clean_markdown import clean_markdown, remove_references_section
from claim_pipeline.preprocess.sentence_split import split_into_sentences_with_ids
from claim_pipeline.utils.io import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv-id", required=True, help="arXiv paper id, e.g. 2503.13507")
    args = parser.parse_args()

    config = load_config()

    print(f"Fetching {args.arxiv_id} from arXiv...")
    pdf_path = fetch_arxiv_pdf(args.arxiv_id, config.data_raw)

    print("Converting PDF to Markdown (marker)...")
    markdown_text = convert_pdf_to_markdown(pdf_path, config.data_interim)

    print("Cleaning Markdown and splitting into sentences...")
    cleaned = clean_markdown(markdown_text)
    cleaned = remove_references_section(cleaned)
    sentences = split_into_sentences_with_ids(cleaned)

    out_path = config.data_interim / args.arxiv_id / "sentences.json"
    save_json(sentences, out_path)

    print(f"Total sentences after cleaning: {len(sentences)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
