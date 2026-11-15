#!/usr/bin/env python
"""Stage 2.1: structured-output call that proposes a dependency graph over
the final atomic claims, each edge grounded in cited source sentence ids.
A single generation call proposes the full edge list directly (majority
voting across repeated generations was removed -- see CLAUDE.md for why).
Usage:

    python scripts/run_build_graph.py --input data/processed/final_claims_deduped.json \\
        --full-sentences data/interim/2503.13507_sentences.json \\
        --doc-summary data/processed/theses_doc_summary.txt
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claim_pipeline.config import load_config
from claim_pipeline.graph.build_graph import generate_proposition_graph, repair_cycles, to_flat_graph
from claim_pipeline.graph.final_output import build_final_output
from claim_pipeline.graph.quality import run_quality_check
from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.utils.io import load_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Path to final_claims.json (Stage 4's output, with provenance)"
    )
    parser.add_argument(
        "--full-sentences", required=True, help="Path to the *_sentences.json full sentence pool from run_ingest.py"
    )
    parser.add_argument(
        "--doc-summary", required=True, help="Path to the whole-document summary text file from Stage 1"
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=10,
        help="BM25 evidence sentences shown per proposition (default 10). Every proposition is shown in the "
        "same prompt regardless of this value -- lowering it shrinks the overall prompt without dropping any "
        "proposition from consideration, unlike relying on llm/local_model.py's input-token truncation backstop "
        "(which truncates from the front, silently excluding whichever propositions land there from ever "
        "getting an edge). Worth lowering for the local backend on a paper with many claims.",
    )
    parser.add_argument(
        "--arxiv-id",
        default=None,
        help="arXiv paper id, used to place default outputs in a per-paper subfolder (data/graph/<arxiv_id>/) so results from different papers don't overwrite each other. Ignored if --output is given.",
    )
    parser.add_argument(
        "--output", default=None, help="Path to write the final proposition graph (default: configs/paths.data_graph[/--arxiv-id]/proposition_graph.json)"
    )
    parser.add_argument(
        "--skip-quality-check",
        action="store_true",
        help="Skip Stage 2.5, the LLM-judge structural fidelity check (Node/Edge/Structural Accuracy) that "
        "otherwise always runs after cycle repair and saves a report to <output>_quality.json. Adds one LLM "
        "call per edge plus two batched calls, on top of graph generation/repair -- only skip "
        "this under real rate-limit/cost pressure, not by default.",
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
    full_sentences = load_json(args.full_sentences)
    doc_summary = Path(args.doc_summary).read_text(encoding="utf-8").strip()

    default_dir = config.data_graph / args.arxiv_id if args.arxiv_id else config.data_graph
    output_path = Path(args.output) if args.output else default_dir / "proposition_graph.json"

    print("Generating proposition graph...")
    raw_graph = generate_proposition_graph(
        claims, full_sentences, doc_summary, pool, config.model,
        candidate_top_k=args.candidate_top_k,
    )
    graph = to_flat_graph(raw_graph)
    print(f"  {len(graph.get('Nodes', []))} nodes, {len(graph.get('Edges', []))} edge entries.")

    print("Checking graph for cycles...")
    graph = repair_cycles(graph, pool, config.model)
    if graph.get("RemovedCycleEdges"):
        print(f"  Removed {len(graph['RemovedCycleEdges'])} edge(s) to break cycle(s):")
        for edge in graph["RemovedCycleEdges"]:
            print(f"    {edge['from']} -> {edge['to']}: {edge['reason']}")
    else:
        print("  No cycles found.")

    print(f"Final graph: {len(graph.get('Nodes', []))} nodes, {len(graph.get('Edges', []))} edge entries.")
    save_json(graph, output_path)
    print(f"Saved proposition graph to {output_path}")

    quality_report = None
    if not args.skip_quality_check:
        print("\nRunning structural fidelity check (Stage 2.5)...")
        quality_report = run_quality_check(graph, claims, full_sentences, doc_summary, pool, config.model)
        flagged = quality_report["Flagged"]
        print(
            f"  Node Accuracy: {len(flagged['nodes'])}/{len(quality_report['NodeAccuracy'])} node(s) below threshold."
        )
        print(
            f"  Edge Accuracy: {len(flagged['edges'])}/{len(quality_report['EdgeAccuracy'])} edge(s) flagged."
        )
        print(f"  Structural Fidelity: score={quality_report['StructuralFidelity'].get('score')}"
              f"{' (flagged)' if flagged['structural'] else ''}")
        quality_path = output_path.with_name(output_path.stem + "_quality.json")
        save_json(quality_report, quality_path)
        print(f"  Saved quality report to {quality_path}")

    final_output = build_final_output(graph, args.arxiv_id, quality_report)
    final_path = output_path.with_name(output_path.stem + "_final.json")
    save_json(final_output, final_path)
    print(f"\nSaved final output ({len(final_output['components'])} component(s), "
          f"{len(final_output['isolated_nodes'])} isolated node(s), "
          f"{'with' if quality_report else 'without'} fidelity scores) to {final_path}")


if __name__ == "__main__":
    main()
