#!/usr/bin/env python
"""Local Flask app for reviewing this pipeline's proposition graphs.

Read-only viewer over whatever's already in data/graph/ -- doesn't touch the
LLM pipeline or require GEMINI_API_KEYS, so it works independently of
run_build_graph.py having been run in the same environment. Discovers papers
by scanning for data/graph/<arxiv_id>/proposition_graph_final.json, so a
newly-processed paper shows up automatically without restarting anything
beyond a page refresh.

Usage:
    python graph_review_app/app.py
    -> http://127.0.0.1:5050
"""
import json
from pathlib import Path

from flask import Flask, jsonify, render_template

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_GRAPH = PROJECT_ROOT / "data" / "graph"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"

app = Flask(__name__)


def _final_output_path(arxiv_id: str) -> Path:
    return DATA_GRAPH / arxiv_id / "proposition_graph_final.json"


def _discover_papers() -> list[str]:
    if not DATA_GRAPH.exists():
        return []
    return sorted(
        p.parent.name for p in DATA_GRAPH.glob("*/proposition_graph_final.json")
    )


def _cited_sentence_ids(final_output: dict) -> set[str]:
    ids = set()
    for component in final_output.get("components", []):
        for edge in component.get("edges", []):
            ids.update(edge.get("evidence", []))
    return ids


def _sentence_lookup(arxiv_id: str, ids: set[str]) -> dict[str, str]:
    sentences_path = DATA_INTERIM / arxiv_id / "sentences.json"
    if not ids or not sentences_path.exists():
        return {}
    with open(sentences_path, encoding="utf-8") as f:
        sentences = json.load(f)
    return {s["id"]: s["text"] for s in sentences if s["id"] in ids}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/papers")
def list_papers():
    papers = []
    for arxiv_id in _discover_papers():
        try:
            with open(_final_output_path(arxiv_id), encoding="utf-8") as f:
                final_output = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        node_count = sum(len(c["nodes"]) for c in final_output.get("components", [])) + len(
            final_output.get("isolated_nodes", [])
        )
        edge_count = sum(len(c["edges"]) for c in final_output.get("components", []))
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "node_count": node_count,
                "edge_count": edge_count,
                "component_count": len(final_output.get("components", [])),
                "structural_score": final_output.get("structural_score"),
            }
        )
    return jsonify(papers)


@app.route("/api/graph/<arxiv_id>")
def get_graph(arxiv_id: str):
    path = _final_output_path(arxiv_id)
    if not path.exists():
        return jsonify({"error": f"No proposition_graph_final.json found for {arxiv_id}"}), 404

    with open(path, encoding="utf-8") as f:
        final_output = json.load(f)

    final_output["sentences"] = _sentence_lookup(arxiv_id, _cited_sentence_ids(final_output))
    return jsonify(final_output)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
