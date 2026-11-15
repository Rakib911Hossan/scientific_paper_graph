"""Stage 2.5 (optional): structural fidelity check over the final proposition
graph, adapted from CDCR-SFT Appendix B.4's LLM-judge rubric (Node/Edge/
Structural Accuracy, 0-10 each). CDCR-SFT scores a generated graph against a
known-correct reference graph; there is no such reference here (the graph is
synthesized from a real paper, not a synthetic puzzle with a gold answer), so
each dimension is redefined reference-free, checked against the paper's own
source material instead of a ground truth:

- Node Accuracy: does each node's "description" still faithfully represent
  the original source claim it was built from. No longer an LLM call --
  build_graph.generate_proposition_graph() builds "description" directly
  from the input claim text rather than asking the model to regenerate it
  (see that function's docstring for the local-backend bug this fixed), so
  "description" is always exactly the source claim verbatim and there is
  nothing left for an LLM judge to usefully catch here. score_node_accuracy()
  short-circuits to a trivial all-match report instead of spending a real
  call confirming something true by construction.
- Edge Accuracy: does each edge's cited evidence support a genuine
  dependency in the SPECIFIC claimed direction, not just topical overlap --
  the one real defect found across manual review of these graphs was a
  direction flip (P11<->P12/13, both directions citing identical evidence),
  never a fabricated or missing-evidence edge, so direction gets its own
  explicit question rather than being folded into a general "is evidence
  sufficient" score. Scored per-edge (one call each) since a batched score
  would average over exactly this kind of isolated, fixable defect.
- Structural Fidelity: does the overall edge shape plausibly track the
  paper's argument structure per the doc summary, scored in one batched
  call. Weakest of the three by construction -- a short doc summary
  necessarily compresses away the fine-grained causal chains (e.g. a
  multi-hop overfitting chain) this is meant to catch, so a low score here
  is lower-confidence than a low Node/Edge score.

Record-only: this module never mutates or drops anything from the graph. It
returns a report for a human (or a future automated step) to act on.
"""

import json
import time

from google.genai import types

from claim_pipeline.graph.prompts import (
    EDGE_ACCURACY_TEMPLATE,
    STRUCTURAL_FIDELITY_TEMPLATE,
)
from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.llm.router import generate_with_retry


def _generate_json(pool: KeyPool, model: str, prompt: str, max_retries: int = 2) -> dict:
    last_error = None
    for attempt in range(max_retries):
        temperature = 0.0 if attempt == 0 else 0.3
        response_text = generate_with_retry(
            pool,
            model,
            prompt,
            generation_config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature,
            ),
        )
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    raise last_error


def score_node_accuracy(graph: dict, claims: list[dict], pool: KeyPool, model: str) -> list[dict]:
    """No LLM call: graph.build_graph.generate_proposition_graph() builds
    each node's "description" directly from claims[i]["text"] rather than
    letting the model regenerate it, so description == source claim
    verbatim by construction, always. Returns a trivial all-10 report in
    the same {"id", "score", "reasoning"} shape the rest of quality.py and
    graph.final_output expect, without spending a real call to confirm
    something that can't be false. claims: the original list passed into
    generate_proposition_graph() (Stage 4's deduped output) -- claims[i]
    lines up with Nodes[i] since node ids are assigned P1..Pn in that same
    input order."""
    nodes = graph.get("Nodes", [])
    return [
        {"id": node["id"], "score": 10, "reasoning": "description is the verbatim source claim text (not model-regenerated)"}
        for node in nodes
    ]


def score_edge_accuracy(
    graph: dict,
    full_sentences: list[dict],
    pool: KeyPool,
    model: str,
) -> list[dict]:
    """One call per edge. Returns a list of
    {"from", "to", "score", "direction_supported", "reasoning", "skipped"}
    -- "skipped" is always False here (kept as a field so downstream
    consumers like graph.final_output don't need a schema change), since
    there is no vote/confidence tier to exempt an edge from scoring now
    that majority voting across repeated generations was removed.

    Each edge's call is wrapped individually against two distinct failure
    modes verified concretely in practice, not just in theory: (1) a
    malformed/truncated JSON response that survives _generate_json()'s own
    retry-at-different-temperature (see quality.py's module docstring and
    CLAUDE.md's "deterministic malformed JSON" note), and (2)
    generate_with_retry() itself exhausting its retries on repeated
    transient API errors (RuntimeError) during a flaky stretch of network
    conditions. Either way, only that one edge is marked errored rather
    than aborting the whole batch -- the same per-item isolation fix
    applied to decontextualize_claim_evidence() for the JSON-truncation
    case specifically.
    """
    node_lookup = {node["id"]: node for node in graph.get("Nodes", [])}
    sentence_lookup = {s["id"]: s["text"] for s in full_sentences}

    results = []
    for edge in graph.get("Edges", []):
        evidence_ids = edge.get("evidence", [])
        evidence_block = "\n".join(f"  [{sid}] {sentence_lookup.get(sid, '(missing sentence)')}" for sid in evidence_ids) or "  (none cited)"
        prompt = EDGE_ACCURACY_TEMPLATE.format(
            from_id=edge["from"],
            from_description=node_lookup.get(edge["from"], {}).get("description", ""),
            to_id=edge["to"],
            to_description=node_lookup.get(edge["to"], {}).get("description", ""),
            evidence_block=evidence_block,
        )
        try:
            decision = _generate_json(pool, model, prompt)
        except (json.JSONDecodeError, RuntimeError) as exc:
            results.append(
                {
                    "from": edge["from"],
                    "to": edge["to"],
                    "skipped": False,
                    "score": None,
                    "direction_supported": None,
                    "reasoning": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        results.append(
            {
                "from": edge["from"],
                "to": edge["to"],
                "skipped": False,
                "score": decision.get("score"),
                "direction_supported": decision.get("direction_supported"),
                "reasoning": decision.get("reasoning", ""),
            }
        )
    return results


def score_structural_fidelity(
    graph: dict, doc_summary: str, edge_scores: list[dict], pool: KeyPool, model: str, edge_threshold: int = 6
) -> dict:
    """One batched call judging the graph's overall shape against the doc
    summary as a proxy for the paper's argument structure. edge_scores:
    score_edge_accuracy()'s output, run first and passed in here so this
    check can't praise an edge Edge Accuracy just rejected -- verified
    concretely that without this, the two checks contradicted each other
    in the same report (Structural Fidelity scored 10/10 citing P14->P37/
    P38 and P24/P27->P35 as evidence of good structure, the exact edges
    Edge Accuracy had scored 0/direction-unsupported two sections earlier)."""
    node_block = "\n".join(f"{n['id']}: {n.get('description', '')}" for n in graph.get("Nodes", []))
    edge_block = "\n".join(f"{e['from']} -> {e['to']}" for e in graph.get("Edges", [])) or "(no edges)"

    rejected = [
        e for e in edge_scores
        if not e.get("skipped") and not e.get("error") and (
            (e.get("score") is not None and e["score"] < edge_threshold) or e.get("direction_supported") is False
        )
    ]
    rejected_edges_block = "\n".join(f"{e['from']} -> {e['to']}: {e.get('reasoning', '')}" for e in rejected) or "(none -- no edges were rejected by the evidence check)"

    prompt = STRUCTURAL_FIDELITY_TEMPLATE.format(
        doc_summary=doc_summary, node_block=node_block, edge_block=edge_block, rejected_edges_block=rejected_edges_block
    )
    return _generate_json(pool, model, prompt)


def run_quality_check(
    graph: dict,
    claims: list[dict],
    full_sentences: list[dict],
    doc_summary: str,
    pool: KeyPool,
    model: str,
    node_threshold: int = 6,
    edge_threshold: int = 6,
    structural_threshold: int = 6,
) -> dict:
    """Orchestrates all three checks and flags anything below its threshold
    for manual review. Does not modify graph or raise on low scores --
    record-only, per the "optional... flag for manual review" framing of
    this step; auto-regeneration on a soft threshold is a separate,
    heavier decision left to a human.

    Each of the three checks is isolated: a JSON failure or exhausted-retry
    RuntimeError in one (e.g. the batched Structural Fidelity call) doesn't
    discard Node/Edge Accuracy results already obtained -- same
    per-item-isolation reasoning as score_edge_accuracy(), one level up.

    Edge Accuracy runs before Structural Fidelity (not just before Node
    Accuracy in the code) specifically so its results can be fed forward --
    see score_structural_fidelity()'s docstring for why that ordering is
    load-bearing, not incidental.
    """
    try:
        node_scores = score_node_accuracy(graph, claims, pool, model)
    except (json.JSONDecodeError, RuntimeError) as exc:
        node_scores = [{"error": f"Node Accuracy check failed: {type(exc).__name__}: {exc}"}]

    edge_scores = score_edge_accuracy(graph, full_sentences, pool, model)

    try:
        structural = score_structural_fidelity(graph, doc_summary, edge_scores, pool, model, edge_threshold=edge_threshold)
    except (json.JSONDecodeError, RuntimeError) as exc:
        structural = {"score": None, "reasoning": f"error: {type(exc).__name__}: {exc}"}

    flagged_nodes = [n for n in node_scores if n.get("score") is not None and n["score"] < node_threshold]
    structural_flagged = structural.get("score") is not None and structural["score"] < structural_threshold

    return {
        "NodeAccuracy": node_scores,
        "EdgeAccuracy": edge_scores,
        "StructuralFidelity": structural,
        "Flagged": {
            "nodes": flagged_nodes,
            "edges": _flagged_edges(edge_scores, edge_threshold),
            "structural": structural_flagged,
        },
    }


def _flagged_edges(edge_scores: list[dict], edge_threshold: int) -> list[dict]:
    return [
        e for e in edge_scores if not e.get("skipped") and (
            e.get("error")
            or (e.get("score") is not None and e["score"] < edge_threshold)
            or e.get("direction_supported") is False
        )
    ]


def retry_errored_edges(
    report: dict,
    graph: dict,
    full_sentences: list[dict],
    pool: KeyPool,
    model: str,
    max_attempts: int = 3,
    backoff_seconds: tuple[int, ...] = (15, 45, 120),
    edge_threshold: int = 6,
) -> dict:
    """Re-runs just the edges in report["EdgeAccuracy"] that came back with
    an "error" field, with real exponential backoff between attempts
    instead of immediately retrying into the same pressure that caused the
    failure. Not called automatically from run_quality_check() -- verified
    concretely across this graph's own test runs that both a JSON-parse
    failure (P28->P35: scored cleanly in one run, bare JSONDecodeError in
    the next) and rate-limit exhaustion (P8->P48: failed 4 separate times
    across ~10 minutes) are transport/parsing noise, not re-judgment -- a
    previously good, well-reasoned verdict should not silently vanish into
    an error on the next run just because this function happened to run
    inside the same pass. Keeping it as an explicit, separate step also
    means it isn't fired automatically on every run, which would compound
    exactly the account-level RPM pressure that caused the errors in the
    first place (see quality.py callers for why this session avoided
    stacking concurrent/back-to-back runs against a single API key).

    Mutates and returns report in place. An edge that still errors after
    max_attempts keeps its latest error message rather than being silently
    dropped or guessed at -- this gives up loudly, not silently.
    """
    edge_lookup = {(e["from"], e["to"]): e for e in graph.get("Edges", [])}
    remaining = [(e["from"], e["to"]) for e in report.get("EdgeAccuracy", []) if e.get("error")]
    if not remaining:
        return report

    resolved: dict[tuple[str, str], dict] = {}
    latest: dict[tuple[str, str], dict] = {}

    for attempt in range(max_attempts):
        if not remaining:
            break
        if attempt > 0:
            time.sleep(backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)])

        retry_graph = {"Nodes": graph.get("Nodes", []), "Edges": [edge_lookup[pair] for pair in remaining if pair in edge_lookup]}
        for result in score_edge_accuracy(retry_graph, full_sentences, pool, model):
            pair = (result["from"], result["to"])
            latest[pair] = result
            if not result.get("error"):
                resolved[pair] = result
        remaining = [pair for pair in remaining if pair not in resolved]

    merged = {**latest, **resolved}
    report["EdgeAccuracy"] = [merged.get((e["from"], e["to"]), e) for e in report["EdgeAccuracy"]]
    report["Flagged"]["edges"] = _flagged_edges(report["EdgeAccuracy"], edge_threshold)
    return report
