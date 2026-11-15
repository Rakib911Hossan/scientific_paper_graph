import json
from collections import defaultdict

from google.genai import types

from claim_pipeline.graph.prompts import CYCLE_REPAIR_TEMPLATE, GRAPH_GENERATION_TEMPLATE
from claim_pipeline.graph.validate import build_adjacency, find_cycle
from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.llm.router import generate_with_retry
from claim_pipeline.resolution.retrieval import BM25Retriever


def generate_proposition_graph(
    claims: list[dict],
    full_sentences: list[dict],
    doc_summary: str,
    pool: KeyPool,
    model: str,
    candidate_top_k: int = 10,
    temperature: float | None = None,
    max_retries: int = 2,
) -> dict:
    """Stage 2.1: one-shot structured-output call that proposes a full edge
    list over the given propositions in a single pass, rather than judging
    each pair independently. claims: list of dicts, each with at least a
    "text" key (e.g. final_claims.json from Stage 4/the final filter) --
    assigned ids P1..Pn in input order. full_sentences: the full sentence
    pool (the *_sentences.json from run_ingest.py) -- BM25-retrieved per
    proposition (using the proposition's own text as query) and shown
    alongside it, so the model has to ground every edge in specific
    sentence ids rather than judging dependency from the bare claim text
    or same-topic proximity alone (same failure mode -- and same fix --
    documented for resolution.evidence.generate_evidence_questions()).
    doc_summary: Stage 1's whole-document summary, given as background so
    the model can recognize dependencies that require paper-level context
    (e.g. an experimental design choice being a precondition for a
    finding). temperature: fixed sampling temperature for the run; when
    None, retries still escalate 0.0 -> 0.3 on a JSON decode failure as
    usual.

    Returns {"Nodes": [...], "Edges": [...]} -- "Nodes" is NOT part of the
    model's response; it's built directly from claims[i]["text"] in input
    order (P1..Pn) after the call, never asked of the model at all. Verified
    concretely as a real failure mode on the local (small-model) backend:
    an earlier version of this function asked the model to also echo back
    each proposition's own text as a "description" field, and under a long,
    many-proposition prompt the model would fail to reproduce most of them
    faithfully, instead repeating the single most textually-salient fact
    from the source sentences across nearly every node (e.g. one node's
    "51.7% on MATH" copied onto ~20 unrelated nodes). Building Nodes from
    the already-known input text sidesteps this class of bug entirely --
    there's nothing left for the model to get wrong about node identity --
    and also shrinks the model's *output* (no need to echo N full claim
    texts back in JSON), which independently reduces truncated/malformed-
    JSON risk. "Edges" is still the model's own generated dict of
    {"node", "inputs", "outputs", "evidence"} entries, each carrying an
    "evidence" list of sentence ids. This is NOT yet validated for
    acyclicity, dangling references, or whether the cited evidence ids
    actually exist/support the edge -- see repair_cycles() for acyclicity.
    """
    retriever = BM25Retriever(full_sentences)
    lines = []
    for i, c in enumerate(claims, start=1):
        candidates = retriever.retrieve(c["text"], top_k=candidate_top_k)
        source_block = "\n".join(f"  [{s['id']}] {s['text']}" for s in candidates)
        lines.append(f"P{i}: {c['text']}\n{source_block}")
    proposition_block = "\n\n".join(lines)

    prompt = GRAPH_GENERATION_TEMPLATE.format(doc_summary=doc_summary, proposition_block=proposition_block)

    nodes = [{"id": f"P{i}", "name": c["text"][:60], "description": c["text"]} for i, c in enumerate(claims, start=1)]

    for attempt in range(max_retries):
        attempt_temperature = temperature if temperature is not None else (0.0 if attempt == 0 else 0.3)
        response_text = generate_with_retry(
            pool,
            model,
            prompt,
            generation_config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=attempt_temperature,
            ),
        )
        try:
            parsed = json.loads(response_text)
            return {"Nodes": nodes, "Edges": parsed.get("Edges", [])}
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise
            continue


def to_flat_graph(raw_graph: dict) -> dict:
    """Reshapes generate_proposition_graph()'s raw per-node {"node": ...,
    "inputs": [...], "outputs": [...], "evidence": [...]} entries into the
    flat {"from", "to", "evidence"} edge schema the rest of the pipeline
    (repair_cycles(), graph.quality, graph.final_output) consumes. There is
    now exactly one generation run (majority voting across repeated runs
    was removed), so there is no vote count or confidence tier to compute
    -- every edge the model proposed is kept as-is, still subject to
    repair_cycles() for acyclicity and graph.quality's per-edge check for
    correctness."""
    edges: dict[tuple[str, str], set[str]] = defaultdict(set)
    for e in raw_graph.get("Edges", []):
        node = e["node"]
        evidence = set(e.get("evidence", []))
        for src in e.get("inputs", []):
            edges[(src, node)] |= evidence
        for dst in e.get("outputs", []):
            edges[(node, dst)] |= evidence

    return {
        "Nodes": raw_graph.get("Nodes", []),
        "Edges": [{"from": a, "to": b, "evidence": sorted(ev)} for (a, b), ev in edges.items()],
    }


def _cycle_repair_prompt(cycle: list[str], cycle_edges: list[dict], node_lookup: dict[str, dict]) -> str:
    blocks = []
    for edge in cycle_edges:
        from_node = node_lookup[edge["from"]]
        to_node = node_lookup[edge["to"]]
        blocks.append(
            f"Edge {edge['from']} -> {edge['to']}\n"
            f"  {edge['from']}: {from_node['description']}\n"
            f"  {edge['to']}: {to_node['description']}\n"
            f"  evidence: {', '.join(edge.get('evidence', [])) or '(none)'}"
        )
    return CYCLE_REPAIR_TEMPLATE.format(cycle_path=" -> ".join(cycle), cycle_block="\n\n".join(blocks))


def repair_cycles(
    graph: dict,
    pool: KeyPool,
    model: str,
    max_iterations: int = 5,
    max_retries: int = 2,
) -> dict:
    """Stage 2.3: generate_proposition_graph() has no built-in cycle check
    (it proposes an edge list based on citable evidence for each edge
    independently, so a longer logical cycle across 3+ nodes can survive
    unnoticed even though every individual edge cited real evidence). This
    repeatedly finds one such cycle (graph.validate.find_cycle()) and asks
    the model, shown each cycle edge's own cited evidence side by side,
    which single edge to remove -- citable evidence does not guarantee
    correct direction. Mutates and returns graph; removed edges are kept
    under graph["RemovedCycleEdges"] (with the model's stated reason)
    rather than silently dropped, so a cycle repair is auditable.

    Raises RuntimeError if max_iterations repair rounds still leave a cycle
    -- this would mean the model keeps naming an edge outside the cycle it
    was shown, or a genuinely pathological edge set; either way, that's a
    bug worth surfacing rather than looping forever or emitting a graph
    that was never actually verified acyclic.
    """
    node_lookup = {node["id"]: node for node in graph.get("Nodes", [])}
    edge_lookup = {(edge["from"], edge["to"]): edge for edge in graph.get("Edges", [])}
    removed = []

    for _ in range(max_iterations):
        adjacency = build_adjacency(graph)
        cycle = find_cycle(adjacency)
        if cycle is None:
            break

        cycle_edges = [edge_lookup[(cycle[i], cycle[i + 1])] for i in range(len(cycle) - 1)]
        prompt = _cycle_repair_prompt(cycle, cycle_edges, node_lookup)

        decision = None
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
                decision = json.loads(response_text)
                break
            except json.JSONDecodeError:
                continue

        target = None
        if decision is not None:
            candidate = (decision.get("remove_edge", {}).get("from"), decision.get("remove_edge", {}).get("to"))
            if candidate in edge_lookup and candidate in {(e["from"], e["to"]) for e in cycle_edges}:
                target = candidate

        if target is None:
            # Model gave no usable/in-cycle answer after retries -- fall back to
            # removing the edge with the fewest cited evidence sentences (weakest
            # grounding), an arbitrary but deterministic tie-break rather than
            # leaving the cycle unresolved.
            target = min(
                ((e["from"], e["to"]) for e in cycle_edges),
                key=lambda pair: len(edge_lookup[pair].get("evidence", [])),
            )
            reason = "auto: weakest-evidence edge in cycle (no usable model decision)"
        else:
            reason = decision.get("reason", "")

        removed_edge = dict(edge_lookup.pop(target))
        removed_edge["reason"] = reason
        removed.append(removed_edge)
        graph["Edges"] = [edge for edge in graph["Edges"] if (edge["from"], edge["to"]) != target]
    else:
        raise RuntimeError(f"Could not break all cycles within {max_iterations} repair iterations")

    if removed:
        graph["RemovedCycleEdges"] = removed
    return graph
