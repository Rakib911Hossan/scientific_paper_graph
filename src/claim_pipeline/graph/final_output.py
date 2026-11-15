"""Combines Stage 2.1's merged/cycle-repaired proposition graph with Stage
2.5's optional quality report into a single downstream-facing output.

Design choices settled by inspecting this exact function's output against
the real 2503.13507 graph before writing it, not decided in the abstract:

- The output is split into one entry per connected component, not one
  blob for the whole graph -- this graph (and most graphs this pipeline
  produces, per CLAUDE.md's own "Residual known issue" on grounding
  coverage) is a forest of small, unrelated dependency clusters rather
  than one connected structure, so a single "depth"/"fidelity" for the
  whole thing conflated clusters that have nothing to do with each other
  (e.g. the P16->P20-23 memory-mitigation cluster and the P24/27/28->P35
  evaluation-agreement cluster are structurally and topically unrelated).
- Nodes/edges are kept in full (id/name/description for nodes; from/to/
  evidence for edges), not stripped down to bare id pairs -- the evidence
  citations are the entire point of this pipeline's grounding, and a
  downstream consumer that can't see them can't tell a trustworthy edge
  from an assertion.
- Fully isolated nodes (no edge in either direction) are listed separately
  under "isolated_nodes" rather than force-fit into a one-node component --
  there's no dependency structure to report for them, and on this graph
  they're the majority of nodes (29/52), so silently making 29 trivial
  single-node "components" would swamp the handful of clusters that
  actually have structure to show.
- "fidelity" carries the SPECIFIC flagged edges within that component (with
  their reasoning), not just an aggregate score -- an aggregate alone can't
  distinguish "8 because two genuinely bad edges got caught" from "8
  because the judge was lenient everywhere," and the flagged list is what
  a consumer would actually act on.
- "edge_score" per component averages the edges Edge Accuracy actually
  scored in that component -- every edge is scored (see graph.quality,
  which scores every edge in the graph now that majority voting across
  repeated generations was removed), so this excludes only edges that
  errored rather than any tier skipped by design.
- "structural_score"/"structural_reasoning" stay at the top level, not
  per-component -- Structural Fidelity is a single holistic judgment over
  the whole graph's shape against the paper's argument (per
  graph.quality.score_structural_fidelity()), it doesn't decompose
  per-cluster.
- "validated_acyclic" is always True for any graph reaching this function,
  since graph.build_graph.repair_cycles() already guarantees it upstream
  -- kept as an explicit per-component field anyway so a downstream
  consumer has a self-documented contract instead of having to know that
  invariant holds implicitly.
"""


def _connected_components(graph: dict) -> list[set[str]]:
    """Groups node ids into connected components, treating edges as
    undirected for the purpose of grouping (a dependency cluster is still
    one cluster regardless of which way its edges point). Only considers
    nodes that appear in at least one edge -- fully isolated nodes are
    handled separately by the caller."""
    adjacency: dict[str, set[str]] = {}
    for edge in graph.get("Edges", []):
        adjacency.setdefault(edge["from"], set()).add(edge["to"])
        adjacency.setdefault(edge["to"], set()).add(edge["from"])

    visited: set[str] = set()
    components = []
    for start in adjacency:
        if start in visited:
            continue
        component: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, set()) - component)
        visited |= component
        components.append(component)
    return components


def _depth(node_ids: set[str], edges: list[dict]) -> int:
    """Longest path length (in nodes) within this component's subgraph.
    Assumes the overall graph is acyclic -- true for any graph that has
    gone through repair_cycles(), required before this function is
    called."""
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        adjacency[edge["from"]].append(edge["to"])

    memo: dict[str, int] = {}

    def longest_path(node: str) -> int:
        if node in memo:
            return memo[node]
        neighbors = adjacency.get(node, [])
        result = 1 if not neighbors else 1 + max(longest_path(n) for n in neighbors)
        memo[node] = result
        return result

    return max((longest_path(node_id) for node_id in node_ids), default=0)


def _component_fidelity(node_ids: set[str], quality_report: dict) -> dict:
    node_scores = [
        n["score"] for n in quality_report.get("NodeAccuracy", []) if n.get("id") in node_ids and n.get("score") is not None
    ]
    edge_scores = [
        e["score"]
        for e in quality_report.get("EdgeAccuracy", [])
        if e.get("from") in node_ids and not e.get("skipped") and e.get("score") is not None
    ]
    flagged_edges = [
        {"from": e["from"], "to": e["to"], "reasoning": e.get("reasoning") or e.get("error", "")}
        for e in quality_report.get("Flagged", {}).get("edges", [])
        if e.get("from") in node_ids
    ]
    return {
        "node_score": round(sum(node_scores) / len(node_scores), 1) if node_scores else None,
        "edge_score": round(sum(edge_scores) / len(edge_scores), 1) if edge_scores else None,
        "flagged_edges": flagged_edges,
    }


def build_final_output(graph: dict, arxiv_id: str, quality_report: dict | None = None) -> dict:
    """graph: the merged, cycle-repaired proposition graph (build_graph.
    repair_cycles()'s output). quality_report: run_quality_check()'s
    output, or None if Stage 2.5 wasn't run for this graph -- per-component
    "fidelity" is omitted entirely in that case rather than filled with
    placeholder scores, so its absence is an honest signal that the graph
    hasn't been quality-checked, not indistinguishable from a
    checked-and-perfect graph."""
    node_lookup = {node["id"]: node for node in graph.get("Nodes", [])}
    removed_cycle_edges = graph.get("RemovedCycleEdges", [])

    components = sorted(_connected_components(graph), key=lambda ids: min(int(i[1:]) for i in ids))

    component_output = []
    for i, node_ids in enumerate(components, start=1):
        component_edges = [e for e in graph.get("Edges", []) if e["from"] in node_ids]
        entry = {
            "component_id": f"C{i}",
            "nodes": [node_lookup[nid] for nid in sorted(node_ids, key=lambda i: int(i[1:]))],
            "edges": component_edges,
            "depth": _depth(node_ids, component_edges),
            "validated_acyclic": True,
            "removed_cycle_edges": sum(1 for e in removed_cycle_edges if e.get("from") in node_ids or e.get("to") in node_ids),
        }
        if quality_report is not None:
            entry["fidelity"] = _component_fidelity(node_ids, quality_report)
        component_output.append(entry)

    connected_ids = {nid for ids in components for nid in ids}
    isolated_nodes = [
        node_lookup[nid] for nid in sorted(node_lookup, key=lambda i: int(i[1:])) if nid not in connected_ids
    ]

    output = {
        "arxiv_id": arxiv_id,
        "components": component_output,
        "isolated_nodes": isolated_nodes,
    }
    if quality_report is not None:
        output["structural_score"] = quality_report.get("StructuralFidelity", {}).get("score")
        output["structural_reasoning"] = quality_report.get("StructuralFidelity", {}).get("reasoning")

    return output
