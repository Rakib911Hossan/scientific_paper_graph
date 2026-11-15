"""Stage 2.3: acyclicity verification over a generated proposition graph.

generate_proposition_graph() proposes each edge independently based on its
own cited evidence, with no whole-graph acyclicity check of its own -- a
longer cycle across 3+ nodes (e.g. P1 -> P2 -> P3 -> P1) can survive even
though every individual edge in it cited real evidence, since no single
edge's citation is wrong in isolation. This module detects such cycles;
build_graph.repair_cycles() uses it to actually break them.
"""


def build_adjacency(graph: dict) -> dict[str, list[str]]:
    """Builds a node-id -> outgoing-neighbor-ids adjacency dict, seeded from
    every id in graph["Nodes"] (not just edge sources) so a pure sink node
    (no outgoing edges) is still a key -- otherwise has_cycle()/find_cycle()
    would KeyError on visited[neighbor] the first time such a node is
    reached as someone else's target."""
    adjacency: dict[str, list[str]] = {node["id"]: [] for node in graph.get("Nodes", [])}
    for edge in graph.get("Edges", []):
        adjacency.setdefault(edge["from"], []).append(edge["to"])
        adjacency.setdefault(edge["to"], [])
    return adjacency


def _find_cycle_dfs(
    adjacency: dict[str, list[str]],
    node: str,
    visited: dict[str, bool],
    rec_stack: dict[str, bool],
    path: list[str],
) -> list[str] | None:
    visited[node] = True
    rec_stack[node] = True
    path.append(node)

    for neighbor in adjacency.get(node, []):
        if not visited[neighbor]:
            cycle = _find_cycle_dfs(adjacency, neighbor, visited, rec_stack, path)
            if cycle:
                return cycle
        elif rec_stack[neighbor]:
            start = path.index(neighbor)
            return path[start:] + [neighbor]

    rec_stack[node] = False
    path.pop()
    return None


def find_cycle(adjacency: dict[str, list[str]]) -> list[str] | None:
    """Returns one cycle as a list of node ids (first == last), or None if
    the graph is a DAG. Only one cycle is returned per call -- repair_cycles()
    calls this repeatedly, re-checking after each edge removal, since
    breaking one cycle can still leave others."""
    visited = {node: False for node in adjacency}
    rec_stack = {node: False for node in adjacency}

    for node in adjacency:
        if not visited[node]:
            cycle = _find_cycle_dfs(adjacency, node, visited, rec_stack, [])
            if cycle:
                return cycle
    return None


def has_cycle(adjacency: dict[str, list[str]]) -> bool:
    return find_cycle(adjacency) is not None
