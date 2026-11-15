"""Load graphs for the review app straight from dev_data (no copy step).

Sources: dev_data/new_papers_graphs/<id>_graph.json and every graph in
dev_data/dev_data_combined.json. Edge evidence is turned into [{id, text}]: sentence ids
resolve against data/interim/<id>/sentences.json when present; section/figure/table
citations are shown as a location to look up in the PDF.
"""
import json
import re
from pathlib import Path

from export_graph_html import longest

ROOT = Path(__file__).resolve().parents[1]
ARXIV = re.compile(r"\d{4}\.\d{4,5}(v\d+)?")


def sentences(aid):
    f = ROOT / "data/interim" / aid / "sentences.json"
    return json.loads(f.read_text()) if f.exists() else []


def convert(g, sents):
    sid = {s["id"]: s["text"] for s in sents}
    for c in g["components"]:
        c.setdefault("fidelity", {"flagged_edges": []})
        for e in c["edges"]:
            ev = []
            for x in e.get("evidence", []):
                if isinstance(x, dict):
                    ev.append(x)
                elif x in sid:
                    ev.append({"id": x, "text": sid[x]})
                elif re.fullmatch(r"S\d+", x):
                    ev.append({"id": x, "text": "(sentence text not available for this paper)"})
                else:
                    ev.append({"id": "Cited location", "text": f"{x} — open the paper PDF and read this part."})
            e["evidence"] = ev
    return g


def repair(g):
    """Make the graph drawable: merge components joined by a cross-component edge, drop
    isolated-node/edge dangling refs, and break any cycle by removing its last edge."""
    comps = g["components"]
    owner = {n["id"]: i for i, c in enumerate(comps) for n in c["nodes"]}
    par = list(range(len(comps)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    edges = [e for c in comps for e in c["edges"] if e["from"] in owner and e["to"] in owner]
    for e in edges:
        par[find(owner[e["from"]])] = find(owner[e["to"]])
    groups = {}
    for i, c in enumerate(comps):
        groups.setdefault(find(i), []).append(c)
    merged = []
    for cs in groups.values():
        m = dict(cs[0])
        m["nodes"] = [n for c in cs for n in c["nodes"]]
        m["edges"] = [e for c in cs for e in c["edges"] if e["from"] in owner and e["to"] in owner]
        for e in edges:  # cross edges were in neither source component's own list
            if find(owner[e["from"]]) == find(owner[cs[0]["nodes"][0]["id"]]) and e not in m["edges"]:
                m["edges"].append(e)
        merged.append(m)
    for m in merged:
        while True:
            try:
                longest(m)
                break
            except RecursionError:
                m["edges"].pop()  # a cycle: drop edges until acyclic (rare)
    g["components"] = merged
    return g


def load_all():
    raw = {}
    for p in sorted((ROOT / "dev_data/new_papers_graphs").glob("*_graph.json")):
        raw[p.name.removesuffix("_graph.json")] = json.loads(p.read_text())
    for j, g in enumerate(json.loads((ROOT / "dev_data/dev_data_combined.json").read_text())):
        i = str(g.get("arxiv_id") or "").removesuffix("v1")
        if not ARXIV.fullmatch(i):
            i = (g.get("source_file") or f"paper{j}").removesuffix("_graph.json").removesuffix(".json")
        if i in raw:  # same paper present twice in the 80 (two graph versions): keep both
            i = f"{i}_{j}"
        raw[i] = g
    return {i: repair(convert(g, sentences(i))) for i, g in raw.items()}, []
