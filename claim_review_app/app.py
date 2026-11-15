#!/usr/bin/env python
"""Local link-review app (same UI/UX as the hf_space Streamlit prototype).

One page per paper: collapsible claim graph, then each link (A -> B) with its evidence on
the left and the Q1-Q5 Yes/No checklist on the right. Answers are saved locally to
claim_review_app/reviews/<reviewer>.jsonl (append-only, latest row per link wins).

    python claim_review_app/app.py   ->  http://127.0.0.1:5051
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request

from data_source import load_all, sentences
from export_graph_html import longest, render_page

HERE = Path(__file__).resolve().parent
REVIEWS = HERE / "reviews"
app = Flask(__name__)
GRAPHS, SKIPPED = load_all()
print(f"{len(GRAPHS)} papers loaded; skipped {len(SKIPPED)} with cross-component edges or cycles")


def review_links(g):
    comps = [c for c in g["components"] if longest(c) >= 2]
    return comps, [e for c in comps for e in c["edges"]]


def safe(name):
    return re.sub(r"[^\w-]", "_", name.strip().casefold())[:60]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/papers")
def papers():
    out = []
    for i, g in GRAPHS.items():
        n = len(review_links(g)[1])
        if n:
            out.append({"id": i, "links": n})
    return jsonify(out)


@app.get("/api/graph/<aid>")
def graph(aid):
    g = GRAPHS.get(aid) or abort(404)
    comps, links = review_links(g)
    return jsonify({
        "id": aid,
        "nodes": {n["id"]: n for c in comps for n in c["nodes"]},
        "links": [{"from": e["from"], "to": e["to"], "evidence": e["evidence"]} for e in links],
        "sentences": sentences(aid),
    })


@app.get("/graph_page/<aid>")
def graph_page(aid):
    g = GRAPHS.get(aid) or abort(404)
    html = render_page(g, aid, True)
    return Response(html or "<p style='font:14px sans-serif'>This paper has no chain of 3+ claims.</p>", mimetype="text/html")


@app.get("/api/reviews")
def get_reviews():
    f = REVIEWS / f"{safe(request.args.get('reviewer', ''))}.jsonl"
    latest = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                latest[(r["arxiv_id"], r["from"], r["to"])] = r
    return jsonify([r for r in latest.values() if r.get("verdict")])


@app.post("/api/reviews")
def post_review():
    r = request.get_json()
    if not r or not r.get("reviewer", "").strip():
        abort(400)
    r["time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    REVIEWS.mkdir(exist_ok=True)
    with (REVIEWS / f"{safe(r['reviewer'])}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, debug=False)
