# Proposition Graph Review App

Local Flask app for reviewing this pipeline's proposition graphs — an
interactive DAG viewer over whatever's already in `data/graph/`. Read-only;
doesn't call the LLM pipeline or require `GEMINI_API_KEYS`.

## Run

```bash
python3 graph_review_app/app.py
```

Then open http://127.0.0.1:5050

## What it does

- Discovers every paper with a `data/graph/<arxiv_id>/proposition_graph_final.json`
  automatically (click "Refresh" after running `run_build_graph.py` on a new
  paper — no restart needed).
- Renders each paper's graph as a hierarchical DAG, one band per connected
  component, nodes ranked by dependency depth.
- Edge color = review status (green = reviewed and supported, red =
  flagged; every edge is individually reviewed by Stage 2.5's Edge
  Accuracy check now that majority voting across repeated generations was
  removed).
- Click a node for its full description and every incoming/outgoing edge.
  Click an edge for its cited evidence sentences and reviewer reasoning
  (for flagged edges).
- "Flagged edges only" filter and node-ID search.
- "+ Paste graph JSON" loads any `proposition_graph_final.json`-shaped file
  ad hoc, for a graph not yet under `data/graph/`.

## Files

- `app.py` — Flask server: serves the page and two read-only JSON endpoints
  (`/api/papers`, `/api/graph/<arxiv_id>`).
- `templates/index.html` — page shell.
- `static/viewer.js` — all rendering/interaction logic (layout, pan/zoom,
  click-to-inspect, filtering, search).
