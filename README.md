# Claim Pipeline

Turns an arXiv paper into a small set of synthesized, high-level thesis
claims, each backed by concrete supporting evidence, decontextualized into
standalone atomic claims and filtered down to the ones worth independently
verifying. See `CLAUDE.md` for a full description of what each stage does
and why the pipeline is designed this way.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEYS
```

**No Gemini API key / running on a local GPU instead:** set `backend: local`
in `configs/default.yaml` (or `CLAIM_PIPELINE_BACKEND=local`) — every LLM
call then runs through `llm/local_model.py` (vLLM) against a Hugging Face
model instead, no `GEMINI_API_KEYS` needed. See "Local backend" in
`CLAUDE.md` for how the switch works, and `docs/colab_local_backend.ipynb`
for a ready-to-run Colab notebook (free-tier T4 GPU) that drives the whole
pipeline this way, including looping over a list of arXiv papers.

## Usage

```bash
# Stages 0-4 (ingest through final filter) -- NOT dedup or the graph stages, see below
python scripts/run_full_pipeline.py --arxiv-id 2503.13507

# Or run stages individually:
python scripts/run_ingest.py --arxiv-id 2503.13507
python scripts/run_generate_theses.py --input data/interim/2503.13507/sentences.json --arxiv-id 2503.13507
python scripts/run_generate_evidence.py \
    --input data/processed/2503.13507/theses.json \
    --full-sentences data/interim/2503.13507/sentences.json \
    --arxiv-id 2503.13507
python scripts/run_decontextualize.py \
    --input data/checkpoints/2503.13507_claim_evidence_checkpoint.jsonl \
    --arxiv-id 2503.13507
python scripts/run_final_filter.py \
    --input data/checkpoints/2503.13507_atomic_claims_checkpoint.jsonl \
    --arxiv-id 2503.13507
```

`run_full_pipeline.py` chains only the first five stages (ingest through
final filter) -- **dedup and the graph-construction stages below aren't
wired into it yet** and have to be run as their own steps:

```bash
# Dedup -- drops near-duplicate claims restating the same fact
python scripts/run_dedup.py \
    --input data/processed/2503.13507/final_claims.json \
    --arxiv-id 2503.13507

# Graph construction (Stage 2.1) + cycle repair (2.3) + quality check (2.5) + final output.
# Always runs cycle repair and the quality check unless --skip-quality-check is passed
# (the quality check adds real LLM-call cost on top of graph generation/repair).
python scripts/run_build_graph.py \
    --input data/processed/2503.13507/final_claims_deduped.json \
    --full-sentences data/interim/2503.13507/sentences.json \
    --doc-summary data/processed/2503.13507/theses_doc_summary.txt \
    --arxiv-id 2503.13507
```

`run_generate_theses.py` also writes `<output-stem>_doc_summary.txt` next to
its `--output` file. `run_generate_evidence.py` picks that up automatically
(same stem, next to `--input`) and feeds it into every Stage 2 answering
call; use `--doc-summary <path>` to point at a different file, or it
proceeds without one if none is found.

Config (model name, thesis/evidence counts, rate limits) lives in
`configs/default.yaml`. API keys are read from the `GEMINI_API_KEYS`
environment variable (comma-separated for round-robin rate limiting; a
single key works too, just serializes through its own rate limit).

## Output shape

`run_generate_evidence.py` checkpoints one JSON record per claim to
`data/checkpoints/claim_evidence_checkpoint.jsonl`:

```json
{
  "id": "C1",
  "claim": "High-performance LLM fine-tuning is highly accessible and cost-effective...",
  "qa_pairs": [
    {"question": "...", "answer": "...", "retried": false},
    {"question": "...", "answer": "...", "retried": false}
  ]
}
```

An empty `"answer"` means the question was genuinely unanswerable from the
paper's extracted text (not a placeholder to be treated as a real value).

`run_decontextualize.py` reads that checkpoint and rewrites every
`(question, answer)` pair into one or more standalone atomic claims,
checkpointing one JSON record per *source* claim to
`data/checkpoints/atomic_claims_checkpoint.jsonl`:

```json
{
  "id": "C2",
  "source_claim": "The winning submissions succeeded by focusing on meticulous data curation...",
  "atomic_claims": [
    {"text": "The winning contestants did not write custom code for fine-tuning.", "question": "...", "answer": "..."},
    {"text": "The top-performing solutions leveraged the QLoRA (Dettmers et al., 2023) quantization method.", "question": "...", "answer": "..."}
  ]
}
```

Each `atomic_claims[].text` is fully self-contained (no dangling pronouns or
vague references), phrased as a direct assertion of fact (not "the paper
suggests X"), and keeps `question`/`answer` for provenance back to the
specific evidence it was derived from. `run_decontextualize.py` also writes
a flat `data/processed/atomic_claims.json` — just the list of claim strings,
with provenance dropped.

`run_final_filter.py` reads that checkpoint, keeps only claims worth
independently verifying (drops opinions, hedges, editorial commentary, and
bare figure/table references), and writes `data/processed/final_claims.json`
(kept claims with `text`/`question`/`answer`/`source_id` provenance) and
`data/processed/final_claims_flat.json` (just the claim strings).

`run_dedup.py` reads `final_claims.json`, merges claims that state the same
specific fact (not just the same topic, keeping the more complete/specific
phrasing of each merged group), and writes
`data/processed/final_claims_deduped.json` (with provenance) and
`data/processed/final_claims_deduped_flat.json` (just the claim strings) —
graph edges built over undeduped claims produce meaningless circular edges
between two nodes stating the same fact, so this runs before graph
construction, not as an optional cleanup step.

`run_build_graph.py` takes the deduped claim set and proposes a dependency
graph over it (which claims are logical prerequisites for verifying which
others) with one evidence-grounded generation call, then cycle-repairs it
and (unless `--skip-quality-check` is passed) runs an LLM-judge structural
fidelity check that scores every edge. Writes to
`data/graph/<arxiv_id>/proposition_graph.json` (`Nodes`, `Edges` with
per-edge evidence, `RemovedCycleEdges`), `proposition_graph_quality.json`
(the quality report, if run), and `proposition_graph_final.json` — the
actual downstream-facing output: the graph decomposed into connected
components with per-component depth/fidelity, isolated nodes listed
separately. See "Stage 2.1" onward in `CLAUDE.md` for the full design
rationale (what the quality check actually catches, why cycle repair is
still needed even with a single generation run).

`python graph_review_app/app.py` (`http://127.0.0.1:5050`) is a local,
read-only Flask app for visually reviewing a paper's
`proposition_graph_final.json` — hierarchical DAG layout, click-to-inspect
nodes/edges with cited evidence text, a flagged-edges-only filter, and a
paste-your-own-JSON option. Auto-discovers any paper under `data/graph/`,
no restart needed after a new one is generated.

## Layout

- `src/claim_pipeline/` — the package:
  - `ingest/`, `preprocess/` — Stage 0: fetch, convert, clean, sentence-split.
  - `ranking/` — Stage 1: `summarize.py` (doc summary), `thesis.py` (claim synthesis), `prompts.py`.
  - `resolution/` — Stages 2-4 + dedup: `evidence.py` (question generation), `qa.py` (answering), `atomic_claims.py` (decontextualization), `final_filter.py` (final filter), `dedup.py` (near-duplicate merge), `retrieval.py` (BM25), `pipeline.py` (orchestration + checkpointing for Stages 2-3), `prompts.py`.
  - `graph/` — Stage 2.1-2.5: `build_graph.py` (proposition graph generation + cycle repair), `validate.py` (cycle detection), `quality.py` (LLM-judge structural fidelity check), `final_output.py` (per-component downstream shape), `prompts.py`.
  - `llm/` — `gemini.py` (Gemini client pool/rate limiting/retry), `router.py` (backend dispatch), `local_model.py` (vLLM/Hugging Face backend), `client_pool.py`.
  - `config.py` — loads `configs/default.yaml` + `.env`.
- `scripts/` — thin CLI entry points per stage (`run_full_pipeline.py` chains Stages 0-4 only, not dedup or the graph stages — see Usage above).
- `configs/default.yaml` — model/backend/rate-limit/thesis/evidence parameters.
- `graph_review_app/` — standalone Flask app for visually reviewing a generated proposition graph; doesn't call the LLM.
- `data/raw|interim|processed|checkpoints|graph/` — pipeline artifacts by stage, namespaced per arXiv id (`--arxiv-id`) so results from different papers don't overwrite each other.
- `docs/colab_local_backend.ipynb` — Colab notebook driving the whole pipeline against the local (non-Gemini) backend.
- `notebooks/` — original prototype notebooks (different, earlier design), kept for historical reference only.
