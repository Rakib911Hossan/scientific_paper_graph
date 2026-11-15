# Thesis + Evidence Claim Pipeline

This codebase turns an arXiv paper into a small set of synthesized, high-level
thesis claims, each backed by concrete supporting evidence (Q&A pairs)
extracted from the paper's own text. It was originally prototyped as two
Colab notebooks implementing a different design (per-sentence extraction +
ambiguity resolution), then restructured into a local Python package
(`src/claim_pipeline/`), then redesigned again after real-paper testing
exposed structural problems with the per-sentence approach. **The current
architecture (described below) is the v2 redesign — see "Why the redesign"
for what changed and why.** The original notebooks are kept under
`notebooks/` for historical reference only.

## Current pipeline overview

Everything below is Stage 1 of a larger plan (thesis synthesis through final
filter produce a flat set of independently-checkable claims); Stage 2 takes
that claim set and builds a validated dependency graph over it. The stage
numbers in the diagram and section headers below are this codebase's own
internal numbering (Stage 0-4 = claim extraction, Stage 2.1 = graph
construction) and predate that reframing, so don't read "Stage 2" in two
different places below as the same thing across sections.

```
arXiv PDF
  -> (Stage 0) PDF text extraction (PyMuPDF)
  -> (Stage 0) clean_markdown / remove_references_section
  -> (Stage 0) sentence segmentation with ids (S1, S2, ...)
  -> (Stage 1) LLM document summary
  -> (Stage 1) LLM thesis-claim synthesis (up to N distinct findings)
  -> theses.json
  -> (Stage 1.1) per-claim evidence-question generation (BM25-grounded, up to 5 questions)
  -> (Stage 1.2) per-question answering (BM25 + doc-summary context, retry-on-empty)
  -> claim_evidence_checkpoint.jsonl
  -> (Stage 1.3) per-QA-pair rewrite into standalone atomic claims (splits multi-fact answers)
  -> atomic_claims_checkpoint.jsonl (with provenance) + atomic_claims.json (flat list)
  -> (Stage 1.4) final filter: keep only claims worth independently verifying
  -> final_claims.json (with provenance) + final_claims_flat.json (flat list)
  -> (dedup) drop claims restating the same fact as another claim in the set
  -> final_claims_deduped.json (with provenance) + _flat.json
  -> (Stage 2.1) proposition-graph construction: one evidence-grounded
     generation call proposes the full edge list
  -> (Stage 2.3) acyclicity verification + LLM-repaired cycle-breaking
  -> data/graph/<arxiv_id>/proposition_graph.json (Nodes, Edges with
     per-edge evidence, RemovedCycleEdges)
  -> (Stage 2.5) LLM-judge structural fidelity check: reference-free
     Node/Edge/Structural Accuracy, evidence-grounded per-edge scoring
     (every edge scored)
  -> proposition_graph_quality.json
  -> (final output) per-connected-component decomposition + depth + fidelity
  -> proposition_graph_final.json
```

Run via `scripts/run_ingest.py`, `scripts/run_generate_theses.py`,
`scripts/run_generate_evidence.py`, `scripts/run_decontextualize.py`,
`scripts/run_final_filter.py`, `scripts/run_dedup.py`,
`scripts/run_build_graph.py` (or `scripts/run_full_pipeline.py` to chain the
first five; dedup and the graph step aren't wired into it yet, see Stage 2.1
below). `run_build_graph.py` always runs cycle repair, the Stage 2.5 quality
check, and writes `proposition_graph_final.json` -- the quality check adds
real LLM-call cost on top of graph generation/repair, but is not optional by
default; pass `--skip-quality-check` to opt out under real rate-limit/cost
pressure. See `README.md` for exact usage.

**File layout for multiple papers**: every script accepts an optional
`--arxiv-id`, used only to namespace *default* output locations so results
from different papers don't silently overwrite each other (explicit
`--output`/`--checkpoint` always wins regardless). `.json` outputs go in a
per-paper subfolder (`data/interim/<arxiv_id>/sentences.json`,
`data/processed/<arxiv_id>/theses.json` etc., `data/graph/<arxiv_id>/
proposition_graph.json`); `.jsonl` checkpoints stay flat with the id as a
filename prefix instead (`data/checkpoints/<arxiv_id>_claim_evidence_checkpoint.jsonl`)
since nothing iterates checkpoints as a per-paper group the way it browses
per-paper json outputs. `run_full_pipeline.py` always passes `--arxiv-id`
through; running an individual stage script directly without it falls back
to the pre-existing bare filename (`theses.json`, `final_claims.json`, ...)
for backward compatibility, which reintroduces the overwrite risk if you're
juggling more than one paper by hand.

## Stage 0 — Ingest (`src/claim_pipeline/ingest/`, `preprocess/`)

Downloads a paper PDF by arXiv id, extracts plain text via PyMuPDF
(`convert_pdf_to_markdown()` — name kept for the function's role in the
pipeline, output is plain text now, not structured Markdown; see "Local
backend" below for why `marker-pdf`/`surya` was replaced), cleans it
(`clean_markdown()` strips
everything before "Abstract", resolves citation anchors, selectively keeps
figure/table captions and equations only when they carry a numeric constant,
strips markdown links/images/HTML/tables), truncates at "References"
(`remove_references_section()`), then splits into sentences with sequential
ids `S1, S2, ...` (`split_into_sentences_with_ids()`, nltk `sent_tokenize`,
filtering section headers, chart-label junk, bare cross-references, and
boilerplate). Output: `data/interim/<arxiv_id>/sentences.json` — the full
sentence pool, used throughout Stage 1 and Stage 2.

## Stage 1 — Thesis synthesis (`src/claim_pipeline/ranking/`)

- **`generate_document_summary()`**: one LLM call producing a 3-5 sentence
  neutral summary of the whole document.
- **`generate_thesis_claims()`** (`ranking/thesis.py`, `ranking/prompts.py`):
  given the full sentence pool + doc summary, one LLM call synthesizes up to
  `max_claims` (default 20, `configs/default.yaml` -> `thesis.max_claims`)
  distinct, standalone, high-level findings the paper actually supports.
  Unlike the old design, these are **generated text, not selected existing
  sentences** — a claim can combine/generalize across multiple source
  sentences (e.g. "fine-tuning is cost-effective: $7-20, under 24h on a
  single GPU" synthesizes facts from several distinct sentences). De-dup is
  baked into the single generation call (explicit "no two claims should
  restate the same point" instruction) rather than a separate pass. The
  prompt allows both empirical findings ("X was observed") and
  recommendation-style findings ("the paper argues X should be done") —
  restricting to empirical-only silently dropped legitimate recommendation
  claims in testing. Output: `theses.json`, plus
  `<output-stem>_doc_summary.txt`.

## Stage 2 — Evidence generation (`src/claim_pipeline/resolution/`)

For each thesis claim:
- **`generate_evidence_questions()`** (`resolution/evidence.py`,
  `resolution/prompts.py`): generates **up to** `n` (default 5,
  `evidence.max_questions_per_claim`) distinct questions whose answers would
  support the claim — not a fixed count; the model returns fewer for
  narrow, single-fact claims and uses the full ceiling for claims combining
  several distinct facts (same "up to N" pattern as thesis generation, which
  replaced an earlier fixed-at-2 design that either padded simple claims
  with a redundant question or under-evidenced richer ones). **Grounded in
  BM25-retrieved sentences relevant to the claim's own text**
  (`evidence.candidate_top_k`, default 8), not just the global doc summary —
  see "Why the redesign" below for why this grounding is load-bearing, not
  optional.
- **`answer_with_bm25()`** (`resolution/qa.py`): for each question, retrieves
  BM25 top-k (`evidence.bm25_top_k`, default 5) over the full sentence pool,
  answers via `answer_question()` (doc summary prepended, `temperature=0.0`,
  `NOT_IN_CONTEXT` sentinel for genuinely unanswerable questions), retries
  once more on the same context if the first attempt comes back empty.
- **`generate_claim_evidence()` / `run_pipeline()`** (`resolution/pipeline.py`):
  orchestrates question-generation -> answering per claim, checkpointing each
  result (`id`, `claim`, `qa_pairs`) to a JSONL file so reruns skip completed
  claims; a per-claim exception is caught, logged, and skipped (not fatal to
  the run, but the claim is redone from scratch on rerun).

## Stage 3 — Decontextualization (`src/claim_pipeline/resolution/atomic_claims.py`)

Takes Stage 2's evidence checkpoint and rewrites every `(question, answer)`
pair into one or more standalone, self-contained atomic claim sentences —
the finest-grained output the pipeline produces, one claim per distinct
checkable fact.

- **`rewrite_qa_to_claims()`**: given the source thesis claim (context only,
  for resolving pronouns/vague references), a question, and its answer,
  produces a list of atomic claims. **Splits the answer into multiple
  claims if it packs multiple distinct facts** — e.g. an answer like "they
  combined curated instruction datasets and filtered out non-English data,
  benchmark contamination, and prohibited content" becomes 3+ separate
  claims, one per filtering step, rather than staying one long compound
  sentence. Resolves pronouns/vague references ("they", "the winning
  teams") to what they actually refer to, using the source claim for
  context. Preserves numbers/units/named entities/technical terms exactly.
  Skips empty answers (nothing to rewrite). **Phrases every claim as a
  direct assertion of the underlying fact, not as a meta-statement about
  the paper's rhetoric** ("RAG can complement fine-tuning...", not "the
  paper suggests combining RAG with fine-tuning...") — this was added
  after Stage 4 (below) was found to be dropping an entire cluster of
  legitimate recommendation-type claims because they were phrased with a
  "the paper suggests/recommends X" wrapper, which reads like an epistemic
  hedge about the world rather than a definite fact about the paper's own
  content.
- **`decontextualize_claim_evidence()` / `run_decontextualization()`**
  (`resolution/pipeline.py`): orchestrates this per source claim across all
  its QA pairs, checkpointing one record per source claim (`id`,
  `source_claim`, `atomic_claims` — each with `text`/`question`/`answer` for
  provenance) to a JSONL file, same resume-on-rerun pattern as Stage 2.
  `run_decontextualize.py` also flattens the result to `atomic_claims.json`
  (just the claim strings, provenance dropped) for consumers that only need
  the final claims.

Verified concretely against real evidence data: a QA pair "Did any
contestants write custom code, and what did they rely on instead? -> No,
they relied on well-established open-source libraries" correctly split into
two atomic claims ("did not write custom code" / "relied on open-source
libraries") rather than staying one compound sentence, and correctly
resolved "they" to "the winning contestants" using the source claim's
context.

## Stage 4 — Final filter (`src/claim_pipeline/resolution/final_filter.py`)

Takes Stage 3's full set of atomic claims and keeps only the ones that
state an objective, checkable fact worth independently verifying, dropping:
opinions/interpretations stated as commentary ("this emphasizes the
importance of X"); genuine epistemic hedges about the world ("may
indicate", "the authors expected to see X" as an unconfirmed prediction);
personal-experience/narrative asides; and bare figure/table references with
no independently-stated fact of their own. **One batched LLM call over the
whole claim list** (not per-claim) — this is a single filtering judgment
over the set, not independent per-item work like Stages 2-3.

- **`filter_atomic_claims()`**: numbers all claims 1..N, asks the model to
  return which indices to KEEP, reconstructs the filtered list in original
  order. Explicitly instructed **not** to drop a claim just because it's
  phrased as "the paper suggests/recommends X" — that's a definite,
  checkable fact about the paper's content, not a hedge about the world
  (this distinction is also handled upstream now, see Stage 3's
  direct-assertion phrasing rule).
- `run_final_filter.py` writes `final_claims.json` (kept claims with
  `text`/`question`/`answer`/`source_id` provenance) and
  `final_claims_flat.json` (just the claim strings).

Verified concretely on real output (66 atomic claims -> 54 kept, 12
dropped): correctly dropped claims like "the low correlation... suggests
that submissions overfitted" (hedge) and "the security exposure emphasized
the need for heightened security measures" (editorial commentary), while
keeping claims like "Approximately 30% of submissions... exceeded the
runtime limit" and "The top-performing solutions... leveraged the QLoRA
quantization method" (concrete, checkable facts).

### A cross-stage bug this surfaced

The first Stage 4 run dropped an entire cluster of legitimate
recommendation claims (RAG/MoE/model merging/pruning/quantization/CUDA
memory management/CPU offloading -- 11 claims) because Stage 3 had phrased
them all as "the paper suggests combining X with fine-tuning," and Stage
4's hedge-detection rule treated that framing as an epistemic hedge. Fixed
in two places for defense in depth: Stage 3's `ATOMIC_CLAIMS_TEMPLATE` now
explicitly requires direct-assertion phrasing (see above), and Stage 4's
`FINAL_FILTER_TEMPLATE` explicitly carves out "the paper suggests/
recommends X" as a keep-worthy fact rather than a hedge, in case
mis-phrased claims slip through from an earlier stage or a future one.
After the fix, all 11 claims survived rephrased as direct assertions (e.g.
"Retrieval-augmented generation (RAG) can be combined with fine-tuning to
evaluate specific AI tasks").

## Dedup (`src/claim_pipeline/resolution/dedup.py`)

Stage 4's output can still contain near-duplicate claims, because Stage 3
rewrites each QA pair independently with no visibility into other QA
pairs' output -- two different questions can each surface the same
underlying fact worded differently (e.g. "dropped to chance level" and
"performed quite poorly, at about chance level" from two separate QA
pairs). This is a separate pass over the *whole* claim set (same batched
pattern as Stage 4's filter, not per-item), run after the final filter and
before Stage 2.1 graph construction -- graph edges built over undeduped
claims produce meaningless circular edges between two nodes stating the
same fact.

- **`dedup_atomic_claims()`**: numbers all claims, asks the model to find
  groups that state the same specific fact (not just the same topic) and
  pick which single claim in each group to keep (preferring the more
  complete/specific phrasing). Claims not part of any group pass through
  unchanged.
- `run_dedup.py` writes `final_claims_deduped.json` (with provenance) and
  `_flat.json`.

Verified concretely on real output (54 -> 48 claims after this pass on the
running example): correctly merged the "dropped to chance level" pair
above, plus three "selected to assess X" / "designed to assess X" pairs
(truthfulness, ethics, bias) that were the same fact stated twice from
different QA pairs sourced from the same underlying sentence.

## Stage 2.1 — Proposition graph construction (`src/claim_pipeline/graph/`)

Takes the deduped claim set and proposes a dependency graph over it: for
each claim ("proposition"), which other propositions in the set are
genuine logical prerequisites for deriving/verifying it. Node ids are
`P1..Pn` assigned in input order. Went through three iterations before
landing on the current design, each one driven by a concrete failure found
by checking generated edges against the paper's actual source sentences
(not just eyeballing plausibility) -- see below.

- **v1 (ungrounded, single-shot)**: one LLM call given only the bare
  proposition list plus the whole-document summary, asked to return a full
  edge list directly. Produced structurally valid output (no cycles, no
  dangling refs) but the edges were unverifiable -- checking a sample
  against source text found genuine misses (e.g. the eval-design timeline
  claims never linked to the overfitting finding they enable) alongside
  edges that looked like same-topic clustering rather than real derivation
  (e.g. linking claims only because they're discussed in the same
  paragraph).
- **v2 (per-proposition BM25 grounding + required evidence citations)**:
  each proposition is now shown with its own BM25 top-k source sentences
  (`graph.build_graph.generate_proposition_graph()`'s `candidate_top_k`,
  default 10) -- same grounding fix as Stage 2's evidence-question
  generation, since ungrounded generation has the same failure signature
  everywhere in this pipeline: a general context block (here, the doc
  summary alone) winning out over specific-but-absent grounding. Every
  edge must now cite the sentence ids (`evidence` field) that justify it;
  the prompt instructs the model to omit an edge entirely rather than
  include one it can't cite. This turned "probably fine" into
  mechanically-checkable: every edge's cited sentence ids were verified
  against the actual source text and held up (zero hallucinated evidence
  ids across runs). Trade-off: this is much more conservative (edges
  dropped from ~36 to ~16-19 on the same 52-claim set) since it now
  produces false negatives (real dependencies the model can't point to one
  specific sentence pair for) instead of the v1 false positives
  (topic-clustering edges).
- **v3 (majority vote across repeated generations, later removed)**: even
  with grounding required, a single run was found to be stochastically
  incomplete -- verified concretely: the Dockerfile-failure chain (untested
  Dockerfiles -> build failures/numerics bugs/unhosted artifact refs) had
  its citable evidence sentence in every relevant node's BM25 context on a
  run that still dropped the edges anyway. `build_majority_graph()` ran
  `generate_proposition_graph()` `n_runs` times (default 3, at different
  temperatures) and `merge_majority_graph()` combined them, tagging each
  edge with a `confidence` tier (`unanimous`/`majority`/`single_run`) and
  resolving direction conflicts by vote count (exact ties excluded from
  `Edges` and surfaced in `AmbiguousDirection` instead of guessed). This
  caught real stochastic dropouts, but was later removed in favor of a
  single generation call: 3x the LLM-call cost/latency per graph for a
  benefit that was hard to distinguish from noise in practice (see the
  residual issue below -- majority voting cannot recover an edge that no
  run proposes evidence for in the first place, which was already this
  approach's dominant failure mode), and a single run's `json.loads()`
  failure (Stage 2.1's retry-on-malformed-JSON re-raises after
  `max_retries`) now aborts the whole graph-generation step 3x less often
  than it did when 3 independent runs each had to succeed. **v3's own
  finding about purpose/goal phrasing still applies and is fixed at the
  prompt level regardless of run count**: a teleological "X was done to
  achieve Y" sentence doesn't by itself establish which of X/Y is the
  prerequisite for verifying the other (verified concretely on `P1->P4`,
  P1 = closed eval set withheld until after submissions closed, P4 =
  closed stage designed to assess generalizability, connected only by
  `S14`, "withheld ... to better ascertain ... generalization ability").
  `GRAPH_GENERATION_TEMPLATE` explicitly instructs: purpose/goal phrasing
  alone, with no other textual signal, should result in omitting the edge
  (same as not being able to cite evidence at all) rather than guessing a
  direction -- this is a property of the single generation call's prompt,
  not something majority voting was providing.

Current output schema is `Nodes` plus a flat `Edges` list of
`{from, to, evidence}` -- `build_graph.to_flat_graph()` reshapes
`generate_proposition_graph()`'s raw per-node `{node, inputs, outputs,
evidence}` format into this directly (no vote count or confidence tier,
since there's exactly one generation run now). `graph.quality`'s Edge
Accuracy check scores every edge for this same reason -- there's no
`unanimous` tier left to exempt an edge from individual review.

### Residual known issue

Grounding is per-proposition BM25 retrieval using the proposition's own
claim text as the query, which can fail to surface a genuine dependency
when the two claims describe the same underlying event in different
vocabulary. Verified concretely: the overfitting-chain claim ("Figure 7
shows low agreement") and the actual overfitting finding ("submissions
performed poorly, at about chance level") sit a few sentences apart in the
source paper, but their respective top-10 BM25 retrievals never overlap
(the connecting sentence ranked ~27th for the Figure-7 claim's query) --
so a single run may never see enough shared context to propose the edge
at all. This is a grounding-coverage gap, not a sampling one -- it was
already the dominant failure mode under majority voting (repeated runs
couldn't recover an edge that no run ever proposed evidence for), so
removing majority voting doesn't materially change this issue's exposure,
just removes the (already-limited) chance of catching an edge that one
lucky run out of several happened to surface.

## Stage 2.3 — Acyclicity verification (`src/claim_pipeline/graph/validate.py`, `build_graph.repair_cycles()`)

`generate_proposition_graph()` proposes each edge independently based on
its own cited evidence, with no whole-graph acyclicity check of its own --
a longer cycle across 3+ nodes can survive even though every individual
edge in it cited real evidence, since no single edge's citation is wrong
in isolation.

- **`validate.build_adjacency()` / `find_cycle()`**: standard DFS
  white/gray/black cycle detection, with one fix worth calling out:
  `visited`/`rec_stack` are seeded from every id in `Nodes`, not just
  edge sources -- seeding from edge sources alone means a pure sink node
  (no outgoing edges, only ever a target) isn't a key in the adjacency
  dict, and `visited[neighbor]` raises `KeyError` the first time such a
  node is reached as someone else's target. `find_cycle()` returns the
  actual cycle path (not just a bool) since `repair_cycles()` needs it to
  build a repair prompt.
- **`repair_cycles()`**: repeatedly finds one cycle, shows the model each
  cycle edge's own cited evidence side by side, and asks which single edge
  to remove -- citable evidence does not guarantee correct direction, so
  even an edge list built from a single generation call (no cross-run
  voting to catch a direction conflict against) can still contain a longer
  cycle across 3+ nodes that a pairwise check would never surface. Removed
  edges are kept under `graph["RemovedCycleEdges"]` with the model's
  stated reason, auditable rather than silently dropped. Falls back to
  dropping the edge with the fewest cited evidence sentences in the cycle
  (weakest grounding, an arbitrary but deterministic tie-break) if the
  model's JSON is unusable or names an edge outside the cycle it was
  shown. `max_iterations` (default 5) is a safety cap, not a target -- a
  real run needing anywhere near that many repair rounds would mean the
  edge set itself is pathological, worth surfacing as a `RuntimeError`
  rather than looping or over-pruning silently.
- Always runs (not optional) as part of `run_build_graph.py`, before the
  graph is saved -- on the one real paper tested against, the merged
  graph came back acyclic with zero repair rounds needed, so this has
  been a no-op safety net in practice so far, not yet exercised against a
  graph that actually needed it.

## Stage 2.5 — Structural fidelity check (`src/claim_pipeline/graph/quality.py`)

LLM-judge rubric adapted from CDCR-SFT Appendix B.4 (Node/Edge/Structural
Accuracy, 0-10 each), run after cycle repair via `run_quality_check()`.
**CDCR-SFT scores a generated graph against a known-correct reference
graph; there is no such reference here** (the graph is synthesized from a
real paper, not a synthetic puzzle with a gold answer), so all three
dimensions are redefined reference-free, checked against the paper's own
source material instead of a ground truth. Not optional -- always runs as
part of `run_build_graph.py` despite the real added LLM-call cost on top of
graph generation/repair; pass `--skip-quality-check` to opt out under real
rate-limit/cost pressure.

- **Node Accuracy**: one batched call over all nodes, checking whether
  each node's generated `description` still faithfully represents the
  original source claim it was built from (paraphrase drift introduced by
  `generate_proposition_graph()` itself). **First version of this prompt
  produced worthless output that looked fine at a glance**: verified
  concretely that all 52 nodes scored a uniform 10/10 with the exact same
  copy-pasted reasoning string ("The node description is identical to the
  source claim.") -- a classic sign the model batch-approved rather than
  comparing each node individually. Fixed by requiring the reasoning name
  a specific fact/number/entity per node and explicitly rejecting a
  reasoning string that would read identically across different nodes;
  verified fixed by rerunning and confirming 52 distinct, source-specific
  reasoning strings (e.g. quoting "Approximately 10%" for one node,
  "300 minutes and 600 questions" for another).
- **Edge Accuracy**: does each edge's cited evidence support a genuine
  dependency in the SPECIFIC claimed direction, not just topical overlap
  -- the one real defect found across extensive manual review of these
  graphs was a direction flip (`P11<->P12/P13`, both directions citing
  identical evidence), never a fabricated or missing-evidence edge, so
  direction gets its own explicit question rather than being folded into
  a general "is evidence sufficient" score. **Scored per-edge (one call
  each) -- every edge in the graph.** An earlier version of this check
  (back when Stage 2.1 still ran majority voting across repeated
  generations, see Stage 2.1's "v3, later removed") skipped edges at
  `unanimous` confidence, on the empirical observation that edges every
  repeated run agreed on never showed the direction-flip defect on manual
  review while every real defect found was on a `majority`/`single_run`/
  tied-direction edge -- that was always an empirical prior from a small
  sample rather than a guarantee (unanimity across 3 runs doesn't rule out
  a shared systematic bias), and it stopped applying entirely once
  majority voting itself was removed and there was no `unanimous` tier
  left to skip. Every edge is now scored individually.
  - **First version of this prompt applied an inconsistent standard
    within a single run.** Verified concretely: `P27->P35`/`P28->P35`
    were accepted because a "This observation" back-reference in the
    source text was treated as sufficient causal/motivational support,
    while `P16->P20-23` was rejected in the SAME run for structurally
    identical back-reference language, with reasoning explicitly
    demanding strict logical necessity ("does not logically depend on or
    require... to be derived"). Fixed by explicitly telling the prompt to
    accept causal/motivational reference language as sufficient and not
    require strict entailment, while still rejecting genuinely disjoint
    "some X... others Y..." listings (verified `P14->P37/P38` -- two
    Dockerfile-defect categories presented as a disjoint list -- stayed
    correctly rejected after this fix). Verified fixed: `P16->P20-23`
    flipped from 0/rejected to 10/accepted immediately after the prompt
    change, with flagged-edge count on the same graph dropping from 8/10
    to 3/10.
- **Structural Fidelity**: one batched call judging whether the graph's
  overall edge shape plausibly tracks the paper's argument per the doc
  summary. Weakest of the three by construction -- a short doc summary
  necessarily compresses away fine-grained multi-hop chains this is meant
  to catch, so a low score here is lower-confidence than a low Node/Edge
  score.
  - **First version extrapolated a rejection onto an edge that was never
    scored.** Verified concretely: with `P16->P20-23` rejected and its
    sibling `P15->P20-23` skipped (unanimous tier, never individually
    checked), the reasoning stated `P15->P20-23` "suffer\[s\] from the
    same logical flaw" as `P16`'s rejected edges -- presenting an
    inference as a finding for an edge that was literally never
    evaluated. Fixed by explicitly forbidding critique/praise of any edge
    not literally present in the "Graph edges" or rejected-edges block
    it's given.
  - **First version used a different dependency-direction convention than
    the rest of the pipeline, and the two checks openly contradicted each
    other as a result.** `GRAPH_GENERATION_TEMPLATE` defines an edge as a
    *verification prerequisite* ("you'd need to know A to verify B"), but
    the original Structural Fidelity prompt (adapted from CDCR-SFT's
    literal causal-graph framing) expected "causes preceding effects" --
    physical/temporal causal order. These frequently agree but can
    diverge for a proposition that is itself a specific causal
    attribution about an already-observed fact. Verified concretely on
    `P11->P12/P13` (P11 = "more than half of Dockerfiles failed to
    build", P12/P13 = "PEFT/Transformers breaking changes caused it"): a
    forced per-edge Edge Accuracy check (evidence-grounded, citing the
    actual source sentence) scored this direction 10/10 every time it was
    checked, while Structural Fidelity called the identical edges a
    "causal reversal" using generic cause-precedes-effect intuition with
    no source-text grounding of its own. Resolved by picking
    verification-prerequisite as canonical (it's the definition the graph
    schema was already built on, and the one the evidence-grounded check
    consistently agreed with) and rewriting `STRUCTURAL_FIDELITY_TEMPLATE`
    to explicitly define direction in those terms, with a worked example
    of the "general fact -> specific cause attributed to it" pattern.
    Verified fixed: after the rewrite, Structural Fidelity started citing
    `P11->P12/P13` as an example of *good* structure instead of a
    reversal.
- **Every per-edge/per-check call is isolated against two failure modes
  verified concretely in practice, not just in theory**: a
  malformed/truncated JSON response that survives one retry-at-a-different-
  temperature (same "deterministic malformed JSON" class documented below
  for `rewrite_qa_to_claims()`), and `generate_with_retry()` itself
  exhausting its retries on repeated transient API errors during a flaky
  stretch. Either failure marks only that one edge/check as `"error"`
  rather than aborting the whole report -- same per-item isolation
  principle as `decontextualize_claim_evidence()`, applied one level up.
  `retry_errored_edges()` is a separate, explicit re-run step (not
  auto-triggered) that retries only the errored edges with real
  exponential backoff (15s/45s/120s) between attempts; kept separate
  rather than automatic so it doesn't compound the account-level rate
  pressure that caused the errors in the first place.
- Also fixed as part of this work, in shared infra rather than
  `quality.py` itself: `llm/gemini.py`'s `generate_with_retry()` was not
  catching `google.genai.errors.ServerError` (5xx, e.g. a live 504
  `DEADLINE_EXCEEDED`) at all -- only `ClientError` (4xx) and httpx-level
  timeouts were retried, so a bare server error propagated uncaught on
  the first attempt with no retry. This affects every stage that calls
  `generate_with_retry()`, not just Stage 2.5; fixed by adding a
  `ServerError` handler alongside the existing `ClientError` one.

### Residual known issue

Regenerating a graph is not a strict-improvement operation, even when the
regeneration is specifically meant to fix one prompt bug. Verified
concretely: regenerating this pipeline's test graph to pick up the
`P1->P4` teleological-phrasing fix (below) did remove that specific bad
edge as intended, but also caused a previously-validated-correct edge
(`P25->P26`, evidence-grounded score 8/10 in Edge Accuracy) to disappear
entirely from all 3 fresh generation runs -- a pure recall loss unrelated
to the prompt change, consistent with the residual grounding-coverage
issue above (LLM graph generation isn't fully stable run-to-run even at
low temperature). A prompt fix should be verified by checking whether the
specific bug it targets is gone, not assumed to make the overall
regenerated graph strictly better than the one it replaced.

## Final output (`src/claim_pipeline/graph/final_output.py`)

`build_final_output()` combines the cycle-repaired graph with the quality
report into a downstream-facing shape, always run by
`run_build_graph.py` regardless of whether `--skip-quality-check` was passed
(`fidelity` fields are simply omitted when there's no quality report,
rather than filled with placeholder scores).

- **Split into one entry per connected component, not one object for the
  whole graph.** Verified concretely on the test graph: it is a forest of
  several small, topically-unrelated dependency clusters (e.g. the
  `P16->P20-23` memory-mitigation cluster and the `P24/27/28->P35`
  evaluation-agreement cluster share nothing) plus a majority of nodes
  (29/52) with no edges at all -- a single graph-wide `depth`/`fidelity`
  would have conflated unrelated clusters and been dominated by whichever
  cluster happened to be worst. Components are found via undirected
  connectivity over `Edges` (a dependency cluster is still one cluster
  regardless of which way its edges point); fully isolated nodes are
  listed separately under `isolated_nodes` rather than force-fit into
  trivial one-node components.
- **`depth`** is the longest path length (in nodes) within that
  component's own subgraph, computed per component -- not hardcoded;
  verified against a synthetic 3-hop chain (`depth: 3`) alongside
  single-hop components (`depth: 2`) in the same run, since every
  component in the one real graph tested against happened to be
  single-hop (`depth: 2` throughout), which wouldn't have caught a
  hardcoding bug on its own.
- **`fidelity` carries the specific flagged edges (with reasoning) inside
  each component, not just an aggregate score** -- an aggregate alone
  can't distinguish "8 because two genuinely bad edges got caught" from
  "8 because the judge was lenient everywhere," and the flagged list is
  what a consumer would actually act on. `edge_score` per component
  averages every edge Edge Accuracy scored within it (every edge in the
  graph, now that Stage 2.1's majority voting -- and the `unanimous` tier
  it used to exempt from scoring -- was removed).
- **`structural_score`/`structural_reasoning` stay at the top level, not
  per-component** -- Structural Fidelity is one holistic judgment over the
  whole graph's shape, it doesn't decompose per cluster.
- **`validated_acyclic` is always `true`** for any graph reaching this
  function, since `repair_cycles()` already guarantees it upstream --
  kept as an explicit per-component field anyway so a downstream consumer
  has a self-documented contract instead of relying on an implicit
  invariant.
- Per-paper output (`data/graph/<arxiv_id>/proposition_graph_final.json`)
  follows the same per-paper file convention as the rest of this pipeline
  (see "File layout for multiple papers" above) -- each paper's DAG gets
  its own complete file with its own component breakdown, not a combined
  batch structure across papers.

## Graph review app (`graph_review_app/`)

A local Flask app for visually reviewing a paper's proposition graph --
separate from the pipeline package (`src/claim_pipeline/`), since it's a
read-only annotator-facing tool, not a pipeline stage. Doesn't call the LLM
or require `GEMINI_API_KEYS`.

- **`app.py`**: two read-only endpoints. `/api/papers` discovers every
  paper with a `data/graph/<arxiv_id>/proposition_graph_final.json` by
  globbing (no hardcoded paper list -- a paper processed by
  `run_build_graph.py` shows up after a page refresh, no restart needed).
  `/api/graph/<arxiv_id>` returns that paper's final output plus a
  `sentences` map (only the specific evidence sentence ids actually cited
  by that paper's edges, looked up from `data/interim/<arxiv_id>/
  sentences.json` -- not the full sentence pool) so the UI can show
  evidence text inline instead of just bare sentence ids.
- **`templates/index.html` + `static/viewer.js`**: hierarchical DAG
  rendering (nodes ranked by dependency depth within each component,
  components shown as separate bands), pan/zoom, click-to-inspect side
  panel for nodes (description + every incoming/outgoing edge) and edges
  (cited evidence text, reviewer reasoning), "flagged edges only" filter,
  node-id search, and a paste-your-own-JSON option for a
  `proposition_graph_final.json` not yet under `data/graph/`.
- **Edge color is a 2-way encoding (green = reviewed and not flagged, red =
  flagged), not a score gradient** -- `proposition_graph_final.json` only
  carries a score/reasoning for edges that were flagged (see
  `graph.final_output`'s design notes above), so a naive gradient would
  have nothing to show for the passed-but-scored edges. An earlier version
  of this view used a 3-way encoding (grey for `unanimous`-confidence
  edges skipped by Stage 2.5's tiering and never individually reviewed);
  that tier no longer exists now that Stage 2.1's majority voting was
  removed and every edge is individually reviewed (see Stage 2.1's "v3,
  later removed" and Stage 2.5's Edge Accuracy notes above).
- Run via `python3 graph_review_app/app.py` -> `http://127.0.0.1:5050`.

## Why the redesign (v1 -> v2)

The original per-sentence design (Stage 1: rank/dedupe existing sentences by
salience; Stage 2: classify each as Atomic/Single/Compound, resolve
ambiguous references via window+BM25 Q&A, rewrite the sentence in place) is
preserved in git history and in the original notebooks, but was replaced
after extensive real-paper debugging surfaced two categories of problem that
kept resurfacing no matter how much the classification/retrieval logic was
patched:

1. **Classification instability.** The Perceptor's Atomic/Single/Compound
   decision flip-flopped across runs on genuinely borderline claims — even
   at `temperature=0.0` — and small few-shot-example wording changes shifted
   decisions in hard-to-predict ways (e.g. rewording one example's *question
   style* from "Whose X suggest Y?" to "Who does 'our' refer to?" flipped an
   unrelated claim's classification from `Single` to `Atomic`).
2. **Few-shot example leakage.** Both the classification and rewrite prompts
   had few-shot examples lifted verbatim from the actual test paper (3/4 and
   5/8 examples respectively), meaning evaluation on that paper's own claims
   was partly memorization of the prompt's own examples, not genuine
   judgment.

The thesis+evidence design sidesteps both: there's no Atomic/Single/Compound
judgment call at all (a claim either has generated supporting evidence or it
doesn't), and the new prompts are zero-shot (no few-shot examples to leak).
**Everything downstream of sentence extraction was reused where it still
applied** — `generate_document_summary()`, `BM25Retriever`, `answer_question()`
(doc-summary injection, retry-on-empty) all carried over unchanged, since
those were validated independently of the classification/rewrite machinery
that got removed (`resolution/perceptor.py`, `resolution/rewrite.py`,
`ranking/salience.py`, `ranking/dedup.py`, `resolution/retrieval.py`'s
`get_window_context()` — none of these exist anymore).

### Bugs found and fixed during the v2 redesign itself

- **Evidence questions were being generated ungrounded from the claim
  alone**, with no document context and no "don't use outside knowledge"
  instruction. This caused the model to generate plausible-sounding but
  wrong questions using parametric knowledge of well-known related work
  (e.g. asking about a "65B parameter model on a single GPU" and "the
  Guanaco model" for a cost-effectiveness claim — both are QLoRA-paper
  results, not this paper's). Fixed by adding the doc summary + an explicit
  no-outside-knowledge instruction to `EVIDENCE_QUESTIONS_TEMPLATE`.
- **Even with the doc summary added, evidence questions for claims not
  covered by the (short, ~4-sentence) doc summary were still wrong —
  consistently and reproducibly (3/3 identical repeat attempts) about a
  *different* claim's topic instead.** Verified concretely: a claim about
  combining fine-tuning with RAG/MoE/model compression generated questions
  about overfitting and data curation instead — both prominent in the doc
  summary, neither relevant to this claim. Root cause: the doc summary only
  covers the paper's ~4 most dominant themes, so for any longer-tail claim
  not covered by it, the model had no claim-specific content to draw a
  question from and defaulted to the summary's own dominant themes. Fixed by
  retrieving BM25 top-k sentences using **the claim's own text as the
  query** and passing those into the question-generation prompt as primary
  grounding, with the doc summary demoted to background-only context. This
  is the same failure signature as `answer_question()`'s doc-summary-driven
  wrong-attribution bug on the old design (see below) — a general context
  block winning out over specific-but-absent grounding, recurring at a
  different pipeline stage.
- **Occasional malformed JSON from `response_mime_type="application/json"`
  responses**, despite that setting nominally guaranteeing valid JSON.
  Usually rare and non-reproducible on retry (same input, clean JSON the
  second time) — `generate_thesis_claims()`, `generate_evidence_questions()`,
  and `rewrite_qa_to_claims()` all retry once on `JSONDecodeError` before
  raising, now at a *different* temperature on the retry (see next item for
  why a same-temperature retry isn't reliable).
- **Some malformed-JSON cases are fully deterministic, not "rare" at all —
  varying temperature or `max_output_tokens` does not fix them.** Verified
  concretely: one specific `(claim, question, answer)` prompt to
  `rewrite_qa_to_claims()` produced byte-for-byte identical truncated JSON
  (missing the closing `}`) across 4 separate calls, including at
  `temperature=0.0`, `temperature=0.3`, and with `max_output_tokens=4096`
  explicitly set. Inspecting the raw API response showed
  `finish_reason=STOP` (not a length or safety cutoff) with only ~60 output
  tokens used — the model itself confidently considers the truncated JSON
  *complete*, so no amount of resampling or budget headroom helps; this is
  a model-generation content bug, not a config problem. Fixed at the
  orchestration level instead: `decontextualize_claim_evidence()` now
  wraps each `(question, answer)` pair's rewrite call individually in its
  own `try/except`, so one QA pair permanently failing this way only drops
  that one pair — it no longer discards the other, successfully-rewritten
  QA pairs for the same source claim (previously, one bad QA pair silently
  made an entire claim's record vanish from the checkpoint, which is what
  surfaced this bug — a 12-claim run produced only 11 checkpoint records).

### Residual known issues (not fully solved, acceptable given evidence)

- A single retry-on-empty in `answer_with_bm25()` does not catch every
  stochastic extraction miss — verified concretely: a question whose answer
  (`S130`/`S132`) was confirmed present in the sentence pool and directly
  relevant still came back empty after one retry. This mirrors the
  non-associative-floating-point-in-batched-serving behavior documented
  below for the old design; a single retry recovers most but not all such
  misses.
- Some referents genuinely aren't stated anywhere in extracted body text
  (e.g. "we"/"our" identity when only the byline states authorship, which
  gets stripped by `clean_markdown()`'s abstract-cutoff) — these correctly
  come back as honest empty answers rather than guessed ones, which is the
  intended behavior, not a bug.

## Notes carried over from the old design (still true for Stage 0/shared infra)

- **`KeyPool` / `RateLimiter`** (`llm/client_pool.py`): round-robins Gemini
  API calls across multiple API keys, each individually rate-limited (~13s
  between calls per key, keeping every key under ~5 req/min). Works with a
  single key too — it just always uses that one key and waits out its own
  rate limit. Every client is created with a bounded request timeout
  (`request_timeout_ms`, default 120s) — verified concretely that without
  one, a dropped connection can hang a call indefinitely rather than
  erroring into the retry logic below (observed a 55-minute stall on a
  single `generate_content` call with negligible CPU time, i.e. genuinely
  blocked on the network, not slow — bounding the timeout turned it into
  an immediate, retryable `httpx.ReadTimeout` instead).
- **`call_with_retry()` / `generate_with_retry()`** (`llm/gemini.py`):
  exponential backoff for transient 503/UNAVAILABLE errors, and per-key
  rate-limited retry with 429/503-specific wait times, respectively.
  `generate_with_retry()` also retries `httpx.TimeoutException` /
  `httpx.TransportError` (the exception `KeyPool`'s bounded timeout above
  actually surfaces) the same way it retries a 503 — without this, the
  bounded-timeout fix alone would just make network hiccups fail fast
  instead of hanging, not actually recover from them.
- A live Gemini API key was previously hardcoded inline in the original
  prototype notebook — this is now sourced from the `GEMINI_API_KEYS`
  environment variable (see `.env.example`), and the original key should be
  treated as compromised/rotated.

## Local backend (no Gemini API, `llm/router.py` + `llm/local_model.py`)

Every LLM call in the pipeline goes through `generate_with_retry()`, imported
by each stage module from `llm/router.py` (not `llm/gemini.py` directly).
`router.py` reads `CLAIM_PIPELINE_BACKEND` (`gemini` default, or `local`) and
dispatches to `llm/gemini.py` (unchanged) or `llm/local_model.py`, without
any stage module or its prompt/config construction needing to know which
backend is live — added specifically so the pipeline can run entirely
without a Gemini API key/cost, on a single local GPU (developed against
Colab's free-tier T4).

- **`config.py` / `configs/default.yaml`**: `backend: gemini | local`.
  `GEMINI_API_KEYS` is only required (and only read) when `backend: gemini`
  — the `local` path needs no API key at all. `local:` block sets
  `model_id` (a Hugging Face Hub repo id, downloaded/cached automatically —
  nothing to pre-download manually), `dtype`, `max_output_tokens`. Every
  field is overridable by an env var of the same shape
  (`CLAIM_PIPELINE_BACKEND`, `CLAIM_PIPELINE_LOCAL_MODEL`,
  `CLAIM_PIPELINE_LOCAL_DTYPE`) without editing the YAML, which is how the
  Colab notebook (below) selects the backend.
- **`llm/local_model.py`**: plain `transformers` + `bitsandbytes` 4-bit
  backend — deliberately **not vLLM**. vLLM was tried first and dropped:
  it pins an exact `torch` build (`torch==2.11.0` as of vllm 0.25.1) with
  its own CUDA-runtime-version requirement, independent of whatever
  `torch`/CUDA the host environment already has. Verified concretely on
  Colab: `pip install vllm` pulled that pinned `torch`, which then failed
  with `ImportError: libcudart.so.13: cannot open shared object file` —
  Colab's CUDA userspace didn't have the exact runtime version that torch
  build's compiled extension expected. This was the third environment
  break caused specifically by installing `vllm` in this session (after
  the Docker requirement and the `transformers` version conflict below),
  with no combination found that avoided all of them — switching to plain
  `transformers` sidesteps this whole class of problem structurally, since
  it has no compiled-extension version pin of its own and just uses
  whatever `torch`/CUDA is already installed. Module-level lazy import
  (module-level singleton `_model`/`_tokenizer`) so importing this module
  doesn't require `transformers`/`torch`/a GPU unless the local backend is
  actually selected at call time. No `KeyPool`/`RateLimiter` equivalent —
  those exist to round-robin/throttle a metered remote API across
  multiple keys, which doesn't apply to a single local GPU.
  **Trade-off versus the abandoned vLLM version: no constrained/guided
  JSON decoding.** `response_mime_type == "application/json"` is honored
  only via prompt instruction (every call site's prompt already asks for
  JSON) plus this pipeline's existing per-call retry-on-malformed-JSON
  logic — the same imperfect-but-workable guarantee Gemini's own JSON mode
  gives in practice (see the malformed-JSON notes elsewhere in this file),
  not a new risk specific to this backend, just no longer mitigated by
  vLLM's `guided_decoding(json_object=True)`.
- **Default model is the plain, unquantized `Qwen/Qwen2.5-3B-Instruct`
  repo — quantization happens at load time via `bitsandbytes`
  (`BitsAndBytesConfig(load_in_4bit=True)`), not a pre-quantized
  checkpoint.** This means any base instruct model repo works here, not
  just ones with a pre-made AWQ/GPTQ checkpoint (an earlier version of
  this backend defaulted to `Qwen/Qwen2.5-7B-Instruct-AWQ` specifically
  for vLLM's auto-detected-quantization path; that constraint doesn't
  apply anymore now that vLLM is gone). **3B, not 7B, despite 4-bit
  quantization fitting a 7B model's weights comfortably on a free-tier
  T4.** Verified concretely: `Qwen2.5-7B-Instruct` in 4-bit hit
  `torch.OutOfMemoryError` specifically during Stage 1's document-summary
  call, not from loading the model itself — Stage 1 feeds the *entire*
  paper's sentence pool into the prompt (see "Stage 1" above), so a single
  prefill attention allocation can spike several GiB independent of model
  weight size; a 7B model leaves too little headroom for that spike on a
  ~15GB T4 even at 4-bit. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is also set (both defensively at module import in `local_model.py` and
  explicitly in the notebook, since it must be set before torch
  initializes CUDA) to reduce allocator fragmentation, and
  `torch.cuda.empty_cache()` runs between retry attempts — neither
  eliminates the underlying long-context memory spike, they just buy
  margin against it. **`dtype` defaults to `float16`, not `bfloat16`** —
  T4 is Turing-generation and has no native bf16 tensor-core support; only
  override to `bfloat16` on a newer GPU (A100/L4/etc.).
- **Stage 0 no longer uses `marker-pdf`/`surya` at all — `ingest/pdf_to_markdown.py`
  extracts plain text via PyMuPDF instead, pipeline-wide (not just under
  `backend: local`).** This wasn't a one-shot fix; it's the end of a chain
  of incompatibilities all rooted in the same conflict, verified concretely
  in order (back when `vllm` was still the local-model backend, before it
  was also dropped, see above): (1) `marker-pdf` latest (2.0+) requires
  `surya-ocr>=0.22`, which redesigned its internal layout/OCR models
  around a served VLM that *always* spawns vLLM-via-Docker on any NVIDIA
  GPU (per `marker-pdf`'s own package metadata: "vLLM (docker) on NVIDIA
  GPUs, llama.cpp elsewhere"), failing on Colab (no Docker) with a
  `SpawnError`; (2) pinning `marker-pdf==1.10.1` (depends on
  `surya-ocr>=0.17,<0.18`, confirmed by downloading and inspecting the
  package directly) avoided that VLM/Docker path, but `surya-ocr` 0.17.x's
  code imports legacy `transformers` APIs (`transformers.onnx.OnnxConfig`,
  then `transformers.pytorch_utils.find_pruneable_heads_and_indices`) that
  `transformers` 5.x removed outright; (3) `vllm` (the local-model backend
  at the time) required `transformers>=5.5.3`, so **no single
  `transformers` version satisfied both `surya-ocr` 0.17.x and `vllm` in
  the same environment** — not a pin-around-able version conflict, two
  hard-incompatible requirements. A stub for the first broken import
  (`transformers.onnx`) held briefly, but each fix just surfaced the next
  legacy symbol `surya` needed, with no reason to expect it to stop. This
  conflict is moot now that `vllm` is gone entirely (above), but plain
  PyMuPDF text extraction was already the fix and stayed the fix — it has
  no `transformers`/ML-model dependency of any kind, so it can't hit this
  or any future version conflict in that space. Trade-off: loses
  `marker`'s structure-aware Markdown output
  (tables/figures/equations rendered cleanly) for plain per-page text —
  acceptable given this pipeline's low accuracy bar; `clean_markdown()`'s
  regex-based cleanup still applies fine to plain text, since none of its
  patterns require literal Markdown syntax to be present, they just no-op
  if it's absent.
- **`docs/colab_local_backend.ipynb`**: end-to-end Colab driver notebook
  for this backend. Gets the repo onto the Colab VM (git clone or unzip an
  uploaded archive), mounts Drive and `rsync`s `data/` there after every
  stage (local disk is wiped on disconnect, Drive isn't), installs deps,
  sets the backend env vars, then either steps through each stage
  individually for one paper (debugging) or runs `run_full_pipeline()` in
  a loop over a `PAPER_LINKS` list of arXiv links/ids (building a dataset
  across many papers, each isolated in its own `try/except` so one
  paper's failure doesn't lose progress on the others). A final summary
  cell reports per-paper sentence/claim/graph counts to distinguish an
  outright failure (a stage that silently produced nothing) from just
  sparse/noisy output — expected and acceptable when running a smaller
  model where raw accuracy isn't the priority.
- **Stage 1's chunk+merge path (`ranking/chunking.py`, `ranking/summarize.py`,
  `ranking/thesis.py`) — the actual fix for Stage 1's OOM risk, not model
  size alone.** Even after switching to a 3B model, Stage 1 still hit
  `torch.OutOfMemoryError` trying to allocate 17+ GiB for a single
  attention op — the root cause is that `generate_document_summary()` and
  `generate_thesis_claims()` feed the model the *entire* paper's sentence
  pool as one prompt (by design, for quality — see "Stage 1" above), which
  for a real paper is long enough (confirmed: ~23k tokens on one test
  paper) that prefill attention memory dominates over model weight memory
  entirely, on any model size that still fits a consumer GPU. Set via
  `chunking.max_sentences` in `configs/default.yaml` or
  `CLAIM_PIPELINE_CHUNK_SIZE_SENTENCES` (`None`/unset = off, the original
  single-call behavior, which is what Gemini's huge context window always
  uses since it never needs this): `chunk_sentences()` splits the
  sentence pool into contiguous chunks; `generate_document_summary()`
  summarizes each chunk independently then merges the partial summaries
  with one more call; `generate_thesis_claims()` generates up to
  `max_claims` candidates per chunk then merges the combined candidate
  pool down to `max_claims` with `_merge_claims()` (new
  `THESIS_MERGE_TEMPLATE` prompt, same numbered-list-selection pattern as
  `resolution/dedup.py`) — only when the combined candidate count exceeds
  `max_claims` in the first place. **Trade-off: the single-call version's
  own "no two claims should restate the same point" instruction only ever
  sees one chunk at a time here**, so cross-chunk duplicate claims are
  caught only by the final merge call, not prevented at generation time —
  acceptable given this backend's low accuracy bar, not free. `llm/
  local_model.py` separately truncates any single prompt still exceeding
  `CLAIM_PIPELINE_LOCAL_MAX_INPUT_TOKENS` (default 8192, truncating from
  the front so every prompt's trailing output-format instructions and the
  chat template's `add_generation_prompt` suffix survive) as a backstop —
  chunking is what actually keeps prompts reasonably sized in normal
  operation; the truncation backstop is defense in depth, not the primary
  mechanism.
- **`resolution/qa.py`'s `_is_empty_answer()` NOT_IN_CONTEXT check was an
  exact match, not a substring check — a real bug surfaced by the local
  backend, not backend-specific in principle.** Gemini reliably outputs
  the bare `NOT_IN_CONTEXT` sentinel (see Stage 2 above) when it can't
  answer, so exact-match worked in practice. A smaller local model instead
  echoed the whole instruction sentence containing that token ("If the
  context does NOT contain the answer, respond with exactly:
  NOT_IN_CONTEXT"), which fails an exact-match check — the non-answer then
  flows through as if it were real, all the way to a bogus final claim
  (verified concretely: a claim whose `text` was literally the string
  `"NOT_IN_CONTEXT"`, sourced from an `answer` that was the echoed
  instruction sentence). Fixed by checking for the sentinel as a substring
  instead of an exact match.
- **`scripts/run_build_graph.py`'s `--candidate-top-k` (new CLI flag,
  default 10, unchanged) — the real fix for Stage 2.1's own version of the
  long-prompt problem, lower-risk than relying on `local_model.py`'s
  input-token truncation.** `generate_proposition_graph()` shows *every*
  proposition in one prompt (needed so any pair can be considered for an
  edge — see "Stage 2.1" above), each with `candidate_top_k` BM25 evidence
  sentences; on a paper with many deduped claims this can exceed even a
  generously-raised truncation backstop. Truncation there is qualitatively
  worse than in Stage 1: it drops whole propositions from the prompt
  (truncating from the front, so whichever propositions land there),
  meaning those claims get *zero* chance of ever appearing in an edge —
  not just seen with less context, silently absent from graph
  consideration entirely. Lowering `--candidate-top-k` instead shrinks the
  prompt at the source while keeping every proposition visible. The Colab
  notebook passes `--candidate-top-k 5` for the local backend.
- **Not yet empirically validated for output quality beyond that one fix**
  — the backend switch and plumbing were verified to run end-to-end, but
  no stage's *output quality* (thesis claims, evidence answers, graph
  edges) has been compared against the Gemini-backed pipeline on the same
  paper. One concrete signal from an actual local-backend run: only 3 of
  ~16 thesis claims produced any surviving evidence/claims after Stage
  2-4 + dedup — worth checking `data/checkpoints/<id>_claim_evidence_checkpoint.jsonl`
  (one record per thesis claim, all its `qa_pairs`) on a real run to see
  whether that's broad Stage 2 answering failure with the smaller model,
  not just this one NOT_IN_CONTEXT leak. Graph construction (Stage 2.1) is
  still the stage most likely to degrade further with a smaller model,
  specifically on the failure modes documented above under "Stage 2.1"
  (edge direction, knowing when to abstain rather than guess) — those were
  hard for Gemini itself even with majority voting and a repair pass.

## Original notebooks (`notebooks/`, historical reference only)

`abd_claim_stage_1_s1.ipynb` and `abd_claim_stage_1_s2.ipynb` implement the
original per-sentence rank/dedupe + classify/resolve/rewrite design described
above under "Why the redesign." They are kept for reference but no longer
reflect `src/claim_pipeline/`'s actual architecture or behavior.
