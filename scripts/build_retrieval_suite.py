"""Build the cleaned 200-graph dataset, a fixed stratified split, and the
Colab notebooks in docs/retrieval_suite/ (one per model group, plus 5-fold
cross-validation) that run every retrieval backbone (BM25/TF-IDF and 12 open
embedding models) on that split. scripts/collect_retrieval_results.py merges
their saved outputs.

Replaces the per-model docs/colab_train_claim_embedder_*.ipynb notebooks,
which each recomputed their own split (so numbers shifted between notebooks)
and, for E5/Qwen, picked the best checkpoint by eval loss on the same
held-out papers they were then scored on.

Sources:
  dev_data/dev_data_combined.json   -> "pipeline" graphs (Gemini pipeline output)
  dev_data/new_papers_graphs/*.json -> "single_pass" graphs
  data/graph/2505.17638/proposition_graph_final.json -> one more "pipeline" graph,
    a full Gemini run (with Stage 2.5 quality check) not yet in dev_data_combined

Split (stratified by source, SEED=42):
  test = 20% of each source; dev = 10% of the remaining train papers, used
  only for checkpoint selection; train = the rest. Also writes 5 stratified
  folds for optional cross-validation.

Each notebook embeds the dataset + split (gzip+base64), so it runs on a fresh
Colab runtime with no upload, Drive mount, or repo access.
"""
import argparse
import base64
import glob
import gzip
import json
import os
import random
import re

SEED = 42
TEST_FRAC = 0.2
DEV_FRAC = 0.1
N_FOLDS = 5


def paper_id(paper: dict, path: str) -> str:
    aid = paper.get("arxiv_id")
    if aid and aid != "unspecified":
        # strip a trailing arXiv version suffix ("2607.04153v1" -> "2607.04153")
        head, sep, tail = aid.rpartition("v")
        return head if sep and tail.isdigit() and "." in head else aid
    stem = os.path.splitext(os.path.basename(paper.get("source_file") or path))[0]
    return stem.removesuffix("_graph")


def slim(paper: dict, pid: str, source: str) -> dict:
    return {
        "paper_id": pid,
        "source": source,
        "components": [
            {
                "nodes": [{"id": n["id"], "description": n["description"]} for n in c.get("nodes", [])],
                "edges": [{"from": e["from"], "to": e["to"]} for e in c.get("edges", [])],
            }
            for c in paper.get("components", [])
        ],
    }


def load_papers(combined_path: str, single_pass_dir: str, extra_pipeline: list) -> list:
    papers, seen = [], set()
    with open(combined_path) as f:
        combined = json.load(f)
    raw = [(p, combined_path, "pipeline") for p in combined]
    for path in extra_pipeline:
        with open(path) as f:
            raw.append((json.load(f), path, "pipeline"))
    for path in sorted(glob.glob(os.path.join(single_pass_dir, "*.json"))):
        with open(path) as f:
            raw.append((json.load(f), path, "single_pass"))
    for paper, path, source in raw:
        pid = paper_id(paper, path)
        if pid in seen:
            print(f"skipping duplicate paper {pid} ({paper.get('source_file') or path})")
            continue
        seen.add(pid)
        papers.append(slim(paper, pid, source))
    return papers


def stratified_split(papers: list, rng: random.Random) -> dict:
    split = {"train": [], "dev": [], "test": []}
    folds = [[] for _ in range(N_FOLDS)]
    for source in sorted({p["source"] for p in papers}):
        ids = sorted(p["paper_id"] for p in papers if p["source"] == source)
        rng.shuffle(ids)
        n_test = round(TEST_FRAC * len(ids))
        rest = ids[n_test:]
        n_dev = max(1, round(DEV_FRAC * len(rest)))
        split["test"] += ids[:n_test]
        split["dev"] += rest[:n_dev]
        split["train"] += rest[n_dev:]
        for i, pid in enumerate(ids):
            folds[i % N_FOLDS].append(pid)

    cv = []
    by_id = {p["paper_id"]: p["source"] for p in papers}
    for k in range(N_FOLDS):
        test = set(folds[k])
        dev = []
        for source in sorted(set(by_id.values())):
            rest = sorted(pid for pid in by_id if by_id[pid] == source and pid not in test)
            rng.shuffle(rest)
            dev += rest[: max(1, round(DEV_FRAC * len(rest)))]
        train = sorted(set(by_id) - test - set(dev))
        cv.append({"train": train, "dev": sorted(dev), "test": sorted(test)})

    return {"seed": SEED, "main": {k: sorted(v) for k, v in split.items()}, "folds": cv}


LABEL_PAPERS = {"pipeline": 4, "single_pass": 6}
LABEL_MIX = {"linked": 12, "hard": 10, "other": 8}   # per paper, ~30 pairs; ~40% linked


def sample_label_pairs(papers: list, split: dict, rng: random.Random) -> list:
    """Blind pairs from test papers for human labelling (linked? which direction?).

    Mixes reference links, same-component unlinked pairs (hard negatives) and other unlinked pairs,
    with A/B order randomised so annotators can't infer direction from position. Only papers with a
    real arXiv id are used, so annotators can open the paper.
    """
    by_id = {p["paper_id"]: p for p in papers}
    out = []
    for source, n_papers in LABEL_PAPERS.items():
        ids = sorted(pid for pid in split["main"]["test"]
                     if by_id[pid]["source"] == source and re.fullmatch(r"\d{4}\.\d{4,5}", pid))
        rng.shuffle(ids)
        for pid in sorted(ids[:n_papers]):
            paper = by_id[pid]
            node_comp, text = {}, {}
            for ci, comp in enumerate(paper["components"]):
                for n in comp["nodes"]:
                    node_comp[n["id"]], text[n["id"]] = ci, n["description"]
            directed = {(e["from"], e["to"]) for c in paper["components"] for e in c["edges"]
                        if e["from"] in text and e["to"] in text and e["from"] != e["to"]}
            linked = {frozenset(e) for e in directed}
            nodes = sorted(text)
            unlinked = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:] if frozenset((a, b)) not in linked]
            hard = [pr for pr in unlinked if node_comp[pr[0]] == node_comp[pr[1]]]
            other = [pr for pr in unlinked if node_comp[pr[0]] != node_comp[pr[1]]]
            picks = [("linked", e) for e in rng.sample(sorted(directed), min(LABEL_MIX["linked"], len(directed)))]
            picks += [("hard", pr) for pr in rng.sample(hard, min(LABEL_MIX["hard"], len(hard)))]
            rest = sum(LABEL_MIX.values()) - len(picks)
            pool = other if len(other) >= rest else other + [pr for pr in hard if ("hard", pr) not in picks]
            picks += [("other", pr) for pr in rng.sample(pool, min(rest, len(pool)))]
            rng.shuffle(picks)
            for kind, (x, y) in picks:
                a, b = (x, y) if rng.random() < 0.5 else (y, x)
                ref_dir = None
                if kind == "linked":
                    ref_dir = "A->B" if (a, b) in directed else "B->A"
                out.append({"pair_id": f"L{len(out) + 1:03d}", "paper_id": pid, "source": source,
                            "a_id": a, "b_id": b, "claim_a": text[a], "claim_b": text[b],
                            "ref_linked": kind == "linked", "ref_direction": ref_dir, "kind": kind})
    return out


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


NOTEBOOK_CELLS = [
    md("""
# Retrieval suite __NUM__: __TITLE__

__INTRO__

Part of `docs/retrieval_suite/` (four notebooks, one Colab session each, replacing
`colab_bm25_tfidf_baseline.ipynb` and the `colab_train_claim_embedder_*.ipynb` notebooks). Every
notebook uses the same embedded dataset, split, metrics and code -- only the model list differs --
and also re-runs BM25 (~1 second) so its comparisons against BM25 work on their own. After running
a notebook, save it (Ctrl+S); `scripts/collect_retrieval_results.py` reads the saved outputs of
all four and merges them into one set of tables. The task is the same as before: given one claim
from a paper's proposition graph, can the model find the claims it is linked to (retrieval), and
can it group the paper's claims into the graph's real components (clustering)?

**What changed vs. the old notebooks**
- **One fixed split for every model**, built by `scripts/build_retrieval_suite.py` and embedded
  below. 200 papers (80 pipeline graphs + 120 single-pass graphs, one duplicate removed),
  stratified by source: 144 train / 16 dev / 40 test.
- **No test leakage in checkpoint selection.** The old E5/Qwen notebooks used `eval_loss` on the
  held-out papers to pick the best checkpoint. Here every fine-tuned model picks its checkpoint on
  the separate **dev** papers; test papers are only used for the final scores.
- **More open models** across the suite: MiniLM, mpnet, SPECTER, SPECTER2, E5, BGE (base/large),
  GTE-ModernBERT, mxbai-embed-large, Qwen3-Embedding-0.6B (fine-tuned) and Qwen3-Embedding-4B
  (zero-shot only).
- **More metrics, with confidence intervals** (section 4).

**Metrics** (all computed per test paper, then averaged over papers unless marked *pooled*)

| metric | what it asks | chance level |
|---|---|---|
| **Filtered MRR** (main) | for each linked pair, rank of the partner among the claims that are *not* linked to the anchor -- other true partners don't count as mistakes | depends on paper size |
| **Recall@10** | share of an anchor's linked claims in its top 10 (denominator capped at 10). Matches Stage 2.1's `candidate_top_k = 10` | depends on paper size |
| **nDCG@10**, **MAP** | rank-weighted versions of the same question, all partners at once | depends on paper size |
| **ROC-AUC** | across all claim pairs in the paper, are linked pairs scored above unlinked ones | always 0.5 -- fair across paper sizes |
| **Hard AUC** | ROC-AUC on pairs from the *same component* only (topically close, unlinked) | always 0.5 |
| **Recall@5 / MRR (legacy, pooled)** | the old directed, unfiltered metrics, pooled over edges -- kept only for comparison with the report | |
| **ARI / NMI** | k-means with the true number of components vs. the real components | ~0 |

Links are treated as undirected (cosine similarity has no direction). Clustering only means
something for pipeline graphs: 117 of the 120 single-pass graphs are one component, so
`n_cluster_papers` shows how many papers each ARI/NMI actually averages over. Main metrics get a
95% bootstrap interval over test papers, and section 9 runs **paired bootstrap tests** (fine-tuned
vs. zero-shot, every model vs. BM25) -- with 40 test papers, a gap whose interval crosses 0 is
not a real difference.

**How to run (VS Code + Colab extension):** Select Kernel → Colab → New Colab Server → GPU (T4).
Run all cells top to bottom, then **Ctrl+S** so the outputs are saved into this file. Results are
printed in full in the last cell. Colab's disk is wiped when the session ends; nothing here needs
to survive except the printed outputs.
"""),
    md("## 1. GPU check and install"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q -U sentence-transformers rank_bm25 scikit-learn datasets

import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: no GPU -- this is a CPU-only server. Create a new Colab server and pick GPU (T4).")
"""),
    md("""
## 2. Settings

Only one model is on the GPU at a time: each is loaded, scored zero-shot, fine-tuned, scored
again, then deleted before the next one loads. Finished (model, variant) rows are skipped on a
re-run, so re-running section 7 after a crash resumes where it stopped, as long as the runtime
itself wasn't reset. To skip a model, remove it from `RUN_MODELS`.

A model that fails (download error, out of memory, ...) is reported and skipped; the loop moves
on to the next one.

__GROUP_NOTE__**Training logs** (train/dev loss per step and which checkpoint was picked) are saved for every
fine-tuned run and printed in the last cell. **Checkpoints** are deleted after each model to save
disk; list models in `KEEP_MODELS` to copy their best fine-tuned weights to Google Drive (the
next cell mounts Drive only if that list is non-empty -- if the mount doesn't work from VS Code,
leave it empty and rely on the logs).
"""),
    code("""
SEED = 42
NOTEBOOK = "__NB_ID__"
RUN_MODELS = __RUN_MODELS__
SPLITS = __SPLITS__

# v2: checkpoint chosen by dev-paper filtered MRR + early stopping (v1 chose by dev loss, which
# kept rising after epoch 1 while training loss fell). The new file names make a re-run in the same
# runtime start fresh instead of resuming v1 rows.
RESULTS_PATH = "/content/results/results_v2.json"
PER_PAPER_PATH = "/content/results/per_paper_v2.json"
LOGS_PATH = "/content/results/train_logs_v2.json"
EARLY_STOP_PATIENCE = 3   # evaluations without a dev-score improvement before training stops

# fine-tuned models to keep (copied to Google Drive after training), e.g. ["mpnet", "qwen3"];
# empty = keep none. Checkpoints otherwise only live on Colab's disk and are deleted after each model.
KEEP_MODELS = []
KEEP_DIR = "/content/drive/MyDrive/claim_embedders"
CKPT_DIR = "/content/ckpt"
N_BOOT = 1000

USE_WANDB = False                       # True: stream training curves to Weights & Biases (see below)
WANDB_PROJECT = "claim-graph-retrieval"
"""),
    md("""
### Weights & Biases (optional)

With `USE_WANDB = True`, every fine-tuning run streams its training loss and dev scores to
[wandb.ai](https://wandb.ai) (one W&B run per model, in project `WANDB_PROJECT`), and the final
test scores are added to each run's summary. The next cell asks once for your API key -- copy it
from https://wandb.ai/authorize. Set `USE_WANDB = False` to skip it; the curves are also printed in
the notebook and saved in the logs either way.
"""),
    code("""
if USE_WANDB:
    !pip install -q -U wandb
    import os, wandb
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_LOG_MODEL"] = "false"   # don't upload checkpoints
    wandb.login()
else:
    print("USE_WANDB is False -- training curves stay in the notebook only")
"""),

    code("""
if KEEP_MODELS:
    from google.colab import drive
    drive.mount("/content/drive")
    import os
    os.makedirs(KEEP_DIR, exist_ok=True)
    print("fine-tuned models will be saved under", KEEP_DIR)
else:
    print("KEEP_MODELS is empty -- no checkpoints will be kept (logs still are)")
"""),
    md("## 3. Embedded dataset and split (generated -- don't edit)"),
    code("""
import base64, gzip, json

DATA_B64 = "__DATA_B64__"

_bundle = json.loads(gzip.decompress(base64.b64decode(DATA_B64)))
PAPERS = {p["paper_id"]: p for p in _bundle["papers"]}
SPLIT = _bundle["split"]
LABEL_PAIRS = _bundle.get("label_pairs", [])   # notebook 5 only: the human-labelling pairs (ids, no labels)
del DATA_B64, _bundle

main = SPLIT["main"]
assert not (set(main["train"]) & set(main["test"])) and not (set(main["dev"]) & set(main["test"]))
assert not (set(main["train"]) & set(main["dev"]))
for name in ("train", "dev", "test"):
    ids = main[name]
    by_src = {s: sum(PAPERS[i]["source"] == s for i in ids) for s in ("pipeline", "single_pass")}
    print(f"{name:5s}: {len(ids):3d} papers  {by_src}")
print(f"{len(SPLIT['folds'])} CV folds available")
"""),
    md("## 4. Pairs, metrics, and the BM25/TF-IDF baseline"),
    code("""
import random, re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

MAIN_METRICS = ["f_mrr", "recall@10", "auc"]
PAPER_METRICS = ["f_mrr", "recall@10", "ndcg@10", "map", "auc", "hard_auc", "ari", "nmi"]


def build_pairs(paper_ids, rng):
    \"\"\"(anchor, positive, hard negative) per real edge -- same construction as the old notebooks.

    Hard negative: a same-component node with no edge to the anchor, falling back to any other node
    in the paper when the component is too small. Pairs without any negative are dropped.
    \"\"\"
    pairs, seen = [], set()
    for pid in paper_ids:
        paper = PAPERS[pid]
        all_nodes = {n["id"]: n for c in paper["components"] for n in c["nodes"]}
        for comp in paper["components"]:
            nodes = {n["id"]: n for n in comp["nodes"]}
            edge_pairs = {(e["from"], e["to"]) for e in comp["edges"]}
            comp_ids = list(nodes)
            for e in comp["edges"]:
                frm, to = nodes.get(e["from"]), nodes.get(e["to"])
                if not frm or not to:
                    continue
                cands = [nid for nid in comp_ids if nid not in (e["from"], e["to"])
                         and (e["from"], nid) not in edge_pairs and (nid, e["from"]) not in edge_pairs]
                if not cands:
                    cands = [nid for nid in all_nodes if nid not in (e["from"], e["to"])]
                if not cands:
                    continue
                key = (frm["description"], to["description"])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((frm["description"], to["description"], all_nodes[rng.choice(cands)]["description"]))
    return pairs


def build_eval_papers(paper_ids):
    out = []
    for pid in paper_ids:
        paper = PAPERS[pid]
        node_ids, node_texts, comp_labels, edges = [], [], [], []
        for ci, comp in enumerate(paper["components"]):
            for n in comp["nodes"]:
                node_ids.append(n["id"])
                node_texts.append(n["description"])
                comp_labels.append(ci)
            edges += [(e["from"], e["to"]) for e in comp["edges"]]
        if len(node_texts) >= 2:
            out.append({"paper": pid, "source": paper["source"], "node_ids": node_ids,
                        "node_texts": node_texts, "comp_labels": comp_labels, "edges": edges})
    return out


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def score_paper(ep, sim, cluster_vecs):
    \"\"\"Every metric for one test paper. `sim` may be asymmetric (BM25); rows are the anchor side.\"\"\"
    n = len(ep["node_texts"])
    idx = {nid: i for i, nid in enumerate(ep["node_ids"])}
    comp = np.array(ep["comp_labels"])

    # legacy: directed edges, unfiltered ranks, pooled over edges later
    hits5, rrs = [], []
    nbrs = [set() for _ in range(n)]
    for frm, to in ep["edges"]:
        if frm not in idx or to not in idx or frm == to:
            continue
        i, j = idx[frm], idx[to]
        ranked = [r for r in np.argsort(-sim[i], kind="stable") if r != i]
        hits5.append(1.0 if j in ranked[:5] else 0.0)
        rrs.append(1.0 / (ranked.index(j) + 1))
        nbrs[i].add(j)
        nbrs[j].add(i)

    # per-anchor ranking metrics, links undirected
    f_rr, r10, nd10, aps = [], [], [], []
    for i in range(n):
        if not nbrs[i]:
            continue
        order = [r for r in np.argsort(-sim[i], kind="stable") if r != i]
        pos = [p for p, r in enumerate(order) if r in nbrs[i]]          # 0-based ranks of true partners
        f_rr += [1.0 / (p - k + 1) for k, p in enumerate(pos)]           # filtered: skip partners ranked above
        cap = min(10, len(pos))
        r10.append(sum(p < 10 for p in pos) / cap)
        nd10.append(sum(1 / np.log2(p + 2) for p in pos if p < 10) / sum(1 / np.log2(t + 2) for t in range(cap)))
        aps.append(np.mean([(k + 1) / (p + 1) for k, p in enumerate(pos)]))

    # pairwise AUC over all claim pairs, and over same-component pairs only
    sym = (sim + sim.T) / 2
    iu, ju = np.triu_indices(n, 1)
    labels = np.array([j in nbrs[i] for i, j in zip(iu, ju)])
    scores = sym[iu, ju]
    auc = roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else None
    same = comp[iu] == comp[ju]
    hard_auc = roc_auc_score(labels[same], scores[same]) if 0 < labels[same].sum() < same.sum() else None

    ari = nmi = None
    k = len(set(ep["comp_labels"]))
    if 1 < k < n:
        pred = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(cluster_vecs)
        ari = adjusted_rand_score(ep["comp_labels"], pred)
        nmi = normalized_mutual_info_score(ep["comp_labels"], pred)

    return {"paper": ep["paper"], "source": ep["source"], "hits5": hits5, "rrs": rrs,
            "f_mrr": _mean(f_rr), "recall@10": _mean(r10), "ndcg@10": _mean(nd10), "map": _mean(aps),
            "auc": auc, "hard_auc": hard_auc, "ari": ari, "nmi": nmi}


def boot_ci(vals, n_boot=N_BOOT):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(SEED)
    means = vals[rng.integers(0, len(vals), (n_boot, len(vals)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def aggregate(per_paper):
    \"\"\"Paper-averaged metrics (+ bootstrap CI for the main ones) and pooled legacy metrics, per source.\"\"\"
    out = {}
    for src in ("all", "pipeline", "single_pass"):
        ps = [p for p in per_paper if src == "all" or p["source"] == src]
        row = {"n_papers": len(ps), "n_edges": sum(len(p["hits5"]) for p in ps)}
        for m in PAPER_METRICS:
            vals = [p[m] for p in ps if p[m] is not None]
            row[m] = _mean(vals)
            if m in MAIN_METRICS:
                row[m + "_ci"] = boot_ci(vals)
        row["n_cluster_papers"] = sum(p["ari"] is not None for p in ps)
        row["recall@5_legacy"] = _mean([h for p in ps for h in p["hits5"]])
        row["mrr_legacy"] = _mean([r for p in ps for r in p["rrs"]])
        out[src] = row
    return out


def _slim(per_paper):
    return [{k: p[k] for k in ["paper", "source"] + PAPER_METRICS} for p in per_paper]


def evaluate_embedder(model, eval_papers, prefix=""):
    per_paper = []
    for ep in eval_papers:
        emb = model.encode([prefix + t for t in ep["node_texts"]], convert_to_numpy=True, batch_size=32)
        per_paper.append(score_paper(ep, cosine_similarity(emb), emb))
    return aggregate(per_paper), _slim(per_paper)


def tokenize(text):
    # same tokenizer as src/claim_pipeline/resolution/retrieval.py's BM25Retriever
    return re.findall(r"\\w+", text.lower())


def evaluate_bm25(eval_papers):
    from rank_bm25 import BM25Okapi
    per_paper = []
    for ep in eval_papers:
        toks = [tokenize(t) for t in ep["node_texts"]]
        bm25 = BM25Okapi(toks)
        sim = np.array([bm25.get_scores(t) for t in toks])
        tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None).fit_transform(ep["node_texts"]).toarray()
        per_paper.append(score_paper(ep, sim, tfidf))
    return aggregate(per_paper), _slim(per_paper)


print("test papers usable for eval:", len(build_eval_papers(main["test"])))
"""),
    md("""
## 5. Model recipes

mpnet, SPECTER, E5 and both Qwen3-0.6B recipes are the ones the old per-model notebooks landed on.
The new models start from a plain, untuned recipe (base size: lr 1e-5, batch 32; large: lr 5e-6,
batch 16, fp16) -- if one regresses below its zero-shot score, that's a recipe to tune, not a
verdict on the model. Every fine-tuned run selects its checkpoint by **filtered MRR on the dev
papers** (evaluated after every epoch, or every 20 steps for the 3-epoch Qwen recipe) and stops
early once that score hasn't improved for `EARLY_STOP_PATIENCE` evaluations -- the epoch counts
below are ceilings, not targets. The first run of this suite selected by dev *loss* instead: it
bottomed out after epoch 1 and then rose while training loss kept falling (overfitting on ~3.3k
pairs), and with only 354 dev pairs it is a noisy stand-in for the retrieval metric we report. No instruction prefixes except E5's required `"query: "` -- the task is symmetric, and
query-side instructions are meant for asymmetric search.
"""),
    code("""
from sentence_transformers import SentenceTransformer, models


def plain(model_id, **kw):
    # load fp32 master weights even when the checkpoint is stored in fp16/bf16 (Transformers v5 keeps
    # the checkpoint's dtype by default, and fp16 AMP training then fails with "Attempting to unscale
    # FP16 gradients" -- seen with mxbai-embed-large in notebook 2)
    kw.setdefault("model_kwargs", {"dtype": torch.float32})
    def load():
        m = SentenceTransformer(model_id, **kw)
        m.max_seq_length = min(m.max_seq_length or 256, 256)  # claims are single sentences
        return m
    return load


def cls_pooled(model_id):
    # for checkpoints without a sentence-transformers config: [CLS] pooling, as SPECTER/SPECTER2 were trained
    def load():
        word = models.Transformer(model_id, max_seq_length=256)
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
        return SentenceTransformer(modules=[word, pool])
    return load


_epoch_eval = dict(eval_strategy="epoch", save_strategy="best")
_base = dict(per_device_train_batch_size=32, **_epoch_eval)
_large = dict(num_train_epochs=4, per_device_train_batch_size=16, learning_rate=5e-6, fp16=True, **_epoch_eval)
# Qwen3: fp32 master weights (checkpoint ships bf16, which T4 can't train in) + fp16 AMP, and gradient
# checkpointing -- needed: without it 0.6B runs out of memory on a T4 (notebook 3). It is switched on
# directly on the underlying Hugging Face model ("grad_ckpt" below), not via the trainer's
# gradient_checkpointing=True, because current Transformers passes `every_n_layers` to
# sentence-transformers' wrapper, which rejects it (notebook 2 crashed on this).
_qwen_load = plain("Qwen/Qwen3-Embedding-0.6B")
_qwen_common = dict(per_device_train_batch_size=8, gradient_accumulation_steps=2, fp16=True,
                    learning_rate=6e-6)

RECIPES = {
    # group A
    "minilm":         {"label": "all-MiniLM-L6-v2", "load": plain("sentence-transformers/all-MiniLM-L6-v2"),
                       "args": dict(num_train_epochs=8, learning_rate=2e-5, **_base)},
    "mpnet":          {"label": "all-mpnet-base-v2", "load": plain("sentence-transformers/all-mpnet-base-v2"),
                       "args": dict(num_train_epochs=8, learning_rate=1e-5, **_base)},
    "specter":        {"label": "SPECTER", "load": cls_pooled("allenai/specter"),
                       "args": dict(num_train_epochs=12, learning_rate=7e-6, **_base)},
    "specter2":       {"label": "SPECTER2 (base)", "load": cls_pooled("allenai/specter2_base"),
                       "args": dict(num_train_epochs=8, learning_rate=1e-5, **_base)},
    "e5":             {"label": "e5-base-v2", "load": plain("intfloat/e5-base-v2"), "prefix": "query: ",
                       "args": dict(num_train_epochs=6, learning_rate=6e-6, **_base)},
    "bge_base":       {"label": "bge-base-en-v1.5", "load": plain("BAAI/bge-base-en-v1.5"),
                       "args": dict(num_train_epochs=6, learning_rate=1e-5, **_base)},
    "gte_modernbert": {"label": "gte-modernbert-base", "load": plain("Alibaba-NLP/gte-modernbert-base"),
                       "args": dict(num_train_epochs=6, learning_rate=1e-5, **_base)},
    # first run at lr 1e-5 collapsed after one epoch (dev fMRR 0.458 zero-shot -> 0.341), unlike every
    # other base model at the same settings -- retried with a much smaller learning rate
    "gte_modernbert_lr2e6": {"label": "gte-modernbert-base (lr 2e-6)", "load": plain("Alibaba-NLP/gte-modernbert-base"),
                             "args": dict(num_train_epochs=6, learning_rate=2e-6, **_base)},
    # group B
    "bge_large":      {"label": "bge-large-en-v1.5", "load": plain("BAAI/bge-large-en-v1.5"), "args": _large},
    "mxbai_large":    {"label": "mxbai-embed-large-v1", "load": plain("mixedbread-ai/mxbai-embed-large-v1"), "args": _large},
    "qwen3":          {"label": "Qwen3-Embedding-0.6B", "load": _qwen_load, "grad_ckpt": True,
                       "args": dict(num_train_epochs=3, weight_decay=0.01, eval_strategy="steps", eval_steps=20,
                                    save_strategy="best", **_qwen_common)},
    "qwen3_6ep":      {"label": "Qwen3-Embedding-0.6B (6 ep)", "load": _qwen_load, "grad_ckpt": True,
                       "args": dict(num_train_epochs=6, **_epoch_eval, **_qwen_common)},
    # group C / optional: evaluation only
    "qwen3_4b_zs":    {"label": "Qwen3-Embedding-4B", "zero_shot_only": True,
                       "load": plain("Qwen/Qwen3-Embedding-4B", model_kwargs={"dtype": torch.float16})},
    "embeddinggemma_zs": {"label": "EmbeddingGemma-300M", "zero_shot_only": True,
                          "load": plain("google/embeddinggemma-300m")},  # no fp16: unsupported by this model
}
"""),
    md("## 6. Runner (zero-shot eval → fine-tune on train, select on dev → eval on test)"),
    code("""
import gc, os, shutil, time, traceback
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from transformers import EarlyStoppingCallback
try:
    from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
except ImportError:
    from sentence_transformers.evaluation import SentenceEvaluator

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
try:
    with open(RESULTS_PATH) as f:
        RESULTS = json.load(f)
    with open(PER_PAPER_PATH) as f:
        PER_PAPER = json.load(f)
except FileNotFoundError:
    RESULTS, PER_PAPER = [], {}
try:
    with open(LOGS_PATH) as f:
        LOGS = json.load(f)
except FileNotFoundError:
    LOGS = {}


class DevRetrievalEvaluator(SentenceEvaluator):
    \"\"\"Filtered MRR / AUC on the dev papers -- the checkpoint-selection metric (see section 5).\"\"\"

    def __init__(self, paper_ids, prefix):
        super().__init__()
        self.papers = build_eval_papers(paper_ids)
        self.prefix = prefix
        self.name = "dev"
        self.primary_metric = "dev_f_mrr"
        self.greater_is_better = True

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        scores, _ = evaluate_embedder(model, self.papers, self.prefix)
        return {"dev_f_mrr": scores["all"]["f_mrr"], "dev_auc": scores["all"]["auc"]}


SELECT_METRIC = "eval_dev_f_mrr"


def _save_logs(run_id, trainer, minutes, dev_zero_shot):
    st = trainer.state
    keep = ("step", "epoch", "loss", "eval_loss", SELECT_METRIC, "eval_dev_auc", "learning_rate", "grad_norm")
    history = [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in h.items() if k in keep}
               for h in st.log_history]
    evals = [h for h in history if SELECT_METRIC in h]
    best = max(evals, key=lambda h: h[SELECT_METRIC]) if evals else None
    LOGS[run_id] = {"select_metric": SELECT_METRIC, "dev_f_mrr_zero_shot": dev_zero_shot,
                    "best_checkpoint": st.best_model_checkpoint, "best_dev_f_mrr": st.best_metric,
                    "best_at_epoch": best and best.get("epoch"), "best_at_step": best and best.get("step"),
                    "total_steps": st.global_step, "epochs_ceiling": trainer.args.num_train_epochs,
                    "stopped_early": bool(evals) and st.global_step < st.max_steps,
                    "train_minutes": round(minutes, 1), "history": history}
    with open(LOGS_PATH, "w") as f:
        json.dump(LOGS, f)
    if best:
        curve = "  ".join(f"{h['epoch']:.1f}:{h[SELECT_METRIC]:.3f}" for h in evals)
        print(f"  dev fMRR zero-shot {dev_zero_shot:.4f} -> best {best[SELECT_METRIC]:.4f} at epoch "
              f"{best['epoch']:.2f} (step {best['step']}/{st.global_step}"
              f"{', stopped early' if LOGS[run_id]['stopped_early'] else ''})")
        print(f"  dev fMRR by epoch: {curve}")


def _run_id(split_name, model_key, variant):
    return f"{split_name}|{model_key}|{variant}"


def _done(split_name, model_key, variant):
    return _run_id(split_name, model_key, variant) in PER_PAPER


def _fmt(v):
    return "  n/a " if v is None else f"{v:.4f}"


def _record(split_name, model_key, label, variant, result, minutes=None):
    scores, per_paper = result
    for src, s in scores.items():
        row = {"split": split_name, "model_key": model_key, "model": label, "variant": variant, "source": src,
               **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}}
        if minutes is not None:
            row["train_minutes"] = round(minutes, 1)
        RESULTS.append(row)
    PER_PAPER[_run_id(split_name, model_key, variant)] = per_paper
    with open(RESULTS_PATH, "w") as f:
        json.dump(RESULTS, f, indent=1)
    with open(PER_PAPER_PATH, "w") as f:
        json.dump(PER_PAPER, f)
    a = scores["all"]
    print(f"  [{split_name}] {label:28s} {variant:10s} fMRR {_fmt(a['f_mrr'])}  R@10 {_fmt(a['recall@10'])}  "
          f"AUC {_fmt(a['auc'])}  hardAUC {_fmt(a['hard_auc'])}  ARI {_fmt(a['ari'])}  (legacy R@5 {_fmt(a['recall@5_legacy'])})")


def _to_ds(pairs, prefix):
    return Dataset.from_dict({"anchor": [prefix + a for a, _, _ in pairs],
                              "positive": [prefix + b for _, b, _ in pairs],
                              "negative": [prefix + c for _, _, c in pairs]})


def _run_model(split_name, split, key, eval_papers):
    r = RECIPES[key]
    prefix = r.get("prefix", "")
    model = trainer = hf = None
    out_dir = f"{CKPT_DIR}/{split_name}_{key}"
    try:
        print(f"== {split_name}: {r['label']}")
        model = r["load"]()
        if torch.cuda.is_available():
            model = model.to("cuda")
        if not _done(split_name, key, "zero-shot"):
            _record(split_name, key, r["label"], "zero-shot", evaluate_embedder(model, eval_papers, prefix))
        if r.get("zero_shot_only") or _done(split_name, key, "fine-tuned"):
            return

        rng = random.Random(SEED)
        train_pairs, dev_pairs = build_pairs(split["train"], rng), build_pairs(split["dev"], rng)
        print(f"  train pairs {len(train_pairs)}  dev pairs {len(dev_pairs)}")
        if r.get("grad_ckpt"):
            hf = getattr(model[0], "auto_model", None) or getattr(model[0], "model")
            hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            hf.config.use_cache = False
            print("  gradient checkpointing on (set on the Hugging Face model directly)")
        dev_eval = DevRetrievalEvaluator(split["dev"], prefix)
        dev_zero_shot = dev_eval(model)["dev_f_mrr"]
        args = SentenceTransformerTrainingArguments(
            output_dir=out_dir, warmup_steps=0.1, logging_steps=10, save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model=SELECT_METRIC, greater_is_better=True,
            report_to="wandb" if USE_WANDB else "none", run_name=f"{NOTEBOOK}-{split_name}-{key}",
            seed=SEED, **r["args"])
        trainer = SentenceTransformerTrainer(
            model=model, args=args, loss=MultipleNegativesRankingLoss(model),
            train_dataset=_to_ds(train_pairs, prefix), eval_dataset=_to_ds(dev_pairs, prefix),
            evaluator=dev_eval, callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)])
        t0 = time.time()
        trainer.train()
        minutes = (time.time() - t0) / 60
        _save_logs(_run_id(split_name, key, "fine-tuned"), trainer, minutes, dev_zero_shot)
        _record(split_name, key, r["label"], "fine-tuned", evaluate_embedder(model, eval_papers, prefix), minutes)
        if USE_WANDB and wandb.run is not None:
            a = [row for row in RESULTS if row["split"] == split_name and row["model_key"] == key
                 and row["variant"] == "fine-tuned" and row["source"] == "all"][-1]
            wandb.run.summary.update({f"test/{m}": a[m] for m in ("f_mrr", "recall@10", "auc", "hard_auc", "ari")}
                                     | {"test/zero_shot_f_mrr": [row for row in RESULTS if row["split"] == split_name
                                        and row["model_key"] == key and row["variant"] == "zero-shot"
                                        and row["source"] == "all"][-1]["f_mrr"], "train_minutes": minutes})
        if key in KEEP_MODELS and split_name == "main":
            model.save(f"{KEEP_DIR}/{key}")  # the best (dev-selected) weights, loaded by load_best_model_at_end
            print(f"  saved to {KEEP_DIR}/{key}")
    except Exception:
        print(f"  !! {r['label']} failed -- skipped, moving on:")
        traceback.print_exc(limit=3)
    finally:
        if USE_WANDB:
            wandb.finish()   # one W&B run per model
        shutil.rmtree(out_dir, ignore_errors=True)  # checkpoints aren't kept -- Qwen's are ~1.2 GB each
        # drop every reference before loading the next model, so only one model is ever on the GPU
        # (hf too: in notebook 3b it still pointed at Qwen here, so cleanup reported 2.24 GiB)
        del trainer, model, hf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  GPU memory still allocated after cleanup: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")


def run_split(split_name, split, model_keys):
    eval_papers = build_eval_papers(split["test"])
    for key in model_keys:
        if key == "bm25":
            if not _done(split_name, "bm25", "lexical"):
                _record(split_name, "bm25", "BM25 / TF-IDF", "lexical", evaluate_bm25(eval_papers))
            continue
        variants = ["zero-shot"] if RECIPES[key].get("zero_shot_only") else ["zero-shot", "fine-tuned"]
        if all(_done(split_name, key, v) for v in variants):
            continue
        _run_model(split_name, split, key, eval_papers)
"""),
    md("## 7. Run (__RUN_DESC__)"),
    code("""
for split_name in SPLITS:
    split = SPLIT["main"] if split_name == "main" else SPLIT["folds"][int(split_name.removeprefix("fold"))]
    run_split(split_name, split, RUN_MODELS)
"""),
    md("""
## 8. Results tables and paired significance tests

Paired bootstrap: resample the 40 test papers with replacement, recompute the mean per-paper
difference between two runs, and report its 95% interval. If the interval excludes 0, the gap
holds up at this test-set size; if it crosses 0, treat the two as tied.
"""),
    code("""
import pandas as pd

pd.set_option("display.width", 250)
df = pd.DataFrame(RESULTS)
main_df = df[df["split"] == "main"]
ret_cols = ["model", "variant", "f_mrr", "f_mrr_ci", "recall@10", "recall@10_ci", "ndcg@10", "map",
            "recall@5_legacy", "mrr_legacy"]
pair_cols = ["model", "variant", "auc", "auc_ci", "hard_auc", "ari", "nmi", "n_cluster_papers", "train_minutes"]
for src in ("all", "pipeline", "single_pass"):
    d = main_df[main_df["source"] == src].sort_values("f_mrr", ascending=False)
    print(f"\\n=== main split, test source = {src}: retrieval ===")
    print(d[ret_cols].to_string(index=False))
    print(f"--- {src}: pairwise + clustering ---")
    print(d[[c for c in pair_cols if c in d]].to_string(index=False))


def paired(run_a, run_b, metric):
    a = {p["paper"]: p[metric] for p in PER_PAPER.get(run_a, []) if p[metric] is not None}
    b = {p["paper"]: p[metric] for p in PER_PAPER.get(run_b, []) if p[metric] is not None}
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    diff = np.array([a[k] - b[k] for k in common])
    ci = boot_ci(diff)
    return {"diff": round(float(diff.mean()), 4), "ci": ci, "holds": ci[0] > 0 or ci[1] < 0, "n": len(common)}


rows = []
for rid in PER_PAPER:
    split_name, key, variant = rid.split("|")
    if split_name != "main" or key == "bm25":
        continue
    label = RECIPES[key]["label"]
    comps = [("vs BM25", _run_id("main", "bm25", "lexical"))]
    if variant == "fine-tuned":
        comps.append(("vs own zero-shot", _run_id("main", key, "zero-shot")))
    for name, other in comps:
        for m in MAIN_METRICS:
            res = paired(rid, other, m)
            if res:
                rows.append({"model": label, "variant": variant, "comparison": name, "metric": m, **res})
SIGNIF = rows
if rows:
    print("\\n=== paired bootstrap (main split, all test papers): run minus comparison ===")
    print(pd.DataFrame(rows).to_string(index=False))

cv_df = df[df["split"].str.startswith("fold") & (df["source"] == "all")]
if len(cv_df):
    agg = cv_df.groupby(["model", "variant"])[MAIN_METRICS].agg(["mean", "std"]).round(4)
    print("\\n=== 5-fold CV, mean / std over folds (all sources) ===")
    print(agg.to_string())
"""),
    md("## 9. Charts (main split, all test papers; error bars = 95% bootstrap CI)"),
    code("""
import matplotlib.pyplot as plt

d = main_df[main_df["source"] == "all"].sort_values("f_mrr").reset_index(drop=True)
colors = {"lexical": "#7f7f7f", "zero-shot": "#6baed6", "fine-tuned": "#2166ac"}
names = d["model"] + " (" + d["variant"] + ")"
panels = [("f_mrr", "Filtered MRR"), ("recall@10", "Recall@10"), ("auc", "ROC-AUC (chance 0.5)"),
          ("hard_auc", "Hard AUC, same-component pairs"), ("ari", "Clustering ARI (pipeline graphs)"),
          ("recall@5_legacy", "Legacy Recall@5 (pooled)")]
fig, axes = plt.subplots(2, 3, figsize=(18, 0.35 * len(d) * 2 + 3))
for ax, (m, title) in zip(axes.flat, panels):
    vals = d[m].astype(float)
    xerr = None
    if m + "_ci" in d:
        lo = [v - c[0] if c else 0 for v, c in zip(vals, d[m + "_ci"])]
        hi = [c[1] - v if c else 0 for v, c in zip(vals, d[m + "_ci"])]
        xerr = [lo, hi]
    ax.barh(range(len(d)), vals, xerr=xerr, color=[colors[v] for v in d["variant"]], capsize=2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(names if ax in axes[:, 0] else [], fontsize=7)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("/content/results/main_split.png", dpi=150)
plt.show()
"""),
    md("## 10. Full results (read back by `scripts/collect_retrieval_results.py`)"),
    code("""
print("NOTEBOOK", NOTEBOOK)
print("RESULTS_JSON_BEGIN")
print(json.dumps(RESULTS))
print("RESULTS_JSON_END")
print("SIGNIF_JSON_BEGIN")
print(json.dumps(SIGNIF))
print("SIGNIF_JSON_END")
print("PER_PAPER_JSON_BEGIN")
print(json.dumps(PER_PAPER))
print("PER_PAPER_JSON_END")
print("TRAIN_LOGS_JSON_BEGIN")
print(json.dumps(LOGS))
print("TRAIN_LOGS_JSON_END")
"""),
]

def _shared(marker: str) -> dict:
    """A cell from NOTEBOOK_CELLS, found by a snippet of its text."""
    hits = [c for c in NOTEBOOK_CELLS if marker in "".join(c["source"])]
    assert len(hits) == 1, marker
    return hits[0]


NOTEBOOK_05_CELLS = [
    md("""
# Retrieval suite 5/5: edge classifier and whole-graph reconstruction

Notebooks 1-4 test **retrieval**: can a model find a claim's linked partners? This notebook tests
the rest of what a graph needs, on the same embedded split (144 train / 16 dev / 40 test papers):

1. **Is there a link?** For every pair of claims in a test paper, a yes/no decision.
2. **Which way does it point?** For every true link, is the prerequisite on the correct side? This
   targets the one real defect found in the pipeline's graphs (reversed edges such as
   `P11->P12/P13`); similarity scores are symmetric and cannot answer it at all.
3. **The whole graph.** Build a graph per test paper (decide links, set directions, break cycles
   the same way `repair_cycles()`'s fallback does -- drop the weakest edge in each cycle) and compare
   it with the reference graph.

**Models.** Cross-encoders read both claims together, so unlike the embedding models they can
learn direction. Each is trained as one 3-way classifier on an *ordered* pair (A, B):
`0 = no link`, `1 = A -> B`, `2 = B -> A`. Each training link appears in both orders (labels 1
and 2), plus 3 unlinked pairs per link (2 from the same component -- topically close -- and 1
from anywhere in the paper). Scores are averaged over both input orders at test time.

| key | model | why |
|---|---|---|
| `deberta_nli` | DeBERTa-v3-base, from an NLI checkpoint | entailment-style training is close to "A is a prerequisite for B" |
| `scibert` | SciBERT | pre-trained on scientific papers |
| `roberta` | RoBERTa-base | general-purpose reference |

**Baselines** (no training): BM25 and zero-shot E5 (the strongest zero-shot embedder in
notebook 1) for links, and an **earlier-claim-first** rule for direction (claim ids follow the
order the claims were extracted in). That rule is a very strong baseline: in the reference graphs
about 96% of links point from the earlier claim to the later one (99% for single-pass graphs,
93% for pipeline graphs). The classifiers only see claim text, never ids or order, so they can't
use it -- the informative direction number is **`dir_acc_backward`**: accuracy on the few links
that point from a later claim to an earlier one, where the rule is wrong by construction.

**Metrics**, each averaged over test papers with a 95% bootstrap interval:
- link: ROC-AUC and average precision (AP) over all claim pairs in the paper;
- direction: accuracy over the true links (chance 0.5);
- graph: undirected and directed edge precision / recall / F1, component ARI (pipeline graphs),
  cycles broken. The link threshold (or edge density, for score-only baselines) is tuned on the
  dev papers, never on test.

Checkpoints are chosen on the dev papers (mean of link AUC and direction accuracy) with early
stopping. The last cell also scores the 300 human-labelling pairs (`docs/retrieval_suite/
human_labels/`) so model predictions can later be compared with the authors' labels.

**How to run:** Select Kernel → Colab → New Colab Server → GPU (T4), run all cells, **Ctrl+S**.
Roughly 15-30 minutes per cross-encoder; remove a key from `RUN_CE` to skip it.

**Caveat:** the reference graphs are LLM output (Gemini pipeline / single-pass), not human gold --
every score here is agreement with those graphs. The human labels are what tell us how good the
reference itself is.
"""),
    md("## 1. GPU check and install"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q -U sentence-transformers transformers accelerate datasets sentencepiece rank_bm25 scikit-learn

import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: no GPU -- this is a CPU-only server. Create a new Colab server and pick GPU (T4).")
"""),
    md("## 2. Settings"),
    code("""
SEED = 42
N_BOOT = 1000
NOTEBOOK = "edge_classifier"

CE_MODELS = {
    "deberta_nli": {"label": "DeBERTa-v3-base (NLI init)", "id": "cross-encoder/nli-deberta-v3-base", "fp16": False},
    "scibert":     {"label": "SciBERT", "id": "allenai/scibert_scivocab_uncased", "fp16": True},
    "roberta":     {"label": "RoBERTa-base", "id": "FacebookAI/roberta-base", "fp16": True},
}
RUN_CE = ["deberta_nli", "scibert", "roberta"]   # DeBERTa-v3 trains in fp32: fp16 is unstable for it

EMBED_BASELINE, EMBED_PREFIX = "intfloat/e5-base-v2", "query: "
NEG_PER_LINK = 3          # unlinked training pairs per link: 2 same-component + 1 anywhere in the paper
EPOCHS = 5                # ceiling -- early stopping picks the best epoch on dev
LR = 2e-5
BATCH = 32
MAX_LEN = 160
EARLY_STOP_PATIENCE = 2
CANDIDATE_TOP_K = 10      # Stage 2.1's candidate_top_k, for the "E5 top-10 candidates" variant

RESULTS_PATH = "/content/results/edge_results.json"
CKPT_DIR = "/content/ckpt_edge"

USE_WANDB = False                       # True: stream training curves to Weights & Biases (see below)
WANDB_PROJECT = "claim-graph-retrieval"
"""),
    md("""
### Weights & Biases (optional)

With `USE_WANDB = True`, every fine-tuning run streams its training loss and dev scores to
[wandb.ai](https://wandb.ai) (one W&B run per model, in project `WANDB_PROJECT`), and the final
test scores are added to each run's summary. The next cell asks once for your API key -- copy it
from https://wandb.ai/authorize. Set `USE_WANDB = False` to skip it; the curves are also printed in
the notebook and saved in the logs either way.
"""),
    code("""
if USE_WANDB:
    !pip install -q -U wandb
    import os, wandb
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_LOG_MODEL"] = "false"   # don't upload checkpoints
    wandb.login()
else:
    print("USE_WANDB is False -- training curves stay in the notebook only")
"""),

    _shared("## 3. Embedded dataset and split"),
    _shared('DATA_B64 = "__DATA_B64__"'),
    md("## 4. Shared helpers (same code as notebooks 1-4)"),
    _shared("def build_pairs(paper_ids, rng):"),
    md("## 5. Paper structures, training examples, graph building and metrics"),
    code("""
import re, random
from collections import defaultdict
from sklearn.metrics import average_precision_score

def node_num(nid):
    m = re.search(r"\\d+", nid)
    return int(m.group()) if m else 0


def structs(paper_ids):
    out = {}
    for ep in build_eval_papers(paper_ids):
        idx = {nid: i for i, nid in enumerate(ep["node_ids"])}
        directed = {(idx[f], idx[t]) for f, t in ep["edges"] if f in idx and t in idx and f != t}
        ep.update(idx=idx, n=len(ep["node_ids"]), directed=directed, und={frozenset(e) for e in directed})
        out[ep["paper"]] = ep
    return out


S_TRAIN, S_DEV, S_TEST = structs(SPLIT["main"]["train"]), structs(SPLIT["main"]["dev"]), structs(SPLIT["main"]["test"])
print(f"papers: train {len(S_TRAIN)}  dev {len(S_DEV)}  test {len(S_TEST)}; "
      f"test links {sum(len(e['directed']) for e in S_TEST.values())}")


def make_examples(S, rng):
    \"\"\"Ordered (A, B, label) examples: 0 no link, 1 A->B, 2 B->A.\"\"\"
    ex = []
    for ep in S.values():
        T, comp, n = ep["node_texts"], ep["comp_labels"], ep["n"]
        for i, j in sorted(ep["directed"]):
            ex += [(T[i], T[j], 1), (T[j], T[i], 2)]
            free = [k for k in range(n) if k not in (i, j) and frozenset((i, k)) not in ep["und"]]
            same = [k for k in free if comp[k] == comp[i]]
            negs = rng.sample(same, min(2, len(same)))
            rest = [k for k in free if k not in negs]
            negs += rng.sample(rest, min(NEG_PER_LINK - len(negs), len(rest)))
            for k in negs:
                ex.append((T[i], T[k], 0) if rng.random() < 0.5 else (T[k], T[i], 0))
    rng.shuffle(ex)
    return ex


def id_order_fwd(ep):
    \"\"\"Earlier-claim-first direction rule: fwd[i, j] = 1 if claim i comes before claim j.\"\"\"
    nums = np.array([node_num(nid) for nid in ep["node_ids"]])
    return (nums[:, None] < nums[None, :]).astype(float)


def find_cycle(n, edges):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
    color = [0] * n
    for s in range(n):
        if color[s]:
            continue
        stack, path = [(s, iter(adj[s]))], [s]
        color[s] = 1
        while stack:
            v, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[v] = 2
                stack.pop()
                path.pop()
            elif color[nxt] == 1:
                cyc = path[path.index(nxt):] + [nxt]
                return list(zip(cyc[:-1], cyc[1:]))
            elif color[nxt] == 0:
                color[nxt] = 1
                stack.append((nxt, iter(adj[nxt])))
                path.append(nxt)
    return None


def build_graph(ep, link, fwd, mode, param, allowed=None):
    \"\"\"Pick links (top round(param * n) pairs for mode="density", or link >= param for "threshold"),
    orient each by fwd, then break cycles by dropping the weakest edge in each one.\"\"\"
    n = ep["n"]
    iu, ju = np.triu_indices(n, 1)
    cand = sorted(((link[i, j], i, j) for i, j in zip(iu, ju) if allowed is None or allowed[i, j]), reverse=True)
    chosen = cand[:max(1, round(param * n))] if mode == "density" else [c for c in cand if c[0] >= param]
    edges = {((i, j) if fwd[i, j] >= fwd[j, i] else (j, i)): s for s, i, j in chosen}
    removed = 0
    while (cyc := find_cycle(n, edges)):
        del edges[min(cyc, key=lambda e: edges[e])]
        removed += 1
    return edges, removed


def _prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def graph_metrics(ep, edges, removed):
    pred_d = set(edges)
    up, ur, uf = _prf({frozenset(e) for e in pred_d}, ep["und"])
    dp, dr, df = _prf(pred_d, ep["directed"])
    parent = list(range(ep["n"]))
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pred_d:
        parent[root(a)] = root(b)
    ari = (adjusted_rand_score(ep["comp_labels"], [root(i) for i in range(ep["n"])])
           if len(set(ep["comp_labels"])) > 1 else None)
    return {"und_p": up, "und_r": ur, "und_f1": uf, "dir_p": dp, "dir_r": dr, "dir_f1": df,
            "comp_ari": ari, "pred_edges": len(pred_d), "gold_edges": len(ep["directed"]), "cycles_broken": removed}


def pair_metrics(ep, link, fwd):
    n = ep["n"]
    iu, ju = np.triu_indices(n, 1)
    y = np.array([frozenset((i, j)) in ep["und"] for i, j in zip(iu, ju)])
    s = link[iu, ju]
    has_both = 0 < y.sum() < len(y)
    dirs = [1.0 if fwd[i, j] > fwd[j, i] else 0.0 for i, j in ep["directed"]]
    nums = [node_num(nid) for nid in ep["node_ids"]]
    back = [1.0 if fwd[i, j] > fwd[j, i] else 0.0 for i, j in ep["directed"] if nums[i] > nums[j]]
    return {"link_auc": roc_auc_score(y, s) if has_both else None,
            "link_ap": average_precision_score(y, s) if y.sum() else None,
            "dir_acc": _mean(dirs), "dir_acc_backward": _mean(back), "n_backward": len(back)}


DENSITY_GRID = np.round(np.arange(0.2, 2.01, 0.05), 3)
THRESH_GRID = np.round(np.arange(0.05, 0.96, 0.025), 3)


def tune(S, scores, mode, allowed_fn=None):
    \"\"\"Pick the density / threshold with the best mean undirected F1 on the dev papers.\"\"\"
    grid = DENSITY_GRID if mode == "density" else THRESH_GRID
    best = None
    for param in grid:
        f1s = []
        for pid, ep in S.items():
            link, fwd = scores[pid]
            edges, removed = build_graph(ep, link, fwd, mode, param, allowed_fn(pid) if allowed_fn else None)
            f1s.append(graph_metrics(ep, edges, removed)["und_f1"])
        if best is None or np.mean(f1s) > best[1]:
            best = (float(param), float(np.mean(f1s)))
    return best
"""),
    md("## 6. Baseline scores (BM25, zero-shot E5, earlier-claim-first direction)"),
    code("""
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

SCORES = {}   # method key -> {"dev": {pid: (link, fwd)}, "test": {...}}


def bm25_scores(S):
    out = {}
    for pid, ep in S.items():
        toks = [tokenize(t) for t in ep["node_texts"]]
        bm25 = BM25Okapi(toks)
        sim = np.array([bm25.get_scores(t) for t in toks])
        sym = (sim + sim.T) / 2
        lo, hi = sym.min(), sym.max()
        out[pid] = ((sym - lo) / (hi - lo) if hi > lo else sym * 0, id_order_fwd(ep))
    return out


SCORES["bm25"] = {"dev": bm25_scores(S_DEV), "test": bm25_scores(S_TEST)}

_e5 = SentenceTransformer(EMBED_BASELINE)
if torch.cuda.is_available():
    _e5 = _e5.to("cuda")


def e5_scores(S):
    out = {}
    for pid, ep in S.items():
        emb = _e5.encode([EMBED_PREFIX + t for t in ep["node_texts"]], convert_to_numpy=True, batch_size=64)
        out[pid] = (cosine_similarity(emb), id_order_fwd(ep))
    return out


SCORES["e5"] = {"dev": e5_scores(S_DEV), "test": e5_scores(S_TEST)}


def topk_allowed(split_name):
    \"\"\"Pairs where either claim is in the other's E5 top-k -- Stage 2.1's candidate restriction.\"\"\"
    def fn(pid):
        sim = SCORES["e5"][split_name][pid][0].copy()
        np.fill_diagonal(sim, -np.inf)
        k = min(CANDIDATE_TOP_K, sim.shape[0] - 1)
        top = np.zeros_like(sim, dtype=bool)
        for i, row in enumerate(np.argsort(-sim, axis=1)[:, :k]):
            top[i, row] = True
        return top | top.T
    return fn


del _e5
import gc; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("baseline scores ready:", list(SCORES))
"""),
    md("## 7. Train the cross-encoders (one at a time; checkpoint picked on dev)"),
    code("""
import json, os, shutil, time, traceback
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
LOGS = {}


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def dev_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    logits = logits[0] if isinstance(logits, tuple) else logits
    p = softmax(logits)
    auc = roc_auc_score(labels > 0, p[:, 1] + p[:, 2])
    pos = labels > 0
    dir_acc = float(np.mean(np.where(p[pos, 1] > p[pos, 2], 1, 2) == labels[pos]))
    return {"link_auc": auc, "dir_acc": dir_acc, "combined": (auc + dir_acc) / 2}


def to_ds(examples, tok):
    ds = Dataset.from_dict({"a": [e[0] for e in examples], "b": [e[1] for e in examples],
                            "label": [e[2] for e in examples]})
    return ds.map(lambda x: tok(x["a"], x["b"], truncation=True, max_length=MAX_LEN),
                  batched=True, remove_columns=["a", "b"])


def predict(model, tok, pairs, fp16, bs=128):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(pairs), bs):
            chunk = pairs[s:s + bs]
            enc = tok([a for a, _ in chunk], [b for _, b in chunk], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(model.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and torch.cuda.is_available()):
                logits = model(**enc).logits.float()
            out.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def ce_scores(model, tok, S, fp16):
    \"\"\"link[i, j] = P(linked), averaged over both input orders; fwd[i, j] = P(i -> j).\"\"\"
    out = {}
    for pid, ep in S.items():
        n, T = ep["n"], ep["node_texts"]
        ij = [(i, j) for i in range(n) for j in range(n) if i != j]
        P = predict(model, tok, [(T[i], T[j]) for i, j in ij], fp16)
        p1, p2 = np.zeros((n, n)), np.zeros((n, n))
        for (i, j), p in zip(ij, P):
            p1[i, j], p2[i, j] = p[1], p[2]
        out[pid] = (((p1 + p2) + (p1 + p2).T) / 2, (p1 + p2.T) / 2)
    return out


rng = random.Random(SEED)
train_ex, dev_ex = make_examples(S_TRAIN, rng), make_examples(S_DEV, rng)
print(f"training examples {len(train_ex)}  dev examples {len(dev_ex)}  "
      f"(label counts train: {[sum(e[2] == k for e in train_ex) for k in range(3)]})")

for key in RUN_CE:
    if key in SCORES:
        continue
    cfg = CE_MODELS[key]
    model = trainer = None
    out_dir = f"{CKPT_DIR}/{key}"
    try:
        print(f"== {cfg['label']}")
        tok = AutoTokenizer.from_pretrained(cfg["id"])
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["id"], num_labels=3, ignore_mismatched_sizes=True, dtype=torch.float32,
            id2label={0: "no_link", 1: "a_to_b", 2: "b_to_a"}, label2id={"no_link": 0, "a_to_b": 1, "b_to_a": 2})
        args = TrainingArguments(
            output_dir=out_dir, num_train_epochs=EPOCHS, learning_rate=LR, warmup_steps=0.1, weight_decay=0.01,
            per_device_train_batch_size=BATCH, per_device_eval_batch_size=128, fp16=cfg["fp16"],
            eval_strategy="epoch", save_strategy="best", save_total_limit=1, load_best_model_at_end=True,
            metric_for_best_model="combined", greater_is_better=True, logging_steps=20,
            report_to="wandb" if USE_WANDB else "none", run_name=f"{NOTEBOOK}-{key}", seed=SEED)
        trainer = Trainer(model=model, args=args, train_dataset=to_ds(train_ex, tok), eval_dataset=to_ds(dev_ex, tok),
                          data_collator=DataCollatorWithPadding(tok), compute_metrics=dev_metrics,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)])
        t0 = time.time()
        trainer.train()
        minutes = (time.time() - t0) / 60
        evals = [h for h in trainer.state.log_history if "eval_combined" in h]
        best = max(evals, key=lambda h: h["eval_combined"])
        LOGS[key] = {"best_at_epoch": best["epoch"], "train_minutes": round(minutes, 1),
                     "history": [{k: v for k, v in h.items() if k in ("epoch", "step", "loss", "eval_loss",
                                  "eval_link_auc", "eval_dir_acc", "eval_combined")} for h in trainer.state.log_history]}
        print(f"  dev link AUC / direction acc by epoch: "
              + "  ".join(f"{h['epoch']:.0f}:{h['eval_link_auc']:.3f}/{h['eval_dir_acc']:.3f}" for h in evals))
        print(f"  best at epoch {best['epoch']:.0f} ({minutes:.1f} min)")
        SCORES[key] = {"dev": ce_scores(model, tok, S_DEV, cfg["fp16"]), "test": ce_scores(model, tok, S_TEST, cfg["fp16"])}
        if USE_WANDB and wandb.run is not None:
            pp = [pair_metrics(S_TEST[pid], *SCORES[key]["test"][pid]) for pid in S_TEST]
            wandb.run.summary.update({f"test/{m}": _mean([p[m] for p in pp if p[m] is not None])
                                      for m in ("link_auc", "link_ap", "dir_acc", "dir_acc_backward")}
                                     | {"train_minutes": minutes, "best_epoch": best["epoch"]})
    except Exception:
        print(f"  !! {cfg['label']} failed -- skipped, moving on:")
        traceback.print_exc(limit=3)
    finally:
        if USE_WANDB:
            wandb.finish()   # one W&B run per model
        shutil.rmtree(out_dir, ignore_errors=True)
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  GPU memory still allocated after cleanup: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
"""),
    md("""
## 8. Evaluate: pair-level metrics, then whole graphs

Graph methods: score-only baselines (BM25, E5) keep the top `density x n` pairs per paper, with
the density tuned on dev, and use the earlier-claim-first direction rule. Cross-encoders decide
links by probability threshold (tuned on dev) and use their own predicted direction; they are also
run restricted to E5 top-10 candidates (Stage 2.1's setup) and with the earlier-claim-first rule
instead of their own direction, which isolates what the learned direction adds.
"""),
    code("""
import pandas as pd

LABELS = {"bm25": "BM25", "e5": "E5 zero-shot", **{k: v["label"] for k, v in CE_MODELS.items()}}
PAIR_ROWS, GRAPH_ROWS, PER_PAPER = [], [], {}


def summarize(per_paper, metrics):
    out = {}
    for src in ("all", "pipeline", "single_pass"):
        ps = [p for p in per_paper if src == "all" or p["source"] == src]
        row = {"source": src, "n_papers": len(ps)}
        for m in metrics:
            vals = [p[m] for p in ps if p[m] is not None]
            row[m] = _mean(vals)
            row[m + "_ci"] = boot_ci(vals)
        out[src] = row
    return out


# direction rules on their own, for reference
for name, fwd_fn in [("earlier-claim-first rule", id_order_fwd)]:
    pp = [{"paper": pid, "source": ep["source"], **pair_metrics(ep, np.zeros((ep["n"], ep["n"])), fwd_fn(ep))}
          for pid, ep in S_TEST.items()]
    PER_PAPER[f"pair|{name}"] = pp
    for src, row in summarize(pp, ["dir_acc", "dir_acc_backward"]).items():
        row["n_backward_links"] = sum(p["n_backward"] for p in pp if src == "all" or p["source"] == src)
        PAIR_ROWS.append({"method": name, **row})

for key in SCORES:
    test = SCORES[key]["test"]
    pp = [{"paper": pid, "source": S_TEST[pid]["source"], **pair_metrics(S_TEST[pid], *test[pid])} for pid in S_TEST]
    PER_PAPER[f"pair|{key}"] = pp
    for src, row in summarize(pp, ["link_auc", "link_ap", "dir_acc", "dir_acc_backward"]).items():
        row["n_backward_links"] = sum(p["n_backward"] for p in pp if src == "all" or p["source"] == src)
        if key in ("bm25", "e5"):
            # baselines use the earlier-claim-first rule, reported above
            row["dir_acc"] = row["dir_acc_ci"] = row["dir_acc_backward"] = row["dir_acc_backward_ci"] = None
        PAIR_ROWS.append({"method": LABELS[key], **row})


def run_graph_method(name, key, mode, use_rule=False, topk=False):
    dev = SCORES[key]["dev"]
    test = SCORES[key]["test"]
    if use_rule:
        dev = {pid: (l, id_order_fwd(S_DEV[pid])) for pid, (l, f) in dev.items()}
        test = {pid: (l, id_order_fwd(S_TEST[pid])) for pid, (l, f) in test.items()}
    param, dev_f1 = tune(S_DEV, dev, mode, topk_allowed("dev") if topk else None)
    allowed = topk_allowed("test") if topk else None
    pp = []
    for pid, ep in S_TEST.items():
        edges, removed = build_graph(ep, *test[pid], mode, param, allowed(pid) if allowed else None)
        pp.append({"paper": pid, "source": ep["source"], **graph_metrics(ep, edges, removed)})
    PER_PAPER[f"graph|{name}"] = pp
    for src, row in summarize(pp, ["und_f1", "dir_f1", "und_p", "und_r", "comp_ari", "cycles_broken"]).items():
        GRAPH_ROWS.append({"method": name, "tuned": f"{mode}={param}", "dev_und_f1": round(dev_f1, 4), **row})


run_graph_method("BM25 + density, earlier-first", "bm25", "density")
run_graph_method("E5 zero-shot + density, earlier-first", "e5", "density")
for key in [k for k in RUN_CE if k in SCORES]:
    lab = LABELS[key]
    run_graph_method(f"{lab}", key, "threshold")
    run_graph_method(f"{lab}, E5 top-{CANDIDATE_TOP_K} candidates", key, "threshold", topk=True)
    run_graph_method(f"{lab} links, earlier-first direction", key, "threshold", use_rule=True)

pd.set_option("display.width", 250)
pair_df, graph_df = pd.DataFrame(PAIR_ROWS), pd.DataFrame(GRAPH_ROWS)
for src in ("all", "pipeline", "single_pass"):
    print(f"\\n=== pair level, test source = {src} ===")
    print(pair_df[pair_df["source"] == src].drop(columns="source").to_string(index=False))
    print(f"\\n=== whole graph, test source = {src} ===")
    d = graph_df[graph_df["source"] == src].drop(columns="source").sort_values("dir_f1", ascending=False)
    print(d.to_string(index=False))
"""),
    md("""
## 9. Paired significance tests (test papers resampled 1000x; `holds` = 95% interval excludes 0)
"""),
    code("""
def paired(run_a, run_b, metric):
    a = {p["paper"]: p[metric] for p in PER_PAPER[run_a] if p[metric] is not None}
    b = {p["paper"]: p[metric] for p in PER_PAPER[run_b] if p[metric] is not None}
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    diff = np.array([a[k] - b[k] for k in common])
    ci = boot_ci(diff)
    return {"diff": round(float(diff.mean()), 4), "ci": ci, "holds": ci[0] > 0 or ci[1] < 0, "n": len(common)}


SIGNIF = []
ce_keys = [k for k in RUN_CE if k in SCORES]
for key in ce_keys:
    lab = LABELS[key]
    tests = [(f"pair|{key}", "pair|earlier-claim-first rule", "dir_acc", "direction vs earlier-first rule"),
             (f"pair|{key}", "pair|e5", "link_auc", "link AUC vs E5"),
             (f"pair|{key}", "pair|bm25", "link_auc", "link AUC vs BM25"),
             (f"graph|{lab}", "graph|E5 zero-shot + density, earlier-first", "und_f1", "graph und-F1 vs E5"),
             (f"graph|{lab}", "graph|E5 zero-shot + density, earlier-first", "dir_f1", "graph dir-F1 vs E5"),
             (f"graph|{lab}", f"graph|{lab} links, earlier-first direction", "dir_f1", "own direction vs rule (dir-F1)"),
             (f"graph|{lab}", f"graph|{lab}, E5 top-{CANDIDATE_TOP_K} candidates", "und_f1", "all pairs vs top-10 candidates")]
    for a, b, m, name in tests:
        res = paired(a, b, m)
        if res:
            SIGNIF.append({"model": lab, "comparison": name, "metric": m, **res})
if SIGNIF:
    print(pd.DataFrame(SIGNIF).to_string(index=False))
"""),
    md("## 10. Scores for the human-labelling pairs, and full results (read back from the saved notebook)"),
    code("""
LABEL_PREDS = []
for lp in LABEL_PAIRS:
    ep = S_TEST.get(lp["paper_id"])
    if ep is None or lp["a_id"] not in ep["idx"] or lp["b_id"] not in ep["idx"]:
        continue
    i, j = ep["idx"][lp["a_id"]], ep["idx"][lp["b_id"]]
    row = {"pair_id": lp["pair_id"]}
    for key in SCORES:
        link, fwd = SCORES[key]["test"][lp["paper_id"]]
        row[f"{key}_link"] = round(float(link[i, j]), 4)
        row[f"{key}_p_a_to_b"] = round(float(fwd[i, j] / (fwd[i, j] + fwd[j, i])), 4) if fwd[i, j] + fwd[j, i] else 0.5
    LABEL_PREDS.append(row)
print(f"{len(LABEL_PREDS)} / {len(LABEL_PAIRS)} labelling pairs scored")

print("NOTEBOOK", NOTEBOOK)
for name, obj in [("EDGE_PAIR", PAIR_ROWS), ("EDGE_GRAPH", GRAPH_ROWS), ("EDGE_SIGNIF", SIGNIF),
                  ("EDGE_LOGS", LOGS), ("EDGE_PER_PAPER", PER_PAPER), ("LABEL_PREDS", LABEL_PREDS)]:
    print(f"{name}_JSON_BEGIN")
    print(json.dumps(obj, default=float))
    print(f"{name}_JSON_END")
"""),
]


NOTEBOOKS = [
    {
        "file": "01_group_a_base_models.ipynb", "num": "1/4", "id": "group_a",
        "title": "base-size models (fine-tuned)",
        "intro": "BM25 baseline plus seven base-size embedding models (22M-150M parameters), each scored "
                 "zero-shot and fine-tuned. The quickest notebook -- run it first.",
        "models": ["bm25", "minilm", "mpnet", "specter", "specter2", "e5", "bge_base", "gte_modernbert"],
        "splits": ["main"], "run_desc": "main 80/20 split",
        "note": "",
    },
    {
        "file": "02_group_b_large_models.ipynb", "num": "2/4", "id": "group_b",
        "title": "large models (fine-tuned)",
        "intro": "BM25 baseline plus the larger models: BGE-large and mxbai-embed-large (335M, fp16 "
                 "training) and both Qwen3-Embedding-0.6B recipes (3 and 6 epochs). The slowest notebook; "
                 "if the session gets cut off, re-run section 7 and finished models are skipped. Also "
                 "starts with a quick re-try of gte-modernbert-base at a lower learning rate, since its "
                 "fine-tuning collapsed in notebook 1.",
        "models": ["bm25", "gte_modernbert_lr2e6", "bge_large", "mxbai_large", "qwen3", "qwen3_6ep"],
        "splits": ["main"], "run_desc": "main 80/20 split",
        "note": "",
    },
    {
        "file": "03_group_c_zero_shot.ipynb", "num": "3/4", "id": "group_c",
        "title": "evaluation-only models (zero-shot) + notebook-2 retries",
        "intro": "BM25 baseline plus models that are only evaluated, not fine-tuned: Qwen3-Embedding-4B "
                 "(too big to fine-tune on a T4; loaded in fp16, ~8 GB). Also re-runs the three notebook-2 "
                 "fine-tunes that crashed on library-version issues, now fixed: mxbai-embed-large (fp16 "
                 "checkpoint -> now loaded in fp32) and both Qwen3-Embedding-0.6B recipes (gradient "
                 "checkpointing removed). If this runs in the same Colab session as notebook 2, their "
                 "zero-shot scores are already saved and only the fine-tuning runs.",
        "models": ["bm25", "mxbai_large", "qwen3", "qwen3_6ep", "qwen3_4b_zs"],
        "splits": ["main"], "run_desc": "main 80/20 split, zero-shot only",
        "note": "Optional: add `\"embeddinggemma_zs\"` (Google EmbeddingGemma-300M) to `RUN_MODELS`. It is "
                "gated on Hugging Face: accept its licence on the model page and run "
                "`from huggingface_hub import login; login()` first.\n\n",
    },
    {
        "file": "03b_qwen_retry.ipynb", "num": "3b/4", "id": "qwen_retry",
        "title": "Qwen3-Embedding-0.6B fine-tuning retry",
        "intro": "Both Qwen3-Embedding-0.6B recipes failed twice: in notebook 2 on a library incompatibility "
                 "(the trainer's gradient-checkpointing call), and in notebook 3 by running out of GPU memory "
                 "once gradient checkpointing was removed. This notebook re-runs just those two, with "
                 "gradient checkpointing switched on directly on the Hugging Face model. If it runs in the "
                 "same Colab session as notebooks 2-3, the zero-shot scores are already saved and only the "
                 "fine-tuning runs (~30-60 min each).",
        "models": ["bm25", "qwen3", "qwen3_6ep"],
        "splits": ["main"], "run_desc": "main 80/20 split",
        "note": "",
    },
    {
        "file": "04_cross_validation.ipynb", "num": "4/4", "id": "cv",
        "title": "5-fold cross-validation",
        "intro": "Optional robustness check. Runs a few representative models on all 5 stratified folds, so "
                 "every paper is tested exactly once, and reports mean / std over folds. About 5x the training "
                 "time of the same models in notebooks 1-2 -- run it last, and only if there's time. Edit "
                 "`RUN_MODELS` to pick which models; the best-performing ones from notebooks 1-3 are the most "
                 "useful to confirm here.",
        "models": ["bm25", "mpnet", "bge_base", "qwen3"],
        "splits": ["fold0", "fold1", "fold2", "fold3", "fold4"], "run_desc": "5 folds",
        "note": "", "chart": False,
    },
    {
        "file": "05_edge_classifier.ipynb", "num": "5/5", "id": "edge_classifier", "title": "", "intro": "",
        "models": [], "splits": [], "run_desc": "", "note": "", "cells": NOTEBOOK_05_CELLS,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", default="dev_data/dev_data_combined.json")
    parser.add_argument("--single-pass-dir", default="dev_data/new_papers_graphs")
    parser.add_argument("--extra-pipeline", nargs="*",
                        default=["data/graph/2505.17638/proposition_graph_final.json"])
    parser.add_argument("--out-dir", default="dev_data/retrieval_suite")
    parser.add_argument("--notebook-dir", default="docs/retrieval_suite")
    parser.add_argument("--force", action="store_true",
                        help="also overwrite notebooks that already contain run outputs")
    parser.add_argument("--only", nargs="*", help="only (re)write these notebook file names")
    args = parser.parse_args()

    papers = load_papers(args.combined, args.single_pass_dir, args.extra_pipeline)
    split = stratified_split(papers, random.Random(SEED))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "dataset.json"), "w") as f:
        json.dump(papers, f, indent=1)
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=1)

    label_pairs = sample_label_pairs(papers, split, random.Random(SEED))
    with open(os.path.join(args.out_dir, "label_pairs.json"), "w") as f:
        json.dump(label_pairs, f, indent=1)
    blob = json.dumps({"papers": papers, "split": split}, separators=(",", ":")).encode()
    # notebook 05 also scores the human-labelling pairs, so it carries them (ids only -- no reference labels)
    blob05 = json.dumps({"papers": papers, "split": split,
                         "label_pairs": [{k: p[k] for k in ("pair_id", "paper_id", "a_id", "b_id")} for p in label_pairs]},
                        separators=(",", ":")).encode()
    data_b64_05 = base64.b64encode(gzip.compress(blob05, 9)).decode()
    data_b64 = base64.b64encode(gzip.compress(blob, 9)).decode()
    os.makedirs(args.notebook_dir, exist_ok=True)
    written = 0
    for spec in NOTEBOOKS:
        if args.only and spec["file"] not in args.only:
            continue
        subs = {"__DATA_B64__": data_b64_05 if spec.get("cells") is not None else data_b64, "__NUM__": spec["num"], "__TITLE__": spec["title"],
                "__INTRO__": spec["intro"], "__GROUP_NOTE__": spec["note"], "__NB_ID__": spec["id"],
                "__RUN_MODELS__": json.dumps(spec["models"]), "__SPLITS__": json.dumps(spec["splits"]),
                "__RUN_DESC__": spec["run_desc"]}
        cells = []
        for c in (spec["cells"] if spec.get("cells") is not None else NOTEBOOK_CELLS):
            text = "".join(c["source"])
            if not spec.get("chart", True) and ("## 9. Charts" in text or "main_split.png" in text):
                continue
            for k, v in subs.items():
                text = text.replace(k, v)
            cells.append(dict(c, source=text.splitlines(keepends=True)))
        nb = {
            "cells": cells,
            "metadata": {
                "accelerator": "GPU",
                "colab": {"gpuType": "T4", "provenance": []},
                "kernelspec": {"display_name": "Python 3", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 0,
        }
        path = os.path.join(args.notebook_dir, spec["file"])
        if os.path.exists(path) and not args.force:
            with open(path) as f:
                if any(c.get("outputs") for c in json.load(f)["cells"]):
                    print(f"keeping {path}: it already has run outputs (pass --force to overwrite)")
                    continue
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        written += 1

    m = split["main"]
    print(f"{len(papers)} papers -> train {len(m['train'])}, dev {len(m['dev'])}, test {len(m['test'])}")
    print(f"wrote {args.out_dir}/dataset.json, {args.out_dir}/split.json, "
          f"{args.out_dir}/label_pairs.json, {written} notebook(s) in {args.notebook_dir}/")


if __name__ == "__main__":
    main()
