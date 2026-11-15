"""BM25 baseline for the same claim-relatedness retrieval/clustering task the
docs/colab_train_claim_embedder_*.ipynb notebooks evaluate embedding models on.

Mirrors their exact methodology (SEED=42, paper-level 80/20 split, Retrieval
Recall@5, Clustering ARI) so the numbers are directly comparable to the
zero-shot/fine-tuned rows those notebooks report -- filling the head-to-head
gap flagged in conversation: the notebooks never measured how the pipeline's
own retrieval mechanism (BM25Okapi, same as
src/claim_pipeline/resolution/retrieval.py's BM25Retriever) does on this task.

Retrieval Recall@5 uses real BM25Okapi term-overlap scores (an honest lexical
baseline). Clustering ARI has no direct BM25 analogue (BM25 scores query vs.
corpus, not a fixed-dimension vector every claim can be embedded into for
KMeans) -- TF-IDF vectors are used there instead, as the standard lexical
stand-in for a vector representation.

Also reports two companion metrics, matching the embedder notebooks: Retrieval
MRR (rank-sensitive companion to Recall@5) and Clustering NMI (companion to
ARI).
"""
import argparse
import json
import random
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SEED = 42


def paper_key(paper: dict) -> str:
    return paper.get("arxiv_id") or paper.get("source_file") or "unknown"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_pairs(papers, rng):
    pairs = []
    for paper in papers:
        pkey = paper_key(paper)
        all_nodes = {}
        for comp in paper.get("components", []):
            for n in comp.get("nodes", []):
                all_nodes[n["id"]] = n
        for comp in paper.get("components", []):
            nodes = {n["id"]: n for n in comp.get("nodes", [])}
            for e in comp.get("edges", []):
                frm, to = nodes.get(e["from"]), nodes.get(e["to"])
                if not frm or not to:
                    continue
                pairs.append({"text_a": frm["description"], "text_b": to["description"], "paper": pkey})
    return pairs


def build_eval_papers(papers, val_papers):
    out = []
    for paper in papers:
        if paper_key(paper) not in val_papers:
            continue
        node_texts, node_ids, comp_labels, edges = [], [], [], []
        for ci, comp in enumerate(paper.get("components", [])):
            for n in comp.get("nodes", []):
                node_ids.append(n["id"])
                node_texts.append(n["description"])
                comp_labels.append(ci)
            for e in comp.get("edges", []):
                edges.append((e["from"], e["to"]))
        if len(node_texts) < 2:
            continue
        out.append({"paper": paper_key(paper), "node_ids": node_ids, "node_texts": node_texts,
                     "comp_labels": comp_labels, "edges": edges})
    return out


def evaluate_bm25(eval_papers):
    recalls, mrrs, chance_rates = [], [], []
    for ep in eval_papers:
        bm25 = BM25Okapi([tokenize(t) for t in ep["node_texts"]])
        id_to_idx = {nid: i for i, nid in enumerate(ep["node_ids"])}
        n = len(ep["node_texts"])
        chance_rates.append(min(5, n - 1) / (n - 1) if n > 1 else 0)
        for frm, to in ep["edges"]:
            if frm not in id_to_idx or to not in id_to_idx:
                continue
            i, j = id_to_idx[frm], id_to_idx[to]
            scores = bm25.get_scores(tokenize(ep["node_texts"][i]))
            ranked = np.argsort(-scores)
            ranked = [r for r in ranked if r != i]
            recalls.append(1.0 if j in ranked[:5] else 0.0)
            mrrs.append(1.0 / (ranked.index(j) + 1))
    return np.mean(recalls), np.mean(mrrs), np.mean(chance_rates), len(recalls)


def evaluate_tfidf_clustering(eval_papers):
    aris, nmis = [], []
    for ep in eval_papers:
        n = len(ep["node_texts"])
        n_true_clusters = len(set(ep["comp_labels"]))
        if not (1 < n_true_clusters < n):
            continue
        vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False).fit_transform(ep["node_texts"])
        pred = KMeans(n_clusters=n_true_clusters, n_init=10, random_state=SEED).fit_predict(vec.toarray())
        aris.append(adjusted_rand_score(ep["comp_labels"], pred))
        nmis.append(normalized_mutual_info_score(ep["comp_labels"], pred))
    return np.mean(aris), np.mean(nmis), len(aris)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dev_data/dev_data_combined.json")
    args = parser.parse_args()

    with open(args.data) as f:
        papers = json.load(f)
    print(f"{len(papers)} papers loaded from {args.data}")

    rng = random.Random(SEED)
    pairs = build_pairs(papers, rng)
    papers_order = sorted({p["paper"] for p in pairs})
    rng.shuffle(papers_order)
    n_val_papers = max(1, int(0.2 * len(papers_order)))
    val_papers = set(papers_order[:n_val_papers])
    print(f"held-out (val) papers: {len(val_papers)}/{len(papers_order)}")

    eval_papers = build_eval_papers(papers, val_papers)
    print(f"{len(eval_papers)} held-out papers with usable eval data\n")

    recall, mrr, chance, n_edges = evaluate_bm25(eval_papers)
    ari, nmi, n_ari_papers = evaluate_tfidf_clustering(eval_papers)

    print("--- BM25 / TF-IDF lexical baseline ---")
    print(f"Retrieval Recall@5: {recall:.4f}  (n={n_edges}, avg chance rate ~{chance:.3f})")
    print(f"Retrieval MRR:      {mrr:.4f}  (n={n_edges}, 0=worst, 1=best -- rank-sensitive companion to Recall@5)")
    print(f"Clustering ARI:     {ari:.4f}  (n={n_ari_papers} papers, TF-IDF vectors, 0=random, 1=perfect)")
    print(f"Clustering NMI:     {nmi:.4f}  (n={n_ari_papers} papers, TF-IDF vectors, 0=no mutual info, 1=perfect -- companion to ARI)")


if __name__ == "__main__":
    main()
