"""Merge the saved outputs of the docs/retrieval_suite/*.ipynb notebooks.

Each notebook prints its results between JSON markers in its last cell
(RESULTS / SIGNIF / PER_PAPER / TRAIN_LOGS). After a notebook has been run on
Colab and saved from VS Code, its outputs live inside the .ipynb file, so this
script reads them straight from there -- no Colab access needed. It merges all
notebooks found, then writes:

  dev_data/retrieval_suite/results_merged.json  every row, per-paper score, training log
  docs/retrieval_suite/RESULTS.md               tables + paired bootstrap tests across notebooks

Cross-notebook paired tests are possible because every notebook scores the
same 40 test papers (same embedded split), so per-paper scores line up.
"""
import argparse
import glob
import json
import os

import numpy as np

SEED = 42
N_BOOT = 1000
MAIN_METRICS = ["f_mrr", "recall@10", "auc"]
MARKERS = ["RESULTS", "SIGNIF", "PER_PAPER", "TRAIN_LOGS"]


def notebook_text(path: str) -> str:
    with open(path) as f:
        nb = json.load(f)
    chunks = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            text = out.get("text")
            if text is None:
                text = out.get("data", {}).get("text/plain", "")
            chunks.append("".join(text) if isinstance(text, list) else text)
    return "".join(chunks)


def extract(text: str, name: str):
    begin, end = f"{name}_JSON_BEGIN", f"{name}_JSON_END"
    if begin not in text or end not in text:
        return None
    return json.loads(text.split(begin, 1)[1].split(end, 1)[0].strip())


def boot_ci(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(SEED)
    means = vals[rng.integers(0, len(vals), (N_BOOT, len(vals)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def paired(per_paper, run_a, run_b, metric):
    a = {p["paper"]: p[metric] for p in per_paper.get(run_a, []) if p[metric] is not None}
    b = {p["paper"]: p[metric] for p in per_paper.get(run_b, []) if p[metric] is not None}
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    diff = np.array([a[k] - b[k] for k in common])
    ci = boot_ci(diff)
    return {"diff": round(float(diff.mean()), 4), "ci": ci, "holds": bool(ci[0] > 0 or ci[1] < 0), "n": len(common)}


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, list):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"
    return str(v)


def table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook-dir", default="docs/retrieval_suite")
    parser.add_argument("--out-json", default="dev_data/retrieval_suite/results_merged.json")
    parser.add_argument("--out-md", default="docs/retrieval_suite/RESULTS.md")
    args = parser.parse_args()

    results, per_paper, logs, found = [], {}, {}, []
    for path in sorted(glob.glob(os.path.join(args.notebook_dir, "*.ipynb"))):
        text = notebook_text(path)
        rows = extract(text, "RESULTS")
        if rows is None:
            if "EDGE_PAIR_JSON_BEGIN" in text:
                print(f"{os.path.basename(path)}: edge-classifier notebook -- its results are reported in the notebook "
                      f"itself and in dev_data/retrieval_suite/edge_results.json, not merged here")
            else:
                print(f"{os.path.basename(path)}: no results saved yet (not run, or not saved after running)")
            continue
        found.append(os.path.basename(path))
        seen = {(r["split"], r["model_key"], r["variant"], r["source"]) for r in results}
        results += [r for r in rows if (r["split"], r["model_key"], r["variant"], r["source"]) not in seen]
        per_paper.update(extract(text, "PER_PAPER") or {})
        logs.update(extract(text, "TRAIN_LOGS") or {})
        print(f"{os.path.basename(path)}: {len({(r['split'], r['model_key'], r['variant']) for r in rows})} runs")
    if not results:
        print("nothing to merge")
        return

    md = [f"# Retrieval suite results\n\nMerged from: {', '.join(found)}\n"]
    main_rows = [r for r in results if r["split"] == "main"]
    for src in ("all", "pipeline", "single_pass"):
        rows = sorted((r for r in main_rows if r["source"] == src), key=lambda r: -(r["f_mrr"] or 0))
        if not rows:
            continue
        md.append(f"## Main split, test source = {src}\n")
        md.append(table(rows, ["model", "variant", "f_mrr", "f_mrr_ci", "recall@10", "ndcg@10", "map",
                               "auc", "auc_ci", "hard_auc", "ari", "n_cluster_papers",
                               "recall@5_legacy", "mrr_legacy"]) + "\n")

    run_ids = sorted(k for k in per_paper if k.startswith("main|"))
    labels = {(r["model_key"], r["variant"]): r["model"] for r in main_rows}
    all_rows = {(r["model_key"], r["variant"]): r for r in main_rows if r["source"] == "all"}
    best = max(all_rows.items(), key=lambda kv: kv[1]["f_mrr"] or 0)[0] if all_rows else None
    sig = []
    for rid in run_ids:
        _, key, variant = rid.split("|")
        if key == "bm25":
            continue
        comps = [("vs BM25", "main|bm25|lexical")]
        if variant == "fine-tuned":
            comps.append(("vs own zero-shot", f"main|{key}|zero-shot"))
        if best and best[0] != "bm25" and (key, variant) != best:
            comps.append((f"vs best ({labels[best]}, {best[1]})", f"main|{best[0]}|{best[1]}"))
        for name, other in comps:
            for m in MAIN_METRICS:
                res = paired(per_paper, rid, other, m)
                if res:
                    sig.append({"model": labels.get((key, variant), key), "variant": variant,
                                "comparison": name, "metric": m, **res})
    if sig:
        md.append("## Paired bootstrap tests (main split, all 40 test papers)\n\n"
                  "`diff` = run minus comparison; `holds` = the 95% interval excludes 0.\n")
        md.append(table(sig, ["model", "variant", "comparison", "metric", "diff", "ci", "holds", "n"]) + "\n")

    cv = [r for r in results if r["split"].startswith("fold") and r["source"] == "all"]
    if cv:
        groups = {}
        for r in cv:
            groups.setdefault((r["model"], r["variant"]), []).append(r)
        cv_rows = []
        for (model, variant), rs in groups.items():
            row = {"model": model, "variant": variant, "folds": len(rs)}
            for m in MAIN_METRICS:
                vals = [r[m] for r in rs if r[m] is not None]
                row[f"{m} mean"] = float(np.mean(vals)) if vals else None
                row[f"{m} std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            cv_rows.append(row)
        md.append("## 5-fold cross-validation (all sources)\n")
        md.append(table(cv_rows, ["model", "variant", "folds"] + [f"{m} {s}" for m in MAIN_METRICS
                                                                   for s in ("mean", "std")]) + "\n")

    if logs:
        md.append("## Training: where the dev-selected checkpoint landed\n\n"
                  "Checkpoints are chosen by filtered MRR on the 16 dev papers, with early stopping. A best "
                  "epoch at the very start suggests the learning rate is too high or the model overfits fast; "
                  "a best epoch at the ceiling means it may still have been improving.\n")
        log_rows = [{"run": rid, "dev fMRR zero-shot": lg.get("dev_f_mrr_zero_shot"),
                     "best dev fMRR": lg.get("best_dev_f_mrr"), "best_at_epoch": lg.get("best_at_epoch"),
                     "epoch ceiling": lg.get("epochs_ceiling"), "stopped_early": lg.get("stopped_early"),
                     "steps": lg.get("total_steps"), "train_minutes": lg.get("train_minutes")}
                    for rid, lg in sorted(logs.items())]
        md.append(table(log_rows, ["run", "dev fMRR zero-shot", "best dev fMRR", "best_at_epoch", "epoch ceiling",
                                   "stopped_early", "steps", "train_minutes"]) + "\n")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"notebooks": found, "results": results, "significance": sig,
                   "per_paper": per_paper, "train_logs": logs}, f, indent=1)
    with open(args.out_md, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
