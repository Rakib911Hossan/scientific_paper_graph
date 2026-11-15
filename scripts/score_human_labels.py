"""Score the filled-in human-labelling workbook against the reference graphs and the models.

Inputs:
  docs/retrieval_suite/human_labels/labelling_sheet.xlsx  (filled in by two annotators)
  dev_data/retrieval_suite/label_pairs.json               (answer key: reference link + direction)
  docs/retrieval_suite/05_edge_classifier.ipynb           (optional: saved run with LABEL_PREDS)

Reports:
  1. Inter-annotator agreement: Cohen's kappa on Linked? (Yes/No; Unsure skipped) and agreement on
     direction where both said Yes.
  2. Reference-graph quality vs. the annotators' consensus (pairs where both agree): how many reference
     links humans confirm (precision), how many human-confirmed links the reference has (recall), and
     direction agreement -- overall and per source (pipeline vs single-pass graphs).
  3. Models vs. consensus, if notebook 5 has been run and saved: link AUC and direction accuracy for
     each model, next to the earlier-claim-first rule.

Writes docs/retrieval_suite/human_labels/HUMAN_LABEL_RESULTS.md. Requires openpyxl.
"""
import argparse
import json
import os
import re

import numpy as np
from openpyxl import load_workbook

DIR_MAP = {"A→B": "A->B", "B→A": "B->A", "A->B": "A->B", "B->A": "B->A"}


def read_sheet(ws):
    out = {}
    header = None
    for row in ws.iter_rows(values_only=True):
        if row and row[0] == "Pair ID":
            header = row
            continue
        if header is None or not row or not row[0]:
            continue
        rec = dict(zip(header, row))
        linked = (rec.get("Linked?") or "").strip()
        direction = DIR_MAP.get((rec.get("Direction") or "").strip())
        out[rec["Pair ID"]] = {"linked": linked or None, "direction": direction}
    return out


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def label_preds(notebook):
    if not os.path.exists(notebook):
        return []
    with open(notebook) as f:
        nb = json.load(f)
    text = ""
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            t = out.get("text", "")
            text += "".join(t) if isinstance(t, list) else t
    if "LABEL_PREDS_JSON_BEGIN" not in text:
        return []
    return json.loads(text.split("LABEL_PREDS_JSON_BEGIN", 1)[1].split("LABEL_PREDS_JSON_END", 1)[0])


def fmt(v):
    return "–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", default="docs/retrieval_suite/human_labels/labelling_sheet.xlsx")
    parser.add_argument("--key", default="dev_data/retrieval_suite/label_pairs.json")
    parser.add_argument("--notebook", default="docs/retrieval_suite/05_edge_classifier.ipynb")
    parser.add_argument("--out", default="docs/retrieval_suite/human_labels/HUMAN_LABEL_RESULTS.md")
    args = parser.parse_args()

    with open(args.key) as f:
        key = {p["pair_id"]: p for p in json.load(f)}
    wb = load_workbook(args.sheet)
    sheets = [ws for ws in wb.worksheets if ws.title.startswith("Annotator")]
    ann = [read_sheet(ws) for ws in sheets]
    names = [(ws["B1"].value or ws.title) for ws in sheets]
    md = ["# Human labels: results\n", f"Annotators: {', '.join(map(str, names))}\n"]

    done = [sum(1 for v in a.values() if v["linked"]) for a in ann]
    md.append("Pairs labelled: " + ", ".join(f"{n}: {d}/{len(key)}" for n, d in zip(names, done)) + "\n")
    if len(ann) < 2 or min(done) == 0:
        md.append("Need two annotators' labels to compute agreement.\n")
    a1, a2 = ann[0], ann[1] if len(ann) > 1 else {}

    # 1. inter-annotator agreement
    both = [pid for pid in key if a1.get(pid, {}).get("linked") in ("Yes", "No")
            and a2.get(pid, {}).get("linked") in ("Yes", "No")]
    k = kappa([a1[p]["linked"] for p in both], [a2[p]["linked"] for p in both])
    yes_both = [p for p in both if a1[p]["linked"] == a2[p]["linked"] == "Yes"]
    dir_both = [p for p in yes_both if a1[p]["direction"] and a2[p]["direction"]]
    dir_agree = (sum(a1[p]["direction"] == a2[p]["direction"] for p in dir_both) / len(dir_both)) if dir_both else None
    md.append("## 1. Inter-annotator agreement\n")
    md.append(f"- Linked? Cohen's kappa: **{fmt(k)}** over {len(both)} pairs both labelled Yes/No "
              f"(raw agreement {fmt(sum(a1[p]['linked'] == a2[p]['linked'] for p in both) / len(both) if both else None)})")
    md.append(f"- Direction agreement where both said Yes: **{fmt(dir_agree)}** over {len(dir_both)} pairs\n")

    # consensus: both agree on Linked?; direction only where both gave the same A->B / B->A
    cons = {}
    for p in both:
        if a1[p]["linked"] != a2[p]["linked"]:
            continue
        d = a1[p]["direction"] if a1[p]["direction"] == a2[p]["direction"] else None
        cons[p] = {"linked": a1[p]["linked"] == "Yes", "direction": d}

    # 2. reference vs consensus
    md.append("## 2. Reference graphs vs. human consensus\n")
    md.append("| source | consensus pairs | reference links confirmed (precision) | human links in reference (recall) | "
              "link agreement | direction agreement |")
    md.append("|---|---|---|---|---|---|")
    for src in ("all", "pipeline", "single_pass"):
        ps = [p for p in cons if src == "all" or key[p]["source"] == src]
        ref_pos = [p for p in ps if key[p]["ref_linked"]]
        hum_pos = [p for p in ps if cons[p]["linked"]]
        prec = sum(cons[p]["linked"] for p in ref_pos) / len(ref_pos) if ref_pos else None
        rec = sum(key[p]["ref_linked"] for p in hum_pos) / len(hum_pos) if hum_pos else None
        acc = sum(cons[p]["linked"] == key[p]["ref_linked"] for p in ps) / len(ps) if ps else None
        dp = [p for p in ps if key[p]["ref_linked"] and cons[p]["linked"] and cons[p]["direction"]]
        dacc = sum(cons[p]["direction"] == key[p]["ref_direction"] for p in dp) / len(dp) if dp else None
        md.append(f"| {src} | {len(ps)} | {fmt(prec)} | {fmt(rec)} | {fmt(acc)} | {fmt(dacc)} (n={len(dp)}) |")
    md.append("")

    # 3. models vs consensus
    preds = {r["pair_id"]: r for r in label_preds(args.notebook)}
    if preds:
        md.append("## 3. Models vs. human consensus (pairs from notebook 5)\n")
        models = sorted({k_.rsplit("_link", 1)[0] for r in preds.values() for k_ in r if k_.endswith("_link")})
        md.append("| model | link AUC vs humans | direction accuracy vs humans |")
        md.append("|---|---|---|")
        ps = [p for p in cons if p in preds]
        dps = [p for p in ps if cons[p]["linked"] and cons[p]["direction"]]
        for m in models:
            a = auc([preds[p][f"{m}_link"] for p in ps], [cons[p]["linked"] for p in ps])
            if m in ("bm25", "e5"):
                dacc = None   # these use the earlier-claim-first rule, reported below
            else:
                dacc = (sum((preds[p][f"{m}_p_a_to_b"] > 0.5) == (cons[p]["direction"] == "A->B") for p in dps)
                        / len(dps)) if dps else None
            md.append(f"| {m} | {fmt(a)} | {fmt(dacc)} |")
        num = lambda nid: int(re.search(r"\d+", nid).group())
        rule = (sum((num(key[p]["a_id"]) < num(key[p]["b_id"])) == (cons[p]["direction"] == "A->B") for p in dps)
                / len(dps)) if dps else None
        md.append(f"| earlier-claim-first rule | – | {fmt(rule)} |")
        md.append(f"\n{len(ps)} consensus pairs scored, {len(dps)} with a consensus direction.\n")
    else:
        md.append("## 3. Models vs. human consensus\n\nNotebook 5 hasn't been run and saved yet -- rerun this "
                  "script after it has.\n")

    with open(args.out, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
