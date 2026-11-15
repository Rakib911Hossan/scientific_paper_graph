"""Write the human-labelling workbook for the 300 test-paper claim pairs.

Reads dev_data/retrieval_suite/label_pairs.json (written by build_retrieval_suite.py) and writes
docs/retrieval_suite/human_labels/labelling_sheet.xlsx with an instructions sheet and one identical
sheet per annotator. The sheet is blind: it shows neither the reference graph's answer nor the claim
ids (ids follow extraction order, which would hint at direction). The answer key stays in
label_pairs.json -- don't send that file to annotators.

Requires openpyxl (pip install openpyxl).
"""
import argparse
import json
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

FONT = "Arial"
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")   # pale yellow: cells annotators fill in
HEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP_TOP = Alignment(wrap_text=True, vertical="top")

LINKED = ["Yes", "No", "Unsure"]
DIRECTION = ["A→B", "B→A", "Unclear"]

INSTRUCTIONS = [
    ("Claim-pair labelling", "title"),
    ("Purpose", "h"),
    ("Our proposition graphs were built by language models, so we don't yet know how accurate their links are. "
     "This sheet asks people to judge a sample of 300 claim pairs from 10 test papers. The answers measure "
     "the quality of the reference graphs and give a human check on the trained models.", "p"),
    ("What to do", "h"),
    ("1. Two people label every pair, each on their own sheet (Annotator 1, Annotator 2). Write your name in cell B1 "
     "of your sheet. Work independently, and don't compare answers until both of you have finished.", "p"),
    ("2. Pairs are grouped by paper. Open the paper from the link in column B and keep it open while you work through its pairs.", "p"),
    ("3. For each pair, fill in the yellow cells: 'Linked?' and, if Yes, 'Direction'. Both are drop-down lists. "
     "'Notes' is optional.", "p"),
    ("4. Judge from the two claims and the paper itself, not from outside knowledge of the topic.", "p"),
    ("Linked? -- what counts as a link", "h"),
    ("Yes: one claim is a genuine prerequisite of the other. You would need to know one of them to derive or verify "
     "the other, and the paper's text supports that connection (for example, one claim describes an experimental "
     "setup that makes the other claim's result measurable, or one names the cause of what the other reports).", "p"),
    ("No: the claims are not connected in that way. Being about the same topic, appearing in the same section, or "
     "being listed side by side ('some X..., others Y...') is NOT a link.", "p"),
    ("Unsure: you can't decide even after checking the paper. Use it sparingly.", "p"),
    ("Direction -- only when Linked? = Yes", "h"),
    ("A→B: Claim A is the prerequisite. You need A to derive or verify B.", "p"),
    ("B→A: Claim B is the prerequisite.", "p"),
    ("Unclear: the claims are linked but the paper doesn't settle which comes first. A sentence of the form "
     "'X was done to achieve Y' on its own usually does not settle it.", "p"),
    ("A general observation and a specific cause attributed to it: the general observation is the prerequisite. "
     "Example: 'more than half the Dockerfiles failed to build' → 'library breaking changes caused the build "
     "failures' is A→B when A is the first claim.", "p"),
    ("Legend", "h"),
    ("Yellow cells are the only ones to edit: your name (B1) and the Linked? / Direction / Notes columns. "
     "Everything else is fixed.", "p"),
    ("Example row (format only -- not one of the 300 pairs)", "h"),
]

EXAMPLE = ["L000", "2503.13507",
           "More than half of the submitted Dockerfiles failed to build.",
           "Breaking changes in the PEFT and Transformers libraries caused the Dockerfile build failures.",
           "Yes", "A→B", "The paper attributes the build failures to these library changes."]

COLUMNS = [("Pair ID", 9), ("Paper (arXiv)", 14), ("Claim A", 60), ("Claim B", 60),
           ("Linked?", 11), ("Direction", 11), ("Notes", 36)]


def style_header(ws, row):
    for col, _ in enumerate(COLUMNS, 1):
        c = ws.cell(row=row, column=col)
        c.font = Font(name=FONT, bold=True)
        c.fill = HEADER_FILL
        c.border = BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center")


def instructions_sheet(wb):
    ws = wb.active
    ws.title = "Instructions"
    for col, (_, width) in enumerate(COLUMNS, 1):
        ws.column_dimensions[chr(64 + col)].width = width
    chars_per_line = sum(w for _, w in COLUMNS) * 1.1
    row = 1
    for i, (text, kind) in enumerate(INSTRUCTIONS):
        if kind == "h" and i > 0:
            row += 1   # blank line before each heading
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(COLUMNS))
        c = ws.cell(row=row, column=1, value=text)
        c.alignment = WRAP_TOP
        c.font = Font(name=FONT, bold=kind != "p", size={"title": 14, "h": 11}.get(kind, 10))
        lines = max(1, -(-len(text) // int(chars_per_line)))   # merged cells don't auto-fit their height
        ws.row_dimensions[row].height = 15 * lines + (6 if kind == "title" else 2)
        row += 1

    for col, (name, _) in enumerate(COLUMNS, 1):
        ws.cell(row=row, column=col, value=name)
    style_header(ws, row)
    for col, value in enumerate(EXAMPLE, 1):
        c = ws.cell(row=row + 1, column=col, value=value)
        c.font = Font(name=FONT, size=10, italic=True)
        c.alignment = WRAP_TOP
        c.border = BORDER
        if col >= 5:
            c.fill = INPUT_FILL


def annotator_sheet(wb, title, pairs):
    ws = wb.create_sheet(title)
    ws["A1"] = "Annotator name:"
    ws["A1"].font = Font(name=FONT, bold=True)
    ws.merge_cells("B1:C1")
    ws["B1"].fill = INPUT_FILL
    ws["B1"].font = Font(name=FONT)
    ws["B1"].border = BORDER
    ws["E1"] = f"{len(pairs)} pairs, {len({p['paper_id'] for p in pairs})} papers. Fill in the yellow cells; see the Instructions sheet."
    ws["E1"].font = Font(name=FONT, italic=True, size=9)

    header_row = 3
    for col, (name, width) in enumerate(COLUMNS, 1):
        ws.cell(row=header_row, column=col, value=name)
        ws.column_dimensions[chr(64 + col)].width = width
    style_header(ws, header_row)

    dv_link = DataValidation(type="list", formula1='"' + ",".join(LINKED) + '"', allow_blank=True)
    dv_dir = DataValidation(type="list", formula1='"' + ",".join(DIRECTION) + '"', allow_blank=True)
    dv_link.error, dv_dir.error = "Choose Yes, No or Unsure.", "Choose A→B, B→A or Unclear."
    ws.add_data_validation(dv_link)
    ws.add_data_validation(dv_dir)

    prev_paper = None
    for r, p in enumerate(pairs, header_row + 1):
        values = [p["pair_id"], p["paper_id"], p["claim_a"], p["claim_b"], None, None, None]
        for col, value in enumerate(values, 1):
            c = ws.cell(row=r, column=col, value=value)
            c.font = Font(name=FONT, size=10)
            c.alignment = WRAP_TOP
            c.border = BORDER
            if col >= 5:
                c.fill = INPUT_FILL
        link = ws.cell(row=r, column=2)
        link.hyperlink = f"https://arxiv.org/abs/{p['paper_id']}"
        link.font = Font(name=FONT, size=10, color="0563C1", underline="single")
        if p["paper_id"] != prev_paper and prev_paper is not None:
            for col in range(1, len(COLUMNS) + 1):
                cell = ws.cell(row=r, column=col)
                cell.border = Border(left=THIN, right=THIN, bottom=THIN, top=Side(style="medium", color="808080"))
        prev_paper = p["paper_id"]
        dv_link.add(f"E{r}")
        dv_dir.add(f"F{r}")

    ws.freeze_panes = ws.cell(row=header_row + 1, column=3)
    ws.auto_filter.ref = f"A{header_row}:G{header_row + len(pairs)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="dev_data/retrieval_suite/label_pairs.json")
    parser.add_argument("--out", default="docs/retrieval_suite/human_labels/labelling_sheet.xlsx")
    parser.add_argument("--annotators", type=int, default=2)
    args = parser.parse_args()

    with open(args.pairs) as f:
        pairs = sorted(json.load(f), key=lambda p: (p["paper_id"], p["pair_id"]))
    wb = Workbook()
    instructions_sheet(wb)
    for k in range(1, args.annotators + 1):
        annotator_sheet(wb, f"Annotator {k}", pairs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wb.save(args.out)
    print(f"wrote {args.out}: {len(pairs)} pairs x {args.annotators} annotator sheets")


if __name__ == "__main__":
    main()
