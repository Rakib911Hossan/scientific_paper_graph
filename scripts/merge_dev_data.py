"""Merge dev_data/new_added_graphs/*.json into dev_data/dev_data_combined.json.

Skips any paper whose arxiv_id/source_file key is already present in the
combined file, so it's safe to re-run after adding more graphs later.
"""
import argparse
import glob
import json
import os


def paper_key(paper: dict) -> str:
    return paper.get("arxiv_id") or paper.get("source_file") or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", default="dev_data/dev_data_combined.json")
    parser.add_argument("--new-graphs-dir", default="dev_data/new_added_graphs")
    args = parser.parse_args()

    with open(args.combined) as f:
        combined = json.load(f)
    existing_keys = {paper_key(p) for p in combined}

    added, skipped = [], []
    for path in sorted(glob.glob(os.path.join(args.new_graphs_dir, "*.json"))):
        with open(path) as f:
            paper = json.load(f)
        key = paper_key(paper)
        if key in existing_keys:
            skipped.append((os.path.basename(path), key))
            continue
        combined.append(paper)
        existing_keys.add(key)
        added.append((os.path.basename(path), key))

    with open(args.combined, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Added {len(added)} papers, skipped {len(skipped)} already present.")
    for fname, key in added:
        print(f"  + {fname} ({key})")
    for fname, key in skipped:
        print(f"  - {fname} ({key}) already in combined, skipped")
    print(f"Total papers in {args.combined}: {len(combined)}")


if __name__ == "__main__":
    main()
