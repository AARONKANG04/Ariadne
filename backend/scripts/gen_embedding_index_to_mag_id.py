#!/usr/bin/env python3
"""
Generate backend/embedding_index_to_mag_id.json from OGB nodeidx2paperid mapping.
Run from repo root with path to the CSV (e.g. ml_pipeline data or ogbn-arxiv mapping).

  python -m scripts.gen_embedding_index_to_mag_id path/to/nodeidx2paperid.csv

Output: backend/embedding_index_to_mag_id.json (array of OpenAlex URLs in node order).
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

# Repo root = parent of backend
BACKEND_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BACKEND_DIR / "embedding_index_to_mag_id.json"


def mag_id_to_openalex_url(mag_id: str) -> str:
    """Convert numeric MAG id to OpenAlex URL."""
    s = str(mag_id).strip()
    if s.startswith("http"):
        return s
    if not s.isdigit() and s.startswith("W"):
        s = s[1:]
    return f"https://openalex.org/W{s}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embedding_index_to_mag_id.json from nodeidx2paperid CSV")
    parser.add_argument("csv_path", type=Path, help="Path to nodeidx2paperid.csv or .csv.gz")
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_PATH, help="Output JSON path")
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Error: {args.csv_path} not found", file=sys.stderr)
        sys.exit(1)

    open_fn = gzip.open if args.csv_path.suffix == ".gz" else open
    mode = "rt" if args.csv_path.suffix == ".gz" else "r"
    with open_fn(args.csv_path, mode, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print("Error: CSV is empty", file=sys.stderr)
        sys.exit(1)

    # Prefer "paper id" then "mag_id" then second column
    keys = list(rows[0].keys())
    mag_col = "paper id" if "paper id" in keys else ("mag_id" if "mag_id" in keys else keys[1])
    node_col = "node idx" if "node idx" in keys else keys[0]
    rows.sort(key=lambda r: int(r.get(node_col, 0)))
    mag_ids = [r[mag_col] for r in rows]
    index_to_mag = [mag_id_to_openalex_url(m) for m in mag_ids]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(index_to_mag, f, indent=0)
    print(f"Wrote {len(index_to_mag)} entries to {args.output}")


if __name__ == "__main__":
    main()
