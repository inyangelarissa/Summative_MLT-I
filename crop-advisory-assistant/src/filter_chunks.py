#!/usr/bin/env python
"""Drop chunks whose TEXT mentions no known crop or disease (reference lists, acknowledgements,
author affiliations...). data_prep.py skips this check when sources.csv supplies a crop for the file.
Run from repo root:  python src/filter_chunks.py data/interim/chunks.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import detect  # same detector the pipeline uses

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/interim/chunks.jsonl")
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
keep = [r for r in rows if detect(r["text"], None)[0] != "general"]
path.with_suffix(".unfiltered.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + "\n", encoding="utf-8")
print(f"{len(rows)} -> {len(keep)} chunks (removed {len(rows) - len(keep)} with no crop/disease mention in the text)")
print("chunks per crop:", dict(Counter(r["crop"] for r in keep)))
print("chunks mentioning a disease:", sum(1 for r in keep if r["diseases"]))