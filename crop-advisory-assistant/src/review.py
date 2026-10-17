#!/usr/bin/env python
"""Manual verification of generated QA pairs (needed for an honest dataset section).

1) Export a sample for a human to check against the source passage:
     python src/review.py export --qa data/interim/qa_raw.jsonl --n 150 --out data/interim/review_sample.csv
   Open the CSV (Excel/Sheets). In `verdict` write one of:  correct | minor_issue | incorrect | unsupported
   Optionally fill `fixed_question` / `fixed_answer` to repair a pair instead of rejecting it.
   (Have a second person review ~30 rows to report inter-annotator agreement.)

2) Apply the review:
     python src/review.py apply --qa data/interim/qa_raw.jsonl --reviewed data/interim/review_sample.csv \
         --out data/interim/qa_clean.jsonl
   Writes the cleaned file (rejected pairs dropped, fixes applied; unreviewed pairs kept) and prints
   the error rate in the reviewed sample, which you report as an estimate of the whole dataset's quality.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

VALID = {"correct", "minor_issue", "incorrect", "unsupported"}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def export(args):
    rows = read_jsonl(args.qa)
    rng = random.Random(args.seed)
    by_crop = defaultdict(list)
    for r in rows:
        by_crop[r["crop"]].append(r)
    sample = []
    per = max(1, args.n // max(1, len(by_crop)))
    for crop, rs in sorted(by_crop.items()):
        rng.shuffle(rs)
        sample += rs[:per]
    rng.shuffle(sample)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "crop", "source_doc", "question", "answer", "passage", "verdict", "fixed_question",
                    "fixed_answer", "notes"])
        for r in sample:
            w.writerow([r["id"], r["crop"], r["source_doc"], r["question"], r["answer"], r["context"][0],
                        "", "", "", ""])
    print(f"exported {len(sample)} rows to {args.out}")


def apply(args):
    rows = read_jsonl(args.qa)
    with open(args.reviewed, newline="", encoding="utf-8-sig") as f:
        rev = {r["id"]: r for r in csv.DictReader(f)}
    counts, out = Counter(), []
    for r in rows:
        rv = rev.get(r["id"])
        verdict = (rv or {}).get("verdict", "").strip().lower()
        if not rv or not verdict:
            out.append(r)
            continue
        if verdict not in VALID:
            raise SystemExit(f"Row {r['id']}: verdict '{verdict}' not in {sorted(VALID)}")
        counts[verdict] += 1
        fq, fa = rv.get("fixed_question", "").strip(), rv.get("fixed_answer", "").strip()
        if verdict in ("incorrect", "unsupported") and not (fq or fa):
            continue
        r = dict(r)
        if fq:
            r["question"] = fq
        if fa:
            r["answer"] = fa
        r["origin"] = r["origin"] + ("+human_fixed" if (fq or fa) else "+human_verified")
        out.append(r)
    n = sum(counts.values())
    report = {"reviewed": n, "counts": dict(counts), "kept_total": len(out), "of_total": len(rows)}
    if n:
        report["error_rate_incorrect_or_unsupported"] = round((counts["incorrect"] + counts["unsupported"]) / n, 3)
        report["fully_correct_rate"] = round(counts["correct"] / n, 3)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(args.out).with_suffix(".review_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--qa", default="data/interim/qa_raw.jsonl")
    e.add_argument("--n", type=int, default=150)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--out", default="data/interim/review_sample.csv")
    e.set_defaults(fn=export)
    a = sub.add_parser("apply")
    a.add_argument("--qa", default="data/interim/qa_raw.jsonl")
    a.add_argument("--reviewed", default="data/interim/review_sample.csv")
    a.add_argument("--out", default="data/interim/qa_clean.jsonl")
    a.set_defaults(fn=apply)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
