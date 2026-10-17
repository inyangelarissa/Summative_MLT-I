import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def jl(path):
    p = Path(path)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else None


def js(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def todo(what):
    return f"**TODO: {what}**"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="data/DATA_CARD.template.md")
    ap.add_argument("--out", default="data/DATA_CARD.md")
    ap.add_argument("--filter_outcomes", default="")
    ap.add_argument("--mt_test_n", type=int, default=0)
    a = ap.parse_args()

    chunks = jl("data/interim/chunks.jsonl")
    qa_raw = jl("data/interim/qa_raw.jsonl")
    qa_reviewed = jl("data/interim/qa_reviewed.jsonl")
    climate, ood = jl("data/interim/climate.jsonl"), jl("data/interim/ood.jsonl")
    review, stats = js("data/interim/qa_reviewed.review_report.json"), js("data/processed/stats.json")

    src = Path("data/raw/sources.csv")
    if src.exists():
        rows = list(csv.DictReader(open(src, encoding="utf-8")))
        cnt = Counter(c["source_doc"] for c in chunks) if chunks else {}
        t = ["| File | Title | Publisher | URL | Licence | Crop | Year | Passages |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            stem = Path(r["filename"]).stem
            t.append(f"| {r['filename']} | {r['title']} | {r['publisher']} | {r['url']} | {r['license']} | {r['crop']} | {r['year']} | {cnt.get(stem, 0)} |")
        sources_table, n_docs = "\n".join(t), len(rows)
    else:
        sources_table, n_docs = todo("create data/raw/sources.csv"), "?"

    if review and review.get("reviewed"):
        c = review["counts"]
        review_line = (f"Manual review of {review['reviewed']} pairs: {review.get('fully_correct_rate', 0):.0%} correct, "
                       f"{review.get('error_rate_incorrect_or_unsupported', 0):.0%} incorrect/unsupported "
                       f"(correct {c.get('correct', 0)}, minor {c.get('minor_issue', 0)}, incorrect {c.get('incorrect', 0)}, unsupported {c.get('unsupported', 0)})")
        err_line = (f"in a manual review of {review['reviewed']} pairs, {review.get('error_rate_incorrect_or_unsupported', 0):.0%} "
                    f"were labelled incorrect or unsupported; they were corrected where a fix was supplied or removed where no correction was supplied.")
    else:
        review_line, err_line = todo("run review.py export/apply and fill verdicts"), todo("state the review error rate")

    if stats:
        hdr = ["| Split | Docs | Rows | By type | By crop | Avg answer words | BM25 R@3 | BM25 MRR@10 |", "|---|---|---|---|---|---|---|---|"]
        for sp, s in stats["splits"].items():
            hdr.append(f"| {sp} | {stats['docs_per_split'].get(sp, '')} | {s['n']} | {s['by_type']} | {s['by_crop']} | "
                       f"{s['avg_answer_words']} | {s.get('bm25_recall@3', '-')} | {s.get('bm25_mrr@10', '-')} |")
        split_table, stats_block = "\n".join(hdr), "```json\n" + json.dumps(stats, indent=2) + "\n```"
    else:
        split_table = stats_block = todo("run make_splits.py to create data/processed/stats.json")

    qa_types = Counter(r.get("qtype", "?") for r in (qa_reviewed or []))
    if qa_reviewed:
        stats_block += f"\n\nQA question types (after review): {dict(qa_types)}"

    sub = {
        "{{SOURCES_TABLE}}": sources_table, "{{N_DOCS}}": str(n_docs),
        "{{N_CHUNKS}}": str(len(chunks)) if chunks else todo("run data_prep.py"),
        "{{N_QA_RAW}}": str(len(qa_raw)) if qa_raw else todo("run build_qa.py"),
        "{{N_QA_CLEAN}}": str(len(qa_reviewed)) if qa_reviewed else todo("run review.py apply"),
        "{{N_CLIMATE}}": str(len(climate)) if climate else todo("run build_climate.py"),
        "{{N_OOD}}": str(len(ood)) if ood else todo("run build_ood.py"),
        "{{REVIEW_LINE}}": review_line, "{{ERR_LINE}}": err_line,
        "{{FILTER_LINE}}": a.filter_outcomes or todo("paste the filter outcomes line via --filter_outcomes"),
        "{{SPLIT_TABLE}}": split_table, "{{STATS_BLOCK}}": stats_block,
        "{{MT_TEST_LINE}}": (f"The native-speaker test set has {a.mt_test_n} items." if a.mt_test_n
                             else todo("build the native-speaker test set and pass --mt_test_n")),
    }
    text = Path(a.template).read_text(encoding="utf-8")
    for k, v in sub.items():
        text = text.replace(k, v)
    Path(a.out).write_text(text, encoding="utf-8")
    left = text.count("**TODO") + text.count("[MANUAL")
    print(f"wrote {a.out}; {left} items still need you (search for 'TODO' and 'MANUAL')")


if __name__ == "__main__":
    main()
