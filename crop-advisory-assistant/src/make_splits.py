#!/usr/bin/env python
"""Stage 5: assemble everything, split BY SOURCE DOCUMENT, attach retrieval context.

  python src/make_splits.py --chunks data/interim/chunks.jsonl --qa data/interim/qa_clean.jsonl \
      --climate data/interim/climate.jsonl --ood data/interim/ood.jsonl --out_dir data/processed

Outputs (data/processed/):
  train/val/test.jsonl      LLM fine-tuning format (see src/prompting.py)
  gate_train/val/test.jsonl domain-gate data: {"text", "label": 1 in-domain / 0 out-of-domain}
  corpus.jsonl              all passages = the retrieval knowledge base
  split_docs.json           which document went to which split
  stats.json                counts, lengths, and BM25 retrieval Recall@k / MRR (baseline B1)

Context policy
  train: gold passage + 2 hard-negative passages (BM25), shuffled -> teaches the model to pick the
         relevant passage. (Limitation: train context always contains the gold passage.)
  val/test: top-3 BM25 passages over the whole corpus -> realistic. `gold_context` keeps the gold
         passage for oracle experiments, and `gold_rank` records where retrieval put it.
Splitting by document stops a passage's questions from landing in both train and test.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm25 import BM25  # noqa: E402


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if path and Path(path).exists() else []


def assign_docs(chunks, val_frac=0.1, test_frac=0.1, seed=42):
    rng = random.Random(seed)
    doc_crop = defaultdict(Counter)
    for c in chunks:
        doc_crop[c["source_doc"]][c["crop"]] += 1
    by_crop = defaultdict(list)
    for d, cnt in doc_crop.items():
        by_crop[cnt.most_common(1)[0][0]].append(d)
    split_of = {}
    for crop, docs in sorted(by_crop.items()):
        docs = sorted(docs)
        rng.shuffle(docs)
        n = len(docs)
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        n_test = max(1, round(n * test_frac)) if n >= 3 else 0
        if n < 3:
            print(f"WARNING: only {n} document(s) for crop '{crop}': all go to train. Add more documents "
                  f"so every crop appears in val/test.")
        for i, d in enumerate(docs):
            split_of[d] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return split_of


def rank_of(gold_id, retrieved_ids):
    return retrieved_ids.index(gold_id) + 1 if gold_id in retrieved_ids else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/interim/chunks.jsonl")
    ap.add_argument("--qa", default="data/interim/qa_clean.jsonl")
    ap.add_argument("--climate", default="data/interim/climate.jsonl")
    ap.add_argument("--ood", default="data/interim/ood.jsonl")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--n_distractors", type=int, default=2)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    chunks = read_jsonl(args.chunks)
    qa, climate, ood = read_jsonl(args.qa), read_jsonl(args.climate), read_jsonl(args.ood)
    if not qa:
        raise SystemExit(f"No QA rows in {args.qa}. Run build_qa.py (and review.py apply) first.")
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    corpus_ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25([c["text"] for c in chunks])

    split_of = assign_docs(chunks, seed=args.seed)
    splits = {"train": [], "val": [], "test": []}

    # ----- in-domain rows: split by source document -----
    for row in qa + climate:
        sp = split_of.get(row["source_doc"])
        if sp is None or row["gold_chunk_id"] not in chunk_by_id:
            continue
        row = dict(row)
        gold = chunk_by_id[row["gold_chunk_id"]]
        row["gold_context"] = [gold["text"]]
        hits = [corpus_ids[i] for i in bm25.top_k(row["question"], k=max(10, args.top_k + args.n_distractors + 2))]
        if sp == "train":
            neg = [h for h in hits if h != gold["chunk_id"]][: args.n_distractors]
            ctx_ids = [gold["chunk_id"]] + neg
            rng.shuffle(ctx_ids)
        else:
            ctx_ids = hits[: args.top_k]
        row["retrieved_ids"] = ctx_ids
        row["gold_rank"] = rank_of(gold["chunk_id"], ctx_ids) if sp != "train" else None
        row["gold_rank_full"] = rank_of(gold["chunk_id"], hits)
        row["context"] = [chunk_by_id[i]["text"] for i in ctx_ids]
        row["split"] = sp
        splits[sp].append(row)

    # ----- OOD rows: random 80/10/10 within each category -----
    by_cat = defaultdict(list)
    for r in ood:
        by_cat[r["category"]].append(r)
    for cat, rows in sorted(by_cat.items()):
        rng.shuffle(rows)
        n = len(rows)
        n_test, n_val = max(1, round(n * 0.1)), max(1, round(n * 0.1))
        for i, r in enumerate(rows):
            sp = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            r = dict(r, split=sp, retrieved_ids=[], gold_context=[], gold_rank=None, gold_rank_full=None)
            splits[sp].append(r)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sp, rows in splits.items():
        rng.shuffle(rows)
        with open(out / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gate, seen = [], set()
        for r in rows:
            key = r["question"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            gate.append({"text": r["question"], "label": 0 if r["type"] == "ood" else 1, "split": sp,
                         "category": r.get("category", r["crop"])})
        with open(out / f"gate_{sp}.jsonl", "w", encoding="utf-8") as f:
            for g in gate:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({k: c.get(k, "") for k in ("chunk_id", "source_doc", "crop", "diseases", "text",
                                                           "title", "url", "license")}, ensure_ascii=False) + "\n")
    (out / "split_docs.json").write_text(json.dumps(split_of, indent=2))

    # ----- leakage checks -----
    docs = {sp: {r["source_doc"] for r in rows if r["type"] != "ood"} for sp, rows in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (docs[a] & docs[b]), f"LEAKAGE: documents shared between {a} and {b}: {docs[a] & docs[b]}"
    qs = {sp: {r["question"].strip().lower() for r in rows if r["type"] == "qa"} for sp, rows in splits.items()}
    dup = qs["train"] & (qs["val"] | qs["test"])
    if dup:
        print(f"WARNING: {len(dup)} identical QA questions appear in train and val/test (e.g. {sorted(dup)[:3]})")

    # ----- stats, incl. BM25 retrieval baseline (B1) -----
    stats = {"n_chunks": len(chunks), "docs_per_split": {sp: len(d) for sp, d in docs.items()}, "splits": {}}
    for sp, rows in splits.items():
        ind = [r for r in rows if r["type"] != "ood"]
        s = {"n": len(rows), "by_type": dict(Counter(r["type"] for r in rows)),
             "by_crop": dict(Counter(r["crop"] for r in rows)),
             "avg_answer_words": round(sum(len(r["answer"].split()) for r in rows) / max(1, len(rows)), 1)}
        if sp != "train" and ind:
            ranks = [r["gold_rank_full"] for r in ind]
            for k in (1, 3, 5):
                s[f"bm25_recall@{k}"] = round(sum(1 for x in ranks if x and x <= k) / len(ranks), 3)
            s["bm25_mrr@10"] = round(sum(1 / x for x in ranks if x) / len(ranks), 3)
        stats["splits"][sp] = s
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote splits to {out}/")


if __name__ == "__main__":
    main()
