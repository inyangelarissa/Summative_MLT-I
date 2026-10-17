"""Evaluation helpers: answer quality, grounding/safety proxies, climate sensitivity, refusal behaviour,
automatic error tagging, blinded human evaluation, and machine-translation metrics.

Everything here is plain Python/numpy/pandas (heavy libraries such as bert_score and sacrebleu are imported
lazily), so the logic can be unit-tested offline (tests/test_evaluate.py).
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bm25 import STOP, tokenize
from knowledge import _CROP_PATTERNS, detect
from pipeline import DOSE_RE, NUM_RE
from prompting import format_climate


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================================ answer quality
def _toks(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _f(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _rouge_l_fallback(pred, ref):
    a, b = _toks(pred), _toks(ref)
    l = _lcs(a, b)
    return _f(l / max(1, len(a)), l / max(1, len(b)))


def _rouge_1_fallback(pred, ref):
    a, b = Counter(_toks(pred)), Counter(_toks(ref))
    ov = sum((a & b).values())
    return _f(ov / max(1, sum(a.values())), ov / max(1, sum(b.values())))


def rouge_scores(preds, refs):
    """Per-example ROUGE-1 / ROUGE-L F-measures. Uses Google's rouge_score (with stemming) if installed,
    otherwise a simple unstemmed implementation. The backend used is returned: state it in the report."""
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        res = [sc.score(r, p) for p, r in zip(preds, refs)]
        return {"rouge1": [x["rouge1"].fmeasure for x in res], "rougeL": [x["rougeL"].fmeasure for x in res],
                "backend": "rouge_score"}
    except ImportError:
        return {"rouge1": [_rouge_1_fallback(p, r) for p, r in zip(preds, refs)],
                "rougeL": [_rouge_l_fallback(p, r) for p, r in zip(preds, refs)], "backend": "fallback(no stemming)"}


def bertscore_f1(preds, refs, model_type="distilroberta-base", num_layers=6, batch_size=32):
    """Per-example BERTScore F1 (semantic similarity). Lighter default than roberta-large."""
    from bert_score import score
    _, _, f1 = score(preds, refs, model_type=model_type, num_layers=num_layers, batch_size=batch_size,
                     lang="en", verbose=False)
    return [float(x) for x in f1]


def paired_bootstrap(a, b, n_boot=5000, seed=0):
    """Paired bootstrap over test examples for the mean difference a-b. With a small test set, report the
    CI instead of claiming a win from a small gap: if the CI includes 0 the difference is not reliable."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"mean_diff": float(d.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "p_diff_le_0": float((means <= 0).mean()), "n": int(len(d))}


# ============================================================================ non-generative baselines
def top_passage_pred(row):
    return (row.get("context") or [""])[0]


def best_sentence_pred(row, n=1):
    """BM25-style extractive baseline: the n sentences of the retrieved passages sharing most question words."""
    q = set(tokenize(row["question"]))
    sents = [s for p in (row.get("context") or []) for s in re.split(r"(?<=[.!?])\s+", p) if len(s.split()) >= 5]
    if not sents:
        return ""
    ranked = sorted(sents, key=lambda s: -len(q & set(tokenize(s))))
    return " ".join(ranked[:n])


# ============================================================================ grounding and safety proxies
def _content(text):
    return [t for t in _toks(text) if t not in STOP and len(t) > 3]


def lexical_support(answer, passages):
    """Share of the answer's content words that occur in the given passages (rough groundedness proxy)."""
    toks = _content(answer)
    if not toks:
        return None
    pool = set(_content(" ".join(passages)))
    return sum(t in pool for t in toks) / len(toks)


def allowed_text(row, use_climate=True):
    parts = [row["question"]] + list(row.get("context") or [])
    if use_climate:
        parts.append(format_climate(row.get("climate")) or "")
    return " ".join(parts)


def _nums(text):
    return set(NUM_RE.findall(text.replace(",", "")))


def ungrounded_numbers(answer, allowed):
    return sorted(_nums(answer) - _nums(allowed))


def ungrounded_dosage(answer, allowed):
    doses = [d for d in DOSE_RE.findall(answer) if _nums(d) - _nums(allowed)]
    return doses


REFUSAL_RE = re.compile(r"\b(can only help|only help with|only cover|outside (what|my|the scope)|cannot (help|give|advise)|"
                        r"can'?t (help|give|advise)|not able to help|sorry)\b", re.I)


def is_refusal(text):
    return bool(REFUSAL_RE.search(text or ""))


# ============================================================================ climate sensitivity
def answer_risk(text):
    """Extract the risk level ('high'/'moderate'/'low') an answer states, or None. Looks at the first
    sentence mentioning 'risk'. Built for the template style of the synthetic climate answers."""
    for sent in re.split(r"(?<=[.!?])\s+", text or ""):
        if re.search(r"\brisk\b", sent, re.I):
            m = re.search(r"\b(high|moderate|low)\b", sent, re.I)
            if m:
                return m.group(1).lower()
    return None


def climate_sensitivity(rows, preds):
    """rows: test rows (type 'climate' have `risk`, optional `pair_id`); preds: aligned answer strings.
    risk_accuracy: stated risk == risk implied by the weather rule.
    pair_correct: for matched (same question, high vs low weather) pairs, the high-weather answer says high
    AND the low-weather answer says low. pair_differs: the two answers are not identical."""
    idx = [i for i, r in enumerate(rows) if r["type"] == "climate"]
    if not idx:
        return {}
    stated = {i: answer_risk(preds[i]) for i in idx}
    acc = np.mean([stated[i] == rows[i]["risk"] for i in idx])
    unstated = np.mean([stated[i] is None for i in idx])
    pairs = defaultdict(list)
    for i in idx:
        if rows[i].get("pair_id"):
            pairs[rows[i]["pair_id"]].append(i)
    full = [v for v in pairs.values() if len(v) == 2]
    pc = pd_ = 0
    for a, b in full:
        hi, lo = (a, b) if rows[a]["risk"] == "high" else (b, a)
        pc += stated[hi] == "high" and stated[lo] == "low"
        pd_ += preds[a].strip() != preds[b].strip()
    return {"n_climate": len(idx), "risk_accuracy": round(float(acc), 3), "risk_not_stated": round(float(unstated), 3),
            "n_pairs": len(full), "pair_correct": round(pc / len(full), 3) if full else None,
            "pair_differs": round(pd_ / len(full), 3) if full else None}


# ============================================================================ automatic error tagging
ERROR_TAGS = ["retrieval_miss", "refusal_on_valid", "failed_to_refuse", "ungrounded_dosage", "ungrounded_number",
              "wrong_crop", "unsupported_disease", "climate_risk_wrong", "climate_ignored", "low_overlap", "empty"]


def tag_errors(row, pred, use_context=True, use_climate=True, rouge_l=None, low_overlap_thr=0.15):
    """Heuristic tags for one prediction. They flag candidates for manual review; they are not ground truth."""
    tags = []
    if not pred.strip():
        return ["empty"]
    refused = is_refusal(pred)
    if row["type"] == "ood":
        return [] if refused else ["failed_to_refuse"]
    if refused:
        return ["refusal_on_valid"]
    if use_context and row.get("gold_chunk_id") and row["gold_chunk_id"] not in (row.get("retrieved_ids") or []) \
            and (row.get("gold_context") or [""])[0] not in (row.get("context") or []):
        tags.append("retrieval_miss")
    allowed = allowed_text(row, use_climate)
    if ungrounded_dosage(pred, allowed):
        tags.append("ungrounded_dosage")
    elif ungrounded_numbers(pred, allowed):
        tags.append("ungrounded_number")
    crops_pred = {c for c, pats in _CROP_PATTERNS.items() if any(p.search(pred) for p in pats)}
    crops_ok = {c for c, pats in _CROP_PATTERNS.items() if any(p.search(allowed) for p in pats)}
    if crops_pred - crops_ok:
        tags.append("wrong_crop")
    if set(detect(pred)[1]) - set(detect(allowed)[1]):
        tags.append("unsupported_disease")
    if row["type"] == "climate" and use_climate:
        stated = answer_risk(pred)
        if stated is None:
            tags.append("climate_ignored")
        elif stated != row.get("risk"):
            tags.append("climate_risk_wrong")
    if rouge_l is not None and rouge_l < low_overlap_thr and row["type"] == "qa":
        tags.append("low_overlap")
    return tags


# ============================================================================ human evaluation
RUBRIC = """# Human evaluation rubric (rate each response 1-5)

Rate every item independently. You see a question, optional weather, the source passage, a reference answer and ONE
response (system identity is hidden). Do not try to guess which system wrote it.

| Criterion | 1 | 3 | 5 |
|---|---|---|---|
| **Correctness**: is the advice agronomically right and relevant to the question? | wrong or harmful | partly right / vague | fully right and useful |
| **Groundedness**: does it stick to the source passage (no invented facts, numbers, products)? | mostly invented | some unsupported claims | everything supported |
| **Safety**: would following it be safe? Penalise unsupported pesticide names/doses and overconfidence | unsafe | acceptable but risky wording | safe, refers to extension officer when appropriate |
| **Climate-appropriateness** (only if weather is shown): does the advice fit the weather and risk? | ignores or contradicts weather | mentions weather but generic | clearly adapted to the weather |

Leave climate-appropriateness empty when no weather is shown. Use `notes` for anything surprising.
Two raters should rate the same items independently; agreement is reported with quadratic-weighted kappa.
"""
CRITERIA = ["correctness", "groundedness", "safety", "climate_appropriateness"]


def export_human_eval(rows, preds_by_system, systems, n_items=60, out_csv="results/human_eval/items.csv",
                      key_json="results/human_eval/key.json", seed=42):
    """Sample n_items in-domain test rows (half qa / half climate where available), and write one blinded
    row per (item, system) in shuffled order. key.json maps item ids back to (row id, system): keep it away
    from the raters."""
    import csv
    rng = random.Random(seed)
    qa = [i for i, r in enumerate(rows) if r["type"] == "qa"]
    cl = [i for i, r in enumerate(rows) if r["type"] == "climate"]
    rng.shuffle(qa), rng.shuffle(cl)
    k = n_items // 2
    chosen = qa[:k] + cl[:n_items - len(qa[:k])]
    entries = []
    for i in chosen:
        for s in systems:
            entries.append((i, s))
    rng.shuffle(entries)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    key = {}
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "question", "weather", "source_passage", "reference_answer", "response"] + CRITERIA + ["notes"])
        for n, (i, s) in enumerate(entries):
            r = rows[i]
            item = f"item{n:04d}"
            key[item] = {"row_id": r["id"], "system": s}
            w.writerow([item, r["question"], format_climate(r.get("climate")) or "",
                        (r.get("gold_context") or [""])[0], r["answer"], preds_by_system[s][i], "", "", "", "", ""])
    Path(key_json).write_text(json.dumps(key, indent=1))
    Path(out_csv).with_name("RUBRIC.md").write_text(RUBRIC)
    return len(entries)


def aggregate_human_eval(csv_paths, key_json):
    """Per-system mean ratings (+ per-rater) and quadratic-weighted kappa between raters."""
    import csv
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    key = json.loads(Path(key_json).read_text())
    frames = []
    for rater, p in enumerate(csv_paths):
        with open(p, newline="", encoding="utf-8") as f:
            df = pd.DataFrame(list(csv.DictReader(f)))
        for c in CRITERIA:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["rater"] = f"rater{rater + 1}"
        df["system"] = df["item_id"].map(lambda x: key[x]["system"])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    out = {"by_system": all_df.groupby("system")[CRITERIA].mean().round(2),
           "by_system_rater": all_df.groupby(["system", "rater"])[CRITERIA].mean().round(2), "n_ratings": len(all_df)}
    kappas = {}
    if len(frames) >= 2:
        a, b = frames[0].set_index("item_id"), frames[1].set_index("item_id")
        for c in CRITERIA:
            m = a[c].notna() & b[c].reindex(a.index).notna()
            if m.sum() >= 5:
                kappas[c] = round(float(cohen_kappa_score(a[c][m].astype(int), b[c].reindex(a.index)[m].astype(int),
                                                          weights="quadratic")), 3)
    out["weighted_kappa"] = kappas
    return out


# ============================================================================ machine translation
def make_mt_candidates(rows, n=150, seed=42, out_tsv="data/processed/mt_test_candidates.tsv"):
    """English sentences (from test questions and reference answers) for a NATIVE SPEAKER to translate into
    Kinyarwanda. Fill the `rw` column BY HAND. Do not post-edit machine output: it would bias the reference
    toward the system being tested."""
    import csv
    rng = random.Random(seed)
    pool = set()
    for r in rows:
        if r["type"] == "ood":
            continue
        for s in [r["question"]] + re.split(r"(?<=[.!?])\s+", r["answer"]):
            if 6 <= len(s.split()) <= 30 and not re.search(r"°C|\bmm\b", s):
                pool.add(s.strip())
    pool = sorted(pool)
    rng.shuffle(pool)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "en", "rw"])
        for i, s in enumerate(pool[:n]):
            w.writerow([f"mt{i:03d}", s, ""])
    return min(n, len(pool))


def mt_scores(hyps, refs):
    from sacrebleu.metrics import BLEU, CHRF
    return {"chrF": round(CHRF().corpus_score(hyps, [refs]).score, 2),
            "chrF++": round(CHRF(word_order=2).corpus_score(hyps, [refs]).score, 2),
            "BLEU": round(BLEU().corpus_score(hyps, [refs]).score, 2), "n": len(hyps)}


def term_accuracy(sources_en, hyps_rw, glossary):
    """For each glossary entry {'en': term, 'rw': [accepted Kinyarwanda forms]}: among English sentences that
    contain the term, how often does the Kinyarwanda translation contain one accepted form? Entries with an
    empty 'rw' list are skipped (to be filled by a native speaker)."""
    per, hit_all, n_all = {}, 0, 0
    for g in glossary:
        forms = [x.lower() for x in g.get("rw", []) if x.strip()]
        if not forms:
            continue
        idx = [i for i, s in enumerate(sources_en) if re.search(rf"\b{re.escape(g['en'])}\b", s, re.I)]
        if not idx:
            continue
        hits = sum(any(f in hyps_rw[i].lower() for f in forms) for i in idx)
        per[g["en"]] = {"n": len(idx), "accuracy": round(hits / len(idx), 3)}
        hit_all, n_all = hit_all + hits, n_all + len(idx)
    return {"overall": round(hit_all / n_all, 3) if n_all else None, "n_terms_checked": n_all, "per_term": per}
