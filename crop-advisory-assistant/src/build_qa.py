#!/usr/bin/env python
"""Stage 2: LLM-assisted question-answer generation from passages, with automatic quality filters.

  python src/build_qa.py --chunks data/interim/chunks.jsonl --out data/interim/qa_raw.jsonl \
      --model Qwen/Qwen2.5-3B-Instruct --pairs_per_chunk 3

Every pair is generated from ONE passage and the answer must be supported by that passage.
Rule-based filters then remove ungrounded answers (e.g. invented numbers/dosages). Automatic
filters are not enough: use src/review.py to manually verify a sample and report the error rate.
The script is resumable (Colab disconnects): re-run the same command to continue.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm25 import STOP  # noqa: E402

GEN_SYSTEM = "You write accurate question-answer pairs for an agricultural advisory dataset. Output JSON only."

GEN_USER = """Below is a passage from an agricultural extension document about crop diseases.

PASSAGE:
\"\"\"{passage}\"\"\"

Write {n} different question-answer pairs that a smallholder farmer might ask and that this passage answers.
Rules:
- Each question must be self-contained: name the crop and/or disease. Never write "the passage", "the text" or "the document".
- Vary the question type: symptoms, cause or favourable conditions, how it spreads, prevention, management, which crops or plant parts are affected, when to act.
- Each answer must be 1-3 sentences and use ONLY facts stated in the passage. Do not add numbers, product names or dosages that are not in the passage.
- If the passage has no useful crop-disease information, return [].
Return ONLY a JSON list: [{{"question": "...", "answer": "...", "type": "symptoms|cause|spread|prevention|management|other"}}]"""

BANNED_Q = re.compile(r"\b(passage|the text|the document|the article|the author|provided|above|below)\b", re.I)
BANNED_A = re.compile(r"\b(the passage|the text|the document|the article|according to the)\b", re.I)


# ---------- parsing and filtering (pure functions, easy to test) ----------
def parse_pairs(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    m = re.search(r"\[.*\]", raw, flags=re.S)
    candidates = [m.group(0)] if m else []
    for cand in candidates:
        try:
            data = json.loads(cand)
            return [d for d in data if isinstance(d, dict) and isinstance(d.get("question"), str)
                    and isinstance(d.get("answer"), str)]
        except json.JSONDecodeError:
            pass
    # fallback: individual objects
    out = []
    for obj in re.findall(r"\{[^{}]*\}", raw, flags=re.S):
        try:
            d = json.loads(obj)
            if isinstance(d.get("question"), str) and isinstance(d.get("answer"), str):
                out.append(d)
        except json.JSONDecodeError:
            continue
    return out


def _content_tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 3]


def _numbers(text):
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def norm_question(q):
    return re.sub(r"\W+", " ", q.lower()).strip()


def check_pair(q, a, passage, min_overlap=0.5):
    """Return (ok, reason)."""
    q, a = q.strip(), a.strip()
    nq, na = len(q.split()), len(a.split())
    if not (5 <= nq <= 40):
        return False, "question_length"
    if not (8 <= na <= 120):
        return False, "answer_length"
    if BANNED_Q.search(q) or BANNED_A.search(a):
        return False, "mentions_passage"
    if not _numbers(a) <= _numbers(passage) | _numbers(q):
        return False, "ungrounded_number"
    toks = _content_tokens(a)
    if toks:
        overlap = sum(t in set(_content_tokens(passage)) for t in toks) / len(toks)
        if overlap < min_overlap:
            return False, "low_overlap"
    return True, "ok"


def run_generation(chunks, generate_fn, out_path, pairs_per_chunk=3, batch_size=8, min_overlap=0.5,
                   min_chunk_words=60, only_disease_chunks=True):
    """Generate, filter, and append to out_path. generate_fn(list[messages]) -> list[str]."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prog_path = out_path.with_suffix(".progress")
    done = set(prog_path.read_text().split()) if prog_path.exists() else set()
    seen_q = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            seen_q.add(norm_question(json.loads(line)["question"]))

    todo = [c for c in chunks if c["chunk_id"] not in done and c["n_words"] >= min_chunk_words
            and (c["diseases"] or not only_disease_chunks)]
    print(f"{len(todo)} chunks to process ({len(done)} already done)")
    stats = {"generated": 0, "kept": 0}
    reasons = {}
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        msgs = [[{"role": "system", "content": GEN_SYSTEM},
                 {"role": "user", "content": GEN_USER.format(passage=c["text"], n=pairs_per_chunk)}] for c in batch]
        outputs = generate_fn(msgs)
        with open(out_path, "a", encoding="utf-8") as f:
            for c, raw in zip(batch, outputs):
                for k, p in enumerate(parse_pairs(raw)[:pairs_per_chunk + 1]):
                    stats["generated"] += 1
                    q, a = p["question"].strip(), p["answer"].strip()
                    if not q.endswith("?"):
                        q += "?"
                    ok, why = check_pair(q, a, c["text"], min_overlap)
                    nq = norm_question(q)
                    if ok and nq in seen_q:
                        ok, why = False, "duplicate_question"
                    reasons[why] = reasons.get(why, 0) + 1
                    if not ok:
                        continue
                    seen_q.add(nq)
                    stats["kept"] += 1
                    row = {"id": f"qa-{c['chunk_id']}-{k}", "type": "qa", "question": q, "answer": a,
                           "context": [c["text"]], "gold_chunk_id": c["chunk_id"], "source_doc": c["source_doc"],
                           "crop": c["crop"], "diseases": c["diseases"], "climate": None,
                           "qtype": p.get("type", "other"), "origin": "llm_generated"}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(prog_path, "a") as pf:
            pf.write("\n".join(c["chunk_id"] for c in batch) + "\n")
        print(f"  batch {i // batch_size + 1}/{-(-len(todo) // batch_size)} | generated {stats['generated']} "
              f"kept {stats['kept']}", flush=True)
    print("filter outcomes:", reasons)
    return stats


# ---------- HF generation backend ----------
def make_hf_generate_fn(model_name, load_in_4bit=False, max_new_tokens=600, temperature=0.4):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"torch_dtype": torch.float16, "device_map": {"": 0}}
    if load_in_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).eval()

    @torch.no_grad()
    def generate(batch_messages):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                             top_p=0.9, pad_token_id=tok.pad_token_id)
        return tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/interim/chunks.jsonl")
    ap.add_argument("--out", default="data/interim/qa_raw.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--pairs_per_chunk", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--min_overlap", type=float, default=0.5)
    ap.add_argument("--include_non_disease_chunks", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N chunks (for testing)")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]
    if args.limit:
        chunks = chunks[: args.limit]
    gen = make_hf_generate_fn(args.model, args.load_in_4bit)
    stats = run_generation(chunks, gen, args.out, args.pairs_per_chunk, args.batch_size, args.min_overlap,
                           only_disease_chunks=not args.include_non_disease_chunks)
    print(stats, "->", args.out)


if __name__ == "__main__":
    main()
