"""Batched answer generation for every system in the evaluation (base / LoRA, with or without passages/weather).

Library use (see notebooks/05_evaluation_and_error_analysis.ipynb):
    rows = prepare_rows(test_rows, context_mode="gold", climate_mode="shuffled")
    model, tok = load_model(BASE, adapter_dir)
    preds = generate_rows(model, tok, rows, use_context=True, use_climate=True)

context_mode : retrieved (default: top-3 BM25/dense passages stored in the row) | gold (oracle passage) | none
climate_mode : as_is | none | shuffled (a different row's weather; the expected risk is recomputed from the rule)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knowledge import assess  # noqa: E402
from prompting import build_messages  # noqa: E402


def prepare_rows(rows, context_mode="retrieved", climate_mode="as_is", seed=42):
    rng = random.Random(seed)
    pool = [r["climate"] for r in rows if r.get("climate")]
    out = []
    for r in rows:
        r = dict(r)
        if context_mode == "gold":
            r["context"] = list(r.get("gold_context") or r.get("context") or [])
        elif context_mode == "none":
            r["context"] = []
        if climate_mode == "none":
            r["climate"] = None
        elif climate_mode == "shuffled" and r.get("climate"):
            others = [c for c in pool if c != r["climate"]]
            r["climate"] = rng.choice(others)
            if r.get("diseases"):
                a = assess(r["crop"], r["diseases"][0], r["climate"])
                if a:
                    r["risk"] = a["risk"]
            r["climate_swapped"] = True
        out.append(r)
    return out


def load_model(base_model, adapter_dir=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter_dir or base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map={"": 0})
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    return model.eval(), tok


def generate_rows(model, tok, rows, use_context, use_climate, batch_size=8, max_new_tokens=256,
                  repetition_penalty=1.05):
    import torch
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["question"]))   # similar lengths -> less padding
    out = [None] * len(rows)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        prompts = [tok.apply_chat_template(build_messages(rows[i], use_context=use_context, use_climate=use_climate),
                                           tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=repetition_penalty, pad_token_id=tok.pad_token_id)
        for i, text in zip(idx, tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[i] = text.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", default="data/processed/test.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--use_context", action="store_true")
    ap.add_argument("--use_climate", action="store_true")
    ap.add_argument("--context_mode", default="retrieved", choices=["retrieved", "gold", "none"])
    ap.add_argument("--climate_mode", default="as_is", choices=["as_is", "none", "shuffled"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.test_file, encoding="utf-8")]
    rows = prepare_rows(rows[: args.limit] if args.limit else rows, args.context_mode, args.climate_mode)
    model, tok = load_model(args.base_model, args.adapter)
    preds = generate_rows(model, tok, rows, args.use_context, args.use_climate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"id": r["id"], "prediction": p}, ensure_ascii=False) + "\n")
    print(f"wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
