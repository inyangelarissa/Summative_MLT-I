#!/usr/bin/env python
"""Assemble a self-contained Hugging Face Space folder.

  python deploy/prepare_space.py --space_dir space \
      --adapter results/runs/e4_lora_rag_climate/adapter --gate results/gate/best \
      --retriever results/retriever/best_dense --corpus data/processed/corpus.jsonl \
      --llm_model Qwen/Qwen2.5-1.5B-Instruct

Then create a Space (SDK: Gradio) and upload the folder, e.g.
  huggingface-cli upload YOUR_USER/crop-advisory-assistant space . --repo-type=space
(files >10 MB such as adapter weights are handled via Git LFS/Xet by the upload tool).
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_FILES = ["bm25.py", "retriever.py", "domain_gate.py", "pipeline.py", "prompting.py", "translate.py",
             "weather.py", "build_climate.py", "knowledge.py"]
REQUIREMENTS = """gradio>=4.44
torch
transformers>=4.46
peft>=0.13
accelerate>=1.0
sentence-transformers
faiss-cpu
scikit-learn
joblib
numpy
sentencepiece
protobuf
"""
README = """---
title: Climate-Aware Crop Disease Advisory Assistant
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

Domain-specific QA assistant for diseases of maize, beans, cassava and potato (retrieval + LoRA-fine-tuned LLM,
weather-aware). University project prototype: confirm advice with an extension officer.
"""


def copy_tree(src, dst, skip=("checkpoints", "errors_test.jsonl", "metrics.json")):
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise SystemExit(f"Missing: {src}")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*skip), dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space_dir", default="space")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit if --llm_model is a merged model on the Hub")
    ap.add_argument("--gate", default="results/gate/best")
    ap.add_argument("--retriever", default="results/retriever/best_dense")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--llm_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--config", default="app/config.json", help="Start from this config (flags are kept)")
    ap.add_argument("--retriever_mode", default=None)
    ap.add_argument("--no_context", action="store_true", help="Model was trained without passages")
    ap.add_argument("--no_climate", action="store_true", help="Model was trained without weather")
    args = ap.parse_args()

    out = Path(args.space_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "src").mkdir(parents=True)
    for f in SRC_FILES:
        shutil.copy(ROOT / "src" / f, out / "src" / f)
    shutil.copy(ROOT / "app" / "app.py", out / "app.py")
    art = out / "artifacts"
    copy_tree(ROOT / args.gate, art / "gate")
    copy_tree(ROOT / args.retriever, art / "retriever")
    (art / "corpus.jsonl").write_bytes((ROOT / args.corpus).read_bytes())
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    cfg.update({"corpus_path": "artifacts/corpus.jsonl", "gate_dir": "artifacts/gate",
                "retriever_dir": "artifacts/retriever", "llm_model": args.llm_model, "adapter_dir": None})
    if args.adapter:
        copy_tree(ROOT / args.adapter, art / "adapter")
        cfg["adapter_dir"] = "artifacts/adapter"
    if args.retriever_mode:
        cfg["retriever_mode"] = args.retriever_mode
    cfg["use_context"], cfg["use_climate"] = not args.no_context, not args.no_climate
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")

    total = 0
    for p in sorted(out.rglob("*")):
        if p.is_file():
            mb = p.stat().st_size / 1e6
            total += mb
            if mb > 5:
                print(f"  {p.relative_to(out)}  {mb:.1f} MB")
    print(f"Space folder ready: {out}/  ({total:.0f} MB). Review config.json (use_context/use_climate must match training).")


if __name__ == "__main__":
    main()
