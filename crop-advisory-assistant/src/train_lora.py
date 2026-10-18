#!/usr/bin/env python
"""LoRA fine-tuning of a small instruct LLM for the crop-disease advisory assistant.

Example (Colab T4):
  python src/train_lora.py \
      --train_file data/processed/train.jsonl --val_file data/processed/val.jsonl \
      --output_dir results/runs/e3_lora_rag --use_context --r 16 --lr 2e-4

Experiment switches:
  (neither flag)              -> E2: question only
  --use_context               -> E3: question + retrieved passages
  --use_context --use_climate -> E4: question + passages + weather context
  --r / --lora_alpha / --lr   -> E5: hyper-parameter study

Only the ANSWER tokens contribute to the loss (prompt tokens are masked with -100),
so the model learns to produce answers rather than to reproduce prompts.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompting import build_messages  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / model
    p.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--use_context", action="store_true", help="Include retrieved passages in the prompt")
    p.add_argument("--use_climate", action="store_true", help="Include weather context in the prompt")
    p.add_argument("--max_len", type=int, default=1024)
    # LoRA
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA: 4-bit NF4 base model (saves VRAM)")
    # optimisation
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--patience", type=int, default=2, help="Early-stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge", action="store_true", help="After training, merge LoRA into base and save")
    return p.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_examples(examples, tok, use_context, use_climate, max_len):
    """Tokenize prompt+answer, masking prompt tokens in the labels."""
    rows, skipped, truncated = [], 0, 0
    for ex in examples:
        messages = build_messages(ex, use_context=use_context, use_climate=use_climate)
        prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tok(ex["answer"].strip() + tok.eos_token, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) > max_len - 32:  # no room for a meaningful answer
            skipped += 1
            continue
        room = max_len - len(prompt_ids)
        if len(answer_ids) > room:
            answer_ids = answer_ids[:room]
            truncated += 1

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        rows.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels})
    print(f"  tokenized {len(rows)} examples | skipped (prompt too long): {skipped} | answers truncated: {truncated}")
    return Dataset.from_list(rows)


def load_model(args, use_bf16):
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    kwargs = {"torch_dtype": dtype}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def merge_adapter(model_name, adapter_dir, out_dir):
    """Merge LoRA weights into the base model for simple deployment."""
    print("Merging adapter into base model ...")
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    print(f"Merged model saved to {out_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    adapter_dir = out / "adapter"
    out.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
    use_fp16 = torch.cuda.is_available() and not use_bf16  # e.g. Colab T4
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU!)'} | "
          f"precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Preparing data ...")
    train_ds = tokenize_examples(read_jsonl(args.train_file), tok, args.use_context, args.use_climate, args.max_len)
    val_ds = tokenize_examples(read_jsonl(args.val_file), tok, args.use_context, args.use_climate, args.max_len)

    model = load_model(args, use_bf16)
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if use_fp16:  # fp16 AMP needs trainable params in fp32
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
    model.enable_input_require_grads()
    trainable, total = count_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    _steps = -(-len(train_ds) // (args.batch_size * args.grad_accum)) * args.epochs
    train_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=max(1, int(args.warmup_ratio * _steps)),
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",          # needs transformers >= 4.46
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60

    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics["eval_loss"]
    metrics = {
        "args": vars(args),
        "trainable_params": trainable,
        "total_params": total,
        "train_minutes": round(minutes, 2),
        "eval_loss": eval_loss,
        "eval_perplexity": math.exp(eval_loss),
        "log_history": trainer.state.log_history,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"Done in {minutes:.1f} min | val loss {eval_loss:.4f} | ppl {math.exp(eval_loss):.2f}")
    print(f"Adapter saved to {adapter_dir}")

    if args.merge:
        del trainer, model
        torch.cuda.empty_cache()
        merge_adapter(args.model_name, str(adapter_dir), str(out / "merged"))


if __name__ == "__main__":
    main()
