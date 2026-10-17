#!/usr/bin/env python
"""Domain gate: decides whether a question is IN-DOMAIN (crop-disease advice) or OUT-OF-DOMAIN.

Label convention: 1 = in-domain, 0 = out-of-domain (OOD). A gate outputs p_in = P(in-domain); the
question is accepted if p_in >= threshold. The costly mistake is letting an OOD question through
(false accept), so the threshold is tuned on the validation set to reach a target OOD recall.

Three interchangeable models (all expose predict_proba(texts) -> p_in):
  tfidf       TF-IDF (word + char n-grams) + logistic regression     (baseline)
  embed       frozen sentence embeddings + logistic regression
  distilbert  DistilBERT fine-tuned end-to-end

CLI example:
  python src/domain_gate.py --method tfidf --data_dir data/processed --out_dir results/gate/tfidf
"""
import argparse
import json
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- data + metrics
def load_split(data_dir, split):
    p = Path(data_dir) / f"gate_{split}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def texts_labels(rows):
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def gate_metrics(y_true, p_in, thr=0.5):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    y = np.asarray(y_true)
    pred = (np.asarray(p_in) >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"threshold": round(float(thr), 4), "n": int(len(y)), "n_ood": int((y == 0).sum()),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "ood_precision": round(float(p[0]), 4), "ood_recall": round(float(r[0]), 4), "ood_f1": round(float(f[0]), 4),
            "in_precision": round(float(p[1]), 4), "in_recall": round(float(r[1]), 4), "in_f1": round(float(f[1]), 4),
            "macro_f1": round(float((f[0] + f[1]) / 2), 4),
            "ood_false_accept_rate": round(float(1 - r[0]), 4),   # OOD questions wrongly let through
            "in_false_reject_rate": round(float(1 - r[1]), 4)}    # valid questions wrongly refused


def tune_threshold(y_true, p_in, min_ood_recall=0.95):
    """Pick the threshold that keeps in-domain recall as high as possible while OOD recall >= target.
    Falls back to the best macro-F1 threshold if the target is unreachable. Tune on VALIDATION only."""
    cands = np.unique(np.concatenate([np.round(np.asarray(p_in), 4), np.linspace(0.05, 0.95, 19)]))
    best, best_key = 0.5, None
    feasible = False
    for t in cands:
        m = gate_metrics(y_true, p_in, t)
        ok = m["ood_recall"] >= min_ood_recall
        key = (ok, m["in_recall"] if ok else m["macro_f1"], m["macro_f1"])
        if best_key is None or key > best_key:
            best, best_key, feasible = float(t), key, ok
    return best, feasible


def per_category(rows, p_in, thr):
    """Accuracy per category (OOD category or crop name): shows WHICH near-domain topics slip through."""
    out = {}
    for cat in sorted({r.get("category", "?") for r in rows}):
        idx = [i for i, r in enumerate(rows) if r.get("category", "?") == cat]
        correct = sum(int((p_in[i] >= thr) == bool(rows[i]["label"])) for i in idx)
        out[cat] = {"n": len(idx), "accuracy": round(correct / len(idx), 3), "label": rows[idx[0]]["label"]}
    return out


def error_rows(rows, p_in, thr):
    errs = []
    for r, p in zip(rows, p_in):
        if int(p >= thr) != r["label"]:
            errs.append({"text": r["text"], "label": r["label"], "p_in": round(float(p), 3),
                         "category": r.get("category"), "error": "false_accept" if r["label"] == 0 else "false_reject"})
    return sorted(errs, key=lambda e: -abs(e["p_in"] - 0.5))


# ----------------------------------------------------------------------------- models
class TfidfGate:
    kind = "tfidf"

    def __init__(self, C=10.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)),
        ])
        self.pipe = Pipeline([("f", feats), ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000))])

    def fit(self, texts, labels, **_):
        self.pipe.fit(texts, labels)
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba(texts)[:, 1]

    def save(self, d):
        import joblib
        joblib.dump(self.pipe, Path(d) / "model.joblib")

    @classmethod
    def load(cls, d, **_):
        import joblib
        g = cls()
        g.pipe = joblib.load(Path(d) / "model.joblib")
        return g


class EmbeddingGate:
    kind = "embed"

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", encode_fn=None, C=10.0):
        from sklearn.linear_model import LogisticRegression
        self.model_name = model_name
        if encode_fn is None:
            from retriever import sentence_transformer_encoder
            encode_fn = sentence_transformer_encoder(model_name)
        self.encode = encode_fn
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)

    def fit(self, texts, labels, **_):
        self.clf.fit(self.encode(texts), labels)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.encode(texts))[:, 1]

    def save(self, d):
        import joblib
        joblib.dump({"clf": self.clf, "model_name": self.model_name}, Path(d) / "model.joblib")

    @classmethod
    def load(cls, d, encode_fn=None, **_):
        import joblib
        blob = joblib.load(Path(d) / "model.joblib")
        g = cls(blob["model_name"], encode_fn)
        g.clf = blob["clf"]
        return g


class TransformerGate:
    kind = "distilbert"

    def __init__(self, model_dir=None, model_name="distilbert-base-uncased", max_len=64):
        self.model_name, self.max_len, self.model_dir = model_name, max_len, model_dir
        self.tok = self.model = None
        if model_dir:
            self._load(model_dir)

    def _load(self, d):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = AutoModelForSequenceClassification.from_pretrained(d).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def fit(self, texts, labels, val_texts=None, val_labels=None, out_dir="results/gate/distilbert_ckpt",
            epochs=4, lr=3e-5, batch_size=32, seed=42, **_):
        import torch
        import torch.nn.functional as F
        from datasets import Dataset
        from sklearn.metrics import f1_score
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                                  Trainer, TrainingArguments, set_seed)
        set_seed(seed)
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

        def make(txts, labs):
            ds = Dataset.from_dict({"text": list(txts), "label": [int(x) for x in labs]})
            return ds.map(lambda b: tok(b["text"], truncation=True, max_length=self.max_len), batched=True,
                          remove_columns=["text"])

        train_ds = make(texts, labels)
        val_ds = make(val_texts, val_labels) if val_texts is not None else None
        labels = np.asarray(labels)
        n0, n1 = max(1, int((labels == 0).sum())), max(1, int((labels == 1).sum()))
        class_w = torch.tensor([len(labels) / (2 * n0), len(labels) / (2 * n1)], dtype=torch.float)

        class WeightedTrainer(Trainer):  # weighted cross-entropy: OOD is the rare class
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                y = inputs.pop("labels")
                out = model(**inputs)
                loss = F.cross_entropy(out.logits, y, weight=class_w.to(out.logits.device))
                return (loss, out) if return_outputs else loss

        def compute_metrics(p):
            return {"macro_f1": float(f1_score(p.label_ids, p.predictions.argmax(-1), average="macro"))}

        args = TrainingArguments(output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr,
                                 per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                 eval_strategy="epoch" if val_ds else "no", save_strategy="epoch" if val_ds else "no",
                                 load_best_model_at_end=bool(val_ds), metric_for_best_model="macro_f1",
                                 greater_is_better=True, save_total_limit=1, weight_decay=0.01, warmup_ratio=0.1,
                                 fp16=torch.cuda.is_available(), report_to="none", seed=seed, logging_steps=20)
        trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                                  data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
        trainer.train()
        self.log_history = trainer.state.log_history
        self.tok, self.model = tok, trainer.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        return self

    def predict_proba(self, texts, batch_size=64):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(list(texts[i:i + batch_size]), truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt").to(self.device)
                out.append(torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def save(self, d):
        self.model.save_pretrained(Path(d) / "hf")
        self.tok.save_pretrained(Path(d) / "hf")

    @classmethod
    def load(cls, d, **_):
        return cls(model_dir=str(Path(d) / "hf"))


KINDS = {"tfidf": TfidfGate, "embed": EmbeddingGate, "distilbert": TransformerGate}


# ----------------------------------------------------------------------------- deployment wrapper
class DomainGate:
    """What the app uses:  gate = DomainGate.load('results/gate/best');  gate.check('my maize has rust')"""

    def __init__(self, model, threshold):
        self.model, self.threshold = model, threshold

    def check(self, text):
        p = float(self.model.predict_proba([text])[0])
        return {"in_domain": p >= self.threshold, "score": round(p, 4)}

    @classmethod
    def load(cls, directory, **kw):
        meta = json.loads((Path(directory) / "gate_meta.json").read_text())
        return cls(KINDS[meta["kind"]].load(directory, **kw), meta["threshold"])


def run_experiment(model, name, data, out_dir, min_ood_recall=0.95, fit_kwargs=None):
    """Fit on train, tune threshold on val, report val+test, save model + metrics + errors.
    `data` = {"train": rows, "val": rows, "test": rows}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr_x, tr_y = texts_labels(data["train"])
    va_x, va_y = texts_labels(data["val"])
    model.fit(tr_x, tr_y, val_texts=va_x, val_labels=va_y, **(fit_kwargs or {}))
    p_val = model.predict_proba(va_x)
    thr, feasible = tune_threshold(va_y, p_val, min_ood_recall)
    result = {"name": name, "kind": model.kind, "tuned_threshold": thr, "target_reachable_on_val": feasible,
              "min_ood_recall_target": min_ood_recall, "val@0.5": gate_metrics(va_y, p_val, 0.5),
              "val@tuned": gate_metrics(va_y, p_val, thr)}
    p_test = None
    if data.get("test"):
        te_x, te_y = texts_labels(data["test"])
        p_test = model.predict_proba(te_x)
        result["test@0.5"] = gate_metrics(te_y, p_test, 0.5)
        result["test@tuned"] = gate_metrics(te_y, p_test, thr)
        result["test_per_category"] = per_category(data["test"], p_test, thr)
        with open(out / "errors_test.jsonl", "w", encoding="utf-8") as f:
            for e in error_rows(data["test"], p_test, thr):
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    model.save(out)
    (out / "gate_meta.json").write_text(json.dumps({"kind": model.kind, "threshold": thr, "name": name}))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result, p_val, p_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(KINDS), required=True)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--min_ood_recall", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    data = {sp: load_split(args.data_dir, sp) for sp in ("train", "val", "test")}
    model = EmbeddingGate(args.embed_model) if args.method == "embed" else KINDS[args.method]()
    fit_kwargs = {"epochs": args.epochs} if args.method == "distilbert" else {}
    result, _, _ = run_experiment(model, args.method, data, args.out_dir, args.min_ood_recall, fit_kwargs)
    print(json.dumps({k: v for k, v in result.items() if k != "test_per_category"}, indent=2))


if __name__ == "__main__":
    main()
