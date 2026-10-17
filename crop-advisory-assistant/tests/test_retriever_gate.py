"""Offline test for retriever.py and domain_gate.py (BM25, hashed-embedding stand-in, TF-IDF gate).

    python tests/test_retriever_gate.py
"""
import json
import re
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from smoke_test import build_toy_dataset  # noqa: E402
from domain_gate import (DomainGate, EmbeddingGate, TfidfGate, gate_metrics, load_split, run_experiment,  # noqa: E402
                         tune_threshold)
from retriever import (BM25Retriever, DenseRetriever, HybridRetriever, evaluate_retrieval,  # noqa: E402
                       load_corpus, miss_analysis, read_jsonl)


def fake_encode(texts, dim=256):
    """Deterministic hashed bag-of-words embedding (stand-in for a real sentence encoder)."""
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, zlib.crc32(w.encode()) % dim] += 1.0
    return out


def main():
    with tempfile.TemporaryDirectory() as tmp:
        processed = build_toy_dataset(tmp)
        corpus = load_corpus(processed / "corpus.jsonl")
        val, test = read_jsonl(processed / "val.jsonl"), read_jsonl(processed / "test.jsonl")

        bm25 = BM25Retriever(corpus)
        dense = DenseRetriever(corpus, encode_fn=fake_encode, name="fake-dense")
        hybrid = HybridRetriever([bm25, dense])
        for r in (bm25, dense, hybrid):
            res = evaluate_retrieval(r, test, corpus)
            print(r.name, res["overall"])
            assert 0 <= res["overall"]["recall@10"] <= 1 and "by_type" in res
        assert len(miss_analysis(bm25, test, corpus, k=1)) >= 0

        dense.save(Path(tmp) / "idx")
        loaded = DenseRetriever.load(Path(tmp) / "idx", encode_fn=fake_encode)
        q = ["What should farmers know about potato late blight?"]
        assert loaded.search(q, 3) == dense.search(q, 3)

        # --- gate ---
        data = {sp: load_split(processed, sp) for sp in ("train", "val", "test")}
        print({sp: (len(v), sum(r["label"] == 0 for r in v)) for sp, v in data.items()}, "(n, n_ood)")
        res, p_val, p_test = run_experiment(TfidfGate(), "tfidf", data, Path(tmp) / "gate_tfidf")
        print("tfidf test@tuned:", res["test@tuned"])
        assert res["test@tuned"]["macro_f1"] > 0.8, "tfidf gate should separate toy in/out-of-domain easily"
        gate = DomainGate.load(Path(tmp) / "gate_tfidf")
        assert gate.check("My potato plants have late blight, what should I do?")["in_domain"]
        assert not gate.check("Who won the football match yesterday?")["in_domain"]

        res2, _, _ = run_experiment(EmbeddingGate(encode_fn=fake_encode), "embed", data, Path(tmp) / "gate_embed")
        print("embed test@tuned:", res2["test@tuned"]["macro_f1"])
        gate2 = DomainGate.load(Path(tmp) / "gate_embed", encode_fn=fake_encode)
        assert "score" in gate2.check("Is my maize sick?")

        m = gate_metrics([1, 1, 0, 0], [0.9, 0.6, 0.4, 0.2])
        assert m["ood_recall"] == 1.0 and m["in_recall"] == 1.0
        thr, ok = tune_threshold([1, 1, 0, 0], [0.9, 0.6, 0.7, 0.2], min_ood_recall=1.0)
        assert ok and thr > 0.7
    print("\nRETRIEVER + GATE TEST PASSED")


if __name__ == "__main__":
    main()
