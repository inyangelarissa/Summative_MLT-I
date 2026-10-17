"""Retrievers (BM25, dense embeddings + FAISS, hybrid) and retrieval evaluation.

Inputs come from stage 6 of the data pipeline:
  data/processed/corpus.jsonl   {chunk_id, source_doc, crop, diseases, text}
  data/processed/{val,test}.jsonl  rows with `question` and `gold_chunk_id`

All retrievers share one interface:  search(list_of_queries, k) -> list of lists of chunk_ids.
The dense retriever takes an injectable `encode_fn` so it can be unit-tested without downloading a model.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bm25 import BM25


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_corpus(path="data/processed/corpus.jsonl"):
    return read_jsonl(path)


# ----------------------------------------------------------------------------- retrievers
class BM25Retriever:
    """Lexical baseline (B1): scores passages by weighted word overlap with the query."""
    name = "bm25"

    def __init__(self, corpus):
        self.ids = [c["chunk_id"] for c in corpus]
        self.bm25 = BM25([c["text"] for c in corpus])

    def search(self, queries, k=10):
        return [[self.ids[i] for i in self.bm25.top_k(q, k)] for q in queries]


def sentence_transformer_encoder(model_name, batch_size=64):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def encode(texts):
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
    return encode


def _normalize(x):
    x = np.asarray(x, dtype="float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class DenseRetriever:
    """Embeds passages and queries into one vector space; relevance = cosine similarity.

    Vectors are L2-normalised, so the inner product (what FAISS IndexFlatIP computes) equals cosine.
    Some models (e.g. BGE) expect an instruction prefix on the QUERY side only: use `query_prefix`.
    """

    def __init__(self, corpus, model_name=None, encode_fn=None, query_prefix="", doc_prefix="",
                 embeddings=None, name=None):
        self.ids = [c["chunk_id"] for c in corpus]
        self.model_name, self.query_prefix, self.doc_prefix = model_name, query_prefix, doc_prefix
        self.name = name or (model_name or "dense").split("/")[-1]
        if encode_fn is None:
            encode_fn = sentence_transformer_encoder(model_name)
        self.encode = encode_fn
        if embeddings is None:
            embeddings = self.encode([doc_prefix + c["text"] for c in corpus])
        self.emb = _normalize(embeddings)
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)
        except ImportError:  # numpy fallback: exact search, fine for a few thousand passages
            self.index = None

    def search(self, queries, k=10):
        q = _normalize(self.encode([self.query_prefix + x for x in queries]))
        k = min(k, len(self.ids))
        if self.index is not None:
            _, idx = self.index.search(q, k)
        else:
            idx = np.argsort(-(q @ self.emb.T), axis=1)[:, :k]
        return [[self.ids[j] for j in row if j >= 0] for row in idx]

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.emb)
        (d / "meta.json").write_text(json.dumps({"ids": self.ids, "model_name": self.model_name,
                                                 "query_prefix": self.query_prefix, "doc_prefix": self.doc_prefix,
                                                 "name": self.name}))

    @classmethod
    def load(cls, directory, encode_fn=None):
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        corpus = [{"chunk_id": i, "text": ""} for i in meta["ids"]]
        return cls(corpus, meta["model_name"], encode_fn, meta["query_prefix"], meta["doc_prefix"],
                   embeddings=np.load(d / "embeddings.npy"), name=meta["name"])


class HybridRetriever:
    """Reciprocal Rank Fusion: score(d) = sum_r  w_r / (rrf_k + rank_r(d)). Combines lexical + semantic."""

    def __init__(self, retrievers, weights=None, rrf_k=60, pool=50, name=None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.rrf_k, self.pool = rrf_k, pool
        self.name = name or "hybrid(" + "+".join(r.name for r in retrievers) + ")"

    def search(self, queries, k=10):
        per = [r.search(queries, self.pool) for r in self.retrievers]
        out = []
        for qi in range(len(queries)):
            score = defaultdict(float)
            for w, res in zip(self.weights, per):
                for rank, cid in enumerate(res[qi]):
                    score[cid] += w / (self.rrf_k + rank + 1)
            out.append([c for c, _ in sorted(score.items(), key=lambda x: -x[1])[:k]])
        return out


# ----------------------------------------------------------------------------- evaluation
def _summ(ranks, drank, ks):
    n = max(1, len(ranks))
    kmax = max(ks)
    s = {"n": len(ranks)}
    for k in ks:
        s[f"recall@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 3)
    s[f"mrr@{kmax}"] = round(sum(1 / r for r in ranks if r) / n, 3)
    for k in ks:
        s[f"disease_hit@{k}"] = round(sum(1 for r in drank if r and r <= k) / n, 3)
    return s


def evaluate_retrieval(retriever, rows, corpus, ks=(1, 3, 5, 10), group_by=("type", "crop")):
    """Recall@k / MRR against the passage each question was generated from (`gold_chunk_id`).

    `disease_hit@k` is a more lenient check: did any of the top-k passages discuss the same disease as
    the gold passage? Several passages can legitimately answer one question, so Recall@k is a lower bound.
    """
    meta = {c["chunk_id"]: c for c in corpus}
    rows = [r for r in rows if r.get("gold_chunk_id") in meta]
    kmax = max(ks)
    retrieved = retriever.search([r["question"] for r in rows], kmax)
    ranks, dranks = [], []
    for r, hits in zip(rows, retrieved):
        gold = meta[r["gold_chunk_id"]]
        ranks.append(hits.index(gold["chunk_id"]) + 1 if gold["chunk_id"] in hits else None)
        gd = set(gold.get("diseases", []))
        dr = next((i + 1 for i, h in enumerate(hits) if gd and gd & set(meta[h].get("diseases", []))), None)
        dranks.append(dr)
    out = {"overall": _summ(ranks, dranks, ks)}
    for key in group_by:
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[r.get(key)].append(i)
        out[f"by_{key}"] = {g: _summ([ranks[i] for i in ix], [dranks[i] for i in ix], ks) for g, ix in groups.items()}
    out["_ranks"] = ranks
    return out


def miss_analysis(retriever, rows, corpus, k=3):
    """Categorise retrieval misses at top-k into: same_doc / same_disease / wrong_disease / nothing_found."""
    meta = {c["chunk_id"]: c for c in corpus}
    rows = [r for r in rows if r.get("gold_chunk_id") in meta]
    retrieved = retriever.search([r["question"] for r in rows], k)
    misses = []
    for r, hits in zip(rows, retrieved):
        gold = meta[r["gold_chunk_id"]]
        if gold["chunk_id"] in hits:
            continue
        gd = set(gold.get("diseases", []))
        if not hits:
            cat = "nothing_found"
        elif any(meta[h]["source_doc"] == gold["source_doc"] for h in hits):
            cat = "same_doc"
        elif gd and any(gd & set(meta[h].get("diseases", [])) for h in hits):
            cat = "same_disease"
        else:
            cat = "wrong_disease"
        misses.append({"id": r["id"], "type": r["type"], "question": r["question"], "category": cat,
                       "gold_chunk": gold["chunk_id"], "gold_text": gold["text"][:300],
                       "top1_chunk": hits[0] if hits else None,
                       "top1_text": meta[hits[0]]["text"][:300] if hits else ""})
    return misses
