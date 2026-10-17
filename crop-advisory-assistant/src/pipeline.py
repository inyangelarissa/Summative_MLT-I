"""End-to-end inference pipeline.

  question (EN or RW) [+ crop] [+ weather]
     -> RW->EN translation (if needed)
     -> domain gate            (out-of-domain -> fixed refusal, LLM never runs)
     -> retriever (BM25 / dense / hybrid) -> top-k passages
     -> LoRA-fine-tuned LLM    (prompt = weather + passages + question, same format as training)
     -> safety filter          (drops dosage-like statements not supported by the passages)
     -> EN->RW translation (if needed)

Every component is injectable, so the pipeline can be tested offline with stand-ins.
The prompt flags (use_context / use_climate) MUST match how the deployed LoRA model was trained.
"""
import json
import re
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

from prompting import build_messages, format_climate

REFUSAL_EN = ("I can only help with crop disease questions for maize, beans, cassava and potato. "
              "Please ask me about those crops.")
DISCLAIMER_EN = ("This advice is general guidance produced by an AI model from agricultural documents and may "
                 "contain errors. For severe or unclear cases, contact your local agricultural extension officer "
                 "(for example RAB) before applying any treatment.")
FALLBACK_EN = ("I could not produce a reliable answer. Please contact your local agricultural extension officer.")
CROPS = ["maize", "beans", "cassava", "potato"]


@dataclass
class PipelineConfig:
    corpus_path: str = "data/processed/corpus.jsonl"
    gate_dir: str = "results/gate/best"
    retriever_dir: str = "results/retriever/best_dense"
    retriever_mode: str = "hybrid"            # bm25 | dense | hybrid
    top_k: int = 3
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"   # base model id, or a merged model id/path
    adapter_dir: str = None                    # LoRA adapter (loaded on top of llm_model); None if merged
    use_context: bool = True                   # must match the training run (E3/E4: True)
    use_climate: bool = True                   # must match the training run (E4: True)
    max_new_tokens: int = 256
    repetition_penalty: float = 1.05
    mt_backend: str = "nllb"
    mt_model: str = "facebook/nllb-200-distilled-600M"
    device: str = "auto"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path, base_dir=None):
        cfg = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        obj = cls(**{k: v for k, v in cfg.items() if k in known})
        base = Path(base_dir) if base_dir else Path(path).resolve().parent
        for name in ("corpus_path", "gate_dir", "retriever_dir", "adapter_dir"):
            v = getattr(obj, name)
            if v and not Path(v).is_absolute():
                setattr(obj, name, str(base / v))
        if obj.llm_model and not Path(obj.llm_model).is_absolute() and (base / obj.llm_model).exists():
            obj.llm_model = str(base / obj.llm_model)
        return obj


# ----------------------------------------------------------------------------- safety filter
DOSE_RE = re.compile(r"\d+(?:[.,]\d+)?\s?(?:(?:ml|g|kg|l|litres?|liters?|ppm|tablespoons?|teaspoons?|cc)\b|%)",
                     re.IGNORECASE)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text):
    return set(NUM_RE.findall(text.replace(",", "")))


def safety_filter(answer, allowed_text):
    """Remove sentences with dosage-like quantities (e.g. '50 ml', '2 kg', '5%') whose numbers do not occur in
    the retrieved passages / question / weather. Returns (clean_answer, flags)."""
    allowed = _numbers(allowed_text)
    kept, removed = [], []
    for sent in re.split(r"(?<=[.!?])\s+", answer.strip()):
        if not sent:
            continue
        doses = DOSE_RE.findall(sent)
        if doses and any(n not in allowed for d in doses for n in _numbers(d)):
            removed.append(sent)
        else:
            kept.append(sent)
    flags = ["removed_ungrounded_dosage"] if removed else []
    clean = " ".join(kept).strip()
    if not clean:
        return FALLBACK_EN, flags + ["no_safe_answer"]
    return clean, flags


# ----------------------------------------------------------------------------- LLM wrapper
class LLMGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tok = self.model = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        c = self.cfg
        dev = "cuda" if (c.device == "auto" and torch.cuda.is_available()) else ("cpu" if c.device == "auto" else c.device)
        tok_src = c.adapter_dir if (c.adapter_dir and (Path(c.adapter_dir) / "tokenizer_config.json").exists()) else c.llm_model
        self.tok = AutoTokenizer.from_pretrained(tok_src)
        model = AutoModelForCausalLM.from_pretrained(c.llm_model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
        if c.adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, c.adapter_dir).merge_and_unload()  # merge: faster inference
        self.model = model.to(dev).eval()

    def generate(self, messages):
        import torch
        if self.model is None:
            self._load()
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            ids = self.model.generate(**enc, max_new_tokens=self.cfg.max_new_tokens, do_sample=False,
                                      repetition_penalty=self.cfg.repetition_penalty,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------- pipeline
class AdvisoryPipeline:
    def __init__(self, cfg, gate=None, retriever=None, generator=None, translator=None, corpus=None):
        self.cfg = cfg
        self._gate, self._retriever, self._generator, self._translator = gate, retriever, generator, translator
        self._corpus = corpus
        self.corpus_by_id = {c["chunk_id"]: c for c in corpus} if corpus else None

    # lazy builders -----------------------------------------------------------------------
    @property
    def gate(self):
        if self._gate is None:
            from domain_gate import DomainGate
            self._gate = DomainGate.load(self.cfg.gate_dir)
        return self._gate

    @property
    def retriever(self):
        if self._retriever is None:
            from retriever import BM25Retriever, DenseRetriever, HybridRetriever, load_corpus
            corpus = self._corpus or load_corpus(self.cfg.corpus_path)
            self.corpus_by_id = {c["chunk_id"]: c for c in corpus}
            bm25 = BM25Retriever(corpus)
            mode = self.cfg.retriever_mode
            if mode != "bm25" and not (Path(self.cfg.retriever_dir) / "meta.json").exists():
                print(f"WARNING: dense index not found at {self.cfg.retriever_dir}; falling back to BM25")
                mode = "bm25"
            if mode == "bm25":
                self._retriever = bm25
            else:
                dense = DenseRetriever.load(self.cfg.retriever_dir)
                self._retriever = dense if mode == "dense" else HybridRetriever([bm25, dense])
        return self._retriever

    @property
    def generator(self):
        if self._generator is None:
            self._generator = LLMGenerator(self.cfg)
        return self._generator

    @property
    def translator(self):
        if self._translator is None:
            from translate import Translator
            self._translator = Translator(self.cfg.mt_backend, self.cfg.mt_model, device=self.cfg.device)
        return self._translator

    def _passages(self, ids):
        _ = self.retriever  # make sure corpus_by_id exists
        return [self.corpus_by_id[i] for i in ids if i in self.corpus_by_id]

    def _tr(self, text, src, tgt):
        return text if src == tgt else self.translator.translate(text, src, tgt)

    def disclaimer(self, lang="en"):
        return self._tr(DISCLAIMER_EN, "en", lang)

    # main entry --------------------------------------------------------------------------
    def answer(self, question, input_lang="en", output_lang="en", crop=None, climate=None):
        t0, timings = time.time(), {}
        question = (question or "").strip()
        trace = {"input_lang": input_lang, "output_lang": output_lang, "crop": crop, "climate": climate}
        if not question:
            return {**trace, "status": "empty", "answer": "", "answer_en": "", "passages": [], "flags": [],
                    "disclaimer": self.disclaimer(output_lang), "timings": timings}

        q_en = self._tr(question, input_lang, "en")
        trace["question_en"] = q_en
        timings["translate_in"] = round(time.time() - t0, 2)

        t = time.time()
        gate = self.gate.check(q_en)
        trace["gate"] = gate
        timings["gate"] = round(time.time() - t, 2)
        if not gate["in_domain"]:
            return {**trace, "status": "out_of_domain", "answer_en": REFUSAL_EN,
                    "answer": self._tr(REFUSAL_EN, "en", output_lang), "passages": [], "flags": [],
                    "disclaimer": self.disclaimer(output_lang), "timings": timings}

        q_aug = q_en if (not crop or crop in q_en.lower()) else f"{q_en} (crop: {crop})"
        t = time.time()
        ids = self.retriever.search([q_aug], self.cfg.top_k)[0]
        passages = self._passages(ids)
        timings["retrieve"] = round(time.time() - t, 2)

        example = {"question": q_aug, "context": [p["text"] for p in passages], "climate": climate}
        messages = build_messages(example, use_context=self.cfg.use_context, use_climate=self.cfg.use_climate)
        t = time.time()
        raw = self.generator.generate(messages)
        timings["generate"] = round(time.time() - t, 2)

        allowed = " ".join([q_aug, format_climate(climate) or ""] + example["context"])
        answer_en, flags = safety_filter(raw, allowed)

        t = time.time()
        answer_out = self._tr(answer_en, "en", output_lang)
        timings["translate_out"] = round(time.time() - t, 2)
        timings["total"] = round(time.time() - t0, 2)
        return {**trace, "status": "answered", "answer_en": answer_en, "answer": answer_out, "raw_answer_en": raw,
                "passages": [{"chunk_id": p["chunk_id"], "source_doc": p.get("source_doc", ""),
                              "title": p.get("title", ""), "url": p.get("url", ""), "text": p["text"]} for p in passages],
                "flags": flags, "disclaimer": self.disclaimer(output_lang), "timings": timings}
