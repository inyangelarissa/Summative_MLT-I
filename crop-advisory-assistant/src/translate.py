"""English <-> Kinyarwanda machine translation (delivery layer only).

Default: NLLB-200 distilled 600M (supports eng_Latn and kin_Latn). Optional backend "opus" uses one
Marian model per direction (verify the model ids exist before relying on them; see config).
Translation is done sentence by sentence: NLLB quality drops and text can be truncated on long inputs.

Limitations to report: MT can mistranslate disease/crop terminology and numbers, so the ASSESSED quality
of this layer (chrF/BLEU on a native-speaker-verified test set) matters more than its mere presence.
"""
import re

LANG_CODES = {"en": "eng_Latn", "rw": "kin_Latn"}
DEFAULT_OPUS = {("en", "rw"): "Helsinki-NLP/opus-mt-en-rw", ("rw", "en"): "Helsinki-NLP/opus-mt-rw-en"}


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class Translator:
    def __init__(self, backend="nllb", model_name="facebook/nllb-200-distilled-600M", opus_models=None,
                 device="auto", max_new_tokens=256, num_beams=4, batch_size=8):
        self.backend, self.model_name = backend, model_name
        self.opus_models = opus_models or DEFAULT_OPUS
        self.max_new_tokens, self.num_beams, self.batch_size = max_new_tokens, num_beams, batch_size
        self.device = device
        self._models, self._cache = {}, {}

    # -- loading ---------------------------------------------------------------------------
    def _device(self):
        import torch
        return "cuda" if (self.device == "auto" and torch.cuda.is_available()) else (
            "cpu" if self.device == "auto" else self.device)

    def _load(self, src, tgt):
        key = self.model_name if self.backend == "nllb" else self.opus_models[(src, tgt)]
        if key not in self._models:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            dev = self._device()
            tok = AutoTokenizer.from_pretrained(key)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                key, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
            self._models[key] = (tok, model)
        return self._models[key]

    # -- translation -----------------------------------------------------------------------
    def _translate_sentences(self, sents, src, tgt):
        import torch
        tok, model = self._load(src, tgt)
        out = [None] * len(sents)
        order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            batch = [sents[i] for i in idx]
            kw = {}
            if self.backend == "nllb":
                tok.src_lang = LANG_CODES[src]
                kw["forced_bos_token_id"] = tok.convert_tokens_to_ids(LANG_CODES[tgt])
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams, **kw)
            for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = t.strip()
        return out

    def translate(self, text, src, tgt):
        if not text or src == tgt:
            return text
        key = (text, src, tgt)
        if key in self._cache:
            return self._cache[key]
        lines = text.split("\n")
        flat, where = [], []
        for li, line in enumerate(lines):
            for s in split_sentences(line):
                flat.append(s)
                where.append(li)
        translated = self._translate_sentences(flat, src, tgt) if flat else []
        rebuilt = [[] for _ in lines]
        for li, t in zip(where, translated):
            rebuilt[li].append(t)
        result = "\n".join(" ".join(parts) for parts in rebuilt)
        self._cache[key] = result
        return result
