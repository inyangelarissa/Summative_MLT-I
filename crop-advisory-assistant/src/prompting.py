"""Shared prompt construction.

Training and inference MUST build prompts with the same function, otherwise the
model sees a different format at test time than it saw during fine-tuning.

Expected JSONL schema (one example per line):
{
  "id": "q0001",
  "question": "How do I recognise late blight on potato?",
  "answer": "Look for water-soaked dark lesions ...",
  "context": ["passage 1 ...", "passage 2 ..."],            # optional (retrieved or gold passages)
  "climate": {"temperature_c": 18, "humidity_pct": 90,       # optional (dict or plain string)
              "rainfall_mm_7d": 60, "season": "long rains"},
  "source_doc": "faq_potato_01"                              # used to split by document
}
"""

SYSTEM_PROMPT = (
    "You are a crop disease advisory assistant for smallholder farmers growing "
    "maize, beans, cassava and potato. Answer only questions about crop diseases, "
    "their symptoms, prevention and management. Use the provided passages and "
    "weather context when they are given. Do not invent pesticide names or dosages "
    "that are not in the passages. For severe or unclear cases, advise contacting "
    "a local agricultural extension officer. If the question is outside crop "
    "disease advice, politely decline."
)

_CLIMATE_LABELS = {
    "temperature_c": ("Temperature", "°C"),
    "humidity_pct": ("Relative humidity", "%"),
    "rainfall_mm_7d": ("Rainfall (last 7 days)", " mm"),
    "season": ("Season", ""),
}


def format_climate(climate):
    """Turn a dict/str climate record into a short text block (or None)."""
    if not climate:
        return None
    if isinstance(climate, str):
        return climate.strip() or None
    parts = []
    for key, value in climate.items():
        if value is None:
            continue
        label, unit = _CLIMATE_LABELS.get(key, (key, ""))
        parts.append(f"{label}: {value}{unit}")
    return "; ".join(parts) if parts else None


def build_user_message(example, use_context=True, use_climate=True,
                       max_passages=3, max_chars_per_passage=800):
    blocks = []
    if use_climate:
        climate_text = format_climate(example.get("climate"))
        if climate_text:
            blocks.append(f"Weather context: {climate_text}")
    if use_context:
        passages = example.get("context") or []
        if isinstance(passages, str):
            passages = [passages]
        passages = [p.strip()[:max_chars_per_passage] for p in passages[:max_passages] if p.strip()]
        if passages:
            numbered = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
            blocks.append(f"Passages:\n{numbered}")
    blocks.append(f"Question: {example['question'].strip()}")
    return "\n\n".join(blocks)


def build_messages(example, use_context=True, use_climate=True, **kwargs):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(example, use_context, use_climate, **kwargs)},
    ]
