#!/usr/bin/env python
"""Stage 3: rule-grounded, climate-conditioned examples (SYNTHETIC).

For a passage about a disease that has a rule in knowledge.CLIMATE_RULES we create the SAME
question under different weather scenarios (high / low / moderate risk). The answer is built
from the rule + safe, generic cultural practices (no pesticide names or dosages). This teaches
the model to condition its advice on weather, and gives you matched pairs (pair_id) for the
climate-sensitivity test.

  python src/build_climate.py --chunks data/interim/chunks.jsonl --out data/interim/climate.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knowledge import CLIMATE_RULES, CROP_DISPLAY, assess  # noqa: E402

Q_TEMPLATES = [
    "Should I be worried about {d} in my {c} field this week?",
    "Is the current weather risky for {d} in my {c} field?",
    "What is the risk of {d} in my {c} field after this weather?",
    "My {c} field has had this weather recently. How likely is {d}?",
    "Do I need to take action against {d} in my {c} field now?",
]
OPENERS = {
    "high": ["The recent weather puts your {c} field at high risk of {d}.",
             "The risk of {d} in your {c} field is high right now."],
    "moderate": ["There is a moderate risk of {d} in your {c} field.",
                 "The weather is partly favourable for {d} in your {c} field, so the risk is moderate."],
    "low": ["The risk of {d} in your {c} field is low at the moment.",
            "The recent weather does not strongly favour {d}, so the risk is low."],
}
FIT = {"high": "match these conditions closely", "moderate": "partly match these conditions",
       "low": "do not match these conditions well"}
ACTIONS = {
    "high": ["Scout your field today and again every 2-3 days, checking lower and inner leaves first.",
             "Remove and destroy badly affected plants or leaves away from the field.",
             "Avoid working in the field while plants are wet, so you do not spread the disease.",
             "Contact your local extension officer or agro-dealer about suitable control options for your area."],
    "moderate": ["Scout your field at least once a week and check for the first symptoms.",
                 "Remove any plants that show symptoms early.",
                 "Keep the field weeded so air can move between plants.",
                 "Ask your local extension officer what to do if symptoms appear."],
    "low": ["Continue routine scouting, because conditions can change quickly when the weather turns.",
            "Keep the field clean and weeded.",
            "Use healthy planting material and avoid moving soil or tools from affected fields."],
}
CLOSERS = ["This weather-based risk is a guide, not a forecast, so confirm with your local extension officer.",
           "Weather is only one factor, so check your plants regularly and ask an extension officer if unsure."]


def season_label(rain):
    return "rainy season" if rain >= 40 else "dry season" if rain < 10 else "between seasons"


def sample_climate(crop, disease, target, rng, tries=2000):
    for _ in range(tries):
        c = {"temperature_c": rng.randint(12, 33), "humidity_pct": rng.randint(35, 98),
             "rainfall_mm_7d": rng.choice([0, 0, 2, 5, 10, 15, 20, 30, 45, 60, 80, 110, 140])}
        if assess(crop, disease, c)["risk"] == target:
            c["season"] = season_label(c["rainfall_mm_7d"])
            return c
    return None


def make_answer(crop, disease, climate, rng):
    a = assess(crop, disease, climate)
    risk, rule = a["risk"], CLIMATE_RULES[(crop, disease)]
    c = CROP_DISPLAY[crop]
    opener = rng.choice(OPENERS[risk]).format(c=c, d=disease)
    reason = (f"{rule['mechanism']} Recent conditions ({climate['temperature_c']}°C, {climate['humidity_pct']}% "
              f"humidity, {climate['rainfall_mm_7d']} mm of rain in the last 7 days) {FIT[risk]}.")
    acts = " ".join(rng.sample(ACTIONS[risk], k=min(3, len(ACTIONS[risk]))))
    return f"{opener} {reason} {acts} {rng.choice(CLOSERS)}", risk


def build_climate_rows(chunks, seed=42, chunks_per_disease=4, extra_per_chunk=1):
    rng = random.Random(seed)
    by_pair = {}
    for ch in chunks:
        for crop, disease in ch.get("disease_pairs", []):
            if (crop, disease) in CLIMATE_RULES:
                by_pair.setdefault((crop, disease), []).append(ch)
    rows = []
    for (crop, disease), cl in sorted(by_pair.items()):
        rng.shuffle(cl)
        for ch in cl[:chunks_per_disease]:
            q = rng.choice(Q_TEMPLATES).format(d=disease, c=CROP_DISPLAY[crop])
            targets = [("high", "low")] + [(rng.choice(["moderate", "high", "low"]),) for _ in range(extra_per_chunk)]
            for gi, group in enumerate(targets):
                pair_id = f"pair-{ch['chunk_id']}-{gi}" if len(group) == 2 else None
                for j, tgt in enumerate(group):
                    clim = sample_climate(crop, disease, tgt, rng)
                    if clim is None:
                        continue
                    ans, risk = make_answer(crop, disease, clim, rng)
                    qq = q if pair_id else rng.choice(Q_TEMPLATES).format(d=disease, c=CROP_DISPLAY[crop])
                    rows.append({"id": f"cl-{ch['chunk_id']}-{gi}-{j}", "type": "climate", "question": qq,
                                 "answer": ans, "context": [ch["text"]], "gold_chunk_id": ch["chunk_id"],
                                 "source_doc": ch["source_doc"], "crop": crop, "diseases": [disease],
                                 "climate": clim, "risk": risk, "pair_id": pair_id, "origin": "rule_template"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/interim/chunks.jsonl")
    ap.add_argument("--out", default="data/interim/climate.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks_per_disease", type=int, default=4)
    ap.add_argument("--extra_per_chunk", type=int, default=1)
    args = ap.parse_args()
    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]
    rows = build_climate_rows(chunks, args.seed, args.chunks_per_disease, args.extra_per_chunk)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(rows)} climate examples | by risk: {dict(Counter(r['risk'] for r in rows))} -> {args.out}")
    print("REMINDER: verify the thresholds in src/knowledge.py against the literature and cite them in your report.")
    missing = [k for k in CLIMATE_RULES if not any(k in {tuple(p) for p in c.get('disease_pairs', [])} for c in chunks)]
    if missing:
        print("No passages found for these ruled diseases (add documents):", missing)


if __name__ == "__main__":
    main()
