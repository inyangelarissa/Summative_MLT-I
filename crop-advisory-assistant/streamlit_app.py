"""Streamlit interface for the climate-aware crop disease advisory assistant.

Two modes share one interface (same features as the Gradio app):
  * Full model   - set FULL_MODEL=1 (env var or Streamlit secret) on a host with enough memory/GPU: the same
                   AdvisoryPipeline as the Gradio app (gate -> hybrid retrieval -> LoRA generator -> safety filter).
  * Lite (default, fits free hosting) - domain gate -> crop-filtered BM25 retrieval -> weather rules ->
                   extractive advice from the retrieved passages. The 1.5B generator is NOT loaded.
"""
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

from weather import RWANDA_LOCATIONS, climate_from_values, fetch_week_summary  # noqa: E402

CROPS = ["maize", "beans", "cassava", "potato"]
WEATHER_MODES = ["No weather", "Enter manually", "Live weather (Open-Meteo)"]
STATUS = {"answered": "✅ In-domain question",
          "out_of_domain": "⛔ Outside my scope (not a maize/bean/cassava/potato disease question)",
          "empty": "Please type a question."}
REFUSAL = ("I can only help with diseases of maize, beans, cassava and potato. "
           "Please ask a question about one of these crops.")
Q_LATE = "Should I be worried about late blight in my potato field this week?"
EXAMPLES = [
    ("Late blight: cool and wet", Q_LATE, "potato", WEATHER_MODES[1], 16, 92, 70),
    ("Late blight: hot and dry", Q_LATE, "potato", WEATHER_MODES[1], 30, 40, 0),
    ("Rust on bean leaves", "What causes rust on bean leaves and how can I prevent it?", "beans", WEATHER_MODES[0], 20, 70, 20),
    ("Cassava mosaic", "How does cassava mosaic disease spread?", "cassava", WEATHER_MODES[0], 20, 70, 20),
    ("Off-topic question", "Who won the football match yesterday?", "Any", WEATHER_MODES[0], 20, 70, 20),
]
GENERIC = {"disease", "virus", "common", "bean", "maize", "cassava", "potato"}
TIP_WORDS = ("control", "manage", "avoid", "remove", "destroy", "resistant", "resistance", "rotation",
             "rotate", "fungicide", "spray", "clean seed", "certified", "sanitation", "scout", "monitor",
             "weed", "spacing", "drain", "irrigat", "burn", "bury", "intercrop", "plant early", "uproot")
SKIP = re.compile(r"\d+(\.\d+)?\s*(g|kg|mg|ml|l)\b|/ha|per hectare|dose|dosage|et al|\(\d{4}\)|\bfig\.|\btable\b", re.I)


def flag_on(name):
    if os.environ.get(name) == "1":
        return True
    try:
        return str(st.secrets.get(name, "")) == "1"
    except Exception:
        return False


# ----------------------------------------------------------------------------- engines
class FullEngine:
    mode = "Full model: domain gate + hybrid retrieval + LoRA-tuned generator + safety filter"

    def __init__(self):
        from pipeline import AdvisoryPipeline, PipelineConfig
        self.pipe = AdvisoryPipeline(PipelineConfig.from_json(str(BASE / "app" / "config.json")))
        _ = self.pipe.gate, self.pipe.retriever

    def answer(self, question, crop, climate):
        return self.pipe.answer(question, "en", "en", crop, climate)


class LiteEngine:
    mode = "Lite (free hosting): domain gate + crop-filtered BM25 retrieval + weather rules"

    def __init__(self):
        from domain_gate import DomainGate
        from retriever import BM25Retriever, load_corpus
        self.corpus = load_corpus(str(BASE / "data" / "processed" / "corpus.jsonl"))
        self.by_id = {c["chunk_id"]: c for c in self.corpus}
        self.retrievers = {"Any": BM25Retriever(self.corpus)}
        for crop in CROPS:
            subset = [c for c in self.corpus if c.get("crop") == crop and c.get("diseases")]
            if len(subset) < 5:
                subset = [c for c in self.corpus if c.get("crop") == crop]
            if subset:
                self.retrievers[crop] = BM25Retriever(subset)
        self.gate = DomainGate.load(str(BASE / "results" / "gate" / "best"))

    def answer(self, question, crop, climate):
        t0, timings = time.time(), {}
        trace = {"crop": crop, "climate": climate, "question_en": question}
        key = crop or "Any"
        gate = self.gate.check(question if not crop else f"{question} ({crop})")
        timings["gate"] = round(time.time() - t0, 3)
        if not gate["in_domain"]:
            return {**trace, "status": "out_of_domain", "gate": gate, "answer": REFUSAL, "answer_en": REFUSAL,
                    "passages": [], "flags": [], "timings": timings}
        t = time.time()
        ids = self.retrievers.get(key, self.retrievers["Any"]).search([question], 3)[0]
        passages = [self.by_id[i] for i in ids if i in self.by_id]
        timings["retrieve"] = round(time.time() - t, 3)
        advice, risk = make_advice(question, key, climate, passages)
        timings["total"] = round(time.time() - t0, 3)
        return {**trace, "status": "answered", "gate": gate, "answer": advice, "answer_en": advice, "risk": risk,
                "passages": [{"chunk_id": p["chunk_id"], "source_doc": p.get("source_doc", ""), "title": p.get("title", ""),
                              "url": p.get("url", ""), "text": p["text"]} for p in passages],
                "flags": [], "timings": timings}


@st.cache_resource
def get_engine():
    if flag_on("FULL_MODEL"):
        return FullEngine()
    return LiteEngine()


# ----------------------------------------------------------------------------- lite-mode advice
def climate_summary(climate):
    return (f"{climate['temperature_c']}°C, {climate['humidity_pct']}% humidity, "
            f"{climate['rainfall_mm_7d']} mm rain in the last 7 days")


def match_disease(question, crop):
    from knowledge import DISEASES
    words = set(re.findall(r"[a-z]+", question.lower()))
    scores = {}
    for name in DISEASES.get(crop, {}):
        hit = sum(1 for w in name.split() if w not in GENERIC and w in words)
        if hit:
            scores[name] = hit
    if not scores:
        return None
    best = max(scores.values())
    top = [n for n, s in scores.items() if s == best]
    return top[0] if len(top) == 1 else None


def pick_disease(question, crop, passages):
    from knowledge import detect
    _, pairs = detect(question, manifest_crop=crop if crop != "Any" else None)
    if pairs:
        return pairs[0]
    if crop != "Any":
        name = match_disease(question, crop)
        if name:
            return (crop, name)
    for passage in passages:
        passage_crop, found = passage.get("crop"), passage.get("diseases") or []
        if passage_crop and found and (crop == "Any" or passage_crop == crop):
            return (passage_crop, found[0])
    return None


def extract_tips(passages, limit=3):
    tips, seen = [], set()
    for passage in passages:
        for sentence in re.split(r"(?<=[.!?])\s+", passage["text"].replace("\n", " ")):
            sentence = sentence.strip()
            low = sentence.lower()
            if 40 <= len(sentence) <= 260 and any(w in low for w in TIP_WORDS) and not SKIP.search(sentence) and low not in seen:
                tips.append(sentence)
                seen.add(low)
        if len(tips) >= limit:
            break
    return tips[:limit]


def make_advice(question, crop, climate, passages):
    from knowledge import CLIMATE_RULES, assess
    selected, risk, parts = pick_disease(question, crop, passages), None, []
    if selected:
        crop_name, disease = selected
        rule = assess(crop_name, disease, climate) if climate else None
        if rule:
            risk = rule["risk"]
            mechanism = CLIMATE_RULES.get(selected, {}).get("mechanism", "")
            parts.append(f"**Weather-based risk: {risk.upper()}** for {disease} in {crop_name}. {mechanism} "
                         f"Current conditions: {climate_summary(climate)}.")
        else:
            parts.append(f"This looks like a question about **{disease}** in {crop_name}.")
    else:
        parts.append("I could not identify one specific disease, so here is what the most relevant sources say.")
    tips = extract_tips(passages)
    if tips:
        parts.append("**What the sources say:**\n" + "\n".join(f"- {t}" for t in tips))
    parts.append("Scout your field regularly, avoid moving plant material or soil between fields, "
                 "and ask a local extension officer about approved control options.")
    return "\n\n".join(parts), risk


# ----------------------------------------------------------------------------- weather and rendering
def resolve_weather(mode, temp, hum, rain, location):
    if mode == WEATHER_MODES[1]:
        return climate_from_values(temp, hum, rain), "Using the weather you entered."
    if mode == WEATHER_MODES[2]:
        try:
            lat, lon = RWANDA_LOCATIONS[location]
            clim = fetch_week_summary(lat, lon)
            if clim:
                return clim, f"Live weather for {location} (last 7 days, regional summary)."
            return None, "Live weather returned no data; answering without weather."
        except Exception as e:
            return None, f"Could not fetch live weather ({type(e).__name__}); answering without weather."
    return None, "No weather used."


def render(result):
    res, climate, wnote = result["res"], result["climate"], result["note"]
    status = STATUS[res["status"]]
    if res.get("gate"):
        status += f"  (gate score {res['gate']['score']})"
    st.markdown(f"**{status}**")
    if res["status"] == "out_of_domain":
        st.warning(res["answer"])
    else:
        st.markdown(res["answer"])
        if res.get("flags"):
            st.caption("⚠️ A statement with an unverified quantity was removed for safety.")
        if res.get("risk"):
            st.metric("Weather-based risk", res["risk"].upper())
    with st.expander("Sources used"):
        if not res["passages"]:
            st.write("_No passages were used._")
        for i, p in enumerate(res["passages"], 1):
            label = p.get("title") or p.get("source_doc") or "Source"
            link = f" ([link]({p['url']}))" if p.get("url") else ""
            text = p["text"]
            st.markdown(f"**[{i}] {label}**{link}\n\n> {text[:450]}{'…' if len(text) > 450 else ''}")
    with st.expander("Weather and technical details"):
        st.caption("Weather used")
        st.json({"note": wnote, "weather": climate})
        st.caption("Pipeline trace (English answer, gate, timings)")
        st.json({k: res.get(k) for k in ("status", "question_en", "gate", "answer_en", "flags", "timings")})


def load_example(i):
    _, q, crop, mode, temp, hum, rain = EXAMPLES[i]
    st.session_state.update(q=q, crop=crop, mode=mode, temp=temp, hum=hum, rain=rain, _run=True)


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="Crop Disease Advisory Assistant", page_icon="🌱", layout="wide")
st.title("🌱 Climate-Aware Crop Disease Advisory Assistant")
st.write("Ask about diseases of **maize, beans, cassava and potato**. Optionally add recent weather so the advice "
         "reflects disease risk.")

try:
    engine = get_engine()
except Exception as exc:
    st.error(f"Could not load the assistant: {type(exc).__name__}: {exc}")
    st.stop()
st.caption(f"Mode: {engine.mode}")

for _k, _v in dict(q="", crop="Any", mode=WEATHER_MODES[0], temp=20, hum=70, rain=20, loc="Kigali").items():
    st.session_state.setdefault(_k, _v)

left, right = st.columns([3, 4], gap="large")
with left:
    question = st.text_area("Your question", key="q", height=110,
                            placeholder="e.g. Should I be worried about late blight in my potato field this week?")
    crop = st.selectbox("Crop (optional)", ["Any"] + CROPS, key="crop")
    mode = st.radio("Weather context", WEATHER_MODES, key="mode")
    temp, hum, rain, location = 20, 70, 20, "Kigali"
    if mode == WEATHER_MODES[1]:
        temp = st.slider("Average temperature (°C)", 5, 38, key="temp")
        hum = st.slider("Average relative humidity (%)", 20, 100, key="hum")
        rain = st.slider("Rainfall in the last 7 days (mm)", 0, 200, key="rain")
    elif mode == WEATHER_MODES[2]:
        location = st.selectbox("Location", list(RWANDA_LOCATIONS), key="loc")
    clicked = st.button("Get advice", type="primary", key="go", use_container_width=True)
    run = st.session_state.pop("_run", False) or clicked

with right:
    if run:
        if not question.strip():
            st.session_state["result"] = None
            st.warning(STATUS["empty"])
        else:
            climate, note = resolve_weather(mode, temp, hum, rain, location)
            with st.spinner("Thinking…"):
                res = engine.answer(question.strip(), None if crop == "Any" else crop, climate)
            st.session_state["result"] = {"res": res, "climate": climate, "note": note}
    if st.session_state.get("result"):
        render(st.session_state["result"])
    else:
        st.info("Type a question and press **Get advice**, or try an example below.")

st.subheader("Try an example")
st.caption("The first two differ only in the weather.")
cols = st.columns(len(EXAMPLES))
for i, (label, *_rest) in enumerate(EXAMPLES):
    cols[i].button(label, key=f"ex{i}", on_click=load_example, args=(i,), use_container_width=True)
