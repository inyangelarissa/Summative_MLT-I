"""Free-hosting Streamlit interface for the retrieval and climate advisory system.

This deployment intentionally omits the 1.5B parameter generator and LoRA adapter.
It keeps the trained domain gate, BM25 retrieval, source evidence, and the
rule-based climate assessment so it can run on a free CPU service.
"""
import sys
from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

from domain_gate import DomainGate  # noqa: E402
from knowledge import CLIMATE_RULES, assess, detect  # noqa: E402
from retriever import BM25Retriever, load_corpus  # noqa: E402
from weather import RWANDA_LOCATIONS, climate_from_values, fetch_week_summary  # noqa: E402

GATE_DIR = BASE / "results" / "gate" / "best"
CORPUS_PATH = BASE / "data" / "processed" / "corpus.jsonl"
DISCLAIMER = (
    "This is a retrieval-based research prototype, not a diagnosis or pesticide "
    "prescription. Confirm symptoms and treatment with a local agricultural extension officer."
)


@st.cache_resource
def load_resources():
    corpus = load_corpus(str(CORPUS_PATH))
    retriever = BM25Retriever(corpus)
    gate = DomainGate.load(str(GATE_DIR))
    return corpus, retriever, gate


def climate_summary(climate):
    return (
        f"{climate['temperature_c']}°C, {climate['humidity_pct']}% humidity, "
        f"{climate['rainfall_mm_7d']} mm rain in the last 7 days"
    )


def make_advice(question, crop, climate, passages):
    detected_crop, pairs = detect(question, manifest_crop=crop if crop != "Any" else None)
    if not pairs and crop != "Any":
        _, pairs = detect(f"{crop} {question}", manifest_crop=crop)

    if not pairs:
        return (
            "I found relevant crop-disease evidence below. Compare the described symptoms "
            "with your plants and seek confirmation before taking action."
        ), None

    selected = pairs[0]
    rule = assess(selected[0], selected[1], climate) if climate else None
    if rule:
        risk = rule["risk"]
        reason = CLIMATE_RULES[selected]["mechanism"]
        advice = (
            f"**Weather-based risk: {risk.upper()}** for {selected[1]} in {selected[0]}. "
            f"{reason} Current conditions are {climate_summary(climate)}. "
            "Scout the field regularly, avoid spreading plant material or soil between fields, "
            "remove badly affected material away from the field, and contact an extension officer "
            "about locally approved control options."
        )
        return advice, risk

    return (
        f"I found evidence about **{selected[1]}** in {selected[0]}. "
        "The available weather rules do not cover this disease, so use the source evidence "
        "and ask an extension officer to confirm the diagnosis and treatment."
    ), None


def resolve_weather(mode, temp, humidity, rain, location):
    if mode == "Enter manually":
        return climate_from_values(temp, humidity, rain), "Using the weather you entered."
    if mode == "Live weather (Open-Meteo)":
        try:
            lat, lon = RWANDA_LOCATIONS[location]
            climate = fetch_week_summary(lat, lon)
            if climate:
                return climate, f"Live regional weather for {location}."
            return None, "No live weather data was returned."
        except Exception as exc:
            return None, f"Live weather unavailable ({type(exc).__name__}); continuing without it."
    return None, "No weather context used."


st.set_page_config(page_title="Crop Disease Advisory Assistant", page_icon="🌱", layout="wide")
st.title("🌱 Climate-Aware Crop Disease Advisory Assistant")
st.write(
    "A free, retrieval-based assistant for maize, beans, cassava, and potato disease questions. "
    "It shows the agricultural passages used to ground each response."
)
st.info("Free deployment mode: domain gate + BM25 retrieval + weather rules. No paid model endpoint is required.")

try:
    corpus, retriever, gate = load_resources()
except Exception as exc:
    st.error(f"Could not load the advisory resources: {type(exc).__name__}: {exc}")
    st.markdown(
        "The deployment must include `data/processed/corpus.jsonl` and "
        "`results/gate/best/gate_meta.json`, `model.joblib`, and the trained gate files. "
        "Commit these paths to GitHub and redeploy."
    )
    st.stop()

question = st.text_area(
    "Your question",
    placeholder="What causes rust on bean leaves and how can I prevent it?",
    height=100,
)
crop = st.selectbox("Crop (optional)", ["Any", "maize", "beans", "cassava", "potato"])
weather_mode = st.radio(
    "Weather context",
    ["No weather", "Enter manually", "Live weather (Open-Meteo)"],
    horizontal=True,
)

temp, humidity, rain, location = 20, 70, 20, "Kigali"
if weather_mode == "Enter manually":
    c1, c2, c3 = st.columns(3)
    temp = c1.slider("Average temperature (°C)", 5, 38, 20)
    humidity = c2.slider("Average humidity (%)", 20, 100, 70)
    rain = c3.slider("Rainfall in last 7 days (mm)", 0, 200, 20)
elif weather_mode == "Live weather (Open-Meteo)":
    location = st.selectbox("Rwanda location", list(RWANDA_LOCATIONS))

if st.button("Get advice", type="primary", disabled=not question.strip()):
    probability = float(gate.model.predict_proba([question])[0])
    threshold = float(getattr(gate, "threshold", 0.5))
    if probability < threshold:
        st.warning(
            "This appears outside the assistant's scope. Please ask about a disease "
            "affecting maize, beans, cassava, or potato."
        )
        st.caption(f"Domain-gate score: {probability:.3f} (threshold: {threshold:.3f})")
        st.stop()

    query = question if crop == "Any" else f"{question} (crop: {crop})"
    ids = retriever.search([query], k=3)[0]
    by_id = {item["chunk_id"]: item for item in corpus}
    passages = [by_id[item_id] for item_id in ids if item_id in by_id]
    climate, weather_note = resolve_weather(weather_mode, temp, humidity, rain, location)
    advice, risk = make_advice(question, crop, climate, passages)

    st.success("In-domain crop disease question")
    st.caption(f"Domain-gate score: {probability:.3f} | {weather_note}")
    st.markdown(advice)
    if risk:
        st.metric("Weather-based risk", risk.upper())
    st.warning(DISCLAIMER)

    with st.expander("Sources used", expanded=True):
        for index, passage in enumerate(passages, 1):
            title = passage.get("title") or passage.get("source_doc", "Source")
            st.markdown(f"**[{index}] {title}**")
            st.write(passage["text"][:900] + ("..." if len(passage["text"]) > 900 else ""))
            if passage.get("url"):
                st.markdown(f"[Open source]({passage['url']})")
