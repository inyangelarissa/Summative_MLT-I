"""Gradio web app for the Climate-Aware Crop Disease Advisory Assistant.

Run locally / in Colab:   python app/app.py            (add --share for a public link)
On Hugging Face Spaces:   deploy/prepare_space.py builds a folder whose app.py is this file.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parent          # Space layout vs repo layout
sys.path.insert(0, str(ROOT / "src"))

from pipeline import CROPS, AdvisoryPipeline, PipelineConfig  # noqa: E402
from weather import RWANDA_LOCATIONS, climate_from_values, fetch_week_summary  # noqa: E402

LANGS = {"English": "en"}
WEATHER_MODES = ["No weather", "Enter manually", "Live weather (Open-Meteo)"]
STATUS = {"answered": "✅ In-domain question", "out_of_domain": "⛔ Outside my scope (not a maize/bean/cassava/potato disease question)",
          "empty": "Please type a question."}
INTRO = ("# 🌱 Climate-Aware Crop Disease Advisory Assistant\n"
         "Ask about diseases of **maize, beans, cassava and potato**. Optionally add recent weather so the advice "
         "reflects disease risk. Answers are given in English.")


def resolve_climate(mode, temp, hum, rain, location, fetch=fetch_week_summary):
    """Returns (climate_dict_or_None, note)."""
    if mode == WEATHER_MODES[1]:
        return climate_from_values(temp, hum, rain), "Using the weather you entered."
    if mode == WEATHER_MODES[2]:
        try:
            lat, lon = RWANDA_LOCATIONS[location]
            clim = fetch(lat, lon)
            if clim:
                return clim, f"Live weather for {location} (last 7 days, regional summary)."
            return None, "Live weather returned no data; answering without weather."
        except Exception as e:  # network down, API change, ...
            return None, f"Could not fetch live weather ({type(e).__name__}); answering without weather."
    return None, "No weather used."


def run_query(pipe, question, in_lang, out_lang, crop, mode, temp, hum, rain, location, fetch=fetch_week_summary):
    """UI-independent handler: returns (answer_md, status, sources_md, weather_info, trace_dict)."""
    crop = None if crop in (None, "", "Any") else crop
    climate, wnote = resolve_climate(mode, temp, hum, rain, location, fetch)
    res = pipe.answer(question, LANGS.get(in_lang, "en"), LANGS.get(out_lang, "en"), crop, climate)
    status = STATUS[res["status"]]
    if res.get("gate"):
        status += f"  (gate score {res['gate']['score']})"
    answer_md = res["answer"]
    if res["status"] == "answered":
        answer_md += f"\n\n---\n*{res['disclaimer']}*"
        if res["flags"]:
            answer_md += "\n\n⚠️ A statement with an unverified quantity was removed for safety."
    sources = ""
    for i, p in enumerate(res["passages"], 1):
        label = p["title"] or p["source_doc"]
        link = f" ([link]({p['url']}))" if p["url"] else ""
        sources += f"**[{i}] {label}**{link}\n\n> {p['text'][:450]}{'…' if len(p['text']) > 450 else ''}\n\n"
    winfo = {"note": wnote, "weather": climate}
    return answer_md, status, sources or "_No passages were used._", winfo, res


def build_demo(pipe):
    import gradio as gr

    with gr.Blocks(title="Crop Disease Advisory Assistant") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(label="Your question", lines=3,
                                      placeholder="e.g. Should I be worried about late blight in my potato field this week?")
                with gr.Row():
                    in_lang = gr.Dropdown(list(LANGS), value="English", label="Question language", visible=False)
                    out_lang = gr.Dropdown(list(LANGS), value="English", label="Answer language", visible=False)
                    crop = gr.Dropdown(["Any"] + CROPS, value="Any", label="Crop (optional)")
                mode = gr.Radio(WEATHER_MODES, value=WEATHER_MODES[0], label="Weather context")
                with gr.Group(visible=False) as manual_box:
                    temp = gr.Slider(5, 38, value=20, step=1, label="Average temperature (°C)")
                    hum = gr.Slider(20, 100, value=70, step=1, label="Average relative humidity (%)")
                    rain = gr.Slider(0, 200, value=20, step=1, label="Rainfall in the last 7 days (mm)")
                with gr.Group(visible=False) as live_box:
                    location = gr.Dropdown(list(RWANDA_LOCATIONS), value="Kigali", label="Location")
                submit = gr.Button("Get advice", variant="primary")
            with gr.Column(scale=4):
                status = gr.Markdown()
                answer = gr.Markdown(label="Answer")
                with gr.Accordion("Sources used", open=False):
                    sources = gr.Markdown()
                with gr.Accordion("Weather and technical details", open=False):
                    winfo = gr.JSON(label="Weather used")
                    trace = gr.JSON(label="Pipeline trace (English answer, gate, timings)")

        def toggle(m):
            return gr.update(visible=m == WEATHER_MODES[1]), gr.update(visible=m == WEATHER_MODES[2])
        mode.change(toggle, mode, [manual_box, live_box])

        def respond(*args):
            a, s, src, w, res = run_query(pipe, *args)
            slim = {k: res.get(k) for k in ("status", "question_en", "gate", "answer_en", "flags", "timings")}
            return s, a, src, w, slim

        inputs = [question, in_lang, out_lang, crop, mode, temp, hum, rain, location]
        outputs = [status, answer, sources, winfo, trace]
        submit.click(respond, inputs, outputs)
        question.submit(respond, inputs, outputs)

        gr.Examples(examples=[
            ["Should I be worried about late blight in my potato field this week?", "English", "English", "potato", WEATHER_MODES[1], 16, 92, 70, "Kigali"],
            ["Should I be worried about late blight in my potato field this week?", "English", "English", "potato", WEATHER_MODES[1], 30, 40, 0, "Kigali"],
            ["What causes rust on bean leaves and how can I prevent it?", "English", "English", "beans", WEATHER_MODES[0], 20, 70, 20, "Kigali"],
            ["How does cassava mosaic disease spread?", "English", "English", "cassava", WEATHER_MODES[0], 20, 70, 20, "Kigali"],
            ["Who won the football match yesterday?", "English", "English", "Any", WEATHER_MODES[0], 20, 70, 20, "Kigali"],
        ], inputs=inputs, label="Try an example (the first two differ only in the weather)")
        gr.Markdown("⚠️ **Prototype for a university project.** It can make mistakes. "
                    "Always confirm with a local agricultural extension officer before treating crops.")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("CROP_ADVISORY_CONFIG", str(HERE / "config.json")))
    ap.add_argument("--share", action="store_true", help="Create a public gradio.live link (handy in Colab)")
    args = ap.parse_args()
    cfg = PipelineConfig.from_json(args.config)
    pipe = AdvisoryPipeline(cfg)
    _ = pipe.gate, pipe.retriever            # load cheap components at start-up; LLM/MT load on first use
    build_demo(pipe).queue().launch(share=args.share)


if __name__ == "__main__":
    main()
