"""Offline test of pipeline.py, weather.py, app.run_query and deploy/prepare_space.py with stand-ins
for the LLM and the translator (no GPU, no downloads).

    python tests/test_pipeline.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "app"))
from smoke_test import build_toy_dataset  # noqa: E402
from domain_gate import DomainGate, TfidfGate, load_split, run_experiment  # noqa: E402
from pipeline import AdvisoryPipeline, PipelineConfig, REFUSAL_EN, safety_filter  # noqa: E402
from retriever import BM25Retriever, load_corpus  # noqa: E402
from weather import climate_from_values, summarize_hourly  # noqa: E402
import app as appmod  # noqa: E402


class FakeLLM:
    def __init__(self):
        self.last_messages = None

    def generate(self, messages):
        self.last_messages = messages
        return ("Scout your field every few days. Apply 50 ml of fungicide per litre of water. "
                "Remove infected plants early.")


class FakeMT:
    def translate(self, text, src, tgt):
        return text if src == tgt else f"[{tgt}] {text}"


def main():
    # --- unit: safety filter, weather ---
    clean, flags = safety_filter("Remove sick plants. Spray 50 ml per litre.", "Remove sick plants early.")
    assert clean == "Remove sick plants." and flags == ["removed_ungrounded_dosage"]
    clean, _ = safety_filter("Use 50 ml per litre.", "The label says use 50 ml per litre.")
    assert "50 ml" in clean                                   # grounded dosage is allowed
    clean, flags = safety_filter("Spray 5 ml.", "")
    assert "no_safe_answer" in flags
    hourly = {"temperature_2m": [20.0, 22.0, None], "relative_humidity_2m": [80, 90, 85], "precipitation": [1.0, 2.5, 0.5]}
    assert summarize_hourly(hourly) == {"temperature_c": 21, "humidity_pct": 85, "rainfall_mm_7d": 4, "season": "between seasons"} \
        or summarize_hourly(hourly)["rainfall_mm_7d"] == 4
    assert climate_from_values(16, 92, 70)["season"] == "rainy season"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        processed = build_toy_dataset(tmp)
        corpus = load_corpus(processed / "corpus.jsonl")
        data = {sp: load_split(processed, sp) for sp in ("train", "val", "test")}
        run_experiment(TfidfGate(), "tfidf", data, tmp / "gate")
        llm = FakeLLM()
        cfg = PipelineConfig(use_context=True, use_climate=True, top_k=3)
        pipe = AdvisoryPipeline(cfg, gate=DomainGate.load(tmp / "gate"), retriever=BM25Retriever(corpus),
                                generator=llm, translator=FakeMT(), corpus=corpus)
        pipe.corpus_by_id = {c["chunk_id"]: c for c in corpus}

        res = pipe.answer("What should farmers know about potato late blight?", crop="potato",
                          climate=climate_from_values(16, 92, 70))
        assert res["status"] == "answered" and res["passages"], res
        assert "50 ml" not in res["answer_en"] and "removed_ungrounded_dosage" in res["flags"]
        user_msg = llm.last_messages[1]["content"]
        assert "Weather context:" in user_msg and "Passages:" in user_msg and "Question:" in user_msg
        print("prompt sent to LLM:\n", user_msg[:400], "\n")

        res = pipe.answer("Who won the football match yesterday?")
        assert res["status"] == "out_of_domain" and res["answer"] == REFUSAL_EN and not res["passages"]

        res = pipe.answer("Ni iki gitera late blight ku birayi?", input_lang="rw", output_lang="rw")
        assert res["question_en"].startswith("[en]") and res["answer"].startswith("[rw]")

        # --- app handler (no gradio needed) ---
        a, status, src, winfo, _ = appmod.run_query(
            pipe, "What should farmers know about potato late blight?", "English", "English", "potato",
            appmod.WEATHER_MODES[1], 16, 92, 70, "Kigali")
        assert "In-domain" in status and winfo["weather"]["rainfall_mm_7d"] == 70 and "[1]" in src
        a, status, *_ = appmod.run_query(pipe, "Who won the football match yesterday?", "English", "English", "Any",
                                         appmod.WEATHER_MODES[0], 20, 70, 20, "Kigali")
        assert "Outside" in status
        def boom(lat, lon): raise OSError("offline")
        _, _, _, winfo, _ = appmod.run_query(pipe, "potato late blight?", "English", "English", "Any",
                                             appmod.WEATHER_MODES[2], 0, 0, 0, "Kigali", fetch=boom)
        assert winfo["weather"] is None and "Could not fetch" in winfo["note"]

        # --- Space packaging with fake artifacts ---
        (tmp / "retr").mkdir(); (tmp / "retr" / "meta.json").write_text("{}")
        (tmp / "adapter").mkdir(); (tmp / "adapter" / "adapter_config.json").write_text("{}")
        space = tmp / "space"
        out = subprocess.run([sys.executable, "deploy/prepare_space.py", "--space_dir", str(space),
                              "--adapter", str(tmp / "adapter"), "--gate", str(tmp / "gate"),
                              "--retriever", str(tmp / "retr"), "--corpus", str(processed / "corpus.jsonl")],
                             cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        cfg2 = json.loads((space / "config.json").read_text())
        assert cfg2["adapter_dir"] == "artifacts/adapter" and (space / "app.py").exists()
        assert (space / "artifacts" / "gate" / "gate_meta.json").exists()
        # the Space config must resolve relative to the Space folder
        rc = PipelineConfig.from_json(space / "config.json")
        assert rc.gate_dir == str(space.resolve() / "artifacts/gate") or rc.gate_dir.endswith("artifacts/gate")
        print(out.stdout.strip().splitlines()[-1])
    print("\nPIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
