"""Offline end-to-end test of the data pipeline with TOY documents and a FAKE generator.

    python tests/smoke_test.py

Checks the plumbing (cleaning, chunking, filters, climate rules, splits, leakage checks) without
a GPU or real data. It says nothing about the quality of real generated data.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_qa import check_pair, parse_pairs, run_generation  # noqa: E402
from knowledge import assess, detect  # noqa: E402

TOY = {
    "potato": ["late blight", "bacterial wilt", "early blight"],
    "maize": ["gray leaf spot", "northern corn leaf blight", "maize lethal necrosis"],
    "beans": ["bean rust", "angular leaf spot", "anthracnose"],
    "cassava": ["cassava bacterial blight", "cassava mosaic disease", "cassava brown streak disease"],
}


def toy_doc(crop, disease, k):
    s = [f"{disease.capitalize()} is an important disease of {crop} in East Africa (toy document {k}).",
         f"Farmers first notice symptoms of {disease} on the leaves of the {crop} plant after several weeks of growth.",
         f"The spread of {disease} depends on weather and on how clean the planting material is in field number {k}.",
         f"To reduce {disease}, farmers should plant healthy seed and remove infected {crop} plants early.",
         f"Regular scouting of the {crop} field helps farmers find {disease} before it spreads widely.",
         f"Extension officers advise farmers to rotate {crop} with other crops to lower {disease} pressure over time."]
    return "\n\n".join(" ".join(s[i:i + 2]) for i in range(0, len(s), 2)) * 2


def fake_generate(batch_messages):
    outs = []
    for m in batch_messages:
        passage = m[-1]["content"].split('"""')[1]
        sents = [x.strip() for x in passage.split(".") if len(x.split()) >= 9]
        pairs = [{"question": f"What do farmers need to know about {s.split()[0].lower()} in case {i}?",
                  "answer": s + ".", "type": "other"} for i, s in enumerate(sents[:3])]
        outs.append("```json\n" + json.dumps(pairs) + "\n```")
    return outs


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def build_toy_dataset(tmp):
    """Run the whole data pipeline on toy documents. Returns the processed-data directory."""
    tmp = Path(tmp)
    raw = tmp / "raw"
    raw.mkdir()
    manifest = ["filename,title,publisher,url,license,crop,year"]
    for crop, diseases in TOY.items():
        for d in diseases:
            for k in range(2):
                name = f"{crop}_{d.replace(' ', '_')}_{k}.txt"
                (raw / name).write_text(toy_doc(crop, d, k))
                manifest.append(f"{name},Toy,Toy,,toy,{crop},2026")
    (raw / "sources.csv").write_text("\n".join(manifest))
    (raw / "noise.txt").write_text("Acknowledgements. " + "We thank everyone for lunch. " * 20)

    print(run(["src/data_prep.py", "--raw_dir", str(raw), "--out", str(tmp / "chunks.jsonl")]))
    chunks = [json.loads(l) for l in open(tmp / "chunks.jsonl")]
    assert chunks and all(c["crop"] != "general" for c in chunks)
    stats = run_generation(chunks, fake_generate, tmp / "qa_raw.jsonl", pairs_per_chunk=3, batch_size=4)
    assert stats["kept"] > 0
    print(run(["src/review.py", "export", "--qa", str(tmp / "qa_raw.jsonl"), "--n", "8", "--out", str(tmp / "rev.csv")]))
    print(run(["src/review.py", "apply", "--qa", str(tmp / "qa_raw.jsonl"), "--reviewed", str(tmp / "rev.csv"),
               "--out", str(tmp / "qa_clean.jsonl")]))
    print(run(["src/build_climate.py", "--chunks", str(tmp / "chunks.jsonl"), "--out", str(tmp / "climate.jsonl")]))
    print(run(["src/build_ood.py", "--out", str(tmp / "ood.jsonl")]))
    out = run(["src/make_splits.py", "--chunks", str(tmp / "chunks.jsonl"), "--qa", str(tmp / "qa_clean.jsonl"),
               "--climate", str(tmp / "climate.jsonl"), "--ood", str(tmp / "ood.jsonl"),
               "--out_dir", str(tmp / "processed")])
    print(out[-300:])
    for sp in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(tmp / "processed" / f"{sp}.jsonl")]
        assert rows, f"{sp} is empty"
    return tmp / "processed"


def main():
    # unit checks
    assert detect("Late blight hit my potato field.")[1] == [("potato", "late blight")]
    assert assess("potato", "late blight", {"temperature_c": 16, "humidity_pct": 92, "rainfall_mm_7d": 70})["risk"] == "high"
    assert assess("potato", "late blight", {"temperature_c": 31, "humidity_pct": 40, "rainfall_mm_7d": 0})["risk"] == "low"
    assert not check_pair("Is blight bad?", "Yes it is quite bad indeed for farmers.", "x")[0]
    assert check_pair("What conditions favour late blight in potato?",
                      "Late blight is favoured by cool, humid weather with long leaf wetness.",
                      "Late blight is favoured by cool, humid weather with long leaf wetness periods.")[0]
    assert not check_pair("What conditions favour late blight in potato?",
                          "Spray 500 ml of fungicide per 20 litres of water on affected potato leaves.",
                          "Late blight is favoured by cool, humid weather.")[0]
    assert len(parse_pairs('text [{"question": "a b c d e f?", "answer": "x"}] more')) == 1

    with tempfile.TemporaryDirectory() as tmp:
        processed = build_toy_dataset(tmp)
        ex = [json.loads(l) for l in open(processed / "test.jsonl")]
        cl = [r for r in ex if r["type"] == "climate"]
        print("example climate row:\n", json.dumps(cl[0], indent=1)[:900] if cl else "no climate rows in test")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
