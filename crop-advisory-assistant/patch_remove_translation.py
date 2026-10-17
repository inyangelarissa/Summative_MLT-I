import json
from pathlib import Path


def sub(path, pairs):
    p = Path(path)
    if not p.exists():
        print("MISSING", path); return
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if old not in t:
            print(f"  (not found in {path}: {old[:50]!r})")
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8"); print("patched", path)


sub("app/app.py", [
    ('LANGS = {"English": "en", "Kinyarwanda": "rw"}', 'LANGS = {"English": "en"}'),
    (' Answers can be given in English or Kinyarwanda (machine translated).', ' Answers are given in English.'),
    ('gr.Dropdown(list(LANGS), value="English", label="Question language")', 'gr.Dropdown(list(LANGS), value="English", label="Question language", visible=False)'),
    ('gr.Dropdown(list(LANGS), value="English", label="Answer language")', 'gr.Dropdown(list(LANGS), value="English", label="Answer language", visible=False)'),
    ('"English", "Kinyarwanda", "cassava"', '"English", "English", "cassava"'),
    ('It can make mistakes, including in Kinyarwanda "\n                    "translations. Always', 'It can make mistakes. "\n                    "Always'),
])
sub("README.md", [
    (" -> optional Kinyarwanda output (NLLB-200).", "."),
    (", a native-speaker Kinyarwanda test set", ""),
])
sub("deploy/prepare_space.py", [("weather-aware, optional Kinyarwanda output)", "weather-aware)")])


def edit_nb(path, drop, texts):
    p = Path(path)
    if not p.exists():
        print("MISSING", path); return
    nb = json.loads(p.read_text(encoding="utf-8"))
    cells = []
    for i, c in enumerate(nb["cells"]):
        s = "".join(c["source"])
        if i in drop and any(k in s.lower() for k in ("kinyarwanda", "translat", "mt_", "nllb")):
            continue
        for old, new in texts:
            s = s.replace(old, new)
        c["source"] = s.splitlines(keepends=True)
        cells.append(c)
    nb["cells"] = cells
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("patched", path, f"({len(nb['cells'])} cells left)")


edit_nb("notebooks/04_pipeline_and_app.ipynb", {9, 10, 11}, [
    ("question (EN/RW) → translate → domain gate", "question → domain gate"),
    ("safety filter → translate → answer", "safety filter → answer"),
])
edit_nb("notebooks/05_evaluation_and_error_analysis.ipynb", {36, 37, 38, 39}, [
    ("B1, B2, E1-E6", "B1, B2, E1-E5"),
    ("| 13 | **Kinyarwanda** MT: NLLB vs OPUS-MT (E6), chrF/BLEU, terminology |\n", ""),
    (", and MT terminology errors", ""),
    (", tokenisation of Kinyarwanda, ...", ", ..."),
    (" the MT test set is small and single-annotator;", ""),
])
edit_nb("notebooks/02_retriever_and_gate.ipynb", set(), [
    (", and include Kinyarwanda-translated-to-English phrasing once the MT part exists", ""),
])
print("done")
