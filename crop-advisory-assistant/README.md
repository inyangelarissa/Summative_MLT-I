# Climate-Aware Crop Disease Advisory Assistant

Domain-specific question answering for diseases of maize, beans, cassava and potato:
domain gate -> retriever (BM25 / dense / hybrid) -> LoRA-fine-tuned LLM conditioned on weather.

- Live app: [[Hugging Face Space link]]
- Demo video: [[link]]
- Report: [[link]]
- Model adapter: [[Hugging Face Hub link]]

## Problem, data, method, results
[[Fill in after running the experiments: 1 paragraph each. See data/DATA_CARD.md and the report.]]

## Repository layout
- `src/` logic (data pipeline, retriever, domain gate, LoRA training, pipeline, evaluation)
- `notebooks/` Colab runners 01-05 (data, retriever+gate, LoRA, pipeline+app, evaluation)
- `app/` Gradio app and config; `deploy/prepare_space.py` builds the Hugging Face Space folder
- `tests/` offline tests (no GPU or data needed)
- `data/` raw documents (not committed), interim and processed data, data card
- `results/` generated tables, figures, predictions, trained models

## Run order
1. `pip install -r requirements-data.txt -r requirements-train.txt -r requirements-app.txt -r requirements-eval.txt`
2. Offline check: `for t in smoke_test test_retriever_gate test_pipeline test_evaluate; do python tests/$t.py; done`
3. Put source documents in `data/raw/` and describe them in `data/raw/sources.csv` (copy `sources_template.csv`; check licences).
4. Run notebooks in order on a GPU runtime (Colab T4 is enough): 01 -> 02 -> 03 -> 02 (save best models) -> 04 -> 05.
   Notebook 01 pauses for a manual review of generated QA pairs; notebook 05 needs human ratings and manual error labels.
5. Launch the app: `python app/app.py --share` (edit `app/config.json` so the adapter path and `use_context` / `use_climate` match your best training run).
6. Deploy: `python deploy/prepare_space.py --space_dir space --adapter results/runs/<best_run>/adapter`, then upload the `space/` folder to a Hugging Face Space.

## Notes
- Weather-rule thresholds in `src/knowledge.py` are starting values: verify and cite them.
- Climate-conditioned examples are synthetic. Advice is not a substitute for an agricultural extension officer.
- Acknowledge all datasets, models and libraries used (see report references).
