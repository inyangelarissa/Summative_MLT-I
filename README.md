# Climate-Aware Crop Disease Advisory Assistant

An agricultural question-answering system for maize, beans, cassava and
potato diseases. The project combines a trained domain gate, information
retrieval, a LoRA-adapted language model, and optional weather context.

## Resources

| Resource | Link or location |
|---|---|
| Free deployment | Deploy [`streamlit_app.py`](streamlit_app.py) on Streamlit Community Cloud |
| Full local app | [`app/app.py`](app/app.py) |
| LoRA training output | [`results/runs/e3_lora_rag/`](results/runs/e3_lora_rag/) |
| Domain-gate model | [`results/gate/best/`](results/gate/best/) |
| Retrieval corpus | [`data/processed/corpus.jsonl`](data/processed/corpus.jsonl) |
| Data card | [`data/DATA_CARD.md`](data/DATA_CARD.md) |
| Report | Add the final report path when it is available |

The free deployment is the recommended public demo. It uses retrieval and
rule-based climate assessment and does not load the 1.5B-parameter language
model, so it can run on a free CPU service. The full generative pipeline is
available locally or on a machine with sufficient memory.

## 1. Problem

Smallholder farmers need accessible information about crop diseases, but
general-purpose chatbots can answer outside their area of expertise, produce
unsupported treatment quantities, or fail to show the evidence behind an
answer. The goal of this project is to build a domain-specific assistant for
maize, beans, cassava and potato disease questions.

The system is designed to:

- reject unrelated questions before expensive model generation;
- retrieve relevant agricultural passages from a curated corpus;
- condition advice on recent weather when climate information is available;
- show the source passages used for the answer;
- avoid presenting ungrounded dosage-like quantities;


The assistant is an educational prototype. It does not replace diagnosis,
agronomic advice, or guidance from a local agricultural extension officer.

## 2. Dataset

The source collection contains agricultural research papers, extension guides,
and crop-disease material relevant to Rwanda and East Africa. The supported
crops are maize, beans, cassava and potato. Source titles, URLs, crop labels,
licensing notes, and intended use are documented in
[`data/DATA_CARD.md`](data/DATA_CARD.md).

The processed retrieval corpus contains **1,515 passages** from **35
documents**. The supervised question-answer data is split into:

| Split | Rows | QA | Climate | Out-of-domain |
|---|---:|---:|---:|---:|
| Train | 1,113 | 950 | 72 | 91 |
| Validation | 269 | 248 | 9 | 12 |
| Test | 312 | 264 | 36 | 12 |

The generated climate examples are synthetic. They are used to test or train
weather-sensitive behavior, not to claim that the rule thresholds are
validated epidemiological models. Climate thresholds in
[`src/knowledge.py`](src/knowledge.py) must be checked against current
agronomic literature before operational use.

Raw source documents are not required by the deployed Streamlit application,
but they are required to reproduce the complete data-preparation pipeline.

## 3. Methods

### Domain gate

The domain gate classifies a question as in-domain or out-of-domain. The
selected gate uses sentence-transformer embeddings from
`sentence-transformers/all-MiniLM-L6-v2` followed by logistic regression. Its
threshold was tuned to reduce false acceptance of unrelated questions.

### Retrieval

Three retriever implementations are available:

- BM25 lexical retrieval;
- dense retrieval using sentence-transformer embeddings and FAISS or NumPy;
- hybrid retrieval using reciprocal-rank fusion.

The full pipeline can select `bm25`, `dense`, or `hybrid` through
[`app/config.json`](app/config.json). The free Streamlit deployment uses BM25
to keep memory and startup time suitable for free CPU hosting.

### Language model and LoRA

The full local pipeline uses `Qwen/Qwen2.5-1.5B-Instruct` with a parameter-
efficient LoRA adapter. The checked-in `e3_lora_rag` run used:

- LoRA rank: 16;
- LoRA alpha: 32;
- LoRA dropout: 0.05;
- learning rate: `2e-4`;
- three training epochs;
- maximum sequence length: 1,024;
- retrieved context enabled;
- climate conditioning disabled for this run;
- seed: 42.

The run trained 18.46 million parameters out of approximately 1.56 billion
total parameters.

### Climate context

Weather can be entered manually or fetched from the free Open-Meteo API for
selected Rwanda locations. The system uses temperature, relative humidity,
rainfall during the previous seven days, and a coarse season label.

For diseases with a configured rule, the application reports low, moderate,
or high weather-based risk. Diseases without a configured rule are not given a
weather risk score.

### Safety handling

The full generated answer passes through a safety filter that removes
dosage-like sentences when their quantities are not supported by the question,
weather block, or retrieved passages. Neither deployment mode is intended to
recommend pesticide dosages.

## 4. Results

### Domain-gate evaluation

The selected gate is stored in [`results/gate/best/`](results/gate/best/).
The test report contains the following results:

| Threshold | Accuracy | In-domain recall | OOD recall | Macro-F1 |
|---|---:|---:|---:|---:|
| 0.5 | 0.9966 | 0.9964 | 1.0000 | 0.9791 |
| Tuned: 0.2031 | 0.9897 | 0.9964 | 0.8333 | 0.9321 |

The tuned threshold preserves in-domain coverage but allows more unrelated
questions through on the test set. The threshold is therefore a safety and
coverage trade-off, not a universal optimum.

### BM25 retrieval

On the processed test split, the BM25 baseline achieved:

| Metric | Score |
|---|---:|
| Recall@1 | 0.267 |
| Recall@3 | 0.400 |
| Recall@5 | 0.480 |
| MRR@10 | 0.358 |

Several passages can be valid evidence for the same question, so exact
gold-passage recall is a lower bound on useful retrieval.

### LoRA run

The metrics for the selected LoRA run are stored in
[`results/runs/e3_lora_rag/metrics.json`](results/runs/e3_lora_rag/metrics.json):

| Metric | Value |
|---|---:|
| Training time | 46.11 minutes |
| Evaluation loss | 1.4730 |
| Evaluation perplexity | 4.3622 |
| Trainable parameters | 18,464,768 |

These are training-run metrics, not a replacement for human evaluation of
agricultural correctness, factuality, or field usefulness.

## 5. Repository structure

```text
crop-advisory-assistant/
├── streamlit_app.py              # Free retrieval and climate-rule app
├── app/
│   ├── app.py                    # Full local Gradio app
│   ├── app_backup.py             # Backup UI implementation
│   └── config.json               # Full pipeline configuration
├── src/
│   ├── pipeline.py               # End-to-end generation pipeline
│   ├── domain_gate.py            # Domain classifier
│   ├── retriever.py              # BM25, dense, and hybrid retrieval
│   ├── bm25.py                   # BM25 implementation
│   ├── knowledge.py              # Crop, disease, and climate rules
│   ├── weather.py                # Manual and Open-Meteo weather utilities
│   ├── prompting.py              # Model prompt construction
│   ├── train_lora.py             # LoRA fine-tuning
│   ├── data_prep.py              # Document preparation
│   ├── build_qa.py               # QA data generation
│   ├── build_climate.py          # Synthetic climate examples
│   └── evaluate.py               # Evaluation and error analysis
├── data/
│   ├── DATA_CARD.md              # Dataset documentation
│   └── processed/corpus.jsonl    # Runtime retrieval corpus
├── results/
│   ├── gate/best/                # Runtime domain-gate files
│   ├── retriever/best_dense/     # Saved dense retriever
│   └── runs/                     # LoRA adapters and metrics
├── notebooks/                    # Data, retrieval, training, and evaluation
├── tests/                        # Offline tests
├── requirements-*.txt            # Stage-specific dependencies
└── README.md
```

## 6. How to reproduce

### Install the complete project

```bash
pip install -r requirements-data.txt
pip install -r requirements-train.txt
pip install -r requirements-app.txt
pip install -r requirements-eval.txt
```

For the free Streamlit app only:

```bash
pip install -r requirements.txt
```

### Prepare data and models

1. Place permitted source documents in `data/raw/` and document them in the
   source manifest.
2. Run `01_data_pipeline.ipynb` to extract, clean, chunk, and review data.
3. Run `02_retriever_and_gate.ipynb` to build the corpus, retrievers, and
   domain-gate model.
4. Run `03_lora_finetune.ipynb` on a GPU runtime to train the adapter.
5. Run `04_pipeline_and_app.ipynb` to check the full application pipeline.
6. Run `05_evaluation_and_error_analysis.ipynb` for evaluation and error
   analysis.

Some stages require a GPU, manual QA review, human ratings, or manual error
labels. Running training notebooks in quick mode can overwrite result files
with numbers that do not match the reported run.

### Run the free app locally

```bash
streamlit run streamlit_app.py
```

The free app requires:

```text
streamlit_app.py
src/bm25.py
src/domain_gate.py
src/knowledge.py
src/retriever.py
src/weather.py
data/processed/corpus.jsonl
results/gate/best/gate_meta.json
results/gate/best/model.joblib
```

### Deploy for free

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from the repository and branch.
4. Set the main file to:

   ```text
   crop-advisory-assistant/streamlit_app.py
   ```

5. Deploy with the root `requirements.txt`.

The GitHub copy must include the processed corpus and the domain-gate files
listed above. The free deployment does not load the Qwen model, LoRA adapter,
dense index, or translation model.

### Run the full local app

Edit [`app/config.json`](app/config.json) so that `adapter_dir`,
`use_context`, and `use_climate` match the selected training run, then run:

```bash
python app/app.py
```

For a temporary Gradio share link:

```bash
python app/app.py --share
```

The full app requires the dependencies in `requirements-app.txt` and enough
memory to load the base language model and adapter.

### Run tests

The tests are executable Python scripts:

```bash
python tests/smoke_test.py
python tests/test_retriever_gate.py
python tests/test_pipeline.py
python tests/test_evaluate.py
```

## 7. Limitations

- The corpus is limited to the documents listed in the data card.
- Source coverage is uneven across crops and diseases.
- The climate rules are approximate starting rules and require literature
  verification.
- Weather values are regional summaries rather than field measurements.
- Open-Meteo live weather requires internet access.
- The domain gate can reject valid questions or accept unrelated questions.
- Retrieval recall is incomplete, especially when wording differs from the
  source documents.
- The free Streamlit app is retrieval- and rule-based; it is not equivalent to
  the full LoRA-generated pipeline.
- The Qwen model and LoRA adapter require substantially more memory than a
  free CPU Streamlit deployment normally provides.
- The system does not validate pesticide products, dosages, resistance, or
  local regulatory requirements.
- Advice should be confirmed with a local agricultural extension officer.

## 8. Existing resources and own work

Existing resources include the agricultural papers and extension sources in the
data card, `Qwen/Qwen2.5-1.5B-Instruct`, the
`sentence-transformers/all-MiniLM-L6-v2` embedding model, and the open-source
libraries used for retrieval, modeling, evaluation, and deployment.

Implemented in this project:

- source-document preparation and corpus construction;
- synthetic QA, out-of-domain, and climate-conditioned examples;
- BM25, dense, and hybrid retrieval;
- the embedding-based domain gate and threshold evaluation;
- LoRA fine-tuning and inference integration;
- weather-aware prompting and rule-based risk assessment;
- generated-answer safety filtering;
- offline tests and evaluation utilities;
- the full local Gradio application;
- the free Streamlit deployment with source evidence.

## 9. References

- Hugging Face Transformers documentation:
  [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- Hugging Face PEFT documentation:
  [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft)
- Qwen2.5 model card:
  [https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Sentence Transformers documentation:
  [https://www.sbert.net/](https://www.sbert.net/)
- Streamlit documentation:
  [https://docs.streamlit.io/](https://docs.streamlit.io/)
- Open-Meteo documentation:
  [https://open-meteo.com/en/docs](https://open-meteo.com/en/docs)

The complete source list and licensing notes are in
[`data/DATA_CARD.md`](data/DATA_CARD.md). All source documents, pretrained
models, and software libraries should be acknowledged according to their
respective licences and citation requirements.
