# DA_EXIT

Standalone DA_EXIT pipeline extracted from the main RAG_lab repo.

This is a development of the work in:
```text
https://arxiv.org/pdf/2412.12559
```

Key idea: discourse-aware sentence expansion is applied after sentence extraction.

## Structure
- `DA_EXIT.py`: main DA_EXIT orchestrator
- `dataset_preprocessing.py`: build processed QA/passages JSONL
- `build_representations.py`: embed passages + build FAISS index
- `baseline_embeddings_RAG.py`: baseline RAG (dense/sparse/hybrid)

Core modules live under:
- `preprocessing_ingestion/`
- `representations_indexing/`
- `sentence_extraction_reranking/`
- `metrics/`
- `LoRa/`

## Data layout
All paths resolve relative to `DA_EXIT/data`.
- Raw datasets: `DA_EXIT/data/raw_datasets/...`
- Processed datasets: `DA_EXIT/data/processed_datasets/...`
- Representations: `DA_EXIT/data/representations/...`
- Models/checkpoints: `DA_EXIT/data/models/...`
- Results: `DA_EXIT/data/results/...`

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Note: `faiss-cpu` is included by default. If you need GPU FAISS, install the GPU variant manually.

## Run
```bash
# Preprocess datasets
python -m DA_EXIT preprocess

# Build representations (embeddings + FAISS)
python -m DA_EXIT build_representations

# Run DA_EXIT
python -m DA_EXIT da_exit

# Baseline RAG
python -m DA_EXIT baseline_rag
```

## Notes
- Local LLM inference uses `SERVER_URL` in the orchestrators (defaults to `http://localhost:8005`).
- Sentence extraction LoRA expects checkpoints under `DA_EXIT/data/models/...`.
