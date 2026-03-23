# DA_EXIT

A discourse-aware retrieval and reasoning pipeline for question answering, based on the work in [this paper](https://arxiv.org/pdf/2412.12559).

The purpose of this project is to 'compress' the amount of information sent to the reader model, providing it with more precise, specific information. A lightweight LoRa is trained to identify useful sentences. At runtime, such sentences are extracted, reranked according to how useful the LoRa considers them to be, and then sent to a more heavyduty reader model. The aim is to reduce compute by passing the reader model less sentences, while retaining accuracy.

The LoRa is trained on MuSiQue passages, and current benchmarking is only done on that. All functionality is included in the codebase to train a new LoRa on a different dataset, and to assess the performance of that. There is currently functionality for a number of datasets besides MuSiQue. MuSiQue was chosen as it's on the more complex side of QA datasets, requiring reasoning that may benefit more from discourse aware sentence extraction.


## Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the environment:
   - On Windows: `.venv\Scripts\Activate.ps1`
   - On Unix/Mac: `source .venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

   Note: For GPU support with FAISS, install `faiss-gpu` manually if needed.


## Usage

The pipeline consists of the following scripts, to be run in order:

1. **dataset_preprocessing.py**: Preprocess raw datasets from `data/raw_datasets/{dataset}/` into QA/passages JSONL files in `data/processed_datasets/{dataset}/{split}/`.
2. **build_representations.py**: Build dense/sparse passage representations from the processed data, saving to `data/representations/datasets/{dataset}/{split}/`.
3. **DA_EXIT.py**: Run the main discourse-aware retrieval and reasoning pipeline using the processed data and representations.
4. **baseline_embeddings_RAG.py**: Run a baseline Retrieval-Augmented Generation pipeline for comparison using the same data.


## LoRa training

This repository includes a full end-to-end training flow for the discourse-aware LoRa sentence usefulness classifier.

1. Preprocess the dataset (default is MuSiQue):
   ```bash
   python dataset_preprocessing.py
   ```
   - Outputs are written under `data/processed_datasets/{dataset}/{split}/`.
   - Default split names in `dataset_preprocessing.py` are `train`, `dev`, `train_sub`, `val`, etc.

2. Train the LoRa model (default uses `microsoft/deberta-v3-base`):
   ```bash
   python src/b2_reranking/useful_sentence_extractor_train.py
   ```
   - Runs a grid search as configured in `RUN_GRID_SEARCH`, `GRID_CONFIGS`, and `BEST_GRID_RUN_KWARGS`.
   - Default dataset in `useful_sentence_extractor_train.py` is `musique`, with `train_sub` and `val`.
   - Outputs include `data/models/useful_sentence_lora/grid/*` and `best_metrics.json`.

3. Check training results:
   - `data/models/useful_sentence_lora/*/best_metrics.json` returns `dev_f1`, `dev_loss`, `tau`, etc.
   - The model is saved with LoRa adapter files (`adapter_config.json`, `adapter_model.safetensors`, tokenizer files) in the run directory.

4. Run inference with the trained LoRa:
   - See `src/b2_reranking/useful_sentence_extractor_infer.py`.
   - Use `load_sentence_lora(checkpoint_dir=...)` and `score_sentences_lora(...)`.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
