# DA_EXIT

A discourse-aware retrieval and reasoning pipeline for question answering, based on the work in [this paper](https://arxiv.org/pdf/2412.12559).

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

This repository exposes a single main entrypoint: `DA_EXIT.py`.

1. Ensure your data is prepared in:
   - `data/processed_datasets/{dataset}/{split}`
   - `data/representations/datasets/{dataset}/{split}`

2. Run the main pipeline:

```bash
python DA_EXIT.py
```

This executes the `main()` loop in `DA_EXIT.py`, which iterates over:
- `DATASETS`
- `SPLITS`
- `RETRIEVER_CONFIG`
- `SENTENCE_MODES`
- `TOP_K_CHUNK_SWEEP`
- `TAU_LOW_SWEEP`
- `SEEDS`

3. Optional: adjust the constants at the top of `DA_EXIT.py` to configure:
- Retriever mix (`RETRIEVER_CONFIG`)
- Passage/retrieval settings
- Reader model / server URL
- LoRA checkpoint path
- Budget thresholds

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
