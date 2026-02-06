"""Path helpers for representation artifacts."""

from __future__ import annotations

import os
from typing import Dict

from DA_EXIT.preprocessing_ingestion.utils.__utils__ import data_path


def dataset_rep_paths(
    dataset: str,
    split: str,
    *,
    passage_source: str = "passages",
) -> Dict[str, str]:
    """Return paths for model-agnostic dataset-level passage representations."""

    base = data_path("representations", "datasets", dataset, split)
    suffix = ""
    if passage_source and passage_source != "passages":
        safe_source = passage_source.replace(os.sep, "_").replace("/", "_")
        base = base / safe_source
        suffix = f"_{safe_source}"
    return {
        "passages_jsonl": str(base / f"{dataset}{suffix}_passages.jsonl"),
        "passages_emb": str(base / f"{dataset}{suffix}_passages_emb.npy"),
        "passages_index": str(base / f"{dataset}{suffix}_faiss_passages.faiss"),
    }
