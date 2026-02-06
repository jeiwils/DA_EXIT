"""DA_EXIT package entrypoint.

Usage:
  python -m DA_EXIT [pipeline]

Pipelines:
  da_exit                Run the DA_EXIT orchestrator
  preprocess             Run dataset_preprocessing
  build_representations  Run build_representations
  baseline_rag           Run baseline_embeddings_RAG
"""

from __future__ import annotations

import sys

from DA_EXIT import DA_EXIT as da_exit_module
from DA_EXIT import dataset_preprocessing as preprocess_module
from DA_EXIT import build_representations as build_repr_module
from DA_EXIT import baseline_embeddings_RAG as baseline_module


def _usage() -> None:
    print(
        "Usage: python -m DA_EXIT [da_exit|preprocess|build_representations|baseline_rag]"
    )


def main() -> None:
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    cmd = sys.argv[1].strip().lower()
    if cmd in {"da_exit", "exit"}:
        da_exit_module.main()
        return
    if cmd in {"preprocess", "dataset_preprocessing"}:
        preprocess_module.main()
        return
    if cmd in {"build_representations", "build_repr", "repr"}:
        build_repr_module.main()
        return
    if cmd in {"baseline_rag", "baseline", "rag"}:
        baseline_module.main()
        return

    _usage()
    sys.exit(1)


if __name__ == "__main__":
    main()
