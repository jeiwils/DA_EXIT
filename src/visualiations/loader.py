from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

SUMMARY_GLOB = "summary_metrics_*_*.json"
_BUDGET_RE = re.compile(r"_b(?P<budget>\d+)")
_TOPK_RE = re.compile(r"_k(?P<top_k>\d+)")
_TAU_RE = re.compile(r"_tau(?P<tau>\d+(?:p\d+)?)")
_SEED_RE = re.compile(r"_seed(?P<seed>\d+)")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_variant_field(variant: str, pattern: re.Pattern[str], field: str) -> str | None:
    match = pattern.search(variant)
    return match.group(field) if match else None


def _parse_budget(variant: str) -> int | None:
    raw = _parse_variant_field(variant, _BUDGET_RE, "budget")
    return int(raw) if raw is not None else None


def _parse_top_k(variant: str) -> int | None:
    raw = _parse_variant_field(variant, _TOPK_RE, "top_k")
    return int(raw) if raw is not None else None


def _parse_tau(variant: str) -> float | None:
    raw = _parse_variant_field(variant, _TAU_RE, "tau")
    if raw is None:
        return None
    return float(raw.replace("p", "."))


def _parse_seed(variant: str) -> int | None:
    raw = _parse_variant_field(variant, _SEED_RE, "seed")
    return int(raw) if raw is not None else None


def _mode_to_type(sentence_mode: str, variant: str) -> str:
    if sentence_mode == "standard_sentences" or variant.startswith("exit_"):
        return "EXIT"
    if sentence_mode == "discourse_aware_sentences" or variant.startswith("da_exit_"):
        return "DA_EXIT"
    return sentence_mode or "unknown"


def discover_summary_files(results_root: Path) -> list[Path]:
    files = []
    for path in results_root.rglob(SUMMARY_GLOB):
        if "visualisations" in path.parts:
            continue
        files.append(path)
    return sorted(files)


def _build_record(summary: dict[str, Any], source_path: Path) -> dict[str, Any]:
    meta = summary.get("meta", {})
    accuracy = summary.get("accuracy", {})
    latency = summary.get("latency", {})
    cost = summary.get("cost", {})
    retrieval = summary.get("retrieval", {})

    variant = str(meta.get("variant", ""))
    sentence_mode = str(meta.get("sentence_mode", ""))

    top_k_meta = meta.get("top_k")
    tau_meta = meta.get("tau_low")
    seed_meta = meta.get("seed")

    top_k = _safe_int(top_k_meta, default=-1) if top_k_meta is not None else (_parse_top_k(variant) or -1)
    tau_low = _safe_float(tau_meta, default=-1.0) if tau_meta is not None else (_parse_tau(variant) or -1.0)
    seed = _safe_int(seed_meta, default=-1) if seed_meta is not None else (_parse_seed(variant) or -1)

    wall_mean = _safe_float(latency.get("wall_time_sec_mean"))
    reader_wall_mean = _safe_float(latency.get("reader_wall_time_sec_mean"))
    sentence_wall_mean = max(wall_mean - reader_wall_mean, 0.0)

    record = {
        "source_path": str(source_path),
        "dataset": str(meta.get("dataset", "")),
        "split": str(meta.get("split", "")),
        "variant": variant,
        "retriever": str(meta.get("retriever", "")),
        "sentence_mode": sentence_mode,
        "run_type": _mode_to_type(sentence_mode, variant),
        "tau_low": tau_low,
        "top_k": top_k,
        "budget": _parse_budget(variant),
        "seed": seed,
        "mean_f1": _safe_float(accuracy.get("mean_f1")),
        "mean_em": _safe_float(accuracy.get("mean_em")),
        "wall_time_sec_mean": wall_mean,
        "reader_wall_time_sec_mean": reader_wall_mean,
        "sentence_wall_time_sec_mean": sentence_wall_mean,
        "overall_tokens_per_query_mean": _safe_float(cost.get("overall_tokens_per_query_mean")),
        "reader_tokens_per_query_mean": _safe_float(cost.get("reader_tokens_per_query_mean")),
        "sentence_extractor_tokens_per_query_mean": _safe_float(cost.get("sentence_extractor_tokens_per_query_mean")),
        "mean_hits_at_k_ratio": _safe_float(retrieval.get("mean_hits_at_k_ratio")),
        "mean_recall_at_k_ratio": _safe_float(retrieval.get("mean_recall_at_k_ratio")),
        "mean_precision_at_k_ratio": _safe_float(retrieval.get("mean_precision_at_k_ratio")),
    }
    return record


def load_summary_frame(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in discover_summary_files(results_root):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(_build_record(summary, summary_path))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = frame.dropna(subset=["budget", "top_k", "tau_low"]).copy()
    frame["budget"] = frame["budget"].astype(int)
    frame["top_k"] = frame["top_k"].astype(int)
    frame["seed"] = frame["seed"].astype(int)
    frame["tau_low"] = frame["tau_low"].astype(float)
    return frame
