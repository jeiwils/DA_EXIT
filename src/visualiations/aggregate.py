from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import VisualizationConfig


GROUP_COLS = [
    "dataset",
    "split",
    "retriever",
    "run_type",
    "budget",
    "top_k",
    "tau_low",
]


def filter_frame(frame: pd.DataFrame, cfg: VisualizationConfig) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy()
    if cfg.dataset is not None:
        out = out[out["dataset"] == cfg.dataset]
    if cfg.split is not None:
        out = out[out["split"] == cfg.split]
    if cfg.retriever is not None:
        out = out[out["retriever"] == cfg.retriever]
    return out


def aggregate_over_seeds(frame: pd.DataFrame, value_columns: Iterable[str]) -> pd.DataFrame:
    if frame.empty:
        return frame

    value_columns = list(value_columns)
    agg_spec: dict[str, list[str]] = {col: ["mean", "std"] for col in value_columns}
    agg_spec["seed"] = ["nunique"]

    grouped = frame.groupby(GROUP_COLS, dropna=False).agg(agg_spec)
    grouped.columns = ["_".join([c for c in col if c]) for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={"seed_nunique": "n_seeds"})

    for col in value_columns:
        std_col = f"{col}_std"
        if std_col in grouped.columns:
            grouped[std_col] = grouped[std_col].fillna(0.0)

    return grouped


def compute_tau_delta(agg: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if agg.empty:
        return agg

    pivot = agg.pivot_table(
        index=["dataset", "split", "retriever", "budget", "top_k", "tau_low"],
        columns="run_type",
        values=metric_col,
        aggfunc="first",
    ).reset_index()

    if "DA_EXIT" not in pivot.columns or "EXIT" not in pivot.columns:
        return pd.DataFrame()

    pivot["delta_metric"] = pivot["DA_EXIT"] - pivot["EXIT"]
    return pivot


def compute_iso_budget_table(agg: pd.DataFrame, metric_col: str, quantiles: Iterable[float]) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []

    for top_k in sorted(agg["top_k"].unique()):
        for tau in sorted(agg["tau_low"].unique()):
            block = agg[(agg["top_k"] == top_k) & (agg["tau_low"] == tau)]
            exit_scores = block[block["run_type"] == "EXIT"][metric_col].dropna()
            if exit_scores.empty:
                continue

            for q in quantiles:
                target = float(exit_scores.quantile(q))
                row: dict[str, float | int | str] = {
                    "top_k": int(top_k),
                    "tau_low": float(tau),
                    "target_quantile": float(q),
                    "target_metric": target,
                }

                for run_type in ["EXIT", "DA_EXIT"]:
                    subset = block[(block["run_type"] == run_type) & (block[metric_col] >= target)]
                    min_budget = float(subset["budget"].min()) if not subset.empty else float("nan")
                    row[f"budget_{run_type.lower()}"] = min_budget

                budget_exit = row.get("budget_exit")
                budget_da = row.get("budget_da_exit")
                if pd.notna(budget_exit) and pd.notna(budget_da):
                    row["budget_saving"] = float(budget_exit) - float(budget_da)
                else:
                    row["budget_saving"] = float("nan")

                rows.append(row)

    return pd.DataFrame(rows)
