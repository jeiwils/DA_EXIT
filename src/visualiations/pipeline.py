from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .aggregate import (
    aggregate_over_seeds,
    compute_iso_budget_table,
    compute_tau_delta,
    filter_frame,
)
from .config import DEFAULT_CONFIG, VisualizationConfig
from .loader import load_summary_frame
from .plots_budget_savings import plot_iso_performance_budgets
from .plots_frontier import (
    plot_quality_vs_latency,
    plot_quality_vs_reader_tokens,
    plot_quality_vs_tokens,
)
from .plots_latency import plot_latency_decomposition
from .plots_tokens import plot_total_vs_reader_tokens
from .plots_tau import plot_tau_delta, plot_tau_sensitivity


VALUE_COLUMNS = [
    "mean_f1",
    "mean_em",
    "wall_time_sec_mean",
    "reader_wall_time_sec_mean",
    "sentence_wall_time_sec_mean",
    "overall_tokens_per_query_mean",
    "reader_tokens_per_query_mean",
    "sentence_extractor_tokens_per_query_mean",
    "mean_hits_at_k_ratio",
    "mean_recall_at_k_ratio",
    "mean_precision_at_k_ratio",
]


def build_visualization_dataset(cfg: VisualizationConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    raw = load_summary_frame(cfg.results_root)
    filtered = filter_frame(raw, cfg)
    return aggregate_over_seeds(filtered, VALUE_COLUMNS)


def build_all_figures(cfg: VisualizationConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    agg = build_visualization_dataset(cfg)
    if agg.empty:
        return {
            "status": "no_data",
            "message": "No summary files found for the selected filters.",
            "artifacts": [],
        }

    metric_col = cfg.metric + "_mean"
    delta = compute_tau_delta(agg, metric_col=metric_col)
    iso = compute_iso_budget_table(agg, metric_col=metric_col, quantiles=cfg.target_quantiles)

    saved: list[str] = []
    for output in plot_quality_vs_latency(agg, cfg):
        saved.append(str(output))
    for output in plot_quality_vs_tokens(agg, cfg):
        saved.append(str(output))
    for output in plot_quality_vs_reader_tokens(agg, cfg):
        saved.append(str(output))
    for output in plot_iso_performance_budgets(iso, cfg):
        saved.append(str(output))
    for output in plot_tau_sensitivity(agg, cfg):
        saved.append(str(output))
    for output in plot_tau_delta(delta, cfg):
        saved.append(str(output))
    for output in plot_latency_decomposition(agg, cfg):
        saved.append(str(output))
    for output in plot_total_vs_reader_tokens(agg, cfg):
        saved.append(str(output))

    csv_out = cfg.output_root / "tables" / "aggregated_visualisation_data.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(csv_out, index=False)
    saved.append(str(csv_out))

    return {
        "status": "ok",
        "n_rows": int(len(agg)),
        "metric": cfg.metric,
        "artifacts": saved,
    }


def run_default_visualisations() -> dict[str, Any]:
    """Convenience entrypoint without CLI arguments."""
    return build_all_figures(DEFAULT_CONFIG)


if __name__ == "__main__":
    result = run_default_visualisations()
    print(f"status={result.get('status')}")
    print(f"artifacts={len(result.get('artifacts', []))}")
