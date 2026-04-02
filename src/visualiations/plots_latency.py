from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns  # type: ignore[import-not-found]

from .config import VisualizationConfig
from .plot_utils import save_figure, setup_style


def plot_latency_decomposition(agg: pd.DataFrame, cfg: VisualizationConfig) -> list[Path]:
    if agg.empty:
        return []

    plot_frame = agg.copy()
    long_df = plot_frame.assign(
        stage="sentence_stage",
        latency_sec=plot_frame["sentence_wall_time_sec_mean_mean"],
    )[["run_type", "budget", "top_k", "tau_low", "stage", "latency_sec"]]

    setup_style(cfg)
    grid = sns.relplot(
        data=long_df,
        x="tau_low",
        y="latency_sec",
        hue="run_type",
        style="budget",
        col="stage",
        row="top_k",
        kind="line",
        marker="o",
        height=2.8,
        aspect=1.3,
        facet_kws={"sharex": True, "sharey": False},
    )
    grid.set_titles("Top-k = {row_name} | {col_name}")
    grid.set_axis_labels("Tau", "Latency (sec/query)")
    grid.figure.suptitle("Sentence-Stage Latency vs Tau", y=1.01)
    out_base = cfg.output_root / "latency" / "latency_decomposition"
    return save_figure(grid.figure, out_base, cfg.formats, cfg.dpi)
