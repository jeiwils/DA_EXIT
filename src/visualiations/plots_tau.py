from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore[import-not-found]

from .config import VisualizationConfig
from .plot_utils import save_figure, setup_style


def plot_tau_sensitivity(agg: pd.DataFrame, cfg: VisualizationConfig) -> list[Path]:
    if agg.empty:
        return []

    setup_style(cfg)
    metric_col = cfg.metric + "_mean"
    grid = sns.relplot(
        data=agg,
        x="tau_low",
        y=metric_col,
        hue="budget",
        style="run_type",
        row="run_type",
        col="top_k",
        kind="line",
        marker="o",
        facet_kws={"sharex": True, "sharey": True},
        height=3.6,
        aspect=1.1,
    )
    grid.set_titles("{row_name} | Top-k = {col_name}")
    grid.set_axis_labels("Tau", cfg.metric.replace("mean_", "").upper())
    grid.figure.suptitle("Tau Sensitivity by Type and Top-k", y=1.03)
    out_base = cfg.output_root / "tau" / "tau_sensitivity"
    return save_figure(grid.figure, out_base, cfg.formats, cfg.dpi)


def plot_tau_delta(delta_frame: pd.DataFrame, cfg: VisualizationConfig) -> list[Path]:
    if delta_frame.empty:
        return []

    setup_style(cfg)
    grid = sns.relplot(
        data=delta_frame,
        x="tau_low",
        y="delta_metric",
        hue="budget",
        col="top_k",
        kind="line",
        marker="o",
        facet_kws={"sharex": True, "sharey": True},
        height=3.8,
        aspect=1.2,
    )

    for ax in grid.axes.flat:
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

    grid.set_titles("Top-k = {col_name}")
    grid.set_axis_labels("Tau", f"Delta {cfg.metric.replace('mean_', '').upper()} (DA_EXIT - EXIT)")
    grid.figure.suptitle("Tau Effect Delta", y=1.03)
    out_base = cfg.output_root / "tau" / "tau_delta"
    return save_figure(grid.figure, out_base, cfg.formats, cfg.dpi)
