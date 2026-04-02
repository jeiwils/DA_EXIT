from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns  # type: ignore[import-not-found]

from .config import VisualizationConfig
from .plot_utils import save_figure, setup_style


def _frontier_plot(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    out_base: Path,
    cfg: VisualizationConfig,
) -> list[Path]:
    if frame.empty:
        return []

    setup_style(cfg)
    grid = sns.relplot(
        data=frame,
        x=x_col,
        y=y_col,
        hue="run_type",
        style="budget",
        col="top_k",
        marker="o",
        kind="scatter",
        facet_kws={"sharex": False, "sharey": True},
        height=4.0,
        aspect=1.2,
    )
    grid.set_titles("Top-k = {col_name}")
    grid.set_axis_labels(x_col.replace("_", " "), y_col.replace("_", " "))
    grid.figure.suptitle(title, y=1.03)
    return save_figure(grid.figure, out_base, cfg.formats, cfg.dpi)


def plot_quality_vs_latency(agg: pd.DataFrame, cfg: VisualizationConfig) -> list[Path]:
    out_base = cfg.output_root / "frontiers" / "quality_vs_latency"
    return _frontier_plot(
        agg,
        x_col="wall_time_sec_mean_mean",
        y_col=cfg.metric + "_mean",
        title="Quality vs Latency Frontier",
        out_base=out_base,
        cfg=cfg,
    )


def plot_quality_vs_tokens(agg: pd.DataFrame, cfg: VisualizationConfig) -> list[Path]:
    out_base = cfg.output_root / "frontiers" / "quality_vs_tokens"
    return _frontier_plot(
        agg,
        x_col="overall_tokens_per_query_mean_mean",
        y_col=cfg.metric + "_mean",
        title="Quality vs Token Cost Frontier",
        out_base=out_base,
        cfg=cfg,
    )


def plot_quality_vs_reader_tokens(agg: pd.DataFrame, cfg: VisualizationConfig) -> list[Path]:
    out_base = cfg.output_root / "frontiers" / "quality_vs_reader_tokens"
    return _frontier_plot(
        agg,
        x_col="reader_tokens_per_query_mean_mean",
        y_col=cfg.metric + "_mean",
        title="Quality vs Reader Token Cost Frontier",
        out_base=out_base,
        cfg=cfg,
    )
