from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns  # type: ignore[import-not-found]

from .config import VisualizationConfig
from .plot_utils import save_figure, setup_style


def plot_total_vs_reader_tokens(agg: pd.DataFrame, cfg: VisualizationConfig) -> list[Path]:
    if agg.empty:
        return []

    long_df = agg.melt(
        id_vars=["run_type", "budget", "top_k", "tau_low"],
        value_vars=[
            "overall_tokens_per_query_mean_mean",
            "reader_tokens_per_query_mean_mean",
        ],
        var_name="token_type",
        value_name="tokens_per_query",
    )
    long_df["token_type"] = long_df["token_type"].map(
        {
            "overall_tokens_per_query_mean_mean": "total_tokens",
            "reader_tokens_per_query_mean_mean": "reader_tokens",
        }
    )

    setup_style(cfg)
    grid = sns.relplot(
        data=long_df,
        x="budget",
        y="tokens_per_query",
        hue="run_type",
        style="tau_low",
        col="token_type",
        row="top_k",
        kind="line",
        marker="o",
        height=2.9,
        aspect=1.35,
        facet_kws={"sharex": True, "sharey": False},
    )
    grid.set_titles("Top-k = {row_name} | {col_name}")
    grid.set_axis_labels("Reader Context Budget", "Tokens / query")
    grid.figure.suptitle("Total vs Reader Tokens (EXIT vs DA_EXIT)", y=1.01)

    out_base = cfg.output_root / "tokens" / "total_vs_reader_tokens"
    return save_figure(grid.figure, out_base, cfg.formats, cfg.dpi)
