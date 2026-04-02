from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns  # type: ignore[import-not-found]

from .config import VisualizationConfig
from .plot_utils import save_figure, setup_style


def plot_iso_performance_budgets(iso_table: pd.DataFrame, cfg: VisualizationConfig) -> list[Path]:
    if iso_table.empty:
        return []

    setup_style(cfg)
    saved: list[Path] = []

    for q in sorted(iso_table["target_quantile"].unique()):
        subset = iso_table[iso_table["target_quantile"] == q].copy()
        long_df = subset.melt(
            id_vars=["top_k", "tau_low", "target_quantile", "target_metric", "budget_saving"],
            value_vars=["budget_exit", "budget_da_exit"],
            var_name="run_type",
            value_name="min_budget",
        )
        long_df["run_type"] = long_df["run_type"].map(
            {"budget_exit": "EXIT", "budget_da_exit": "DA_EXIT"}
        )
        long_df = long_df.dropna(subset=["min_budget"])
        if long_df.empty:
            continue

        grid = sns.catplot(
            data=long_df,
            x="tau_low",
            y="min_budget",
            hue="run_type",
            col="top_k",
            kind="bar",
            height=3.8,
            aspect=1.1,
            sharey=True,
        )
        grid.set_titles("Top-k = {col_name}")
        grid.set_axis_labels("Tau", "Minimum Budget for Target F1")
        grid.figure.suptitle(f"Iso-Performance Budget Requirement (EXIT quantile={q:.2f})", y=1.03)

        out_base = cfg.output_root / "savings" / f"iso_budget_quantile_{q:.2f}".replace(".", "p")
        saved.extend(save_figure(grid.figure, out_base, cfg.formats, cfg.dpi))

    out_table = cfg.output_root / "tables" / "iso_budget_savings.csv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    iso_table.to_csv(out_table, index=False)
    saved.append(out_table)
    return saved
