from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns

from .config import VisualizationConfig


def setup_style(cfg: VisualizationConfig) -> None:
    sns.set_theme(style=cfg.style, context=cfg.context, palette=cfg.palette)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_base: Path, formats: Iterable[str], dpi: int) -> list[Path]:
    ensure_dir(out_base.parent)
    saved: list[Path] = []
    for ext in formats:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved.append(out_path)
    return saved
