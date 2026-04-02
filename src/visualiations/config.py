from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class VisualizationConfig:
    """Shared configuration for DA_EXIT plotting modules."""

    results_root: Path = Path("data/results")
    output_root: Path = Path("data/results/visualisations")
    metric: str = "mean_f1"
    dataset: str | None = None
    split: str | None = None
    retriever: str | None = None
    formats: Sequence[str] = field(default_factory=lambda: ("png", "pdf"))
    dpi: int = 300
    style: str = "whitegrid"
    context: str = "talk"
    palette: str = "Set2"
    target_quantiles: Sequence[float] = field(default_factory=lambda: (0.50, 0.75))


DEFAULT_CONFIG = VisualizationConfig()
