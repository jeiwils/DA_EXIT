from .config import DEFAULT_CONFIG, VisualizationConfig

__all__ = [
    "VisualizationConfig",
    "DEFAULT_CONFIG",
    "build_visualization_dataset",
    "build_all_figures",
    "run_default_visualisations",
]


def __getattr__(name: str):
    # Lazy import prevents runpy warnings when executing pipeline as a module.
    if name in {
        "build_visualization_dataset",
        "build_all_figures",
        "run_default_visualisations",
    }:
        from .pipeline import (
            build_all_figures,
            build_visualization_dataset,
            run_default_visualisations,
        )

        return {
            "build_visualization_dataset": build_visualization_dataset,
            "build_all_figures": build_all_figures,
            "run_default_visualisations": run_default_visualisations,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
