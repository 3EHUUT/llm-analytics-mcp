"""Thin adapters for chart and correlation visualization operations."""

from pathlib import Path

from src.schemas import ChartResult, CorrelationResult
from src.skills import VisualizationSkill
from src.storage import DatasetRegistry


class VisualizationTools:
    """Expose visualization operations over an explicitly shared registry."""

    def __init__(
        self,
        registry: DatasetRegistry,
        output_dir: str | Path | None = None,
    ) -> None:
        self.visualization = VisualizationSkill(registry, output_dir)

    def plot_trend(
        self,
        dataset_id: str,
        x_column: str,
        y_column: str,
        aggregation: str = "sum",
        frequency: str | None = None,
    ) -> ChartResult:
        """Create a time-series trend chart with computed summary values."""

        return self.visualization.plot_trend(
            dataset_id,
            x_column,
            y_column,
            aggregation=aggregation,
            frequency=frequency,
        )

    def plot_distribution(
        self,
        dataset_id: str,
        column: str,
        bins: int = 20,
    ) -> ChartResult:
        """Create a numeric histogram with exact summary statistics."""

        return self.visualization.plot_distribution(dataset_id, column, bins)

    def plot_bar(
        self,
        dataset_id: str,
        category_column: str,
        value_column: str,
        aggregation: str = "sum",
        top_n: int = 10,
    ) -> ChartResult:
        """Create a top-category bar chart from shared aggregation logic."""

        return self.visualization.plot_bar(
            dataset_id,
            category_column,
            value_column,
            aggregation=aggregation,
            top_n=top_n,
        )

    def correlation_analysis(self, dataset_id: str) -> CorrelationResult:
        """Compute correlation statistics and render their heatmap."""

        return self.visualization.correlation_analysis(dataset_id)
