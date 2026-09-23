"""Thin adapters for deterministic analysis operations."""

from src.schemas import AggregationResult, CorrelationStats
from src.skills import AnalysisSkill
from src.storage import DatasetRegistry


class AnalysisTools:
    """Expose analysis operations over an explicitly shared registry."""

    def __init__(self, registry: DatasetRegistry) -> None:
        self.analysis = AnalysisSkill(registry)

    def aggregate_data(
        self,
        dataset_id: str,
        group_by: list[str],
        metric: str,
        aggregation: str = "sum",
        sort_desc: bool = True,
        limit: int = 20,
    ) -> AggregationResult:
        """Aggregate a metric by categorical or datetime dimensions."""

        return self.analysis.aggregate(
            dataset_id,
            group_by,
            metric,
            aggregation=aggregation,
            sort_desc=sort_desc,
            limit=limit,
        )

    def correlation_stats(self, dataset_id: str) -> CorrelationStats:
        """Compute Pearson statistics without creating a chart."""

        return self.analysis.correlations(dataset_id)
