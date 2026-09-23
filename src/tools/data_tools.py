"""Thin adapters for data-oriented MCP tools."""

from src.schemas import CleaningReport, DataProfile, DatasetInfo
from src.skills import DataCleaningSkill, DataLoadingSkill, DataProfilingSkill
from src.storage import DatasetRegistry


class DataTools:
    """Compose data skills behind the public tool-shaped operations."""

    def __init__(self, registry: DatasetRegistry) -> None:
        self.loading = DataLoadingSkill(registry)
        self.profiling = DataProfilingSkill(registry)
        self.cleaning = DataCleaningSkill(registry)

    def load_data(self, file_path: str) -> DatasetInfo:
        """Load a CSV or XLSX file and register it for later operations."""

        return self.loading.run(file_path)

    def describe_data(self, dataset_id: str) -> DataProfile:
        """Inspect schema, quality metrics, and compact dataset statistics."""

        return self.profiling.run(dataset_id)

    def clean_data(
        self,
        dataset_id: str,
        handle_missing: bool = True,
        remove_duplicates: bool = True,
        detect_outliers: bool = True,
    ) -> CleaningReport:
        """Clean a registered dataset using deterministic policies."""

        return self.cleaning.run(
            dataset_id,
            handle_missing=handle_missing,
            remove_duplicates=remove_duplicates,
            detect_outliers=detect_outliers,
        )
