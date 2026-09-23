"""Create compact deterministic profiles of registered datasets."""

import pandas as pd

from src.exceptions import EmptyDatasetError
from src.schemas import (
    ColumnInfo,
    DataProfile,
    NumericSummary,
    SemanticType,
    ValueCount,
)
from src.skills.base import BaseSkill, infer_semantic_type, to_json_scalar


class DataProfilingSkill(BaseSkill):
    """Compute schema, quality metrics, and compact column summaries."""

    max_top_values = 5

    def run(self, dataset_id: str) -> DataProfile:
        """Profile a registered dataset without exposing its DataFrame."""

        dataframe = self.registry.get(dataset_id)
        if dataframe.empty:
            raise EmptyDatasetError(dataset_id)

        return DataProfile(
            dataset_id=dataset_id,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            columns=[self._profile_column(dataframe[column]) for column in dataframe],
            duplicate_row_count=int(dataframe.duplicated().sum()),
        )

    def _profile_column(self, series: pd.Series) -> ColumnInfo:
        semantic_type = infer_semantic_type(series)
        row_count = len(series)
        null_count = int(series.isna().sum())
        column = ColumnInfo(
            name=str(series.name),
            dtype=str(series.dtype),
            semantic_type=semantic_type,
            null_count=null_count,
            missing_percentage=round((null_count / row_count) * 100, 2),
            unique_count=int(series.nunique(dropna=True)),
        )

        if semantic_type is SemanticType.NUMERIC:
            column.numeric_summary = self._numeric_summary(series)
        elif semantic_type in {SemanticType.CATEGORICAL, SemanticType.BOOLEAN}:
            column.top_values = self._top_values(series)
        elif semantic_type is SemanticType.DATETIME:
            column.min_date, column.max_date = self._datetime_range(series)

        return column

    @staticmethod
    def _numeric_summary(series: pd.Series) -> NumericSummary:
        numeric = series.dropna()
        if numeric.empty:
            return NumericSummary()
        return NumericSummary(
            min=to_json_scalar(numeric.min()),
            max=to_json_scalar(numeric.max()),
            mean=to_json_scalar(numeric.mean()),
            median=to_json_scalar(numeric.median()),
            std=to_json_scalar(numeric.std()),
        )

    def _top_values(self, series: pd.Series) -> list[ValueCount]:
        counts = series.value_counts(dropna=True).head(self.max_top_values)
        return [
            ValueCount(value=to_json_scalar(value), count=int(count))
            for value, count in counts.items()
        ]

    @staticmethod
    def _datetime_range(series: pd.Series) -> tuple[str | None, str | None]:
        values = series.dropna()
        if values.empty:
            return None, None
        minimum = to_json_scalar(values.min())
        maximum = to_json_scalar(values.max())
        return (
            minimum if isinstance(minimum, str) else None,
            maximum if isinstance(maximum, str) else None,
        )
