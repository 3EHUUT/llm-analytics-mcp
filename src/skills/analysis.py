"""Deterministic aggregation and statistical analysis."""

import pandas as pd

from src.exceptions import (
    AnalysisError,
    ColumnNotFoundError,
    EmptyDatasetError,
    InvalidColumnTypeError,
)
from src.schemas import (
    AggregationResult,
    CorrelationPair,
    CorrelationStats,
)
from src.skills.base import BaseSkill, to_json_scalar


class AnalysisSkill(BaseSkill):
    """Perform exact aggregations and numeric correlation calculations."""

    supported_aggregations = {
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "count",
    }

    def aggregate(
        self,
        dataset_id: str,
        group_by: list[str],
        metric: str,
        aggregation: str = "sum",
        sort_desc: bool = True,
        limit: int = 20,
    ) -> AggregationResult:
        """Aggregate a metric by one or more columns."""

        dataframe = self.registry.get(dataset_id)
        if dataframe.empty:
            raise EmptyDatasetError(dataset_id)

        self._validate_aggregation_request(
            dataframe,
            group_by,
            metric,
            aggregation,
            limit,
        )

        try:
            result = (
                dataframe.groupby(
                    group_by,
                    dropna=False,
                    observed=True,
                    sort=False,
                )[metric]
                .agg(aggregation)
                .reset_index(name=metric)
            )
        except (TypeError, ValueError) as error:
            raise AnalysisError(
                "The requested aggregation could not be computed.",
                aggregation=aggregation,
                metric=metric,
            ) from error

        result = result.sort_values(
            by=metric,
            ascending=not sort_desc,
            na_position="last",
            kind="mergesort",
        ).head(limit)
        rows = [
            {str(key): to_json_scalar(value) for key, value in row.items()}
            for row in result.to_dict(orient="records")
        ]
        return AggregationResult(
            dataset_id=dataset_id,
            group_by=list(group_by),
            metric=metric,
            aggregation=aggregation,
            rows=rows,
        )

    def correlations(self, dataset_id: str) -> CorrelationStats:
        """Compute Pearson correlations for real numeric columns."""

        dataframe = self.registry.get(dataset_id)
        if dataframe.empty:
            raise EmptyDatasetError(dataset_id)

        numeric_columns = [
            str(column)
            for column in dataframe.columns
            if pd.api.types.is_numeric_dtype(dataframe[column].dtype)
            and not pd.api.types.is_bool_dtype(dataframe[column].dtype)
            and not pd.api.types.is_complex_dtype(dataframe[column].dtype)
        ]
        if len(numeric_columns) < 2:
            raise AnalysisError(
                "Correlation analysis requires at least two numeric columns.",
                numeric_columns=numeric_columns,
            )

        correlation_frame = dataframe[numeric_columns].corr(method="pearson")
        matrix = {
            row_name: {
                column_name: self._correlation_value(
                    correlation_frame.loc[row_name, column_name]
                )
                for column_name in numeric_columns
            }
            for row_name in numeric_columns
        }

        pairs: list[CorrelationPair] = []
        for first_index, first_column in enumerate(numeric_columns):
            for second_column in numeric_columns[first_index + 1 :]:
                value = self._correlation_value(
                    correlation_frame.loc[first_column, second_column]
                )
                if value is None:
                    continue
                pairs.append(
                    CorrelationPair(
                        column_1=first_column,
                        column_2=second_column,
                        correlation=value,
                    )
                )
        pairs.sort(key=lambda pair: abs(pair.correlation), reverse=True)

        return CorrelationStats(
            dataset_id=dataset_id,
            columns=numeric_columns,
            matrix=matrix,
            strongest_pairs=pairs,
        )

    def _validate_aggregation_request(
        self,
        dataframe: pd.DataFrame,
        group_by: list[str],
        metric: str,
        aggregation: str,
        limit: int,
    ) -> None:
        available_columns = [str(column) for column in dataframe.columns]
        if not group_by:
            raise AnalysisError("At least one group_by column is required.")
        if len(group_by) != len(set(group_by)):
            raise AnalysisError("group_by columns must be unique.")

        for column in [*group_by, metric]:
            if column not in dataframe.columns:
                raise ColumnNotFoundError(column, available_columns)
        if metric in group_by:
            raise AnalysisError(
                "The metric column cannot also be a group_by column.",
                metric=metric,
            )
        if aggregation not in self.supported_aggregations:
            raise AnalysisError(
                f"Aggregation '{aggregation}' is not supported.",
                aggregation=aggregation,
                supported_aggregations=sorted(self.supported_aggregations),
            )
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise AnalysisError("limit must be a positive integer.", limit=limit)

        metric_dtype = dataframe[metric].dtype
        if aggregation != "count" and (
            not pd.api.types.is_numeric_dtype(metric_dtype)
            or pd.api.types.is_bool_dtype(metric_dtype)
            or pd.api.types.is_complex_dtype(metric_dtype)
        ):
            raise InvalidColumnTypeError(
                metric,
                ["numeric"],
                str(metric_dtype),
            )

    @staticmethod
    def _correlation_value(value: object) -> float | None:
        converted = to_json_scalar(value)
        if isinstance(converted, (int, float)) and not isinstance(converted, bool):
            return float(converted)
        return None
