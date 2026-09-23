"""Headless matplotlib visualizations for registered datasets."""

from math import isfinite
from pathlib import Path
from textwrap import fill
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.config import OUTPUT_DIR
from src.exceptions import (
    AnalysisError,
    ColumnNotFoundError,
    EmptyDatasetError,
    InvalidColumnTypeError,
)
from src.schemas import ChartResult, CorrelationResult, SemanticType
from src.skills.analysis import AnalysisSkill
from src.skills.base import BaseSkill, infer_semantic_type, to_json_scalar
from src.storage import DatasetRegistry


class VisualizationSkill(BaseSkill):
    """Create PNG charts and return the exact values used to plot them."""

    supported_frequencies = {"D", "W", "M", "Q", "Y"}

    def __init__(
        self,
        registry: DatasetRegistry,
        output_dir: str | Path | None = None,
    ) -> None:
        super().__init__(registry)
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.analysis = AnalysisSkill(registry)

    def plot_trend(
        self,
        dataset_id: str,
        x_column: str,
        y_column: str,
        aggregation: str = "sum",
        frequency: str | None = None,
    ) -> ChartResult:
        """Plot a numeric metric over a datetime-compatible column."""

        dataframe = self._get_dataframe(dataset_id)
        self._require_column(dataframe, x_column)
        self._require_numeric_column(dataframe, y_column)
        if aggregation not in AnalysisSkill.supported_aggregations:
            raise AnalysisError(
                f"Aggregation '{aggregation}' is not supported.",
                aggregation=aggregation,
                supported_aggregations=sorted(
                    AnalysisSkill.supported_aggregations
                ),
            )

        normalized_frequency = frequency.upper() if frequency else None
        if (
            normalized_frequency
            and normalized_frequency not in self.supported_frequencies
        ):
            raise AnalysisError(
                f"Frequency '{frequency}' is not supported.",
                frequency=frequency,
                supported_frequencies=sorted(self.supported_frequencies),
            )

        datetimes = self._coerce_datetime(dataframe[x_column], x_column)
        working = pd.DataFrame(
            {"_time": datetimes, "_value": dataframe[y_column]}
        ).dropna(subset=["_time"])
        if working["_value"].dropna().empty:
            raise AnalysisError(
                f"Column '{y_column}' contains no usable numeric values.",
                column=y_column,
            )

        try:
            if normalized_frequency:
                periods = working["_time"].dt.to_period(normalized_frequency)
                trend = working.groupby(periods, sort=True)["_value"].agg(
                    aggregation
                )
                trend.index = trend.index.to_timestamp(how="start")
            else:
                trend = working.groupby("_time", sort=True)["_value"].agg(
                    aggregation
                )
        except (TypeError, ValueError) as error:
            raise AnalysisError(
                "The requested trend aggregation could not be computed.",
                aggregation=aggregation,
            ) from error

        trend = trend[trend.map(self._is_finite_number)]
        if trend.empty:
            raise AnalysisError("No usable values remain for the trend chart.")

        plotted_data = [
            {
                x_column: to_json_scalar(timestamp),
                y_column: to_json_scalar(value),
            }
            for timestamp, value in trend.items()
        ]
        summary = self._trend_summary(trend)

        figure, axis = plt.subplots(figsize=(8, 4.5))
        try:
            axis.plot(trend.index, trend.values, marker="o", linewidth=1.8)
            axis.set_title(f"{aggregation.title()} of {y_column} over {x_column}")
            axis.set_xlabel(x_column)
            axis.set_ylabel(y_column)
            axis.grid(True, alpha=0.25)
            figure.autofmt_xdate()
            file_path = self._save_figure(figure, "trend")
        finally:
            plt.close(figure)

        return ChartResult(
            dataset_id=dataset_id,
            chart_type="line",
            file_path=file_path,
            summary=summary,
            data=plotted_data,
        )

    def plot_distribution(
        self,
        dataset_id: str,
        column: str,
        bins: int = 20,
    ) -> ChartResult:
        """Plot a histogram and return its numeric summary and bin counts."""

        dataframe = self._get_dataframe(dataset_id)
        self._require_numeric_column(dataframe, column)
        if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
            raise AnalysisError("bins must be a positive integer.", bins=bins)

        values = dataframe[column].dropna()
        values = values[values.map(self._is_finite_number)]
        if values.empty:
            raise AnalysisError(
                f"Column '{column}' contains no usable numeric values.",
                column=column,
            )

        summary = {
            "count": int(values.count()),
            "mean": to_json_scalar(values.mean()),
            "median": to_json_scalar(values.median()),
            "std": to_json_scalar(values.std()),
            "min": to_json_scalar(values.min()),
            "max": to_json_scalar(values.max()),
        }

        figure, axis = plt.subplots(figsize=(7, 4.5))
        try:
            counts, edges, _ = axis.hist(values, bins=bins, edgecolor="black")
            axis.set_title(f"Distribution of {column}")
            axis.set_xlabel(column)
            axis.set_ylabel("Frequency")
            file_path = self._save_figure(figure, "distribution")
        finally:
            plt.close(figure)

        histogram_data = [
            {
                "bin_start": to_json_scalar(edges[index]),
                "bin_end": to_json_scalar(edges[index + 1]),
                "count": int(count),
            }
            for index, count in enumerate(counts)
        ]
        return ChartResult(
            dataset_id=dataset_id,
            chart_type="histogram",
            file_path=file_path,
            summary=summary,
            data=histogram_data,
        )

    def plot_bar(
        self,
        dataset_id: str,
        category_column: str,
        value_column: str,
        aggregation: str = "sum",
        top_n: int = 10,
    ) -> ChartResult:
        """Plot top grouped values using the shared aggregation logic."""

        dataframe = self._get_dataframe(dataset_id)
        self._require_column(dataframe, category_column)
        category_type = infer_semantic_type(dataframe[category_column])
        if category_type not in {
            SemanticType.CATEGORICAL,
            SemanticType.DATETIME,
        }:
            raise InvalidColumnTypeError(
                category_column,
                ["categorical", "datetime"],
                str(dataframe[category_column].dtype),
            )
        if aggregation == "count":
            self._require_column(dataframe, value_column)
        else:
            self._require_numeric_column(dataframe, value_column)
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise AnalysisError("top_n must be a positive integer.", top_n=top_n)

        aggregated = self.analysis.aggregate(
            dataset_id,
            [category_column],
            value_column,
            aggregation=aggregation,
            sort_desc=True,
            limit=top_n,
        )
        plotted_data = [
            row
            for row in aggregated.rows
            if self._is_finite_number(row[value_column])
        ]
        if not plotted_data:
            raise AnalysisError("No usable values remain for the bar chart.")

        labels = [
            fill(
                "(missing)"
                if row[category_column] is None
                else str(row[category_column]),
                width=20,
            )
            for row in plotted_data
        ]
        values = [row[value_column] for row in plotted_data]

        figure, axis = plt.subplots(figsize=(8, 4.8))
        try:
            axis.bar(labels, values)
            axis.set_title(
                f"Top {len(plotted_data)} {category_column} by "
                f"{aggregation} of {value_column}"
            )
            axis.set_xlabel(category_column)
            axis.set_ylabel(f"{aggregation.title()} of {value_column}")
            axis.tick_params(axis="x", labelrotation=35)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
            file_path = self._save_figure(figure, "bar")
        finally:
            plt.close(figure)

        return ChartResult(
            dataset_id=dataset_id,
            chart_type="bar",
            file_path=file_path,
            summary={
                "aggregation": aggregation,
                "top_category": plotted_data[0][category_column],
                "top_value": plotted_data[0][value_column],
            },
            data=plotted_data,
        )

    def correlation_analysis(self, dataset_id: str) -> CorrelationResult:
        """Render a heatmap from the shared deterministic correlation result."""

        statistics = self.analysis.correlations(dataset_id)
        matrix_values = [
            [
                statistics.matrix[row_name][column_name]
                if statistics.matrix[row_name][column_name] is not None
                else float("nan")
                for column_name in statistics.columns
            ]
            for row_name in statistics.columns
        ]

        size = max(5.0, min(10.0, len(statistics.columns) * 1.15))
        figure, axis = plt.subplots(figsize=(size, size))
        try:
            image = axis.imshow(
                matrix_values,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                aspect="auto",
            )
            axis.set_title("Pearson Correlation Matrix")
            axis.set_xlabel("Numeric columns")
            axis.set_ylabel("Numeric columns")
            positions = range(len(statistics.columns))
            axis.set_xticks(positions, labels=statistics.columns, rotation=45)
            axis.set_yticks(positions, labels=statistics.columns)
            figure.colorbar(image, ax=axis, label="Pearson correlation")

            if len(statistics.columns) <= 10:
                for row_index, row_name in enumerate(statistics.columns):
                    for column_index, column_name in enumerate(
                        statistics.columns
                    ):
                        value = statistics.matrix[row_name][column_name]
                        if value is not None:
                            axis.text(
                                column_index,
                                row_index,
                                f"{value:.2f}",
                                ha="center",
                                va="center",
                                fontsize=8,
                            )
            file_path = self._save_figure(figure, "correlation")
        finally:
            plt.close(figure)

        return CorrelationResult(
            **statistics.model_dump(),
            file_path=file_path,
        )

    def _get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        dataframe = self.registry.get(dataset_id)
        if dataframe.empty:
            raise EmptyDatasetError(dataset_id)
        return dataframe

    @staticmethod
    def _require_column(dataframe: pd.DataFrame, column: str) -> None:
        if column not in dataframe.columns:
            raise ColumnNotFoundError(
                column,
                [str(available) for available in dataframe.columns],
            )

    def _require_numeric_column(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> None:
        self._require_column(dataframe, column)
        dtype = dataframe[column].dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            raise InvalidColumnTypeError(column, ["numeric"], str(dtype))

    @staticmethod
    def _coerce_datetime(series: pd.Series, column: str) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return series
        if not (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            raise InvalidColumnTypeError(
                column,
                ["datetime-compatible"],
                str(series.dtype),
            )

        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        non_null_count = int(series.notna().sum())
        parse_ratio = (
            float(parsed.notna().sum() / non_null_count)
            if non_null_count
            else 0.0
        )
        if parse_ratio < 0.8:
            raise InvalidColumnTypeError(
                column,
                ["datetime-compatible"],
                str(series.dtype),
            )
        return parsed

    @staticmethod
    def _trend_summary(trend: pd.Series) -> dict[str, str | int | float | None]:
        first_value = to_json_scalar(trend.iloc[0])
        last_value = to_json_scalar(trend.iloc[-1])
        change_pct = None
        if (
            isinstance(first_value, (int, float))
            and not isinstance(first_value, bool)
            and isinstance(last_value, (int, float))
            and not isinstance(last_value, bool)
            and first_value != 0
        ):
            change_pct = ((last_value - first_value) / first_value) * 100

        return {
            "first_value": first_value,
            "last_value": last_value,
            "change_pct": change_pct,
            "min_value": to_json_scalar(trend.min()),
            "max_value": to_json_scalar(trend.max()),
            "peak_period": to_json_scalar(trend.idxmax()),
        }

    def _save_figure(self, figure: Figure, prefix: str) -> str:
        file_path = self.output_dir / f"{prefix}_{uuid4().hex}.png"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            figure.tight_layout()
            figure.savefig(file_path, format="png", dpi=140)
        except (OSError, ValueError) as error:
            raise AnalysisError(
                "The chart image could not be saved.",
                file_path=str(file_path),
            ) from error
        return str(file_path.resolve())

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        if value is None or isinstance(value, bool):
            return False
        try:
            return isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
