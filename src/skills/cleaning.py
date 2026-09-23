"""Deterministic cleaning for registered datasets."""

import pandas as pd

from src.exceptions import EmptyDatasetError
from src.schemas import (
    CleaningReport,
    MissingValueAction,
    OutlierInfo,
    SemanticType,
)
from src.skills.base import BaseSkill, infer_semantic_type, to_json_scalar


class DataCleaningSkill(BaseSkill):
    """Clean missing values and duplicates, and report retained outliers."""

    def run(
        self,
        dataset_id: str,
        handle_missing: bool = True,
        remove_duplicates: bool = True,
        detect_outliers: bool = True,
    ) -> CleaningReport:
        """Clean a registered dataset and replace its stored DataFrame."""

        dataframe = self.registry.get(dataset_id)
        if dataframe.empty:
            raise EmptyDatasetError(dataset_id)

        rows_before = len(dataframe)
        duplicates_removed = 0
        if remove_duplicates:
            duplicates_removed = int(dataframe.duplicated().sum())
            dataframe = dataframe.drop_duplicates().copy()

        missing_actions: dict[str, MissingValueAction] = {}
        if handle_missing:
            dataframe, missing_actions = self._handle_missing_values(dataframe)

        outliers: dict[str, OutlierInfo] = {}
        if detect_outliers:
            outliers = self._detect_outliers(dataframe)

        self.registry.replace(dataset_id, dataframe)
        return CleaningReport(
            dataset_id=dataset_id,
            rows_before=rows_before,
            rows_after=len(dataframe),
            duplicates_removed=duplicates_removed,
            missing_values_handled=missing_actions,
            outliers=outliers,
        )

    def _handle_missing_values(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, MissingValueAction]]:
        cleaned = dataframe.copy()
        actions: dict[str, MissingValueAction] = {}

        for column_name in cleaned.columns:
            series = cleaned[column_name]
            missing_count = int(series.isna().sum())
            if missing_count == 0:
                continue

            semantic_type = infer_semantic_type(series)
            if semantic_type is SemanticType.NUMERIC:
                filled, fill_value, strategy = self._fill_numeric(series)
            elif semantic_type in {
                SemanticType.CATEGORICAL,
                SemanticType.BOOLEAN,
            }:
                filled, fill_value, strategy = self._fill_categorical(series)
            elif semantic_type is SemanticType.DATETIME:
                filled, fill_value, strategy = series, None, "left_missing"
            else:
                filled, fill_value, strategy = (
                    series,
                    None,
                    "skipped_unsupported_type",
                )

            cleaned[column_name] = filled
            actions[str(column_name)] = MissingValueAction(
                count=missing_count,
                strategy=strategy,
                fill_value=to_json_scalar(fill_value),
            )

        return cleaned, actions

    @staticmethod
    def _fill_numeric(
        series: pd.Series,
    ) -> tuple[pd.Series, object | None, str]:
        usable_values = series.dropna()
        if usable_values.empty:
            return series, None, "skipped_no_median"

        median = usable_values.median()
        if pd.isna(median):
            return series, None, "skipped_no_median"

        try:
            return series.fillna(median), median, "median"
        except TypeError:
            if pd.api.types.is_integer_dtype(series.dtype):
                converted = series.astype("Float64")
                return converted.fillna(float(median)), median, "median"
            return series, None, "skipped_incompatible_type"

    @staticmethod
    def _fill_categorical(
        series: pd.Series,
    ) -> tuple[pd.Series, object | None, str]:
        mode = series.mode(dropna=True)
        fill_value: object = mode.iloc[0] if not mode.empty else "Unknown"
        strategy = "mode" if not mode.empty else "constant_unknown"

        try:
            if isinstance(series.dtype, pd.CategoricalDtype):
                if fill_value not in series.cat.categories:
                    series = series.cat.add_categories([fill_value])
            return series.fillna(fill_value), fill_value, strategy
        except (TypeError, ValueError):
            return series, None, "skipped_incompatible_type"

    @staticmethod
    def _detect_outliers(dataframe: pd.DataFrame) -> dict[str, OutlierInfo]:
        results: dict[str, OutlierInfo] = {}
        for column_name in dataframe.columns:
            series = dataframe[column_name]
            if infer_semantic_type(series) is not SemanticType.NUMERIC:
                continue

            values = series.dropna()
            count = 0
            if not values.empty:
                try:
                    first_quartile = values.quantile(0.25)
                    third_quartile = values.quantile(0.75)
                    iqr = third_quartile - first_quartile
                    if not pd.isna(iqr):
                        lower_bound = first_quartile - (1.5 * iqr)
                        upper_bound = third_quartile + (1.5 * iqr)
                        count = int(
                            ((values < lower_bound) | (values > upper_bound)).sum()
                        )
                except (TypeError, ValueError):
                    count = 0

            results[str(column_name)] = OutlierInfo(count=count)
        return results
