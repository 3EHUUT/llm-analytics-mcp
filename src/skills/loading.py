"""Load local tabular files and register their DataFrames."""

import re
from pathlib import Path

import pandas as pd

from src.exceptions import (
    AnalysisError,
    DataFileNotFoundError,
    EmptyDatasetError,
    UnsupportedFileFormatError,
)
from src.schemas import ColumnInfo, DatasetInfo
from src.skills.base import BaseSkill, infer_semantic_type


class DataLoadingSkill(BaseSkill):
    """Load supported local files and store parsed data in the registry."""

    supported_extensions = (".csv", ".xlsx")
    _date_name_tokens = {
        "date",
        "datetime",
        "day",
        "month",
        "time",
        "timestamp",
        "year",
    }
    _date_value_pattern = re.compile(
        r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
        r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?:[ T].*)?$|"
        r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}",
        re.IGNORECASE,
    )

    def run(self, file_path: str) -> DatasetInfo:
        """Load a CSV or XLSX file and return compact registered metadata."""

        path = Path(file_path).expanduser()
        if not path.is_file():
            raise DataFileNotFoundError(file_path)

        extension = path.suffix.lower()
        if extension not in self.supported_extensions:
            raise UnsupportedFileFormatError(
                extension or "<none>",
                self.supported_extensions,
            )

        dataframe = self._read_file(path, extension)
        if dataframe.empty:
            raise EmptyDatasetError()

        dataframe.columns = [str(column) for column in dataframe.columns]
        dataframe = self._parse_datetime_columns(dataframe)
        dataset_id = self.registry.add(
            dataframe,
            {
                "file_name": path.name,
                "source_path": str(path.resolve()),
            },
        )

        return DatasetInfo(
            dataset_id=dataset_id,
            file_name=path.name,
            rows=len(dataframe),
            columns=[self._column_info(dataframe[column]) for column in dataframe],
        )

    @staticmethod
    def _read_file(path: Path, extension: str) -> pd.DataFrame:
        try:
            if extension == ".csv":
                return pd.read_csv(path)
            return pd.read_excel(path, engine="openpyxl")
        except pd.errors.EmptyDataError as error:
            raise EmptyDatasetError() from error
        except (OSError, UnicodeError, ValueError, ImportError) as error:
            raise AnalysisError(
                f"Could not load data file '{path.name}'.",
                file_path=str(path),
                reason=str(error),
            ) from error

    def _parse_datetime_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        parsed_dataframe = dataframe.copy()
        for column in parsed_dataframe.columns:
            series = parsed_dataframe[column]
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                continue
            if not (
                pd.api.types.is_object_dtype(series.dtype)
                or pd.api.types.is_string_dtype(series.dtype)
            ):
                continue

            non_null = series.dropna()
            if non_null.empty:
                continue

            text_values = non_null.astype("string").str.strip()
            name_tokens = set(re.split(r"[^a-z0-9]+", str(column).lower()))
            has_date_name = bool(name_tokens & self._date_name_tokens)
            date_value_ratio = float(
                text_values.str.match(self._date_value_pattern, na=False).mean()
            )
            if not has_date_name and date_value_ratio < 0.8:
                continue

            parsed_non_null = pd.to_datetime(
                text_values,
                errors="coerce",
                format="mixed",
            )
            parse_ratio = float(parsed_non_null.notna().mean())
            if parse_ratio >= 0.8:
                parsed_dataframe[column] = pd.to_datetime(
                    series,
                    errors="coerce",
                    format="mixed",
                )
        return parsed_dataframe

    @staticmethod
    def _column_info(series: pd.Series) -> ColumnInfo:
        row_count = len(series)
        null_count = int(series.isna().sum())
        return ColumnInfo(
            name=str(series.name),
            dtype=str(series.dtype),
            semantic_type=infer_semantic_type(series),
            null_count=null_count,
            missing_percentage=round((null_count / row_count) * 100, 2),
            unique_count=int(series.nunique(dropna=True)),
        )
