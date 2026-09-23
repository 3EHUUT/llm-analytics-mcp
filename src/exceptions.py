"""Domain exceptions exposed through the future MCP adapter layer."""

from collections.abc import Iterable
from typing import Any, ClassVar


class AnalyticsMCPError(Exception):
    """Base class for expected, user-facing analytics errors."""

    error_code: ClassVar[str] = "analytics_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        """Return a compact payload suitable for an MCP response."""

        return {
            "error": self.error_code,
            "message": self.message,
            **self.details,
        }


class DatasetNotFoundError(AnalyticsMCPError):
    """Raised when a dataset identifier is not present in the registry."""

    error_code = "dataset_not_found"

    def __init__(self, dataset_id: str) -> None:
        super().__init__(
            f"Dataset '{dataset_id}' does not exist.",
            dataset_id=dataset_id,
        )


class DataFileNotFoundError(AnalyticsMCPError):
    """Raised when a requested local data file does not exist."""

    error_code = "file_not_found"

    def __init__(self, file_path: str) -> None:
        super().__init__(
            f"Data file '{file_path}' does not exist.",
            file_path=file_path,
        )


class UnsupportedFileFormatError(AnalyticsMCPError):
    """Raised when a data file has an unsupported extension."""

    error_code = "unsupported_file_format"

    def __init__(
        self,
        extension: str,
        supported_formats: Iterable[str] = (".csv", ".xlsx"),
    ) -> None:
        formats = list(supported_formats)
        super().__init__(
            f"File format '{extension}' is not supported.",
            extension=extension,
            supported_formats=formats,
        )


class ColumnNotFoundError(AnalyticsMCPError):
    """Raised when a requested column is absent from a dataset."""

    error_code = "column_not_found"

    def __init__(self, column: str, available_columns: Iterable[str]) -> None:
        super().__init__(
            f"Column '{column}' does not exist.",
            column=column,
            available_columns=list(available_columns),
        )


class InvalidColumnTypeError(AnalyticsMCPError):
    """Raised when a column cannot be used by an operation."""

    error_code = "invalid_column_type"

    def __init__(
        self,
        column: str,
        expected_types: Iterable[str],
        actual_type: str,
    ) -> None:
        expected = list(expected_types)
        super().__init__(
            f"Column '{column}' must have one of these types: {', '.join(expected)}.",
            column=column,
            expected_types=expected,
            actual_type=actual_type,
        )


class EmptyDatasetError(AnalyticsMCPError):
    """Raised when an operation requires rows but the dataset is empty."""

    error_code = "empty_dataset"

    def __init__(self, dataset_id: str | None = None) -> None:
        details = {"dataset_id": dataset_id} if dataset_id is not None else {}
        super().__init__("The dataset contains no rows.", **details)


class AnalysisError(AnalyticsMCPError):
    """Raised when a requested deterministic analysis cannot be completed."""

    error_code = "analysis_error"
