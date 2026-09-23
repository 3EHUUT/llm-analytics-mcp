"""Pydantic contracts for JSON-compatible public tool results."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


JsonScalar = str | int | float | bool | None


class ResultModel(BaseModel):
    """Base model for strict public result contracts."""

    model_config = ConfigDict(extra="forbid")


class SemanticType(str, Enum):
    """Supported semantic column categories."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class NumericSummary(ResultModel):
    """Basic statistics for a numeric column."""

    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None


class ValueCount(ResultModel):
    """A JSON-safe value and its observed count."""

    value: JsonScalar
    count: int = Field(ge=0)


class ColumnInfo(ResultModel):
    """Compact schema and profiling information for one column."""

    name: str
    dtype: str
    semantic_type: SemanticType
    null_count: int = Field(ge=0)
    missing_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    unique_count: int | None = Field(default=None, ge=0)
    numeric_summary: NumericSummary | None = None
    top_values: list[ValueCount] = Field(default_factory=list)
    min_date: str | None = None
    max_date: str | None = None


class DatasetInfo(ResultModel):
    """Metadata returned after a dataset is loaded and registered."""

    dataset_id: str
    file_name: str
    rows: int = Field(ge=0)
    columns: list[ColumnInfo]


class DataProfile(ResultModel):
    """Structured overview of a registered dataset."""

    dataset_id: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[ColumnInfo]
    duplicate_row_count: int = Field(ge=0)


class MissingValueAction(ResultModel):
    """How missing values in a column were handled."""

    count: int = Field(ge=0)
    strategy: str
    fill_value: JsonScalar = None


class OutlierInfo(ResultModel):
    """Outlier detection result for a numeric column."""

    count: int = Field(ge=0)
    method: str = "IQR"
    action: str = "kept"


class CleaningReport(ResultModel):
    """Structured record of deterministic cleaning operations."""

    dataset_id: str
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
    duplicates_removed: int = Field(ge=0)
    missing_values_handled: dict[str, MissingValueAction] = Field(
        default_factory=dict
    )
    outliers: dict[str, OutlierInfo] = Field(default_factory=dict)


class AggregationResult(ResultModel):
    """Exact grouped values returned by an aggregation."""

    dataset_id: str
    group_by: list[str]
    metric: str
    aggregation: str
    rows: list[dict[str, JsonScalar]]


class ChartResult(ResultModel):
    """A chart path accompanied by values the LLM can interpret directly."""

    dataset_id: str
    chart_type: str
    file_path: str
    summary: dict[str, JsonScalar] = Field(default_factory=dict)
    data: list[dict[str, JsonScalar]] = Field(default_factory=list)


class CorrelationPair(ResultModel):
    """Correlation between two distinct numeric columns."""

    column_1: str
    column_2: str
    correlation: float


class CorrelationStats(ResultModel):
    """Computed correlation values without visualization metadata."""

    dataset_id: str
    columns: list[str]
    matrix: dict[str, dict[str, float | None]]
    strongest_pairs: list[CorrelationPair] = Field(default_factory=list)


class CorrelationResult(CorrelationStats):
    """Correlation statistics combined with a generated chart path."""

    file_path: str
