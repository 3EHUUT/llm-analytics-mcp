"""Shared foundations for deterministic analytics skills."""

from math import isfinite
from typing import Any

import pandas as pd

from src.schemas import JsonScalar, SemanticType
from src.storage import DatasetRegistry


class BaseSkill:
    """Base class for skills that operate on registered datasets."""

    def __init__(self, registry: DatasetRegistry) -> None:
        self.registry = registry


def infer_semantic_type(series: pd.Series) -> SemanticType:
    """Map a pandas dtype to the compact semantic types exposed publicly."""

    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return SemanticType.BOOLEAN
    if pd.api.types.is_numeric_dtype(dtype):
        return SemanticType.NUMERIC
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return SemanticType.DATETIME
    if (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    ):
        return SemanticType.CATEGORICAL
    return SemanticType.UNKNOWN


def to_json_scalar(value: Any) -> JsonScalar:
    """Convert a pandas or NumPy scalar to a JSON-compatible primitive."""

    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
