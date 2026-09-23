"""In-memory storage for pandas DataFrames."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from uuid import uuid4

import pandas as pd

from src.exceptions import DatasetNotFoundError


class DatasetRegistry:
    """Store DataFrames in memory and address them by opaque dataset IDs."""

    def __init__(self) -> None:
        self._datasets: dict[str, pd.DataFrame] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add(
        self,
        dataframe: pd.DataFrame,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Store a DataFrame and return a newly generated dataset ID."""

        self._validate_dataframe(dataframe)
        dataset_id = self._new_dataset_id()
        self._datasets[dataset_id] = dataframe.copy(deep=True)
        self._metadata[dataset_id] = deepcopy(dict(metadata or {}))
        return dataset_id

    def get(self, dataset_id: str) -> pd.DataFrame:
        """Return a defensive copy of a registered DataFrame."""

        self._require_dataset(dataset_id)
        return self._datasets[dataset_id].copy(deep=True)

    def replace(self, dataset_id: str, dataframe: pd.DataFrame) -> None:
        """Replace a registered DataFrame while preserving its metadata."""

        self._require_dataset(dataset_id)
        self._validate_dataframe(dataframe)
        self._datasets[dataset_id] = dataframe.copy(deep=True)

    def exists(self, dataset_id: str) -> bool:
        """Return whether a dataset ID is currently registered."""

        return dataset_id in self._datasets

    def get_metadata(self, dataset_id: str) -> dict[str, Any]:
        """Return a defensive copy of a dataset's metadata."""

        self._require_dataset(dataset_id)
        return deepcopy(self._metadata[dataset_id])

    def _new_dataset_id(self) -> str:
        dataset_id = f"ds_{uuid4().hex}"
        while dataset_id in self._datasets:
            dataset_id = f"ds_{uuid4().hex}"
        return dataset_id

    def _require_dataset(self, dataset_id: str) -> None:
        if not self.exists(dataset_id):
            raise DatasetNotFoundError(dataset_id)

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
