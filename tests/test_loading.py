"""Tests for loading CSV and XLSX datasets."""

import json

import pandas as pd
import pytest

from src.exceptions import (
    DataFileNotFoundError,
    EmptyDatasetError,
    UnsupportedFileFormatError,
)
from src.schemas import SemanticType
from src.skills import DataLoadingSkill
from src.storage import DatasetRegistry
from src.tools.data_tools import DataTools


@pytest.fixture
def source_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Region": ["North", "South"],
            "Sales": [100.0, 125.0],
        }
    )


def test_load_csv_registers_parsed_dataframe(
    tmp_path,
    source_dataframe: pd.DataFrame,
) -> None:
    file_path = tmp_path / "sales.csv"
    source_dataframe.to_csv(file_path, index=False)
    registry = DatasetRegistry()

    result = DataLoadingSkill(registry).run(str(file_path))

    assert result.file_name == "sales.csv"
    assert result.rows == 2
    assert registry.exists(result.dataset_id)
    stored = registry.get(result.dataset_id)
    assert pd.api.types.is_datetime64_any_dtype(stored["Date"].dtype)


def test_load_xlsx_registers_dataframe(
    tmp_path,
    source_dataframe: pd.DataFrame,
) -> None:
    file_path = tmp_path / "sales.xlsx"
    source_dataframe.to_excel(file_path, index=False)
    registry = DatasetRegistry()

    result = DataLoadingSkill(registry).run(str(file_path))

    assert result.rows == 2
    assert registry.exists(result.dataset_id)
    assert list(registry.get(result.dataset_id).columns) == [
        "Date",
        "Region",
        "Sales",
    ]


def test_public_adapters_share_registered_state(
    tmp_path,
    source_dataframe: pd.DataFrame,
) -> None:
    file_path = tmp_path / "adapter.csv"
    source_dataframe.to_csv(file_path, index=False)

    tools = DataTools(DatasetRegistry())
    loaded = tools.load_data(str(file_path))
    profile = tools.describe_data(loaded.dataset_id)

    assert profile.dataset_id == loaded.dataset_id
    assert profile.row_count == loaded.rows
    json.dumps(loaded.model_dump(mode="json"))
    json.dumps(profile.model_dump(mode="json"))


def test_unsupported_file_format_is_rejected(tmp_path) -> None:
    file_path = tmp_path / "sales.txt"
    file_path.write_text("Sales\n100\n", encoding="utf-8")

    with pytest.raises(UnsupportedFileFormatError) as error:
        DataLoadingSkill(DatasetRegistry()).run(str(file_path))

    assert error.value.to_dict()["supported_formats"] == [".csv", ".xlsx"]


def test_missing_file_is_reported(tmp_path) -> None:
    file_path = tmp_path / "missing.csv"

    with pytest.raises(DataFileNotFoundError):
        DataLoadingSkill(DatasetRegistry()).run(str(file_path))


def test_empty_dataset_is_rejected(tmp_path) -> None:
    file_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["Date", "Sales"]).to_csv(file_path, index=False)

    with pytest.raises(EmptyDatasetError):
        DataLoadingSkill(DatasetRegistry()).run(str(file_path))


def test_loading_metadata_reports_semantic_types(
    tmp_path,
    source_dataframe: pd.DataFrame,
) -> None:
    file_path = tmp_path / "types.csv"
    source_dataframe.to_csv(file_path, index=False)

    result = DataLoadingSkill(DatasetRegistry()).run(str(file_path))
    types = {column.name: column.semantic_type for column in result.columns}

    assert types == {
        "Date": SemanticType.DATETIME,
        "Region": SemanticType.CATEGORICAL,
        "Sales": SemanticType.NUMERIC,
    }
