"""Tests for compact dataset profiling."""

import json

import pandas as pd
import pytest

from src.exceptions import DatasetNotFoundError, EmptyDatasetError
from src.schemas import SemanticType
from src.skills import DataProfilingSkill
from src.storage import DatasetRegistry


@pytest.fixture
def profile_result():
    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", None, "2024-01-01"]
            ),
            "Region": ["North", "South", "North", "North"],
            "Sales": [10.0, 20.0, None, 10.0],
        }
    )
    registry = DatasetRegistry()
    dataset_id = registry.add(dataframe)
    return DataProfilingSkill(registry).run(dataset_id)


def test_profile_reports_shape_types_missing_and_duplicates(profile_result) -> None:
    profile = profile_result
    columns = {column.name: column for column in profile.columns}

    assert profile.row_count == 4
    assert profile.column_count == 3
    assert profile.duplicate_row_count == 1
    assert columns["Date"].semantic_type is SemanticType.DATETIME
    assert columns["Region"].semantic_type is SemanticType.CATEGORICAL
    assert columns["Sales"].semantic_type is SemanticType.NUMERIC
    assert columns["Sales"].dtype == "float64"
    assert columns["Sales"].null_count == 1
    assert columns["Sales"].missing_percentage == 25.0
    assert columns["Sales"].unique_count == 2


def test_profile_reports_numeric_statistics(profile_result) -> None:
    sales = next(
        column for column in profile_result.columns if column.name == "Sales"
    )

    assert sales.numeric_summary is not None
    assert sales.numeric_summary.min == 10.0
    assert sales.numeric_summary.max == 20.0
    assert sales.numeric_summary.mean == pytest.approx(13.333333)
    assert sales.numeric_summary.median == 10.0
    assert sales.numeric_summary.std == pytest.approx(5.773503)


def test_profile_reports_top_categorical_values(profile_result) -> None:
    region = next(
        column for column in profile_result.columns if column.name == "Region"
    )

    assert [(item.value, item.count) for item in region.top_values] == [
        ("North", 3),
        ("South", 1),
    ]


def test_profile_reports_datetime_range(profile_result) -> None:
    date = next(column for column in profile_result.columns if column.name == "Date")

    assert date.min_date == "2024-01-01T00:00:00"
    assert date.max_date == "2024-01-02T00:00:00"


def test_profile_is_json_compatible(profile_result) -> None:
    serialized = json.dumps(profile_result.model_dump(mode="json"))

    assert "DataFrame" not in serialized


def test_unknown_dataset_id_is_reported() -> None:
    with pytest.raises(DatasetNotFoundError):
        DataProfilingSkill(DatasetRegistry()).run("ds_missing")


def test_registered_empty_dataset_is_rejected() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(pd.DataFrame(columns=["Sales"]))

    with pytest.raises(EmptyDatasetError):
        DataProfilingSkill(registry).run(dataset_id)
