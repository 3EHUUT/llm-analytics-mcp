"""Tests for deterministic dataset cleaning."""

import json

import pandas as pd
import pytest

from src.exceptions import DatasetNotFoundError
from src.skills import DataCleaningSkill
from src.storage import DatasetRegistry
from src.tools.data_tools import DataTools


def test_exact_duplicates_are_removed_and_stored() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame(
            {
                "Region": ["North", "North", "South"],
                "Sales": [10.0, 10.0, 20.0],
            }
        )
    )

    report = DataCleaningSkill(registry).run(dataset_id)

    assert report.duplicates_removed == 1
    assert report.rows_before == 3
    assert report.rows_after == 2
    assert len(registry.get(dataset_id)) == 2


def test_numeric_missing_values_are_filled_with_median() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(pd.DataFrame({"Sales": [10.0, None, 30.0]}))

    report = DataCleaningSkill(registry).run(
        dataset_id,
        remove_duplicates=False,
    )

    stored = registry.get(dataset_id)
    assert stored["Sales"].tolist() == [10.0, 20.0, 30.0]
    assert str(stored["Sales"].dtype) == "float64"
    assert report.missing_values_handled["Sales"].strategy == "median"
    assert report.missing_values_handled["Sales"].fill_value == 20.0


def test_fractional_median_uses_compatible_nullable_numeric_dtype() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame({"Quantity": pd.Series([1, None, 2], dtype="Int64")})
    )

    DataCleaningSkill(registry).run(dataset_id)

    stored = registry.get(dataset_id)
    assert stored["Quantity"].tolist() == [1.0, 1.5, 2.0]
    assert str(stored["Quantity"].dtype) == "Float64"


def test_numeric_column_without_median_is_left_missing() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame({"Sales": pd.Series([None, None], dtype="Float64")})
    )

    report = DataCleaningSkill(registry).run(dataset_id)

    assert registry.get(dataset_id)["Sales"].isna().all()
    assert report.missing_values_handled["Sales"].strategy == "skipped_no_median"


def test_categorical_missing_values_use_mode_without_dtype_change() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame(
            {"Region": pd.Series(["North", None, "North"], dtype="category")}
        )
    )

    report = DataCleaningSkill(registry).run(
        dataset_id,
        remove_duplicates=False,
    )

    stored = registry.get(dataset_id)
    assert stored["Region"].tolist() == ["North", "North", "North"]
    assert str(stored["Region"].dtype) == "category"
    assert report.missing_values_handled["Region"].strategy == "mode"
    assert report.missing_values_handled["Region"].fill_value == "North"


def test_categorical_column_without_mode_uses_unknown() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(pd.DataFrame({"Region": [None, None]}))

    report = DataCleaningSkill(registry).run(
        dataset_id,
        remove_duplicates=False,
    )

    assert registry.get(dataset_id)["Region"].tolist() == ["Unknown", "Unknown"]
    assert report.missing_values_handled["Region"].strategy == "constant_unknown"


def test_boolean_column_without_mode_is_left_unchanged() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame({"Active": pd.Series([None, None], dtype="boolean")})
    )

    report = DataCleaningSkill(registry).run(dataset_id)

    assert registry.get(dataset_id)["Active"].isna().all()
    assert (
        report.missing_values_handled["Active"].strategy
        == "skipped_incompatible_type"
    )


def test_datetime_missing_values_remain_missing() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", None])})
    )

    report = DataCleaningSkill(registry).run(dataset_id)

    assert registry.get(dataset_id)["Date"].isna().sum() == 1
    assert report.missing_values_handled["Date"].strategy == "left_missing"


def test_iqr_outliers_are_reported_but_not_removed() -> None:
    registry = DatasetRegistry()
    dataframe = pd.DataFrame({"Sales": [10.0, 11.0, 12.0, 13.0, 100.0]})
    dataset_id = registry.add(dataframe)

    report = DataCleaningSkill(registry).run(dataset_id)

    assert report.outliers["Sales"].count == 1
    assert report.outliers["Sales"].method == "IQR"
    assert report.outliers["Sales"].action == "kept"
    assert report.rows_after == report.rows_before == 5
    assert registry.get(dataset_id)["Sales"].tolist() == dataframe["Sales"].tolist()


def test_remove_duplicates_can_be_disabled() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(pd.DataFrame({"Sales": [10.0, 10.0]}))

    report = DataCleaningSkill(registry).run(
        dataset_id,
        remove_duplicates=False,
    )

    assert report.duplicates_removed == 0
    assert len(registry.get(dataset_id)) == 2


def test_missing_value_handling_can_be_disabled() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(pd.DataFrame({"Sales": [10.0, None, 30.0]}))

    report = DataCleaningSkill(registry).run(
        dataset_id,
        handle_missing=False,
    )

    assert registry.get(dataset_id)["Sales"].isna().sum() == 1
    assert report.missing_values_handled == {}


def test_outlier_detection_can_be_disabled() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(pd.DataFrame({"Sales": [1.0, 2.0, 100.0]}))

    report = DataCleaningSkill(registry).run(
        dataset_id,
        detect_outliers=False,
    )

    assert report.outliers == {}
    assert len(registry.get(dataset_id)) == 3


def test_data_tools_cleaning_uses_composed_registry(tmp_path) -> None:
    file_path = tmp_path / "sales.csv"
    pd.DataFrame({"Sales": [10.0, None, 30.0]}).to_csv(file_path, index=False)
    registry = DatasetRegistry()
    tools = DataTools(registry)

    loaded = tools.load_data(str(file_path))
    report = tools.clean_data(loaded.dataset_id)
    profile = tools.describe_data(loaded.dataset_id)

    assert report.dataset_id == loaded.dataset_id
    assert profile.columns[0].null_count == 0
    assert registry.get(loaded.dataset_id)["Sales"].tolist() == [10.0, 20.0, 30.0]


def test_unknown_dataset_id_propagates_domain_error() -> None:
    with pytest.raises(DatasetNotFoundError):
        DataCleaningSkill(DatasetRegistry()).run("ds_missing")


def test_cleaning_report_is_json_compatible() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame(
            {
                "Region": ["North", None, "North"],
                "Sales": [10.0, None, 100.0],
            }
        )
    )

    report = DataCleaningSkill(registry).run(dataset_id)
    serialized = json.dumps(report.model_dump(mode="json"))

    assert report.dataset_id in serialized
    assert "DataFrame" not in serialized
