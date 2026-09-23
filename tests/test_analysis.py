"""Tests for deterministic aggregation and correlation calculations."""

import json

import pandas as pd
import pytest

from src.exceptions import (
    AnalysisError,
    ColumnNotFoundError,
    DatasetNotFoundError,
    InvalidColumnTypeError,
)
from src.skills import AnalysisSkill
from src.storage import DatasetRegistry
from src.tools.analysis_tools import AnalysisTools
from src.tools.data_tools import DataTools


@pytest.fixture
def sales_analysis() -> tuple[AnalysisSkill, str]:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame(
            {
                "Region": ["North", "North", "South", "South"],
                "Product": ["Laptop", "Monitor", "Laptop", "Monitor"],
                "Sales": [10.0, 30.0, 20.0, 40.0],
                "Quantity": [1, 3, 2, 4],
            }
        )
    )
    return AnalysisSkill(registry), dataset_id


def test_sum_aggregation_by_categorical_column(sales_analysis) -> None:
    skill, dataset_id = sales_analysis

    result = skill.aggregate(dataset_id, ["Region"], "Sales")

    assert result.rows == [
        {"Region": "South", "Sales": 60.0},
        {"Region": "North", "Sales": 40.0},
    ]


def test_mean_aggregation(sales_analysis) -> None:
    skill, dataset_id = sales_analysis

    result = skill.aggregate(
        dataset_id,
        ["Region"],
        "Sales",
        aggregation="mean",
    )

    assert result.rows == [
        {"Region": "South", "Sales": 30.0},
        {"Region": "North", "Sales": 20.0},
    ]


def test_multiple_group_by_columns(sales_analysis) -> None:
    skill, dataset_id = sales_analysis

    result = skill.aggregate(
        dataset_id,
        ["Region", "Product"],
        "Sales",
        sort_desc=False,
    )

    assert result.group_by == ["Region", "Product"]
    assert result.rows[0] == {
        "Region": "North",
        "Product": "Laptop",
        "Sales": 10.0,
    }
    assert result.rows[-1] == {
        "Region": "South",
        "Product": "Monitor",
        "Sales": 40.0,
    }


def test_sort_direction_and_limit(sales_analysis) -> None:
    skill, dataset_id = sales_analysis

    result = skill.aggregate(
        dataset_id,
        ["Product"],
        "Quantity",
        sort_desc=False,
        limit=1,
    )

    assert result.rows == [{"Product": "Laptop", "Quantity": 3}]


def test_count_aggregation_allows_categorical_metric() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame(
            {
                "Region": ["North", "North", "South"],
                "Product": ["Laptop", None, "Monitor"],
            }
        )
    )

    result = AnalysisSkill(registry).aggregate(
        dataset_id,
        ["Region"],
        "Product",
        aggregation="count",
    )

    assert result.rows == [
        {"Region": "North", "Product": 1},
        {"Region": "South", "Product": 1},
    ]


@pytest.mark.parametrize("column", ["MissingGroup", "MissingMetric"])
def test_invalid_aggregation_column_is_reported(
    sales_analysis,
    column: str,
) -> None:
    skill, dataset_id = sales_analysis
    group_by = [column] if column == "MissingGroup" else ["Region"]
    metric = column if column == "MissingMetric" else "Sales"

    with pytest.raises(ColumnNotFoundError):
        skill.aggregate(dataset_id, group_by, metric)


def test_unsupported_aggregation_is_reported(sales_analysis) -> None:
    skill, dataset_id = sales_analysis

    with pytest.raises(AnalysisError) as error:
        skill.aggregate(dataset_id, ["Region"], "Sales", aggregation="mode")

    assert error.value.to_dict()["supported_aggregations"] == [
        "count",
        "max",
        "mean",
        "median",
        "min",
        "sum",
    ]


def test_nonnumeric_metric_is_rejected_for_sum() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame({"Region": ["North"], "Product": ["Laptop"]})
    )

    with pytest.raises(InvalidColumnTypeError):
        AnalysisSkill(registry).aggregate(
            dataset_id,
            ["Region"],
            "Product",
        )


@pytest.mark.parametrize("operation", ["aggregate", "correlations"])
def test_unknown_dataset_id_propagates_domain_error(operation: str) -> None:
    skill = AnalysisSkill(DatasetRegistry())

    with pytest.raises(DatasetNotFoundError):
        if operation == "aggregate":
            skill.aggregate("ds_missing", ["Region"], "Sales")
        else:
            skill.correlations("ds_missing")


def test_aggregation_result_is_json_compatible(sales_analysis) -> None:
    skill, dataset_id = sales_analysis

    result = skill.aggregate(dataset_id, ["Region"], "Sales")
    serialized = json.dumps(result.model_dump(mode="json"))

    assert result.dataset_id in serialized
    assert "DataFrame" not in serialized


@pytest.fixture
def correlation_result():
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame(
            {
                "Sales": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Profit": [2.0, 4.0, 6.0, 8.0, 10.0],
                "Quantity": [5.0, 1.0, 4.0, 2.0, 3.0],
                "Constant": [7.0, 7.0, 7.0, 7.0, 7.0],
                "Region": ["N", "S", "E", "W", "N"],
            }
        )
    )
    return AnalysisSkill(registry).correlations(dataset_id)


def test_correlation_matrix_uses_only_numeric_columns(correlation_result) -> None:
    assert correlation_result.columns == [
        "Sales",
        "Profit",
        "Quantity",
        "Constant",
    ]
    assert "Region" not in correlation_result.matrix
    assert set(correlation_result.matrix) == set(correlation_result.columns)


def test_correlation_pairs_exclude_diagonal_and_duplicates(
    correlation_result,
) -> None:
    pairs = correlation_result.strongest_pairs

    assert all(pair.column_1 != pair.column_2 for pair in pairs)
    unordered_pairs = [frozenset((pair.column_1, pair.column_2)) for pair in pairs]
    assert len(unordered_pairs) == len(set(unordered_pairs))


def test_correlation_pairs_are_sorted_by_absolute_strength(
    correlation_result,
) -> None:
    strengths = [abs(pair.correlation) for pair in correlation_result.strongest_pairs]

    assert strengths == sorted(strengths, reverse=True)
    assert correlation_result.strongest_pairs[0].column_1 == "Sales"
    assert correlation_result.strongest_pairs[0].column_2 == "Profit"
    assert correlation_result.strongest_pairs[0].correlation == pytest.approx(1.0)


def test_nan_correlations_are_json_safe_and_omitted_from_pairs(
    correlation_result,
) -> None:
    assert correlation_result.matrix["Constant"]["Sales"] is None
    assert correlation_result.matrix["Constant"]["Constant"] is None
    assert all(
        "Constant" not in {pair.column_1, pair.column_2}
        for pair in correlation_result.strongest_pairs
    )
    json.dumps(correlation_result.model_dump(mode="json"))


def test_dataset_without_two_numeric_columns_is_rejected() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame({"Region": ["North", "South"], "Sales": [1.0, 2.0]})
    )

    with pytest.raises(AnalysisError, match="at least two numeric columns"):
        AnalysisSkill(registry).correlations(dataset_id)


def test_data_and_analysis_tools_share_injected_registry(tmp_path) -> None:
    file_path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "Region": ["North", "South", "North"],
            "Sales": [10.0, 20.0, 30.0],
        }
    ).to_csv(file_path, index=False)
    registry = DatasetRegistry()
    data_tools = DataTools(registry)
    analysis_tools = AnalysisTools(registry)

    loaded = data_tools.load_data(str(file_path))
    result = analysis_tools.aggregate_data(
        loaded.dataset_id,
        ["Region"],
        "Sales",
    )

    assert result.rows == [
        {"Region": "North", "Sales": 40.0},
        {"Region": "South", "Sales": 20.0},
    ]
