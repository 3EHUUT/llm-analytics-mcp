"""Tests for headless chart generation and structured chart results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.exceptions import (
    AnalysisError,
    ColumnNotFoundError,
    InvalidColumnTypeError,
)
from src.skills import AnalysisSkill, VisualizationSkill
from src.storage import DatasetRegistry
from src.tools.analysis_tools import AnalysisTools
from src.tools.data_tools import DataTools
from src.tools.visualization_tools import VisualizationTools


def assert_nonempty_png(file_path: str) -> None:
    path = Path(file_path)
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


@pytest.fixture
def visualization_setup(tmp_path):
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"]
                ),
                "Region": ["North", "South", "North", "South"],
                "Sales": [10.0, 20.0, 30.0, 40.0],
                "Profit": [2.0, 4.0, 6.0, 8.0],
                "Quantity": [4.0, 1.0, 3.0, 2.0],
                "Constant": [5.0, 5.0, 5.0, 5.0],
            }
        )
    )
    return registry, dataset_id, VisualizationTools(registry, tmp_path)


def test_plot_trend_creates_png_and_correct_summary(visualization_setup) -> None:
    _, dataset_id, tools = visualization_setup
    open_figures = plt.get_fignums()

    result = tools.plot_trend(dataset_id, "Date", "Sales")

    assert_nonempty_png(result.file_path)
    assert result.chart_type == "line"
    assert result.summary == {
        "first_value": 10.0,
        "last_value": 40.0,
        "change_pct": 300.0,
        "min_value": 10.0,
        "max_value": 40.0,
        "peak_period": "2024-02-15T00:00:00",
    }
    assert len(result.data) == 4
    assert plt.get_fignums() == open_figures
    json.dumps(result.model_dump(mode="json"))


def test_plot_trend_aggregates_by_frequency(visualization_setup) -> None:
    _, dataset_id, tools = visualization_setup

    result = tools.plot_trend(
        dataset_id,
        "Date",
        "Sales",
        frequency="M",
    )

    assert [row["Sales"] for row in result.data] == [30.0, 70.0]
    assert result.summary["first_value"] == 30.0
    assert result.summary["last_value"] == 70.0
    assert result.summary["change_pct"] == pytest.approx(133.333333)
    assert result.summary["peak_period"] == "2024-02-01T00:00:00"
    assert_nonempty_png(result.file_path)


@pytest.mark.parametrize("frequency", ["D", "W", "M", "Q", "Y"])
def test_plot_trend_supports_required_frequencies(
    visualization_setup,
    frequency: str,
) -> None:
    _, dataset_id, tools = visualization_setup

    result = tools.plot_trend(
        dataset_id,
        "Date",
        "Sales",
        frequency=frequency,
    )

    assert result.data
    assert_nonempty_png(result.file_path)


def test_plot_trend_handles_zero_first_value(tmp_path) -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "Sales": [0.0, 10.0],
            }
        )
    )

    result = VisualizationSkill(registry, tmp_path).plot_trend(
        dataset_id,
        "Date",
        "Sales",
    )

    assert result.summary["change_pct"] is None
    assert_nonempty_png(result.file_path)


@pytest.mark.parametrize(
    ("x_column", "y_column"),
    [("Sales", "Profit"), ("Date", "Region")],
)
def test_plot_trend_rejects_invalid_column_types(
    visualization_setup,
    x_column: str,
    y_column: str,
) -> None:
    _, dataset_id, tools = visualization_setup

    with pytest.raises(InvalidColumnTypeError):
        tools.plot_trend(dataset_id, x_column, y_column)


def test_plot_trend_rejects_invalid_frequency(visualization_setup) -> None:
    _, dataset_id, tools = visualization_setup

    with pytest.raises(AnalysisError, match="Frequency"):
        tools.plot_trend(dataset_id, "Date", "Sales", frequency="hour")


def test_plot_distribution_creates_png_and_summary(visualization_setup) -> None:
    _, dataset_id, tools = visualization_setup

    result = tools.plot_distribution(dataset_id, "Sales", bins=2)

    assert_nonempty_png(result.file_path)
    assert result.chart_type == "histogram"
    assert result.summary["count"] == 4
    assert result.summary["mean"] == 25.0
    assert result.summary["median"] == 25.0
    assert result.summary["std"] == pytest.approx(12.909944)
    assert result.summary["min"] == 10.0
    assert result.summary["max"] == 40.0
    assert len(result.data) == 2
    json.dumps(result.model_dump(mode="json"))


def test_generated_chart_names_are_unique(visualization_setup) -> None:
    _, dataset_id, tools = visualization_setup

    first = tools.plot_distribution(dataset_id, "Sales")
    second = tools.plot_distribution(dataset_id, "Sales")

    assert first.file_path != second.file_path
    assert_nonempty_png(first.file_path)
    assert_nonempty_png(second.file_path)


def test_plot_distribution_rejects_invalid_type(visualization_setup) -> None:
    _, dataset_id, tools = visualization_setup

    with pytest.raises(InvalidColumnTypeError):
        tools.plot_distribution(dataset_id, "Region")


@pytest.mark.parametrize("bins", [0, -1, True])
def test_plot_distribution_rejects_invalid_bins(
    visualization_setup,
    bins,
) -> None:
    _, dataset_id, tools = visualization_setup

    with pytest.raises(AnalysisError, match="bins"):
        tools.plot_distribution(dataset_id, "Sales", bins=bins)


def test_plot_bar_reuses_aggregation_and_applies_top_n(
    visualization_setup,
) -> None:
    registry, dataset_id, tools = visualization_setup
    expected = AnalysisTools(registry).aggregate_data(
        dataset_id,
        ["Region"],
        "Sales",
        limit=1,
    )

    result = tools.plot_bar(
        dataset_id,
        "Region",
        "Sales",
        top_n=1,
    )

    assert_nonempty_png(result.file_path)
    assert result.data == expected.rows
    assert len(result.data) == 1
    assert result.summary == {
        "aggregation": "sum",
        "top_category": "South",
        "top_value": 60.0,
    }
    json.dumps(result.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("category_column", "value_column"),
    [("Quantity", "Sales"), ("Region", "Region")],
)
def test_plot_bar_rejects_invalid_column_types(
    visualization_setup,
    category_column: str,
    value_column: str,
) -> None:
    _, dataset_id, tools = visualization_setup

    with pytest.raises(InvalidColumnTypeError):
        tools.plot_bar(dataset_id, category_column, value_column)


@pytest.mark.parametrize("top_n", [0, -1, True])
def test_plot_bar_rejects_invalid_top_n(
    visualization_setup,
    top_n,
) -> None:
    _, dataset_id, tools = visualization_setup

    with pytest.raises(AnalysisError, match="top_n"):
        tools.plot_bar(dataset_id, "Region", "Sales", top_n=top_n)


def test_visualizations_report_missing_columns(visualization_setup) -> None:
    _, dataset_id, tools = visualization_setup

    with pytest.raises(ColumnNotFoundError):
        tools.plot_distribution(dataset_id, "Revenue")


def test_visualizations_reject_unsupported_aggregation(
    visualization_setup,
) -> None:
    _, dataset_id, tools = visualization_setup

    with pytest.raises(AnalysisError, match="not supported"):
        tools.plot_bar(
            dataset_id,
            "Region",
            "Sales",
            aggregation="mode",
        )


def test_correlation_analysis_creates_matching_heatmap(
    visualization_setup,
) -> None:
    registry, dataset_id, tools = visualization_setup
    expected = AnalysisSkill(registry).correlations(dataset_id)

    result = tools.correlation_analysis(dataset_id)

    assert_nonempty_png(result.file_path)
    assert result.matrix == expected.matrix
    assert result.strongest_pairs == expected.strongest_pairs
    assert result.matrix["Constant"]["Sales"] is None
    json.dumps(result.model_dump(mode="json"))


def test_data_and_visualization_tools_share_registry(tmp_path) -> None:
    input_path = tmp_path / "sales.csv"
    output_path = tmp_path / "charts"
    pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Sales": [10.0, 20.0],
        }
    ).to_csv(input_path, index=False)
    registry = DatasetRegistry()
    data_tools = DataTools(registry)
    analysis_tools = AnalysisTools(registry)
    visualization_tools = VisualizationTools(registry, output_path)

    loaded = data_tools.load_data(str(input_path))
    aggregate = analysis_tools.aggregate_data(
        loaded.dataset_id,
        ["Date"],
        "Sales",
    )
    chart = visualization_tools.plot_trend(
        loaded.dataset_id,
        "Date",
        "Sales",
    )

    assert aggregate.dataset_id == loaded.dataset_id
    assert chart.dataset_id == loaded.dataset_id
    assert_nonempty_png(chart.file_path)
    assert output_path.exists()
