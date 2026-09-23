"""Tests for deterministic sample generation and the end-to-end demo."""

import json
from pathlib import Path

import pandas as pd

from demo.run_demo import DEMO_CHART_NAMES, run_demo
from scripts.generate_sample_data import generate_sales_data
from src.skills import DataLoadingSkill
from src.storage import DatasetRegistry


EXPECTED_COLUMNS = [
    "Date",
    "Product",
    "Region",
    "Sales",
    "Quantity",
    "Profit",
]


def test_sample_generation_is_deterministic_and_has_expected_quality_issues(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    generate_sales_data(first_path)
    generate_sales_data(second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    dataframe = pd.read_csv(first_path)
    assert list(dataframe.columns) == EXPECTED_COLUMNS
    assert 180 <= len(dataframe) <= 200
    assert dataframe.duplicated().sum() == 3
    assert dataframe[["Sales", "Quantity", "Profit"]].isna().sum().sum() == 4
    assert dataframe["Date"].min().startswith("2023-")
    assert dataframe["Date"].max().startswith("2024-")

    outlier_columns = []
    for column in ["Sales", "Profit"]:
        values = dataframe[column].dropna()
        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        upper_bound = third_quartile + 1.5 * (third_quartile - first_quartile)
        if int((values > upper_bound).sum()) > 0:
            outlier_columns.append(column)
    assert outlier_columns == ["Sales", "Profit"]


def test_generated_dataset_loads_through_data_loading_skill(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "sales_data.csv"
    generated = generate_sales_data(data_path)
    registry = DatasetRegistry()

    result = DataLoadingSkill(registry).run(str(data_path))

    assert result.rows == len(generated)
    assert registry.exists(result.dataset_id)
    stored = registry.get(result.dataset_id)
    assert pd.api.types.is_datetime64_any_dtype(stored["Date"].dtype)


def test_full_demo_creates_json_summary_and_four_nonempty_charts(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "sales_data.csv"
    output_dir = tmp_path / "output"
    results_path = tmp_path / "demo_results.json"
    generate_sales_data(data_path)

    summary = run_demo(data_path, output_dir, results_path)

    persisted = json.loads(results_path.read_text(encoding="utf-8"))
    assert persisted == summary
    assert summary["original_row_count"] == 195
    assert summary["duplicate_count"] == 3
    assert set(summary["chart_paths"]) == set(DEMO_CHART_NAMES)
    assert len(list(output_dir.glob("*.png"))) == 4

    for chart_name, file_name in DEMO_CHART_NAMES.items():
        chart_path = output_dir / file_name
        assert Path(summary["chart_paths"][chart_name]) == chart_path.resolve()
        assert chart_path.exists()
        assert chart_path.stat().st_size > 0
