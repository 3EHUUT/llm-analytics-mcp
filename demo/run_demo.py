"""Run the sample dataset through the real public tool-layer classes."""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR  # noqa: E402
from src.storage import DatasetRegistry  # noqa: E402
from src.tools.analysis_tools import AnalysisTools  # noqa: E402
from src.tools.data_tools import DataTools  # noqa: E402
from src.tools.visualization_tools import VisualizationTools  # noqa: E402


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sales_data.csv"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "demo" / "demo_results.json"
DEMO_CHART_NAMES = {
    "sales_trend": "demo_sales_trend.png",
    "region_sales": "demo_region_sales.png",
    "profit_distribution": "demo_profit_distribution.png",
    "correlation_heatmap": "demo_correlation_heatmap.png",
}


def _replace_demo_chart(generated_path: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Path(generated_path).replace(destination)
    return destination.resolve()


def _result_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def run_demo(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
) -> dict[str, object]:
    """Execute the deterministic backend workflow and write its summary."""

    chart_directory = Path(output_dir)
    chart_directory.mkdir(parents=True, exist_ok=True)
    for existing_chart in chart_directory.glob("demo_*.png"):
        existing_chart.unlink()

    registry = DatasetRegistry()
    data_tools = DataTools(registry)
    analysis_tools = AnalysisTools(registry)
    visualization_tools = VisualizationTools(registry, chart_directory)

    loaded = data_tools.load_data(str(data_path))
    profile = data_tools.describe_data(loaded.dataset_id)
    missing_summary = {
        column.name: {
            "count": column.null_count,
            "percentage": column.missing_percentage,
        }
        for column in profile.columns
        if column.null_count
    }
    cleaning = data_tools.clean_data(loaded.dataset_id)

    region_sales = analysis_tools.aggregate_data(
        loaded.dataset_id,
        ["Region"],
        "Sales",
    )
    product_profit = analysis_tools.aggregate_data(
        loaded.dataset_id,
        ["Product"],
        "Profit",
    )

    trend = visualization_tools.plot_trend(
        loaded.dataset_id,
        "Date",
        "Sales",
        frequency="M",
    )
    bar = visualization_tools.plot_bar(
        loaded.dataset_id,
        "Region",
        "Sales",
    )
    distribution = visualization_tools.plot_distribution(
        loaded.dataset_id,
        "Profit",
    )
    correlation = visualization_tools.correlation_analysis(loaded.dataset_id)

    generated_charts = {
        "sales_trend": trend.file_path,
        "region_sales": bar.file_path,
        "profit_distribution": distribution.file_path,
        "correlation_heatmap": correlation.file_path,
    }
    chart_paths = {
        chart_name: _result_path(
            _replace_demo_chart(
                generated_charts[chart_name],
                chart_directory / file_name,
            )
        )
        for chart_name, file_name in DEMO_CHART_NAMES.items()
    }

    summary: dict[str, object] = {
        "dataset_id": loaded.dataset_id,
        "original_row_count": profile.row_count,
        "duplicate_count": profile.duplicate_row_count,
        "missing_value_summary": missing_summary,
        "cleaning_result": cleaning.model_dump(mode="json"),
        "top_regions_by_sales": region_sales.rows,
        "top_products_by_profit": product_profit.rows,
        "trend_summary": trend.summary,
        "correlation_strongest_pairs": [
            pair.model_dump(mode="json")
            for pair in correlation.strongest_pairs
        ],
        "chart_paths": chart_paths,
    }

    destination = Path(results_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Run the default demo and print where its summary was written."""

    run_demo()
    print(f"Demo complete. Results written to {DEFAULT_RESULTS_PATH}.")


if __name__ == "__main__":
    main()
