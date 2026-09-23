"""Application composition and stdio MCP server entry point."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from src.exceptions import AnalyticsMCPError
from src.schemas import (
    AggregationResult,
    ChartResult,
    CleaningReport,
    CorrelationResult,
    DataProfile,
    DatasetInfo,
)
from src.storage import DatasetRegistry
from src.tools.analysis_tools import AnalysisTools
from src.tools.data_tools import DataTools
from src.tools.visualization_tools import VisualizationTools


ResultT = TypeVar("ResultT")


def _domain_call(operation: Callable[[], ResultT]) -> ResultT:
    """Convert expected domain failures into model-readable MCP tool errors."""

    try:
        return operation()
    except AnalyticsMCPError as error:
        payload = json.dumps(error.to_dict(), ensure_ascii=False)
        raise ToolError(payload) from None


def _create_mcp_server(
    data: DataTools,
    analysis: AnalysisTools,
    visualization: VisualizationTools,
) -> MCPServer:
    server = MCPServer(
        name="llm-analytics",
        title="LLM Analytics",
        description="Deterministic local dataset analysis tools.",
        version="0.1.0",
    )

    @server.tool(
        description=(
            "Load a local CSV or XLSX dataset and register it for analysis. "
            "Returns a dataset_id and compact schema metadata. Use this first "
            "when analyzing a new file."
        )
    )
    def load_data(file_path: str) -> DatasetInfo:
        return _domain_call(lambda: data.load_data(file_path))

    @server.tool(
        description=(
            "Inspect a loaded dataset's schema, semantic types, missing values, "
            "duplicates, numeric statistics, categorical values, and date ranges. "
            "Use after load_data and before choosing analyses."
        )
    )
    def describe_data(dataset_id: str) -> DataProfile:
        return _domain_call(lambda: data.describe_data(dataset_id))

    @server.tool(
        description=(
            "Clean a loaded dataset using deterministic rules. Can remove exact "
            "duplicates, fill safe missing values, and detect IQR outliers. "
            "Outliers are reported but not removed. Use when profiling identifies "
            "data quality issues."
        )
    )
    def clean_data(
        dataset_id: str,
        handle_missing: bool = True,
        remove_duplicates: bool = True,
        detect_outliers: bool = True,
    ) -> CleaningReport:
        return _domain_call(
            lambda: data.clean_data(
                dataset_id,
                handle_missing=handle_missing,
                remove_duplicates=remove_duplicates,
                detect_outliers=detect_outliers,
            )
        )

    @server.tool(
        description=(
            "Compute exact grouped statistics for a metric. Use to compare regions, "
            "products, categories, or other dimensions. Prefer this tool for exact "
            "numeric comparisons rather than estimating values from charts."
        )
    )
    def aggregate_data(
        dataset_id: str,
        group_by: list[str],
        metric: str,
        aggregation: str = "sum",
        sort_desc: bool = True,
        limit: int = 20,
    ) -> AggregationResult:
        return _domain_call(
            lambda: analysis.aggregate_data(
                dataset_id,
                group_by,
                metric,
                aggregation=aggregation,
                sort_desc=sort_desc,
                limit=limit,
            )
        )

    @server.tool(
        description=(
            "Create a time-series chart for a numeric metric over a date/time "
            "dimension. Returns both a PNG path and exact structured values. "
            "Use for trends over time."
        )
    )
    def plot_trend(
        dataset_id: str,
        x_column: str,
        y_column: str,
        aggregation: str = "sum",
        frequency: str | None = None,
    ) -> ChartResult:
        return _domain_call(
            lambda: visualization.plot_trend(
                dataset_id,
                x_column,
                y_column,
                aggregation=aggregation,
                frequency=frequency,
            )
        )

    @server.tool(
        description=(
            "Create a histogram for a numeric column. Returns the PNG path and "
            "exact descriptive statistics. Use to inspect numeric distributions."
        )
    )
    def plot_distribution(
        dataset_id: str,
        column: str,
        bins: int = 20,
    ) -> ChartResult:
        return _domain_call(
            lambda: visualization.plot_distribution(dataset_id, column, bins)
        )

    @server.tool(
        description=(
            "Create a categorical comparison chart using deterministic aggregation. "
            "Returns the PNG path and exact grouped values. Use to compare products, "
            "regions, or similar categories."
        )
    )
    def plot_bar(
        dataset_id: str,
        category_column: str,
        value_column: str,
        aggregation: str = "sum",
        top_n: int = 10,
    ) -> ChartResult:
        return _domain_call(
            lambda: visualization.plot_bar(
                dataset_id,
                category_column,
                value_column,
                aggregation=aggregation,
                top_n=top_n,
            )
        )

    @server.tool(
        description=(
            "Compute Pearson correlations across numeric columns and generate a "
            "heatmap. Returns the matrix, strongest pairs, and PNG path. "
            "Correlation does not imply causation."
        )
    )
    def correlation_analysis(dataset_id: str) -> CorrelationResult:
        return _domain_call(
            lambda: visualization.correlation_analysis(dataset_id)
        )

    return server


def create_server(output_dir: str | Path | None = None) -> MCPServer:
    """Create an isolated MCP application composition, primarily for tests."""

    app_registry = DatasetRegistry()
    return _create_mcp_server(
        DataTools(app_registry),
        AnalysisTools(app_registry),
        VisualizationTools(app_registry, output_dir),
    )


registry = DatasetRegistry()
data_tools = DataTools(registry)
analysis_tools = AnalysisTools(registry)
visualization_tools = VisualizationTools(registry)
mcp = _create_mcp_server(data_tools, analysis_tools, visualization_tools)


def main() -> None:
    """Run the MCP server over stdio."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
