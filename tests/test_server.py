"""In-process tests for the MCP server wiring."""

import json
from pathlib import Path

import pandas as pd
import pytest
from mcp import Client

from src.server import create_server


EXPECTED_TOOLS = {
    "load_data",
    "describe_data",
    "clean_data",
    "aggregate_data",
    "plot_trend",
    "plot_distribution",
    "plot_bar",
    "correlation_analysis",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mcp_server(tmp_path):
    return create_server(output_dir=tmp_path / "charts")


@pytest.fixture
async def client(mcp_server):
    async with Client(mcp_server, raise_exceptions=True) as connected:
        yield connected


@pytest.fixture
def sales_file(tmp_path) -> Path:
    file_path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
            "Region": ["North", "South", "North", "North"],
            "Sales": [10.0, 20.0, None, None],
            "Quantity": [1, 2, 3, 3],
            "Profit": [2.0, 4.0, 6.0, 6.0],
        }
    ).to_csv(file_path, index=False)
    return file_path


async def load_dataset(client: Client, sales_file: Path) -> dict:
    result = await client.call_tool(
        "load_data",
        {"file_path": str(sales_file)},
    )
    assert not result.is_error
    assert result.structured_content is not None
    return result.structured_content


@pytest.mark.anyio
async def test_server_exposes_exactly_expected_tools(client: Client) -> None:
    result = await client.list_tools()

    assert {tool.name for tool in result.tools} == EXPECTED_TOOLS


@pytest.mark.anyio
async def test_load_and_describe_share_mcp_state(
    client: Client,
    sales_file: Path,
) -> None:
    loaded = await load_dataset(client, sales_file)

    described = await client.call_tool(
        "describe_data",
        {"dataset_id": loaded["dataset_id"]},
    )

    assert not described.is_error
    assert described.structured_content is not None
    assert described.structured_content["dataset_id"] == loaded["dataset_id"]
    assert described.structured_content["row_count"] == 4
    json.dumps(described.structured_content)


@pytest.mark.anyio
async def test_clean_and_aggregate_work_after_mcp_load(
    client: Client,
    sales_file: Path,
) -> None:
    loaded = await load_dataset(client, sales_file)
    dataset_id = loaded["dataset_id"]

    cleaned = await client.call_tool("clean_data", {"dataset_id": dataset_id})
    aggregated = await client.call_tool(
        "aggregate_data",
        {
            "dataset_id": dataset_id,
            "group_by": ["Region"],
            "metric": "Sales",
        },
    )

    assert not cleaned.is_error
    assert cleaned.structured_content["duplicates_removed"] == 1
    assert not aggregated.is_error
    assert aggregated.structured_content["dataset_id"] == dataset_id
    assert aggregated.structured_content["rows"] == [
        {"Region": "North", "Sales": 25.0},
        {"Region": "South", "Sales": 20.0},
    ]


@pytest.mark.anyio
async def test_visualization_and_correlation_work_through_mcp(
    client: Client,
    sales_file: Path,
) -> None:
    loaded = await load_dataset(client, sales_file)
    dataset_id = loaded["dataset_id"]

    chart = await client.call_tool(
        "plot_distribution",
        {"dataset_id": dataset_id, "column": "Profit", "bins": 3},
    )
    correlation = await client.call_tool(
        "correlation_analysis",
        {"dataset_id": dataset_id},
    )

    assert not chart.is_error
    assert_nonempty_png(chart.structured_content["file_path"])
    assert not correlation.is_error
    assert_nonempty_png(correlation.structured_content["file_path"])
    assert correlation.structured_content["matrix"]
    json.dumps(chart.structured_content)
    json.dumps(correlation.structured_content)


@pytest.mark.anyio
async def test_domain_error_is_a_clean_mcp_tool_failure(client: Client) -> None:
    result = await client.call_tool(
        "describe_data",
        {"dataset_id": "ds_missing"},
    )

    assert result.is_error
    assert result.structured_content is None
    error_text = result.content[0].text
    assert "dataset_not_found" in error_text
    assert "ds_missing" in error_text
    assert "Traceback" not in error_text


def assert_nonempty_png(file_path: str) -> None:
    path = Path(file_path)
    assert path.exists()
    assert path.stat().st_size > 0
