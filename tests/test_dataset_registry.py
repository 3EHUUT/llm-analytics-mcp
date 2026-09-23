"""Tests for in-memory dataset storage."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.exceptions import DatasetNotFoundError
from src.storage import DatasetRegistry


def test_add_registers_dataframe_with_unique_dataset_id() -> None:
    registry = DatasetRegistry()
    dataframe = pd.DataFrame({"Sales": [100.0, 125.0]})

    first_id = registry.add(dataframe, {"source": "first.csv"})
    second_id = registry.add(dataframe, {"source": "second.csv"})

    assert first_id.startswith("ds_")
    assert second_id.startswith("ds_")
    assert first_id != second_id
    assert registry.exists(first_id)
    assert registry.exists(second_id)


def test_registry_owns_dataframe_and_metadata_copies() -> None:
    registry = DatasetRegistry()
    dataframe = pd.DataFrame({"Region": ["North"], "Sales": [100.0]})
    metadata = {"source": "sales.csv", "tags": ["demo"]}

    dataset_id = registry.add(dataframe, metadata)
    dataframe.loc[0, "Sales"] = 999.0
    metadata["tags"].append("changed")

    retrieved = registry.get(dataset_id)
    retrieved.loc[0, "Sales"] = 500.0
    retrieved_metadata = registry.get_metadata(dataset_id)
    retrieved_metadata["tags"].append("also-changed")

    assert registry.get(dataset_id).loc[0, "Sales"] == 100.0
    assert registry.get_metadata(dataset_id) == {
        "source": "sales.csv",
        "tags": ["demo"],
    }


def test_replace_updates_dataframe_and_preserves_metadata() -> None:
    registry = DatasetRegistry()
    dataset_id = registry.add(
        pd.DataFrame({"Sales": [100.0]}),
        {"source": "sales.csv"},
    )
    replacement = pd.DataFrame({"Sales": [125.0, 150.0]})

    registry.replace(dataset_id, replacement)

    assert_frame_equal(registry.get(dataset_id), replacement)
    assert registry.get_metadata(dataset_id) == {"source": "sales.csv"}


@pytest.mark.parametrize("operation", ["get", "replace", "get_metadata"])
def test_unknown_dataset_id_raises_domain_error(operation: str) -> None:
    registry = DatasetRegistry()

    with pytest.raises(DatasetNotFoundError) as error:
        if operation == "replace":
            registry.replace("ds_missing", pd.DataFrame())
        else:
            getattr(registry, operation)("ds_missing")

    assert error.value.to_dict() == {
        "error": "dataset_not_found",
        "message": "Dataset 'ds_missing' does not exist.",
        "dataset_id": "ds_missing",
    }


def test_registry_rejects_non_dataframe_values() -> None:
    registry = DatasetRegistry()

    with pytest.raises(TypeError, match="pandas DataFrame"):
        registry.add([{"Sales": 100.0}])  # type: ignore[arg-type]
