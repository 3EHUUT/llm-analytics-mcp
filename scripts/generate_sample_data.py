"""Generate the deterministic sales dataset used by the demo."""

from datetime import date
from pathlib import Path
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "sales_data.csv"
RANDOM_SEED = 20240715
BASE_ROW_COUNT = 192

PRODUCTS = {
    "Laptop": {"unit_price": 920.0, "quantity_mean": 5.0, "margin": 0.14},
    "Monitor": {"unit_price": 310.0, "quantity_mean": 5.0, "margin": 0.20},
    "Keyboard": {"unit_price": 72.0, "quantity_mean": 5.0, "margin": 0.28},
    "Mouse": {"unit_price": 44.0, "quantity_mean": 5.0, "margin": 0.32},
    "Headphones": {"unit_price": 135.0, "quantity_mean": 5.0, "margin": 0.24},
}
REGION_FACTORS = {
    "North": 1.05,
    "South": 0.94,
    "East": 1.00,
    "West": 1.10,
}
MONTHLY_PRODUCTS = [
    "Laptop",
    "Monitor",
    "Keyboard",
    "Mouse",
    "Headphones",
    "Laptop",
    "Monitor",
    "Headphones",
]
MONTHLY_DAYS = [2, 5, 9, 12, 16, 19, 23, 27]


def generate_sales_data(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Generate, save, and return a reproducible sales DataFrame."""

    rng = random.Random(RANDOM_SEED)
    region_names = list(REGION_FACTORS)
    rows: list[dict[str, object]] = []

    for index in range(BASE_ROW_COUNT):
        month_index, month_slot = divmod(index, len(MONTHLY_PRODUCTS))
        row_date = date(
            2023 + month_index // 12,
            1 + month_index % 12,
            MONTHLY_DAYS[month_slot],
        )
        product = MONTHLY_PRODUCTS[month_slot]
        region = region_names[(month_slot + month_index) % len(region_names)]
        product_profile = PRODUCTS[product]

        quantity = max(
            1,
            round(
                rng.gauss(
                    product_profile["quantity_mean"],
                    product_profile["quantity_mean"] * 0.32,
                )
            ),
        )
        growth_factor = 1.0 + (0.14 * month_index / 23)
        demand_noise = rng.uniform(0.90, 1.10)
        sales = (
            product_profile["unit_price"]
            * quantity
            * REGION_FACTORS[region]
            * growth_factor
            * demand_noise
        )
        profit_noise = rng.gauss(0.0, sales * 0.045)
        profit = max(1.0, sales * product_profile["margin"] + profit_noise)

        rows.append(
            {
                "Date": row_date,
                "Product": product,
                "Region": region,
                "Sales": round(sales, 2),
                "Quantity": quantity,
                "Profit": round(profit, 2),
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    dataframe["Quantity"] = dataframe["Quantity"].astype("Int64")

    for row_index, multiplier in [(21, 4.2), (96, 4.8), (164, 4.5)]:
        dataframe.loc[row_index, "Sales"] = round(
            dataframe.loc[row_index, "Sales"] * multiplier,
            2,
        )
        dataframe.loc[row_index, "Profit"] = round(
            dataframe.loc[row_index, "Profit"] * (multiplier * 0.90),
            2,
        )

    dataframe.loc[34, "Sales"] = pd.NA
    dataframe.loc[77, "Profit"] = pd.NA
    dataframe.loc[119, "Quantity"] = pd.NA
    dataframe.loc[151, "Sales"] = pd.NA

    duplicate_rows = dataframe.iloc[[12, 83, 171]].copy()
    dataframe = pd.concat([dataframe, duplicate_rows], ignore_index=True)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(
        destination,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.2f",
        lineterminator="\n",
    )
    return dataframe


def main() -> None:
    """Generate the default sample file and print a compact summary."""

    dataframe = generate_sales_data()
    print(
        f"Generated {len(dataframe)} rows at {DEFAULT_OUTPUT_PATH} "
        f"({int(dataframe.duplicated().sum())} duplicate rows)."
    )


if __name__ == "__main__":
    main()
