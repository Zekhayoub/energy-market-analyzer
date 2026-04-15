"""Shared test helpers for synthetic data generation.

All test helpers are centralized here. Tests import from
this module instead of creating their own data. This ensures
consistency across all test files.

This file will grow as new parts of the project add their
own helpers (e.g., create_synthetic_multimodal_data for HMM,
create_obvious_outliers for IF).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_synthetic_price_series(
    n_hours: int = 1000,
    regime_type: str = "stable",
    random_state: int = 42,
) -> pd.Series:
    """Generate a synthetic price series for testing.

    Args:
        n_hours: Number of hourly data points.
        regime_type: One of "stable", "volatile", "crisis".
            - "stable": mean=50, std=5
            - "volatile": mean=100, std=25
            - "crisis": mean=250, std=80
        random_state: Random seed for reproducibility.

    Returns:
        pd.Series with DatetimeIndex (hourly, UTC) and
        price values in EUR/MWh.
    """
    rng = np.random.default_rng(random_state)

    params = {
        "stable": (50, 5),
        "volatile": (100, 25),
        "crisis": (250, 80),
    }
    if regime_type not in params:
        raise ValueError(f"regime_type must be one of {list(params.keys())}")

    mean, std = params[regime_type]
    prices = rng.normal(mean, std, n_hours)
    prices = np.maximum(prices, -50)  # Negative prices are possible but bounded

    index = pd.date_range(
        start="2020-01-01",
        periods=n_hours,
        freq="h",
        tz=None,  # UTC without timezone info (project convention)
    )

    return pd.Series(prices, index=index, name="price_eur_mwh")


def create_synthetic_master_dataframe(
    n_days: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a minimal master DataFrame for testing.

    Creates a DataFrame with the basic columns needed by
    the feature engineering pipeline. Not all columns are
    realistic — this is for testing logic, not results.

    Args:
        n_days: Number of days (each day = 24 rows).
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame with hourly index and basic columns.
    """
    rng = np.random.default_rng(random_state)
    n_hours = n_days * 24

    index = pd.date_range(
        start="2020-01-01",
        periods=n_hours,
        freq="h",
    )

    df = pd.DataFrame(
        {
            "price_eur_mwh": rng.normal(50, 15, n_hours),
            "load_mw": rng.normal(10000, 2000, n_hours),
            "gas_generation_mw": rng.normal(3000, 500, n_hours),
            "total_generation_mw": rng.normal(12000, 2000, n_hours),
            "renewable_generation_mw": rng.normal(4000, 1500, n_hours),
            "nuclear_generation_mw": rng.normal(4000, 500, n_hours),
            "net_import_total_mw": rng.normal(500, 1000, n_hours),
            "ttf_price": rng.normal(25, 5, n_hours),
            "carbon_price": rng.normal(60, 10, n_hours),
            "temperature_2m": rng.normal(12, 8, n_hours),
            "wind_speed_100m": rng.normal(8, 4, n_hours),
            "shortwave_radiation": np.maximum(rng.normal(200, 150, n_hours), 0),
            "cloud_cover": rng.uniform(0, 100, n_hours),
        },
        index=index,
    )

    return df
