"""Ingestion pipeline nodes.

Each function fetches data from one source (ENTSO-E, yfinance,
or Open-Meteo), cleans it, and returns a DataFrame ready for
merging.

All timestamps are converted to UTC. No timezone-aware
DatetimeIndex — use naive UTC everywhere for consistency.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=FutureWarning, module="entsoe")


def _check_hourly_continuity(
    df: pd.DataFrame | pd.Series,
    label: str,
) -> None:
    """Log a warning if the hourly index has gaps.

    is_monotonic_increasing only checks order, not
    completeness. A 3-week gap passes silently. This helper
    detects missing hours so downstream rolling windows don't
    silently mix non-contiguous dates.
    """
    expected_len = len(pd.date_range(df.index[0], df.index[-1], freq="h"))
    actual_len = len(df)
    if actual_len < expected_len:
        gap_hours = expected_len - actual_len
        logger.warning(
            "%s index has %d missing hours out of %d expected (%.1f%% gap)",
            label,
            gap_hours,
            expected_len,
            100 * gap_hours / expected_len,
        )


def _resample_to_hourly(
    df: pd.DataFrame | pd.Series,
    label: str,
) -> pd.DataFrame | pd.Series:
    """Resample sub-hourly data (e.g. 15min) to hourly.

    ENTSO-E returns 15-min resolution for load and
    generation in some countries (BE, DE). The project convention
    is hourly — resample here to avoid schema mismatches in the
    master DataFrame merge.
    """
    freq = pd.infer_freq(df.index)
    if freq and "15" in str(freq):
        logger.info("Resampling %s from 15min to 1h", label)
        return df.resample("h").mean()
    return df


def _fetch_with_pagination(
    query_fn: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame | pd.Series | None],
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_retries: int = 3,
    backoff_factor: int = 2,
    label: str = "data",
) -> pd.DataFrame | pd.Series:
    """Fetch data from ENTSO-E with yearly pagination and retry.

    ENTSO-E returns errors for requests spanning more than ~1 year.
    This helper splits the range into ~11-month chunks, retries on
    failure with exponential backoff, and concatenates results.

    Args:
        query_fn: Callable(start, end) -> DataFrame/Series.
            Must accept two pd.Timestamp args with tz.
        start: Start of period (tz-aware).
        end: End of period (tz-aware).
        max_retries: Max retry attempts per chunk.
        backoff_factor: Exponential backoff base (seconds).
        label: Label for log messages.

    Returns:
        Concatenated, sorted, deduplicated result in UTC naive.

    Raises:
        ValueError: If no data returned for the entire period.
        RuntimeError: If any chunk fails after all retries.
    """
    all_data = []
    current_start = start

    while current_start < end:
        # Use 11-month chunks instead of calendar year.
        # A calendar year jump (year+1) produces 366 days on leap years,
        # which can exceed the ENTSO-E API limit (~365 days).
        current_end = min(
            current_start + pd.DateOffset(months=11),
            end,
        )

        logger.info(
            "Fetching %s: %s to %s",
            label,
            current_start.date(),
            current_end.date(),
        )

        for attempt in range(1, max_retries + 1):
            try:
                result = query_fn(current_start, current_end)
                if result is not None and len(result) > 0:
                    all_data.append(result)
                    break
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to fetch {label} for "
                        f"{current_start.date()}-{current_end.date()} "
                        f"after {max_retries} attempts: {e}"
                    ) from e
                else:
                    wait = backoff_factor**attempt
                    logger.info("Retry %d/%d in %ds: %s", attempt, max_retries, wait, e)
                    time.sleep(wait)

        current_start = current_end

    if not all_data:
        raise ValueError(f"No {label} returned between {start.date()} and {end.date()}")

    combined = pd.concat(all_data, copy=False)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    # Convert to UTC naive
    if combined.index.tz is not None:
        combined.index = combined.index.tz_convert("UTC").tz_localize(None)

    return combined


def fetch_day_ahead_prices(
    params: dict[str, Any],
    api_key: str,
) -> pd.DataFrame:
    """Fetch Belgian day-ahead electricity prices from ENTSO-E.

    Args:
        params: Ingestion parameters from parameters.yml.
        api_key: ENTSO-E API key from credentials.yml.

    Returns:
        pd.DataFrame with columns:
            - price_eur_mwh: float (day-ahead price)
        Index: DatetimeIndex, hourly, naive UTC.

    Raises:
        ValueError: If API returns no data for the entire period.
    """
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_key)

    country = params["country"]
    start = pd.Timestamp(params["start_date"], tz="Europe/Brussels")
    end = pd.Timestamp(params["end_date"], tz="Europe/Brussels")

    result = _fetch_with_pagination(
        query_fn=lambda s, e: client.query_day_ahead_prices(country, start=s, end=e),
        start=start,
        end=end,
        max_retries=params.get("retry_max_attempts", 3),
        backoff_factor=params.get("retry_backoff_factor", 2),
        label=f"day-ahead prices {country}",
    )

    # Structure as DataFrame
    df = pd.DataFrame({"price_eur_mwh": result.values}, index=result.index)
    df.index.name = "timestamp"

    assert df.index.is_monotonic_increasing, "Price index not monotonic after UTC conversion"
    assert not df["price_eur_mwh"].isna().all(), "All prices are NaN"

    _check_hourly_continuity(df, f"day-ahead prices {country}")

    logger.info(
        "Fetched %d hourly prices for %s (%s to %s)",
        len(df),
        country,
        df.index[0],
        df.index[-1],
    )

    return df
