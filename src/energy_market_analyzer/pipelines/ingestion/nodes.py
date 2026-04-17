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

import numpy as np
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


def fetch_generation_by_tech(
    params: dict[str, Any],
    api_key: str,
) -> pd.DataFrame:
    """Fetch electricity generation by technology from ENTSO-E.

    Returns generation in MW for each technology. Technologies
    that don't exist in the country (e.g., geothermal in Belgium)
    are returned as columns with 0 values.

    Args:
        params: Ingestion parameters (country, dates, retry_*,
            expected_generation_techs).
        api_key: ENTSO-E API key.

    Returns:
        pd.DataFrame with one column per technology (MW).
        Index: DatetimeIndex, hourly, naive UTC.
    """
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_key)

    country = params["country"]
    start = pd.Timestamp(params["start_date"], tz="Europe/Brussels")
    end = pd.Timestamp(params["end_date"], tz="Europe/Brussels")

    expected_techs = params["expected_generation_techs"]

    result = _fetch_with_pagination(
        query_fn=lambda s, e: client.query_generation(country, start=s, end=e),
        start=start,
        end=end,
        max_retries=params.get("retry_max_attempts", 3),
        backoff_factor=params.get("retry_backoff_factor", 2),
        label=f"generation {country}",
    )

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    result.index.name = "timestamp"

    result = _resample_to_hourly(result, f"generation {country}")

    assert result.index.is_monotonic_increasing, "Generation index not monotonic"

    _check_hourly_continuity(result, f"generation {country}")

    # --- FIX: missing technologies ---
    # Belgium has no geothermal, France had no offshore until recently.
    # Add missing columns with 0 to ensure all countries share the
    # same schema. NaN would be incorrect: 0 MW of geothermal is
    # a fact, not missing data.
    for tech in expected_techs:
        if tech not in result.columns:
            result[tech] = 0.0
            logger.info("Technology '%s' not found for %s — filled with 0", tech, country)

    # Drop unexpected columns not in expected_techs.
    # The API may return "Other" or "Waste" categories that vary
    # between countries — filter to guarantee a stable schema.
    result = result[expected_techs]

    # result = result.reindex(columns=expected_techs, fill_value=0.0)

    logger.info(
        "Fetched generation for %s: %d rows, %d technologies",
        country,
        len(result),
        len(result.columns),
    )

    return result


def fetch_load(
    params: dict[str, Any],
    api_key: str,
) -> pd.DataFrame:
    """Fetch electricity demand (load) from ENTSO-E.

    Args:
        params: Ingestion parameters.
        api_key: ENTSO-E API key.

    Returns:
        pd.DataFrame with column 'load_mw' (demand in MW).
        Index: DatetimeIndex, hourly, naive UTC.
    """
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_key)

    country = params["country"]
    start = pd.Timestamp(params["start_date"], tz="Europe/Brussels")
    end = pd.Timestamp(params["end_date"], tz="Europe/Brussels")

    result = _fetch_with_pagination(
        query_fn=lambda s, e: client.query_load(country, start=s, end=e),
        start=start,
        end=end,
        max_retries=params.get("retry_max_attempts", 3),
        backoff_factor=params.get("retry_backoff_factor", 2),
        label=f"load {country}",
    )

    result = _resample_to_hourly(result, f"load {country}")

    # Some countries return multiple load columns (e.g.,
    # consumption profiles). Sum them for total demand instead of
    # blindly taking the first column with iloc[:, 0].
    if isinstance(result, pd.Series):
        df = pd.DataFrame({"load_mw": result.values}, index=result.index)
    else:
        df = pd.DataFrame({"load_mw": result.sum(axis=1).values}, index=result.index)

    df.index.name = "timestamp"

    assert df.index.is_monotonic_increasing, "Load index not monotonic"
    assert not df["load_mw"].isna().all(), "All load values are NaN"

    # Check hourly continuity
    _check_hourly_continuity(df, f"load {country}")

    logger.info("Fetched %d hourly load values for %s", len(df), country)

    return df


def fetch_crossborder_flows(
    params: dict[str, Any],
    api_key: str,
) -> pd.DataFrame:
    """Fetch BIDIRECTIONAL cross-border flows from ENTSO-E.

    For each neighbor, fetches flows in BOTH directions:
        - neighbor -> country (imports)
        - country -> neighbor (exports)

    Then computes net flows:
        net_flow_X = flow_X_to_country - flow_country_to_X

    Positive net flow = country is importing (net importer).
    Negative net flow = country is exporting (net exporter).

    For Belgium with neighbors [FR, NL, DE_LU]:
        6 API calls (3 import + 3 export)
        3 net flow columns + 1 total net import column

    Args:
        params: Ingestion parameters with 'neighbors' list.
        api_key: ENTSO-E API key.

    Returns:
        pd.DataFrame with columns:
            - flow_fr_to_be, flow_be_to_fr (raw flows)
            - net_flow_fr (import - export)
            - net_import_total (sum of all net flows)
        Index: DatetimeIndex, hourly, naive UTC.
    """
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_key)

    country = params["country"]
    neighbors = params["neighbors"]
    start = pd.Timestamp(params["start_date"], tz="Europe/Brussels")
    end = pd.Timestamp(params["end_date"], tz="Europe/Brussels")
    max_retries = params.get("retry_max_attempts", 3)
    backoff = params.get("retry_backoff_factor", 2)

    def _safe_fetch_flow(from_c: str, to_c: str) -> pd.Series | None:
        """Fetch a single directional flow, return None on failure."""
        try:
            return _fetch_with_pagination(
                query_fn=lambda s, e: client.query_crossborder_flows(
                    from_c,
                    to_c,
                    start=s,
                    end=e,
                ),
                start=start,
                end=end,
                max_retries=max_retries,
                backoff_factor=backoff,
                label=f"flow {from_c}->{to_c}",
            )
        except (ValueError, RuntimeError):
            logger.warning("No flow data for %s->%s", from_c, to_c)
            return None

    # --- Fetch bidirectional flows ---
    flows_dict: dict[str, pd.Series] = {}

    for neighbor in neighbors:
        n_lower = neighbor.lower().replace("_", "")
        c_lower = country.lower()

        import_col = f"flow_{n_lower}_to_{c_lower}"
        export_col = f"flow_{c_lower}_to_{n_lower}"

        # Import: neighbor -> country
        logger.info("Fetching flow %s -> %s (import)", neighbor, country)
        import_flow = _safe_fetch_flow(neighbor, country)
        if import_flow is not None:
            flows_dict[import_col] = import_flow
        else:
            logger.warning("Missing import flow %s->%s, inserting NaN", neighbor, country)

        # Export: country -> neighbor
        logger.info("Fetching flow %s -> %s (export)", country, neighbor)
        export_flow = _safe_fetch_flow(country, neighbor)
        if export_flow is not None:
            flows_dict[export_col] = export_flow
        else:
            logger.warning("Missing export flow %s->%s, inserting NaN", country, neighbor)

    if not flows_dict:
        raise ValueError(f"No cross-border flow data for {country}")

    df = pd.DataFrame(flows_dict)

    df = _resample_to_hourly(df, f"flows {country}")

    # --- Compute net flows ---
    for neighbor in neighbors:
        n_lower = neighbor.lower().replace("_", "")
        c_lower = country.lower()

        import_col = f"flow_{n_lower}_to_{c_lower}"
        export_col = f"flow_{c_lower}_to_{n_lower}"

        if import_col in df.columns and export_col in df.columns:
            df[f"net_flow_{n_lower}"] = df[import_col].sub(
                df[export_col],
                fill_value=0,
            )
        elif import_col in df.columns:
            df[f"net_flow_{n_lower}"] = df[import_col]
        elif export_col in df.columns:
            df[f"net_flow_{n_lower}"] = -df[export_col]
        else:
            df[f"net_flow_{n_lower}"] = np.nan

    net_flow_cols = [c for c in df.columns if c.startswith("net_flow_")]
    if net_flow_cols:
        df["net_import_total"] = df[net_flow_cols].sum(axis=1, min_count=1)

    df.index.name = "timestamp"

    assert df.index.is_monotonic_increasing, "Flows index not monotonic"

    _check_hourly_continuity(df, f"flows {country}")

    logger.info(
        "Fetched %d bidirectional flows for %s (%d neighbors, %d API calls)",
        len(df),
        country,
        len(neighbors),
        len(neighbors) * 2,
    )

    return df


def fetch_gas_price_ttf(
    params: dict[str, Any],
) -> pd.DataFrame:
    """Fetch TTF natural gas prices from Yahoo Finance.

    TTF (Title Transfer Facility) is the European benchmark for
    gas prices. It drives the electricity price via the merit order.

    Args:
        params: Ingestion parameters (start_date, end_date).

    Returns:
        pd.DataFrame with column 'ttf_close' (EUR/MWh).
        Index: DatetimeIndex, DAILY, naive UTC (midnight-aligned).
    """
    import yfinance as yf

    start = str(params["start_date"])

    end = (pd.Timestamp(params["end_date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Fetching TTF gas prices from yfinance: %s to %s", start, end)

    ticker = yf.Ticker("TTF=F")

    max_retries = params.get("retry_max_attempts", 3)
    backoff = params.get("retry_backoff_factor", 2)

    for attempt in range(1, max_retries + 1):
        df = ticker.history(start=start, end=end, auto_adjust=False)
        if df is not None and len(df) > 0:
            break
        if attempt < max_retries:
            wait = backoff**attempt
            logger.info("yfinance TTF retry %d/%d in %ds", attempt, max_retries, wait)
            time.sleep(wait)
    else:
        raise ValueError(
            "No TTF data returned from yfinance after "
            f"{max_retries} attempts. Ticker TTF=F may have changed."
        )

    if "Close" not in df.columns:
        raise ValueError(
            f"Expected 'Close' column in yfinance response. " f"Got columns: {df.columns.tolist()}."
        )

    result = pd.DataFrame({"ttf_close": df["Close"].values}, index=df.index)

    if result.index.tz is not None:
        result.index = result.index.tz_localize(None).normalize()

    result.index.name = "timestamp"

    assert not result["ttf_close"].isna().all(), "All TTF prices are NaN"

    logger.info("Fetched %d daily TTF prices", len(result))

    return result
