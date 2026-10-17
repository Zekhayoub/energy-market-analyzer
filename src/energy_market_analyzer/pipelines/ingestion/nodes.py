"""Ingestion pipeline nodes.

Each source is downloaded once and kept as published (data/01_raw/). The
hourly master table (data/03_primary/) and its quality report are then
built from these raw tables, without any network call.

Time conventions (conventions.md, V3-1):
- every instant is a DatetimeIndex in UTC, with its time zone;
- delivery days are calendar days in the market time zone (Brussels);
- daily commodity prices are indexed by `trading_date`, a calendar date.

The master table keeps realized values. The 11:00 rule (decisions.md, §6)
is applied when features are built, not here.
"""

from __future__ import annotations

import functools
import io
import logging
import re
import time
import warnings
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from pydantic import SecretStr
from yfinance.exceptions import YFRateLimitError

logger = logging.getLogger(__name__)

# --- Index and column names ---
TIMESTAMP = "timestamp"
TRADING_DATE = "trading_date"
PRICE = "price_eur_mwh"
LOAD = "load_mw"
LOAD_FORECAST = "load_forecast_mw"
TTF = "ttf_eur_mwh"
EUA = "eua_eur_t"
NET_IMPORT_TOTAL = "net_import_total"
WEATHER_VARIABLES = ("temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover")
UTC_ZONE = "UTC"
DATE_FORMAT = "%Y-%m-%d"
ONE_DAY = pd.Timedelta(days=1)

# --- Source formats ---
ENTSOE_ACTUAL_LOAD = "Actual Load"
ENTSOE_FORECASTED_LOAD = "Forecasted Load"
ENTSOE_ACTUAL_AGGREGATED = "Actual Aggregated"
ENTSOE_DAY_AHEAD_PROCESS = "A01"
YAHOO_CLOSE = "Close"
YAHOO_DAILY = "1d"
EEX_DATE = "Date"
EEX_CONTRACT = "Contract"
EEX_STATUS = "Status"
EEX_SUCCESSFUL = "successful"
EEX_PRICE_PREFIX = "Auction Price"
EXCEL_SUFFIXES = (".xlsx", ".xls")
WEEKEND_FIRST_DAY = 5  # Saturday in pandas dayofweek (Monday = 0)
# EEX workbooks carry no default cell style: openpyxl warns about it on every file.
EEX_HARMLESS_WARNING = "Workbook contains no default style"
YEAR_PATTERN = re.compile(r"20\d{2}")
OPEN_METEO_WIND_UNIT = "ms"

# --- Market facts ---
QUARTER_HOURS_PER_HOUR = 4
HOURLY_UNITS_PER_HOUR = 1

# --- Network ---
HTTP_ERROR_STATUS = 400
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
ERROR_BODY_CHARS = 300
# Network errors of requests and of curl_cffi (used by yfinance) both derive from OSError.
NETWORK_ERRORS = (OSError, YFRateLimitError)

# --- Secrets ---
# entsoe-py sends the key as the `securityToken` URL parameter: any error message
# that quotes the URL quotes the key.
SECURITY_TOKEN = re.compile(r"(securityToken=)[^&\s'\"]+")
REDACTED_TOKEN = r"\1***"
# entsoe-py logs its request parameters (key included) at DEBUG level, and urllib3
# logs the URL. These loggers never go below INFO, whatever the logging setup.
LOGGERS_THAT_PRINT_THE_KEY = ("entsoe", "urllib3")
for _logger_name in LOGGERS_THAT_PRINT_THE_KEY:
    logging.getLogger(_logger_name).setLevel(logging.INFO)

EntsoeQuery = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame | pd.Series]


class NoDataError(ValueError):
    """A source returned no data at all for the requested period."""


# ==================================================
# Shared helpers: secrets, time, network
# ==================================================


def redact_secrets(text: str) -> str:
    """Mask the ENTSO-E key in any text (`securityToken=***`)."""
    return SECURITY_TOKEN.sub(REDACTED_TOKEN, text)


def delivery_period(params: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start of start_date and start of the day after end_date, in market time.

    Args:
        params: Ingestion parameters (start_date, end_date, market_timezone).

    Returns:
        The half-open period [start, end) covering every delivery day.
    """
    timezone = params["market_timezone"]
    period_start = pd.Timestamp(params["start_date"], tz=timezone)
    day_after_end = pd.Timestamp(params["end_date"]) + ONE_DAY
    period_end = day_after_end.tz_localize(timezone)
    return period_start, period_end


def hourly_grid(params: dict[str, Any]) -> pd.DatetimeIndex:
    """Every hour of the delivery period, in UTC (23 or 25 hours on clock-change days).

    Args:
        params: Ingestion parameters.

    Returns:
        A continuous hourly UTC index named `timestamp`.
    """
    period_start, period_end = delivery_period(params)
    return pd.date_range(
        period_start.tz_convert(UTC_ZONE),
        period_end.tz_convert(UTC_ZONE),
        freq="h",
        inclusive="left",
        name=TIMESTAMP,
    )


def _local_midnight_utc(day: Any, timezone: str) -> pd.Timestamp:
    """Start of a calendar day in market time, as a UTC instant."""
    local_midnight = pd.Timestamp(day, tz=timezone)
    return local_midnight.tz_convert(UTC_ZONE)


def _is_transient(error: BaseException) -> bool:
    """Connection problems, timeouts, rate limits and 5xx answers deserve a new attempt."""
    if isinstance(error, requests.HTTPError):
        status = error.response.status_code if error.response is not None else None
        return status in RETRYABLE_HTTP_STATUS
    return True


def _call_with_retry[T](fetch: Callable[[], T], label: str, params: dict[str, Any]) -> T:
    """Call `fetch`, retrying transient network errors with a doubling wait.

    Args:
        fetch: Function doing one network request.
        label: What is fetched, for log and error messages.
        params: Ingestion parameters (retry_max_attempts, retry_backoff_seconds).

    Returns:
        What `fetch` returns.

    Raises:
        RuntimeError: On a permanent error, or when every attempt failed. Its message
            is cleaned of the API key and the original error is not attached to it.
    """
    max_attempts = params["retry_max_attempts"]
    first_wait = params["retry_backoff_seconds"]
    for attempt in range(1, max_attempts + 1):
        try:
            return fetch()
        except NETWORK_ERRORS as error:
            safe_error = redact_secrets(f"{type(error).__name__}: {error}")
            is_transient = _is_transient(error)
        # Raised outside the `except` block on purpose: the original error, whose
        # message may contain the API key, is then not chained to the new one.
        is_last_attempt = attempt == max_attempts
        if is_last_attempt or not is_transient:
            raise RuntimeError(f"Failed to fetch {label} after {attempt} attempt(s): {safe_error}")
        wait_seconds = first_wait * 2 ** (attempt - 1)
        logger.warning(
            "Fetching %s failed (%s). Retry %d/%d in %.0f s",
            label,
            safe_error,
            attempt,
            max_attempts - 1,
            wait_seconds,
        )
        time.sleep(wait_seconds)
    raise RuntimeError(f"No attempt made to fetch {label}: retry_max_attempts must be >= 1")


def _raise_for_status(response: requests.Response) -> None:
    """Like `response.raise_for_status()`, with the start of the answer in the message."""
    is_error = response.status_code >= HTTP_ERROR_STATUS
    if is_error:
        answer_start = response.text[:ERROR_BODY_CHARS]
        message = f"HTTP {response.status_code} for {response.url}: {answer_start}"
        raise requests.HTTPError(message, response=response)


def _download_bytes(url: str, label: str, params: dict[str, Any]) -> bytes:
    """Download a file, with timeout and retries."""

    def download() -> bytes:
        response = requests.get(url, timeout=params["http_timeout_seconds"])
        _raise_for_status(response)
        return response.content

    return _call_with_retry(download, label, params)


# ==================================================
# ENTSO-E
# ==================================================


def _entsoe_client(api_key: SecretStr, params: dict[str, Any]) -> EntsoePandasClient:
    """ENTSO-E client with a timeout: without one, a silent request blocks the run forever."""
    return EntsoePandasClient(
        api_key=api_key.get_secret_value(),
        timeout=params["http_timeout_seconds"],
    )


def _request_windows(
    start: pd.Timestamp, end: pd.Timestamp, months: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) into windows of `months` calendar months, in market time.

    entsoe-py pads price requests by one day on each side and splits any request
    longer than a year in two; the hour at the split is then lost (31 December,
    00:00). Windows well under a year (parameter entsoe_window_months) avoid it.
    """
    month_starts = pd.date_range(start, end, freq=f"{months}MS")
    inner_boundaries = [boundary for boundary in month_starts if start < boundary < end]
    boundaries = [start, *inner_boundaries, end]
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def _clean_time_index(
    data: pd.DataFrame | pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame | pd.Series:
    """Convert to UTC, sort, drop repeated timestamps and keep [start, end)."""
    utc_data = data.tz_convert(UTC_ZONE)
    sorted_data = utc_data.sort_index()
    # Consecutive yearly answers share their boundary timestamp.
    is_repeated = sorted_data.index.duplicated(keep="first")
    unique_data = sorted_data[~is_repeated]
    is_in_period = (unique_data.index >= start) & (unique_data.index < end)
    period_data = unique_data[is_in_period]
    return period_data.rename_axis(TIMESTAMP)


def _download_entsoe(
    query: EntsoeQuery,
    start: pd.Timestamp,
    end: pd.Timestamp,
    params: dict[str, Any],
    label: str,
) -> pd.DataFrame | pd.Series:
    """Download [start, end) window by window. A window without published data is skipped.

    Args:
        query: Function (window_start, window_end) -> answer of one ENTSO-E request.
        start: First instant (market time).
        end: First instant after the period (market time).
        params: Ingestion parameters.
        label: What is fetched, for log and error messages.

    Returns:
        The answers concatenated, in UTC, sorted, without repeated timestamps.

    Raises:
        NoDataError: If no window returned any data.
        RuntimeError: If a request fails (see `_call_with_retry`).
    """
    window_answers = []
    windows = _request_windows(start, end, params["entsoe_window_months"])
    for window_start, window_end in windows:
        window_label = f"{label} {window_start.date()} to {window_end.date()}"
        logger.info("Fetching %s", window_label)
        fetch_window = functools.partial(query, window_start, window_end)
        try:
            window_answer = _call_with_retry(fetch_window, window_label, params)
        except NoMatchingDataError:
            logger.warning("ENTSO-E published no %s", window_label)
            continue
        window_answers.append(window_answer)
    if not window_answers:
        raise NoDataError(f"ENTSO-E returned no {label} between {start.date()} and {end.date()}")
    combined = pd.concat(window_answers)
    return _clean_time_index(combined, start, end)


def _select_column(table: pd.DataFrame, source_column: str, target_column: str) -> pd.DataFrame:
    """Keep one ENTSO-E column under the project's name.

    Raises:
        ValueError: If the column is not in the answer.
    """
    if source_column not in table.columns:
        raise ValueError(
            f"ENTSO-E answer without '{source_column}' column, got {list(table.columns)}"
        )
    selected = table[[source_column]]
    return selected.rename(columns={source_column: target_column})


def fetch_day_ahead_prices(params: dict[str, Any], entsoe_api_key: SecretStr) -> pd.DataFrame:
    """Day-ahead prices of the bidding zone, at their published resolution (decision D2).

    One row per market time unit: 60 minutes until the quarter-hour go-live,
    15 minutes after. Known: published around 12:45 the day before delivery.

    Args:
        params: Ingestion parameters.
        entsoe_api_key: ENTSO-E key, masked.

    Returns:
        Column `price_eur_mwh` (EUR/MWh), index `timestamp` (UTC, start of the time unit).

    Raises:
        NoDataError: If ENTSO-E returned no price at all.
        RuntimeError: If a request fails.
    """
    client = _entsoe_client(entsoe_api_key, params)
    country = params["country"]

    def query(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        return client.query_day_ahead_prices(country, start=start, end=end)

    period_start, period_end = delivery_period(params)
    label = f"day-ahead prices {country}"
    prices = _download_entsoe(query, period_start, period_end, params, label)
    return prices.to_frame(PRICE)


def fetch_actual_load(params: dict[str, Any], entsoe_api_key: SecretStr) -> pd.DataFrame:
    """Actual total load of the bidding zone, at its published resolution.

    Known: realized value, published within one hour after real time.

    Args:
        params: Ingestion parameters.
        entsoe_api_key: ENTSO-E key, masked.

    Returns:
        Column `load_mw`, index `timestamp` (UTC).

    Raises:
        NoDataError: If ENTSO-E returned no load at all.
        ValueError: If the answer has no 'Actual Load' column.
    """
    client = _entsoe_client(entsoe_api_key, params)
    country = params["country"]

    def query(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return client.query_load(country, start=start, end=end)

    period_start, period_end = delivery_period(params)
    load = _download_entsoe(query, period_start, period_end, params, f"actual load {country}")
    return _select_column(load, ENTSOE_ACTUAL_LOAD, LOAD)


def fetch_load_forecast(params: dict[str, Any], entsoe_api_key: SecretStr) -> pd.DataFrame:
    """Day-ahead forecast of the total load, per delivery time unit (decision D3).

    Known: published at the latest 2 hours before the day-ahead gate closure
    (12:00), so before the 11:00 decision. Limit: ENTSO-E keeps the last version
    published, possibly revised after 11:00.

    Args:
        params: Ingestion parameters.
        entsoe_api_key: ENTSO-E key, masked.

    Returns:
        Column `load_forecast_mw`, index `timestamp` (UTC, delivery time unit).

    Raises:
        NoDataError: If ENTSO-E returned no forecast at all.
        ValueError: If the answer has no 'Forecasted Load' column.
    """
    client = _entsoe_client(entsoe_api_key, params)
    country = params["country"]

    def query(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return client.query_load_forecast(
            country, start=start, end=end, process_type=ENTSOE_DAY_AHEAD_PROCESS
        )

    period_start, period_end = delivery_period(params)
    label = f"day-ahead load forecast {country}"
    forecast = _download_entsoe(query, period_start, period_end, params, label)
    return _select_column(forecast, ENTSOE_FORECASTED_LOAD, LOAD_FORECAST)


def _actual_aggregated(published: pd.DataFrame) -> pd.DataFrame:
    """Keep generation only: pumped storage is also published as consumption (pumping).

    Raises:
        ValueError: If a two-level answer has no 'Actual Aggregated' columns.
    """
    has_two_levels = isinstance(published.columns, pd.MultiIndex)
    if not has_two_levels:
        return published
    second_level = published.columns.get_level_values(1)
    if ENTSOE_ACTUAL_AGGREGATED not in second_level:
        raise ValueError(f"ENTSO-E generation without '{ENTSOE_ACTUAL_AGGREGATED}' columns")
    return published.xs(ENTSOE_ACTUAL_AGGREGATED, axis=1, level=1)


def _generation_columns(published: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """One column per production type of `column_map`, renamed to gen_<type>_mw.

    A type absent from the whole answer (one year) was not produced: 0 MW.
    Hours missing for a published type stay NaN.
    """
    generation = _actual_aggregated(published)
    unexpected_types = sorted(set(generation.columns) - set(column_map))
    if unexpected_types:
        logger.warning("Production types not in generation_columns, dropped: %s", unexpected_types)
    unpublished_types = sorted(set(column_map) - set(generation.columns))
    if unpublished_types:
        logger.info("Production types not published, set to 0 MW: %s", unpublished_types)
    complete_generation = generation.reindex(columns=list(column_map), fill_value=0.0)
    return complete_generation.rename(columns=column_map)


def fetch_generation(params: dict[str, Any], entsoe_api_key: SecretStr) -> pd.DataFrame:
    """Actual generation per production type, at its published resolution.

    Known: realized values, published within one hour after real time.

    Args:
        params: Ingestion parameters (generation_columns maps types to columns).
        entsoe_api_key: ENTSO-E key, masked.

    Returns:
        One column `gen_<type>_mw` per entry of generation_columns, index `timestamp` (UTC).

    Raises:
        NoDataError: If ENTSO-E returned no generation at all.
    """
    client = _entsoe_client(entsoe_api_key, params)
    country = params["country"]
    column_map = params["generation_columns"]

    def query(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        published = client.query_generation(country, start=start, end=end)
        return _generation_columns(published, column_map)

    period_start, period_end = delivery_period(params)
    label = f"generation {country}"
    return _download_entsoe(query, period_start, period_end, params, label)


def _zone_slug(zone: str) -> str:
    """'DE_LU' -> 'delu', used in column names."""
    lower_zone = zone.lower()
    return lower_zone.replace("_", "")


def flow_column(exporter: str, importer: str) -> str:
    """Name of the raw physical flow column, e.g. flow_fr_to_be_mw."""
    return f"flow_{_zone_slug(exporter)}_to_{_zone_slug(importer)}_mw"


def net_flow_column(neighbor: str) -> str:
    """Name of the net flow column of a border, e.g. net_flow_fr."""
    return f"net_flow_{_zone_slug(neighbor)}"


def _empty_flow() -> pd.Series:
    """A flow without any value, indexed like the other raw tables."""
    empty_index = pd.DatetimeIndex([], tz=UTC_ZONE, name=TIMESTAMP)
    return pd.Series(index=empty_index, dtype=float)


def _border_start(in_service_from: Any, period_start: pd.Timestamp, timezone: str) -> pd.Timestamp:
    """First instant to request for a border: period start, or commissioning day if later."""
    if in_service_from is None:
        return period_start
    service_start = pd.Timestamp(in_service_from, tz=timezone)
    return max(period_start, service_start)


def _download_flow(
    client: EntsoePandasClient,
    exporter: str,
    importer: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    params: dict[str, Any],
) -> pd.Series:
    """Physical flow in one direction; empty (with a warning) if never published."""

    def query(window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.Series:
        return client.query_crossborder_flows(
            exporter, importer, start=window_start, end=window_end
        )

    label = f"physical flow {exporter}->{importer}"
    try:
        return _download_entsoe(query, start, end, params, label)
    except NoDataError:
        logger.warning("No %s published at all: column left empty (see quality report)", label)
        return _empty_flow()


def fetch_crossborder_flows(params: dict[str, Any], entsoe_api_key: SecretStr) -> pd.DataFrame:
    """Physical flows on each border, in both directions, at their published resolution.

    A border is only requested from the day its interconnector entered service
    (decision D11). Known: realized values, published after real time.

    Args:
        params: Ingestion parameters (country, borders).
        entsoe_api_key: ENTSO-E key, masked.

    Returns:
        Columns `flow_<from>_to_<to>_mw` (MW, >= 0), index `timestamp` (UTC).

    Raises:
        NoDataError: If no direction of any border returned data.
    """
    client = _entsoe_client(entsoe_api_key, params)
    country = params["country"]
    timezone = params["market_timezone"]
    period_start, period_end = delivery_period(params)
    flows = {}
    for neighbor, in_service_from in params["borders"].items():
        border_start = _border_start(in_service_from, period_start, timezone)
        for exporter, importer in ((neighbor, country), (country, neighbor)):
            column = flow_column(exporter, importer)
            flows[column] = _download_flow(
                client, exporter, importer, border_start, period_end, params
            )
    has_any_flow = any(len(flow) > 0 for flow in flows.values())
    if not has_any_flow:
        raise NoDataError(f"ENTSO-E returned no cross-border flow for {country}")
    flow_table = pd.DataFrame(flows)
    return flow_table.rename_axis(TIMESTAMP)


# ==================================================
# Gas (Yahoo Finance) and carbon (EEX)
# ==================================================


def _checked_daily_history(daily: pd.DataFrame, params: dict[str, Any], label: str) -> pd.DataFrame:
    """Sort a daily history, check it covers the period, and keep the period only.

    Raises:
        NoDataError: If the history is empty.
        ValueError: If a trading date appears twice, or if the history starts more than
            max_history_gap_days after start_date or ends that much before end_date.
    """
    has_value = daily.notna().all(axis=1)
    valued_days = daily[has_value]
    sorted_days = valued_days.sort_index()
    if sorted_days.empty:
        raise NoDataError(f"{label}: no data returned")
    if sorted_days.index.has_duplicates:
        raise ValueError(f"{label}: several values for the same trading date")
    first_day = pd.Timestamp(params["start_date"])
    last_day = pd.Timestamp(params["end_date"])
    tolerance = pd.Timedelta(days=params["max_history_gap_days"])
    starts_too_late = sorted_days.index[0] > first_day + tolerance
    ends_too_early = sorted_days.index[-1] < last_day - tolerance
    if starts_too_late or ends_too_early:
        raise ValueError(
            f"{label}: history covers {sorted_days.index[0].date()} to "
            f"{sorted_days.index[-1].date()}, the period is {first_day.date()} to "
            f"{last_day.date()} (tolerance {tolerance.days} days). Change the source or "
            "the period in parameters.yml (decisions D4 and D8)."
        )
    is_in_period = (sorted_days.index >= first_day) & (sorted_days.index <= last_day)
    period_days = sorted_days[is_in_period]
    logger.info("%s: %d trading days", label, len(period_days))
    return period_days


def fetch_ttf_prices(params: dict[str, Any]) -> pd.DataFrame:
    """Daily close of the front-month Dutch TTF gas future, from Yahoo Finance (decision D4).

    Known: the close of trading day D is set in the evening of D, after the
    11:00 decision of D. Yahoo data is for personal research use, and the
    series jumps when the front contract rolls over.

    Args:
        params: Ingestion parameters (ttf_ticker, period, network settings).

    Returns:
        Column `ttf_eur_mwh` (EUR/MWh), index `trading_date` (calendar date).

    Raises:
        ValueError: If the answer has no 'Close' column or does not cover the period.
        RuntimeError: If the download fails.
    """
    ticker_symbol = params["ttf_ticker"]
    first_day = pd.Timestamp(params["start_date"])
    # Yahoo excludes the end date: ask up to the day after end_date.
    day_after_end = pd.Timestamp(params["end_date"]) + ONE_DAY
    ticker = yf.Ticker(ticker_symbol)

    def download() -> pd.DataFrame:
        return ticker.history(
            start=first_day.strftime(DATE_FORMAT),
            end=day_after_end.strftime(DATE_FORMAT),
            interval=YAHOO_DAILY,
            auto_adjust=False,
            timeout=params["http_timeout_seconds"],
            raise_errors=True,
        )

    history = _call_with_retry(download, f"{ticker_symbol} history", params)
    if YAHOO_CLOSE not in history.columns:
        raise ValueError(f"Yahoo answer without '{YAHOO_CLOSE}' column: {list(history.columns)}")
    trading_dates = pd.DatetimeIndex(history.index.date, name=TRADING_DATE)
    closes = pd.DataFrame({TTF: history[YAHOO_CLOSE].to_numpy()}, index=trading_dates)
    return _checked_daily_history(closes, params, f"TTF ({ticker_symbol})")


def _years_in_name(file_name: str) -> range:
    """Years covered by a workbook, read from its name ('...2019...' or '...2012-2025...')."""
    base_name = file_name.rsplit("/", maxsplit=1)[-1]
    years = [int(year) for year in YEAR_PATTERN.findall(base_name)]
    if not years:
        return range(0)
    return range(min(years), max(years) + 1)


def _archived_eua_workbooks(years: list[int], params: dict[str, Any]) -> list[tuple[str, bytes]]:
    """Workbooks of the EEX archive that cover `years`.

    Raises:
        ValueError: If a requested year is in no workbook of the archive.
    """
    if not years:
        return []
    archive = _download_bytes(params["eua_archive_url"], "EEX auction archive", params)
    with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
        workbook_names = [
            name for name in zip_file.namelist() if name.lower().endswith(EXCEL_SUFFIXES)
        ]
        selected_names = [name for name in workbook_names if set(_years_in_name(name)) & set(years)]
        covered_years = {year for name in selected_names for year in _years_in_name(name)}
        missing_years = sorted(set(years) - covered_years)
        if missing_years:
            raise ValueError(
                f"EEX archive has no workbook for {missing_years}. "
                f"Workbooks in the archive: {workbook_names}"
            )
        return [(name, zip_file.read(name)) for name in selected_names]


def _yearly_eua_workbook(year: int, params: dict[str, Any]) -> tuple[str, bytes]:
    """Workbook of one year published by EEX."""
    url = params["eua_yearly_url"].format(year=year)
    content = _download_bytes(url, f"EEX auction report {year}", params)
    return url, content


def _auction_sheet_table(sheet: pd.DataFrame) -> pd.DataFrame | None:
    """Table below the header row of an auction sheet (None if the sheet has none).

    EEX puts a few title rows above the column names: the header row is the
    first one holding the 'Auction Price ...' column.
    """
    sheet_text = sheet.astype(str)
    is_price_header = sheet_text.apply(lambda column: column.str.startswith(EEX_PRICE_PREFIX))
    header_positions = np.flatnonzero(is_price_header.any(axis=1).to_numpy())
    if len(header_positions) == 0:
        return None
    header_position = header_positions[0]
    column_names = sheet_text.iloc[header_position].str.strip()
    rows_below_header = sheet.iloc[header_position + 1 :]
    return rows_below_header.set_axis(column_names, axis=1)


def _is_successful(auctions: pd.DataFrame) -> pd.Series:
    """True for the auctions that took place.

    Recent reports have a Status column ('successful' or 'cancelled'). Archived
    reports (2018 and before) have none: there, a cancelled auction simply has no
    clearing price, which `_general_allowance_prices` also requires.
    """
    has_status = EEX_STATUS in auctions.columns
    if not has_status:
        return pd.Series(True, index=auctions.index)
    statuses = auctions[EEX_STATUS].astype(str).str.strip().str.lower()
    return statuses == EEX_SUCCESSFUL


def _auction_dates(values: pd.Series, source: str) -> pd.Series:
    """Auction dates as calendar dates (NaT for rows that are not auctions, e.g. notes).

    Excel cells usually hold real dates. Text dates are read day first
    (European format, 02/01/2018 = 2 January).

    Raises:
        ValueError: If a date falls on a weekend: auctions only take place on
            weekdays, so the dates were misread.
    """
    parsed = pd.to_datetime(values, dayfirst=True, format="mixed", errors="coerce")
    dates = parsed.dt.normalize()
    is_weekend = dates.dt.dayofweek >= WEEKEND_FIRST_DAY
    if is_weekend.any():
        examples = values[is_weekend].head(3).tolist()
        raise ValueError(
            f"EEX workbook {source}: {int(is_weekend.sum())} auction dates fall on a weekend, "
            f"the date column was misread (examples: {examples})"
        )
    return dates


def _general_allowance_prices(auctions: pd.DataFrame, contract: str, source: str) -> pd.DataFrame:
    """Date and clearing price of the auctions of `contract` that took place (may be empty).

    Raises:
        ValueError: If expected columns are missing, or a date is misread.
    """
    price_columns = [name for name in auctions.columns if name.startswith(EEX_PRICE_PREFIX)]
    required_columns = [EEX_DATE, EEX_CONTRACT]
    missing_columns = [name for name in required_columns if name not in auctions.columns]
    if missing_columns or len(price_columns) != 1:
        raise ValueError(
            f"EEX workbook {source}: expected columns {required_columns} and one "
            f"'{EEX_PRICE_PREFIX} ...' column, got {list(auctions.columns)}"
        )
    contracts = auctions[EEX_CONTRACT].astype(str).str.strip()
    # A cancelled auction or a note row has no numeric price: it becomes NaN.
    clearing_prices = pd.to_numeric(auctions[price_columns[0]], errors="coerce")
    auction_dates = _auction_dates(auctions[EEX_DATE], source)
    is_general_allowance = contracts == contract
    has_price = clearing_prices > 0
    has_date = auction_dates.notna()
    is_selected = is_general_allowance & _is_successful(auctions) & has_price & has_date
    return pd.DataFrame(
        {
            TRADING_DATE: auction_dates[is_selected].to_numpy(),
            EUA: clearing_prices[is_selected].to_numpy(),
        }
    )


def _contracts_found(auction_tables: list[pd.DataFrame]) -> list[str]:
    """Distinct contract codes of a workbook, to explain an empty selection."""
    contract_columns = [
        table[EEX_CONTRACT].dropna().astype(str).str.strip()
        for table in auction_tables
        if EEX_CONTRACT in table.columns
    ]
    if not contract_columns:
        return []
    all_contracts = pd.concat(contract_columns)
    return sorted(all_contracts.unique().tolist())


def _read_eua_auctions(content: bytes, source: str, contract: str) -> pd.DataFrame:
    """Auction dates and clearing prices of every sheet of one EEX workbook.

    Raises:
        ValueError: If no sheet has an auction table, or none holds a priced
            auction of `contract`.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=EEX_HARMLESS_WARNING, category=UserWarning)
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    sheet_tables = [_auction_sheet_table(sheet) for sheet in sheets.values()]
    auction_tables = [table for table in sheet_tables if table is not None]
    if not auction_tables:
        raise ValueError(f"EEX workbook {source}: no sheet with a '{EEX_PRICE_PREFIX} ...' column")
    prices_per_sheet = [
        _general_allowance_prices(table, contract, source) for table in auction_tables
    ]
    workbook_prices = pd.concat(prices_per_sheet, ignore_index=True)
    if workbook_prices.empty:
        raise ValueError(
            f"EEX workbook {source}: no auction of contract {contract} with a price. "
            f"Contracts in the workbook: {_contracts_found(auction_tables)}"
        )
    logger.info("EEX workbook %s: %d auctions of %s", source, len(workbook_prices), contract)
    return workbook_prices


def fetch_eua_prices(params: dict[str, Any]) -> pd.DataFrame:
    """Daily EU allowance price: clearing price of the EEX primary auctions (decision D4).

    General allowances only (contract `eua_contract`); the EU, German and Polish
    auctions of the same day are averaged. Known: auction results are published
    on the auction day, around noon.

    Args:
        params: Ingestion parameters (eua_* settings, period, network settings).

    Returns:
        Column `eua_eur_t` (EUR/t), index `trading_date` (calendar date).

    Raises:
        ValueError: If a workbook has an unexpected layout or the history does not cover
            the period.
        RuntimeError: If a download fails.
    """
    first_year = pd.Timestamp(params["start_date"]).year
    last_year = pd.Timestamp(params["end_date"]).year
    last_archived_year = params["eua_archive_last_year"]
    years = list(range(first_year, last_year + 1))
    archived_years = [year for year in years if year <= last_archived_year]
    recent_years = [year for year in years if year > last_archived_year]
    archived_workbooks = _archived_eua_workbooks(archived_years, params)
    recent_workbooks = [_yearly_eua_workbook(year, params) for year in recent_years]
    auction_tables = [
        _read_eua_auctions(content, source, params["eua_contract"])
        for source, content in archived_workbooks + recent_workbooks
    ]
    auctions = pd.concat(auction_tables, ignore_index=True)
    daily_prices = auctions.groupby(TRADING_DATE)[EUA].mean()
    return _checked_daily_history(daily_prices.to_frame(), params, "EUA (EEX auctions)")


# ==================================================
# Weather (Open-Meteo)
# ==================================================


def _weather_table(answer: dict[str, Any]) -> pd.DataFrame:
    """Hourly weather table from an Open-Meteo JSON answer.

    Raises:
        ValueError: If the answer misses the hourly data or a variable.
        NoDataError: If the answer holds no hour.
    """
    if "hourly" not in answer:
        raise ValueError(f"Open-Meteo answer without hourly data, keys: {list(answer)}")
    hourly = answer["hourly"]
    missing_variables = [name for name in WEATHER_VARIABLES if name not in hourly]
    if missing_variables:
        raise ValueError(f"Open-Meteo answer without {missing_variables}")
    # Times are in UTC because the request asks for timezone=UTC.
    timestamps = pd.to_datetime(hourly["time"], utc=True)
    index = pd.DatetimeIndex(timestamps, name=TIMESTAMP)
    # Missing values come as null: dtype=float turns them into NaN.
    values = {name: np.asarray(hourly[name], dtype=float) for name in WEATHER_VARIABLES}
    table = pd.DataFrame(values, index=index)
    if table.empty:
        raise NoDataError("Open-Meteo returned no weather hour")
    return table


def fetch_weather(params: dict[str, Any]) -> pd.DataFrame:
    """Observed hourly weather at one point, from the Open-Meteo archive (ERA5).

    Known: realized values (ERA5 is published about 5 days after real time).
    Used for the retrospective analysis only (decision D3).

    Args:
        params: Ingestion parameters (weather point and model, period).

    Returns:
        Columns temperature_2m (degC), wind_speed_100m (m/s), shortwave_radiation
        (W/m2), cloud_cover (%), index `timestamp` (UTC).

    Raises:
        ValueError: If the answer is incomplete.
        RuntimeError: If the download fails.
    """
    weather = params["weather"]
    # Open-Meteo works in whole UTC days: one more day on each side covers the
    # offset of the market time zone. build_master keeps the delivery period only.
    first_day = pd.Timestamp(params["start_date"]) - ONE_DAY
    last_day = pd.Timestamp(params["end_date"]) + ONE_DAY
    query = {
        "latitude": weather["latitude"],
        "longitude": weather["longitude"],
        "start_date": first_day.strftime(DATE_FORMAT),
        "end_date": last_day.strftime(DATE_FORMAT),
        "hourly": ",".join(WEATHER_VARIABLES),
        "models": weather["model"],
        "timezone": UTC_ZONE,
        # Open-Meteo answers in km/h unless asked otherwise.
        "wind_speed_unit": OPEN_METEO_WIND_UNIT,
    }

    def download() -> dict[str, Any]:
        response = requests.get(
            weather["archive_url"], params=query, timeout=params["http_timeout_seconds"]
        )
        _raise_for_status(response)
        return response.json()

    answer = _call_with_retry(download, "Open-Meteo weather", params)
    return _weather_table(answer)


# ==================================================
# Master table
# ==================================================


def _require_utc_index(table: pd.DataFrame, label: str) -> None:
    """Raise ValueError unless `table` is indexed by UTC instants."""
    index = table.index
    is_utc = isinstance(index, pd.DatetimeIndex) and str(index.tz) == UTC_ZONE
    if not is_utc:
        raise ValueError(
            f"{label}: index must be a DatetimeIndex in UTC, got "
            f"{type(index).__name__} with tz={getattr(index, 'tz', None)}"
        )


def _require_trading_dates(table: pd.DataFrame, label: str) -> None:
    """Raise ValueError unless `table` is indexed by unique calendar dates."""
    index = table.index
    is_date_index = isinstance(index, pd.DatetimeIndex) and index.tz is None
    if not is_date_index:
        raise ValueError(f"{label}: index must hold calendar dates (no time zone)")
    is_midnight = index == index.normalize()
    if not is_midnight.all() or index.has_duplicates:
        raise ValueError(f"{label}: index must hold unique dates without time of day")


def _require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    """Raise ValueError if `table` misses one of `columns`."""
    missing_columns = [name for name in columns if name not in table.columns]
    if missing_columns:
        raise ValueError(f"{label}: missing columns {missing_columns}, re-run its fetch node")


def _hourly_mean(table: pd.DataFrame, grid: pd.DatetimeIndex, label: str) -> pd.DataFrame:
    """Mean of the values of each hour, on the hourly grid.

    Only values of the same hour are averaged: no value comes from another hour.
    """
    _require_utc_index(table, label)
    hourly = table.resample("h").mean()
    return hourly.reindex(grid)


def _hourly_generation(
    generation: pd.DataFrame, grid: pd.DatetimeIndex, params: dict[str, Any]
) -> pd.DataFrame:
    """Hourly generation; unpublished output of the zero_when_unpublished units is 0 MW.

    Elia publishes no value for these units when they produce nothing (offshore
    wind at calm, pumped storage while pumping or idle). This only applies to
    hours where the generation document was published: a missing document
    keeps every column NaN.
    """
    hourly_generation = _hourly_mean(generation, grid, "generation")
    is_document_published = hourly_generation.notna().any(axis=1)
    zero_columns = params["zero_when_unpublished"]
    _require_columns(hourly_generation, zero_columns, "generation")
    for column in zero_columns:
        is_unpublished = hourly_generation[column].isna() & is_document_published
        logger.info("%s: %d unpublished hours set to 0 MW", column, int(is_unpublished.sum()))
        hourly_generation.loc[is_unpublished, column] = 0.0
    return hourly_generation


def price_completeness(prices: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Per hour: mean price, number of published time units, and completeness.

    An hour is complete with 1 hourly price before the quarter-hour go-live and
    with its 4 quarter-hour prices after (decision D2).

    Args:
        prices: Raw day-ahead prices (`fetch_day_ahead_prices`).
        params: Ingestion parameters (quarter_hour_prices_from, market_timezone).

    Returns:
        Columns `mean`, `published`, `is_complete`, indexed by UTC hour.
    """
    _require_utc_index(prices, "prices")
    price_values = prices[PRICE]
    published = price_values.resample("h").count()
    hourly_mean = price_values.resample("h").mean()
    quarter_hour_start = _local_midnight_utc(
        params["quarter_hour_prices_from"], params["market_timezone"]
    )
    is_quarter_hour_market = published.index >= quarter_hour_start
    expected = np.where(is_quarter_hour_market, QUARTER_HOURS_PER_HOUR, HOURLY_UNITS_PER_HOUR)
    is_complete = published.to_numpy() == expected
    return pd.DataFrame({"mean": hourly_mean, "published": published, "is_complete": is_complete})


def _hourly_prices(
    prices: pd.DataFrame, grid: pd.DatetimeIndex, params: dict[str, Any]
) -> pd.DataFrame:
    """Hourly price; NaN when one of the hour's time units is missing (the target is exact)."""
    completeness = price_completeness(prices, params)
    complete_prices = completeness["mean"].where(completeness["is_complete"])
    is_partial = (completeness["published"] > 0) & ~completeness["is_complete"]
    partial_hours = int(is_partial.sum())
    if partial_hours:
        logger.warning("%d hours miss part of their prices: set to NaN", partial_hours)
    hourly_prices = complete_prices.reindex(grid)
    return hourly_prices.to_frame(PRICE)


def _is_before_service(index: pd.DatetimeIndex, in_service_from: Any, timezone: str) -> np.ndarray:
    """True for the hours before an interconnector entered service."""
    if in_service_from is None:
        return np.zeros(len(index), dtype=bool)
    service_start = _local_midnight_utc(in_service_from, timezone)
    return np.asarray(index < service_start)


def _net_flows(hourly_flows: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Net import per border and in total, in MW (positive = import), decision D11."""
    country = params["country"]
    timezone = params["market_timezone"]
    net_flows = {}
    for neighbor, in_service_from in params["borders"].items():
        import_column = flow_column(neighbor, country)
        export_column = flow_column(country, neighbor)
        _require_columns(hourly_flows, [import_column, export_column], "flows")
        # NaN as soon as one direction is missing: a missing value is never replaced by 0.
        net_flow = hourly_flows[import_column] - hourly_flows[export_column]
        is_before_service = _is_before_service(hourly_flows.index, in_service_from, timezone)
        # Before commissioning there is no line: the flow is 0 by construction.
        net_flows[net_flow_column(neighbor)] = net_flow.mask(is_before_service, 0.0)
    net_table = pd.DataFrame(net_flows)
    total_import = net_table.sum(axis=1, skipna=False)
    return net_table.assign(**{NET_IMPORT_TOTAL: total_import})


def _on_market_days(
    daily: pd.DataFrame, grid: pd.DatetimeIndex, params: dict[str, Any], label: str
) -> pd.DataFrame:
    """Place the value of trading day D on every hour of day D in market time.

    Days without a session (weekends, holidays) stay NaN. No shift here.
    """
    _require_trading_dates(daily, label)
    local_days = grid.tz_convert(params["market_timezone"]).date
    market_days = pd.DatetimeIndex(local_days)
    values_per_hour = daily.reindex(market_days)
    return values_per_hour.set_axis(grid)


def _check_master(master: pd.DataFrame, grid: pd.DatetimeIndex) -> None:
    """Raise ValueError if the master table breaks its contract."""
    if not master.index.equals(grid):
        raise ValueError("master index differs from the continuous hourly UTC grid")
    if master.columns.has_duplicates:
        duplicated_names = master.columns[master.columns.duplicated()].tolist()
        raise ValueError(f"master has duplicated columns: {duplicated_names}")
    if master[PRICE].isna().all():
        raise ValueError("master has no price at all: check entsoe_prices_raw")


def build_master(
    prices: pd.DataFrame,
    load: pd.DataFrame,
    load_forecast: pd.DataFrame,
    generation: pd.DataFrame,
    flows: pd.DataFrame,
    ttf: pd.DataFrame,
    eua: pd.DataFrame,
    weather: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Merge every source on one continuous hourly UTC grid.

    - An hour absent from every source stays a row of NaN: gaps remain visible.
    - Hourly values average the values of the same hour only (no bfill, no
      interpolation, no value borrowed from another hour).
    - Prices: an hour needs all its time units, else NaN (decision D2).
    - Generation: unpublished output of zero_when_unpublished units is 0 MW
      (decision D12).
    - Flows: net flow per border and total imports (decision D11).
    - Gas and carbon: price of trading day D on the hours of day D in market
      time, NaN on days without a session. No shift: the 11:00 rule is applied
      when features are built (decisions.md, §6).

    Args:
        prices: Raw day-ahead prices.
        load: Raw actual load.
        load_forecast: Raw day-ahead load forecast.
        generation: Raw generation per production type.
        flows: Raw physical flows, both directions per border.
        ttf: Daily TTF closes.
        eua: Daily EUA auction prices.
        weather: Raw hourly weather.
        params: Ingestion parameters.

    Returns:
        The master table described in conventions.md, section 7.

    Raises:
        ValueError: If an input is not indexed as expected or no price is available.
    """
    grid = hourly_grid(params)
    hourly_flows = _hourly_mean(flows, grid, "flows")
    hourly_tables = [
        _hourly_prices(prices, grid, params),
        _hourly_mean(load_forecast, grid, "load forecast"),
        _hourly_mean(load, grid, "load"),
        _hourly_generation(generation, grid, params),
        _net_flows(hourly_flows, params),
        _on_market_days(ttf, grid, params, "ttf"),
        _on_market_days(eua, grid, params, "eua"),
        _hourly_mean(weather, grid, "weather"),
    ]
    master = pd.concat(hourly_tables, axis=1)
    _check_master(master, grid)
    logger.info("Master table built: %d hours x %d columns", len(master), len(master.columns))
    return master


# ==================================================
# Quality report
# ==================================================


def _json_number(value: float) -> float | None:
    """A float rounded for the report, or None (JSON has no NaN)."""
    if pd.isna(value):
        return None
    return round(float(value), 2)


def _iso(instant: pd.Timestamp | None) -> str | None:
    """ISO text of an instant, or None."""
    if instant is None:
        return None
    return instant.isoformat()


def _longest_run(flags: pd.Series) -> int:
    """Length of the longest run of consecutive True values."""
    run_id = (~flags).cumsum()
    run_lengths = flags.groupby(run_id).sum()
    if run_lengths.empty:
        return 0
    return int(run_lengths.max())


def _column_quality(values: pd.Series) -> dict[str, Any]:
    """Missing hours, longest gap, first and last valid hour of one column."""
    is_missing = values.isna()
    missing_hours = int(is_missing.sum())
    valid_hours = values.index[~is_missing]
    has_values = len(valid_hours) > 0
    return {
        "missing_hours": missing_hours,
        "missing_pct": round(100 * missing_hours / len(values), 2),
        "longest_gap_hours": _longest_run(is_missing),
        "first_valid": _iso(valid_hours[0]) if has_values else None,
        "last_valid": _iso(valid_hours[-1]) if has_values else None,
    }


def _price_summary(
    hourly_prices: pd.Series, prices: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Negative hours, range and incomplete quarter-hour hours of the price."""
    completeness = price_completeness(prices, params)
    is_partial = (completeness["published"] > 0) & ~completeness["is_complete"]
    is_negative = hourly_prices < 0
    return {
        "negative_hours": int(is_negative.sum()),
        "min": _json_number(hourly_prices.min()),
        "max": _json_number(hourly_prices.max()),
        "mean": _json_number(hourly_prices.mean()),
        "incomplete_hours": int(is_partial.sum()),
    }


def _border_summary(master: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Service date and missing hours after it, per border."""
    summary = {}
    for neighbor, in_service_from in params["borders"].items():
        net_flow = master[net_flow_column(neighbor)]
        summary[neighbor] = {
            "in_service_from": None if in_service_from is None else str(in_service_from),
            "missing_hours_in_service": int(net_flow.isna().sum()),
        }
    return summary


def _daily_summary(values: pd.Series, params: dict[str, Any]) -> dict[str, Any]:
    """Number of trading days, first and last one, of a daily price placed on hours."""
    valued_hours = values.dropna().index
    local_hours = valued_hours.tz_convert(params["market_timezone"])
    trading_days = local_hours.normalize().unique()
    has_days = len(trading_days) > 0
    return {
        "trading_days": len(trading_days),
        "first_day": str(trading_days[0].date()) if has_days else None,
        "last_day": str(trading_days[-1].date()) if has_days else None,
    }


def audit_quality(
    master: pd.DataFrame, prices: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Data quality report of the master table (conventions.md, section 7).

    Args:
        master: Master table (`build_master`).
        prices: Raw day-ahead prices, to count incomplete quarter-hour hours.
        params: Ingestion parameters.

    Returns:
        A JSON-ready report: period, per-column gaps, prices, borders, gas and carbon.
    """
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "period": {
            "first_hour_utc": _iso(master.index[0]),
            "last_hour_utc": _iso(master.index[-1]),
            "hours": len(master),
        },
        "columns": {name: _column_quality(master[name]) for name in master.columns},
        "prices": _price_summary(master[PRICE], prices, params),
        "borders": _border_summary(master, params),
        "commodities": {name: _daily_summary(master[name], params) for name in (TTF, EUA)},
    }
    price_quality = report["columns"][PRICE]
    logger.info(
        "Quality: %d hours, %d without price (%.2f%%), %d negative-price hours",
        len(master),
        price_quality["missing_hours"],
        price_quality["missing_pct"],
        report["prices"]["negative_hours"],
    )
    return report
