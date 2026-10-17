"""Tests of the ingestion nodes: the hard cases listed in decisions.md, section 5.

No network call: ENTSO-E, Yahoo and EEX are replaced by mocks.
"""

from __future__ import annotations

import io
import logging
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests
from entsoe.decorators import documents_limited, year_limited
from entsoe.exceptions import NoMatchingDataError
from pydantic import SecretStr

from energy_market_analyzer.pipelines.ingestion.nodes import (
    EUA,
    NET_IMPORT_TOTAL,
    PRICE,
    TRADING_DATE,
    TTF,
    NoDataError,
    build_master,
    fetch_day_ahead_prices,
    fetch_eua_prices,
    flow_column,
    hourly_grid,
)
from tests.helpers import make_ingestion_params, make_raw_sources, utc_index

NODES = "energy_market_analyzer.pipelines.ingestion.nodes"
ENTSOE_CLIENT = f"{NODES}.EntsoePandasClient"
FAKE_KEY = "FAKE-KEY-used-only-in-tests"
EEX_HEADER = ["Date", "Time", "Auction Name", "Contract", "Status", "Auction Price €/tCO2"]
# Archived reports (2018 and before): an empty first column and no Status column.
EEX_ARCHIVED_HEADER = [None, "Date", "Time", "Auction Name", "Contract", "Auction Price €/tCO2"]
EEX_TITLE_ROWS = 5
EEX_ARCHIVE_URL = "https://example.test/eex-archive.zip"


def _utc(text: str) -> pd.Timestamp:
    """UTC instant from text, e.g. '2025-10-01 05:00'."""
    return pd.Timestamp(text, tz="UTC")


def test_quarter_hour_prices_become_exact_hourly_means() -> None:
    """Hourly market before 1 October 2025, quarter-hours after: one exact price per hour."""
    params = make_ingestion_params(start_date="2025-09-30", end_date="2025-10-01")
    hourly_market = pd.DataFrame({PRICE: 50.0}, index=utc_index("2025-09-29 22:00", 24))
    quarter_hours = utc_index("2025-09-30 22:00", 96, freq="15min")
    quarter_hour_market = pd.DataFrame({PRICE: [10.0, 20.0, 30.0, 40.0] * 24}, index=quarter_hours)
    # One quarter-hour missing at 05:15 UTC on 1 October: that hour cannot be exact.
    is_missing_unit = quarter_hour_market.index == _utc("2025-10-01 05:15")
    incomplete_market = quarter_hour_market[~is_missing_unit]
    raw_prices = pd.concat([hourly_market, incomplete_market])
    sources = make_raw_sources(params)

    master = build_master(**{**sources, "prices": raw_prices}, params=params)

    assert len(master) == 48
    assert master.loc[_utc("2025-09-30 10:00"), PRICE] == 50.0
    assert master.loc[_utc("2025-10-01 08:00"), PRICE] == pytest.approx(25.0)
    assert np.isnan(master.loc[_utc("2025-10-01 05:00"), PRICE])


class _EntsoeLikePrices:
    """Fake day-ahead price service that behaves like entsoe-py 0.8.

    ENTSO-E answers [start, end) with one price per hour. entsoe-py pads the
    request by one day on each side, splits it into one-year blocks (keeping each
    block but the first left-open), converts to Brussels time and truncates.
    """

    @year_limited
    @documents_limited(100)
    def _server(self, start: pd.Timestamp, end: pd.Timestamp, offset: int = 0) -> pd.Series:
        if offset > 0:
            raise NoMatchingDataError
        hours = pd.date_range(start, end, freq="h", inclusive="left")
        return pd.Series(np.arange(len(hours), dtype=float), index=hours)

    def query_day_ahead_prices(
        self, country: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        padded = self._server(start=start - pd.Timedelta(days=1), end=end + pd.Timedelta(days=1))
        local = padded.tz_convert("Europe/Brussels")
        return local.truncate(before=start, after=end)


def test_no_hour_is_lost_at_clock_changes_or_year_ends() -> None:
    """Every hour of 2024-2025 comes back once, in UTC.

    Covers the 23-hour and 25-hour days of the clock changes, and the hour that
    entsoe-py drops on 31 December when one request spans more than a year.
    """
    params = make_ingestion_params(start_date="2024-01-01", end_date="2025-12-31")
    expected_hours = hourly_grid(params)

    with patch(ENTSOE_CLIENT, return_value=_EntsoeLikePrices()):
        prices = fetch_day_ahead_prices(params, SecretStr(FAKE_KEY))

    missing_hours = expected_hours.difference(prices.index)
    assert list(missing_hours) == []
    assert prices.index.equals(expected_hours)
    october_change_day = (
        prices.index.tz_convert("Europe/Brussels").date == pd.Timestamp("2025-10-26").date()
    )
    assert october_change_day.sum() == 25
    spring_grid = hourly_grid(make_ingestion_params(start_date="2025-03-30", end_date="2025-03-30"))
    assert len(spring_grid) == 23


def test_empty_entsoe_answer_raises_a_clear_error() -> None:
    """No data for a year is skipped (no retry); no data at all stops ingestion."""
    params = make_ingestion_params(start_date="2024-12-30", end_date="2025-01-02")
    second_year_hours = pd.date_range(
        "2025-01-01", "2025-01-03", freq="h", tz="Europe/Brussels", inclusive="left"
    )
    second_year = pd.Series(60.0, index=second_year_hours)

    with patch(ENTSOE_CLIENT) as client_class:
        query = client_class.return_value.query_day_ahead_prices
        query.side_effect = [NoMatchingDataError(), second_year]
        partial_prices = fetch_day_ahead_prices(params, SecretStr(FAKE_KEY))
        query.side_effect = NoMatchingDataError()
        with pytest.raises(NoDataError, match="ENTSO-E returned no day-ahead prices BE"):
            fetch_day_ahead_prices(params, SecretStr(FAKE_KEY))

    assert partial_prices.index[0] == _utc("2024-12-31 23:00")
    assert len(partial_prices) == 48
    assert query.call_count == 4


def test_missing_values_stay_nan_and_known_zeros_become_zero() -> None:
    """Only facts become 0: a line not yet built, an offshore park at calm.

    ALEGrO (DE_LU) delivers from 18 November 2020. A missing flow direction, or a
    generation document never published, stays NaN.
    """
    params = make_ingestion_params(
        start_date="2020-11-17",
        end_date="2020-11-18",
        borders={"FR": None, "DE_LU": "2020-11-18"},
    )
    sources = make_raw_sources(params)
    missing_hour = _utc("2020-11-18 10:00")
    flows = sources["flows"].copy()
    flows.loc[missing_hour, flow_column("BE", "FR")] = np.nan
    alegro_start = _utc("2020-11-17 23:00")  # 18 November, 00:00 in Brussels
    calm_hour = _utc("2020-11-17 05:00")
    unpublished_hour = _utc("2020-11-17 06:00")
    generation = sources["generation"].copy()
    is_calm_hour = generation.index.floor("h") == calm_hour
    is_unpublished_hour = generation.index.floor("h") == unpublished_hour
    generation.loc[is_calm_hour, "gen_wind_offshore_mw"] = np.nan
    generation.loc[is_unpublished_hour, :] = np.nan

    master = build_master(**{**sources, "flows": flows, "generation": generation}, params=params)

    is_before_alegro = master.index < alegro_start
    assert (master.loc[is_before_alegro, "net_flow_delu"] == 0.0).all()
    assert master.loc[~is_before_alegro, "net_flow_delu"].notna().all()
    assert np.isnan(master.loc[missing_hour, "net_flow_fr"])
    assert np.isnan(master.loc[missing_hour, NET_IMPORT_TOTAL])
    other_hour = _utc("2020-11-18 11:00")
    expected_total = master.loc[other_hour, ["net_flow_fr", "net_flow_delu"]].sum()
    assert master.loc[other_hour, NET_IMPORT_TOTAL] == pytest.approx(expected_total)
    assert master.loc[calm_hour, "gen_wind_offshore_mw"] == 0.0
    assert master.loc[calm_hour, "gen_gas_mw"] > 0
    assert master.loc[unpublished_hour, ["gen_wind_offshore_mw", "gen_gas_mw"]].isna().all()


def test_api_key_never_appears_in_logs_or_errors(caplog: pytest.LogCaptureFixture) -> None:
    """entsoe-py puts the key in the URL: errors and logs must show securityToken=*** only."""
    params = make_ingestion_params(retry_max_attempts=2, retry_backoff_seconds=0)
    url = f"https://web-api.tp.entsoe.eu/api?documentType=A44&securityToken={FAKE_KEY}"
    response = requests.Response()
    response.status_code = 503
    server_error = requests.HTTPError(f"503 Server Error for url: {url}", response=response)

    def failing_request(*args: object, **kwargs: object) -> None:
        # entsoe-py logs the request parameters, key included, at DEBUG level.
        logging.getLogger("entsoe.entsoe").debug("Request params %s", {"securityToken": FAKE_KEY})
        raise server_error

    caplog.set_level(logging.DEBUG)
    with patch(ENTSOE_CLIENT) as client_class:
        query = client_class.return_value.query_day_ahead_prices
        query.side_effect = failing_request
        with pytest.raises(RuntimeError, match=r"securityToken=\*\*\*") as raised:
            fetch_day_ahead_prices(params, SecretStr(FAKE_KEY))

    error = raised.value
    assert FAKE_KEY not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert FAKE_KEY not in caplog.text
    assert "securityToken=***" in caplog.text
    assert query.call_count == 2


def _eex_workbook(header: list[object], rows: list[list[object]]) -> bytes:
    """EEX-like workbook: a few title rows, then `header` and `rows`."""
    title_rows = [["Emission Spot Primary Market Auction Report"]] + [[]] * (EEX_TITLE_ROWS - 1)
    sheet = pd.DataFrame(title_rows + [header] + rows)
    workbook = io.BytesIO()
    sheet.to_excel(workbook, header=False, index=False)
    return workbook.getvalue()


def _eex_archive(workbook: bytes) -> bytes:
    """Zip laid out like the EEX archive: one folder, one workbook per year."""
    archive = io.BytesIO()
    folder = "emission-spot-primary-market-auction-report-2012-2024-data"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            f"{folder}/emission-spot-primary-market-auction-report-2024-data.xlsx", workbook
        )
    return archive.getvalue()


def _eex_answer(url: str, timeout: float) -> MagicMock:
    """Fake EEX server: the archive (2024, old layout) and the 2025 report (new layout)."""
    archived_rows = [
        [None, "31/12/2024", "09:00", "EU Auction", "T3PA", 66.0],
        [None, "31/12/2024", "11:00", "PL Auction", "T3PA", None],  # cancelled: no price
        [None, "Prices in EUR per tonne", None, None, None, None],  # note below the table
    ]
    recent_rows = [
        [pd.Timestamp("2025-01-02"), "09:00", "Auction CAP3 EU", "T3PA", "successful", 68.0],
        [pd.Timestamp("2025-01-03"), "09:00", "Auction CAP3 EU", "T3PA", "successful", 70.0],
        [pd.Timestamp("2025-01-03"), "09:00", "Auction DE", "T3PA", "successful", 72.0],
        [pd.Timestamp("2025-01-03"), "09:00", "Auction EUAA", "T3AA", "successful", 99.0],
        [pd.Timestamp("2025-01-06"), "09:00", "Auction PL", "T3PA", "cancelled", 99.0],
        [pd.Timestamp("2025-01-06"), "09:00", "Auction CAP3 EU", "T3PA", "successful", 75.0],
    ]
    is_archive = url == EEX_ARCHIVE_URL
    content = (
        _eex_archive(_eex_workbook(EEX_ARCHIVED_HEADER, archived_rows))
        if is_archive
        else _eex_workbook(EEX_HEADER, recent_rows)
    )
    return MagicMock(status_code=200, content=content)


def test_gas_and_carbon_prices_sit_on_their_brussels_trading_day() -> None:
    """Price of trading day D on the 24 hours of D in Brussels, NaN without session, no shift.

    EEX: archived (2024) and recent (2025) report layouts, same-day auctions averaged.
    """
    params = make_ingestion_params(
        start_date="2024-12-31",
        end_date="2025-01-06",
        eua_archive_url=EEX_ARCHIVE_URL,
        eua_archive_last_year=2024,
    )
    trading_days = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"], name=TRADING_DATE)
    ttf = pd.DataFrame({TTF: [40.0, 41.0, 43.0]}, index=trading_days)

    with patch(f"{NODES}.requests.get", side_effect=_eex_answer):
        eua = fetch_eua_prices(params)
    sources = make_raw_sources(params)
    master = build_master(**{**sources, "ttf": ttf, "eua": eua}, params=params)

    assert eua[EUA].to_dict() == {
        pd.Timestamp("2024-12-31"): 66.0,
        pd.Timestamp("2025-01-02"): 68.0,
        pd.Timestamp("2025-01-03"): 71.0,
        pd.Timestamp("2025-01-06"): 75.0,
    }
    friday_hours = master.index[
        (master.index >= _utc("2025-01-02 23:00")) & (master.index < _utc("2025-01-03 23:00"))
    ]
    assert len(friday_hours) == 24
    assert (master.loc[friday_hours, TTF] == 41.0).all()
    assert (master.loc[friday_hours, EUA] == 71.0).all()
    assert master.loc[_utc("2025-01-02 22:00"), TTF] == 40.0  # Thursday 23:00 in Brussels
    saturday_hours = (master.index >= _utc("2025-01-03 23:00")) & (
        master.index < _utc("2025-01-04 23:00")
    )
    assert master.loc[saturday_hours, [TTF, EUA]].isna().all().all()
