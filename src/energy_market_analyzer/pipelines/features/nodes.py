"""Features pipeline nodes: one feature table per strategy.

Information rule (decisions.md, section 6): for a day delivered on J+1, every
input of models A and B is known at 11:00 Brussels on the decision day J.

Words used in the column names, relative to the decision day J:
- today: J. Its day-ahead prices are known (published on J-1 around 12:45);
  its realized load, generation, flows and weather are not.
- yesterday: J-1, the last complete day of realized values.
- week_ago: the predicted day minus 7 days.
- last close / last price: the last gas or carbon price strictly before the
  reference day. The session of day J closes after 11:00.

Days are calendar days in Brussels time, never 24 rows of the UTC grid: on the
day of the autumn clock change, a 24-row shift would give the last hour the
price of midnight of the same day, published after the decision.

Tables:
- features_model_a: one row per delivered hour (UTC); target = its price.
- features_model_b: one row per (decision_day, horizon_days); target = mean
  price of the day decision_day + horizon_days.
- features_model_c: one row per hour, causal: the value of hour t only uses
  data up to t, and fuel prices of the previous days.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import holidays
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from energy_market_analyzer.pipelines.ingestion.nodes import (
    EUA,
    LOAD,
    LOAD_FORECAST,
    NET_IMPORT_TOTAL,
    ONE_DAY,
    PRICE,
    TTF,
    UTC_ZONE,
    WEATHER_VARIABLES,
    WEEKEND_FIRST_DAY,
)

logger = logging.getLogger(__name__)

# --- Index names and durations ---
DAY = "day"
DECISION_DAY = "decision_day"
HORIZON_DAYS = "horizon_days"
ONE_HOUR = pd.Timedelta(hours=1)
ONE_WEEK = pd.Timedelta(days=7)

# --- Targets ---
TARGET_PRICE = "target_price_eur_mwh"
TARGET_MEAN_PRICE = "target_mean_price_eur_mwh"

# --- Calendar of the predicted day, in Brussels time ---
HOUR = "hour"
DAY_OF_WEEK = "day_of_week"
IS_WEEKEND = "is_weekend"
IS_HOLIDAY = "is_holiday"
MONTH = "month"
DAY_CALENDAR = (DAY_OF_WEEK, IS_WEEKEND, IS_HOLIDAY, MONTH)

# --- Hourly values ---
PRICE_CHANGE = "price_change_1h"
PRICE_VOLATILITY = "price_volatility"
RESIDUAL_LOAD = "residual_load_mw"
GAS_TO_LOAD = "gas_to_load"
NUCLEAR_TO_LOAD = "nuclear_to_load"
WIND_SOLAR_TO_LOAD = "wind_solar_to_load"
RATIOS_TO_LOAD = (GAS_TO_LOAD, NUCLEAR_TO_LOAD, WIND_SOLAR_TO_LOAD)
TTF_LAST = "ttf_last_close_eur_mwh"
EUA_LAST = "eua_last_price_eur_t"
CLEAN_SPARK_SPREAD = "clean_spark_spread"

# --- Daily market: one row per calendar day ---
PRICE_MEAN = "price_mean"
PRICE_STATISTICS = {PRICE_MEAN: "mean", "price_min": "min", "price_max": "max", "price_std": "std"}
LOAD_MEAN = "load_mean_mw"
RESIDUAL_LOAD_MEAN = "residual_load_mean_mw"
NET_IMPORT_MEAN = "net_import_mean_mw"
TTF_AGE = "ttf_age_days"
EUA_AGE = "eua_age_days"

# --- Known at 11:00 on the decision day (models A and B) ---
TODAY_PRICES = {name: f"{name}_today" for name in PRICE_STATISTICS}
PRICE_MEAN_CHANGE_TODAY = "price_mean_change_today"
YESTERDAY_REALIZED = {
    LOAD_MEAN: "load_mean_yesterday_mw",
    RESIDUAL_LOAD_MEAN: "residual_load_mean_yesterday_mw",
    NET_IMPORT_MEAN: "net_import_mean_yesterday_mw",
    GAS_TO_LOAD: "gas_to_load_yesterday",
    NUCLEAR_TO_LOAD: "nuclear_to_load_yesterday",
    WIND_SOLAR_TO_LOAD: "wind_solar_to_load_yesterday",
}
CLEAN_SPARK_SPREAD_TODAY = "clean_spark_spread_today"
KNOWN_AT_DECISION = (
    *TODAY_PRICES.values(),
    PRICE_MEAN_CHANGE_TODAY,
    *YESTERDAY_REALIZED.values(),
    TTF_LAST,
    EUA_LAST,
    CLEAN_SPARK_SPREAD_TODAY,
)
PRICE_SAME_HOUR_TODAY = "price_same_hour_today"
PRICE_SAME_HOUR_WEEK_AGO = "price_same_hour_week_ago"
PRICE_MEAN_WEEK_AGO = "price_mean_week_ago"

# --- Feature lists: single source of truth for the tables, reports and tests ---
FEATURES_MODEL_A = (
    HOUR,
    *DAY_CALENDAR,
    PRICE_SAME_HOUR_TODAY,
    PRICE_SAME_HOUR_WEEK_AGO,
    LOAD_FORECAST,
    *KNOWN_AT_DECISION,
)
FEATURES_MODEL_B = (*DAY_CALENDAR, PRICE_MEAN_WEEK_AGO, *KNOWN_AT_DECISION)
FEATURES_MODEL_C = (
    PRICE,
    PRICE_CHANGE,
    PRICE_VOLATILITY,
    LOAD,
    RESIDUAL_LOAD,
    NET_IMPORT_TOTAL,
    *RATIOS_TO_LOAD,
    TTF_LAST,
    EUA_LAST,
    CLEAN_SPARK_SPREAD,
    *WEATHER_VARIABLES,
    HOUR,
    *DAY_CALENDAR,
)

# --- Reports ---
CORRELATION_REPORT_THRESHOLD = 0.9
ADF_COLUMNS = (PRICE, PRICE_CHANGE, PRICE_VOLATILITY, GAS_TO_LOAD, CLEAN_SPARK_SPREAD, TTF_LAST)
ADF_MAX_LAG_HOURS = 48  # two days of hourly lags
ADF_SIGNIFICANCE = 0.05


# ==================================================
# Calendar days
# ==================================================


def market_days(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    """Calendar day in market time of each UTC instant, as dates without time zone.

    Args:
        index: UTC instants.
        timezone: Market time zone (Europe/Brussels).

    Returns:
        Local midnights without time zone, one per instant. The 23 or 25 hours
        of a clock-change day share the same day.
    """
    local_index = index.tz_convert(timezone)
    local_midnights = local_index.normalize()
    # Calendar dates, not instants: the one case without time zone (conventions.md, V3-1).
    return local_midnights.tz_localize(None)


def _calendar(days: pd.DatetimeIndex, params: dict[str, Any]) -> pd.DataFrame:
    """Day of week, weekend, public holiday and month of each calendar day.

    Known: in advance, for any predicted day.
    """
    years = sorted(set(days.year))
    holiday_calendar = holidays.country_holidays(params["holiday_country"], years=years)
    holiday_days = pd.DatetimeIndex(sorted(holiday_calendar))
    is_weekend = days.dayofweek >= WEEKEND_FIRST_DAY
    is_holiday = days.isin(holiday_days)
    calendar = {
        DAY_OF_WEEK: days.dayofweek,
        IS_WEEKEND: is_weekend.astype(int),
        IS_HOLIDAY: is_holiday.astype(int),
        MONTH: days.month,
    }
    return pd.DataFrame(calendar, index=days)


# ==================================================
# Master checks and hourly values
# ==================================================


def _require_master(master: pd.DataFrame, params: dict[str, Any]) -> None:
    """Raise ValueError unless `master` is the continuous hourly UTC table of the ingestion."""
    index = master.index
    is_utc = isinstance(index, pd.DatetimeIndex) and str(index.tz) == UTC_ZONE
    if not is_utc:
        raise ValueError("master: index must be a DatetimeIndex in UTC, re-run the ingestion")
    steps = index[1:] - index[:-1]
    is_continuous = bool((steps == ONE_HOUR).all())
    if not is_continuous:
        raise ValueError("master: index must be a continuous hourly grid, re-run the ingestion")
    generation = [params["gas_column"], params["nuclear_column"], *params["wind_solar_columns"]]
    required = [PRICE, LOAD, LOAD_FORECAST, NET_IMPORT_TOTAL, TTF, EUA, *WEATHER_VARIABLES]
    missing_columns = [name for name in [*required, *generation] if name not in master.columns]
    if missing_columns:
        raise ValueError(f"master: missing columns {missing_columns}, re-run the ingestion")


def _wind_solar(master: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Wind (onshore and offshore) plus solar generation per hour, NaN if one is missing."""
    return master[params["wind_solar_columns"]].sum(axis=1, skipna=False)


def _ratio_to_load(values: pd.Series, load: pd.Series) -> pd.Series:
    """values / load; NaN where the load is missing or not positive."""
    positive_load = load.where(load > 0)
    return values / positive_load


def _hourly_fundamentals(master: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Residual load and ratios of gas, nuclear and wind + solar to the load, per hour.

    Ratios are taken to the load, not to total generation: the total depends on
    irregularly published types ("Other" almost empty in 2021-2022). A missing
    value stays missing (V3-4).

    Known: realized values, after the hour.
    """
    load = master[LOAD]
    wind_solar = _wind_solar(master, params)
    residual_load = load - wind_solar
    fundamentals = {
        RESIDUAL_LOAD: residual_load,
        GAS_TO_LOAD: _ratio_to_load(master[params["gas_column"]], load),
        NUCLEAR_TO_LOAD: _ratio_to_load(master[params["nuclear_column"]], load),
        WIND_SOLAR_TO_LOAD: _ratio_to_load(wind_solar, load),
    }
    return pd.DataFrame(fundamentals)


def clean_spark_spread(
    price: pd.Series, ttf: pd.Series, eua: pd.Series, params: dict[str, Any]
) -> pd.Series:
    """Margin of a gas plant per MWh sold, decision D7.

    price - heat_rate x gas price - heat_rate x emission factor x carbon price.

    Args:
        price: Electricity price (EUR/MWh).
        ttf: Gas price (EUR/MWh of gas).
        eua: Carbon price (EUR/t).
        params: Features parameters (heat_rate, emission_factor_t_per_mwh_gas).

    Returns:
        The clean spark spread (EUR/MWh), NaN where an input is missing.
    """
    gas_cost = params["heat_rate"] * ttf
    carbon_cost = params["heat_rate"] * params["emission_factor_t_per_mwh_gas"] * eua
    return price - gas_cost - carbon_cost


# ==================================================
# Daily market
# ==================================================


def _complete_stat(values: pd.Series, keys: Any, statistic: str) -> pd.Series:
    """Statistic of each group; NaN if one value of the group is missing.

    A day with a missing hour has no exact mean, minimum or maximum: it stays
    NaN rather than being computed on the hours left (V3-4).
    """
    grouped = values.groupby(keys)
    group_values = grouped.agg(statistic)
    is_complete = grouped.count() == grouped.size()
    return group_values.where(is_complete)


def _daily_prices(master: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Mean, minimum, maximum and standard deviation of the hourly prices of each day."""
    prices = master[PRICE]
    statistics = {
        name: _complete_stat(prices, days, statistic)
        for name, statistic in PRICE_STATISTICS.items()
    }
    return pd.DataFrame(statistics)


def _daily_realized(
    master: pd.DataFrame, days: pd.DatetimeIndex, params: dict[str, Any]
) -> pd.DataFrame:
    """Daily load, residual load, net imports and ratios to the load, complete days only.

    Daily ratios divide daily energies (sums), so each hour weighs its MWh.
    """
    load = master[LOAD]
    wind_solar = _wind_solar(master, params)
    residual_load = load - wind_solar
    load_energy = _complete_stat(load, days, "sum")
    gas_energy = _complete_stat(master[params["gas_column"]], days, "sum")
    nuclear_energy = _complete_stat(master[params["nuclear_column"]], days, "sum")
    wind_solar_energy = _complete_stat(wind_solar, days, "sum")
    realized = {
        LOAD_MEAN: _complete_stat(load, days, "mean"),
        RESIDUAL_LOAD_MEAN: _complete_stat(residual_load, days, "mean"),
        NET_IMPORT_MEAN: _complete_stat(master[NET_IMPORT_TOTAL], days, "mean"),
        GAS_TO_LOAD: _ratio_to_load(gas_energy, load_energy),
        NUCLEAR_TO_LOAD: _ratio_to_load(nuclear_energy, load_energy),
        WIND_SOLAR_TO_LOAD: _ratio_to_load(wind_solar_energy, load_energy),
    }
    return pd.DataFrame(realized)


def _session_prices(hourly_prices: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Price of each day's session, NaN on days without a session.

    The master places a session price on every hour of its day.
    """
    return hourly_prices.groupby(days).first()


def last_price_before(session_prices: pd.Series, max_age_days: int) -> pd.DataFrame:
    """Last session price strictly before each day, and its age in days.

    The session of a day closes after the 11:00 decision: a day only sees the
    previous sessions. No price is carried more than `max_age_days` after its
    session: older, the value is NaN (limited forward fill, V3-4).

    Args:
        session_prices: Price per consecutive calendar day, NaN without a session.
        max_age_days: Age of the oldest usable session, in days.

    Returns:
        Columns `price` and `age_days`, indexed like `session_prices`.
    """
    days = session_prices.index.to_series()
    session_days = days.where(session_prices.notna())
    last_session_day = session_days.ffill().shift(1)
    carried_price = session_prices.ffill().shift(1)
    age_days = (days - last_session_day).dt.days
    is_too_old = age_days > max_age_days
    last_price = carried_price.mask(is_too_old)
    used_age = age_days.mask(is_too_old)
    return pd.DataFrame({"price": last_price, "age_days": used_age})


def _daily_fuel(
    master: pd.DataFrame, days: pd.DatetimeIndex, params: dict[str, Any]
) -> pd.DataFrame:
    """Last gas close and last carbon price before each day, with their age in days."""
    ttf_sessions = _session_prices(master[TTF], days)
    eua_sessions = _session_prices(master[EUA], days)
    ttf = last_price_before(ttf_sessions, params["ttf_max_age_days"])
    eua = last_price_before(eua_sessions, params["eua_max_age_days"])
    fuel = {
        TTF_LAST: ttf["price"],
        TTF_AGE: ttf["age_days"],
        EUA_LAST: eua["price"],
        EUA_AGE: eua["age_days"],
    }
    return pd.DataFrame(fuel)


def build_daily_market(master: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """One row per calendar day in Brussels: prices, realized values and fuel prices.

    The price and realized columns describe day d itself. The fuel columns hold
    the last prices strictly before d; the clean spark spread of d pairs the
    prices of d, auctioned on d-1, with them.

    Args:
        master: Hourly master table (ingestion).
        params: Features parameters.

    Returns:
        A table indexed by `day` (local date, without time zone).

    Raises:
        ValueError: If the master table is not the expected hourly UTC table.
    """
    _require_master(master, params)
    days = market_days(master.index, params["market_timezone"])
    daily_prices = _daily_prices(master, days)
    daily_realized = _daily_realized(master, days, params)
    daily_fuel = _daily_fuel(master, days, params)
    daily = pd.concat([daily_prices, daily_realized, daily_fuel], axis=1)
    spread = clean_spark_spread(daily[PRICE_MEAN], daily[TTF_LAST], daily[EUA_LAST], params)
    daily_market = daily.assign(**{CLEAN_SPARK_SPREAD: spread})
    days_without_price = int(daily[PRICE_MEAN].isna().sum())
    logger.info(
        "Daily market: %d days, %d without a complete price", len(daily), days_without_price
    )
    return daily_market.rename_axis(DAY)


# ==================================================
# Models A and B: known at 11:00 on the decision day
# ==================================================


def _known_at_decision(daily_market: pd.DataFrame) -> pd.DataFrame:
    """Features known at 11:00 on each decision day J, indexed by J.

    Known at 11:00 J: the day-ahead prices of J (published on J-1), the
    realized values of the complete day J-1, the last fuel prices before J.
    """
    days = daily_market.index
    previous_days = days - ONE_DAY
    today_prices = daily_market[list(TODAY_PRICES)].rename(columns=TODAY_PRICES)
    previous_price_mean = daily_market[PRICE_MEAN].reindex(previous_days).to_numpy()
    price_mean_change = daily_market[PRICE_MEAN] - previous_price_mean
    previous_realized = daily_market[list(YESTERDAY_REALIZED)].reindex(previous_days)
    yesterday = previous_realized.set_axis(days).rename(columns=YESTERDAY_REALIZED)
    known = pd.concat([today_prices, yesterday, daily_market[[TTF_LAST, EUA_LAST]]], axis=1)
    known_at_decision = known.assign(
        **{
            PRICE_MEAN_CHANGE_TODAY: price_mean_change,
            CLEAN_SPARK_SPREAD_TODAY: daily_market[CLEAN_SPARK_SPREAD],
        }
    )
    return known_at_decision.rename_axis(DECISION_DAY)


def _lookup(table: pd.Series, days: pd.DatetimeIndex, hours: pd.Index) -> np.ndarray:
    """Values of a (day, clock hour) table for each pair (day, hour); NaN if absent."""
    keys = pd.MultiIndex.from_arrays([days, hours])
    return table.reindex(keys).to_numpy()


def build_features_model_a(
    master: pd.DataFrame, daily_market: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Model A: features of each delivered hour of J+1, known at 11:00 on J.

    Known at 11:00 J: calendar of the delivered hour; prices of J (same clock
    hour, daily statistics, change of the daily mean vs J-1); price of the same
    clock hour one week before the delivered day; ENTSO-E load forecast of the
    delivered hour (published before 11:00, decision D3); realized values of
    the complete day J-1; last fuel prices before J.

    Same clock hour: on the spring clock change there is no 02:00 (NaN); on the
    autumn one, the two 02:00 hours are averaged.

    Args:
        master: Hourly master table (ingestion).
        daily_market: Output of `build_daily_market`.
        params: Features parameters.

    Returns:
        One row per delivered hour (UTC index): FEATURES_MODEL_A, then TARGET_PRICE.
    """
    _require_master(master, params)
    timezone = params["market_timezone"]
    delivery_days = market_days(master.index, timezone)
    decision_days = delivery_days - ONE_DAY
    clock_hours = master.index.tz_convert(timezone).hour
    by_day_and_hour = _complete_stat(master[PRICE], [delivery_days, clock_hours], "mean")
    calendar = _calendar(delivery_days, params).set_axis(master.index)
    known_by_day = _known_at_decision(daily_market).reindex(decision_days)
    known = known_by_day.set_axis(master.index)
    hourly = calendar.assign(
        **{
            HOUR: clock_hours.to_numpy(),
            PRICE_SAME_HOUR_TODAY: _lookup(by_day_and_hour, decision_days, clock_hours),
            PRICE_SAME_HOUR_WEEK_AGO: _lookup(
                by_day_and_hour, delivery_days - ONE_WEEK, clock_hours
            ),
            LOAD_FORECAST: master[LOAD_FORECAST],
            TARGET_PRICE: master[PRICE],
        }
    )
    table = pd.concat([hourly, known], axis=1)
    logger.info(
        "Model A features: %d delivered hours x %d features", len(table), len(FEATURES_MODEL_A)
    )
    return table[[*FEATURES_MODEL_A, TARGET_PRICE]]


def _horizon_rows(
    known: pd.DataFrame, daily_market: pd.DataFrame, horizon: int, params: dict[str, Any]
) -> pd.DataFrame:
    """Model B rows of one horizon: decision day J, predicted day J + horizon.

    Raises:
        ValueError: If the week-ago price of the predicted day falls after J.
    """
    is_week_ago_known = horizon * ONE_DAY <= ONE_WEEK
    if not is_week_ago_known:
        raise ValueError(f"horizon {horizon}: the week-ago price would not be known on J")
    decision_days = known.index
    target_days = decision_days + horizon * ONE_DAY
    is_inside_data = target_days.isin(daily_market.index)
    calendar = _calendar(target_days, params).set_axis(decision_days)
    week_ago_mean = daily_market[PRICE_MEAN].reindex(target_days - ONE_WEEK).to_numpy()
    target_mean = daily_market[PRICE_MEAN].reindex(target_days).to_numpy()
    columns = pd.concat([calendar, known], axis=1)
    rows = columns.assign(
        **{
            PRICE_MEAN_WEEK_AGO: week_ago_mean,
            TARGET_MEAN_PRICE: target_mean,
            HORIZON_DAYS: horizon,
        }
    )
    inside_rows = rows[is_inside_data]
    return inside_rows.set_index(HORIZON_DAYS, append=True)


def build_features_model_b(daily_market: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Model B: features of the daily mean price of J + horizon, known at 11:00 on J.

    Known at 11:00 J: calendar of the predicted day; prices of J; mean price of
    the predicted day one week before (on or before J); realized values of the
    complete day J-1; last fuel prices before J. No load forecast: ENTSO-E only
    publishes it for J+1.

    The target is NaN when one hour of the predicted day has no price.

    Args:
        daily_market: Output of `build_daily_market`.
        params: Features parameters (multi_day_horizons: 2, 3 and 4 days).

    Returns:
        One row per (decision_day, horizon_days), predicted day inside the data:
        FEATURES_MODEL_B, then TARGET_MEAN_PRICE.
    """
    known = _known_at_decision(daily_market)
    horizon_tables = [
        _horizon_rows(known, daily_market, horizon, params)
        for horizon in params["multi_day_horizons"]
    ]
    table = pd.concat(horizon_tables).sort_index()
    logger.info("Model B features: %d rows x %d features", len(table), len(FEATURES_MODEL_B))
    return table[[*FEATURES_MODEL_B, TARGET_MEAN_PRICE]]


# ==================================================
# Model C: hourly and causal
# ==================================================


def build_features_model_c(
    master: pd.DataFrame, daily_market: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Model C: one row per hour, for regimes, anomalies and retrospective analysis.

    Causal: the value of hour t only uses data up to t. The price change and
    its volatility look backwards; fuel prices are the last ones before the day
    of t, and the clean spark spread pairs them with the price of hour t,
    auctioned the day before.

    Known: price columns of day D at 12:45 on D-1, fuel columns at the end of
    D-1, realized values (load, generation, flows, weather) after the hour.

    Args:
        master: Hourly master table (ingestion).
        daily_market: Output of `build_daily_market`.
        params: Features parameters (volatility_window_hours, ...).

    Returns:
        One row per hour (UTC index), columns FEATURES_MODEL_C.
    """
    _require_master(master, params)
    timezone = params["market_timezone"]
    days = market_days(master.index, timezone)
    price = master[PRICE]
    price_change = price.diff()
    window = params["volatility_window_hours"]
    volatility = price_change.rolling(window, min_periods=window).std()
    fuel = daily_market[[TTF_LAST, EUA_LAST]].reindex(days).set_axis(master.index)
    spread = clean_spark_spread(price, fuel[TTF_LAST], fuel[EUA_LAST], params)
    calendar = _calendar(days, params).set_axis(master.index)
    fundamentals = _hourly_fundamentals(master, params)
    observed = master[[PRICE, LOAD, NET_IMPORT_TOTAL, *WEATHER_VARIABLES]]
    columns = pd.concat([observed, fundamentals, fuel, calendar], axis=1)
    table = columns.assign(
        **{
            PRICE_CHANGE: price_change,
            PRICE_VOLATILITY: volatility,
            CLEAN_SPARK_SPREAD: spread,
            HOUR: master.index.tz_convert(timezone).hour,
        }
    )
    logger.info("Model C features: %d hours x %d features", len(table), len(FEATURES_MODEL_C))
    return table[list(FEATURES_MODEL_C)]


# ==================================================
# Reports
# ==================================================


def _missing_summary(values: pd.Series) -> dict[str, Any]:
    """Number and share of missing values of one column."""
    missing_count = int(values.isna().sum())
    return {"count": missing_count, "pct": round(100 * missing_count / len(values), 2)}


def _json_label(value: Any) -> Any:
    """JSON value of one index label: ISO instant, calendar date or integer."""
    if isinstance(value, pd.Timestamp):
        is_calendar_date = value.tz is None
        return str(value.date()) if is_calendar_date else value.isoformat()
    return int(value)


def _index_boundary(index: pd.Index, position: int) -> Any:
    """First or last index label, one named entry per level of a MultiIndex."""
    label = index[position]
    if isinstance(index, pd.MultiIndex):
        return {name: _json_label(part) for name, part in zip(index.names, label, strict=True)}
    return _json_label(label)


def _table_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Rows, first and last index, missing values per column."""
    return {
        "rows": len(table),
        "first": _index_boundary(table.index, 0),
        "last": _index_boundary(table.index, -1),
        "missing": {name: _missing_summary(table[name]) for name in table.columns},
    }


def _fuel_summary(price: pd.Series, age_days: pd.Series) -> dict[str, Any]:
    """Days without a usable price and oldest session actually used."""
    oldest_age = age_days.max()
    return {
        "days_without_value": int(price.isna().sum()),
        "max_age_used_days": None if pd.isna(oldest_age) else int(oldest_age),
    }


def correlated_pairs(features: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    """Pairs of columns whose absolute Pearson correlation reaches `threshold`.

    Args:
        features: Feature columns.
        threshold: Minimum absolute correlation reported.

    Returns:
        One entry per pair, strongest first.
    """
    correlations = features.corr()
    columns = list(correlations.columns)
    pairs = []
    for position, first in enumerate(columns):
        for second in columns[position + 1 :]:
            correlation = correlations.loc[first, second]
            is_strong = abs(correlation) >= threshold
            if is_strong:
                pairs.append(
                    {"features": [first, second], "correlation": round(float(correlation), 3)}
                )
    return sorted(pairs, key=lambda pair: -abs(pair["correlation"]))


def _before_holdout_a(features_model_a: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Model A rows delivered before the reserved period."""
    delivery_days = market_days(features_model_a.index, params["market_timezone"])
    is_development = delivery_days < pd.Timestamp(params["holdout_start"])
    return features_model_a[is_development]


def _predicted_days_b(features_model_b: pd.DataFrame) -> pd.DatetimeIndex:
    """Predicted day of each model B row."""
    decision_days = features_model_b.index.get_level_values(DECISION_DAY)
    horizons = features_model_b.index.get_level_values(HORIZON_DAYS)
    return decision_days + pd.to_timedelta(horizons, unit="D")


def build_features_report(
    features_model_a: pd.DataFrame,
    features_model_b: pd.DataFrame,
    features_model_c: pd.DataFrame,
    daily_market: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Report of the feature tables: size, missing values, fuel prices, correlations.

    Correlations only use rows predicted before the reserved period (decision
    D9): they inform choices, so they never look at it.

    Args:
        features_model_a: Output of `build_features_model_a`.
        features_model_b: Output of `build_features_model_b`.
        features_model_c: Output of `build_features_model_c`.
        daily_market: Output of `build_daily_market`.
        params: Features parameters (holdout_start, market_timezone).

    Returns:
        A JSON-ready report.
    """
    holdout_start = pd.Timestamp(params["holdout_start"])
    development_a = _before_holdout_a(features_model_a, params)
    is_development_b = _predicted_days_b(features_model_b) < holdout_start
    development_b = features_model_b[is_development_b]
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "holdout_start": str(holdout_start.date()),
        "tables": {
            "features_model_a": _table_summary(features_model_a),
            "features_model_b": _table_summary(features_model_b),
            "features_model_c": _table_summary(features_model_c),
        },
        "fuel": {
            "ttf": _fuel_summary(daily_market[TTF_LAST], daily_market[TTF_AGE]),
            "eua": _fuel_summary(daily_market[EUA_LAST], daily_market[EUA_AGE]),
        },
        "correlated_pairs_before_holdout": {
            "threshold": CORRELATION_REPORT_THRESHOLD,
            "features_model_a": correlated_pairs(
                development_a[list(FEATURES_MODEL_A)], CORRELATION_REPORT_THRESHOLD
            ),
            "features_model_b": correlated_pairs(
                development_b[list(FEATURES_MODEL_B)], CORRELATION_REPORT_THRESHOLD
            ),
        },
    }
    _log_report(report)
    return report


def _log_report(report: dict[str, Any]) -> None:
    """One log line per table and per fuel."""
    for name, summary in report["tables"].items():
        worst_name, worst = max(summary["missing"].items(), key=lambda item: item[1]["count"])
        logger.info(
            "%s: %d rows, most missing: %s (%.2f%%)",
            name,
            summary["rows"],
            worst_name,
            worst["pct"],
        )
    for name, summary in report["fuel"].items():
        logger.info(
            "%s: %d days without a usable price, oldest session used %s days before",
            name,
            summary["days_without_value"],
            summary["max_age_used_days"],
        )


# ==================================================
# Diagnostic for notebook 2 (not wired in the pipeline)
# ==================================================


def _adf_result(values: pd.Series) -> dict[str, Any]:
    """Augmented Dickey-Fuller test of one series (lag chosen by AIC)."""
    observed = values.dropna()
    statistic, p_value, used_lag, observations, critical_values, _ = adfuller(
        observed, maxlag=ADF_MAX_LAG_HOURS, autolag="AIC"
    )
    return {
        "statistic": round(float(statistic), 3),
        "p_value": round(float(p_value), 4),
        "used_lag": int(used_lag),
        "observations": int(observations),
        "critical_values": {
            level: round(float(value), 3) for level, value in critical_values.items()
        },
        "is_stationary": bool(p_value < ADF_SIGNIFICANCE),
    }


def stationarity_report(features_model_c: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """ADF stationarity test of the main table C columns, before the reserved period.

    Diagnostic for notebook 2, not wired in the pipeline. Missing hours are
    dropped: the few gaps join their neighbours, which the test tolerates.

    Args:
        features_model_c: Output of `build_features_model_c`.
        params: Features parameters (holdout_start, market_timezone).

    Returns:
        One ADF result per column of ADF_COLUMNS.
    """
    days = market_days(features_model_c.index, params["market_timezone"])
    is_development = days < pd.Timestamp(params["holdout_start"])
    development = features_model_c[is_development]
    return {name: _adf_result(development[name]) for name in ADF_COLUMNS}
