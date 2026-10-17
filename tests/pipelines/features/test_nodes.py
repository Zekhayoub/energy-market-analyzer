"""Tests of the features nodes: the 11:00 rule, clock changes, fuel prices, missing values.

The master tables are built by the real `build_master` from synthetic sources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_market_analyzer.pipelines.features.nodes import (
    CLEAN_SPARK_SPREAD_TODAY,
    DECISION_DAY,
    EUA_LAST,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    FEATURES_MODEL_C,
    GAS_TO_LOAD,
    LOAD_MEAN,
    PRICE_MEAN,
    PRICE_SAME_HOUR_TODAY,
    PRICE_VOLATILITY,
    RESIDUAL_LOAD,
    TARGET_MEAN_PRICE,
    TTF_LAST,
    build_daily_market,
    build_features_model_a,
    build_features_model_b,
    build_features_model_c,
    market_days,
)
from energy_market_analyzer.pipelines.ingestion.nodes import EUA, LOAD, LOAD_FORECAST, PRICE, TTF
from tests.helpers import make_features_params, make_master

TIMEZONE = "Europe/Brussels"
ONE_DAY = pd.Timedelta(days=1)
# Columns known before their own hour: day-ahead prices, load forecast, fuel sessions.
PUBLISHED_IN_ADVANCE = (PRICE, LOAD_FORECAST, TTF, EUA)


def _local_start(day: str, days_after: int = 0) -> pd.Timestamp:
    """Midnight in Brussels of `day` plus `days_after` calendar days, as a UTC instant."""
    calendar_day = pd.Timestamp(day) + days_after * ONE_DAY
    local_midnight = calendar_day.tz_localize(TIMEZONE)
    return local_midnight.tz_convert("UTC")


def _all_tables(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily market and the three feature tables built from `master`."""
    params = make_features_params()
    daily_market = build_daily_market(master, params)
    return {
        "daily": daily_market,
        "a": build_features_model_a(master, daily_market, params),
        "b": build_features_model_b(daily_market, params),
        "c": build_features_model_c(master, daily_market, params),
    }


def _hide_after_decision(master: pd.DataFrame, decision_day: str) -> pd.DataFrame:
    """Replace by noise everything that is not known at 11:00 on `decision_day` (J).

    Unknown at 11:00 J: day-ahead prices of J+1 and after (published at 12:45 J),
    realized values of J and after, fuel sessions of J and after (closed after
    11:00), load forecasts of J+2 and after.
    """
    rng = np.random.default_rng(7)
    hidden = master.copy()
    hours = master.index
    realized_columns = [name for name in master.columns if name not in PUBLISHED_IN_ADVANCE]
    is_from_j = hours >= _local_start(decision_day)
    is_from_j1 = hours >= _local_start(decision_day, days_after=1)
    is_from_j2 = hours >= _local_start(decision_day, days_after=2)
    hidden.loc[is_from_j, realized_columns] = rng.uniform(
        0, 5000, (is_from_j.sum(), len(realized_columns))
    )
    hidden.loc[is_from_j, [TTF, EUA]] = rng.uniform(0, 500, (is_from_j.sum(), 2))
    hidden.loc[is_from_j1, PRICE] = rng.uniform(-500, 3000, is_from_j1.sum())
    hidden.loc[is_from_j2, LOAD_FORECAST] = rng.uniform(0, 20000, is_from_j2.sum())
    return hidden


@pytest.mark.parametrize(
    ("start_date", "end_date", "decision_day"),
    [
        ("2025-03-15", "2025-04-08", "2025-03-29"),  # J+1 = spring clock change (23 hours)
        ("2025-03-15", "2025-04-08", "2025-03-30"),  # J = spring clock change
        ("2025-10-12", "2025-11-05", "2025-10-25"),  # J+1 = autumn clock change (25 hours)
        ("2025-10-12", "2025-11-05", "2025-10-26"),  # J = autumn clock change
        ("2025-10-12", "2025-11-05", "2025-10-31"),  # J+1 = All Saints' Day, a Saturday
    ],
)
def test_models_a_and_b_only_see_what_is_known_at_11h(
    start_date: str, end_date: str, decision_day: str
) -> None:
    """Features of J+1 (model A) and of decision day J (model B) ignore the unknown."""
    master = make_master(start_date, end_date)
    tables = _all_tables(master)
    hidden_tables = _all_tables(_hide_after_decision(master, decision_day))
    delivery_days = market_days(master.index, TIMEZONE)
    is_delivered_j1 = delivery_days == pd.Timestamp(decision_day) + ONE_DAY
    model_a_rows = tables["a"].loc[is_delivered_j1, list(FEATURES_MODEL_A)]
    hidden_a_rows = hidden_tables["a"].loc[is_delivered_j1, list(FEATURES_MODEL_A)]
    model_b_rows = tables["b"].xs(pd.Timestamp(decision_day), level=DECISION_DAY)
    hidden_b_rows = hidden_tables["b"].xs(pd.Timestamp(decision_day), level=DECISION_DAY)

    pd.testing.assert_frame_equal(hidden_a_rows, model_a_rows)
    pd.testing.assert_frame_equal(
        hidden_b_rows[list(FEATURES_MODEL_B)], model_b_rows[list(FEATURES_MODEL_B)]
    )
    # The comparison is not made on empty rows (spring: no 02:00 on J, one NaN).
    assert model_a_rows.notna().to_numpy().mean() > 0.99
    assert model_b_rows[list(FEATURES_MODEL_B)].notna().all(axis=None)
    # And the features do react to what is known: the prices of J.
    is_day_j = delivery_days == pd.Timestamp(decision_day)
    repriced_master = master.copy()
    repriced_master.loc[is_day_j, PRICE] = master.loc[is_day_j, PRICE] + 50.0
    repriced_a_rows = _all_tables(repriced_master)["a"].loc[is_delivered_j1]
    assert not repriced_a_rows[list(FEATURES_MODEL_A)].equals(model_a_rows)


def test_model_c_only_uses_the_past() -> None:
    """The value of hour t only uses data up to t, and fuel sessions before the day of t."""
    master = make_master("2025-10-12", "2025-11-05")
    last_known_hour = pd.Timestamp("2025-10-27 09:00", tz="UTC")
    rng = np.random.default_rng(11)
    hidden = master.copy()
    is_after = master.index > last_known_hour
    is_same_day_or_after = master.index >= _local_start("2025-10-27")
    hidden.loc[is_after] = rng.uniform(0, 5000, (is_after.sum(), master.shape[1]))
    hidden.loc[is_same_day_or_after, [TTF, EUA]] = rng.uniform(
        0, 500, (is_same_day_or_after.sum(), 2)
    )

    model_c = _all_tables(master)["c"].loc[:last_known_hour]
    hidden_model_c = _all_tables(hidden)["c"].loc[:last_known_hour]

    pd.testing.assert_frame_equal(hidden_model_c, model_c)
    assert model_c.loc[last_known_hour, list(FEATURES_MODEL_C)].notna().all()
    assert model_c[PRICE_VOLATILITY].iloc[:24].isna().all()


def test_same_hour_follows_the_brussels_clock() -> None:
    """On 23- and 25-hour days, 'same hour today' is the same clock hour of J, never J+1."""
    master = make_master("2025-03-25", "2025-10-30")
    local_index = master.index.tz_convert(TIMEZONE)
    day_numbers = (local_index.normalize().tz_localize(None) - pd.Timestamp("2025-01-01")).days
    # Price = 100 x day number + clock hour: each value tells its day and hour.
    master[PRICE] = 100.0 * day_numbers + local_index.hour
    model_a = _all_tables(master)["a"]
    clock_hours = model_a.index.tz_convert(TIMEZONE).hour

    def price_of(day: str, hour: int) -> float:
        return 100.0 * (pd.Timestamp(day) - pd.Timestamp("2025-01-01")).days + hour

    autumn_change = model_a[market_days(model_a.index, TIMEZONE) == pd.Timestamp("2025-10-26")]
    last_hour = autumn_change.iloc[-1]
    assert len(autumn_change) == 25
    assert last_hour[PRICE_SAME_HOUR_TODAY] == price_of("2025-10-25", 23)
    # A 24-row shift would have given the price of 00:00 on the delivered day itself.
    assert master[PRICE].shift(24).loc[autumn_change.index[-1]] == price_of("2025-10-26", 0)
    day_after_spring_change = market_days(model_a.index, TIMEZONE) == pd.Timestamp("2025-03-31")
    spring_values = model_a.loc[day_after_spring_change, PRICE_SAME_HOUR_TODAY]
    spring_hours = clock_hours[day_after_spring_change]
    expected_spring = [
        price_of("2025-03-30", hour) if hour != 2 else np.nan for hour in spring_hours
    ]
    np.testing.assert_array_equal(spring_values.to_numpy(), expected_spring)


def test_fuel_is_the_last_session_before_the_decision_day() -> None:
    """Monday sees Friday's close; a session older than the age limit is not used."""
    master = make_master("2025-09-29", "2025-10-20")
    session_days = market_days(master.index, TIMEZONE)
    master[TTF] = np.nan
    master[EUA] = np.nan
    master.loc[session_days == pd.Timestamp("2025-10-10"), [TTF, EUA]] = [30.0, 80.0]  # Friday
    master.loc[session_days == pd.Timestamp("2025-10-13"), TTF] = 35.0  # Monday
    params = make_features_params(ttf_max_age_days=3, eua_max_age_days=3)

    daily = build_daily_market(master, params)
    model_a = build_features_model_a(master, daily, params)

    monday = daily.loc["2025-10-13"]
    tuesday = daily.loc["2025-10-14"]
    assert (monday[TTF_LAST], monday[EUA_LAST]) == (30.0, 80.0)
    assert tuesday[TTF_LAST] == 35.0
    assert np.isnan(tuesday[EUA_LAST])  # Friday's auction is 4 days old
    expected_spread = monday[PRICE_MEAN] - 2.0 * 30.0 - 2.0 * 0.202 * 80.0
    tuesday_delivery = model_a[market_days(model_a.index, TIMEZONE) == pd.Timestamp("2025-10-14")]
    assert tuesday_delivery[TTF_LAST].eq(30.0).all()
    assert tuesday_delivery[CLEAN_SPARK_SPREAD_TODAY].to_numpy() == pytest.approx(expected_spread)


def test_missing_hours_and_ratios_to_load() -> None:
    """One missing hour makes its day NaN; ratios divide by the load, exactly."""
    master = make_master("2025-10-01", "2025-10-20")
    days = market_days(master.index, TIMEZONE)
    is_oct_8 = days == pd.Timestamp("2025-10-08")
    is_oct_9 = days == pd.Timestamp("2025-10-09")
    gap_hour = master.index[is_oct_8][5]
    master.loc[gap_hour, "gen_gas_mw"] = np.nan
    master.loc[gap_hour, "gen_solar_mw"] = np.nan
    master.loc[master.index[is_oct_9][10], PRICE] = np.nan
    master.loc[days == pd.Timestamp("2025-10-10"), ["gen_gas_mw", LOAD]] = [3000.0, 10000.0]

    tables = _all_tables(master)

    daily = tables["daily"]
    assert np.isnan(daily.loc["2025-10-08", GAS_TO_LOAD])
    assert daily.loc["2025-10-08", LOAD_MEAN] > 0
    assert (
        daily.loc["2025-10-09", ["price_mean", "price_min", "price_max", "price_std"]].isna().all()
    )
    assert daily.loc["2025-10-10", GAS_TO_LOAD] == pytest.approx(0.3)
    assert np.isnan(tables["c"].loc[gap_hour, RESIDUAL_LOAD])
    model_b = tables["b"]
    assert np.isnan(model_b.loc[(pd.Timestamp("2025-10-07"), 2), TARGET_MEAN_PRICE])
    assert model_b.loc[(pd.Timestamp("2025-10-08"), 2), TARGET_MEAN_PRICE] > 0
