"""Tests of the anomaly nodes: past-only scores, anomalies relative to the regime, the
closest-regime fallback and the events.

Synthetic hourly markets with known regimes come from `make_hourly_market`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_market_analyzer.pipelines.anomaly.nodes import (
    ANOMALOUS_HOURS,
    ANOMALY_MODEL,
    ANOMALY_SCORE,
    ANOMALY_THRESHOLD,
    DURATION_HOURS,
    EVENT_ID,
    FIRST_HOUR,
    IS_ANOMALY,
    LAST_HOUR,
    PEAK_HOUR,
    PEAK_PRICE,
    PRICE_PERCENTILE,
    SCORE_COLUMNS,
    TOP_DRIVER,
    TOP_DRIVER_PERCENTILE,
    analyze_development_period,
    build_anomaly_inputs,
    build_anomaly_report,
    detect_anomalies,
    group_anomaly_events,
)
from energy_market_analyzer.pipelines.features.nodes import (
    CLEAN_SPARK_SPREAD,
    PRICE_CHANGE,
    market_days,
)
from energy_market_analyzer.pipelines.ingestion.nodes import PRICE
from energy_market_analyzer.pipelines.regime.nodes import REGIME_FILTERED, REGIME_SMOOTHED
from tests.helpers import MARKET_TIMEZONE, make_anomaly_params, make_hourly_market

CALM, VOLATILE, CRISIS = 0, 1, 2
# Eight months from 1 January 2019 (243 days); scored from March, reserved period from July.
MARKET_DAYS = 243
ONE_HOUR = pd.Timedelta(hours=1)


def _cycle(pattern: list[int], days: int) -> list[int]:
    """`pattern` of daily regimes repeated over `days` days."""
    repeats = days // len(pattern) + 1
    return (pattern * repeats)[:days]


def _scores(
    table_c: pd.DataFrame, regimes: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Inputs, hourly scores and refit log of the anomaly nodes."""
    inputs = build_anomaly_inputs(table_c, regimes, params)
    scores, refits = detect_anomalies(inputs, params)
    return inputs, scores, refits


def _report(table_c: pd.DataFrame, regimes: pd.DataFrame, params: dict) -> dict:
    """Full report, through every node of the pipeline."""
    inputs, scores, refits = _scores(table_c, regimes, params)
    development = analyze_development_period(inputs, scores, params)
    anomalies, events = group_anomaly_events(scores, inputs, params)
    return build_anomaly_report(anomalies, events, inputs, refits, development, params)


def _noise_after(
    table_c: pd.DataFrame, regimes: pd.DataFrame, last_known: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random values in table C after `last_known`, random regimes after its day."""
    rng = np.random.default_rng(7)
    is_after = table_c.index > last_known
    noisy_table_c = table_c.copy()
    noisy_table_c.loc[is_after] = rng.uniform(-500, 3000, (is_after.sum(), table_c.shape[1]))
    last_known_day = market_days(pd.DatetimeIndex([last_known]), MARKET_TIMEZONE)[0]
    is_after_day = regimes.index > last_known_day
    random_labels = pd.array(rng.integers(0, 3, is_after_day.sum()), dtype="Int8")
    noisy_regimes = regimes.copy()
    noisy_regimes.loc[is_after_day, REGIME_FILTERED] = random_labels
    return noisy_table_c, noisy_regimes


def _with_price(table_c: pd.DataFrame, hours: pd.DatetimeIndex, price: float) -> pd.DataFrame:
    """Table C with `price` at `hours`; price change and spread follow, fuel costs unchanged."""
    fuel = table_c[PRICE] - table_c[CLEAN_SPARK_SPREAD]
    prices = table_c[PRICE].copy()
    prices.loc[hours] = price
    return table_c.assign(
        **{PRICE: prices, PRICE_CHANGE: prices.diff(), CLEAN_SPARK_SPREAD: prices - fuel}
    )


def _local_hours(day: pd.Timestamp, first: int, count: int) -> pd.DatetimeIndex:
    """`count` consecutive hours of a Brussels day from local hour `first`, in UTC."""
    start = pd.Timestamp(day, tz=MARKET_TIMEZONE) + first * ONE_HOUR
    return pd.date_range(start, periods=count, freq="h").tz_convert("UTC")


def test_scores_only_use_the_past_and_the_filtered_regime() -> None:
    """Noise after hour t leaves every output up to t unchanged.

    t is the first hour of a Brussels month (22:00 UTC the day before) and a
    mid-month hour. The same noise moves the later hours, so the test bites.
    The smoothed regime is never read; changing hour t itself moves its score;
    a one-month training window only learns from the previous month; the
    development section of the report ignores the reserved period.
    """
    table_c, regimes = make_hourly_market(
        _cycle([CALM] * 10 + [VOLATILE] * 10 + [CRISIS] * 10, MARKET_DAYS)
    )
    params = make_anomaly_params()
    _, scores, _ = _scores(table_c, regimes, params)
    month_start = pd.Timestamp("2019-06-01", tz=MARKET_TIMEZONE).tz_convert("UTC")
    mid_month = pd.Timestamp("2019-06-17 10:00", tz="UTC")
    for last_known in (month_start, mid_month):
        noisy_table_c, noisy_regimes = _noise_after(table_c, regimes, last_known)
        _, noisy_scores, _ = _scores(noisy_table_c, noisy_regimes, params)
        is_known = scores.index <= last_known

        pd.testing.assert_frame_equal(noisy_scores[is_known], scores[is_known])
        assert not noisy_scores.loc[~is_known, ANOMALY_SCORE].equals(
            scores.loc[~is_known, ANOMALY_SCORE]
        )

    shuffled = regimes[REGIME_SMOOTHED].sample(frac=1, random_state=3).array
    smoothed_noise = regimes.assign(**{REGIME_SMOOTHED: shuffled})
    assert smoothed_noise[REGIME_SMOOTHED].dtype == regimes[REGIME_SMOOTHED].dtype
    _, smoothed_scores, _ = _scores(table_c, smoothed_noise, params)
    pd.testing.assert_frame_equal(smoothed_scores, scores)

    spike = pd.DatetimeIndex([mid_month])
    _, spiked_scores, _ = _scores(_with_price(table_c, spike, 2000.0), regimes, params)
    assert spiked_scores.loc[mid_month, ANOMALY_SCORE] > scores.loc[mid_month, ANOMALY_SCORE]
    assert spiked_scores.loc[mid_month, IS_ANOMALY]

    _, _, expanding_refits = _scores(table_c, regimes, params)
    _, _, windowed_refits = _scores(table_c, regimes, make_anomaly_params(training_window_months=1))
    may_hours = int((market_days(table_c.index, MARKET_TIMEZONE).month == 5).sum())
    for refits, is_windowed in ((expanding_refits, False), (windowed_refits, True)):
        june_training_hours = sum(
            record["training_hours"] for record in refits if record["month"] == "2019-06"
        )
        assert (june_training_hours <= may_hours) == is_windowed

    last_development_hour = pd.Timestamp(params["holdout_start"], tz=MARKET_TIMEZONE) - ONE_HOUR
    report = _report(table_c, regimes, params)
    noisy_table_c, noisy_regimes = _noise_after(table_c, regimes, last_development_hour)
    noisy_report = _report(noisy_table_c, noisy_regimes, params)
    assert noisy_report["choices_before_holdout"] == report["choices_before_holdout"]
    assert set(report["choices_before_holdout"]["training_window_comparison"]["candidates"]) == {
        "expanding",
        "1_months",
    }
    assert noisy_report["holdout_descriptive_only"] != report["holdout_descriptive_only"]


def test_anomalies_are_relative_to_the_regime() -> None:
    """200 EUR/MWh is flagged on a calm day, driven by the price, and ordinary on a crisis day.

    On these stationary regimes, each per-regime model flags about the
    contamination share of its hours; one model for all regimes over-flags the
    rarer, wider crisis regime.
    """
    states = _cycle([CALM] * 12 + [VOLATILE] * 4 + [CRISIS] * 4, MARKET_DAYS)
    table_c, regimes = make_hourly_market(states)
    days = regimes.index
    june = days[(days >= "2019-06-01") & (days < "2019-07-01")]
    calm_day = june[regimes.loc[june, REGIME_FILTERED].to_numpy() == CALM][0]
    crisis_day = june[regimes.loc[june, REGIME_FILTERED].to_numpy() == CRISIS][0]
    calm_plateau = _local_hours(calm_day, first=4, count=4)
    crisis_plateau = _local_hours(crisis_day, first=4, count=4)
    plateaus = calm_plateau.append(crisis_plateau)
    params = make_anomaly_params()
    inputs, scores, _ = _scores(_with_price(table_c, plateaus, 200.0), regimes, params)

    calm_middle = scores.loc[calm_plateau[1:3]]
    assert calm_middle[IS_ANOMALY].all()
    assert (calm_middle[TOP_DRIVER] == PRICE).all()
    assert (calm_middle[TOP_DRIVER_PERCENTILE] == 100.0).all()
    crisis_middle = scores.loc[crisis_plateau[1:3]]
    assert not crisis_middle[IS_ANOMALY].any()
    assert crisis_middle[PRICE_PERCENTILE].between(10, 90).all()

    contamination = params["contamination"]
    ablation = analyze_development_period(inputs, scores, params)["global_model_ablation"]
    for share in ablation["per_regime_models"].values():
        assert contamination / 2 <= share <= contamination * 2
    assert ablation["one_model_for_all_regimes"]["crisis"] > 2 * contamination


def test_regime_with_few_hours_is_scored_by_the_closest_regime() -> None:
    """A regime under min_training_hours (here 240 h) borrows the closest model in volatility order.

    Crisis (72 training hours) is scored by volatile; volatile (72 hours) between
    eligible calm and crisis is scored by crisis, the more volatile. The choice
    ignores the hours of the scored month. With no eligible regime nothing is
    scored, and a feature that is a calendar column or not in table C stops the run.
    """
    params = make_anomaly_params()
    march = pd.Timestamp("2019-03-01", tz=MARKET_TIMEZONE)
    cases = (
        (
            [CALM] * 15 + [VOLATILE] * 15 + [CALM] * 13 + [CRISIS] * 3 + [VOLATILE] * 13,
            CRISIS,
            VOLATILE,
        ),
        ([CALM] * 28 + [VOLATILE] * 3 + [CRISIS] * 28, VOLATILE, CRISIS),
    )
    for training_days, scarce, expected_model in cases:
        march_days = [scarce] * 10 + [CALM] * 11 + [CRISIS] * 10
        table_c, regimes = make_hourly_market(training_days + march_days)
        _, scores, _ = _scores(table_c, regimes, params)
        march_scores = scores[scores.index >= march]
        is_scarce = march_scores[REGIME_FILTERED] == scarce

        assert (march_scores.loc[is_scarce, ANOMALY_MODEL] == expected_model).all()
        assert (
            march_scores.loc[~is_scarce, ANOMALY_MODEL]
            == march_scores.loc[~is_scarce, REGIME_FILTERED]
        ).all()

    no_model_params = make_anomaly_params(min_training_hours=10**5)
    inputs, no_model_scores, _ = _scores(table_c, regimes, no_model_params)
    anomalies, events = group_anomaly_events(no_model_scores, inputs, no_model_params)
    assert no_model_scores[ANOMALY_SCORE].isna().all()
    assert no_model_scores[IS_ANOMALY].isna().all()
    assert events.empty

    for features, message in (
        (["hour", PRICE], "calendar"),
        (["load_mw", PRICE], "not in features_model_c"),
    ):
        with pytest.raises(ValueError, match=message):
            build_anomaly_inputs(table_c, regimes, make_anomaly_params(features=features))


def test_events_bridge_short_gaps_and_stop_at_unknown_hours() -> None:
    """Events over the 25-hour night of 26 October 2025, with event_max_gap_hours = 4.

    - 4 normal hours are bridged, 5 are not;
    - the duration counts real hours (7 hours from 00:00 to 05:00 that night);
    - an unscored hour ends an event, and so does holdout_start (27 October).
    """
    params = make_anomaly_params(holdout_start="2025-10-27")
    hours = pd.date_range("2025-10-25 22:00", "2025-10-27 23:00", freq="h", tz="UTC")
    anomalous = pd.DatetimeIndex(
        [
            "2025-10-25 22:00",
            "2025-10-25 23:00",
            "2025-10-26 04:00",  # 4 normal hours bridged
            "2025-10-26 10:00",  # 5 normal hours: new event
            "2025-10-26 22:00",  # last development hour (23:00 in Brussels)
            "2025-10-26 23:00",  # midnight of holdout_start in Brussels: new event
            "2025-10-27 12:00",
            "2025-10-27 14:00",  # an unscored hour between them
        ],
        tz="UTC",
    )
    unscored = pd.Timestamp("2025-10-27 13:00", tz="UTC")
    peak = pd.Timestamp("2025-10-25 23:00", tz="UTC")
    is_anomalous = hours.isin(anomalous)
    flags = pd.array(np.where(hours == unscored, None, is_anomalous), dtype="boolean")
    scores = pd.DataFrame(
        {
            REGIME_FILTERED: pd.array(np.ones(len(hours)), dtype="Int8"),
            ANOMALY_MODEL: pd.array(np.ones(len(hours)), dtype="Int8"),
            ANOMALY_SCORE: np.where(hours == peak, 0.7, np.where(is_anomalous, 0.6, 0.4)),
            ANOMALY_THRESHOLD: 0.55,
            IS_ANOMALY: flags,
            PRICE_PERCENTILE: 50.0,
            TOP_DRIVER: pd.array(np.where(is_anomalous, PRICE, None), dtype="string"),
            TOP_DRIVER_PERCENTILE: np.where(is_anomalous, 100.0, np.nan),
        },
        index=hours,
    )[list(SCORE_COLUMNS)]
    prices = pd.DataFrame({PRICE: np.arange(len(hours), dtype=float)}, index=hours)
    anomalies, events = group_anomaly_events(scores, prices, params)

    expected_events = [0, 0, 0, 1, 2, 3, 4, 5]
    assert anomalies.loc[anomalous, EVENT_ID].tolist() == expected_events
    assert anomalies.loc[~is_anomalous, EVENT_ID].isna().all()
    first_event = events.loc[0]
    assert first_event[FIRST_HOUR] == anomalous[0]
    assert first_event[LAST_HOUR] == anomalous[2]
    assert first_event[DURATION_HOURS] == 7
    assert first_event[ANOMALOUS_HOURS] == 3
    assert first_event[PEAK_HOUR] == peak
    assert first_event[PEAK_PRICE] == prices.loc[peak, PRICE]
    assert events.index.tolist() == list(range(6))
