"""Tests of the backtest nodes: schedules decided with what is known at 11:00 J, known answers
of the optimizer and of the alert rule, days left out, and a report whose choices ignore the
reserved period.

Predictions and alerts come from `make_backtest_market` (a synthetic model A).
"""

from __future__ import annotations

from dataclasses import replace
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from energy_market_analyzer.pipelines.backtest.nodes import (
    ALERT_RULE_NAIVE,
    ALERT_RULE_SELECTED,
    HOURS,
    OPTIMIZED_NAIVE,
    OPTIMIZED_SELECTED,
    PERFECT_FORESIGHT,
    PRED_NAIVE,
    PRED_SELECTED,
    SAVINGS_TOLERANCE_EUR,
    STRATEGIES,
    STRATEGY_NAMES,
    Client,
    Schedule,
    alert_rule_schedule,
    build_backtest_inputs,
    build_backtest_report,
    check_schedule,
    client_profile,
    compare_strategies,
    day_savings,
    moved_column,
    optimal_schedule,
    payback_summary,
    run_backtest,
    savings_column,
    split_days,
    strategy_schedule,
)
from energy_market_analyzer.pipelines.backtest.pipeline import create_pipeline
from energy_market_analyzer.pipelines.features.nodes import TARGET_PRICE
from energy_market_analyzer.pipelines.prediction.models import IS_CHEAPEST_HOUR, TARGET_DAY
from tests.helpers import (
    BACKTEST_HOLDOUT,
    make_alerts,
    make_backtest_market,
    make_backtest_params,
)

# One MWh of flexible energy per hour, at most two in one hour, moved by up to two hours.
SMALL_CLIENT = Client(hourly_load_mwh=10.0, flexible_mwh=1.0, max_flexible_mwh=2.0, window_hours=2)
SHIFT_COST = 5.0


def _run(predictions: pd.DataFrame, alerts: pd.DataFrame, params: dict) -> dict[str, Any]:
    """Every node of the pipeline, in order."""
    inputs, coverage = build_backtest_inputs(predictions, alerts, params)
    daily, sensitivity = run_backtest(inputs, params)
    report = build_backtest_report(daily, sensitivity, coverage, params)
    return {"inputs": inputs, "coverage": coverage, "daily": daily, "report": report}


@pytest.fixture(scope="module")
def market() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly predictions and alerts of a synthetic model A, March to November 2019."""
    return make_backtest_market()


@pytest.fixture(scope="module")
def outputs(market: tuple) -> dict[str, Any]:
    """Outputs of the whole pipeline on the synthetic predictions."""
    return _run(*market, make_backtest_params())


def test_schedules_only_use_what_is_known_at_eleven(market: tuple, outputs: dict) -> None:
    """Noise in the realized prices leaves every schedule unchanged but perfect foresight's:
    only the savings move, because the realized price evaluates the schedules.

    Perfect foresight is the only strategy that reads it to decide, and no strategy saves
    more than perfect foresight on any day. The schedules react to the predictions; the
    alerts need no realized level.
    """
    params = make_backtest_params()
    predictions, alerts = market
    rng = np.random.default_rng(4)
    noisy = predictions.assign(**{TARGET_PRICE: rng.uniform(-200, 800, len(predictions))})
    noisy_outputs = _run(noisy, alerts, params)
    daily = outputs["daily"]
    noisy_daily = noisy_outputs["daily"]
    deciding_on_predictions = [name for name in STRATEGY_NAMES if name != PERFECT_FORESIGHT]
    moved = [moved_column(name) for name in deciding_on_predictions]
    pd.testing.assert_frame_equal(noisy_daily[moved], daily[moved])
    for name in STRATEGY_NAMES:
        assert not noisy_daily[savings_column(name)].equals(daily[savings_column(name)])
    assert [strategy.name for strategy in STRATEGIES if strategy.signal == TARGET_PRICE] == [
        PERFECT_FORESIGHT
    ]
    ceiling = daily[savings_column(PERFECT_FORESIGHT)]
    for name in STRATEGY_NAMES:
        assert (daily[savings_column(name)] <= ceiling + SAVINGS_TOLERANCE_EUR).all()

    client = client_profile(params)
    day = pd.Timestamp("2019-06-12")
    known_day = split_days(outputs["inputs"])[day]
    noisy_day = split_days(noisy_outputs["inputs"])[day]
    for strategy in STRATEGIES:
        schedule = strategy_schedule(strategy, known_day, client, SHIFT_COST)
        noisy_schedule = strategy_schedule(strategy, noisy_day, client, SHIFT_COST)
        is_same = np.array_equal(schedule.flexible_load, noisy_schedule.flexible_load)
        assert is_same == (strategy.name != PERFECT_FORESIGHT)

    changed = predictions.copy()
    is_day = (changed[TARGET_DAY] == day).to_numpy()
    changed.loc[is_day, PRED_SELECTED] = changed.loc[is_day, PRED_SELECTED].to_numpy()[::-1]
    changed_daily = _run(changed, make_alerts(changed, params), params)["daily"]
    column = savings_column(OPTIMIZED_SELECTED)
    other_days = daily.index != day
    assert changed_daily.loc[day, column] != daily.loc[day, column]
    pd.testing.assert_series_equal(
        changed_daily.loc[other_days, column], daily.loc[other_days, column]
    )


def test_optimizer_gives_known_schedules_and_counts_losses() -> None:
    """The energy of the expensive hour goes to the cheap hour within the window, up to the
    maximum power; nothing moves beyond the window or for a gain below the shift cost; a
    wrong prediction loses money (no clipping). Days of 23 and 25 hours keep their energy."""
    prices = np.array([50.0, 50.0, 200.0, 50.0, 0.0, 50.0])
    schedule = optimal_schedule(prices, SMALL_CLIENT, SHIFT_COST)
    np.testing.assert_allclose(schedule.flexible_load, [1, 1, 0, 1, 2, 1], atol=1e-9)
    assert schedule.moved_mwh == pytest.approx(1.0)
    assert day_savings(schedule, prices, SMALL_CLIENT, SHIFT_COST) == pytest.approx(195.0)
    reversed_prices = np.array([50.0, 50.0, 0.0, 50.0, 200.0, 50.0])
    assert day_savings(schedule, reversed_prices, SMALL_CLIENT, SHIFT_COST) == pytest.approx(-205.0)

    costly = optimal_schedule(prices, SMALL_CLIENT, shift_cost=200.0)
    assert costly.moved_mwh == 0.0
    assert day_savings(costly, prices, SMALL_CLIENT, 200.0) == 0.0

    one_hour_window = replace(SMALL_CLIENT, window_hours=1)
    far_cheap_hour = np.array([200.0, 50.0, 50.0, 50.0, 50.0, 0.0])
    near_only = optimal_schedule(far_cheap_hour, one_hour_window, SHIFT_COST)
    np.testing.assert_allclose(near_only.flexible_load, [0, 2, 1, 1, 0, 2], atol=1e-9)

    rng = np.random.default_rng(1)
    for hours in (23, 25):
        day_prices = rng.normal(80.0, 30.0, hours)
        day_schedule = optimal_schedule(day_prices, SMALL_CLIENT, SHIFT_COST)
        check_schedule(day_schedule, SMALL_CLIENT)
        assert len(day_schedule.flexible_load) == hours
        assert day_savings(day_schedule, day_prices, SMALL_CLIENT, SHIFT_COST) > 0
    with pytest.raises(RuntimeError, match="Invalid schedule"):
        check_schedule(Schedule(np.array([1.0, 1.5, 1.0]), 0.5), SMALL_CLIENT)


def test_alert_rule_follows_the_published_alerts(market: tuple) -> None:
    """Most expensive flagged hour first, to the cheapest flagged hour within the window, up to
    the maximum power, only for a predicted gain above the shift cost. Alerts that do not come
    from the predictions (stale file, other hour counts) stop the pipeline."""
    prices = np.array([10.0, 100.0, 30.0, 90.0, 5.0, 80.0, 20.0, 70.0])
    expensive = np.array([False, True, False, True, False, False, False, False])
    cheapest = np.array([True, False, False, False, True, False, False, False])
    schedule = alert_rule_schedule(prices, cheapest, expensive, SMALL_CLIENT, SHIFT_COST)
    np.testing.assert_allclose(schedule.flexible_load, [2, 0, 1, 0, 2, 1, 1, 1])
    assert day_savings(schedule, prices, SMALL_CLIENT, SHIFT_COST) == pytest.approx(165.0)

    shared_cheap_hour = np.array([False, False, True, False, False, False, False, False])
    full = alert_rule_schedule(prices, shared_cheap_hour, expensive, SMALL_CLIENT, SHIFT_COST)
    np.testing.assert_allclose(full.flexible_load, [1, 0, 2, 1, 1, 1, 1, 1])
    costly = alert_rule_schedule(prices, cheapest, expensive, SMALL_CLIENT, shift_cost=88.0)
    np.testing.assert_allclose(costly.flexible_load, [2, 0, 1, 1, 1, 1, 1, 1])

    predictions, alerts = market
    params = make_backtest_params()
    stale = alerts.copy()
    first_hour = stale.index[0]
    stale.loc[first_hour, IS_CHEAPEST_HOUR] = not bool(stale.loc[first_hour, IS_CHEAPEST_HOUR])
    with pytest.raises(ValueError, match="does not match the selected predictions"):
        build_backtest_inputs(predictions, stale, params)
    with pytest.raises(ValueError, match="same hours"):
        build_backtest_inputs(predictions, alerts.iloc[1:], params)
    with pytest.raises(ValueError, match="does not match the selected predictions"):
        build_backtest_inputs(predictions, alerts, make_backtest_params(cheapest_hours=6))


def test_days_without_every_hour_price_and_prediction_are_left_out(market: tuple) -> None:
    """A day with a missing hour, a missing realized price or a missing prediction is left out
    for every strategy and counted by reason; clock-change days of 23 and 25 hours are kept."""
    predictions, _ = market
    params = make_backtest_params()
    missing_hour = pd.Timestamp("2019-05-02")
    missing_price = pd.Timestamp("2019-05-03")
    missing_naive = pd.Timestamp("2019-05-04")
    dropped_row = predictions.index[(predictions[TARGET_DAY] == missing_hour).to_numpy()][5]
    broken = predictions.drop(dropped_row)
    price_row = broken.index[(broken[TARGET_DAY] == missing_price).to_numpy()][7]
    naive_row = broken.index[(broken[TARGET_DAY] == missing_naive).to_numpy()][3]
    broken.loc[price_row, TARGET_PRICE] = np.nan
    broken.loc[naive_row, PRED_NAIVE] = np.nan

    result = _run(broken, make_alerts(broken, params), params)
    coverage = result["coverage"]
    daily = result["daily"]
    assert coverage["excluded_days"] == {
        "2019-05-02": "missing_hours",
        "2019-05-03": "missing_realized_price",
        "2019-05-04": "missing_prediction",
    }
    assert not daily.index.isin([missing_hour, missing_price, missing_naive]).any()
    assert coverage["evaluated_days"] == len(daily) == coverage["predicted_days"] - 3
    assert coverage["holdout_days"] == int((daily.index >= BACKTEST_HOLDOUT).sum())
    assert daily.loc["2019-03-31", HOURS] == 23
    assert daily.loc["2019-10-27", HOURS] == 25


def test_report_choices_ignore_the_reserved_period_and_payback_is_exact(
    market: tuple, outputs: dict
) -> None:
    """Noise in every delivered day of the reserved period changes nothing in the choices
    (rules of run 6 included); the reserved section moves. A leak of the realized prices into
    the reserved predictions raises the alarm. The yearly gain, its interval and the payback
    match a calculation by hand. The pipeline reads only predictions and alerts."""
    predictions, _ = market
    params = make_backtest_params()
    rng = np.random.default_rng(8)
    is_holdout = (predictions[TARGET_DAY] >= BACKTEST_HOLDOUT).to_numpy()
    noisy = predictions.copy()
    for column in (TARGET_PRICE, PRED_SELECTED, PRED_NAIVE):
        noisy.loc[is_holdout, column] = rng.uniform(-100, 500, is_holdout.sum())
    noisy_report = _run(noisy, make_alerts(noisy, params), params)["report"]
    report = outputs["report"]
    assert noisy_report["choices_before_holdout"] == report["choices_before_holdout"]
    assert noisy_report["holdout_descriptive_only"] != report["holdout_descriptive_only"]
    assert set(report["choices_before_holdout"]["rules_for_run_6"]) == {
        "1_leak_alarm",
        "2_model_value",
        "3_morning_report_strategy",
    }
    assert create_pipeline().inputs() == {"predictions_model_a", "alerts", "params:backtest"}
    assert not report["holdout_descriptive_only"]["alarms"]["investigate_bug"]
    leaky = predictions.copy()
    leaky.loc[is_holdout, PRED_SELECTED] = leaky.loc[is_holdout, TARGET_PRICE]
    leaky_report = _run(leaky, make_alerts(leaky, params), params)["report"]
    assert leaky_report["holdout_descriptive_only"]["alarms"]["investigate_bug"]

    days = pd.date_range("2020-01-01", periods=60, freq="D", name=TARGET_DAY)
    draws = pd.Series(rng.normal(60.0, 30.0, len(days)), index=days)
    gains = draws.rolling(3, min_periods=1).mean()
    daily = pd.DataFrame(
        {
            savings_column(OPTIMIZED_SELECTED): 100.0,
            savings_column(OPTIMIZED_NAIVE): 100.0 - gains,
            savings_column(ALERT_RULE_SELECTED): 40.0,
            savings_column(ALERT_RULE_NAIVE): 40.0,
        },
        index=days,
    )
    comparison = compare_strategies(daily, OPTIMIZED_SELECTED, OPTIMIZED_NAIVE, params)
    lag = comparison["diebold_mariano"]["lag_days"]
    ones = np.ones((len(gains), 1))
    hac = sm.OLS(gains.to_numpy(), ones).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}
    )
    half_width = NormalDist().inv_cdf(0.975) * float(hac.bse[0])
    expected = [365 * (gains.mean() - half_width), 365 * (gains.mean() + half_width)]
    assert comparison["gain_per_year_eur"] == pytest.approx(365 * gains.mean(), abs=0.01)
    assert comparison["gain_per_year_interval_eur"] == pytest.approx(expected, abs=0.01)
    assert comparison["first_significantly_better"]

    payback = payback_summary(daily, params)
    no_shift = payback[OPTIMIZED_SELECTED]["vs_no_shift"]
    assert no_shift["payback_months"] == pytest.approx(30000.0 / (36500.0 / 12), abs=0.05)
    assert payback[ALERT_RULE_SELECTED]["vs_naive"]["payback_months"] is None
