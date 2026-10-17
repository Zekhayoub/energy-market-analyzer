"""Tests of the regime nodes: past-only filtered regime, known regimes, forward filter,
daily observations and the development-period analysis.

Synthetic regimes come from `make_regime_observations`; the observation test
goes through the real chain build_master -> table C -> observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_market_analyzer.pipelines.features.nodes import (
    CLEAN_SPARK_SPREAD,
    PRICE_CHANGE,
    build_daily_market,
    build_features_model_c,
    market_days,
)
from energy_market_analyzer.pipelines.ingestion.nodes import PRICE
from energy_market_analyzer.pipelines.regime.nodes import (
    CLEAN_SPARK_SPREAD_MEAN,
    FILTERED,
    PRICE_CHANGE_STD,
    PRICE_MEAN,
    REGIME_FILTERED,
    REGIME_SMOOTHED,
    SMOOTHED,
    analyze_development_period,
    build_regime_observations,
    filter_probabilities,
    fit_filtered_regimes,
    fit_hmm,
    fit_smoothed_regimes,
    hmm_inputs,
    regime_columns,
    to_hmmlearn,
)
from tests.helpers import (
    REGIME_NAMES,
    make_features_params,
    make_master,
    make_regime_observations,
    make_regime_params,
)

TIMEZONE = "Europe/Brussels"
# 18 blocks of 30 days from 1 January 2019; the first 6 months see the three regimes.
STATES = [0, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 0, 2]
BLOCK_DAYS = 30
# The anti-leak test stops early in 2020: 13 blocks, the last calm block ends on 26 December 2019.
LEAK_TEST_BLOCKS = 13
FILTERED_PROBABILITIES = regime_columns(FILTERED, REGIME_NAMES)[1:]
SMOOTHED_PROBABILITIES = regime_columns(SMOOTHED, REGIME_NAMES)[1:]


def _noise_after(observations: pd.DataFrame, last_known_day: str) -> pd.DataFrame:
    """Replace every observation after `last_known_day` by large random values."""
    rng = np.random.default_rng(7)
    is_after = observations.index > pd.Timestamp(last_known_day)
    noisy = observations.copy()
    noisy.loc[is_after] = rng.uniform(0, 3000, (is_after.sum(), len(observations.columns)))
    return noisy


def _labels(table: pd.DataFrame, column: str) -> np.ndarray:
    """Regime labels as floats, NaN where unknown."""
    return table[column].to_numpy(dtype=float, na_value=np.nan)


def test_filtered_regime_only_uses_past_days() -> None:
    """Noise after day J leaves the filtered regime of J and before unchanged.

    J is a first of the month (a refit day) and a mid-month day. The same noise
    moves the smoothed regime before J, so the test would catch a model that
    looks ahead; and changing day J itself moves its filtered regime.
    """
    observations, _ = make_regime_observations(STATES[:LEAK_TEST_BLOCKS], BLOCK_DAYS)
    params = make_regime_params()
    filtered, _ = fit_filtered_regimes(observations, params)
    smoothed, _ = fit_smoothed_regimes(observations, params)
    for last_known_day in ("2019-10-01", "2019-11-17"):
        noisy = _noise_after(observations, last_known_day)
        noisy_filtered, _ = fit_filtered_regimes(noisy, params)
        noisy_smoothed, _ = fit_smoothed_regimes(noisy, params)
        is_known = filtered.index <= pd.Timestamp(last_known_day)

        pd.testing.assert_frame_equal(noisy_filtered[is_known], filtered[is_known])
        smoothed_before = smoothed.loc[is_known, SMOOTHED_PROBABILITIES].to_numpy()
        noisy_smoothed_before = noisy_smoothed.loc[is_known, SMOOTHED_PROBABILITIES].to_numpy()
        assert not np.allclose(noisy_smoothed_before, smoothed_before)

    calm_day = pd.Timestamp("2019-12-10")
    crisis_values = {PRICE_MEAN: 400.0, PRICE_CHANGE_STD: 120.0, CLEAN_SPARK_SPREAD_MEAN: 150.0}
    changed = observations.copy()
    changed.loc[calm_day, list(crisis_values)] = list(crisis_values.values())
    changed_filtered, _ = fit_filtered_regimes(changed, params)
    assert filtered.loc[calm_day, REGIME_FILTERED] == 0
    assert changed_filtered.loc[calm_day, REGIME_FILTERED] == 2


def test_regimes_recover_known_states_in_volatility_order() -> None:
    """On data drawn from 3 known regimes, both outputs find them, 0 = calmest.

    A fit from another seed gives the same labels: no label switching.
    """
    observations, true_states = make_regime_observations(STATES, BLOCK_DAYS)
    params = make_regime_params()
    filtered, _ = fit_filtered_regimes(observations, params)
    smoothed, _ = fit_smoothed_regimes(observations, params)
    other_seed, _ = fit_smoothed_regimes(observations, make_regime_params(random_state=50))
    filtered_labels = _labels(filtered, REGIME_FILTERED)
    smoothed_labels = _labels(smoothed, REGIME_SMOOTHED)
    is_filtered = ~np.isnan(filtered_labels)

    assert is_filtered.sum() > len(true_states) / 2
    assert (filtered_labels[is_filtered] == true_states[is_filtered]).mean() > 0.9
    assert (smoothed_labels == true_states).mean() > 0.97
    assert (_labels(other_seed, REGIME_SMOOTHED) == smoothed_labels).mean() > 0.99
    assert filtered[FILTERED_PROBABILITIES].dropna().sum(axis=1).round(9).eq(1).all()


def test_forward_filter_matches_hmmlearn_and_skips_missing_days() -> None:
    """The filtered probability of day t equals hmmlearn's posterior of the days 0..t at t.

    An incomplete day gets no regime and does not break the following days.
    """
    observations, _ = make_regime_observations(STATES[:8], BLOCK_DAYS)
    params = make_regime_params()
    inputs = hmm_inputs(observations)
    fitted = fit_hmm(inputs, len(REGIME_NAMES), params, seed=0)
    model = to_hmmlearn(fitted, params["covariance_type"])
    filtered = filter_probabilities(fitted, inputs)
    for day in (0, 10, 100, len(inputs) - 1):
        posterior_at_day = model.predict_proba(inputs[: day + 1])[-1]
        np.testing.assert_allclose(filtered[day], posterior_at_day, atol=1e-8)

    missing_day = 100
    with_gap = inputs.copy()
    with_gap[missing_day] = np.nan
    gap_filtered = filter_probabilities(fitted, with_gap)
    assert np.isnan(gap_filtered[missing_day]).all()
    np.testing.assert_allclose(gap_filtered[:missing_day], filtered[:missing_day])
    np.testing.assert_allclose(gap_filtered[missing_day + 1 :].sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("start_date", "end_date", "clock_change_day", "hours"),
    [
        ("2025-03-27", "2025-04-01", "2025-03-30", 23),
        ("2025-10-23", "2025-10-28", "2025-10-26", 25),
    ],
)
def test_daily_observations_follow_brussels_days(
    start_date: str, end_date: str, clock_change_day: str, hours: int
) -> None:
    """Observations of a 23- or 25-hour day, complete-day rule, nothing read after the day."""
    master = make_master(start_date, end_date)
    features_params = make_features_params()
    daily_market = build_daily_market(master, features_params)
    table_c = build_features_model_c(master, daily_market, features_params)
    regime_params = make_regime_params()
    observations = build_regime_observations(table_c, regime_params)
    day = pd.Timestamp(clock_change_day)
    is_day = market_days(table_c.index, TIMEZONE) == day
    hours_of_day = table_c[is_day]

    assert len(hours_of_day) == hours
    expected = {
        PRICE_MEAN: hours_of_day[PRICE].mean(),
        PRICE_CHANGE_STD: hours_of_day[PRICE_CHANGE].std(),
        CLEAN_SPARK_SPREAD_MEAN: hours_of_day[CLEAN_SPARK_SPREAD].mean(),
    }
    pd.testing.assert_series_equal(
        observations.loc[day], pd.Series(expected, name=day), check_names=False
    )

    noisy_table_c = table_c.copy()
    is_after_day = market_days(table_c.index, TIMEZONE) > day
    noisy_table_c.loc[is_after_day, [PRICE, PRICE_CHANGE, CLEAN_SPARK_SPREAD]] = 999.0
    noisy_observations = build_regime_observations(noisy_table_c, regime_params)
    pd.testing.assert_frame_equal(noisy_observations.loc[:day], observations.loc[:day])

    # The last price of the day is missing: the day is incomplete, and so is the
    # volatility of the next day, whose first change starts from that price.
    gap_master = master.copy()
    last_hour = hours_of_day.index[-1]
    gap_master.loc[last_hour, PRICE] = np.nan
    gap_daily_market = build_daily_market(gap_master, features_params)
    gap_table_c = build_features_model_c(gap_master, gap_daily_market, features_params)
    gap_observations = build_regime_observations(gap_table_c, regime_params)
    next_day = day + pd.Timedelta(days=1)
    assert gap_observations.loc[day].isna().all()
    assert np.isnan(gap_observations.loc[next_day, PRICE_CHANGE_STD])
    assert not np.isnan(gap_observations.loc[next_day, PRICE_MEAN])


def test_development_analysis_ignores_the_holdout_and_finds_three_states() -> None:
    """Noise in the reserved period changes nothing in the development analysis.

    On data drawn from 3 regimes, the BIC is lowest at 3 states.
    """
    observations, _ = make_regime_observations(STATES, BLOCK_DAYS)
    params = make_regime_params(holdout_start="2020-01-01")
    filtered, _ = fit_filtered_regimes(observations, params)
    analysis = analyze_development_period(observations, filtered, params)
    noisy = _noise_after(observations, "2019-12-31")
    noisy_filtered, _ = fit_filtered_regimes(noisy, params)
    noisy_analysis = analyze_development_period(noisy, noisy_filtered, params)

    assert noisy_analysis == analysis
    assert analysis["last_day"] == "2019-12-31"
    assert analysis["bic_best_n_states"] == 3
    assert min(analysis["stability_across_seeds"].values()) > 0.99
    assert analysis["filtered_vs_hindsight"]["by_year"]["2019"] > 0.9
