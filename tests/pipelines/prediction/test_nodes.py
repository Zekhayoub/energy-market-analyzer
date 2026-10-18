"""Tests of the prediction nodes: the 11:00 rule through the monthly refits, the reserved
period, known answers of the baselines and of the selection rule, the conformal
calibration and the alerts.

Feature tables come from the real feature nodes (`make_prediction_market`); the
filtered regimes are known blocks of days.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from energy_market_analyzer.pipelines.features.nodes import (
    DAY_OF_WEEK,
    DECISION_DAY,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    HORIZON_DAYS,
    PRICE_SAME_HOUR_TODAY,
    PRICE_SAME_HOUR_WEEK_AGO,
    TARGET_MEAN_PRICE,
    TARGET_PRICE,
    market_days,
)
from energy_market_analyzer.pipelines.ingestion.nodes import ONE_DAY
from energy_market_analyzer.pipelines.prediction.evaluation import (
    calibrate_intervals,
    diebold_mariano,
    error_metrics,
    is_development,
    select_variant,
)
from energy_market_analyzer.pipelines.prediction.models import (
    ANCHOR,
    GLOBAL,
    HYBRID,
    INTERVAL_CALIBRATED,
    INTERVAL_LOWER,
    INTERVAL_LOWER_RAW,
    INTERVAL_UPPER,
    INTERVAL_UPPER_RAW,
    IS_CHEAPEST_HOUR,
    IS_MOST_EXPENSIVE_HOUR,
    MEAN_TODAY,
    NAIVE_SIMILAR_DAY,
    NO_REGIME,
    RANK_IN_DAY,
    REALIZED_LEVEL,
    SECOND_SEED,
    TARGET_DAY,
    TASK_A,
    TASK_B,
    THRESHOLD_CRITICAL,
    THRESHOLD_HIGH,
    THRESHOLD_REGIME,
    baseline_predictions,
    prediction_column,
)
from energy_market_analyzer.pipelines.prediction.nodes import (
    alert_confidence,
    alert_levels,
    build_alerts,
    build_model_a_inputs,
    build_model_b_inputs,
    build_prediction_report,
    daily_ranks,
    exceedance_probability,
    finalize_predictions,
    month_thresholds,
    walk_forward_model_a,
    walk_forward_model_b,
)
from energy_market_analyzer.pipelines.prediction.pipeline import create_pipeline
from energy_market_analyzer.pipelines.regime.nodes import FILTERED, SMOOTHED, regime_columns
from tests.helpers import (
    MARKET_TIMEZONE,
    PREDICTION_HOLDOUT,
    REGIME_NAMES,
    make_prediction_market,
    make_prediction_params,
)

ONE_HOUR = pd.Timedelta(hours=1)


def _run(
    features_a: pd.DataFrame, features_b: pd.DataFrame, regimes: pd.DataFrame, params: dict
) -> dict[str, Any]:
    """Every node of the pipeline, in order."""
    inputs_a = build_model_a_inputs(features_a, regimes, params)
    inputs_b = build_model_b_inputs(features_b, regimes, params)
    raw_a, refits_a = walk_forward_model_a(inputs_a, params)
    raw_b, refits_b = walk_forward_model_b(inputs_b, params)
    predictions_a, predictions_b, choices = finalize_predictions(raw_a, raw_b, params)
    alerts, thresholds = build_alerts(predictions_a, inputs_a, regimes, params)
    report = build_prediction_report(
        predictions_a, predictions_b, alerts, choices, refits_a, refits_b, thresholds, params
    )
    return {"a": predictions_a, "b": predictions_b, "alerts": alerts, "report": report}


@pytest.fixture(scope="module")
def market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Feature tables A and B and the regimes, January to October 2018."""
    return make_prediction_market()


@pytest.fixture(scope="module")
def outputs(market: tuple) -> dict[str, Any]:
    """Outputs of the whole pipeline on the synthetic market."""
    return _run(*market, make_prediction_params())


def _float_columns(table: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    """Columns among `columns` stored as floats (calendar columns are integers)."""
    return [name for name in columns if table[name].dtype.kind == "f"]


def _target_days_b(features_b: pd.DataFrame) -> pd.DatetimeIndex:
    """Predicted day of each row of table B."""
    decision_days = features_b.index.get_level_values(DECISION_DAY)
    horizons = pd.to_timedelta(features_b.index.get_level_values(HORIZON_DAYS), unit="D")
    return decision_days + horizons


def _noise_from(
    market: tuple, first_unknown_day: pd.Timestamp, seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random values in everything unknown at 11:00 on the day before `first_unknown_day`.

    Unknown at 11:00 J (J = first_unknown_day - 1): the prices of J+1 and after
    (targets), the features decided after J, the regimes of J+1 and after.
    """
    features_a, features_b, regimes = market
    rng = np.random.default_rng(seed)
    days_a = market_days(features_a.index, MARKET_TIMEZONE)
    hidden_a = features_a.copy()
    is_unknown_price = days_a >= first_unknown_day
    hidden_a.loc[is_unknown_price, TARGET_PRICE] = rng.uniform(-500, 3000, is_unknown_price.sum())
    is_later_decision = days_a >= first_unknown_day + ONE_DAY
    columns_a = _float_columns(features_a, FEATURES_MODEL_A)
    hidden_a.loc[is_later_decision, columns_a] = rng.uniform(
        0, 1000, (is_later_decision.sum(), len(columns_a))
    )
    hidden_b = features_b.copy()
    is_unknown_target = _target_days_b(features_b) >= first_unknown_day
    hidden_b.loc[is_unknown_target, TARGET_MEAN_PRICE] = rng.uniform(
        -100, 1000, is_unknown_target.sum()
    )
    is_later_b = features_b.index.get_level_values(DECISION_DAY) >= first_unknown_day
    columns_b = _float_columns(features_b, FEATURES_MODEL_B)
    hidden_b.loc[is_later_b, columns_b] = rng.uniform(0, 1000, (is_later_b.sum(), len(columns_b)))
    return hidden_a, hidden_b, _random_regimes_from(regimes, first_unknown_day, rng)


def _random_regimes_from(
    regimes: pd.DataFrame, first_day: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    """Random filtered regimes from `first_day` on, random smoothed regimes on every day.

    The smoothed regime of a day depends on the days after it: any change of the
    future may change it, so it must never be read.
    """
    noisy = regimes.copy()
    is_after = regimes.index >= first_day
    for kind, rows in ((FILTERED, is_after), (SMOOTHED, np.ones(len(regimes), dtype=bool))):
        label_column, *probability_columns = regime_columns(kind, REGIME_NAMES)
        probabilities = rng.dirichlet(np.ones(len(probability_columns)), rows.sum())
        noisy.loc[rows, probability_columns] = probabilities
        noisy.loc[rows, label_column] = pd.array(probabilities.argmax(axis=1), dtype="Int8")
    return noisy


def test_outputs_of_a_day_only_use_what_is_known_at_eleven(market: tuple, outputs: dict) -> None:
    """Noise in everything unknown at 11:00 J leaves the predictions, intervals and alerts of
    J+1 (model A) and of the decision day J (model B) unchanged.

    J is the last day before a month (refit at the month boundary) and a mid-month day.
    The same noise moves later rows, so the test bites. The outputs react to what is
    known: the probabilities of the regime of J move the hybrid variant, a past price
    moves the global model. Equal outputs also show the run is reproducible.
    """
    params = make_prediction_params()
    for decision_day in (pd.Timestamp("2018-08-31"), pd.Timestamp("2018-07-15")):
        next_day = decision_day + ONE_DAY
        hidden = _run(*_noise_from(market, next_day), params)
        rows_a = outputs["a"][TARGET_DAY] == next_day
        later_a = outputs["a"][TARGET_DAY] > next_day
        rows_b = outputs["b"].index.get_level_values(DECISION_DAY) == decision_day
        known_a = outputs["a"].columns.drop(TARGET_PRICE)
        known_alerts = outputs["alerts"].columns.drop(REALIZED_LEVEL)
        known_b = outputs["b"].columns.drop(TARGET_MEAN_PRICE)

        pd.testing.assert_frame_equal(
            hidden["a"].loc[rows_a, known_a], outputs["a"].loc[rows_a, known_a]
        )
        pd.testing.assert_frame_equal(
            hidden["alerts"].loc[rows_a, known_alerts], outputs["alerts"].loc[rows_a, known_alerts]
        )
        pd.testing.assert_frame_equal(
            hidden["b"].loc[rows_b, known_b], outputs["b"].loc[rows_b, known_b]
        )
        global_column = prediction_column(GLOBAL)
        assert (
            not hidden["a"]
            .loc[later_a, global_column]
            .equals(outputs["a"].loc[later_a, global_column])
        )

    features_a, features_b, regimes = market
    decision_day = pd.Timestamp("2018-07-15")
    rows_a = outputs["a"][TARGET_DAY] == decision_day + ONE_DAY
    probability_columns = regime_columns(FILTERED, params["regime_names"])[1:]
    shifted = regimes.copy()
    shifted.loc[decision_day, probability_columns] = [0.05, 0.05, 0.9]
    shifted_outputs = _run(features_a, features_b, shifted, params)
    hybrid_column = prediction_column(HYBRID)
    assert not np.allclose(
        shifted_outputs["a"].loc[rows_a, hybrid_column], outputs["a"].loc[rows_a, hybrid_column]
    )
    past_hour = features_a.index[market_days(features_a.index, MARKET_TIMEZONE) == "2018-05-10"]
    past_prices = features_a.assign(
        **{TARGET_PRICE: features_a[TARGET_PRICE].where(~features_a.index.isin(past_hour), 2000.0)}
    )
    past_outputs = _run(past_prices, features_b, regimes, params)
    global_column = prediction_column(GLOBAL)
    assert not np.allclose(
        past_outputs["a"].loc[rows_a, global_column], outputs["a"].loc[rows_a, global_column]
    )


def test_model_b_learns_only_known_targets_and_is_split_by_predicted_day(
    market: tuple, outputs: dict
) -> None:
    """Rows decided before a month but predicting a day of it never train the month's models.

    Their targets are unknown on the first decision days of the month. Noise in
    them leaves the month's predictions unchanged; noise in known targets moves
    them. A row decided before holdout_start for a day after it belongs to the
    reserved period.
    """
    features_a, features_b, regimes = market
    params = make_prediction_params()
    month_start = pd.Timestamp("2018-08-01")
    decision_days = features_b.index.get_level_values(DECISION_DAY)
    target_days = _target_days_b(features_b)
    rng = np.random.default_rng(3)
    inputs_b = build_model_b_inputs(features_b, regimes, params)
    raw_b, _ = walk_forward_model_b(inputs_b, params)
    in_month = (raw_b[DECISION_DAY] >= month_start) & (raw_b[DECISION_DAY] < "2018-09-01")
    point_columns = [column for column in raw_b.columns if column.startswith("pred_")]
    for is_changed, should_move in (
        ((decision_days < month_start) & (target_days >= month_start), False),
        ((target_days < month_start) & (target_days >= "2018-07-01"), True),
    ):
        noisy_b = features_b.copy()
        noisy_b.loc[is_changed, TARGET_MEAN_PRICE] = rng.uniform(500, 1000, is_changed.sum())
        noisy_raw, _ = walk_forward_model_b(build_model_b_inputs(noisy_b, regimes, params), params)
        is_equal = noisy_raw.loc[in_month, point_columns].equals(raw_b.loc[in_month, point_columns])
        assert is_equal != should_move

    table_b = outputs["b"].reset_index()
    straddling = table_b[table_b[DECISION_DAY] == pd.Timestamp(PREDICTION_HOLDOUT) - 3 * ONE_DAY]
    expected = (straddling[TARGET_DAY] < PREDICTION_HOLDOUT).to_numpy()
    assert expected.any()
    assert not expected.all()
    np.testing.assert_array_equal(is_development(straddling, params), expected)


def test_choices_ignore_the_reserved_period(market: tuple, outputs: dict) -> None:
    """Noise in every predicted day of the reserved period changes nothing in the choices.

    The selected variants, tree curves, calibration candidates and every development
    figure stay the same; the reserved section moves. The pipeline never reads an
    output of the anomaly pipeline.
    """
    params = make_prediction_params()
    noisy = _run(*_noise_from(market, pd.Timestamp(PREDICTION_HOLDOUT), seed=11), params)
    choices = outputs["report"]["choices_before_holdout"]
    noisy_choices = noisy["report"]["choices_before_holdout"]

    assert noisy_choices == choices
    assert (
        noisy["report"]["holdout_descriptive_only"] != outputs["report"]["holdout_descriptive_only"]
    )
    assert create_pipeline().inputs() == {
        "features_model_a",
        "features_model_b",
        "regimes",
        "params:prediction",
    }


def _selection_table(differences: np.ndarray) -> pd.DataFrame:
    """Model B rows (one per day) whose no_regime error exceeds global's by `differences`."""
    days = pd.date_range("2019-01-01", periods=len(differences), freq="D")
    return pd.DataFrame(
        {
            DECISION_DAY: days,
            TARGET_DAY: days + 2 * ONE_DAY,
            TARGET_MEAN_PRICE: 0.0,
            prediction_column(MEAN_TODAY): 9.0,
            prediction_column(GLOBAL): 5.0,
            prediction_column(NO_REGIME): 5.0 + differences,
            ANCHOR: 1.0,
        }
    )


def test_baselines_tests_and_selection_give_known_answers() -> None:
    """Similar-day naive, Diebold-Mariano against statsmodels, rMAE and the selection rule.

    The rule keeps the simplest candidate (no_regime) when global is better by
    noise only, and global when the gap is significant; not when two seeds of global
    already differ by more than that gap (training noise).
    """
    days = pd.date_range("2019-01-07", periods=7, freq="D")
    week = pd.DataFrame(
        {DAY_OF_WEEK: days.dayofweek, PRICE_SAME_HOUR_TODAY: 1.0, PRICE_SAME_HOUR_WEEK_AGO: 2.0}
    ).assign(**{ANCHOR: 1.5})
    similar_day = baseline_predictions(week, TASK_A)[prediction_column(NAIVE_SIMILAR_DAY)]
    np.testing.assert_array_equal(similar_day, [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])

    params = make_prediction_params()
    rng = np.random.default_rng(5)
    differential = pd.Series(rng.normal(0.3, 1.0, 200)).rolling(3, min_periods=1).mean()
    test = diebold_mariano(differential, params)
    ones = np.ones((len(differential), 1))
    hac = sm.OLS(differential.to_numpy(), ones).fit(
        cov_type="HAC", cov_kwds={"maxlags": test["lag_days"], "use_correction": False}
    )
    assert test["statistic"] == pytest.approx(float(hac.tvalues[0]), abs=1e-3)
    assert test["p_value"] == pytest.approx(float(hac.pvalues[0]), abs=1e-4)

    draws = rng.normal(0.0, 1.0, 120)
    centred = draws - draws.mean()
    noise_only = _selection_table(centred + 0.05)
    significant = _selection_table(centred + 1.0)
    noise_selection = select_variant(noise_only, TASK_B, params)
    assert noise_selection["best"] == GLOBAL
    assert noise_selection["selected"] == NO_REGIME
    assert select_variant(significant, TASK_B, params)["selected"] == GLOBAL
    noisy_training = significant.assign(**{prediction_column(SECOND_SEED): 6.5})
    noisy_selection = select_variant(noisy_training, TASK_B, params)
    assert noisy_selection["training_noise_mae"] == pytest.approx(1.5)
    assert noisy_selection["selected"] == NO_REGIME
    metrics = error_metrics(significant, TASK_B, GLOBAL)
    assert metrics["mae"] == pytest.approx(5.0)
    assert metrics["rmae"] == pytest.approx(5.0 / 9.0, abs=1e-3)
    assert metrics["bias"] == pytest.approx(5.0)


def _interval_table(days: int, raw_half_width: float, scale: float) -> pd.DataFrame:
    """Model A rows, 24 per day: actual price ~ N(0, scale), raw interval +/- raw_half_width."""
    rng = np.random.default_rng(9)
    target_days = np.repeat(pd.date_range("2019-01-02", periods=days, freq="D"), 24)
    return pd.DataFrame(
        {
            TARGET_DAY: target_days,
            DECISION_DAY: target_days - ONE_DAY,
            TARGET_PRICE: rng.normal(0.0, scale, len(target_days)),
            INTERVAL_LOWER_RAW: -raw_half_width,
            INTERVAL_UPPER_RAW: raw_half_width,
        }
    )


def test_conformal_calibration_uses_known_errors_and_reaches_the_coverage() -> None:
    """Too narrow raw intervals are widened to about 80 % with errors known at 11:00 J.

    The interval of day D only uses errors of days up to D - 1: noise in D and
    after changes nothing, a change on D - 1 moves it. Too wide intervals shrink,
    never below min_width_share of their raw width.
    """
    params = make_prediction_params()
    window = 30
    table = _interval_table(days=200, raw_half_width=5.0, scale=10.0)
    intervals = calibrate_intervals(table, TASK_A, window, params)
    calibrated = table.assign(**intervals)[intervals[INTERVAL_CALIBRATED].to_numpy()]
    inside = calibrated[TARGET_PRICE].between(
        calibrated[INTERVAL_LOWER], calibrated[INTERVAL_UPPER]
    )
    assert inside.mean() == pytest.approx(0.8, abs=0.03)

    day = pd.Timestamp("2019-05-01")
    is_day = (table[TARGET_DAY] == day).to_numpy()
    noisy = table.copy()
    noisy.loc[(table[TARGET_DAY] >= day).to_numpy(), TARGET_PRICE] = 1e6
    noisy_intervals = calibrate_intervals(noisy, TASK_A, window, params)
    pd.testing.assert_frame_equal(noisy_intervals[is_day], intervals[is_day])
    moved = table.copy()
    moved.loc[(table[TARGET_DAY] == day - ONE_DAY).to_numpy(), TARGET_PRICE] = 1e6
    moved_intervals = calibrate_intervals(moved, TASK_A, window, params)
    assert not moved_intervals[is_day].equals(intervals[is_day])

    wide = _interval_table(days=60, raw_half_width=50.0, scale=1.0)
    shrunk = calibrate_intervals(wide, TASK_A, window, params)
    widths = (shrunk[INTERVAL_UPPER] - shrunk[INTERVAL_LOWER])[shrunk[INTERVAL_CALIBRATED]]
    assert widths.min() == pytest.approx(params["interval"]["min_width_share"] * 100.0)
    assert shrunk["guarded"].any()


def test_alert_thresholds_probabilities_confidence_and_ranks() -> None:
    """Thresholds from past prices of the regime (closest regime, or all hours, when too few);
    probability read on the interval; confidence of each level; ranks on 23- and 25-hour days."""
    params = make_prediction_params()
    rng = np.random.default_rng(2)
    days = pd.date_range("2018-01-01", "2018-04-30", freq="D")
    hourly_days = np.repeat(days, 24)
    regimes = np.where(np.repeat(days.day, 24) <= 15, 0.0, 1.0)
    regimes[np.repeat(days.day, 24) == 31] = 2.0
    reference = pd.DataFrame(
        {
            TARGET_DAY: hourly_days,
            "price": np.where(regimes == 0, 40.0, 90.0) + rng.normal(0, 5, len(hourly_days)),
            "regime": regimes,
        }
    )
    month = pd.Timestamp("2018-04-01")
    records = pd.DataFrame(month_thresholds(reference, month, params)).set_index("regime")
    in_window = (reference[TARGET_DAY] >= "2018-02-01") & (reference[TARGET_DAY] < month)
    calm_prices = reference.loc[in_window & (reference["regime"] == 0), "price"]
    assert records.loc[0, THRESHOLD_HIGH] == pytest.approx(calm_prices.quantile(0.8))
    assert records.loc[0, THRESHOLD_CRITICAL] == pytest.approx(calm_prices.quantile(0.95))
    assert records.loc[2, THRESHOLD_REGIME] == "volatile"
    assert records.loc[-1, THRESHOLD_REGIME] == "all"
    noisy = reference.assign(price=reference["price"].where(~(reference[TARGET_DAY] >= month), 1e4))
    pd.testing.assert_frame_equal(
        pd.DataFrame(month_thresholds(noisy, month, params)).set_index("regime"), records
    )

    thresholds = np.array([5.0, 20.0, 60.0, 15.0, 20.0])
    lower = np.array([10.0, 10.0, 10.0, 10.0, 30.0])
    point = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
    upper = np.array([50.0, 50.0, 50.0, 50.0, 40.0])
    probabilities = exceedance_probability(thresholds, lower, point, upper, params)
    np.testing.assert_allclose(probabilities, [0.9, 0.5, 0.1, 0.7, 0.9])

    prices = np.array([150.0, 100.0, 50.0])
    levels = alert_levels(prices, np.full(3, 90.0), np.full(3, 140.0))
    assert levels.tolist() == ["critical", "high", "normal"]
    confidence = alert_confidence(levels, np.array([0.9, 0.75, 0.6]), np.full(3, 0.9), params)
    assert confidence.tolist() == ["high", "medium", "low"]

    local_hours = pd.date_range(
        pd.Timestamp("2018-03-25", tz=MARKET_TIMEZONE),
        pd.Timestamp("2018-03-26", tz=MARKET_TIMEZONE),
        freq="h",
        inclusive="left",
    ).append(
        pd.date_range(
            pd.Timestamp("2018-10-28", tz=MARKET_TIMEZONE),
            pd.Timestamp("2018-10-30", tz=MARKET_TIMEZONE),
            freq="h",
            inclusive="left",
        )
    )
    hours = local_hours.tz_convert("UTC")
    hour_days = pd.Series(market_days(hours, MARKET_TIMEZONE), index=hours)
    predicted = pd.Series(np.arange(len(hours), 0, -1, dtype=float), index=hours)
    predicted.iloc[-3] = np.nan
    ranks = daily_ranks(predicted, hour_days, params)
    spring = ranks[hour_days == "2018-03-25"]
    autumn = ranks[hour_days == "2018-10-28"]
    assert (len(spring), len(autumn)) == (23, 25)
    assert spring[RANK_IN_DAY].tolist() == list(range(23, 0, -1))
    assert int(autumn[IS_CHEAPEST_HOUR].sum()) == 8
    assert autumn[IS_MOST_EXPENSIVE_HOUR].iloc[:4].all()
    assert ranks.loc[hour_days == "2018-10-29", RANK_IN_DAY].isna().all()
