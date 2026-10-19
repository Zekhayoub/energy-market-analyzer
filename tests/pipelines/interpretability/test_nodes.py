"""Tests of the interpretability nodes: exact explanations of the published predictions, the
11:00 rule through the refits and in the morning report, the SHAP report and the rules of
run 7, and known cases of the morning report.

Predictions come from the real prediction nodes on the synthetic market of the prediction
tests (`make_prediction_market`); the savings from the real backtest nodes.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from energy_market_analyzer.pipelines.backtest.nodes import (
    build_backtest_inputs,
    client_profile,
    optimal_schedule,
    run_backtest,
)
from energy_market_analyzer.pipelines.features.nodes import (
    DECISION_DAY,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    HORIZON_DAYS,
    HOUR,
    PRICE_SAME_HOUR_TODAY,
    TARGET_MEAN_PRICE,
    TARGET_PRICE,
    TODAY_PRICES,
    TTF_LAST,
)
from energy_market_analyzer.pipelines.ingestion.nodes import ONE_DAY, PRICE
from energy_market_analyzer.pipelines.interpretability.explanations import (
    BASE_VALUE,
    EXACTNESS_TOLERANCE_EUR_MWH,
    PRED_SELECTED,
    explain_month,
    explain_task,
    families_of,
    family_sums,
    feature_contributions,
    indexed_inputs,
    interventional_contributions,
    month_rows,
    shap_column,
)
from energy_market_analyzer.pipelines.interpretability.morning_report import (
    PRED_NAIVE,
    build_morning_report,
    hour_labels,
    labels_of,
)
from energy_market_analyzer.pipelines.interpretability.nodes import (
    FAMILY_LEVEL,
    FEATURE_LEVEL,
    agreement,
    build_shap_report,
    driver_stability,
    explain_model_a,
    explain_model_b,
    explanation_section,
    hour_alignment_rule,
    method_sensitivity,
)
from energy_market_analyzer.pipelines.prediction.models import (
    ANCHOR,
    AVERAGE,
    GLOBAL,
    HYBRID,
    NO_REGIME,
    PER_REGIME,
    REALIZED_LEVEL,
    TARGET_DAY,
    TASK_A,
    TASK_B,
    WEIGHTED,
    prediction_column,
    regime_feature_columns,
)
from energy_market_analyzer.pipelines.prediction.nodes import (
    build_alerts,
    build_model_a_inputs,
    build_model_b_inputs,
    build_prediction_report,
    finalize_predictions,
    walk_forward_model_a,
    walk_forward_model_b,
)
from energy_market_analyzer.pipelines.regime.nodes import (
    FILTERED,
    REGIME_FILTERED,
    SMOOTHED,
    regime_columns,
)
from tests.helpers import (
    INTERPRETABILITY_DECISION_DAY,
    MARKET_TIMEZONE,
    PREDICTION_HOLDOUT,
    REGIME_NAMES,
    make_anomalies,
    make_anomaly_params,
    make_backtest_params,
    make_interpretability_params,
    make_prediction_market,
    make_prediction_params,
)

DECISION = pd.Timestamp(INTERPRETABILITY_DECISION_DAY)
ONE_HOUR = pd.Timedelta(hours=1)
CHECKED_MONTH = pd.Timestamp("2018-07-01")
ROUNDING_EUR_MWH = 0.05  # the report rounds each part to the cent
REGIME_CONTRIBUTIONS = (50.0, 40.0, 30.0, 20.0)  # regime of J, then its probabilities
TINY_ROWS = 600


@pytest.fixture(scope="module")
def chain() -> dict[str, Any]:
    """Predictions, alerts, report and backtest of the synthetic market, and their SHAP tables."""
    features_a, features_b, regimes = make_prediction_market()
    prediction_params = make_prediction_params()
    params = make_interpretability_params()
    inputs_a = build_model_a_inputs(features_a, regimes, prediction_params)
    inputs_b = build_model_b_inputs(features_b, regimes, prediction_params)
    raw_a, refits_a = walk_forward_model_a(inputs_a, prediction_params)
    raw_b, refits_b = walk_forward_model_b(inputs_b, prediction_params)
    predictions_a, predictions_b, choices = finalize_predictions(raw_a, raw_b, prediction_params)
    alerts, thresholds = build_alerts(predictions_a, inputs_a, regimes, prediction_params)
    prediction_report = build_prediction_report(
        predictions_a,
        predictions_b,
        alerts,
        choices,
        refits_a,
        refits_b,
        thresholds,
        prediction_params,
    )
    shap_a, checks_a = explain_model_a(
        inputs_a, predictions_a, prediction_report, prediction_params, params
    )
    shap_b, checks_b = explain_model_b(
        inputs_b, predictions_b, prediction_report, prediction_params, params
    )
    backtest_params = make_backtest_params(holdout_start=PREDICTION_HOLDOUT)
    backtest_inputs, _ = build_backtest_inputs(predictions_a, alerts, backtest_params)
    backtest_daily, _ = run_backtest(backtest_inputs, backtest_params)
    return {
        "features_a": features_a,
        "features_b": features_b,
        "regimes": regimes,
        "prediction_params": prediction_params,
        "params": params,
        "inputs_a": inputs_a,
        "inputs_b": inputs_b,
        "predictions_a": predictions_a,
        "predictions_b": predictions_b,
        "alerts": alerts,
        "prediction_report": prediction_report,
        "shap_a": shap_a,
        "checks_a": checks_a,
        "shap_b": shap_b,
        "checks_b": checks_b,
        "backtest_params": backtest_params,
        "backtest_daily": backtest_daily,
    }


def _rebuilt(table: pd.DataFrame) -> pd.Series:
    """Anchor + base value + every contribution of a SHAP table."""
    return table[ANCHOR] + table[BASE_VALUE] + feature_contributions(table).sum(axis=1)


def _with_variant(report: dict[str, Any], task_name: str, variant: str) -> dict[str, Any]:
    """Copy of the prediction report that selects `variant` for one model."""
    changed = copy.deepcopy(report)
    changed["choices_before_holdout"][task_name]["selection"]["selected"] = variant
    return changed


def _method_check(params: dict[str, Any]) -> dict[str, Any]:
    """Settings of the interventional check, as the nodes pass them."""
    return {
        "day_of_month": params["method_check_day_of_month"],
        "background_rows": params["method_check_background_rows"],
    }


# ==================================================
# 1. Explanations of the published predictions
# ==================================================


def test_explanations_add_up_to_the_published_predictions(chain: dict[str, Any]) -> None:
    """Every published row is explained, its decomposition adds up to pred_selected, the refit
    gives back the published prediction; a stale or inconsistent table stops the pipeline."""
    prediction_params = chain["prediction_params"]
    holdout = pd.Timestamp(PREDICTION_HOLDOUT)
    for name in ("a", "b"):
        table, checks = chain[f"shap_{name}"], chain[f"checks_{name}"]
        predictions = chain[f"predictions_{name}"]
        gaps = (_rebuilt(table) - table[PRED_SELECTED]).abs()
        assert table.index.sort_values().equals(predictions.index.sort_values())
        assert gaps.max() <= EXACTNESS_TOLERANCE_EUR_MWH
        assert checks["max_refit_difference_eur_mwh"] <= EXACTNESS_TOLERANCE_EUR_MWH
        assert checks["variant"] == NO_REGIME
        development_rows = int((table[TARGET_DAY] < holdout).sum())
        assert len(checks["second_seed"]) == development_rows
        checked_days = table.loc[checks["interventional"].index, TARGET_DAY]
        assert len(checked_days) > 0
        assert (checked_days < holdout).all()

    # Every other candidate is explained as exactly: balanced weights, and the mixtures of
    # XGBoosts (experts picked or mixed by the regime of J, mean of the candidates).
    predictions_a = chain["predictions_a"]
    check = _method_check(chain["params"])
    for variant in (WEIGHTED, PER_REGIME, HYBRID, AVERAGE):
        selected = predictions_a[prediction_column(variant)]
        published = predictions_a.assign(**{PRED_SELECTED: selected})
        report = _with_variant(chain["prediction_report"], TASK_A.name, variant)
        table, checks = explain_task(
            chain["inputs_a"], published, report, TASK_A, prediction_params, check
        )
        assert (_rebuilt(table) - table[PRED_SELECTED]).abs().max() <= EXACTNESS_TOLERANCE_EUR_MWH
        assert checks["interventional_reconstruction_error_eur_mwh"] <= EXACTNESS_TOLERANCE_EUR_MWH
        assert shap_column(REGIME_FILTERED) in table

    # Filtered probabilities of the HMM go far below the float32 range (1e-60, run 7): the
    # interventional check reads them as XGBoost does, so that its leaves are the prediction's.
    rng = np.random.default_rng(0)
    is_on = rng.integers(0, 2, TINY_ROWS).astype(bool)
    tiny = pd.DataFrame(
        {
            "x": rng.normal(size=TINY_ROWS),
            "probability": np.where(is_on, 1.0, 10.0 ** -rng.uniform(40, 80, TINY_ROWS)),
        }
    )
    model = XGBRegressor(
        n_estimators=100, max_depth=3, objective="reg:absoluteerror", random_state=0
    ).fit(tiny, 3 * tiny["x"] + 5 * is_on)
    explained = tiny.iloc[-100:]
    values = interventional_contributions(
        model, tiny.iloc[:-100], explained, list(tiny.columns), prediction_params, check
    )
    tiny_gaps = (values.sum(axis=1) - model.predict(explained)).abs()
    assert tiny_gaps.max() <= EXACTNESS_TOLERANCE_EUR_MWH

    # Stale or inconsistent predictions stop the pipeline.
    is_checked_month = (predictions_a[TARGET_DAY] >= CHECKED_MONTH).to_numpy()
    shifted = predictions_a.copy()
    for column in (PRED_SELECTED, prediction_column(NO_REGIME)):
        shifted.loc[is_checked_month, column] += 0.5
    report = chain["prediction_report"]
    inputs_a = chain["inputs_a"]
    with pytest.raises(ValueError, match="stale"):
        explain_task(inputs_a, shifted, report, TASK_A, prediction_params, check)
    last_month = predictions_a[TARGET_DAY] >= pd.Timestamp("2018-10-01")
    with pytest.raises(ValueError, match="do not cover the rows"):
        explain_task(inputs_a, predictions_a[~last_month], report, TASK_A, prediction_params, check)
    mixed = predictions_a.copy()
    mixed.iloc[0, mixed.columns.get_loc(PRED_SELECTED)] += 1.0
    with pytest.raises(ValueError, match="pred_selected is not pred_no_regime"):
        explain_task(inputs_a, mixed, report, TASK_A, prediction_params, check)
    unknown = _with_variant(report, TASK_A.name, "ridge")
    with pytest.raises(ValueError, match="unknown selected variant 'ridge'"):
        explain_task(inputs_a, predictions_a, unknown, TASK_A, prediction_params, check)


# ==================================================
# 2. The 11:00 rule through the refits
# ==================================================


def _float_columns(table: pd.DataFrame, columns: list[str]) -> list[str]:
    """Columns among `columns` stored as floats."""
    return [name for name in columns if table[name].dtype.kind == "f"]


def test_explanations_only_use_what_is_known_before_the_month(chain: dict[str, Any]) -> None:
    """The explanation of a month does not move when every target of the month and later and
    every later feature is replaced; a row's explanation only reacts to its own features;
    the smoothed regime is never read, the filtered regime of J is."""
    params = chain["prediction_params"]
    seed = params["random_state"]
    rng = np.random.default_rng(3)
    inputs = chain["inputs_a"]
    training, rows = month_rows(inputs, CHECKED_MONTH, TASK_A, params)
    base = explain_month(training, rows, TASK_A, NO_REGIME, params, seed)

    noisy = inputs.copy()
    is_unknown = (noisy[TARGET_DAY] >= CHECKED_MONTH).to_numpy()
    noisy.loc[is_unknown, TARGET_PRICE] = rng.normal(500, 300, int(is_unknown.sum()))
    is_later = (noisy[TARGET_DAY] >= CHECKED_MONTH + pd.offsets.MonthBegin(1)).to_numpy()
    for column in _float_columns(noisy, list(FEATURES_MODEL_A)):
        noisy.loc[is_later, column] = rng.normal(0, 1000, int(is_later.sum()))
    again = explain_month(
        *month_rows(noisy, CHECKED_MONTH, TASK_A, params), TASK_A, NO_REGIME, params, seed
    )
    pd.testing.assert_frame_equal(again.contributions, base.contributions)

    # The interventional check of the 15th only uses training rows as reference: the other
    # rows of the month (decided later) do not move it.
    check = _method_check(chain["params"])
    is_checked = (rows[TARGET_DAY].dt.day == check["day_of_month"]).to_numpy()
    checked = explain_month(
        training, rows, TASK_A, NO_REGIME, params, seed, rows[is_checked], check
    )
    other_rows = noisy.copy()
    is_other = (other_rows[TARGET_DAY] >= CHECKED_MONTH).to_numpy() & ~inputs.index.isin(
        rows.index[is_checked]
    )
    for column in _float_columns(other_rows, list(FEATURES_MODEL_A)):
        other_rows.loc[is_other, column] = rng.normal(0, 1000, int(is_other.sum()))
    other_training, other_month = month_rows(other_rows, CHECKED_MONTH, TASK_A, params)
    is_other_checked = (other_month[TARGET_DAY].dt.day == check["day_of_month"]).to_numpy()
    checked_again = explain_month(
        other_training,
        other_month,
        TASK_A,
        NO_REGIME,
        params,
        seed,
        other_month[is_other_checked],
        check,
    )
    pd.testing.assert_frame_equal(checked_again.interventional, checked.interventional)

    hour = rows.index[100]
    changed = inputs.copy()
    for column in _float_columns(changed, list(FEATURES_MODEL_A)):
        changed.loc[hour, column] += 300.0
    after = explain_month(
        *month_rows(changed, CHECKED_MONTH, TASK_A, params), TASK_A, NO_REGIME, params, seed
    )
    moved = (after.contributions - base.contributions).abs().max(axis=1) > 0
    assert moved[hour]
    assert int(moved.sum()) == 1

    # Model B with the regime columns of J (global): smoothed regimes scrambled, then the
    # features of one decision day changed.
    features_b, regimes = chain["features_b"], chain["regimes"]
    inputs_b = indexed_inputs(build_model_b_inputs(features_b, regimes, params), TASK_B)
    base_b = explain_month(
        *month_rows(inputs_b, CHECKED_MONTH, TASK_B, params), TASK_B, GLOBAL, params, seed
    )
    scrambled = regimes.copy()
    for column in regime_columns(SMOOTHED, REGIME_NAMES):
        scrambled[column] = scrambled[column].sample(frac=1.0, random_state=5).to_numpy()
    inputs_scrambled = indexed_inputs(build_model_b_inputs(features_b, scrambled, params), TASK_B)
    again_b = explain_month(
        *month_rows(inputs_scrambled, CHECKED_MONTH, TASK_B, params), TASK_B, GLOBAL, params, seed
    )
    pd.testing.assert_frame_equal(again_b.contributions, base_b.contributions)
    assert shap_column(REGIME_FILTERED) in base_b.contributions
    day = CHECKED_MONTH + pd.Timedelta(days=10)
    is_day = features_b.index.get_level_values(DECISION_DAY) == day
    shifted = features_b.copy()
    for column in _float_columns(shifted, list(FEATURES_MODEL_B)):
        shifted.loc[is_day, column] += 300.0
    inputs_shifted = indexed_inputs(build_model_b_inputs(shifted, regimes, params), TASK_B)
    after_b = explain_month(
        *month_rows(inputs_shifted, CHECKED_MONTH, TASK_B, params), TASK_B, GLOBAL, params, seed
    )
    moved_b = (after_b.contributions - base_b.contributions).abs().max(axis=1) > 0
    moved_days = moved_b[moved_b].index.get_level_values(DECISION_DAY).unique()
    assert list(moved_days) == [day]


# ==================================================
# 3. SHAP report
# ==================================================


def _day_table(days: int, contributions: dict[str, np.ndarray]) -> pd.DataFrame:
    """SHAP table of model A over `days` days of 24 hours from 1 June 2018, anchor 50, base 0."""
    first = pd.Timestamp("2018-06-01", tz=MARKET_TIMEZONE)
    hours = pd.date_range(first, periods=24 * days, freq="h").tz_convert("UTC")
    target_days = hours.tz_convert(MARKET_TIMEZONE).normalize().tz_localize(None)
    shap = {shap_column(name): values for name, values in contributions.items()}
    table = pd.DataFrame(shap, index=hours)
    total = table.sum(axis=1)
    return table.assign(
        **{
            DECISION_DAY: target_days - ONE_DAY,
            TARGET_DAY: target_days,
            REGIME_FILTERED: pd.array(np.zeros(len(hours), dtype=int), dtype="Int8"),
            ANCHOR: 50.0,
            PRED_SELECTED: 50.0 + total,
            BASE_VALUE: 0.0,
        }
    )


def test_shap_report_adds_up_and_ignores_the_reserved_period(chain: dict[str, Any]) -> None:
    """Families add up to the features; a feature constant within the day has no within-day
    importance; the choices section does not move with noise in the reserved period; every
    feature has a family and a label."""
    params = chain["params"]
    contributions = feature_contributions(chain["shap_a"])
    families = families_of(list(contributions.columns), params)
    sums = family_sums(contributions, families)
    np.testing.assert_allclose(sums.sum(axis=1), contributions.sum(axis=1), atol=1e-9)

    hours = np.tile(np.arange(24), 3)
    day_level = np.repeat([5.0, -3.0, 1.0], 24)
    table = _day_table(
        3, {HOUR: np.sin(hours / 24 * 2 * np.pi), TODAY_PRICES["price_mean"]: day_level}
    )
    values = pd.DataFrame(
        {HOUR: hours, TODAY_PRICES["price_mean"]: day_level + 60.0}, index=table.index
    )
    section = explanation_section(table, values, TASK_A, params, development=False)
    within = section["within_day_importance"]["features"]
    day = section["day_level_importance"]["features"]
    assert within[TODAY_PRICES["price_mean"]]["mean_abs_eur_mwh"] == 0.0
    assert within[HOUR]["mean_abs_eur_mwh"] > 0.5
    assert day[TODAY_PRICES["price_mean"]]["mean_abs_eur_mwh"] == pytest.approx(3.0)
    assert day[HOUR]["mean_abs_eur_mwh"] == pytest.approx(0.0, abs=1e-9)

    report = build_shap_report(
        chain["shap_a"],
        chain["shap_b"],
        chain["inputs_a"],
        chain["inputs_b"],
        chain["checks_a"],
        chain["checks_b"],
        params,
    )
    rng = np.random.default_rng(9)
    holdout = pd.Timestamp(PREDICTION_HOLDOUT)
    noisy_tables = {}
    for name in ("a", "b"):
        table = chain[f"shap_{name}"].copy()
        is_holdout = (table[TARGET_DAY] >= holdout).to_numpy()
        for column in feature_contributions(table).columns:
            table.loc[is_holdout, shap_column(column)] = rng.normal(0, 100, int(is_holdout.sum()))
        noisy_tables[name] = table
    noisy_inputs_a = chain["inputs_a"].copy()
    is_holdout_a = (noisy_inputs_a[TARGET_DAY] >= holdout).to_numpy()
    noisy_inputs_a.loc[is_holdout_a, PRICE_SAME_HOUR_TODAY] = rng.normal(
        0, 100, int(is_holdout_a.sum())
    )
    noisy = build_shap_report(
        noisy_tables["a"],
        noisy_tables["b"],
        noisy_inputs_a,
        chain["inputs_b"],
        chain["checks_a"],
        chain["checks_b"],
        params,
    )
    assert noisy["choices_before_holdout"] == report["choices_before_holdout"]
    assert noisy["holdout_descriptive_only"] != report["holdout_descriptive_only"]

    labels = labels_of(params)
    explained_b = [*FEATURES_MODEL_B, HORIZON_DAYS, *regime_feature_columns(params)]
    for features in (list(FEATURES_MODEL_A), explained_b):
        assert set(families_of(features, params)) <= set(labels)
        assert all(feature in labels for feature in features)
    with pytest.raises(ValueError, match="without a family"):
        families_of(["unknown_feature"], params)


# ==================================================
# 4. Rules of run 7
# ==================================================


def test_rules_of_run_7_on_known_cases() -> None:
    """Hour alignment passes when the contribution grows with the value inside the day and
    fails when it is reversed; driver agreement and the display level; method sensitivity."""
    params = make_interpretability_params()
    rng = np.random.default_rng(1)
    days = 30
    values = rng.normal(80, 20, 24 * days)
    noise = rng.normal(0, 1, 24 * days)
    table = _day_table(days, {PRICE_SAME_HOUR_TODAY: 0.5 * (values - 80) + noise})
    feature_values = pd.DataFrame({PRICE_SAME_HOUR_TODAY: values}, index=table.index)
    assert hour_alignment_rule(table, feature_values)["passes"]
    reversed_table = table.assign(**{shap_column(PRICE_SAME_HOUR_TODAY): -values})
    reversed_rule = hour_alignment_rule(reversed_table, feature_values)
    assert reversed_rule["investigate_bug"]

    first = pd.DataFrame(rng.normal(0, 1, (100, 4)), columns=["w", "x", "y", "z"])
    assert agreement(first, first, 3) == {"first_driver": 1.0, "top_overlap": 1.0}
    second = first.copy()
    largest = first.abs().to_numpy().argmax(axis=1)
    for row in range(30):
        second.iloc[row, largest[row]] = 0.0
    assert agreement(first, second, 1)["first_driver"] == pytest.approx(0.7)

    contributions = {
        HOUR: rng.normal(0, 3, 24 * days),
        PRICE_SAME_HOUR_TODAY: rng.normal(0, 2, 24 * days),
        TTF_LAST: np.repeat(rng.normal(0, 4, days), 24),
    }
    table_a = _day_table(days, contributions)
    table_b = _day_table(days, contributions)
    same = {"second_seed": feature_contributions(table_a).rename(columns=shap_column)}
    stable = driver_stability(table_a, same, table_b, same, params)
    assert stable["recommended_drivers"] == {"day": FEATURE_LEVEL, "hour": FEATURE_LEVEL}
    assert not stable["change_configuration"]
    swapped_values = feature_contributions(table_a)[[PRICE_SAME_HOUR_TODAY, HOUR, TTF_LAST]]
    swapped = swapped_values.set_axis([HOUR, PRICE_SAME_HOUR_TODAY, TTF_LAST], axis=1)
    unstable_seed = {"second_seed": swapped.rename(columns=shap_column)}
    unstable = driver_stability(table_a, unstable_seed, table_b, same, params)
    assert unstable["recommended_drivers"]["hour"] == FAMILY_LEVEL
    assert unstable["change_configuration"]

    sample = table_a.iloc[:48]
    identical = method_sensitivity(table_a, sample, params)
    assert identical["largest_family_share_gap"] == 0.0
    moved = sample.assign(**{shap_column(TTF_LAST): sample[shap_column(TTF_LAST)] * 3})
    assert method_sensitivity(table_a, moved, params)["largest_family_share_gap"] > 0.05


# ==================================================
# 5 and 6. Morning report
# ==================================================


def _morning_inputs(
    chain: dict[str, Any], anomalies: pd.DataFrame, decision_day: Any = DECISION
) -> dict[str, Any]:
    """Keyword arguments of `build_morning_report` for one decision day."""
    morning = {**make_interpretability_params()["morning_report"], "decision_day": decision_day}
    features_a = chain["features_a"]
    return {
        "predictions_model_a": chain["predictions_a"],
        "predictions_model_b": chain["predictions_b"],
        "alerts": chain["alerts"],
        "shap_model_a": chain["shap_a"],
        "shap_model_b": chain["shap_b"],
        "features_model_a": features_a,
        "features_model_b": chain["features_b"],
        "regimes": chain["regimes"],
        "anomalies": anomalies,
        "features_model_c": pd.DataFrame({PRICE: features_a[TARGET_PRICE]}),
        "backtest_daily": chain["backtest_daily"],
        "prediction_report": chain["prediction_report"],
        "params": make_interpretability_params(morning_report=morning),
        "backtest_params": chain["backtest_params"],
        "anomaly_params": make_anomaly_params(),
    }


def _with_regime_drivers(shap_a: pd.DataFrame) -> pd.DataFrame:
    """SHAP table A of a variant that sees the regime of J (global, weighted, hybrid, average):
    the regime of J and its probabilities become the largest drivers, the sum is unchanged."""
    columns = regime_feature_columns(make_interpretability_params())
    contributions = dict(zip(columns, REGIME_CONTRIBUTIONS, strict=True))
    with_regime = shap_a.assign(
        **{shap_column(name): value for name, value in contributions.items()}
    )
    return with_regime.assign(**{BASE_VALUE: shap_a[BASE_VALUE] - sum(REGIME_CONTRIBUTIONS)})


def _local(day: pd.Timestamp, hour: int) -> pd.Timestamp:
    """Instant of `hour` o'clock on a calendar day in Brussels, in UTC."""
    return (day + pd.Timedelta(hours=hour)).tz_localize(MARKET_TIMEZONE).tz_convert("UTC")


def _scramble(
    table: pd.DataFrame, is_unknown: np.ndarray, columns: list[str], rng: np.random.Generator
) -> pd.DataFrame:
    """`table` with `columns` replaced on the unknown rows: floats by random numbers, other
    types by the values of other rows (dtype kept). The day columns that identify a row
    (decision day, delivered day) are kept: they are known by construction."""
    noisy = table.copy()
    positions = np.flatnonzero(is_unknown)
    replaced = [column for column in columns if column not in (DECISION_DAY, TARGET_DAY)]
    for column in replaced:
        location = noisy.columns.get_loc(column)
        values = noisy[column]
        if values.dtype.kind == "f":
            noisy.iloc[positions, location] = rng.normal(100, 80, len(positions))
            continue
        shuffled = values.iloc[rng.permutation(len(values))]
        noisy.iloc[positions, location] = shuffled.iloc[positions].to_numpy()
    return noisy


def _unknown_at_eleven(inputs: dict[str, Any], decision_day: pd.Timestamp) -> dict[str, Any]:
    """Every input with what is unknown at 11:00 on `decision_day` replaced."""
    rng = np.random.default_rng(21)
    delivery = decision_day + ONE_DAY
    cutoff = decision_day.tz_localize(MARKET_TIMEZONE)
    noisy = dict(inputs)
    predictions_a = inputs["predictions_model_a"]
    is_later_a = (predictions_a[DECISION_DAY] > decision_day).to_numpy()
    realized = _scramble(
        predictions_a, (predictions_a[TARGET_DAY] >= delivery).to_numpy(), [TARGET_PRICE], rng
    )
    noisy["predictions_model_a"] = _scramble(realized, is_later_a, list(realized.columns), rng)
    alerts = inputs["alerts"]
    every_row = np.ones(len(alerts), dtype=bool)
    realized_levels = _scramble(alerts, every_row, [REALIZED_LEVEL], rng)
    noisy["alerts"] = _scramble(realized_levels, is_later_a, list(alerts.columns), rng)
    predictions_b = inputs["predictions_model_b"]
    other_b = predictions_b.index.get_level_values(DECISION_DAY) != decision_day
    realized_b = _scramble(
        predictions_b, np.ones(len(predictions_b), dtype=bool), [TARGET_MEAN_PRICE], rng
    )
    noisy["predictions_model_b"] = _scramble(realized_b, other_b, list(predictions_b.columns), rng)
    shap_a = inputs["shap_model_a"]
    noisy["shap_model_a"] = _scramble(
        shap_a, (shap_a[DECISION_DAY] != decision_day).to_numpy(), list(shap_a.columns), rng
    )
    shap_b = inputs["shap_model_b"]
    noisy["shap_model_b"] = _scramble(
        shap_b,
        shap_b.index.get_level_values(DECISION_DAY) != decision_day,
        list(shap_b.columns),
        rng,
    )
    features_a = inputs["features_model_a"]
    days_a = features_a.index.tz_convert(MARKET_TIMEZONE).normalize().tz_localize(None)
    target_a = _scramble(features_a, np.ones(len(features_a), dtype=bool), [TARGET_PRICE], rng)
    noisy["features_model_a"] = _scramble(
        target_a, np.asarray(days_a > delivery), list(features_a.columns), rng
    )
    features_b = inputs["features_model_b"]
    target_b = _scramble(features_b, np.ones(len(features_b), dtype=bool), [TARGET_MEAN_PRICE], rng)
    other_days_b = features_b.index.get_level_values(DECISION_DAY) != decision_day
    noisy["features_model_b"] = _scramble(target_b, other_days_b, list(features_b.columns), rng)
    regimes = inputs["regimes"]
    smoothed = _scramble(
        regimes, np.ones(len(regimes), dtype=bool), regime_columns(SMOOTHED, REGIME_NAMES), rng
    )
    noisy["regimes"] = _scramble(
        smoothed,
        np.asarray(regimes.index > decision_day),
        regime_columns(FILTERED, REGIME_NAMES),
        rng,
    )
    anomalies = inputs["anomalies"]
    noisy["anomalies"] = _scramble(
        anomalies, np.asarray(anomalies.index >= cutoff), list(anomalies.columns), rng
    )
    table_c = inputs["features_model_c"]
    noisy["features_model_c"] = _scramble(
        table_c, np.asarray(table_c.index >= cutoff), [PRICE], rng
    )
    daily = inputs["backtest_daily"]
    noisy["backtest_daily"] = _scramble(
        daily, np.asarray(daily.index > decision_day), list(daily.columns), rng
    )
    return noisy


def test_morning_report_only_uses_what_is_known_at_eleven(chain: dict[str, Any]) -> None:
    """Replacing everything unknown at 11:00 J changes nothing in the report, with the selected
    variant and with one that sees the regime of J; a prediction of J+1 or an anomalous hour
    of J-1 does change it."""
    yesterday = DECISION - ONE_DAY
    flagged = [_local(yesterday, 20), _local(yesterday, 21), _local(DECISION, 2)]
    anomalies = make_anomalies(chain["features_a"].index, flagged)
    inputs = _morning_inputs(chain, anomalies)
    report, markdown = build_morning_report(**inputs)
    assert report["anomalies_yesterday"]["anomalous_hours"] == 2
    assert report["anomalies_yesterday"]["events"][0]["provisional"]

    noisy_report, noisy_markdown = build_morning_report(**_unknown_at_eleven(inputs, DECISION))
    assert noisy_report == report
    assert noisy_markdown == markdown
    with_regime = {**inputs, "shap_model_a": _with_regime_drivers(chain["shap_a"])}
    regime_report, regime_markdown = build_morning_report(**with_regime)
    noisy_regime = build_morning_report(**_unknown_at_eleven(with_regime, DECISION))
    assert noisy_regime == (regime_report, regime_markdown)

    predictions_a = chain["predictions_a"]
    is_delivery_hour = (predictions_a[TARGET_DAY] == DECISION + ONE_DAY).to_numpy()
    first_hour = predictions_a.index[is_delivery_hour][0]
    moved = predictions_a.copy()
    moved.loc[first_hour, PRED_SELECTED] += 40.0
    moved_report, _ = build_morning_report(**{**inputs, "predictions_model_a": moved})
    assert moved_report["tomorrow"] != report["tomorrow"]
    more = make_anomalies(chain["features_a"].index, [*flagged, _local(yesterday, 8)])
    more_report, _ = build_morning_report(**{**inputs, "anomalies": more})
    assert more_report["anomalies_yesterday"]["anomalous_hours"] == 3


def test_morning_report_known_cases(chain: dict[str, Any]) -> None:
    """Default day, hours of the clock-change days, schedule of the optimizer and hours where
    it differs from the naive's, parts that add up, events final or provisional, and
    missing data reported or refused."""
    hours = chain["features_a"].index
    no_anomaly = make_anomalies(hours, [])
    latest, _ = build_morning_report(**_morning_inputs(chain, no_anomaly, decision_day=None))
    assert latest["header"]["decision_day"] == "2018-10-27"
    labels = [entry["hour"] for entry in latest["tomorrow"]["hours"]]
    assert len(labels) == 25
    assert {"02:00 CEST", "02:00 CET"} <= set(labels)
    spring = pd.date_range(
        pd.Timestamp("2019-03-31", tz=MARKET_TIMEZONE), periods=23, freq="h"
    ).tz_convert("UTC")
    spring_labels = hour_labels(spring, MARKET_TIMEZONE)
    assert len(set(spring_labels)) == 23
    assert "02:00" not in spring_labels

    report, _ = build_morning_report(**_morning_inputs(chain, no_anomaly))
    predictions_a = chain["predictions_a"]
    is_delivery = (predictions_a[TARGET_DAY] == DECISION + ONE_DAY).to_numpy()
    rows = predictions_a[is_delivery].sort_index()
    client = client_profile(chain["backtest_params"])
    model = optimal_schedule(rows[PRED_SELECTED].to_numpy(), client, 5.0)
    naive = optimal_schedule(rows[PRED_NAIVE].to_numpy(), client, 5.0)
    entries = report["tomorrow"]["hours"]
    loads = np.array([entry["flexible_load_mw"]["model"] for entry in entries])
    np.testing.assert_allclose(loads, model.flexible_load, atol=1e-3)
    differs = np.abs(model.flexible_load - naive.flexible_load) > 1e-6
    differing = [hour["hour"] for hour in report["schedule"]["naive"]["differing_hours"]]
    assert differing == [entries[position]["hour"] for position in np.flatnonzero(differs)]

    level = report["tomorrow"]["level"]
    parts = level["mean_price_today_eur_mwh"] + level["base_value_eur_mwh"] + level["other_eur_mwh"]
    parts += sum(driver["contribution_eur_mwh"] for driver in level["drivers"])
    assert parts == pytest.approx(level["mean_predicted_price_eur_mwh"], abs=ROUNDING_EUR_MWH)
    for entry in entries:
        shape = entry["shape"]
        hour_parts = shape["other_eur_mwh"] + sum(
            d["contribution_eur_mwh"] for d in shape["drivers"]
        )
        assert hour_parts == pytest.approx(entry["gap_to_day_mean_eur_mwh"], abs=ROUNDING_EUR_MWH)

    # A variant that sees the regime of J: its values come from the regimes table (J), the
    # regime shown by its name; the hours' shapes do not change (same regime all day).
    regime_report, regime_markdown = build_morning_report(
        **{
            **_morning_inputs(chain, no_anomaly),
            "shap_model_a": _with_regime_drivers(chain["shap_a"]),
        }
    )
    regime_of_j = chain["regimes"].loc[DECISION, regime_feature_columns(chain["params"])]
    label_driver, probability_driver, *_ = regime_report["tomorrow"]["level"]["drivers"]
    assert label_driver["name"] == REGIME_FILTERED
    assert label_driver["value"] == REGIME_NAMES[int(regime_of_j.iloc[0])]
    assert f"filtered regime today ({label_driver['value']}) +50.0" in regime_markdown
    assert probability_driver["value"] == pytest.approx(regime_of_j.iloc[1], abs=0.005)
    assert regime_report["tomorrow"]["hours"] == report["tomorrow"]["hours"]

    yesterday = DECISION - ONE_DAY
    final_hours = [_local(yesterday, 10), _local(yesterday, 11)]
    final, _ = build_morning_report(**_morning_inputs(chain, make_anomalies(hours, final_hours)))
    final_event = final["anomalies_yesterday"]["events"][0]
    assert final_event["anomalous_hours"] == 2
    assert not final_event["provisional"]
    late_hours = [_local(yesterday, 19)]
    late, _ = build_morning_report(**_morning_inputs(chain, make_anomalies(hours, late_hours)))
    assert late["anomalies_yesterday"]["events"][0]["provisional"]

    incomplete = predictions_a.copy()
    incomplete.loc[rows.index[5], PRED_SELECTED] = np.nan
    with pytest.raises(ValueError, match="23 of 24 hours predicted"):
        build_morning_report(
            **{**_morning_inputs(chain, no_anomaly), "predictions_model_a": incomplete}
        )
    no_naive = predictions_a.copy()
    no_naive.loc[rows.index[5], PRED_NAIVE] = np.nan
    without_naive, _ = build_morning_report(
        **{**_morning_inputs(chain, no_anomaly), "predictions_model_a": no_naive}
    )
    assert not without_naive["schedule"]["naive"]["available"]
    predictions_b = chain["predictions_b"].copy()
    predictions_b.loc[(DECISION, 3), PRED_SELECTED] = np.nan
    partial, _ = build_morning_report(
        **{**_morning_inputs(chain, no_anomaly), "predictions_model_b": predictions_b}
    )
    availability = [entry["available"] for entry in partial["next_days"]]
    assert availability == [True, False, True]
