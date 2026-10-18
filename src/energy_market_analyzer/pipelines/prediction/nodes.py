"""Prediction pipeline nodes: walk-forward predictions of models A and B, intervals, alerts.

Information rule (decisions.md, section 6): for a day delivered on J+1, every
input is known at 11:00 Brussels on the decision day J.

- Inputs: feature tables A and B (features pipeline) and the filtered regime of
  the decision day J (regime pipeline, known at 11:00 J); never the smoothed
  regime, never an output of the anomaly pipeline (decision D24).
- Walk-forward (decision D25): on the first day of each month, every model is
  refitted on the rows whose predicted day is before that day, then predicts the
  month. Development: predicted days before holdout_start; the refits go on
  through the reserved period with the same settings (decision D9).
- The variant kept is chosen by a rule fixed in advance, on the development
  period only (decision D27); intervals are calibrated each day with the errors
  known at 11:00 J (decision D28).
- Alerts (decision D29): level of each predicted hour against quantiles of the
  realized prices of its regime over the previous 24 months, probability of
  exceeding them, and rank of the hour in its day.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from energy_market_analyzer.pipelines.anomaly.nodes import closest_group
from energy_market_analyzer.pipelines.features.nodes import (
    DECISION_DAY,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    HORIZON_DAYS,
    PRICE_SAME_HOUR_TODAY,
    TARGET_MEAN_PRICE,
    TARGET_PRICE,
    market_days,
)
from energy_market_analyzer.pipelines.ingestion.nodes import ONE_DAY, UTC_ZONE
from energy_market_analyzer.pipelines.prediction.evaluation import (
    SHARE_DECIMALS,
    alerts_summary,
    calibrate_intervals,
    epoch_summary,
    holdout_alarms,
    holdout_day,
    interval_metrics,
    is_development,
    period_summary,
    rounded,
    rules_for_run_5,
    select_variant,
    tree_curve,
)
from energy_market_analyzer.pipelines.prediction.models import (
    ALERT_CONFIDENCE,
    ALERT_LEVEL,
    ALERT_LEVELS,
    ANCHOR,
    CONFIDENCE_CLASSES,
    INTERVAL_CALIBRATED,
    INTERVAL_LOWER,
    INTERVAL_LOWER_RAW,
    INTERVAL_UPPER,
    INTERVAL_UPPER_RAW,
    IS_CHEAPEST_HOUR,
    IS_MOST_EXPENSIVE_HOUR,
    LEVEL_TARGET,
    NO_PRICE,
    PRED_PRICE,
    PROB_ABOVE_CRITICAL,
    PROB_ABOVE_HIGH,
    RANK_IN_DAY,
    REALIZED_LEVEL,
    RIDGE,
    SELECTED,
    TARGET_DAY,
    TASK_A,
    TASK_B,
    THRESHOLD_CRITICAL,
    THRESHOLD_HIGH,
    THRESHOLD_REGIME,
    TRAINING_END,
    TREES_PREFIX,
    MonthData,
    Task,
    baseline_predictions,
    fit_month,
    prediction_column,
    regime_feature_columns,
    window_label,
)
from energy_market_analyzer.pipelines.regime.nodes import LABEL_DTYPE, REGIME_FILTERED

logger = logging.getLogger(__name__)

MONTH_FORMAT = "%Y-%m"
DAY_UNIT = "ns"  # same resolution as the other day columns
TEXT_DTYPE = "string"
FLAG_DTYPE = "boolean"
RANK_DTYPE = "Int8"
ALL_REGIMES = "all"  # thresholds from every hour of the window: regime unknown or too rare
UNKNOWN_REGIME = -1
MEDIAN_LEVEL = 0.5  # the point prediction is a median (absolute error loss)
PROBABILITY_DECIMALS = 6
SECONDS_DECIMALS = 1
# Context columns copied into the prediction tables, after their index.
CONTEXT_A = (DECISION_DAY, TARGET_DAY, REGIME_FILTERED, TARGET_PRICE, PRICE_SAME_HOUR_TODAY, ANCHOR)
CONTEXT_B = (DECISION_DAY, HORIZON_DAYS, TARGET_DAY, REGIME_FILTERED, TARGET_MEAN_PRICE, ANCHOR)
ALERT_COLUMNS = (
    DECISION_DAY,
    REGIME_FILTERED,
    PRED_PRICE,
    INTERVAL_LOWER,
    INTERVAL_UPPER,
    THRESHOLD_REGIME,
    THRESHOLD_HIGH,
    THRESHOLD_CRITICAL,
    PROB_ABOVE_HIGH,
    PROB_ABOVE_CRITICAL,
    ALERT_LEVEL,
    ALERT_CONFIDENCE,
    RANK_IN_DAY,
    IS_CHEAPEST_HOUR,
    IS_MOST_EXPENSIVE_HOUR,
    REALIZED_LEVEL,
)


# ==================================================
# Inputs: features and the filtered regime of the decision day J
# ==================================================


def _require_regimes(regimes: pd.DataFrame, params: dict[str, Any]) -> None:
    """Raise ValueError unless the regimes table holds the filtered regime and its probabilities."""
    missing_columns = [name for name in regime_feature_columns(params) if name not in regimes]
    if missing_columns:
        raise ValueError(f"regimes: missing columns {missing_columns}, re-run the regime pipeline")


def _regime_of_decision_day(
    regimes: pd.DataFrame, decision_days: pd.DatetimeIndex, params: dict[str, Any]
) -> pd.DataFrame:
    """Filtered regime of each row's decision day J and its probabilities (known at 11:00 J).

    The smoothed regime, which looks at later days, is never read.
    """
    columns = regime_feature_columns(params)
    return regimes[columns].reindex(decision_days)


def build_model_a_inputs(
    features_model_a: pd.DataFrame, regimes: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Model A rows: features of the delivered hour and the filtered regime of J.

    Known at 11:00 J: the features (features pipeline), the filtered regime of J
    (decision D19). The delivered day is J + 1 in Brussels.

    Args:
        features_model_a: Feature table A (one row per delivered hour, UTC index).
        regimes: Daily regimes table (regime pipeline).
        params: Prediction parameters.

    Returns:
        Table A with decision_day, target_day and the regime columns of J added.

    Raises:
        ValueError: If a table is not as expected.
    """
    index = features_model_a.index
    is_utc = isinstance(index, pd.DatetimeIndex) and str(index.tz) == UTC_ZONE
    if not is_utc:
        raise ValueError("features_model_a: index must be a DatetimeIndex in UTC")
    missing_columns = [
        name for name in (*FEATURES_MODEL_A, TARGET_PRICE) if name not in features_model_a
    ]
    if missing_columns:
        raise ValueError(f"features_model_a: missing columns {missing_columns}, re-run features")
    _require_regimes(regimes, params)
    target_days = market_days(index, params["market_timezone"])
    decision_days = target_days - ONE_DAY
    regime_of_j = _regime_of_decision_day(regimes, decision_days, params).set_axis(index)
    days = pd.DataFrame({DECISION_DAY: decision_days, TARGET_DAY: target_days}, index=index)
    inputs = pd.concat([features_model_a, days, regime_of_j], axis=1)
    logger.info("Model A inputs: %d delivered hours", len(inputs))
    return inputs


def build_model_b_inputs(
    features_model_b: pd.DataFrame, regimes: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Model B rows: features of the predicted day and the filtered regime of J.

    The horizon becomes a feature: one model learns the three horizons.

    Args:
        features_model_b: Feature table B (index decision_day, horizon_days).
        regimes: Daily regimes table (regime pipeline).
        params: Prediction parameters.

    Returns:
        One row per (decision_day, horizon_days), as columns, with target_day and
        the regime columns of J added.

    Raises:
        ValueError: If a table is not as expected.
    """
    if list(features_model_b.index.names) != [DECISION_DAY, HORIZON_DAYS]:
        raise ValueError("features_model_b: index must be (decision_day, horizon_days)")
    missing_columns = [
        name for name in (*FEATURES_MODEL_B, TARGET_MEAN_PRICE) if name not in features_model_b
    ]
    if missing_columns:
        raise ValueError(f"features_model_b: missing columns {missing_columns}, re-run features")
    _require_regimes(regimes, params)
    flat = features_model_b.reset_index()
    horizons = pd.to_timedelta(flat[HORIZON_DAYS], unit="D")
    decision_days = pd.DatetimeIndex(flat[DECISION_DAY])
    regime_of_j = _regime_of_decision_day(regimes, decision_days, params).set_axis(flat.index)
    with_target_day = flat.assign(**{TARGET_DAY: flat[DECISION_DAY] + horizons})
    inputs = pd.concat([with_target_day, regime_of_j], axis=1)
    logger.info("Model B inputs: %d rows", len(inputs))
    return inputs


# ==================================================
# Walk-forward: monthly refit on the past
# ==================================================


def test_months(period_days: pd.Series, params: dict[str, Any]) -> list[pd.Timestamp]:
    """First day of each tested month: min_train_months after the month of the first row.

    Raises:
        ValueError: If no month is left to test.
    """
    first_month = period_days.min().to_period("M") + params["min_train_months"]
    months = period_days.dt.to_period("M").unique()
    month_starts = sorted(month.start_time for month in months if month >= first_month)
    if not month_starts:
        raise ValueError(f"No month left to test after {params['min_train_months']} months")
    return month_starts


def training_mask(
    target_days: pd.Series, month_start: pd.Timestamp, window_months: int | None
) -> np.ndarray:
    """True for the rows whose predicted day is before `month_start`, within the window.

    The price of a predicted day d is published on d - 1 at 12:45: every row
    before the month is known at 11:00 on the last day before it.
    """
    is_past = target_days < month_start
    if window_months is None:
        return is_past.to_numpy()
    window_start = month_start - pd.DateOffset(months=window_months)
    is_in_window = target_days >= window_start
    return (is_past & is_in_window).to_numpy()


def _month_data(
    inputs: pd.DataFrame, month_start: pd.Timestamp, task: Task, params: dict[str, Any]
) -> MonthData:
    """Training rows, rows of the month and, for development months of A, the other windows."""
    next_start = month_start + pd.offsets.MonthBegin(1)
    period_days = inputs[task.period_column]
    is_month = ((period_days >= month_start) & (period_days < next_start)).to_numpy()
    is_development_month = month_start < holdout_day(params)
    window = params["training_window_months"]
    training = inputs[training_mask(inputs[TARGET_DAY], month_start, window)]
    other_windows = {}
    if task.noise_checks and is_development_month:
        other_windows = {
            candidate: inputs[training_mask(inputs[TARGET_DAY], month_start, candidate)]
            for candidate in params["window_candidates_months"]
            if candidate != window
        }
    return MonthData(training, inputs[is_month], other_windows, is_development_month)


def walk_forward(
    inputs: pd.DataFrame, task: Task, params: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Predictions of every model, month by month, each model refitted on the past.

    Args:
        inputs: Output of `build_model_a_inputs` or `build_model_b_inputs`.
        task: TASK_A or TASK_B.
        params: Prediction parameters.

    Returns:
        One row per predicted row: context columns, baselines, every model's
        prediction, raw interval and tree curve; and one record per refit.
    """
    tables, records = [], []
    for month_start in test_months(inputs[task.period_column], params):
        started = time.perf_counter()
        data = _month_data(inputs, month_start, task, params)
        table, record = fit_month(data, task, params)
        seconds = round(time.perf_counter() - started, SECONDS_DECIMALS)
        month = month_start.strftime(MONTH_FORMAT)
        training_end = pd.Timestamp(record["training_end"]).as_unit(DAY_UNIT)
        tables.append(table.assign(**{TRAINING_END: training_end}))
        records.append({"month": month, **record, "seconds": seconds})
        logger.info(
            "%s %s: %d training rows, %d rows predicted (%.1f s)",
            task.name,
            month,
            record["training_rows"],
            record["predicted_rows"],
            seconds,
        )
    predictions = pd.concat(tables)
    predicted_inputs = inputs.loc[predictions.index]
    context_columns = list(CONTEXT_A if task.name == TASK_A.name else CONTEXT_B)
    baselines = baseline_predictions(predicted_inputs, task)
    table = pd.concat([predicted_inputs[context_columns], baselines, predictions], axis=1)
    return table, records


def walk_forward_model_a(
    prediction_inputs_a: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Model A: walk-forward predictions of every hourly model (see `walk_forward`)."""
    return walk_forward(prediction_inputs_a, TASK_A, params)


def walk_forward_model_b(
    prediction_inputs_b: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Model B: walk-forward predictions of every daily model (see `walk_forward`)."""
    return walk_forward(prediction_inputs_b, TASK_B, params)


# ==================================================
# Selected variant and calibrated intervals
# ==================================================


def _calibration_candidates(
    table: pd.DataFrame, task: Task, params: dict[str, Any]
) -> dict[str, Any]:
    """Coverage of the calibrated interval with each candidate window, development rows only."""
    development = table[is_development(table, params)]
    candidates = {}
    for days in params["interval"]["calibration_candidates_days"]:
        intervals = calibrate_intervals(development, task, days, params)
        calibrated = development.assign(**intervals)[intervals[INTERVAL_CALIBRATED].to_numpy()]
        candidates[str(days)] = interval_metrics(
            calibrated, task, INTERVAL_LOWER, INTERVAL_UPPER, params
        )
    return candidates


def finalize_task(
    raw: pd.DataFrame, task: Task, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Selected variant (development rule), calibrated interval, choices of the task.

    The selection and the candidate windows only read development rows; the
    calibrated interval of a row only uses errors known at its decision time.
    """
    development = raw[is_development(raw, params)]
    selection = select_variant(development, task, params)
    selected = raw[prediction_column(selection["selected"])]
    with_selected = raw.assign(**{prediction_column(SELECTED): selected})
    calibration_days = params["interval"]["calibration_days"]
    intervals = calibrate_intervals(with_selected, task, calibration_days, params)
    is_guarded = intervals.pop("guarded")
    is_calibrated_development = intervals[INTERVAL_CALIBRATED] & is_development(raw, params)
    choices = {
        "selection": selection,
        "tree_curve": tree_curve(development, task),
        "calibration_candidates": _calibration_candidates(with_selected, task, params),
        "share_of_calibrated_rows_at_width_floor": rounded(
            is_guarded[is_calibrated_development].mean(), SHARE_DECIMALS
        ),
    }
    table = with_selected.assign(**intervals)
    logger.info("%s: variant selected on the development period: %s", task.name, selection)
    return table[_ordered_columns(table, task)], choices


def _ordered_columns(table: pd.DataFrame, task: Task) -> list[str]:
    """Context, then baselines, ridge, candidates, ablations, checks, selected, intervals.

    The tree curve stays out: it only feeds rule 3 of run 5.
    """
    context = list(CONTEXT_A if task.name == TASK_A.name else CONTEXT_B)
    first_models = [*task.baselines, RIDGE, *task.candidates, LEVEL_TARGET, NO_PRICE]
    first_columns = [prediction_column(model) for model in first_models]
    interval_columns = [
        INTERVAL_LOWER_RAW,
        INTERVAL_UPPER_RAW,
        INTERVAL_LOWER,
        INTERVAL_UPPER,
        INTERVAL_CALIBRATED,
    ]
    placed = {
        *context,
        TRAINING_END,
        *first_columns,
        *interval_columns,
        prediction_column(SELECTED),
    }
    checks = [
        column
        for column in table.columns
        if column not in placed and not column.startswith(TREES_PREFIX)
    ]
    selected = [prediction_column(SELECTED)]
    return [*context, TRAINING_END, *first_columns, *checks, *selected, *interval_columns]


def finalize_predictions(
    raw_predictions_a: pd.DataFrame, raw_predictions_b: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prediction tables of models A and B, and the choices made on the development period.

    Args:
        raw_predictions_a: First output of `walk_forward_model_a`.
        raw_predictions_b: First output of `walk_forward_model_b`.
        params: Prediction parameters.

    Returns:
        predictions_model_a (UTC index), predictions_model_b (index decision_day,
        horizon_days) and the choices of each model.
    """
    table_a, choices_a = finalize_task(raw_predictions_a, TASK_A, params)
    table_b, choices_b = finalize_task(raw_predictions_b, TASK_B, params)
    indexed_b = table_b.set_index([DECISION_DAY, HORIZON_DAYS]).sort_index()
    choices = {TASK_A.name: choices_a, TASK_B.name: choices_b}
    return table_a, indexed_b, choices


# ==================================================
# Alerts
# ==================================================


def reference_prices(prediction_inputs_a: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Realized price of every delivered hour and the filtered regime of its own day."""
    days = prediction_inputs_a[TARGET_DAY]
    day_regimes = regimes[REGIME_FILTERED].reindex(days).to_numpy(dtype=float, na_value=np.nan)
    return pd.DataFrame(
        {TARGET_DAY: days, "price": prediction_inputs_a[TARGET_PRICE], "regime": day_regimes},
        index=prediction_inputs_a.index,
    )


def month_thresholds(
    reference: pd.DataFrame, month_start: pd.Timestamp, params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Alert thresholds of one month, for each regime of J and for an unknown regime.

    Reference: realized prices of the delivered days of the previous
    reference_window_months months, in the days of the regime (the closest
    regime with enough hours, else every hour). Known on the last day before the month.
    """
    alerts = params["alerts"]
    regime_names = params["regime_names"]
    is_window = training_mask(reference[TARGET_DAY], month_start, alerts["reference_window_months"])
    window = reference[is_window & reference["price"].notna().to_numpy()]
    counts = window["regime"].value_counts()
    eligible = [
        state
        for state in range(len(regime_names))
        if counts.get(float(state), 0) >= alerts["min_reference_hours"]
    ]
    records = []
    for state in [*range(len(regime_names)), UNKNOWN_REGIME]:
        source = closest_group(state, eligible) if state != UNKNOWN_REGIME else None
        prices = (
            window["price"] if source is None else window.loc[window["regime"] == source, "price"]
        )
        records.append(
            {
                "month": month_start,
                "regime": state,
                THRESHOLD_REGIME: ALL_REGIMES if source is None else regime_names[source],
                THRESHOLD_HIGH: float(prices.quantile(alerts["high_quantile"])),
                THRESHOLD_CRITICAL: float(prices.quantile(alerts["critical_quantile"])),
                "reference_hours": len(prices),
            }
        )
    return records


def exceedance_probability(
    threshold: np.ndarray,
    lower: np.ndarray,
    point: np.ndarray,
    upper: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """Probability that the price is at or above `threshold`, read on the predictive interval.

    The distribution goes linearly through (lower, lower_quantile), (point, 0.5)
    and (upper, upper_quantile), the three values sorted. Beyond the interval
    nothing is known: the probability stays between 1 - upper_quantile and
    1 - lower_quantile (0.1 and 0.9 with the default interval).
    """
    interval = params["interval"]
    low_level = interval["lower_quantile"]
    high_level = interval["upper_quantile"]
    knots = np.sort(np.column_stack([lower, point, upper]), axis=1)
    first, middle, last = knots[:, 0], knots[:, 1], knots[:, 2]
    rising = low_level + (MEDIAN_LEVEL - low_level) * _fraction(threshold, first, middle)
    falling = MEDIAN_LEVEL + (high_level - MEDIAN_LEVEL) * _fraction(threshold, middle, last)
    below_threshold = np.where(
        threshold <= first,
        low_level,
        np.where(threshold <= middle, rising, np.where(threshold <= last, falling, high_level)),
    )
    probability = np.round(1.0 - below_threshold, PROBABILITY_DECIMALS)
    is_known = ~np.isnan(np.column_stack([threshold, lower, point, upper])).any(axis=1)
    return np.where(is_known, probability, np.nan)


def _fraction(values: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Position of `values` between `start` and `end` (0 to 1); 1 when the segment is empty."""
    length = end - start
    has_length = length > 0
    safe_length = np.where(has_length, length, 1.0)
    return np.where(has_length, np.clip((values - start) / safe_length, 0.0, 1.0), 1.0)


def alert_levels(prices: np.ndarray, high: np.ndarray, critical: np.ndarray) -> pd.Series:
    """normal / high / critical from a price and the two thresholds; missing if one is unknown."""
    levels = np.where(
        prices >= critical,
        ALERT_LEVELS[2],
        np.where(prices >= high, ALERT_LEVELS[1], ALERT_LEVELS[0]),
    )
    is_known = ~np.isnan(np.column_stack([prices, high, critical])).any(axis=1)
    return pd.Series(np.where(is_known, levels, None)).astype(TEXT_DTYPE)


def alert_confidence(
    levels: pd.Series, above_high: np.ndarray, above_critical: np.ndarray, params: dict[str, Any]
) -> pd.Series:
    """Confidence of each level: probability that the hour really belongs to it or above.

    critical: P(price >= critical threshold); high: P(price >= high threshold);
    normal: P(price < high threshold). Medium and high from the configured
    probabilities (0.7 and 0.9).
    """
    level_values = levels.to_numpy(dtype=object, na_value=None)
    probability = np.where(
        level_values == ALERT_LEVELS[2],
        above_critical,
        np.where(level_values == ALERT_LEVELS[1], above_high, 1.0 - above_high),
    )
    medium, high = params["alerts"]["confidence_probabilities"]
    classes = np.where(
        probability >= high,
        CONFIDENCE_CLASSES[2],
        np.where(probability >= medium, CONFIDENCE_CLASSES[1], CONFIDENCE_CLASSES[0]),
    )
    is_known = pd.notna(level_values) & ~np.isnan(probability)
    return pd.Series(np.where(is_known, classes, None)).astype(TEXT_DTYPE)


def daily_ranks(predicted: pd.Series, days: pd.Series, params: dict[str, Any]) -> pd.DataFrame:
    """Rank of each hour's predicted price in its Brussels day (1 = cheapest), and the
    cheapest and most expensive hours. A day with an hour without prediction is not ranked.

    Days of 23 and 25 hours keep their own number of hours; on a tie the earlier hour ranks first.
    """
    alerts = params["alerts"]
    grouped = predicted.groupby(days)
    is_complete = (grouped.transform("count") == grouped.transform("size")).to_numpy()
    ranks = grouped.rank(method="first").to_numpy()
    hours = grouped.transform("size").to_numpy()
    complete_ranks = np.where(is_complete, ranks, np.nan)
    is_cheapest = np.where(is_complete, ranks <= alerts["cheapest_hours"], np.nan)
    is_expensive = np.where(is_complete, ranks > hours - alerts["most_expensive_hours"], np.nan)
    columns = {
        RANK_IN_DAY: pd.array(complete_ranks, dtype=RANK_DTYPE),
        IS_CHEAPEST_HOUR: pd.array(is_cheapest, dtype=FLAG_DTYPE),
        IS_MOST_EXPENSIVE_HOUR: pd.array(is_expensive, dtype=FLAG_DTYPE),
    }
    return pd.DataFrame(columns, index=predicted.index)


def build_alerts(
    predictions_model_a: pd.DataFrame,
    prediction_inputs_a: pd.DataFrame,
    regimes: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Alert of each predicted hour of model A, with its confidence and its rank in the day.

    Known at 11:00 J: the selected prediction and its calibrated interval, the
    regime of J, the thresholds of the month (prices of the previous months).
    realized_level uses the realized price: for the evaluation only.

    Args:
        predictions_model_a: First output of `finalize_predictions`.
        prediction_inputs_a: Output of `build_model_a_inputs` (realized prices of every hour).
        regimes: Daily regimes table (regime pipeline).
        params: Prediction parameters.

    Returns:
        The alerts table (ALERT_COLUMNS, UTC index) and the thresholds of each month.
    """
    reference = reference_prices(prediction_inputs_a, regimes)
    months = sorted(set(predictions_model_a[TARGET_DAY].dt.to_period("M").dt.start_time))
    records = [record for month in months for record in month_thresholds(reference, month, params)]
    thresholds = pd.DataFrame(records).set_index(["month", "regime"])
    row_months = predictions_model_a[TARGET_DAY].dt.to_period("M").dt.start_time
    row_regimes = predictions_model_a[REGIME_FILTERED].astype(float).fillna(UNKNOWN_REGIME)
    keys = pd.MultiIndex.from_arrays([row_months, row_regimes.astype(int)])
    row_thresholds = thresholds.reindex(keys).set_axis(predictions_model_a.index)
    alerts = _alert_columns(predictions_model_a, row_thresholds, params)
    thresholds_log = [
        {**record, "month": record["month"].strftime(MONTH_FORMAT)} for record in records
    ]
    logger.info("Alerts: %s", alerts[ALERT_LEVEL].value_counts().to_dict())
    return alerts, thresholds_log


def _alert_columns(
    predictions: pd.DataFrame, thresholds: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Levels, probabilities, confidence, ranks and realized level of each predicted hour."""
    point = predictions[prediction_column(SELECTED)].to_numpy(dtype=float)
    lower = predictions[INTERVAL_LOWER].to_numpy(dtype=float)
    upper = predictions[INTERVAL_UPPER].to_numpy(dtype=float)
    high = thresholds[THRESHOLD_HIGH].to_numpy(dtype=float)
    critical = thresholds[THRESHOLD_CRITICAL].to_numpy(dtype=float)
    above_high = exceedance_probability(high, lower, point, upper, params)
    above_critical = exceedance_probability(critical, lower, point, upper, params)
    levels = alert_levels(point, high, critical).set_axis(predictions.index)
    realized = predictions[TASK_A.target].to_numpy(dtype=float)
    columns = {
        DECISION_DAY: predictions[DECISION_DAY],
        REGIME_FILTERED: predictions[REGIME_FILTERED].astype(LABEL_DTYPE),
        PRED_PRICE: point,
        INTERVAL_LOWER: lower,
        INTERVAL_UPPER: upper,
        THRESHOLD_REGIME: thresholds[THRESHOLD_REGIME].astype(TEXT_DTYPE),
        THRESHOLD_HIGH: high,
        THRESHOLD_CRITICAL: critical,
        PROB_ABOVE_HIGH: above_high,
        PROB_ABOVE_CRITICAL: above_critical,
        ALERT_LEVEL: levels,
        ALERT_CONFIDENCE: alert_confidence(levels, above_high, above_critical, params).set_axis(
            predictions.index
        ),
        REALIZED_LEVEL: alert_levels(realized, high, critical).set_axis(predictions.index),
    }
    table = pd.DataFrame(columns, index=predictions.index)
    ranks = daily_ranks(predictions[prediction_column(SELECTED)], predictions[TARGET_DAY], params)
    return pd.concat([table, ranks], axis=1)[list(ALERT_COLUMNS)]


# ==================================================
# Report
# ==================================================


def _coverage(table: pd.DataFrame, task: Task, params: dict[str, Any]) -> dict[str, Any]:
    """Tested months and predicted rows of a task, by period."""
    months = table[task.period_column].dt.strftime(MONTH_FORMAT)
    is_development_row = is_development(table, params)
    return {
        "first_tested_month": months.min(),
        "last_tested_month": months.max(),
        "tested_months": int(months.nunique()),
        "predicted_rows": len(table),
        "development_rows": int(is_development_row.sum()),
        "holdout_rows": int((~is_development_row).sum()),
        "rows_without_anchor": int(table[ANCHOR].isna().sum()),
    }


def _period(table: pd.DataFrame, params: dict[str, Any], development: bool) -> pd.DataFrame:
    """Rows of one period, split by the predicted day."""
    is_development_row = is_development(table, params)
    return table[is_development_row if development else ~is_development_row]


def build_prediction_report(
    predictions_model_a: pd.DataFrame,
    predictions_model_b: pd.DataFrame,
    alerts: pd.DataFrame,
    prediction_choices: dict[str, Any],
    refits_a: list[dict[str, Any]],
    refits_b: list[dict[str, Any]],
    alert_thresholds: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Report of the predictions and alerts, in two sections and the refit logs.

    - choices_before_holdout: the only section to read when choosing; nothing in
      it uses a predicted day of the reserved period.
    - holdout_descriptive_only: the same figures on the reserved period, read
      after the choices; the headline results are the selected variants there.

    Args:
        predictions_model_a: First output of `finalize_predictions`.
        predictions_model_b: Second output of `finalize_predictions`.
        alerts: First output of `build_alerts`.
        prediction_choices: Third output of `finalize_predictions`.
        refits_a: Second output of `walk_forward_model_a`.
        refits_b: Second output of `walk_forward_model_b`.
        alert_thresholds: Second output of `build_alerts`.
        params: Prediction parameters.

    Returns:
        A JSON-ready report.
    """
    tables = {TASK_A.name: predictions_model_a, TASK_B.name: predictions_model_b.reset_index()}
    sections = {}
    for name, development in (("development", True), ("holdout", False)):
        section = {
            task.name: period_summary(_period(tables[task.name], params, development), task, params)
            for task in (TASK_A, TASK_B)
        }
        period_a = _period(tables[TASK_A.name], params, development)
        section["alerts"] = alerts_summary(alerts.loc[period_a.index], period_a, params)
        sections[name] = section
    development_section = sections["development"]
    for task in (TASK_A, TASK_B):
        development_rows = _period(tables[task.name], params, development=True)
        development_section[task.name] = {
            **prediction_choices[task.name],
            **development_section[task.name],
            "by_development_epoch": epoch_summary(development_rows, task, params),
        }
    development_section["rules_for_run_5"] = rules_for_run_5(
        development_section, prediction_choices, params
    )
    sections["holdout"]["alarms"] = holdout_alarms(development_section, sections["holdout"])
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "holdout_start": str(params["holdout_start"]),
        "regime_names": list(params["regime_names"]),
        "training_window": window_label(params["training_window_months"]),
        "coverage": {
            task.name: _coverage(tables[task.name], task, params) for task in (TASK_A, TASK_B)
        },
        "choices_before_holdout": development_section,
        "holdout_descriptive_only": sections["holdout"],
        "refits": {"model_a": refits_a, "model_b": refits_b, "alert_thresholds": alert_thresholds},
    }
    _log_report(report)
    return report


def _log_report(report: dict[str, Any]) -> None:
    """Main figures in the log; warnings on the rules of run 5 that ask for a decision."""
    choices = report["choices_before_holdout"]
    for name in (TASK_A.name, TASK_B.name):
        metrics = choices[name]["metrics"]
        selected = choices[name]["selection"]["selected"]
        logger.info(
            "Before holdout, %s: selected %s, MAE %s (rMAE %s); reference MAE %s",
            name,
            selected,
            metrics[SELECTED]["mae"],
            metrics[SELECTED]["rmae"],
            {model: values["mae"] for model, values in metrics.items()},
        )
    rules = choices["rules_for_run_5"]
    logger.info("Rules of run 5: %s", rules)
    if rules["1_leak_alarm"]["investigate_leak"]:
        logger.warning("Relative MAE too good to be true: look for a leak before going on")
    if not rules["2_baselines"]["beats_reference"] or not rules["2_baselines"]["beats_ridge"]:
        logger.warning("The selected model does not beat every baseline: %s", rules["2_baselines"])
    for name, rule in rules["3_trees"].items():
        if rule["suggested_n_estimators"] != rule["configured_n_estimators"]:
            logger.warning("%s: the tree curve suggests another n_estimators: %s", name, rule)
    if rules["4_training_window"]["switch_to"] is not None:
        logger.warning("Another training window is better: %s", rules["4_training_window"])
    for name, rule in rules["6_interval_calibration"].items():
        if rule["suggested_calibration_days"] != rule["configured_calibration_days"]:
            logger.warning("%s: another calibration window is closer to 80 %%: %s", name, rule)
    confidence_rule = rules["7_alert_confidence"]
    if not confidence_rule["confidence_is_informative"]:
        logger.warning("Alert precision does not grow with confidence: %s", confidence_rule)
