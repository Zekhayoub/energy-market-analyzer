"""Evaluation of the predictions: metrics, tests, the selection rule and the intervals.

- Development and reserved periods are split by the predicted day: a prediction
  made in September 2025 for October 2025 belongs to the reserved period, so the
  choices never see a price after holdout_start (decision D9).
- MAE in EUR/MWh is the main metric (the unit the procurement team pays in);
  rMAE divides it by the MAE of a naive baseline on the same rows (Lago et al.,
  2021). No MAPE: prices near zero or negative make it explode.
- Diebold-Mariano test on the daily mean of the absolute error differences, with
  a Newey-West (Bartlett) long-run variance.
- Selection rule (decision D27): the simplest candidate that is not worse than
  the best one on the development period, "worse" meaning significantly worse
  and by more than the training noise (two seeds of the global model).
- Intervals (decision D28): split-conformal calibration of the raw quantile
  interval, scaled by its width, refreshed each day with the out-of-sample errors
  known at 11:00 on the decision day (Romano, Patterson and Candès, 2019).

References:
    Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. JBES 13(3).
    Newey, W. K. and West, K. D. (1994). Automatic lag selection in covariance matrix
    estimation. Review of Economic Studies 61(4).
    Romano, Y., Patterson, E. and Candès, E. (2019). Conformalized quantile regression.
    NeurIPS 2019.
    Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction,
    and estimation. JASA 102(477).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from energy_market_analyzer.pipelines.features.nodes import (
    DECISION_DAY,
    HORIZON_DAYS,
    PRICE_SAME_HOUR_TODAY,
)
from energy_market_analyzer.pipelines.prediction.models import (
    ALERT_CONFIDENCE,
    ALERT_LEVEL,
    ALERT_LEVELS,
    ANCHOR,
    CONFIDENCE_CLASSES,
    GLOBAL,
    HYBRID,
    INTERVAL_CALIBRATED,
    INTERVAL_LOWER,
    INTERVAL_LOWER_RAW,
    INTERVAL_UPPER,
    INTERVAL_UPPER_RAW,
    LEVEL_TARGET,
    NO_PRICE,
    NO_REGIME,
    PER_REGIME,
    PROB_ABOVE_CRITICAL,
    PROB_ABOVE_HIGH,
    RANK_IN_DAY,
    REALIZED_LEVEL,
    RIDGE,
    SECOND_SEED,
    SELECTED,
    TARGET_DAY,
    TASK_A,
    TREES_PREFIX,
    Task,
    prediction_column,
)
from energy_market_analyzer.pipelines.regime.nodes import REGIME_FILTERED

# --- Report ---
METRIC_DECIMALS = 3
SHARE_DECIMALS = 4
HOURS_PER_WEEK = 168
# Bins of the reliability table of the alert probabilities (they lie between 0.1 and 0.9).
PROBABILITY_BINS = (0.0, 0.3, 0.5, 0.7, 1.0)
# Rules fixed before run 5 (decisions.md), read on the development period only:
# - without wind and solar forecasts (D3), a relative MAE under 0.6 would be too
#   good: look for a leak before going on;
# - a tree curve still falling, or lowest earlier, by more than 1 % moves n_estimators
#   (x 1.5 in the first case);
# - another calibration window replaces the configured one only if its coverage is
#   closer to the target by more than 2 points;
# - the reserved period better than the development period by more than 0.1 in
#   relative MAE is an alarm (a bug to look for), never a reason to change a setting.
SUSPICIOUS_RMAE = 0.6
TREE_GAIN_LIMIT = 0.01
TREE_INCREASE_FACTOR = 1.5
COVERAGE_TOLERANCE = 0.02
HOLDOUT_GAP_ALARM = 0.1
# Newey-West rule of thumb for the lag of the long-run variance: 4 x (n / 100)^(2/9).
NEWEY_WEST_FACTOR = 4.0
NEWEY_WEST_SCALE = 100.0
NEWEY_WEST_EXPONENT = 2.0 / 9.0
# Hours of the day grouped in blocks of six (night, morning, afternoon, evening).
HOUR_BLOCK = 6
MONTH_FORMAT = "%Y-%m"
DAY_FORMAT = "%Y-%m-%d"


# ==================================================
# Periods and rows
# ==================================================


def holdout_day(params: dict[str, Any]) -> pd.Timestamp:
    """First predicted day of the reserved period, as a calendar date."""
    return pd.Timestamp(params["holdout_start"])


def is_development(table: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    """True for the rows whose predicted day is before holdout_start."""
    return (table[TARGET_DAY] < holdout_day(params)).to_numpy()


def complete_rows(table: pd.DataFrame, task: Task, models: list[str]) -> pd.DataFrame:
    """Rows with a known target and a prediction from every model: the same rows for all."""
    columns = [task.target, *(prediction_column(model) for model in models)]
    is_complete = table[columns].notna().all(axis=1)
    return table[is_complete]


def day_keys(table: pd.DataFrame, task: Task) -> pd.Series:
    """Day of each row for the daily tests: delivered day (A), decision day (B)."""
    return table[task.period_column]


# ==================================================
# Metrics
# ==================================================


def rounded(value: Any, decimals: int = METRIC_DECIMALS) -> float | None:
    """Python float rounded for the JSON report; None if missing or infinite."""
    if value is None:
        return None
    number = float(value)
    return round(number, decimals) if math.isfinite(number) else None


def mae(table: pd.DataFrame, task: Task, model: str) -> float:
    """Mean absolute error of a model on the rows of `table`."""
    errors = table[prediction_column(model)] - table[task.target]
    return float(errors.abs().mean())


def error_metrics(table: pd.DataFrame, task: Task, model: str) -> dict[str, Any]:
    """MAE, RMSE, rMAE (against the task's reference), bias and direction, on `table`'s rows.

    Direction: share of rows where the model says correctly whether the price
    is above or below the known price of J (same hour of J for A, mean of J for B).
    """
    predicted = table[prediction_column(model)]
    actual = table[task.target]
    errors = predicted - actual
    reference_mae = mae(table, task, task.reference)
    known_price = table[PRICE_SAME_HOUR_TODAY] if task.name == TASK_A.name else table[ANCHOR]
    actual_moves = np.sign(actual - known_price)
    has_move = (actual_moves != 0) & known_price.notna()
    same_direction = np.sign(predicted - known_price) == actual_moves
    model_mae = float(errors.abs().mean())
    return {
        "rows": len(table),
        "mae": rounded(model_mae),
        "rmse": rounded(np.sqrt((errors**2).mean())),
        "rmae": rounded(model_mae / reference_mae) if reference_mae > 0 else None,
        "bias": rounded(errors.mean()),
        "direction_accuracy": rounded(same_direction[has_move].mean(), SHARE_DECIMALS),
    }


# ==================================================
# Diebold-Mariano test
# ==================================================


def daily_differential(table: pd.DataFrame, task: Task, first: str, second: str) -> pd.Series:
    """Daily mean of |error of first| - |error of second|, on the rows where both predict.

    Positive = first is worse. One value per day: hours of a day are not
    independent, days are closer to it.
    """
    rows = complete_rows(table, task, [first, second])
    first_errors = (rows[prediction_column(first)] - rows[task.target]).abs()
    second_errors = (rows[prediction_column(second)] - rows[task.target]).abs()
    differences = first_errors - second_errors
    return differences.groupby(day_keys(rows, task)).mean()


def newey_west_lag(days: int, params: dict[str, Any]) -> int:
    """Lag of the long-run variance: Newey-West rule of thumb, at least min_dm_lag_days.

    A longer lag makes the test more prudent, so it favours the simpler model.
    """
    rule_of_thumb = NEWEY_WEST_FACTOR * (days / NEWEY_WEST_SCALE) ** NEWEY_WEST_EXPONENT
    return max(params["min_dm_lag_days"], int(rule_of_thumb))


def long_run_variance(values: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance of daily values (Bartlett weights up to `lag` days).

    Days close in time are correlated: the variance of their mean is larger than
    the plain variance divided by the number of days.
    """
    days = len(values)
    centred = values - values.mean()
    autocovariances = [centred[k:] @ centred[: days - k] / days for k in range(lag + 1)]
    bartlett = [1.0 - k / (lag + 1) for k in range(1, lag + 1)]
    return float(autocovariances[0] + 2 * float(np.dot(bartlett, autocovariances[1:])))


def diebold_mariano(differential: pd.Series, params: dict[str, Any]) -> dict[str, Any]:
    """Diebold-Mariano test that the mean daily loss difference is zero (two-sided).

    Returns:
        Days, mean difference (EUR/MWh, positive = first worse), statistic,
        p-value and lag; statistic and p-value are None with too few days.
    """
    values = differential.to_numpy(dtype=float)
    days = len(values)
    lag = newey_west_lag(days, params)
    result = {"days": days, "mean_difference": rounded(values.mean()) if days else None}
    if days <= lag + 1:
        return {**result, "statistic": None, "p_value": None, "lag_days": lag}
    variance = long_run_variance(values, lag)
    if variance <= 0:
        return {**result, "statistic": None, "p_value": None, "lag_days": lag}
    statistic = values.mean() / math.sqrt(variance / days)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return {
        **result,
        "statistic": rounded(statistic),
        "p_value": rounded(p_value, SHARE_DECIMALS),
        "lag_days": lag,
    }


def compare(
    table: pd.DataFrame, task: Task, first: str, second: str, params: dict[str, Any]
) -> dict[str, Any]:
    """MAE of two models on the rows where both predict, and the Diebold-Mariano test."""
    rows = complete_rows(table, task, [first, second])
    if rows.empty:
        return {"rows": 0}
    test = diebold_mariano(daily_differential(rows, task, first, second), params)
    return {
        "rows": len(rows),
        f"mae_{first}": rounded(mae(rows, task, first)),
        f"mae_{second}": rounded(mae(rows, task, second)),
        "diebold_mariano": test,
    }


def is_significantly_worse(test: dict[str, Any], params: dict[str, Any]) -> bool:
    """True when the first model of a Diebold-Mariano test is worse at the significance level."""
    p_value = test["p_value"]
    mean_difference = test["mean_difference"]
    if p_value is None or mean_difference is None:
        return False
    return mean_difference > 0 and p_value < params["significance"]


# ==================================================
# Selection rule
# ==================================================


def training_noise(table: pd.DataFrame, task: Task) -> float:
    """MAE gap between the global model and the same model refitted with the second seed.

    Diebold-Mariano treats each model as fixed; the subsampling of rows and
    columns makes two fits differ by chance. 0 when no second seed was fitted (B).
    """
    second_column = prediction_column(SECOND_SEED)
    if second_column not in table or table[second_column].isna().all():
        return 0.0
    rows = complete_rows(table, task, [GLOBAL, SECOND_SEED])
    return abs(mae(rows, task, SECOND_SEED) - mae(rows, task, GLOBAL))


def select_variant(table: pd.DataFrame, task: Task, params: dict[str, Any]) -> dict[str, Any]:
    """The simplest candidate not worse than the best beyond chance (decision D27).

    A candidate is worse than the best when the Diebold-Mariano test says so
    and its MAE gap also exceeds the training noise (two seeds of the global
    model differ by that much by chance alone). Candidates are ordered from
    the simplest. Applied to the rows given, which the caller restricts to the
    development period.

    Args:
        table: Prediction rows (development period only).
        task: TASK_A or TASK_B.
        params: Prediction parameters (significance, min_dm_lag_days).

    Returns:
        MAE of each candidate, the best one, the test of each against it, the
        training noise and the selected candidate.

    Raises:
        ValueError: If no row has a prediction from every candidate.
    """
    candidates = list(task.candidates)
    rows = complete_rows(table, task, [*candidates, task.reference])
    if rows.empty:
        raise ValueError(f"{task.name}: no development row to select a variant on")
    errors = {name: mae(rows, task, name) for name in candidates}
    best = min(candidates, key=errors.get)
    noise = training_noise(table, task)
    tests = {
        name: diebold_mariano(daily_differential(rows, task, name, best), params)
        for name in candidates
        if name != best
    }
    worse = [
        name
        for name, test in tests.items()
        if is_significantly_worse(test, params) and errors[name] - errors[best] > noise
    ]
    selected = next(name for name in candidates if name not in worse)
    return {
        "rows": len(rows),
        "mae": {name: rounded(value) for name, value in errors.items()},
        "best": best,
        "tests_against_best": tests,
        "significance": params["significance"],
        "training_noise_mae": rounded(noise),
        "worse_than_best": worse,
        "selected": selected,
    }


# ==================================================
# Conformal calibration of the intervals
# ==================================================


def day_numbers(days: pd.Series) -> np.ndarray:
    """Calendar days as integers (days since 1970-01-01)."""
    return days.to_numpy(dtype="datetime64[D]").astype(np.int64)


def conformity_scores(
    lower: np.ndarray, upper: np.ndarray, actual: np.ndarray, min_width: float
) -> np.ndarray:
    """Distance of the actual price outside its interval, in interval widths.

    Negative inside the interval, positive outside (scaled version of the
    conformalized quantile regression score).
    """
    width = np.maximum(upper - lower, min_width)
    distance = np.maximum(lower - actual, actual - upper)
    return distance / width


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """The ceil((n + 1) x coverage)-th smallest score: the split-conformal offset.

    With fewer scores than that rank needs, the largest score (never infinite).
    """
    count = len(scores)
    rank = math.ceil((count + 1) * coverage)
    ordered = np.sort(scores)
    return float(ordered[min(rank, count) - 1])


def daily_offsets(
    score_days: np.ndarray,
    scores: np.ndarray,
    decision_days: np.ndarray,
    window_days: int,
    params: dict[str, Any],
) -> dict[int, float]:
    """Calibration offset of each decision day J from the scores of the days (J - window, J].

    The score of a predicted day d is known at 11:00 on J when d <= J: its price
    was published on d - 1. No offset (NaN) before min_calibration_days days of scores.
    """
    interval = params["interval"]
    coverage = interval["upper_quantile"] - interval["lower_quantile"]
    order = np.argsort(score_days, kind="stable")
    sorted_days = score_days[order]
    sorted_scores = scores[order]
    first_day = sorted_days[0] if len(sorted_days) else None
    offsets = {}
    for day in np.unique(decision_days):
        start = np.searchsorted(sorted_days, day - window_days, side="right")
        end = np.searchsorted(sorted_days, day, side="right")
        has_history = (
            first_day is not None and day - first_day + 1 >= interval["min_calibration_days"]
        )
        is_usable = has_history and end > start
        window = sorted_scores[start:end]
        offsets[int(day)] = conformal_quantile(window, coverage) if is_usable else np.nan
    return offsets


def _calibrate_group(
    table: pd.DataFrame, task: Task, window_days: int, params: dict[str, Any]
) -> pd.DataFrame:
    """Calibrated interval of the rows of one group (all rows for A, one horizon for B)."""
    interval = params["interval"]
    lower = table[INTERVAL_LOWER_RAW].to_numpy(dtype=float)
    upper = table[INTERVAL_UPPER_RAW].to_numpy(dtype=float)
    actual = table[task.target].to_numpy(dtype=float)
    scores = conformity_scores(lower, upper, actual, interval["min_raw_width_eur_mwh"])
    has_score = ~np.isnan(scores)
    target_days = day_numbers(table[TARGET_DAY])
    decision_days = day_numbers(table[DECISION_DAY])
    offsets = daily_offsets(
        target_days[has_score], scores[has_score], decision_days, window_days, params
    )
    row_offsets = np.array([offsets[day] for day in decision_days.tolist()], dtype=float)
    # The interval keeps at least min_width_share of its raw width: (1 + 2 x offset) >= share.
    floor = -(1.0 - interval["min_width_share"]) / 2.0
    is_calibrated = ~np.isnan(row_offsets)
    is_guarded = is_calibrated & (row_offsets < floor)
    used_offsets = np.where(is_calibrated, np.maximum(row_offsets, floor), 0.0)
    width = np.maximum(upper - lower, interval["min_raw_width_eur_mwh"])
    columns = {
        INTERVAL_LOWER: lower - used_offsets * width,
        INTERVAL_UPPER: upper + used_offsets * width,
        INTERVAL_CALIBRATED: is_calibrated,
        "guarded": is_guarded,
    }
    return pd.DataFrame(columns, index=table.index)


def calibrate_intervals(
    table: pd.DataFrame, task: Task, window_days: int, params: dict[str, Any]
) -> pd.DataFrame:
    """Calibrated interval of every row: each day, the offset that would have covered
    the configured share of the out-of-sample errors of the last window_days days.

    Model B is calibrated horizon by horizon: errors grow with the horizon.

    Returns:
        INTERVAL_LOWER, INTERVAL_UPPER, INTERVAL_CALIBRATED and "guarded" (the
        floor on the width was used), same index as `table`.
    """
    if task.name == TASK_A.name:
        return _calibrate_group(table, task, window_days, params)
    groups = [
        _calibrate_group(group, task, window_days, params)
        for _, group in table.groupby(HORIZON_DAYS)
    ]
    return pd.concat(groups).loc[table.index]


def interval_metrics(
    table: pd.DataFrame, task: Task, lower_column: str, upper_column: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Coverage, mean width, pinball losses of both bounds and interval (Winkler) score.

    The interval score adds to the width 2 / alpha times the distance of each
    actual price outside the interval: a wide interval covers but is penalized.
    """
    interval = params["interval"]
    lower_quantile = interval["lower_quantile"]
    upper_quantile = interval["upper_quantile"]
    alpha = 1.0 - (upper_quantile - lower_quantile)
    rows = table[table[[task.target, lower_column, upper_column]].notna().all(axis=1)]
    if rows.empty:
        return {"rows": 0}
    actual = rows[task.target].to_numpy(dtype=float)
    lower = rows[lower_column].to_numpy(dtype=float)
    upper = rows[upper_column].to_numpy(dtype=float)
    below = np.maximum(lower - actual, 0.0)
    above = np.maximum(actual - upper, 0.0)
    return {
        "rows": len(rows),
        "coverage": rounded(((actual >= lower) & (actual <= upper)).mean(), SHARE_DECIMALS),
        "mean_width": rounded((upper - lower).mean()),
        "pinball_lower": rounded(_pinball(actual, lower, lower_quantile)),
        "pinball_upper": rounded(_pinball(actual, upper, upper_quantile)),
        "interval_score": rounded((upper - lower + 2.0 / alpha * (below + above)).mean()),
    }


def _pinball(actual: np.ndarray, quantile_values: np.ndarray, level: float) -> float:
    """Mean pinball (quantile) loss of predicted quantiles at `level`."""
    gaps = actual - quantile_values
    return float(np.maximum(level * gaps, (level - 1.0) * gaps).mean())


# ==================================================
# Breakdowns
# ==================================================


def group_keys(table: pd.DataFrame, task: Task, by: str, params: dict[str, Any]) -> pd.Series:
    """Key of each row: year or month of the predicted day, regime of J, hour block or horizon."""
    if by == "year":
        return table[TARGET_DAY].dt.year.astype(str)
    if by == "month":
        return table[TARGET_DAY].dt.strftime(MONTH_FORMAT)
    if by == "regime":
        names = dict(enumerate(params["regime_names"]))
        labels = table[REGIME_FILTERED].astype(float)
        return labels.map(names).fillna("unknown")
    if by == "hour_block":
        hours = table.index.tz_convert(params["market_timezone"]).hour
        starts = hours // HOUR_BLOCK * HOUR_BLOCK
        names = [f"{start:02d}-{start + HOUR_BLOCK - 1:02d}" for start in starts]
        return pd.Series(names, index=table.index)
    if by == "hour":
        hours = table.index.tz_convert(params["market_timezone"]).hour
        return pd.Series(hours.astype(str), index=table.index)
    if by == "horizon":
        return table[HORIZON_DAYS].astype(str)
    raise ValueError(f"Unknown breakdown '{by}'")


def mae_by(
    table: pd.DataFrame, task: Task, models: list[str], by: str, params: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """MAE and rMAE of each model in each group of rows (rows already common to the models)."""
    keys = group_keys(table, task, by, params)
    summary = {}
    for key, rows in table.groupby(keys, sort=True):
        reference_mae = mae(rows, task, task.reference)
        summary[str(key)] = {
            model: {
                "mae": rounded(mae(rows, task, model)),
                "rmae": rounded(mae(rows, task, model) / reference_mae) if reference_mae else None,
            }
            for model in models
        }
        summary[str(key)]["rows"] = len(rows)
    return summary


def residuals(
    table: pd.DataFrame, task: Task, params: dict[str, Any], worst_days: int
) -> dict[str, Any]:
    """Bias of the selected model by group, autocorrelation of its daily error, worst days.

    A daily error that persists from one day to the next means information is missing.
    """
    errors = table[prediction_column(SELECTED)] - table[task.target]
    breakdown = "hour" if task.name == TASK_A.name else "horizon"
    bias = {
        by: errors.groupby(group_keys(table, task, by, params)).mean().map(rounded).to_dict()
        for by in (breakdown, "regime", "year")
    }
    daily_error = errors.groupby(day_keys(table, task)).mean()
    daily_mae = errors.abs().groupby(day_keys(table, task)).mean()
    reference_mae = (
        ((table[prediction_column(task.reference)] - table[task.target]).abs())
        .groupby(day_keys(table, task))
        .mean()
    )
    mean_price = table[task.target].groupby(day_keys(table, task)).mean()
    worst = daily_mae.sort_values(ascending=False).head(worst_days)
    return {
        "bias": bias,
        "daily_error_autocorrelation": {
            "lag_1_day": rounded(daily_error.autocorr(1)),
            "lag_7_days": rounded(daily_error.autocorr(7)),
        },
        "worst_days": [
            {
                "day": day.strftime(DAY_FORMAT),
                "mae_selected": rounded(value),
                f"mae_{task.reference}": rounded(reference_mae[day]),
                "mean_price": rounded(mean_price[day]),
            }
            for day, value in worst.items()
        ],
    }


def development_epochs(table: pd.DataFrame, params: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Rows of each development epoch, cut at the development_epochs dates."""
    boundaries = [pd.Timestamp(day) for day in params["development_epochs"]]
    starts = [table[TARGET_DAY].min(), *boundaries]
    ends = [*boundaries, holdout_day(params)]
    epochs = {}
    for start, end in zip(starts, ends, strict=True):
        is_epoch = (table[TARGET_DAY] >= start) & (table[TARGET_DAY] < end)
        label = (
            f"{start.strftime(DAY_FORMAT)}_to_{(end - pd.Timedelta(days=1)).strftime(DAY_FORMAT)}"
        )
        epochs[label] = table[is_epoch]
    return epochs


def common_models(task: Task) -> list[str]:
    """Models evaluated together on the same rows: baselines, candidates, ablations, selected."""
    return [*task.baselines, *task.candidates, LEVEL_TARGET, NO_PRICE, SELECTED]


# ==================================================
# Report sections of one task and one period
# ==================================================


def _comparisons(table: pd.DataFrame, task: Task, params: dict[str, Any]) -> dict[str, Any]:
    """Pairwise tests, each on the rows where both models predict.

    Ablations: anchored target (level_target vs global), price features
    (no_price vs level_target, same target), regime columns (no_regime vs
    global), training noise (second seed) and training windows (development only).
    """
    pairs = [
        (SELECTED, task.reference),
        (RIDGE, SELECTED),
        (LEVEL_TARGET, GLOBAL),
        (NO_PRICE, LEVEL_TARGET),
        (NO_REGIME, GLOBAL),
        (SECOND_SEED, GLOBAL),
        *((name, GLOBAL) for name in _window_models(table)),
    ]
    present = [
        (first, second)
        for first, second in pairs
        if prediction_column(first) in table and table[prediction_column(first)].notna().any()
    ]
    return {
        f"{first}_vs_{second}": compare(table, task, first, second, params)
        for first, second in present
    }


def _window_models(table: pd.DataFrame) -> list[str]:
    """Global models refitted on another training window, present in `table`."""
    prefix = prediction_column(f"{GLOBAL}_")
    return [
        column.removeprefix(prediction_column(""))
        for column in table.columns
        if column.startswith(prefix) and column != prediction_column(SECOND_SEED)
    ]


def _interval_summary(table: pd.DataFrame, task: Task, params: dict[str, Any]) -> dict[str, Any]:
    """Raw and calibrated intervals on the same (calibrated) rows, by group; point outside."""
    calibrated = table[table[INTERVAL_CALIBRATED].astype(bool)]
    breakdown = "hour_block" if task.name == TASK_A.name else "horizon"
    by_group = {}
    for by in ("year", "regime", breakdown):
        keys = group_keys(calibrated, task, by, params)
        by_group[by] = {
            str(key): interval_metrics(rows, task, INTERVAL_LOWER, INTERVAL_UPPER, params)
            for key, rows in calibrated.groupby(keys, sort=True)
        }
    point = calibrated[prediction_column(SELECTED)]
    is_outside = (point < calibrated[INTERVAL_LOWER]) | (point > calibrated[INTERVAL_UPPER])
    return {
        "rows_not_calibrated": int((~table[INTERVAL_CALIBRATED].astype(bool)).sum()),
        "raw": interval_metrics(calibrated, task, INTERVAL_LOWER_RAW, INTERVAL_UPPER_RAW, params),
        "calibrated": interval_metrics(calibrated, task, INTERVAL_LOWER, INTERVAL_UPPER, params),
        "calibrated_by": by_group,
        "selected_outside_interval": rounded(is_outside.mean(), SHARE_DECIMALS),
    }


def period_summary(table: pd.DataFrame, task: Task, params: dict[str, Any]) -> dict[str, Any]:
    """Everything the report says about one task on one period.

    The main metrics compare every common model on the same rows; other models
    (ridge, ablations computed on part of the rows) are compared pairwise.
    """
    models = common_models(task)
    rows = complete_rows(table, task, models)
    breakdown = "hour_block" if task.name == TASK_A.name else "horizon"
    main = [SELECTED, task.reference]
    return {
        "rows": len(table),
        "evaluated_rows": len(rows),
        "metrics": {model: error_metrics(rows, task, model) for model in models},
        "by_year": mae_by(rows, task, models, "year", params),
        "by_regime": mae_by(rows, task, models, "regime", params),
        f"by_{breakdown}": mae_by(rows, task, main, breakdown, params),
        "monthly_mae": mae_by(rows, task, main, "month", params),
        "comparisons": _comparisons(table, task, params),
        "residuals": residuals(rows, task, params, params["worst_days"]),
        "intervals": _interval_summary(table, task, params),
    }


def epoch_summary(table: pd.DataFrame, task: Task, params: dict[str, Any]) -> dict[str, Any]:
    """Per development epoch: the variant the rule would select, and what the regime adds.

    Does the variant selected on all development months hold on 2023-2025, the
    epoch closest to the reserved period? Do the regime columns help before and
    after mid-2021 (notes of commit 3)?
    """
    regime_pairs = [(NO_REGIME, GLOBAL), *((name, GLOBAL) for name in (PER_REGIME, HYBRID))]
    summary = {}
    for label, rows in development_epochs(table, params).items():
        if complete_rows(rows, task, [*task.candidates, task.reference]).empty:
            summary[label] = {"rows": 0}
            continue
        selection = select_variant(rows, task, params)
        summary[label] = {
            "selected": selection["selected"],
            "best": selection["best"],
            "mae": selection["mae"],
            "regime_columns": {
                f"{first}_vs_{second}": compare(rows, task, first, second, params)
                for first, second in regime_pairs
                if first in task.candidates
            },
        }
    return summary


# ==================================================
# Alerts
# ==================================================


def _hours_per_week(rows: int) -> float:
    """Number of weeks covered by `rows` hours."""
    return rows / HOURS_PER_WEEK


def _realized_at_least(levels: pd.Series, level: str) -> pd.Series:
    """True where the realized level is at least `level` (normal < high < critical)."""
    rank = {name: position for position, name in enumerate(ALERT_LEVELS)}
    return levels.map(rank) >= rank[level]


def _precision_recall(alerts: pd.DataFrame) -> dict[str, Any]:
    """Precision and recall of the high (or critical) and critical alerts."""
    summary = {}
    for level in ALERT_LEVELS[1:]:
        is_alert = _realized_at_least(alerts[ALERT_LEVEL], level)
        is_realized = _realized_at_least(alerts[REALIZED_LEVEL], level)
        hits = int((is_alert & is_realized).sum())
        summary[f"at_least_{level}"] = {
            "alerts_per_week": rounded(is_alert.sum() / _hours_per_week(len(alerts))),
            "realized_per_week": rounded(is_realized.sum() / _hours_per_week(len(alerts))),
            "precision": rounded(hits / is_alert.sum(), SHARE_DECIMALS) if is_alert.any() else None,
            "recall": rounded(hits / is_realized.sum(), SHARE_DECIMALS)
            if is_realized.any()
            else None,
        }
    return summary


def _precision_by_confidence(alerts: pd.DataFrame) -> dict[str, Any]:
    """For high and critical alerts: share realized at least at the alert's level, by confidence.

    Rule 7 of run 5: this share must not decrease from low to high confidence.
    """
    is_alert = alerts[ALERT_LEVEL] != ALERT_LEVELS[0]
    raised = alerts[is_alert]
    rank = {name: position for position, name in enumerate(ALERT_LEVELS)}
    is_hit = raised[REALIZED_LEVEL].map(rank) >= raised[ALERT_LEVEL].map(rank)
    summary = {}
    for confidence in CONFIDENCE_CLASSES:
        is_class = raised[ALERT_CONFIDENCE] == confidence
        summary[confidence] = {
            "alerts": int(is_class.sum()),
            "precision": rounded(is_hit[is_class].mean(), SHARE_DECIMALS)
            if is_class.any()
            else None,
        }
    return summary


def _brier(alerts: pd.DataFrame, probability: str, level: str, nominal_rate: float) -> dict:
    """Brier score of a probability and its skill against the nominal rate of the threshold.

    The nominal rate (e.g. 5 % above the 95 % quantile) is known before the
    period: skill > 0 means the probability is more useful than that constant.
    """
    outcome = _realized_at_least(alerts[REALIZED_LEVEL], level).astype(float)
    forecast = alerts[probability]
    brier = float(((forecast - outcome) ** 2).mean())
    constant = float(((nominal_rate - outcome) ** 2).mean())
    bins = pd.cut(forecast, PROBABILITY_BINS, include_lowest=True)
    reliability = {
        str(interval): {
            "rows": len(group),
            "mean_probability": rounded(group.mean(), SHARE_DECIMALS),
            "realized_share": rounded(outcome[group.index].mean(), SHARE_DECIMALS),
        }
        for interval, group in forecast.groupby(bins, observed=True)
    }
    return {
        "brier_score": rounded(brier, SHARE_DECIMALS),
        "brier_skill_vs_nominal_rate": rounded(1.0 - brier / constant, SHARE_DECIMALS)
        if constant > 0
        else None,
        "reliability": reliability,
    }


def _ranking(alerts: pd.DataFrame, predictions: pd.DataFrame, params: dict[str, Any]) -> dict:
    """Hit rates of the cheapest and most expensive predicted hours, against the naive baseline.

    On the days with a rank for every hour and a known price for every hour.
    """
    alert_settings = params["alerts"]
    days = predictions[TARGET_DAY]
    actual = predictions[TASK_A.target]
    naive = predictions[prediction_column(TASK_A.reference)]
    is_complete_day = actual.notna().groupby(days).transform("all") & naive.notna().groupby(
        days
    ).transform("all")
    has_rank = alerts[RANK_IN_DAY].notna().groupby(days).transform("all")
    rows = alerts.index[(is_complete_day & has_rank).to_numpy()]
    if rows.empty:
        return {"days": 0}
    day_of_rows = days[rows]
    realized_rank = actual[rows].groupby(day_of_rows).rank(method="first")
    naive_rank = naive[rows].groupby(day_of_rows).rank(method="first")
    model_rank = alerts.loc[rows, RANK_IN_DAY].astype(float)
    hours = realized_rank.groupby(day_of_rows).transform("size")
    summary = {"days": int(day_of_rows.nunique())}
    for name, ranks in (("selected", model_rank), (TASK_A.reference, naive_rank)):
        summary[name] = {
            "cheapest_hit_rate": _hit_rate(
                ranks, realized_rank, day_of_rows, hours, alert_settings, cheapest=True
            ),
            "most_expensive_hit_rate": _hit_rate(
                ranks, realized_rank, day_of_rows, hours, alert_settings, cheapest=False
            ),
            "mean_rank_correlation": rounded(
                _rank_correlation(ranks, realized_rank, day_of_rows), SHARE_DECIMALS
            ),
        }
    return summary


def _hit_rate(
    ranks: pd.Series,
    realized_rank: pd.Series,
    days: pd.Series,
    hours: pd.Series,
    alert_settings: dict[str, Any],
    cheapest: bool,
) -> float | None:
    """Share of the predicted cheapest (or most expensive) hours that really were, per day."""
    if cheapest:
        count = alert_settings["cheapest_hours"]
        is_predicted = ranks <= count
        is_realized = realized_rank <= count
    else:
        count = alert_settings["most_expensive_hours"]
        is_predicted = ranks > hours - count
        is_realized = realized_rank > hours - count
    hits = (is_predicted & is_realized).groupby(days).sum()
    return rounded((hits / count).mean(), SHARE_DECIMALS)


def _rank_correlation(ranks: pd.Series, realized_rank: pd.Series, days: pd.Series) -> float:
    """Mean over days of the Spearman correlation between predicted and realized hour ranks."""
    frame = pd.DataFrame({"predicted": ranks, "realized": realized_rank, "day": days})
    correlations = frame.groupby("day").apply(
        lambda day: day["predicted"].corr(day["realized"]), include_groups=False
    )
    return float(correlations.mean())


def alerts_summary(
    alerts: pd.DataFrame, predictions: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Alerts of one period: rates, precision, confidence, probabilities and daily ranks."""
    alert_settings = params["alerts"]
    known = alerts[alerts[[ALERT_LEVEL, REALIZED_LEVEL]].notna().all(axis=1)]
    if known.empty:
        return {"hours": 0}
    years = predictions.loc[known.index, TARGET_DAY].dt.year.astype(str)
    base_rates = {
        year: {
            level: rounded(_realized_at_least(rows[REALIZED_LEVEL], level).mean(), SHARE_DECIMALS)
            for level in ALERT_LEVELS[1:]
        }
        for year, rows in known.groupby(years)
    }
    return {
        "hours": len(known),
        "realized_share_above_thresholds_by_year": base_rates,
        "levels": _precision_recall(known),
        "precision_by_confidence": _precision_by_confidence(known),
        "probability_above_high": _brier(
            known, PROB_ABOVE_HIGH, ALERT_LEVELS[1], 1.0 - alert_settings["high_quantile"]
        ),
        "probability_above_critical": _brier(
            known, PROB_ABOVE_CRITICAL, ALERT_LEVELS[2], 1.0 - alert_settings["critical_quantile"]
        ),
        "daily_ranks": _ranking(alerts.loc[predictions.index], predictions, params),
    }


# ==================================================
# Rules of run 5, fixed before the run (development period only)
# ==================================================


def tree_curve(table: pd.DataFrame, task: Task) -> dict[str, float | None]:
    """MAE of the global model with its first n trees, on the rows where every point predicts."""
    columns = [column for column in table.columns if column.startswith(TREES_PREFIX)]
    is_complete = table[[task.target, *columns]].notna().all(axis=1)
    rows = table[is_complete]
    return {
        column.removeprefix(TREES_PREFIX): rounded((rows[column] - rows[task.target]).abs().mean())
        for column in columns
    }


def _tree_rule(curve: dict[str, float | None], settings: dict[str, Any]) -> dict[str, Any]:
    """Rule 3: keep n_estimators unless the curve still falls, or bottoms out earlier, by > 1 %."""
    errors = {int(trees): value for trees, value in curve.items() if value is not None}
    configured = settings["xgboost"]["n_estimators"]
    step = settings["tree_checkpoint_step"]
    if configured not in errors or configured - step not in errors:
        return {
            "configured_n_estimators": configured,
            "suggested_n_estimators": configured,
            "reason": "curve too short",
        }
    last_gain = 1.0 - errors[configured] / errors[configured - step]
    best_trees = min(errors, key=errors.get)
    best_gain = 1.0 - errors[best_trees] / errors[configured]
    suggested = configured
    if last_gain > TREE_GAIN_LIMIT:
        suggested = int(configured * TREE_INCREASE_FACTOR)
    elif best_gain > TREE_GAIN_LIMIT:
        suggested = best_trees
    return {
        "configured_n_estimators": configured,
        "gain_of_last_step": rounded(last_gain, SHARE_DECIMALS),
        "best_trees": best_trees,
        "gain_of_best_trees": rounded(best_gain, SHARE_DECIMALS),
        "limit": TREE_GAIN_LIMIT,
        "suggested_n_estimators": suggested,
    }


def _window_rule(comparisons: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Rule 4: switch every model to another training window if it beats the configured one.

    "Beats": lower MAE and the configured global model significantly worse.
    """
    switch_to = None
    tests = {}
    for name, comparison in comparisons.items():
        is_window = name.startswith(f"{GLOBAL}_") and name.endswith(f"_vs_{GLOBAL}")
        if not is_window or name.startswith(SECOND_SEED) or comparison["rows"] == 0:
            continue
        test = comparison["diebold_mariano"]
        tests[name] = test
        mean_difference = test["mean_difference"]
        p_value = test["p_value"]
        is_better = mean_difference is not None and mean_difference < 0
        is_significant = p_value is not None and p_value < params["significance"]
        if is_better and is_significant:
            window_model_name = name.removesuffix(f"_vs_{GLOBAL}")
            switch_to = window_model_name.removeprefix(f"{GLOBAL}_")
    return {
        "configured": params["training_window_months"],
        "tests": tests,
        "switch_to": switch_to,
    }


def _calibration_rule(candidates: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Rule 6: keep calibration_days unless another window is closer to the target coverage
    by more than COVERAGE_TOLERANCE."""
    interval = params["interval"]
    target = interval["upper_quantile"] - interval["lower_quantile"]
    gaps = {
        days: abs(metrics["coverage"] - target)
        for days, metrics in candidates.items()
        if metrics.get("coverage") is not None
    }
    configured = str(interval["calibration_days"])
    closest = min(gaps, key=gaps.get) if gaps else configured
    keep = configured not in gaps or gaps[configured] - gaps[closest] <= COVERAGE_TOLERANCE
    return {
        "configured_calibration_days": int(configured),
        "target_coverage": rounded(target, SHARE_DECIMALS),
        "coverage_gap_by_days": {days: rounded(gap, SHARE_DECIMALS) for days, gap in gaps.items()},
        "suggested_calibration_days": int(configured if keep else closest),
    }


def _confidence_rule(alerts: dict[str, Any]) -> dict[str, Any]:
    """Rule 7: alert precision must not decrease from low to high confidence."""
    by_confidence = alerts.get("precision_by_confidence", {})
    precisions = [
        by_confidence[name]["precision"]
        for name in CONFIDENCE_CLASSES
        if by_confidence.get(name, {}).get("precision") is not None
    ]
    is_ordered = all(
        later >= earlier for earlier, later in zip(precisions, precisions[1:], strict=False)
    )
    return {"precision_by_confidence": precisions, "confidence_is_informative": is_ordered}


def rules_for_run_5(
    development: dict[str, Any], choices: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """The rules fixed before run 5 (decisions.md), read on the development period only.

    Args:
        development: Development summaries of model_a, model_b and alerts.
        choices: Output of `finalize_predictions` (selection, tree curves, calibration).
        params: Prediction parameters.

    Returns:
        One entry per rule; each says what to do, nothing is changed automatically
        except the selected variant (rule 5).
    """
    model_a = development["model_a"]
    selected_rmae = model_a["metrics"][SELECTED]["rmae"]
    comparisons = model_a["comparisons"]
    reference_test = comparisons[f"{SELECTED}_vs_{TASK_A.reference}"]["diebold_mariano"]
    ridge_test = comparisons.get(f"{RIDGE}_vs_{SELECTED}", {}).get("diebold_mariano")
    return {
        "1_leak_alarm": {
            "selected_rmae": selected_rmae,
            "limit": SUSPICIOUS_RMAE,
            "investigate_leak": selected_rmae is not None and selected_rmae < SUSPICIOUS_RMAE,
        },
        "2_baselines": {
            "beats_reference": selected_rmae is not None
            and selected_rmae < 1.0
            and _is_better(reference_test, params),
            "beats_ridge": ridge_test is not None and is_significantly_worse(ridge_test, params),
        },
        "3_trees": {
            name: _tree_rule(choices[name]["tree_curve"], params[name])
            for name in ("model_a", "model_b")
        },
        "4_training_window": _window_rule(comparisons, params),
        "5_selected_variants": {
            name: choices[name]["selection"]["selected"] for name in ("model_a", "model_b")
        },
        "6_interval_calibration": {
            name: _calibration_rule(choices[name]["calibration_candidates"], params)
            for name in ("model_a", "model_b")
        },
        "7_alert_confidence": _confidence_rule(development["alerts"]),
    }


def _is_better(test: dict[str, Any], params: dict[str, Any]) -> bool:
    """True when the first model of the test is significantly better than the second."""
    p_value = test["p_value"]
    mean_difference = test["mean_difference"]
    if p_value is None or mean_difference is None:
        return False
    return mean_difference < 0 and p_value < params["significance"]


def holdout_alarms(development: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    """Alarm (a reason to look for a bug, never to change a setting): the reserved period
    much better than the development period, in relative MAE."""
    development_rmae = development["model_a"]["metrics"][SELECTED]["rmae"]
    holdout_rmae = holdout["model_a"]["metrics"][SELECTED]["rmae"]
    is_suspicious = (
        development_rmae is not None
        and holdout_rmae is not None
        and holdout_rmae < development_rmae - HOLDOUT_GAP_ALARM
    )
    return {
        "development_rmae": development_rmae,
        "holdout_rmae": holdout_rmae,
        "limit": HOLDOUT_GAP_ALARM,
        "holdout_much_better": is_suspicious,
    }
