"""Prediction models: baselines and one monthly refit of every model (decisions D25 to D27).

Each month, every model is refitted on the rows whose predicted day is before the
first day of the month (Brussels), then predicts the rows of the month:

- models A (hourly price of J+1) and B (daily mean price of J+2 to J+4), both
  decided at 11:00 on J;
- XGBoost with the absolute error loss: the prediction is a median, robust to
  price spikes and consistent with the MAE;
- anchored target: the models learn the gap to the mean price of J (known at
  11:00 J) and add it back, because a tree never predicts outside the values it
  was trained on and price levels move by a factor 4 within months;
- fixed hyperparameters, no early stopping: no validation set is chosen, so no
  stopping can look at the tested month;
- intervals from LightGBM quantile models, calibrated afterwards (evaluation.py).

Nothing here reads an output of the anomaly pipeline (decision D24).

References:
    Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. KDD 2016.
    Lago, J., Marcjasz, G., De Schutter, B. and Weron, R. (2021). Forecasting day-ahead
    electricity prices: a review of state-of-the-art algorithms, best practices and an
    open-access benchmark. Applied Energy 293.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from energy_market_analyzer.pipelines.features.nodes import (
    CLEAN_SPARK_SPREAD_TODAY,
    DAY_OF_WEEK,
    DECISION_DAY,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    HORIZON_DAYS,
    HOUR,
    MONTH,
    PRICE_MEAN,
    PRICE_MEAN_CHANGE_TODAY,
    PRICE_MEAN_WEEK_AGO,
    PRICE_SAME_HOUR_TODAY,
    PRICE_SAME_HOUR_WEEK_AGO,
    TARGET_MEAN_PRICE,
    TARGET_PRICE,
    TODAY_PRICES,
)
from energy_market_analyzer.pipelines.regime.nodes import FILTERED, REGIME_FILTERED, regime_columns

logger = logging.getLogger(__name__)

# --- Columns of the inputs and prediction tables ---
TARGET_DAY = "target_day"  # predicted day: the delivered day (A), decision_day + horizon (B)
ANCHOR = TODAY_PRICES[PRICE_MEAN]  # mean price of J, known at 11:00 J
TRAINING_END = "training_end"  # last predicted day of the training rows
INTERVAL_LOWER_RAW = "interval_lower_raw"
INTERVAL_UPPER_RAW = "interval_upper_raw"
INTERVAL_LOWER = "interval_lower"
INTERVAL_UPPER = "interval_upper"
INTERVAL_CALIBRATED = "interval_calibrated"
TREES_PREFIX = "trees_"  # predictions of the global model with its first n trees (in memory)

# --- Alerts table (one row per predicted hour of model A) ---
PRED_PRICE = "pred_price"
THRESHOLD_REGIME = "threshold_regime"
THRESHOLD_HIGH = "threshold_high"
THRESHOLD_CRITICAL = "threshold_critical"
PROB_ABOVE_HIGH = "prob_above_high"
PROB_ABOVE_CRITICAL = "prob_above_critical"
ALERT_LEVEL = "alert_level"
ALERT_CONFIDENCE = "alert_confidence"
RANK_IN_DAY = "rank_in_day"
IS_CHEAPEST_HOUR = "is_cheapest_hour"
IS_MOST_EXPENSIVE_HOUR = "is_most_expensive_hour"
REALIZED_LEVEL = "realized_level"
ALERT_LEVELS = ("normal", "high", "critical")
CONFIDENCE_CLASSES = ("low", "medium", "high")

# --- Model names ---
NAIVE_SIMILAR_DAY = "naive_similar_day"
NAIVE_WEEK_AGO = "naive_week_ago"
MEAN_TODAY = "mean_today"
RIDGE = "ridge"
NO_REGIME = "no_regime"
GLOBAL = "global"
WEIGHTED = "weighted"
PER_REGIME = "per_regime"
HYBRID = "hybrid"
AVERAGE = "average"
LEVEL_TARGET = "level_target"
NO_PRICE = "no_price"
SECOND_SEED = "global_second_seed"
SELECTED = "selected"

# --- Prices of the feature tables (the "fundamentals only" ablation drops them) ---
PRICE_FEATURES_A = (
    PRICE_SAME_HOUR_TODAY,
    PRICE_SAME_HOUR_WEEK_AGO,
    *TODAY_PRICES.values(),
    PRICE_MEAN_CHANGE_TODAY,
    CLEAN_SPARK_SPREAD_TODAY,
)
PRICE_FEATURES_B = (
    PRICE_MEAN_WEEK_AGO,
    *TODAY_PRICES.values(),
    PRICE_MEAN_CHANGE_TODAY,
    CLEAN_SPARK_SPREAD_TODAY,
)

# --- Fitting ---
XGBOOST_OBJECTIVE = "reg:absoluteerror"
XGBOOST_TREE_METHOD = "hist"
LIGHTGBM_OBJECTIVE = "quantile"
LIGHTGBM_BAGGING_FREQUENCY = 1  # draw a new subsample of rows for every tree
LIGHTGBM_SILENT = -1
# Delivered days for which the similar-day naive takes the week before (Lago et al.,
# 2021): Monday, Saturday and Sunday (pandas dayofweek, Monday = 0).
WEEK_AGO_WEEKDAYS = (0, 5, 6)
ROUND_SECONDS = 1


@dataclass(frozen=True)
class Task:
    """What changes between model A and model B.

    Attributes:
        name: Report key, "model_a" or "model_b"; also the parameter section.
        target: Target column.
        features: Features of the models without regime.
        price_features: Features dropped by the "fundamentals only" ablation.
        one_hot: Calendar columns the ridge sees as categories.
        period_column: Day that sets the tested month (delivered day, decision day).
        candidates: Variants that can be selected, from the simplest.
        reference: Baseline of the relative MAE.
        baselines: Baselines that need no fitting.
        regime_experts: Whether the weighted, per-regime, hybrid and average variants exist.
        noise_checks: Whether a second seed and the other training windows are fitted.
    """

    name: str
    target: str
    features: tuple[str, ...]
    price_features: tuple[str, ...]
    one_hot: tuple[str, ...]
    period_column: str
    candidates: tuple[str, ...]
    reference: str
    baselines: tuple[str, ...]
    regime_experts: bool
    noise_checks: bool


TASK_A = Task(
    name="model_a",
    target=TARGET_PRICE,
    features=FEATURES_MODEL_A,
    price_features=PRICE_FEATURES_A,
    one_hot=(HOUR, DAY_OF_WEEK, MONTH),
    period_column=TARGET_DAY,
    candidates=(NO_REGIME, GLOBAL, WEIGHTED, PER_REGIME, HYBRID, AVERAGE),
    reference=NAIVE_SIMILAR_DAY,
    baselines=(NAIVE_SIMILAR_DAY, NAIVE_WEEK_AGO, MEAN_TODAY),
    regime_experts=True,
    noise_checks=True,
)
TASK_B = Task(
    name="model_b",
    target=TARGET_MEAN_PRICE,
    features=(*FEATURES_MODEL_B, HORIZON_DAYS),
    price_features=PRICE_FEATURES_B,
    one_hot=(DAY_OF_WEEK, MONTH, HORIZON_DAYS),
    period_column=DECISION_DAY,
    candidates=(NO_REGIME, GLOBAL),
    reference=MEAN_TODAY,
    baselines=(MEAN_TODAY, NAIVE_WEEK_AGO),
    regime_experts=False,
    noise_checks=False,
)


def prediction_column(model: str) -> str:
    """Column of a model's predictions, e.g. 'pred_global'."""
    return f"pred_{model}"


def window_label(window_months: int | None) -> str:
    """Name of a training window: 'expanding' (every past row) or e.g. '24_months'."""
    if window_months is None:
        return "expanding"
    return f"{window_months}_months"


def window_model(window_months: int | None) -> str:
    """Name of the global model refitted on another training window, e.g. 'global_24_months'."""
    return f"{GLOBAL}_{window_label(window_months)}"


def regime_feature_columns(params: dict[str, Any]) -> list[str]:
    """Filtered regime of J and its probabilities, e.g. regime_filtered, prob_filtered_calm."""
    return regime_columns(FILTERED, params["regime_names"])


def model_features(task: Task, model: str, params: dict[str, Any]) -> list[str]:
    """Columns a fitted model sees.

    Without regime: no_regime, ridge and the per-regime experts (the regime picks
    the expert). Without prices: no_price, which also drops the regime (computed
    from prices). Every other model adds the filtered regime of J.
    """
    if model in (NO_REGIME, RIDGE, PER_REGIME):
        return list(task.features)
    if model == NO_PRICE:
        return [name for name in task.features if name not in task.price_features]
    return [*task.features, *regime_feature_columns(params)]


# ==================================================
# Baselines: no fitting
# ==================================================


def baseline_predictions(inputs: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Predictions of the baselines, from features known at 11:00 J.

    Model A: similar day (same hour of J from Tuesday to Friday, of the delivered
    day minus 7 from Saturday to Monday), same hour a week before, mean of J.
    Model B: mean of J, mean of the predicted day a week before.
    """
    if task.name == TASK_A.name:
        uses_week_ago = inputs[DAY_OF_WEEK].isin(WEEK_AGO_WEEKDAYS)
        similar_day = inputs[PRICE_SAME_HOUR_TODAY].where(
            ~uses_week_ago, inputs[PRICE_SAME_HOUR_WEEK_AGO]
        )
        baselines = {
            NAIVE_SIMILAR_DAY: similar_day,
            NAIVE_WEEK_AGO: inputs[PRICE_SAME_HOUR_WEEK_AGO],
            MEAN_TODAY: inputs[ANCHOR],
        }
    else:
        baselines = {MEAN_TODAY: inputs[ANCHOR], NAIVE_WEEK_AGO: inputs[PRICE_MEAN_WEEK_AGO]}
    columns = {prediction_column(name): values for name, values in baselines.items()}
    return pd.DataFrame(columns, index=inputs.index)


# ==================================================
# One model
# ==================================================


def _matrix(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Columns as floats; nullable integers become NaN where missing."""
    return table[columns].astype(float)


def model_target(table: pd.DataFrame, task: Task, anchored: bool) -> pd.Series:
    """Target a model learns: gap to the mean price of J (anchored) or the price itself."""
    if anchored:
        return table[task.target] - table[ANCHOR]
    return table[task.target]


def fit_xgboost(
    features: pd.DataFrame,
    target: pd.Series,
    settings: dict[str, Any],
    seed: int,
    weights: np.ndarray | None = None,
) -> XGBRegressor:
    """XGBoost with the absolute error loss, on the rows whose target is known.

    Missing features are allowed: each split learns where to send them.
    """
    is_known = target.notna().to_numpy()
    model = XGBRegressor(
        objective=XGBOOST_OBJECTIVE,
        tree_method=XGBOOST_TREE_METHOD,
        random_state=seed,
        **settings,
    )
    known_weights = None if weights is None else weights[is_known]
    model.fit(features[is_known], target[is_known], sample_weight=known_weights)
    return model


def xgboost_predictions(
    training: pd.DataFrame,
    rows: pd.DataFrame,
    columns: list[str],
    task: Task,
    settings: dict[str, Any],
    seed: int,
    anchored: bool = True,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, XGBRegressor]:
    """Fit one XGBoost on `training` and predict the price of `rows`.

    Returns:
        The predicted prices (anchor added back when anchored) and the model.
    """
    target = model_target(training, task, anchored)
    model = fit_xgboost(_matrix(training, columns), target, settings, seed, weights)
    raw_predictions = model.predict(_matrix(rows, columns))
    if not anchored:
        return raw_predictions.astype(float), model
    return raw_predictions + rows[ANCHOR].to_numpy(dtype=float), model


def tree_curve_predictions(
    model: XGBRegressor, rows: pd.DataFrame, columns: list[str], step: int
) -> dict[str, np.ndarray]:
    """Anchored predictions of `model` with its first step, 2 x step, ... trees.

    Free: one fit, several predictions. Rule 3 of run 5 reads the curve.
    """
    features = _matrix(rows, columns)
    anchor = rows[ANCHOR].to_numpy(dtype=float)
    n_trees = model.get_params()["n_estimators"]
    curve = {}
    for trees in range(step, n_trees + 1, step):
        gaps = model.predict(features, iteration_range=(0, trees))
        curve[f"{TREES_PREFIX}{trees}"] = gaps + anchor
    return curve


def balanced_weights(labels: pd.Series, n_regimes: int) -> np.ndarray:
    """Weight of each training row: every regime weighs the same in total.

    Weights average 1 over the labelled rows; a row without regime (before the
    first filtered regime) weighs 1. Computed on the training rows only.
    """
    counts = labels.value_counts()
    present_regimes = [state for state in range(n_regimes) if counts.get(state, 0) > 0]
    labelled_rows = int(counts.sum())
    regime_weights = {
        state: labelled_rows / (len(present_regimes) * counts[state]) for state in present_regimes
    }
    weights = labels.astype(float).map(regime_weights)
    return weights.fillna(1.0).to_numpy(dtype=float)


def regime_experts(
    training: pd.DataFrame,
    rows: pd.DataFrame,
    task: Task,
    params: dict[str, Any],
    global_predictions: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """One XGBoost per filtered regime of J, each fitted on the rows of that regime.

    A regime with fewer than min_regime_training_hours known training rows uses
    the global model: a model learnt on one season of a regime would predict the
    others badly.

    Returns:
        Predictions of every expert for every row (one column per regime) and,
        per regime, "own" or "global".
    """
    settings = params[task.name]["xgboost"]
    columns = model_features(task, PER_REGIME, params)
    labels = training[REGIME_FILTERED]
    has_target = model_target(training, task, anchored=True).notna()
    expert_columns, sources = [], []
    for state in range(len(params["regime_names"])):
        is_state = (labels == state).fillna(False).to_numpy(dtype=bool)
        known_rows = int((is_state & has_target.to_numpy()).sum())
        if known_rows < params["min_regime_training_hours"]:
            expert_columns.append(global_predictions)
            sources.append(GLOBAL)
            continue
        expert, _ = xgboost_predictions(
            training[is_state], rows, columns, task, settings, params["random_state"]
        )
        expert_columns.append(expert)
        sources.append("own")
    return np.column_stack(expert_columns), sources


def pick_expert(experts: np.ndarray, labels: pd.Series, fallback: np.ndarray) -> np.ndarray:
    """Prediction of the expert of each row's regime; `fallback` where the regime is unknown."""
    label_values = labels.to_numpy(dtype=float, na_value=np.nan)
    is_known = ~np.isnan(label_values)
    positions = np.where(is_known, label_values, 0).astype(int)
    picked = experts[np.arange(len(positions)), positions]
    return np.where(is_known, picked, fallback)


def mix_experts(experts: np.ndarray, probabilities: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Experts weighted by the filtered probabilities of J; `fallback` where they are unknown."""
    is_known = ~np.isnan(probabilities).any(axis=1)
    weights = np.nan_to_num(probabilities, nan=0.0)
    mixed = (experts * weights).sum(axis=1)
    return np.where(is_known, mixed, fallback)


def ridge_predictions(
    training: pd.DataFrame, rows: pd.DataFrame, task: Task, params: dict[str, Any]
) -> np.ndarray:
    """Linear baseline: ridge on the same features, calendar as categories, anchored target.

    Scaled and one-hot encoded with the training rows only. A row with a missing
    feature is not used for training and gets no prediction: a linear model has
    no way to handle it.
    """
    columns = list(task.features)
    target = model_target(training, task, anchored=True)
    is_complete = training[columns].notna().all(axis=1) & target.notna()
    complete_training = training[is_complete]
    numeric = [name for name in columns if name not in task.one_hot]
    scaler = StandardScaler().fit(_matrix(complete_training, numeric))
    categories = {name: np.sort(complete_training[name].unique()) for name in task.one_hot}
    design = _ridge_design(complete_training, numeric, scaler, categories)
    model = Ridge(alpha=params["ridge_alpha"]).fit(design, target[is_complete])
    has_features = rows[columns].notna().all(axis=1).to_numpy()
    filled_rows = _matrix(rows, columns).fillna(0.0)  # masked below: only shapes the design
    row_design = _ridge_design(filled_rows, numeric, scaler, categories)
    predictions = model.predict(row_design) + rows[ANCHOR].to_numpy(dtype=float)
    return np.where(has_features, predictions, np.nan)


def _ridge_design(
    table: pd.DataFrame,
    numeric: list[str],
    scaler: StandardScaler,
    categories: dict[str, np.ndarray],
) -> np.ndarray:
    """Scaled numeric columns, then one 0/1 column per training category of each calendar column."""
    scaled = scaler.transform(_matrix(table, numeric))
    indicators = [
        (table[name].to_numpy()[:, None] == values[None, :]).astype(float)
        for name, values in categories.items()
    ]
    return np.hstack([scaled, *indicators])


def quantile_interval(
    training: pd.DataFrame, rows: pd.DataFrame, task: Task, params: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Raw interval: LightGBM quantile models on the anchored target, regime included.

    The two bounds are sorted on each row: two quantile models fitted apart can cross.
    """
    settings = params[task.name]["lightgbm"]
    interval = params["interval"]
    columns = model_features(task, GLOBAL, params)
    target = model_target(training, task, anchored=True)
    is_known = target.notna().to_numpy()
    features = _matrix(training, columns)[is_known]
    row_features = _matrix(rows, columns)
    anchor = rows[ANCHOR].to_numpy(dtype=float)
    bounds = []
    for quantile in (interval["lower_quantile"], interval["upper_quantile"]):
        model = LGBMRegressor(
            objective=LIGHTGBM_OBJECTIVE,
            alpha=quantile,
            subsample_freq=LIGHTGBM_BAGGING_FREQUENCY,
            random_state=params["random_state"],
            deterministic=True,
            force_col_wise=True,
            verbose=LIGHTGBM_SILENT,
            **settings,
        )
        model.fit(features, target[is_known])
        bounds.append(model.predict(row_features) + anchor)
    return np.minimum(*bounds), np.maximum(*bounds)


# ==================================================
# One month: every model of a task
# ==================================================


@dataclass(frozen=True)
class MonthData:
    """Rows of one monthly refit.

    Attributes:
        training: Training rows of the configured window (predicted day before the month).
        rows: Rows of the tested month.
        other_windows: Training rows of the other candidate windows, by window
            (development months of model A only, else empty).
        is_development: Whether the month starts before holdout_start.
    """

    training: pd.DataFrame
    rows: pd.DataFrame
    other_windows: dict[int | None, pd.DataFrame]
    is_development: bool


def _main_models(
    data: MonthData, task: Task, params: dict[str, Any]
) -> tuple[dict[str, np.ndarray], XGBRegressor]:
    """Ridge, no_regime, global and the two ablations (level target, fundamentals only)."""
    settings = params[task.name]["xgboost"]
    seed = params["random_state"]
    predictions = {RIDGE: ridge_predictions(data.training, data.rows, task, params)}
    no_regime_columns = model_features(task, NO_REGIME, params)
    predictions[NO_REGIME], _ = xgboost_predictions(
        data.training, data.rows, no_regime_columns, task, settings, seed
    )
    global_columns = model_features(task, GLOBAL, params)
    predictions[GLOBAL], global_model = xgboost_predictions(
        data.training, data.rows, global_columns, task, settings, seed
    )
    for model in (LEVEL_TARGET, NO_PRICE):
        columns = model_features(task, model, params)
        predictions[model], _ = xgboost_predictions(
            data.training, data.rows, columns, task, settings, seed, anchored=False
        )
    return predictions, global_model


def _regime_variants(
    data: MonthData, task: Task, params: dict[str, Any], predictions: dict[str, np.ndarray]
) -> list[str]:
    """Weighted, per-regime, hybrid and average variants, added to `predictions`.

    Returns:
        "own" or "global" for each regime's expert.
    """
    settings = params[task.name]["xgboost"]
    columns = model_features(task, WEIGHTED, params)
    weights = balanced_weights(data.training[REGIME_FILTERED], len(params["regime_names"]))
    predictions[WEIGHTED], _ = xgboost_predictions(
        data.training, data.rows, columns, task, settings, params["random_state"], weights=weights
    )
    experts, sources = regime_experts(data.training, data.rows, task, params, predictions[GLOBAL])
    labels = data.rows[REGIME_FILTERED]
    probabilities = _matrix(data.rows, regime_feature_columns(params)[1:]).to_numpy()
    predictions[PER_REGIME] = pick_expert(experts, labels, predictions[GLOBAL])
    predictions[HYBRID] = mix_experts(experts, probabilities, predictions[GLOBAL])
    averaged = [predictions[model] for model in task.candidates if model != AVERAGE]
    predictions[AVERAGE] = np.mean(averaged, axis=0)
    return sources


def _noise_checks(
    data: MonthData, task: Task, params: dict[str, Any], predictions: dict[str, np.ndarray]
) -> None:
    """Global model with the second seed, and on the other training windows (development only)."""
    settings = params[task.name]["xgboost"]
    columns = model_features(task, GLOBAL, params)
    predictions[SECOND_SEED], _ = xgboost_predictions(
        data.training, data.rows, columns, task, settings, params["stability_seed"]
    )
    for window_months, training in data.other_windows.items():
        predictions[window_model(window_months)], _ = xgboost_predictions(
            training, data.rows, columns, task, settings, params["random_state"]
        )


def fit_month(
    data: MonthData, task: Task, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit every model of a task on the training rows and predict the tested month.

    Args:
        data: Training rows and rows of the month.
        task: TASK_A or TASK_B.
        params: Prediction parameters.

    Returns:
        One row per row of the month: one prediction column per model, the raw
        interval and the tree curve of the global model; and the record of the refit.
    """
    predictions, global_model = _main_models(data, task, params)
    sources = _regime_variants(data, task, params, predictions) if task.regime_experts else []
    if task.noise_checks and data.is_development:
        _noise_checks(data, task, params, predictions)
    lower, upper = quantile_interval(data.training, data.rows, task, params)
    curve = tree_curve_predictions(
        global_model,
        data.rows,
        model_features(task, GLOBAL, params),
        params[task.name]["tree_checkpoint_step"],
    )
    columns = {prediction_column(name): values for name, values in predictions.items()}
    columns.update({INTERVAL_LOWER_RAW: lower, INTERVAL_UPPER_RAW: upper, **curve})
    table = pd.DataFrame(columns, index=data.rows.index)
    has_target = model_target(data.training, task, anchored=True).notna()
    known_days = data.training.loc[has_target, TARGET_DAY]
    record = {
        "training_rows": int(has_target.sum()),
        "training_end": str(known_days.max().date()),
        "predicted_rows": len(data.rows),
        "experts": dict(zip(params["regime_names"], sources, strict=False)),
    }
    return table, record
