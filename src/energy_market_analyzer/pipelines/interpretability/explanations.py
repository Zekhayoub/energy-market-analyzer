"""SHAP explanations of the published predictions of models A and B (decision D33).

What is explained: the variant selected on the development period (pred_selected),
nothing else. Each month its XGBoosts are refitted exactly as in the prediction
pipeline (same functions, rows and seed; no saved model, decision D30), then the
rows of the month are explained. A variant made of several XGBoosts (per-regime
experts, hybrid, average) is a weighted sum of models: SHAP values are linear in the
model, so the same weighted sum of their explanations is exact.

- Exact TreeSHAP (path-dependent), computed by XGBoost itself (pred_contribs),
  identical to shap.TreeExplainer. It needs no background sample and does not
  build impossible inputs (a maximum price of J below its mean).
- Anchored target (decision D26): predicted price = mean price of J (known at
  11:00 J) + expected gap of the month's model + the sum of the contributions.
  SHAP explains what moves J+1 away from J; the level of J, gas included, is the anchor.
- Checks that stop the pipeline: the refit gives back the published prediction,
  and the decomposition adds up to it, both within EXACTNESS_TOLERANCE_EUR_MWH.
- Robustness, rows predicted before holdout_start only: the same explanations with the
  second seed (are the drivers shown stable?) and interventional SHAP on one day per
  month (does the attribution method change the families' shares?).

Correlated features (the prices of J) share their credit in a way that is not
unique: the reports also sum the contributions by family of features, an exact
decomposition. SHAP describes the model, not the market: a contribution is
associated with a prediction, it is not its cause.

References:
    Lundberg, S. M. et al. (2020). From local explanations to global understanding with
    explainable AI for trees. Nature Machine Intelligence 2.
    Chen, H., Janizek, J. D., Lundberg, S. and Lee, S.-I. (2020). True to the model or true
    to the data? arXiv:2006.16234.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from xgboost import DMatrix, XGBRegressor

from energy_market_analyzer.pipelines.features.nodes import (
    CLEAN_SPARK_SPREAD_TODAY,
    DAY_CALENDAR,
    DECISION_DAY,
    EUA_LAST,
    HORIZON_DAYS,
    HOUR,
    PRICE_MEAN_CHANGE_TODAY,
    PRICE_MEAN_WEEK_AGO,
    PRICE_SAME_HOUR_TODAY,
    PRICE_SAME_HOUR_WEEK_AGO,
    TODAY_PRICES,
    TTF_LAST,
    YESTERDAY_REALIZED,
)
from energy_market_analyzer.pipelines.ingestion.nodes import LOAD_FORECAST
from energy_market_analyzer.pipelines.prediction.evaluation import holdout_day
from energy_market_analyzer.pipelines.prediction.models import (
    ANCHOR,
    AVERAGE,
    GLOBAL,
    HYBRID,
    NO_REGIME,
    PER_REGIME,
    SELECTED,
    TARGET_DAY,
    TASK_A,
    WEIGHTED,
    Task,
    balanced_weights,
    fit_xgboost,
    model_features,
    model_target,
    prediction_column,
    regime_feature_columns,
)
from energy_market_analyzer.pipelines.prediction.nodes import test_months, training_mask
from energy_market_analyzer.pipelines.regime.nodes import REGIME_FILTERED

logger = logging.getLogger(__name__)

# --- Columns of the SHAP tables ---
SHAP_PREFIX = "shap_"
BASE_VALUE = "base_value"  # expected gap of the month's model over its training rows
PRED_SELECTED = prediction_column(SELECTED)
CONTEXT_A = (DECISION_DAY, TARGET_DAY, REGIME_FILTERED, ANCHOR, PRED_SELECTED)
CONTEXT_B = (TARGET_DAY, REGIME_FILTERED, ANCHOR, PRED_SELECTED)

# --- What is explained ---
# Variants that are one XGBoost; the others (per-regime experts, hybrid, average) are
# mixtures of XGBoosts, explained by the same mixture of their explanations.
SINGLE_MODEL_VARIANTS = (NO_REGIME, GLOBAL, WEIGHTED)
# The refit gives back the published prediction, and base + contributions + anchor add up
# to it, within this gap (XGBoost computes in single precision).
EXACTNESS_TOLERANCE_EUR_MWH = 0.01
INTERVENTIONAL = "interventional"
XGBOOST_DTYPE = np.float32  # XGBoost reads every feature value in float32
MONTH_FORMAT = "%Y-%m"
SECONDS_DECIMALS = 1
GAP_DECIMALS = 6

# --- Families of features: features that carry the same kind of information ---
HOUR_FAMILY = "hour"
DAY_CALENDAR_FAMILY = "day_calendar"
PRICES_TODAY_FAMILY = "prices_today"
PRICE_WEEK_AGO_FAMILY = "price_week_ago"
LOAD_FORECAST_FAMILY = "load_forecast"
REALIZED_YESTERDAY_FAMILY = "realized_yesterday"
FUELS_FAMILY = "fuels_and_spread"
HORIZON_FAMILY = "horizon"
REGIME_FAMILY = "regime_today"
# Features of model A whose value changes from one hour of the delivered day to the next;
# every other feature is the same for all the hours of a day.
WITHIN_DAY_FEATURES = (HOUR, PRICE_SAME_HOUR_TODAY, PRICE_SAME_HOUR_WEEK_AGO, LOAD_FORECAST)


def shap_column(feature: str) -> str:
    """Column of a feature's contribution in the SHAP tables, e.g. 'shap_hour'."""
    return f"{SHAP_PREFIX}{feature}"


def contribution_columns(table: pd.DataFrame) -> list[str]:
    """Contribution columns of a SHAP table, in table order."""
    return [name for name in table.columns if name.startswith(SHAP_PREFIX)]


def feature_contributions(table: pd.DataFrame) -> pd.DataFrame:
    """Contributions of a SHAP table with the feature names as columns."""
    columns = contribution_columns(table)
    names = [name.removeprefix(SHAP_PREFIX) for name in columns]
    return table[columns].set_axis(names, axis=1)


def feature_families(params: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Every family of features of models A and B, in display order.

    The clean spark spread of J sits with the fuels: it is the margin of a gas plant,
    although its price part is the price of J.
    """
    return {
        HOUR_FAMILY: (HOUR,),
        DAY_CALENDAR_FAMILY: DAY_CALENDAR,
        PRICES_TODAY_FAMILY: (
            PRICE_SAME_HOUR_TODAY,
            *TODAY_PRICES.values(),
            PRICE_MEAN_CHANGE_TODAY,
        ),
        PRICE_WEEK_AGO_FAMILY: (PRICE_SAME_HOUR_WEEK_AGO, PRICE_MEAN_WEEK_AGO),
        LOAD_FORECAST_FAMILY: (LOAD_FORECAST,),
        REALIZED_YESTERDAY_FAMILY: tuple(YESTERDAY_REALIZED.values()),
        FUELS_FAMILY: (TTF_LAST, EUA_LAST, CLEAN_SPARK_SPREAD_TODAY),
        HORIZON_FAMILY: (HORIZON_DAYS,),
        REGIME_FAMILY: tuple(regime_feature_columns(params)),
    }


def families_of(features: list[str], params: dict[str, Any]) -> dict[str, list[str]]:
    """Families of `features`, each with its features; families without any are left out.

    Raises:
        ValueError: If a feature belongs to no family.
    """
    all_families = feature_families(params)
    family_by_feature = {
        feature: family for family, members in all_families.items() for feature in members
    }
    orphans = [feature for feature in features if feature not in family_by_feature]
    if orphans:
        raise ValueError(f"Features without a family in explanations.py: {orphans}")
    families = {}
    for family in all_families:
        members = [feature for feature in features if family_by_feature[feature] == family]
        if members:
            families[family] = members
    return families


def family_sums(contributions: pd.DataFrame, families: dict[str, list[str]]) -> pd.DataFrame:
    """Sum of the contributions of each family (exact: SHAP values add up)."""
    sums = {family: contributions[members].sum(axis=1) for family, members in families.items()}
    return pd.DataFrame(sums, index=contributions.index)


# ==================================================
# One refit and its explanations
# ==================================================


def selected_variant(prediction_report: dict[str, Any], task: Task) -> str:
    """Variant of the task selected on the development period (prediction report).

    Raises:
        ValueError: If it is not one of the task's candidates.
    """
    selection = prediction_report["choices_before_holdout"][task.name]["selection"]
    variant = selection["selected"]
    if variant not in task.candidates:
        raise ValueError(
            f"{task.name}: unknown selected variant '{variant}' (candidates "
            f"{list(task.candidates)})"
        )
    return variant


def explained_features(task: Task, variant: str, params: dict[str, Any]) -> list[str]:
    """Features whose contributions are reported: those of the variant's XGBoost, or, for a
    mixture of XGBoosts, every feature of the task and the regime of J (a contribution is
    zero where a model of the mixture does not see the feature)."""
    if variant in SINGLE_MODEL_VARIANTS:
        return model_features(task, variant, params)
    return model_features(task, GLOBAL, params)


def tree_contributions(model: XGBRegressor, rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Exact TreeSHAP of `model` on `rows`: one contribution per feature and the base value.

    XGBoost's pred_contribs returns one column per feature, then the expected
    value of the model; together they add up to the model's output, the gap to
    the anchor (EUR/MWh).
    """
    matrix = rows[columns].astype(float)
    data = DMatrix(matrix, missing=np.nan)
    values = model.get_booster().predict(data, pred_contribs=True)
    names = [*(shap_column(name) for name in columns), BASE_VALUE]
    return pd.DataFrame(values.astype(float), index=rows.index, columns=names)


def interventional_contributions(
    model: XGBRegressor,
    training: pd.DataFrame,
    rows: pd.DataFrame,
    columns: list[str],
    params: dict[str, Any],
    check: dict[str, Any],
) -> pd.DataFrame:
    """Interventional SHAP of `rows` against training rows drawn with the prediction seed.

    Robustness check of the attribution method only: the features are moved one
    by one to the values of the background rows, whatever their correlations.
    Values are read in float32, as XGBoost reads them: a filtered probability of 1e-60
    is 0 for XGBoost but not in float64, which would send a row down another branch
    than the prediction's (gap of 0.34 EUR/MWh on model B in run 7).
    """
    rng = np.random.default_rng(params["random_state"])
    size = min(check["background_rows"], len(training))
    positions = np.sort(rng.choice(len(training), size=size, replace=False))
    background = training.iloc[positions][columns].astype(XGBOOST_DTYPE)
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation=INTERVENTIONAL)
    values = explainer.shap_values(rows[columns].astype(XGBOOST_DTYPE), check_additivity=False)
    names = [shap_column(name) for name in columns]
    table = pd.DataFrame(np.asarray(values, dtype=float), index=rows.index, columns=names)
    expected = float(np.ravel(explainer.expected_value)[0])
    return table.assign(**{BASE_VALUE: expected})


# ==================================================
# A variant as XGBoosts and their weights
# ==================================================


@dataclass(frozen=True)
class Component:
    """One XGBoost of a variant and its weight in the prediction of each row of the month.

    Attributes:
        model: The fitted XGBoost (it predicts the gap to the anchor).
        columns: Columns it sees.
        training: Its training rows (reference of the interventional check).
        weights: Its weight in each row's prediction, one per row of the month.
    """

    model: XGBRegressor
    columns: list[str]
    training: pd.DataFrame
    weights: np.ndarray


def _fit_model(
    training: pd.DataFrame, task: Task, model_name: str, params: dict[str, Any], seed: int
) -> tuple[XGBRegressor, list[str]]:
    """One XGBoost of the prediction pipeline (same columns, target, weights, settings, seed)."""
    columns = model_features(task, model_name, params)
    target = model_target(training, task, anchored=True)
    sample_weights = None
    if model_name == WEIGHTED:
        sample_weights = balanced_weights(training[REGIME_FILTERED], len(params["regime_names"]))
    settings = params[task.name]["xgboost"]
    model = fit_xgboost(training[columns].astype(float), target, settings, seed, sample_weights)
    return model, columns


def _experts(
    training: pd.DataFrame, task: Task, params: dict[str, Any], seed: int
) -> list[tuple[XGBRegressor, list[str], pd.DataFrame] | None]:
    """One XGBoost per filtered regime of J, as `regime_experts` of the prediction pipeline;
    None for a regime with too few known training rows (the global model replaces it)."""
    labels = training[REGIME_FILTERED]
    has_target = model_target(training, task, anchored=True).notna().to_numpy()
    experts = []
    for state in range(len(params["regime_names"])):
        is_state = (labels == state).fillna(False).to_numpy(dtype=bool)
        known_rows = int((is_state & has_target).sum())
        if known_rows < params["min_regime_training_hours"]:
            experts.append(None)
            continue
        subset = training[is_state]
        model, columns = _fit_model(subset, task, PER_REGIME, params, seed)
        experts.append((model, columns, subset))
    return experts


def _per_regime_weights(rows: pd.DataFrame, params: dict[str, Any]) -> tuple[np.ndarray, ...]:
    """Weight of each regime's expert in each row (the row's regime of J), and of the global
    model (rows without regime), as `pick_expert` of the prediction pipeline."""
    labels = rows[REGIME_FILTERED].to_numpy(dtype=float, na_value=np.nan)
    states = np.arange(len(params["regime_names"]))
    expert_weights = (labels[:, None] == states[None, :]).astype(float)
    global_weights = np.isnan(labels).astype(float)
    return expert_weights, global_weights


def _hybrid_weights(rows: pd.DataFrame, params: dict[str, Any]) -> tuple[np.ndarray, ...]:
    """Weight of each regime's expert in each row (filtered probability of J), and of the
    global model (rows without probabilities), as `mix_experts` of the prediction pipeline."""
    probabilities = rows[regime_feature_columns(params)[1:]].astype(float).to_numpy()
    is_known = ~np.isnan(probabilities).any(axis=1)
    expert_weights = np.where(is_known[:, None], np.nan_to_num(probabilities, nan=0.0), 0.0)
    global_weights = (~is_known).astype(float)
    return expert_weights, global_weights


def _mixture(
    experts: list[tuple[XGBRegressor, list[str], pd.DataFrame] | None],
    expert_weights: np.ndarray,
    global_weights: np.ndarray,
    global_component: Component,
    share: float,
) -> list[Component]:
    """Experts and the global model with their weights; a missing expert's weight goes to
    the global model, as in the prediction pipeline."""
    fallback = global_weights.copy()
    components = []
    for position, expert in enumerate(experts):
        weights = expert_weights[:, position]
        if expert is None:
            fallback = fallback + weights
            continue
        model, columns, subset = expert
        components.append(Component(model, columns, subset, share * weights))
    global_model, global_columns = global_component.model, global_component.columns
    components.append(
        Component(global_model, global_columns, global_component.training, share * fallback)
    )
    return components


def variant_components(
    training: pd.DataFrame,
    rows: pd.DataFrame,
    task: Task,
    variant: str,
    params: dict[str, Any],
    seed: int,
) -> list[Component]:
    """XGBoosts of a variant and their weight in each row, refitted as the prediction pipeline
    does: one model; the regime experts picked (per_regime) or mixed by the filtered
    probabilities of J (hybrid); or the mean of the other candidates (average)."""
    every_row = np.ones(len(rows))
    if variant in SINGLE_MODEL_VARIANTS:
        model, columns = _fit_model(training, task, variant, params, seed)
        return [Component(model, columns, training, every_row)]
    global_model, global_columns = _fit_model(training, task, GLOBAL, params, seed)
    global_component = Component(global_model, global_columns, training, every_row)
    experts = _experts(training, task, params, seed)
    mixtures = {
        PER_REGIME: _per_regime_weights(rows, params),
        HYBRID: _hybrid_weights(rows, params),
    }
    if variant in mixtures:
        return _mixture(experts, *mixtures[variant], global_component, 1.0)
    candidates = [name for name in task.candidates if name != AVERAGE]
    share = 1.0 / len(candidates)
    components = []
    for name in candidates:
        if name in mixtures:
            components.extend(_mixture(experts, *mixtures[name], global_component, share))
        elif name == GLOBAL:
            components.append(Component(global_model, global_columns, training, share * every_row))
        else:
            model, columns = _fit_model(training, task, name, params, seed)
            components.append(Component(model, columns, training, share * every_row))
    return components


@dataclass(frozen=True)
class MonthExplanation:
    """Explanations of the rows of one month.

    Attributes:
        predicted: Prices predicted by the refit (anchor added back).
        contributions: Base value and one contribution per feature, one row per row.
        interventional: Interventional contributions of the checked rows (maybe none).
    """

    predicted: pd.Series
    contributions: pd.DataFrame
    interventional: pd.DataFrame


def explain_month(
    training: pd.DataFrame,
    rows: pd.DataFrame,
    task: Task,
    variant: str,
    params: dict[str, Any],
    seed: int,
    checked_rows: pd.DataFrame | None = None,
    check: dict[str, Any] | None = None,
) -> MonthExplanation:
    """Refit the variant on `training` with `seed` (as the prediction pipeline does) and
    explain `rows`; interventional SHAP of `checked_rows` when given.

    A mixture of XGBoosts is explained exactly by the same mixture of their explanations
    (SHAP values are linear in the model). For the hybrid variant the regime probabilities
    are weights of the mixture, not contributions.

    Args:
        training: Training rows (predicted day before the month).
        rows: Rows of the month.
        task: TASK_A or TASK_B.
        variant: Variant explained (one of the task's candidates).
        params: Prediction parameters.
        seed: Seed of the XGBoosts.
        checked_rows: Rows of the method check, or None.
        check: method_check_* settings (day_of_month, background_rows), or None.

    Returns:
        The refit's predictions and the explanations of the month.
    """
    columns = explained_features(task, variant, params)
    names = [*(shap_column(name) for name in columns), BASE_VALUE]
    has_check = checked_rows is not None and not checked_rows.empty
    checked_positions = rows.index.get_indexer(checked_rows.index) if has_check else None
    gap = np.zeros(len(rows))
    contributions = pd.DataFrame(0.0, index=rows.index, columns=names)
    interventional = (
        pd.DataFrame(0.0, index=checked_rows.index, columns=names) if has_check else None
    )
    for component in variant_components(training, rows, task, variant, params, seed):
        if not component.weights.any():
            continue
        predicted_gap = component.model.predict(rows[component.columns].astype(float))
        gap = gap + component.weights * predicted_gap
        values = tree_contributions(component.model, rows, component.columns)
        contributions = contributions.add(values.mul(component.weights, axis=0), fill_value=0.0)
        if has_check:
            checked = interventional_contributions(
                component.model, component.training, checked_rows, component.columns, params, check
            )
            checked_weights = component.weights[checked_positions]
            interventional = interventional.add(
                checked.mul(checked_weights, axis=0), fill_value=0.0
            )
    anchor = rows[ANCHOR].to_numpy(dtype=float)
    predicted = pd.Series(anchor + gap, index=rows.index)
    checked_table = interventional[names] if has_check else pd.DataFrame()
    return MonthExplanation(predicted, contributions[names], checked_table)


# ==================================================
# Every month of a task
# ==================================================


def indexed_inputs(inputs: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Inputs with the index of the task's prediction table.

    Model A: one row per delivered hour (already). Model B: flat inputs, given the
    (decision_day, horizon_days) index of predictions_model_b; the columns stay.
    """
    if task.name == TASK_A.name:
        return inputs
    index = pd.MultiIndex.from_arrays(
        [inputs[DECISION_DAY], inputs[HORIZON_DAYS]], names=[DECISION_DAY, HORIZON_DAYS]
    )
    return inputs.set_axis(index)


def month_rows(
    inputs: pd.DataFrame, month_start: pd.Timestamp, task: Task, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows (predicted day before the month, in the window) and rows of the month,
    exactly as the walk-forward of the prediction pipeline selects them."""
    next_start = month_start + pd.offsets.MonthBegin(1)
    period_days = inputs[task.period_column]
    is_month = ((period_days >= month_start) & (period_days < next_start)).to_numpy()
    window = params["training_window_months"]
    is_training = training_mask(inputs[TARGET_DAY], month_start, window)
    return inputs[is_training], inputs[is_month]


@dataclass(frozen=True)
class Explanations:
    """Explanations of the rows of several months.

    Attributes:
        contributions: Base value and contributions of every explained row.
        predicted: Prices predicted by the refits.
        interventional: Interventional contributions of the checked rows.
        refits: One record per month.
    """

    contributions: pd.DataFrame
    predicted: pd.Series
    interventional: pd.DataFrame
    refits: list[dict[str, Any]]


def _checked_rows(
    rows: pd.DataFrame,
    month_start: pd.Timestamp,
    task: Task,
    params: dict[str, Any],
    check: dict[str, Any] | None,
) -> pd.DataFrame | None:
    """Rows of the method check: the configured day of a month, predicted day before
    holdout_start (a model B row decided in August for September is a reserved row, D25)."""
    if check is None or month_start >= holdout_day(params):
        return None
    is_checked_day = (rows[task.period_column].dt.day == check["day_of_month"]).to_numpy()
    is_development_row = (rows[TARGET_DAY] < holdout_day(params)).to_numpy()
    return rows[is_checked_day & is_development_row]


def explain_months(
    inputs: pd.DataFrame,
    task: Task,
    variant: str,
    params: dict[str, Any],
    seed: int,
    months: list[pd.Timestamp],
    check: dict[str, Any] | None = None,
) -> Explanations:
    """Refit the variant at the start of each month and explain the rows of the month.

    Args:
        inputs: Output of `indexed_inputs`.
        task: TASK_A or TASK_B.
        variant: Variant explained.
        params: Prediction parameters.
        seed: Seed of every refit.
        months: First day of each month to explain.
        check: method_check_* settings, or None for no interventional check.

    Returns:
        The explanations of every row of the months.
    """
    parts, predicted, interventional, refits = [], [], [], []
    for month_start in months:
        started = time.perf_counter()
        training, rows = month_rows(inputs, month_start, task, params)
        checked = _checked_rows(rows, month_start, task, params, check)
        month = explain_month(training, rows, task, variant, params, seed, checked, check)
        seconds = round(time.perf_counter() - started, SECONDS_DECIMALS)
        parts.append(month.contributions)
        predicted.append(month.predicted)
        interventional.append(month.interventional)
        refits.append(
            {
                "month": month_start.strftime(MONTH_FORMAT),
                "training_rows": len(training),
                "rows": len(rows),
                "seconds": seconds,
            }
        )
        logger.info(
            "%s %s (seed %d): %d rows explained (%.1f s)",
            task.name,
            refits[-1]["month"],
            seed,
            len(rows),
            seconds,
        )
    return Explanations(pd.concat(parts), pd.concat(predicted), pd.concat(interventional), refits)


# ==================================================
# Checks
# ==================================================


def _check_selected(predictions: pd.DataFrame, variant: str, task: Task) -> None:
    """Raise ValueError unless pred_selected is the variant's column, row for row."""
    selected = predictions[PRED_SELECTED]
    variant_predictions = predictions[prediction_column(variant)]
    is_same = selected.eq(variant_predictions) | (selected.isna() & variant_predictions.isna())
    if not is_same.all():
        raise ValueError(
            f"{task.name}: pred_selected is not pred_{variant} on {int((~is_same).sum())} rows, "
            f"although the prediction report selects '{variant}': rerun the prediction pipeline"
        )


def _check_months(
    months: list[pd.Timestamp], predictions: pd.DataFrame, inputs: pd.DataFrame, task: Task
) -> None:
    """Raise ValueError unless the predictions cover exactly the walk-forward's rows."""
    index = inputs.index
    period_days = inputs[task.period_column]
    first_month, end = months[0], months[-1] + pd.offsets.MonthBegin(1)
    is_tested = ((period_days >= first_month) & (period_days < end)).to_numpy()
    tested_rows = index[is_tested]
    if not tested_rows.sort_values().equals(predictions.index.sort_values()):
        raise ValueError(
            f"{task.name}: the predictions do not cover the rows of the tested months "
            f"({len(predictions)} rows vs {len(tested_rows)}): the predictions table is stale, "
            f"rerun the prediction pipeline"
        )


def largest_gap(first: pd.Series, second: pd.Series) -> float:
    """Largest absolute gap between two aligned series where both are known (0 if none)."""
    gaps = (first - second).abs()
    return float(gaps.max()) if gaps.notna().any() else 0.0


def check_refit(refit: pd.Series, published: pd.Series, task: Task, variant: str) -> float:
    """Largest gap between the refit and the published predictions.

    Raises:
        ValueError: Beyond EXACTNESS_TOLERANCE_EUR_MWH, or where only one is missing:
            the predictions table does not come from these inputs and parameters.
    """
    aligned = published.reindex(refit.index)
    is_missing_once = refit.isna() != aligned.isna()
    gap = largest_gap(refit, aligned)
    if is_missing_once.any() or gap > EXACTNESS_TOLERANCE_EUR_MWH:
        raise ValueError(
            f"{task.name}: the refit of '{variant}' differs from the published predictions "
            f"(largest gap {gap:.4f} EUR/MWh, {int(is_missing_once.sum())} rows missing in one "
            f"only): the predictions table is stale, rerun the prediction pipeline"
        )
    return gap


def reconstruction_gap(
    contributions: pd.DataFrame, anchor: pd.Series, prediction: pd.Series
) -> float:
    """Largest gap between anchor + base value + contributions and the prediction."""
    columns = [*contribution_columns(contributions), BASE_VALUE]
    rebuilt = anchor.reindex(contributions.index) + contributions[columns].sum(axis=1)
    return largest_gap(rebuilt, prediction.reindex(contributions.index))


def development_rows(table: pd.DataFrame, inputs: pd.DataFrame, params: dict[str, Any]) -> Any:
    """Rows of `table` whose predicted day is before holdout_start (split by the predicted
    day, D25: a model B row decided in August for September belongs to the reserved period)."""
    target_days = inputs[TARGET_DAY].reindex(table.index)
    is_development_row = (target_days < holdout_day(params)).to_numpy()
    return table[is_development_row]


def _require_exact(gap: float, task: Task, what: str) -> float:
    """Raise RuntimeError if a decomposition does not add up to its prediction."""
    if gap > EXACTNESS_TOLERANCE_EUR_MWH:
        raise RuntimeError(
            f"{task.name}: {what} does not add up to the prediction (gap {gap:.4f} EUR/MWh)"
        )
    return gap


# ==================================================
# A task: published explanations and robustness checks
# ==================================================


def explain_task(
    inputs: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_report: dict[str, Any],
    task: Task,
    params: dict[str, Any],
    check: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """SHAP table of the published predictions of a task, and the robustness checks.

    Args:
        inputs: Model inputs of the task (prediction pipeline's input nodes).
        predictions: predictions_model_a or predictions_model_b.
        prediction_report: Report of the prediction pipeline (variant selected).
        task: TASK_A or TASK_B.
        params: Prediction parameters.
        check: method_check_* settings (day_of_month, background_rows).

    Returns:
        The SHAP table (context columns, base value, one contribution per feature) and the
        checks: variant, features, refits, exactness, second seed and interventional SHAP
        (development months).

    Raises:
        ValueError: If the predictions do not come from these inputs and parameters.
    """
    variant = selected_variant(prediction_report, task)
    _check_selected(predictions, variant, task)
    indexed = indexed_inputs(inputs, task)
    months = test_months(indexed[task.period_column], params)
    _check_months(months, predictions, indexed, task)
    published = explain_months(
        indexed, task, variant, params, params["random_state"], months, check
    )
    refit_gap = check_refit(
        published.predicted, predictions[prediction_column(variant)], task, variant
    )
    anchor = predictions[ANCHOR]
    gap = reconstruction_gap(published.contributions, anchor, predictions[PRED_SELECTED])
    _require_exact(gap, task, "base value + contributions + anchor")
    development_months = [month for month in months if month < holdout_day(params)]
    second = explain_months(
        indexed, task, variant, params, params["stability_seed"], development_months
    )
    second_contributions = development_rows(second.contributions, indexed, params)
    second_predicted = development_rows(second.predicted, indexed, params)
    second_gap = reconstruction_gap(second_contributions, anchor, second_predicted)
    _require_exact(second_gap, task, "the second seed's decomposition")
    interventional_gap = reconstruction_gap(published.interventional, anchor, published.predicted)
    context = list(CONTEXT_A if task.name == TASK_A.name else CONTEXT_B)
    table = pd.concat([predictions[context], published.contributions], axis=1)
    checks = {
        "variant": variant,
        "features": explained_features(task, variant, params),
        "refits": published.refits,
        "max_refit_difference_eur_mwh": round(refit_gap, GAP_DECIMALS),
        "max_reconstruction_error_eur_mwh": round(gap, GAP_DECIMALS),
        "second_seed": second_contributions,
        "second_seed_reconstruction_error_eur_mwh": round(second_gap, GAP_DECIMALS),
        "interventional": published.interventional,
        "interventional_reconstruction_error_eur_mwh": round(interventional_gap, GAP_DECIMALS),
    }
    logger.info(
        "%s: '%s' explained on %d rows; refit gap %.6f, decomposition gap %.6f EUR/MWh",
        task.name,
        variant,
        len(table),
        refit_gap,
        gap,
    )
    return table.sort_index(), checks
