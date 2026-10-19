"""Interpretability nodes: SHAP tables of the published predictions and the SHAP report.

Information rule (decisions.md, section 6): each explained row only uses what its
prediction used, known at 11:00 on its decision day J; the model that explains a
month is fitted on the days predicted before that month (explanations.py).

The report reads the explanations three ways (decision D33):
- per row: the usual SHAP importance;
- level of a delivered day (model A): mean contribution over the day's hours, what
  moves the whole day away from today (the budget question);
- shape of a day (model A): contribution of an hour minus its mean over the day,
  what orders the hours (the schedule question: a move within the day only depends
  on that order, decision D32).
Euros stay the unit (as the MAE, D27): the crisis weighs more in the totals, so every
figure is also given by year, epoch, regime and month.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from energy_market_analyzer.pipelines.features.nodes import (
    PRICE_SAME_HOUR_TODAY,
    PRICE_SAME_HOUR_WEEK_AGO,
)
from energy_market_analyzer.pipelines.interpretability.explanations import (
    BASE_VALUE,
    EXACTNESS_TOLERANCE_EUR_MWH,
    PRED_SELECTED,
    explain_task,
    families_of,
    family_sums,
    feature_contributions,
    indexed_inputs,
)
from energy_market_analyzer.pipelines.prediction.evaluation import (
    SHARE_DECIMALS,
    development_epochs,
    group_keys,
    is_development,
    rounded,
)
from energy_market_analyzer.pipelines.prediction.models import (
    ANCHOR,
    TARGET_DAY,
    TASK_A,
    TASK_B,
    Task,
)

logger = logging.getLogger(__name__)

# --- Report ---
TOP_FEATURES = 5  # features listed for each group of rows (every family is listed)
FEATURE_LEVEL = "feature"
FAMILY_LEVEL = "family"
# Rules fixed before run 7 (decisions.md), read on the development period only:
# - inside a day, the contribution of the same hour of J grows with its value (the daily
#   shape repeats from one day to the next: the naive captures most of the ceiling, D32); if
#   not, hours are misaligned somewhere: look for the bug before going on. The same hour a
#   week before is reported too, without a rule (it matters for Saturday to Monday only);
# - the first driver shown must stay the same with the second training seed in at least
#   DRIVER_AGREEMENT_MIN of the cases, else the morning report shows families;
# - a family share that moves by more than METHOD_SHARE_GAP between path-dependent and
#   interventional SHAP makes the ranking depend on the method: documented, nothing changes.
ROW_ORDER_FEATURE = PRICE_SAME_HOUR_TODAY
REPORTED_ROW_ORDER_FEATURES = (PRICE_SAME_HOUR_TODAY, PRICE_SAME_HOUR_WEEK_AGO)
DRIVER_AGREEMENT_MIN = 0.8
METHOD_SHARE_GAP = 0.05
METHOD_DESCRIPTION = {
    "values": (
        "exact TreeSHAP (path-dependent) of the selected variant, refitted each month as in "
        "the prediction pipeline; computed by XGBoost (pred_contribs)"
    ),
    "decomposition": (
        "predicted price = mean price of J (anchor) + base value (expected gap of the "
        "month's model) + sum of the contributions"
    ),
    "day_level": "mean contribution over the hours of the delivered day (model A)",
    "within_day": "contribution of an hour minus its mean over the delivered day (model A)",
    "families": "sum of the contributions of the features of a family (exact)",
    "reading": (
        "SHAP describes the model, not the market: a contribution is associated with a "
        "prediction, it is not its cause"
    ),
}


# ==================================================
# Nodes: explanations of models A and B
# ==================================================


def _method_check(params: dict[str, Any]) -> dict[str, Any]:
    """Settings of the interventional SHAP check."""
    return {
        "day_of_month": params["method_check_day_of_month"],
        "background_rows": params["method_check_background_rows"],
    }


def explain_model_a(
    explanation_inputs_a: pd.DataFrame,
    predictions_model_a: pd.DataFrame,
    prediction_report: dict[str, Any],
    prediction_params: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exact SHAP values of every published prediction of model A (see `explain_task`).

    Args:
        explanation_inputs_a: Model A inputs (features of the hour, filtered regime of J).
        predictions_model_a: Hourly predictions of model A (prediction pipeline).
        prediction_report: Report of the prediction pipeline (variant selected).
        prediction_params: Prediction parameters (the refits use them unchanged).
        params: Interpretability parameters.

    Returns:
        shap_model_a (one row per predicted hour) and the checks, in memory.
    """
    return explain_task(
        explanation_inputs_a,
        predictions_model_a,
        prediction_report,
        TASK_A,
        prediction_params,
        _method_check(params),
    )


def explain_model_b(
    explanation_inputs_b: pd.DataFrame,
    predictions_model_b: pd.DataFrame,
    prediction_report: dict[str, Any],
    prediction_params: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exact SHAP values of every published prediction of model B (see `explain_task`).

    Returns:
        shap_model_b (index decision_day, horizon_days) and the checks, in memory.
    """
    return explain_task(
        explanation_inputs_b,
        predictions_model_b,
        prediction_report,
        TASK_B,
        prediction_params,
        _method_check(params),
    )


# ==================================================
# Importances
# ==================================================


def importance(values: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Mean absolute value of each column (EUR/MWh) and its share of their sum, largest first."""
    means = values.abs().mean()
    total = float(means.sum())
    ordered = means.sort_values(ascending=False, kind="stable")
    return {
        str(name): {
            "mean_abs_eur_mwh": rounded(value),
            "share": rounded(value / total, SHARE_DECIMALS) if total > 0 else None,
        }
        for name, value in ordered.items()
    }


def _first(summary: dict[str, Any], count: int) -> dict[str, Any]:
    """The first `count` entries of an ordered summary."""
    return dict(list(summary.items())[:count])


def day_means(values: pd.DataFrame, days: pd.Series) -> pd.DataFrame:
    """Mean of each column over the hours of each delivered day (index: day)."""
    return values.groupby(days.to_numpy()).mean()


def within_day(values: pd.DataFrame, days: pd.Series) -> pd.DataFrame:
    """Each value minus its mean over the hours of its delivered day."""
    means = values.groupby(days.to_numpy()).transform("mean")
    return values - means


def _features_and_families(values: pd.DataFrame, families: pd.DataFrame) -> dict[str, Any]:
    """Importance of the features and of the families, on the same rows."""
    return {"features": importance(values), "families": importance(families)}


def direction(contributions: pd.DataFrame, values: pd.DataFrame) -> dict[str, float | None]:
    """Spearman correlation between each feature's value and its contribution.

    Descriptive: a monotone tendency over the rows, which mixes the price levels of
    different years; the notebook shows the whole relation.
    """
    correlations = {}
    for feature in contributions.columns:
        feature_values = values[feature].astype(float)
        has_spread = feature_values.nunique() > 1 and contributions[feature].nunique() > 1
        correlation = (
            feature_values.corr(contributions[feature], method="spearman") if has_spread else None
        )
        correlations[feature] = rounded(correlation)
    return correlations


def grouped(
    contributions: pd.DataFrame, families: pd.DataFrame, keys: pd.Series
) -> dict[str, dict[str, Any]]:
    """For each group of rows: number of rows, every family and the main features."""
    summary = {}
    for key, rows in contributions.groupby(keys.to_numpy(), sort=True):
        summary[str(key)] = {
            "rows": len(rows),
            "families": importance(families.loc[rows.index]),
            "top_features": _first(importance(rows), TOP_FEATURES),
        }
    return summary


def families_by_month(families: pd.DataFrame, months: pd.Series) -> dict[str, dict[str, Any]]:
    """Mean absolute contribution of each family, month by month (EUR/MWh)."""
    means = families.abs().groupby(months.to_numpy(), sort=True).mean()
    return {
        str(month): {str(family): rounded(value) for family, value in row.items()}
        for month, row in means.iterrows()
    }


def _by_epoch(
    table: pd.DataFrame,
    contributions: pd.DataFrame,
    families: pd.DataFrame,
    values: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Main features, every family and the direction of each feature, per development epoch."""
    summary = {}
    for label, rows in development_epochs(table, params).items():
        index = rows.index
        summary[label] = {
            "rows": len(rows),
            "families": importance(families.loc[index]),
            "top_features": _first(importance(contributions.loc[index]), TOP_FEATURES),
            "direction": direction(contributions.loc[index], values.loc[index]),
        }
    return summary


def _day_views(
    table: pd.DataFrame, contributions: pd.DataFrame, families: pd.DataFrame
) -> dict[str, Any]:
    """Importances of the level and of the shape of the delivered days (model A)."""
    days = table[TARGET_DAY]
    return {
        "day_level_importance": _features_and_families(
            day_means(contributions, days), day_means(families, days)
        ),
        "within_day_importance": _features_and_families(
            within_day(contributions, days), within_day(families, days)
        ),
    }


def explanation_section(
    table: pd.DataFrame,
    values: pd.DataFrame,
    task: Task,
    params: dict[str, Any],
    development: bool,
) -> dict[str, Any]:
    """Everything the report says about the explanations of one model on one period.

    Args:
        table: SHAP rows of the period (model B flat: target_day and horizon_days columns).
        values: Feature values of the same rows, in the same order.
        task: TASK_A or TASK_B.
        params: Interpretability parameters.
        development: Whether the rows are those of the development period (epochs).

    Returns:
        A JSON-ready section.
    """
    if table.empty:
        return {"rows": 0}
    contributions = feature_contributions(table)
    families = family_sums(contributions, families_of(list(contributions.columns), params))
    gaps = table[PRED_SELECTED] - table[ANCHOR]
    section = {
        "rows": len(table),
        "mean_absolute_gap_eur_mwh": rounded(gaps.abs().mean()),
        "mean_base_value_eur_mwh": rounded(table[BASE_VALUE].mean()),
        "importance": _features_and_families(contributions, families),
    }
    if task.name == TASK_A.name:
        section.update(_day_views(table, contributions, families))
    section["direction"] = direction(contributions, values)
    breakdowns = ("regime", "year", "hour_block" if task.name == TASK_A.name else "horizon")
    for by in breakdowns:
        keys = group_keys(table, task, by, params)
        section[f"by_{by}"] = grouped(contributions, families, keys)
    if development:
        section["by_epoch"] = _by_epoch(table, contributions, families, values, params)
    months = group_keys(table, task, "month", params)
    section["families_by_month"] = families_by_month(families, months)
    return section


# ==================================================
# Rules of run 7
# ==================================================


def within_day_rank_correlation(
    values: pd.Series, contributions: pd.Series, days: pd.Series
) -> pd.Series:
    """Spearman correlation between a feature and its contribution inside each day (index: day).

    Missing when the feature or its contribution does not change within the day.
    """
    frame = pd.DataFrame(
        {"value": values.astype(float).to_numpy(), "contribution": contributions.to_numpy()},
    )
    day_keys = days.to_numpy()
    is_known = frame["value"].notna().to_numpy()
    known = frame[is_known]
    known_days = day_keys[is_known]
    ranks = known.groupby(known_days).rank()
    centered = ranks - ranks.groupby(known_days).transform("mean")
    covariance = (centered["value"] * centered["contribution"]).groupby(known_days).sum()
    value_spread = (centered["value"] ** 2).groupby(known_days).sum()
    contribution_spread = (centered["contribution"] ** 2).groupby(known_days).sum()
    denominator = np.sqrt(value_spread * contribution_spread)
    return covariance / denominator.where(denominator > 0)


def hour_alignment_rule(table: pd.DataFrame, values: pd.DataFrame) -> dict[str, Any]:
    """Rule 2: inside a day, the contribution of the same hour of J grows with its value
    (median of the daily Spearman correlations above 0); the same hour a week before is
    reported without a rule."""
    contributions = feature_contributions(table)
    days = table[TARGET_DAY]
    medians = {}
    for feature in REPORTED_ROW_ORDER_FEATURES:
        if feature not in contributions:
            continue
        correlations = within_day_rank_correlation(values[feature], contributions[feature], days)
        medians[feature] = rounded(correlations.median())
    checked = medians.get(ROW_ORDER_FEATURE)
    passes = checked is not None and checked > 0
    return {
        "median_within_day_rank_correlation": medians,
        "checked_feature": ROW_ORDER_FEATURE,
        "passes": passes,
        "investigate_bug": not passes,
    }


def _top_mask(values: np.ndarray, count: int) -> np.ndarray:
    """True for the `count` largest values of each row (ties: first column first)."""
    order = np.argsort(-values, axis=1, kind="stable")[:, :count]
    mask = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    return mask


def agreement(first: pd.DataFrame, second: pd.DataFrame, top: int) -> dict[str, float | None]:
    """How often the first driver (largest absolute contribution) is the same in both tables,
    and the mean overlap of their `top` drivers (intersection over union)."""
    if first.empty:
        return {"first_driver": None, "top_overlap": None}
    first_values = first.abs().to_numpy()
    second_values = second[first.columns].abs().to_numpy()
    is_same_first = np.argmax(first_values, axis=1) == np.argmax(second_values, axis=1)
    count = min(top, first.shape[1])
    first_top = _top_mask(first_values, count)
    second_top = _top_mask(second_values, count)
    overlap = (first_top & second_top).sum(axis=1) / (first_top | second_top).sum(axis=1)
    return {
        "first_driver": rounded(is_same_first.mean(), SHARE_DECIMALS),
        "top_overlap": rounded(overlap.mean(), SHARE_DECIMALS),
    }


def _levels_agreement(
    published: pd.DataFrame, second: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Agreement of the drivers between the two seeds, as features and as families."""
    top = params["top_drivers"]
    families = families_of(list(published.columns), params)
    return {
        FEATURE_LEVEL: agreement(published, second, top),
        FAMILY_LEVEL: agreement(
            family_sums(published, families), family_sums(second, families), top
        ),
    }


def _display_level(first_driver_agreements: list[float | None]) -> str:
    """Features when every measured agreement reaches DRIVER_AGREEMENT_MIN, else families."""
    known = [value for value in first_driver_agreements if value is not None]
    if not known or min(known) >= DRIVER_AGREEMENT_MIN:
        return FEATURE_LEVEL
    return FAMILY_LEVEL


def driver_stability(
    table_a: pd.DataFrame,
    checks_a: dict[str, Any],
    table_b: pd.DataFrame,
    checks_b: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Rule 3: are the drivers the morning report shows the same with the second seed?

    Level of a day (model A day means, model B rows) and shape of the hours (model A),
    on the development months refitted with both seeds.
    """
    second_a = feature_contributions(checks_a["second_seed"])
    published_a = feature_contributions(table_a).loc[second_a.index]
    days = table_a.loc[second_a.index, TARGET_DAY]
    second_b = feature_contributions(checks_b["second_seed"])
    published_b = feature_contributions(table_b).loc[second_b.index]
    agreements = {
        "model_a_day_level": _levels_agreement(
            day_means(published_a, days), day_means(second_a, days), params
        ),
        "model_a_hour_shape": _levels_agreement(
            within_day(published_a, days), within_day(second_a, days), params
        ),
        "model_b_day_level": _levels_agreement(published_b, second_b, params),
    }
    day_agreements = [
        agreements[name][FEATURE_LEVEL]["first_driver"]
        for name in ("model_a_day_level", "model_b_day_level")
    ]
    hour_agreement = agreements["model_a_hour_shape"][FEATURE_LEVEL]["first_driver"]
    recommended = {"day": _display_level(day_agreements), "hour": _display_level([hour_agreement])}
    configured = dict(params["morning_report"]["drivers"])
    return {
        "agreement": agreements,
        "limit": DRIVER_AGREEMENT_MIN,
        "recommended_drivers": recommended,
        "configured_drivers": configured,
        "change_configuration": recommended != configured,
    }


def _shares(values: pd.DataFrame) -> pd.Series:
    """Share of each column in the sum of the mean absolute values."""
    means = values.abs().mean()
    return means / means.sum()


def method_sensitivity(
    table: pd.DataFrame, interventional: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Rule 4: family shares with path-dependent and interventional SHAP, on the same rows."""
    if interventional.empty:
        return {"rows": 0, "largest_family_share_gap": None}
    path = feature_contributions(table).loc[interventional.index]
    other = feature_contributions(interventional)[path.columns]
    families = families_of(list(path.columns), params)
    path_shares = _shares(family_sums(path, families))
    other_shares = _shares(family_sums(other, families))
    gaps = (path_shares - other_shares).abs()
    rank_correlation = path.abs().mean().corr(other.abs().mean(), method="spearman")
    return {
        "rows": len(path),
        "family_shares": {
            "path_dependent": {
                name: rounded(value, SHARE_DECIMALS) for name, value in path_shares.items()
            },
            "interventional": {
                name: rounded(value, SHARE_DECIMALS) for name, value in other_shares.items()
            },
        },
        "largest_family_share_gap": rounded(gaps.max(), SHARE_DECIMALS),
        "feature_importance_rank_correlation": rounded(rank_correlation),
    }


def _exactness(checks: dict[str, Any]) -> dict[str, Any]:
    """Largest gaps of the exactness checks of one model (EUR/MWh)."""
    keys = (
        "max_refit_difference_eur_mwh",
        "max_reconstruction_error_eur_mwh",
        "second_seed_reconstruction_error_eur_mwh",
        "interventional_reconstruction_error_eur_mwh",
    )
    return {key: checks[key] for key in keys}


def rules_for_run_7(
    tables: dict[str, pd.DataFrame],
    checks: dict[str, dict[str, Any]],
    development_a: tuple[pd.DataFrame, pd.DataFrame],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Rules fixed before run 7, computed on the development period only.

    1. Exactness: the refits give back the published predictions and the decompositions
       add up to them (checked by the code, which stops otherwise).
    2. Hour alignment: inside a day, the same hour's contribution grows with its value.
    3. Driver stability: the first driver shown stays the same with the second seed.
    4. Attribution method: family shares barely move with interventional SHAP.
    """
    names = (TASK_A.name, TASK_B.name)
    sensitivity = {
        name: method_sensitivity(tables[name], checks[name]["interventional"], params)
        for name in names
    }
    gaps = [sensitivity[name]["largest_family_share_gap"] for name in names]
    depends_on_method = any(gap is not None and gap > METHOD_SHARE_GAP for gap in gaps)
    return {
        "1_exactness": {
            "tolerance_eur_mwh": EXACTNESS_TOLERANCE_EUR_MWH,
            **{name: _exactness(checks[name]) for name in names},
        },
        "2_hour_alignment": hour_alignment_rule(*development_a),
        "3_driver_stability": driver_stability(
            tables[TASK_A.name],
            checks[TASK_A.name],
            tables[TASK_B.name],
            checks[TASK_B.name],
            params,
        ),
        "4_attribution_method": {
            "limit": METHOD_SHARE_GAP,
            **sensitivity,
            "ranking_depends_on_method": depends_on_method,
        },
    }


# ==================================================
# Report
# ==================================================


def _flat_rows(
    table: pd.DataFrame, inputs: pd.DataFrame, task: Task
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP rows and the feature values of the same rows, in the same order.

    Model B is flattened: its decision day and horizon become columns.
    """
    features = list(feature_contributions(table).columns)
    values = indexed_inputs(inputs, task).reindex(table.index)[features]
    if task.name == TASK_A.name:
        return table, values
    return table.reset_index(), values.reset_index(drop=True)


def _coverage(table: pd.DataFrame, checks: dict[str, Any], params: dict[str, Any]) -> dict:
    """Explained rows and months of one model, by period."""
    is_development_row = is_development(table, params)
    refits = checks["refits"]
    return {
        "variant": checks["variant"],
        "features": checks["features"],
        "families": families_of(checks["features"], params),
        "explained_rows": len(table),
        "development_rows": int(is_development_row.sum()),
        "holdout_rows": int((~is_development_row).sum()),
        "rows_without_prediction": int(table[PRED_SELECTED].isna().sum()),
        "explained_months": len(refits),
        "first_month": refits[0]["month"] if refits else None,
        "last_month": refits[-1]["month"] if refits else None,
        "second_seed_rows": len(checks["second_seed"]),
        "interventional_rows": len(checks["interventional"]),
    }


def build_shap_report(
    shap_model_a: pd.DataFrame,
    shap_model_b: pd.DataFrame,
    explanation_inputs_a: pd.DataFrame,
    explanation_inputs_b: pd.DataFrame,
    shap_checks_a: dict[str, Any],
    shap_checks_b: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Report of the explanations, in two sections and the refit logs.

    - choices_before_holdout: the only section to read when choosing (rules of run 7);
      every row in it has a predicted day before holdout_start, explained by a model
      fitted on earlier days.
    - holdout_descriptive_only: the same figures on the reserved period, read after.

    Args:
        shap_model_a: First output of `explain_model_a`.
        shap_model_b: First output of `explain_model_b`.
        explanation_inputs_a: Model A inputs (feature values).
        explanation_inputs_b: Model B inputs (feature values).
        shap_checks_a: Second output of `explain_model_a`.
        shap_checks_b: Second output of `explain_model_b`.
        params: Interpretability parameters.

    Returns:
        A JSON-ready report.
    """
    tasks = {TASK_A.name: TASK_A, TASK_B.name: TASK_B}
    tables = {TASK_A.name: shap_model_a, TASK_B.name: shap_model_b}
    inputs = {TASK_A.name: explanation_inputs_a, TASK_B.name: explanation_inputs_b}
    checks = {TASK_A.name: shap_checks_a, TASK_B.name: shap_checks_b}
    development, holdout, coverage = {}, {}, {}
    development_a = (pd.DataFrame(), pd.DataFrame())
    for name, task in tasks.items():
        table, values = _flat_rows(tables[name], inputs[name], task)
        is_development_row = is_development(table, params)
        development_rows = (table[is_development_row], values[is_development_row])
        holdout_rows = (table[~is_development_row], values[~is_development_row])
        development[name] = explanation_section(*development_rows, task, params, development=True)
        holdout[name] = explanation_section(*holdout_rows, task, params, development=False)
        coverage[name] = _coverage(table, checks[name], params)
        if name == TASK_A.name:
            development_a = development_rows
    development["rules_for_run_7"] = rules_for_run_7(tables, checks, development_a, params)
    report = {
        "holdout_start": str(params["holdout_start"]),
        "regime_names": list(params["regime_names"]),
        "method": METHOD_DESCRIPTION,
        "coverage": coverage,
        "choices_before_holdout": development,
        "holdout_descriptive_only": holdout,
        "refits": {name: checks[name]["refits"] for name in tasks},
    }
    _log_report(report)
    return report


def _log_report(report: dict[str, Any]) -> None:
    """Main figures in the log; warnings on the rules of run 7 that ask for attention."""
    development = report["choices_before_holdout"]
    for name in (TASK_A.name, TASK_B.name):
        families = development[name].get("importance", {}).get("families", {})
        logger.info(
            "Before holdout, %s: mean absolute contribution by family (EUR/MWh) %s",
            name,
            {family: values["mean_abs_eur_mwh"] for family, values in families.items()},
        )
    rules = development["rules_for_run_7"]
    logger.info("Rules of run 7: %s", rules)
    if rules["2_hour_alignment"]["investigate_bug"]:
        logger.warning("Hour alignment rule failed: look for misaligned hours before going on")
    stability = rules["3_driver_stability"]
    if stability["change_configuration"]:
        logger.warning(
            "Driver stability: set morning_report.drivers to %s", stability["recommended_drivers"]
        )
    if rules["4_attribution_method"]["ranking_depends_on_method"]:
        logger.warning("Family shares depend on the attribution method: document it")
