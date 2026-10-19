"""Morning report: what the procurement team reads at 11:00 on the decision day J (D34).

Everything in it is known at 11:00 J (decisions.md, section 6), which a test checks
by scrambling the rest:
- market of J: filtered regime of J, prices of J (published on J-1), last gas and carbon prices;
- J+1 hour by hour (model A): selected prediction, calibrated interval, alert, rank,
  similar-day naive, and the SHAP explanation split in two exact parts: the level of
  the day (what moves J+1 away from J) and the shape (what places an hour above or
  below the rest of the day, the only thing a move within the day depends on);
- schedule of the flexible load: the optimizer on the selected prediction (rule 3 of
  run 6), where it differs from the naive's schedule and why;
- J+2 to J+4 (model B) with their drivers;
- yesterday's anomalies (model C): hours up to midnight only, grouped into events
  with the hours known at 11:00 J; an event that may continue into J is provisional;
- track record: backtest savings of the delivered days up to J (their prices were
  published on the day before).

The report is a replay of one past morning, chosen by decision_day. The JSON is the
content; the Markdown is a view of the JSON only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from energy_market_analyzer.pipelines.anomaly.nodes import (
    ANOMALY_SCORE,
    IS_ANOMALY,
    PRICE_PERCENTILE,
    TOP_DRIVER,
    TOP_DRIVER_PERCENTILE,
    assign_events,
)
from energy_market_analyzer.pipelines.backtest.nodes import (
    ENERGY_TOLERANCE_MWH,
    OPTIMIZED_NAIVE,
    OPTIMIZED_SELECTED,
    PERFECT_FORESIGHT,
    SAVINGS_TOLERANCE_EUR,
    Schedule,
    check_schedule,
    client_profile,
    day_savings,
    expected_hours,
    optimal_schedule,
    savings_column,
)
from energy_market_analyzer.pipelines.features.nodes import (
    CLEAN_SPARK_SPREAD_TODAY,
    DAY_OF_WEEK,
    DECISION_DAY,
    EUA_LAST,
    GAS_TO_LOAD,
    HORIZON_DAYS,
    HOUR,
    IS_HOLIDAY,
    IS_WEEKEND,
    LOAD_MEAN,
    MONTH,
    NET_IMPORT_MEAN,
    NUCLEAR_TO_LOAD,
    PRICE_MEAN_CHANGE_TODAY,
    PRICE_MEAN_WEEK_AGO,
    PRICE_SAME_HOUR_TODAY,
    PRICE_SAME_HOUR_WEEK_AGO,
    RESIDUAL_LOAD_MEAN,
    TODAY_PRICES,
    TTF_LAST,
    WIND_SOLAR_TO_LOAD,
    YESTERDAY_REALIZED,
)
from energy_market_analyzer.pipelines.ingestion.nodes import LOAD_FORECAST, ONE_DAY, PRICE
from energy_market_analyzer.pipelines.interpretability.explanations import (
    BASE_VALUE,
    DAY_CALENDAR_FAMILY,
    FUELS_FAMILY,
    HORIZON_FAMILY,
    HOUR_FAMILY,
    LOAD_FORECAST_FAMILY,
    PRED_SELECTED,
    PRICE_WEEK_AGO_FAMILY,
    PRICES_TODAY_FAMILY,
    REALIZED_YESTERDAY_FAMILY,
    REGIME_FAMILY,
    families_of,
    feature_contributions,
)
from energy_market_analyzer.pipelines.interpretability.nodes import FAMILY_LEVEL
from energy_market_analyzer.pipelines.prediction.evaluation import DAY_FORMAT, rounded
from energy_market_analyzer.pipelines.prediction.models import (
    ALERT_CONFIDENCE,
    ALERT_LEVEL,
    ALERT_LEVELS,
    ANCHOR,
    INTERVAL_LOWER,
    INTERVAL_UPPER,
    IS_CHEAPEST_HOUR,
    IS_MOST_EXPENSIVE_HOUR,
    NAIVE_SIMILAR_DAY,
    PROB_ABOVE_CRITICAL,
    PROB_ABOVE_HIGH,
    RANK_IN_DAY,
    TARGET_DAY,
    TASK_A,
    TASK_B,
    THRESHOLD_CRITICAL,
    THRESHOLD_HIGH,
    THRESHOLD_REGIME,
    TRAINING_END,
    prediction_column,
    regime_feature_columns,
)
from energy_market_analyzer.pipelines.regime.nodes import REGIME_FILTERED

logger = logging.getLogger(__name__)

PRED_NAIVE = prediction_column(NAIVE_SIMILAR_DAY)
DECISION_TIME = "11:00"  # decision time on J, market time (decisions.md, section 6)
ONE_HOUR = pd.Timedelta(hours=1)
HOUR_FORMAT = "%H:%M"
ZONE_FORMAT = "%Z"
LOCAL_FORMAT = "%Y-%m-%d %H:%M"
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
UNKNOWN = "unknown"
DEVELOPMENT = "development"
HOLDOUT = "holdout"
ALERT_ORDER = ALERT_LEVELS[1:]  # high and critical hours are listed apart
PRICE_DECIMALS = 2
POWER_DECIMALS = 3
PROBABILITY_DECIMALS = 3

# --- Labels for the readers (English, as the repository) ---
TODAY_LABELS = {
    "price_mean": ("mean price today", "EUR/MWh"),
    "price_min": ("lowest price today", "EUR/MWh"),
    "price_max": ("highest price today", "EUR/MWh"),
    "price_std": ("price spread today (std)", "EUR/MWh"),
}
YESTERDAY_LABELS = {
    LOAD_MEAN: ("load yesterday", "MW"),
    RESIDUAL_LOAD_MEAN: ("residual load yesterday", "MW"),
    NET_IMPORT_MEAN: ("net imports yesterday", "MW"),
    GAS_TO_LOAD: ("gas share of load yesterday", ""),
    NUCLEAR_TO_LOAD: ("nuclear share of load yesterday", ""),
    WIND_SOLAR_TO_LOAD: ("wind and solar share of load yesterday", ""),
}
FEATURE_LABELS = {
    HOUR: ("hour of day", ""),
    DAY_OF_WEEK: ("day of week", ""),
    IS_WEEKEND: ("weekend", ""),
    IS_HOLIDAY: ("public holiday", ""),
    MONTH: ("month", ""),
    PRICE_SAME_HOUR_TODAY: ("same hour today", "EUR/MWh"),
    PRICE_SAME_HOUR_WEEK_AGO: ("same hour a week before", "EUR/MWh"),
    PRICE_MEAN_WEEK_AGO: ("mean price of the day a week before", "EUR/MWh"),
    LOAD_FORECAST: ("load forecast", "MW"),
    PRICE_MEAN_CHANGE_TODAY: ("change of the mean price since yesterday", "EUR/MWh"),
    TTF_LAST: ("gas TTF, last close", "EUR/MWh"),
    EUA_LAST: ("carbon EUA, last price", "EUR/t"),
    CLEAN_SPARK_SPREAD_TODAY: ("clean spark spread today", "EUR/MWh"),
    HORIZON_DAYS: ("horizon", "days"),
    **{TODAY_PRICES[stat]: label for stat, label in TODAY_LABELS.items()},
    **{YESTERDAY_REALIZED[name]: label for name, label in YESTERDAY_LABELS.items()},
}
FAMILY_LABELS = {
    HOUR_FAMILY: "hour of day",
    DAY_CALENDAR_FAMILY: "day calendar",
    PRICES_TODAY_FAMILY: "prices of today",
    PRICE_WEEK_AGO_FAMILY: "price a week before",
    LOAD_FORECAST_FAMILY: "load forecast",
    REALIZED_YESTERDAY_FAMILY: "realized yesterday",
    FUELS_FAMILY: "fuels and spread",
    HORIZON_FAMILY: "horizon",
    REGIME_FAMILY: "regime of today",
}
INTEGER_FEATURES = (HOUR, MONTH, HORIZON_DAYS)
FLAG_FEATURES = (IS_WEEKEND, IS_HOLIDAY)
NOTES = (
    "Every figure is known at 11:00 on the decision day: the predictions, intervals and "
    "alerts of the selected models, the filtered regime of today, the prices of today and "
    "the realized data of yesterday.",
    "Drivers are SHAP contributions: they describe the model, not the market. A driver is "
    "associated with the forecast; it is not its cause.",
    "Each explanation adds up: mean price of tomorrow = mean price today + the model's "
    "expected gap + the drivers shown + other; an hour's gap to the day's mean = its drivers "
    "+ other. Figures are rounded.",
    "Naive: the same hour of today from Tuesday to Friday, of the same day a week before from "
    "Saturday to Monday; what the client does without the project.",
    "The expected saving is computed at the predicted prices. The saving actually made is "
    "known after delivery (track record).",
    "Anomalies are hours unusual for their regime (model C). An event marked provisional may "
    "go on into today.",
)


def labels_of(params: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Label and unit of every feature, regime column and family."""
    label_column, *probability_columns = regime_feature_columns(params)
    regime_labels = {label_column: ("filtered regime today", "")}
    for name, column in zip(params["regime_names"], probability_columns, strict=True):
        regime_labels[column] = (f"probability of the {name} regime today", "")
    family_labels = {family: (label, "") for family, label in FAMILY_LABELS.items()}
    return {**FEATURE_LABELS, **regime_labels, **family_labels}


# ==================================================
# Decision day and the rows read
# ==================================================


def complete_delivery_days(predictions_a: pd.DataFrame, timezone: str) -> pd.DatetimeIndex:
    """Delivered days whose every hour has a selected prediction."""
    is_known = predictions_a[PRED_SELECTED].notna()
    counts = is_known.groupby(predictions_a[TARGET_DAY]).sum()
    days = pd.DatetimeIndex(counts.index)
    expected = expected_hours(days, timezone)
    is_complete = counts.to_numpy() == expected.to_numpy()
    return days[is_complete]


def complete_decision_days_b(predictions_b: pd.DataFrame) -> pd.DatetimeIndex:
    """Decision days whose every horizon of model B has a selected prediction."""
    flat = predictions_b.reset_index()
    horizons = flat[HORIZON_DAYS].nunique()
    counts = flat[PRED_SELECTED].notna().groupby(flat[DECISION_DAY]).sum()
    is_complete = counts.to_numpy() == horizons
    return pd.DatetimeIndex(counts.index[is_complete])


def resolve_decision_day(
    predictions_a: pd.DataFrame, predictions_b: pd.DataFrame, params: dict[str, Any]
) -> pd.Timestamp:
    """Configured decision day, or the last one with a complete J+1 and every horizon of B.

    Raises:
        ValueError: If no day has both.
    """
    configured = params["morning_report"]["decision_day"]
    if configured is not None:
        return pd.Timestamp(str(configured)).normalize()
    delivery_days = complete_delivery_days(predictions_a, params["market_timezone"])
    candidates = (delivery_days - ONE_DAY).intersection(complete_decision_days_b(predictions_b))
    if candidates.empty:
        raise ValueError("No decision day has a complete J+1 and every horizon of model B")
    return candidates.max()


def delivery_rows(
    predictions_a: pd.DataFrame, delivery_day: pd.Timestamp, params: dict[str, Any]
) -> pd.DataFrame:
    """Predictions of the hours of the delivered day J+1, in time order.

    Raises:
        ValueError: If an hour of the day has no selected prediction.
    """
    is_delivery = (predictions_a[TARGET_DAY] == delivery_day).to_numpy()
    rows = predictions_a[is_delivery].sort_index()
    expected = int(
        expected_hours(pd.DatetimeIndex([delivery_day]), params["market_timezone"]).iloc[0]
    )
    known = int(rows[PRED_SELECTED].notna().sum())
    if len(rows) != expected or known != expected:
        raise ValueError(
            f"Delivery day {delivery_day.strftime(DAY_FORMAT)}: {known} of {expected} hours "
            f"predicted; choose another morning_report.decision_day"
        )
    return rows


def hour_labels(instants: pd.DatetimeIndex, timezone: str) -> list[str]:
    """Local hour of each instant ("18:00"); the zone is added when a label repeats
    (the 25-hour day of the autumn clock change: "02:00 CEST", "02:00 CET")."""
    local = instants.tz_convert(timezone)
    labels = pd.Index(local.strftime(HOUR_FORMAT))
    zones = local.strftime(ZONE_FORMAT)
    is_repeated = labels.duplicated(keep=False)
    return [
        f"{label} {zone}" if repeated else label
        for label, zone, repeated in zip(labels, zones, is_repeated, strict=True)
    ]


def _day_text(day: pd.Timestamp) -> str:
    """Calendar day as text."""
    return pd.Timestamp(day).strftime(DAY_FORMAT)


def _local_text(instant: pd.Timestamp, timezone: str) -> str:
    """Instant as local date and time."""
    return instant.tz_convert(timezone).strftime(LOCAL_FORMAT)


def _json_value(value: Any) -> Any:
    """Python value for the JSON report: float rounded, None when missing."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    return rounded(value, PRICE_DECIMALS)


# ==================================================
# Drivers
# ==================================================


def _shown(
    contributions: pd.Series, values: pd.Series, level: str, params: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Contributions and values at the display level: features, or families (no value)."""
    if level != FAMILY_LEVEL:
        return contributions, values
    families = families_of(list(contributions.index), params)
    sums = {family: contributions[members].sum() for family, members in families.items()}
    family_values = pd.Series(np.nan, index=list(sums))
    return pd.Series(sums), family_values


def _driver_value(name: str, value: Any, params: dict[str, Any]) -> Any:
    """Value of a driver for the JSON report; the filtered regime by its name."""
    if name == REGIME_FILTERED and value is not None and pd.notna(value):
        return params["regime_names"][int(value)]
    return _json_value(value)


def drivers(
    contributions: pd.Series, values: pd.Series, level: str, params: dict[str, Any]
) -> dict[str, Any]:
    """The top_drivers largest contributions (EUR/MWh), their values, and the sum of the rest.

    Args:
        contributions: One contribution per feature.
        values: Value of each feature (missing when it is not one value, e.g. over a day).
        level: "feature" or "family".
        params: Interpretability parameters.

    Returns:
        drivers (name, label, unit, value, contribution) and other_eur_mwh.
    """
    shown, shown_values = _shown(contributions, values, level, params)
    order = shown.abs().sort_values(ascending=False, kind="stable").index
    top = list(order[: params["top_drivers"]])
    labels = labels_of(params)
    entries = []
    for name in top:
        label, unit = labels.get(name, (name, ""))
        entries.append(
            {
                "name": name,
                "label": label,
                "unit": unit,
                "value": _driver_value(name, shown_values.get(name), params),
                "contribution_eur_mwh": rounded(shown[name], PRICE_DECIMALS),
            }
        )
    other = shown.drop(top).sum()
    return {"drivers": entries, "other_eur_mwh": rounded(other, PRICE_DECIMALS)}


def constant_values(features: pd.DataFrame) -> pd.Series:
    """Value of each feature over the rows; missing when it changes from row to row."""
    first = features.iloc[0]
    is_constant = features.nunique(dropna=False) <= 1
    return first.where(is_constant)


def regime_values(
    regimes: pd.DataFrame, decision_day: pd.Timestamp, params: dict[str, Any]
) -> pd.Series:
    """Filtered regime of J and its probabilities (known at 11:00 J; the smoothed regime,
    which looks at later days, is never read). Missing when J has no filtered regime."""
    return regimes.reindex([decision_day])[regime_feature_columns(params)].iloc[0]


def driver_values_a(
    features_model_a: pd.DataFrame,
    regimes: pd.DataFrame,
    index: pd.DatetimeIndex,
    decision_day: pd.Timestamp,
    explained: list[str],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Values of the features explained for the hours of J+1: the feature table and, for a
    variant that sees it (global, weighted, hybrid, average), the filtered regime of J.

    Raises:
        KeyError: If an explained feature is in neither table.
    """
    regime_of_j = regime_values(regimes, decision_day, params)
    hours = features_model_a.reindex(index)
    with_regime = hours.assign(**{name: regime_of_j[name] for name in regime_of_j.index})
    return with_regime[explained]


# ==================================================
# Sections
# ==================================================


def _header(
    rows_a: pd.DataFrame,
    rows_b: pd.DataFrame,
    decision_day: pd.Timestamp,
    prediction_report: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Decision and delivery days, period, and the models that predicted them."""
    delivery_day = decision_day + ONE_DAY
    is_holdout = delivery_day >= pd.Timestamp(str(params["holdout_start"]))
    choices = prediction_report["choices_before_holdout"]
    training_end_b = rows_b[TRAINING_END].dropna()
    return {
        "decision_day": _day_text(decision_day),
        "decision_weekday": WEEKDAYS[decision_day.dayofweek],
        "decision_time": DECISION_TIME,
        "market_timezone": params["market_timezone"],
        "delivery_day": _day_text(delivery_day),
        "delivery_weekday": WEEKDAYS[delivery_day.dayofweek],
        "delivery_hours": len(rows_a),
        "period": HOLDOUT if is_holdout else DEVELOPMENT,
        "model_a": {
            "variant": choices[TASK_A.name]["selection"]["selected"],
            "trained_on_days_up_to": _day_text(rows_a[TRAINING_END].iloc[0]),
        },
        "model_b": {
            "variant": choices[TASK_B.name]["selection"]["selected"],
            "trained_on_days_up_to": (
                _day_text(training_end_b.iloc[0]) if not training_end_b.empty else None
            ),
        },
    }


def _market(
    features_rows: pd.DataFrame,
    regimes: pd.DataFrame,
    decision_day: pd.Timestamp,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Filtered regime of J and its probabilities, prices of J, last fuel prices."""
    first = features_rows.iloc[0]
    label_column, *probability_columns = regime_feature_columns(params)
    regime_row = regimes.reindex([decision_day]).iloc[0]
    label = regime_row[label_column]
    names = params["regime_names"]
    return {
        "regime": UNKNOWN if pd.isna(label) else names[int(label)],
        "regime_probabilities": {
            name: rounded(regime_row[column], PROBABILITY_DECIMALS)
            for name, column in zip(names, probability_columns, strict=True)
        },
        **{
            f"{stat}_today_eur_mwh": _json_value(first[TODAY_PRICES[stat]]) for stat in TODAY_LABELS
        },
        "price_mean_change_today_eur_mwh": _json_value(first[PRICE_MEAN_CHANGE_TODAY]),
        "gas_ttf_last_close_eur_mwh": _json_value(first[TTF_LAST]),
        "carbon_eua_last_price_eur_t": _json_value(first[EUA_LAST]),
        "clean_spark_spread_today_eur_mwh": _json_value(first[CLEAN_SPARK_SPREAD_TODAY]),
    }


def _schedules(rows_a: pd.DataFrame, backtest_params: dict[str, Any]) -> dict[str, Any]:
    """Optimal schedules of the flexible load for the selected and for the naive prices."""
    client = client_profile(backtest_params)
    shift_cost = float(backtest_params["shift_cost_eur_mwh"])
    predicted = rows_a[PRED_SELECTED].to_numpy(dtype=float)
    model = optimal_schedule(predicted, client, shift_cost)
    check_schedule(model, client)
    naive_prices = rows_a[PRED_NAIVE].to_numpy(dtype=float)
    has_naive = not np.isnan(naive_prices).any()
    naive = optimal_schedule(naive_prices, client, shift_cost) if has_naive else None
    return {
        "client": client,
        "shift_cost": shift_cost,
        "model": model,
        "naive": naive,
        "expected_saving": day_savings(model, predicted, client, shift_cost),
    }


def _power(value: float) -> float:
    """Flexible load in MW, rounded; a solver's -0.0 (tolerance of the optimizer) becomes 0.0."""
    return float(np.round(value, POWER_DECIMALS)) + 0.0


def _shape(shap_rows: pd.DataFrame) -> pd.DataFrame:
    """Contribution of each hour minus its mean over the day: the shape of the day."""
    contributions = feature_contributions(shap_rows)
    return contributions - contributions.mean()


def _hour_entries(
    rows_a: pd.DataFrame,
    alert_rows: pd.DataFrame,
    shap_rows: pd.DataFrame,
    features_rows: pd.DataFrame,
    schedules: dict[str, Any],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """One entry per hour of J+1: prediction, interval, naive, alert, rank, flexible load
    and the drivers of its gap to the day's mean."""
    labels = hour_labels(rows_a.index, params["market_timezone"])
    shape = _shape(shap_rows)
    level = params["morning_report"]["drivers"]["hour"]
    day_mean = rows_a[PRED_SELECTED].mean()
    naive_load = schedules["naive"].flexible_load if schedules["naive"] is not None else None
    entries = []
    for position, instant in enumerate(rows_a.index):
        row = rows_a.loc[instant]
        alert = alert_rows.loc[instant]
        rank = alert[RANK_IN_DAY]
        entries.append(
            {
                "hour": labels[position],
                "start_utc": instant.isoformat(),
                "predicted_eur_mwh": rounded(row[PRED_SELECTED], PRICE_DECIMALS),
                "interval_80_eur_mwh": [
                    rounded(row[INTERVAL_LOWER], PRICE_DECIMALS),
                    rounded(row[INTERVAL_UPPER], PRICE_DECIMALS),
                ],
                "naive_eur_mwh": _json_value(row[PRED_NAIVE]),
                "alert_level": None if pd.isna(alert[ALERT_LEVEL]) else str(alert[ALERT_LEVEL]),
                "alert_confidence": (
                    None if pd.isna(alert[ALERT_CONFIDENCE]) else str(alert[ALERT_CONFIDENCE])
                ),
                "probability_above_high": rounded(alert[PROB_ABOVE_HIGH], PROBABILITY_DECIMALS),
                "probability_above_critical": rounded(
                    alert[PROB_ABOVE_CRITICAL], PROBABILITY_DECIMALS
                ),
                "rank_in_day": None if pd.isna(rank) else int(rank),
                "cheapest": _json_value(alert[IS_CHEAPEST_HOUR]),
                "most_expensive": _json_value(alert[IS_MOST_EXPENSIVE_HOUR]),
                "flexible_load_mw": {
                    "flat": _power(schedules["client"].flexible_mwh),
                    "model": _power(schedules["model"].flexible_load[position]),
                    "naive": _power(naive_load[position]) if naive_load is not None else None,
                },
                "gap_to_day_mean_eur_mwh": rounded(row[PRED_SELECTED] - day_mean, PRICE_DECIMALS),
                "shape": drivers(shape.loc[instant], features_rows.loc[instant], level, params),
            }
        )
    return entries


def _tomorrow(
    rows_a: pd.DataFrame,
    alert_rows: pd.DataFrame,
    shap_rows: pd.DataFrame,
    features_rows: pd.DataFrame,
    schedules: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """J+1: level of the day and its drivers, thresholds, every hour, the alerts."""
    contributions = feature_contributions(shap_rows)
    level = params["morning_report"]["drivers"]["day"]
    first_alert = alert_rows.iloc[0]
    hours = _hour_entries(rows_a, alert_rows, shap_rows, features_rows, schedules, params)
    alerts = [entry for entry in hours if entry["alert_level"] in ALERT_ORDER]
    return {
        "level": {
            "mean_predicted_price_eur_mwh": rounded(rows_a[PRED_SELECTED].mean(), PRICE_DECIMALS),
            "mean_price_today_eur_mwh": rounded(rows_a[ANCHOR].iloc[0], PRICE_DECIMALS),
            "base_value_eur_mwh": rounded(shap_rows[BASE_VALUE].mean(), PRICE_DECIMALS),
            **drivers(contributions.mean(), constant_values(features_rows), level, params),
        },
        "thresholds": {
            "regime_source": str(first_alert[THRESHOLD_REGIME]),
            "high_eur_mwh": rounded(first_alert[THRESHOLD_HIGH], PRICE_DECIMALS),
            "critical_eur_mwh": rounded(first_alert[THRESHOLD_CRITICAL], PRICE_DECIMALS),
        },
        "alerts": [entry["hour"] for entry in alerts],
        "hours": hours,
    }


def _differing_hour(entry: dict[str, Any]) -> dict[str, Any]:
    """An hour where the model's schedule differs from the naive's: loads, prices, drivers."""
    load = entry["flexible_load_mw"]
    return {
        "hour": entry["hour"],
        "model_mw": load["model"],
        "naive_mw": load["naive"],
        "predicted_eur_mwh": entry["predicted_eur_mwh"],
        "naive_eur_mwh": entry["naive_eur_mwh"],
        "gap_to_day_mean_eur_mwh": entry["gap_to_day_mean_eur_mwh"],
        "shape": entry["shape"],
    }


def _schedule_section(hours: list[dict[str, Any]], schedules: dict[str, Any]) -> dict[str, Any]:
    """Schedule of the flexible load, and the hours where it differs from the naive's."""
    client = schedules["client"]
    model: Schedule = schedules["model"]
    naive: Schedule | None = schedules["naive"]
    section = {
        "strategy": "optimal schedule for the selected prediction (rule 3 of run 6)",
        "average_flexible_power_mw": rounded(client.flexible_mwh, POWER_DECIMALS),
        "max_flexible_power_mw": rounded(client.max_flexible_mwh, POWER_DECIMALS),
        "window_hours": client.window_hours,
        "shift_cost_eur_mwh": schedules["shift_cost"],
        "moved_mwh": rounded(model.moved_mwh, POWER_DECIMALS),
        "expected_saving_eur": rounded(schedules["expected_saving"], PRICE_DECIMALS),
    }
    if naive is None:
        section["naive"] = {"available": False}
        return section
    differs = np.abs(model.flexible_load - naive.flexible_load) > ENERGY_TOLERANCE_MWH
    section["naive"] = {
        "available": True,
        "moved_mwh": rounded(naive.moved_mwh, POWER_DECIMALS),
        "differing_hours": [
            _differing_hour(hours[position]) for position in np.flatnonzero(differs)
        ],
    }
    return section


def _next_days(
    predictions_b: pd.DataFrame,
    shap_b: pd.DataFrame,
    features_b: pd.DataFrame,
    regimes: pd.DataFrame,
    decision_day: pd.Timestamp,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """J+2 to J+4 (model B): mean predicted price, interval and drivers of each day."""
    horizons = sorted(
        int(value) for value in predictions_b.index.get_level_values(HORIZON_DAYS).unique()
    )
    regime_of_j = regime_values(regimes, decision_day, params)
    level = params["morning_report"]["drivers"]["day"]
    entries = []
    for horizon in horizons:
        key = (decision_day, horizon)
        target_day = decision_day + pd.Timedelta(days=horizon)
        is_known = key in predictions_b.index and pd.notna(predictions_b.loc[key, PRED_SELECTED])
        entry = {
            "horizon_days": horizon,
            "target_day": _day_text(target_day),
            "available": is_known,
        }
        if is_known:
            row = predictions_b.loc[key]
            contributions = feature_contributions(shap_b.loc[[key]]).iloc[0]
            horizon_value = pd.Series({HORIZON_DAYS: horizon})
            all_values = pd.concat([features_b.loc[key], regime_of_j, horizon_value])
            values = all_values.reindex(contributions.index)
            entry.update(
                {
                    "predicted_eur_mwh": rounded(row[PRED_SELECTED], PRICE_DECIMALS),
                    "interval_80_eur_mwh": [
                        rounded(row[INTERVAL_LOWER], PRICE_DECIMALS),
                        rounded(row[INTERVAL_UPPER], PRICE_DECIMALS),
                    ],
                    "mean_price_today_eur_mwh": rounded(row[ANCHOR], PRICE_DECIMALS),
                    "base_value_eur_mwh": rounded(shap_b.loc[key, BASE_VALUE], PRICE_DECIMALS),
                    **drivers(contributions, values, level, params),
                }
            )
        entries.append(entry)
    return entries


def _event_entry(
    window: pd.DataFrame,
    events: pd.Series,
    event_id: int,
    prices: pd.Series,
    cutoff: pd.Timestamp,
    params: dict[str, Any],
    max_gap_hours: int,
) -> dict[str, Any]:
    """One anomaly event touching yesterday, from the hours known at 11:00 J."""
    timezone = params["market_timezone"]
    in_event = events.eq(event_id).fillna(False).to_numpy(dtype=bool)
    hours = window[in_event]
    first, last = hours.index.min(), hours.index.max()
    peak = hours[ANOMALY_SCORE].astype(float).idxmax()
    peak_row = hours.loc[peak]
    regime = peak_row[REGIME_FILTERED]
    may_continue = last + (max_gap_hours + 1) * ONE_HOUR >= cutoff
    return {
        "first_hour": _local_text(first, timezone),
        "last_hour": _local_text(last, timezone),
        "anomalous_hours": len(hours),
        "peak_hour": _local_text(peak, timezone),
        "peak_score": rounded(peak_row[ANOMALY_SCORE], PROBABILITY_DECIMALS),
        "peak_price_eur_mwh": _json_value(prices.get(peak)),
        "peak_price_percentile": _json_value(peak_row[PRICE_PERCENTILE]),
        "top_driver": None if pd.isna(peak_row[TOP_DRIVER]) else str(peak_row[TOP_DRIVER]),
        "top_driver_percentile": _json_value(peak_row[TOP_DRIVER_PERCENTILE]),
        "regime": UNKNOWN if pd.isna(regime) else params["regime_names"][int(regime)],
        "provisional": bool(may_continue),
        "may_have_started_earlier": bool(first == window.index[0]),
    }


def _anomalies(
    anomalies: pd.DataFrame,
    features_c: pd.DataFrame,
    decision_day: pd.Timestamp,
    params: dict[str, Any],
    anomaly_params: dict[str, Any],
) -> dict[str, Any]:
    """Anomalous hours of J-1, grouped into events with the hours known at 11:00 J.

    The hours of J are left out: their realized data are not known at 11:00 J. The events
    table of the anomaly pipeline looks at later hours: it is not read.
    """
    timezone = params["market_timezone"]
    lookback = params["morning_report"]["anomaly_lookback_days"]
    max_gap = anomaly_params["event_max_gap_hours"]
    cutoff = decision_day.tz_localize(timezone)
    yesterday_start = (decision_day - ONE_DAY).tz_localize(timezone)
    window_start = (decision_day - (lookback + 1) * ONE_DAY).tz_localize(timezone)
    is_window = (anomalies.index >= window_start) & (anomalies.index < cutoff)
    window = anomalies[is_window].sort_index()
    events = assign_events(window[IS_ANOMALY], np.zeros(len(window), dtype=int), max_gap)
    is_yesterday = np.asarray(window.index >= yesterday_start)
    flags = window.loc[is_yesterday, IS_ANOMALY]
    event_ids = events[is_yesterday].dropna().unique()
    prices = features_c[PRICE]
    return {
        "day": _day_text(decision_day - ONE_DAY),
        "hours": int(is_yesterday.sum()),
        "scored_hours": int(flags.notna().sum()),
        "anomalous_hours": int(flags.fillna(False).astype(bool).sum()),
        "unscored_hours": int(flags.isna().sum()),
        "events": [
            _event_entry(window, events, int(event_id), prices, cutoff, params, max_gap)
            for event_id in event_ids
        ],
    }


def _track_record(
    backtest_daily: pd.DataFrame, decision_day: pd.Timestamp, params: dict[str, Any]
) -> dict[str, Any]:
    """Backtest savings of the delivered days up to J (their prices were published the day
    before each of them)."""
    days = params["morning_report"]["track_record_days"]
    first_day = decision_day - (days - 1) * ONE_DAY
    is_window = (backtest_daily.index >= first_day) & (backtest_daily.index <= decision_day)
    rows = backtest_daily[is_window]
    names = {
        "model": OPTIMIZED_SELECTED,
        "naive": OPTIMIZED_NAIVE,
        "perfect_foresight": PERFECT_FORESIGHT,
    }
    totals = {key: float(rows[savings_column(name)].sum()) for key, name in names.items()}
    ceiling = totals["perfect_foresight"]
    losses = rows[savings_column(OPTIMIZED_SELECTED)] < -SAVINGS_TOLERANCE_EUR
    return {
        "first_day": _day_text(first_day),
        "last_day": _day_text(decision_day),
        "days_requested": days,
        "days_evaluated": len(rows),
        "savings_eur": {key: rounded(value, PRICE_DECIMALS) for key, value in totals.items()},
        "share_of_perfect_foresight": {
            key: rounded(totals[key] / ceiling, PROBABILITY_DECIMALS) if ceiling > 0 else None
            for key in ("model", "naive")
        },
        "loss_days_model": int(losses.sum()),
    }


# ==================================================
# Node
# ==================================================


def build_morning_report(
    predictions_model_a: pd.DataFrame,
    predictions_model_b: pd.DataFrame,
    alerts: pd.DataFrame,
    shap_model_a: pd.DataFrame,
    shap_model_b: pd.DataFrame,
    features_model_a: pd.DataFrame,
    features_model_b: pd.DataFrame,
    regimes: pd.DataFrame,
    anomalies: pd.DataFrame,
    features_model_c: pd.DataFrame,
    backtest_daily: pd.DataFrame,
    prediction_report: dict[str, Any],
    params: dict[str, Any],
    backtest_params: dict[str, Any],
    anomaly_params: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Morning report of the decision day J: JSON content and its Markdown view.

    Reads only persisted tables, so that a past morning can be replayed alone
    (`--nodes=morning_report_node`). Realized prices of J+1, realized_level, the anomalies
    of J and every later row are never read.

    Args:
        predictions_model_a: Hourly predictions of model A.
        predictions_model_b: Daily predictions of model B.
        alerts: Hourly alerts of model A.
        shap_model_a: SHAP table of model A.
        shap_model_b: SHAP table of model B.
        features_model_a: Feature table A (values of the drivers).
        features_model_b: Feature table B (values of the drivers).
        regimes: Daily regimes (filtered regime of J and its probabilities).
        anomalies: Hourly anomalies (model C).
        features_model_c: Table C (price of the anomalous hours).
        backtest_daily: Daily backtest savings (track record).
        prediction_report: Report of the prediction pipeline (variants).
        params: Interpretability parameters.
        backtest_params: Backtest parameters (client, shift cost).
        anomaly_params: Anomaly parameters (event_max_gap_hours).

    Returns:
        The report (JSON-ready) and its Markdown text.

    Raises:
        ValueError: If J+1 is not fully predicted.
    """
    decision_day = resolve_decision_day(predictions_model_a, predictions_model_b, params)
    delivery_day = decision_day + ONE_DAY
    rows_a = delivery_rows(predictions_model_a, delivery_day, params)
    index = rows_a.index
    explained = list(feature_contributions(shap_model_a).columns)
    features_rows = driver_values_a(
        features_model_a, regimes, index, decision_day, explained, params
    )
    shap_rows = shap_model_a.reindex(index)
    alert_rows = alerts.reindex(index)
    is_decision_b = predictions_model_b.index.get_level_values(DECISION_DAY) == decision_day
    rows_b = predictions_model_b[is_decision_b]
    schedules = _schedules(rows_a, backtest_params)
    tomorrow = _tomorrow(rows_a, alert_rows, shap_rows, features_rows, schedules, params)
    report = {
        "header": _header(rows_a, rows_b, decision_day, prediction_report, params),
        "market_today": _market(features_rows, regimes, decision_day, params),
        "tomorrow": tomorrow,
        "schedule": _schedule_section(tomorrow["hours"], schedules),
        "next_days": _next_days(
            rows_b, shap_model_b, features_model_b, regimes, decision_day, params
        ),
        "anomalies_yesterday": _anomalies(
            anomalies, features_model_c, decision_day, params, anomaly_params
        ),
        "track_record": _track_record(backtest_daily, decision_day, params),
        "notes": list(NOTES),
    }
    logger.info(
        "Morning report of %s: delivery %s, %d alerts, %.1f MWh moved",
        report["header"]["decision_day"],
        report["header"]["delivery_day"],
        len(tomorrow["alerts"]),
        schedules["model"].moved_mwh,
    )
    return report, render_markdown(report)


# ==================================================
# Markdown view
# ==================================================


def _number(value: float | None, decimals: int = 1) -> str:
    """Number with thousands separators; 'n/a' when missing."""
    if value is None:
        return "n/a"
    return f"{value:,.{decimals}f}"


def _signed(value: float | None) -> str:
    """Signed number with one decimal; 'n/a' when missing."""
    if value is None:
        return "n/a"
    return f"{value:+,.1f}"


def _value_text(driver: dict[str, Any]) -> str:
    """Value of a driver for the text, with its unit; empty when there is none."""
    value = driver["value"]
    if value is None:
        return ""
    if isinstance(value, str):
        return f" ({value})"
    if driver["name"] in INTEGER_FEATURES:
        return f" ({int(value)})"
    if driver["name"] in FLAG_FEATURES:
        return f" ({'yes' if value else 'no'})"
    if driver["name"] == DAY_OF_WEEK:
        return f" ({WEEKDAYS[int(value)]})"
    unit = f" {driver['unit']}" if driver["unit"] else ""
    decimals = 2 if abs(value) < 1 else 1
    return f" ({_number(value, decimals)}{unit})"


def drivers_text(explanation: dict[str, Any]) -> str:
    """Drivers and the rest, e.g. 'hour of day (18) +60.2; other +4.3'."""
    parts = [
        f"{driver['label']}{_value_text(driver)} {_signed(driver['contribution_eur_mwh'])}"
        for driver in explanation["drivers"]
    ]
    parts.append(f"other {_signed(explanation['other_eur_mwh'])}")
    return "; ".join(parts)


def _header_lines(report: dict[str, Any]) -> list[str]:
    """Title, delivery day and models."""
    header = report["header"]
    model_a, model_b = header["model_a"], header["model_b"]
    return [
        f"# Morning report: {header['decision_weekday']} {header['decision_day']}, "
        f"{header['decision_time']} ({header['market_timezone']})",
        "",
        f"Delivery: {header['delivery_weekday']} {header['delivery_day']} "
        f"({header['delivery_hours']} hours), {header['period']} period.  ",
        f"Models: A `{model_a['variant']}` trained on the days up to "
        f"{model_a['trained_on_days_up_to']}; B `{model_b['variant']}` trained on the days up "
        f"to {model_b['trained_on_days_up_to']}.",
        "",
    ]


def _market_lines(report: dict[str, Any]) -> list[str]:
    """Regime, prices and fuels of today."""
    market = report["market_today"]
    probabilities = " / ".join(
        f"{name} {_number(value, 2)}" for name, value in market["regime_probabilities"].items()
    )
    return [
        "## Market today",
        "",
        f"- Filtered regime: **{market['regime']}** ({probabilities})",
        f"- Prices today: mean {_number(market['price_mean_today_eur_mwh'])}, lowest "
        f"{_number(market['price_min_today_eur_mwh'])}, highest "
        f"{_number(market['price_max_today_eur_mwh'])} EUR/MWh; change of the mean since "
        f"yesterday {_signed(market['price_mean_change_today_eur_mwh'])}",
        f"- Fuels: gas TTF {_number(market['gas_ttf_last_close_eur_mwh'])} EUR/MWh, carbon EUA "
        f"{_number(market['carbon_eua_last_price_eur_t'])} EUR/t, clean spark spread "
        f"{_number(market['clean_spark_spread_today_eur_mwh'])} EUR/MWh",
        "",
    ]


def _alert_lines(tomorrow: dict[str, Any]) -> list[str]:
    """High and critical hours of J+1."""
    hours = {entry["hour"]: entry for entry in tomorrow["hours"]}
    if not tomorrow["alerts"]:
        return ["No high or critical hour.", ""]
    lines = []
    for label in tomorrow["alerts"]:
        entry = hours[label]
        lower, upper = entry["interval_80_eur_mwh"]
        lines.append(
            f"- **{label} {entry['alert_level']}**: {_number(entry['predicted_eur_mwh'])} EUR/MWh "
            f"({_number(lower)} to {_number(upper)}), confidence {entry['alert_confidence']}; "
            f"{_signed(entry['gap_to_day_mean_eur_mwh'])} on the day's mean: "
            f"{drivers_text(entry['shape'])}"
        )
    return [*lines, ""]


def _hour_table_lines(tomorrow: dict[str, Any]) -> list[str]:
    """Table of the hours of J+1."""
    lines = [
        "| Hour | Predicted | 80 % interval | Naive | Alert | Confidence | Rank | "
        "Flexible MW (model / naive) | Gap to the day's mean: drivers |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in tomorrow["hours"]:
        lower, upper = entry["interval_80_eur_mwh"]
        load = entry["flexible_load_mw"]
        rank = entry["rank_in_day"] if entry["rank_in_day"] is not None else "n/a"
        lines.append(
            f"| {entry['hour']} | {_number(entry['predicted_eur_mwh'])} | {_number(lower)} to "
            f"{_number(upper)} | {_number(entry['naive_eur_mwh'])} | {entry['alert_level']} | "
            f"{entry['alert_confidence']} | {rank} | {_number(load['model'], 2)} / "
            f"{_number(load['naive'], 2)} | {_signed(entry['gap_to_day_mean_eur_mwh'])}: "
            f"{drivers_text(entry['shape'])} |"
        )
    return [*lines, ""]


def _tomorrow_lines(report: dict[str, Any]) -> list[str]:
    """Level of J+1, thresholds, alerts and the hour table."""
    tomorrow = report["tomorrow"]
    level = tomorrow["level"]
    thresholds = tomorrow["thresholds"]
    return [
        "## Tomorrow (model A)",
        "",
        f"**Mean predicted price {_number(level['mean_predicted_price_eur_mwh'])} EUR/MWh** = "
        f"{_number(level['mean_price_today_eur_mwh'])} (mean price today) "
        f"{_signed(level['base_value_eur_mwh'])} (model's expected gap); "
        f"{drivers_text(level)}.",
        "",
        f"Alert thresholds (regime {thresholds['regime_source']}, previous 24 months): high "
        f"{_number(thresholds['high_eur_mwh'])}, critical "
        f"{_number(thresholds['critical_eur_mwh'])} EUR/MWh.",
        "",
        "### Alerts",
        "",
        *_alert_lines(tomorrow),
        "### Hour by hour",
        "",
        *_hour_table_lines(tomorrow),
    ]


def _schedule_lines(report: dict[str, Any]) -> list[str]:
    """Schedule of the flexible load and the hours where it differs from the naive's."""
    schedule = report["schedule"]
    naive = schedule["naive"]
    lines = [
        "## Flexible load schedule",
        "",
        f"Optimizer on the model's prices: {_number(schedule['moved_mwh'], 2)} MWh moved, expected "
        f"saving {_number(schedule['expected_saving_eur'], 2)} EUR at the predicted prices (net of "
        f"{_number(schedule['shift_cost_eur_mwh'])} EUR/MWh moved). Flexible power "
        f"{_number(schedule['average_flexible_power_mw'], 2)} MW on average, at most "
        f"{_number(schedule['max_flexible_power_mw'], 2)} MW; each MWh moves "
        f"{schedule['window_hours']} hours at most.",
        "",
    ]
    if not naive["available"]:
        return [*lines, "Naive schedule: not available (an hour of the naive is missing).", ""]
    differing = naive["differing_hours"]
    if not differing:
        return [*lines, "Same schedule as the naive's.", ""]
    rows = [
        f"| {hour['hour']} | {_number(hour['model_mw'], 2)} | {_number(hour['naive_mw'], 2)} | "
        f"{_number(hour['predicted_eur_mwh'])} | {_number(hour['naive_eur_mwh'])} | "
        f"{_signed(hour['gap_to_day_mean_eur_mwh'])}: {drivers_text(hour['shape'])} |"
        for hour in differing
    ]
    return [
        *lines,
        f"Hours where it differs from the naive schedule ({_number(naive['moved_mwh'], 2)} MWh "
        f"moved by the naive):",
        "",
        "| Hour | Model MW | Naive MW | Predicted | Naive | Gap to the day's mean: drivers |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def _next_days_lines(report: dict[str, Any]) -> list[str]:
    """Days J+2 to J+4 (model B)."""
    lines = [
        "## Next days (model B)",
        "",
        "| Day | Horizon | Mean predicted | 80 % interval | Drivers (vs today's mean) |",
        "|---|---|---|---|---|",
    ]
    for entry in report["next_days"]:
        if not entry["available"]:
            lines.append(f"| {entry['target_day']} | {entry['horizon_days']} | not available | | |")
            continue
        lower, upper = entry["interval_80_eur_mwh"]
        lines.append(
            f"| {entry['target_day']} | {entry['horizon_days']} | "
            f"{_number(entry['predicted_eur_mwh'])} | {_number(lower)} to {_number(upper)} | "
            f"{_signed(entry['base_value_eur_mwh'])} expected gap; {drivers_text(entry)} |"
        )
    return [*lines, ""]


def _anomaly_lines(report: dict[str, Any]) -> list[str]:
    """Anomalous hours and events of yesterday."""
    anomalies = report["anomalies_yesterday"]
    lines = [
        "## Yesterday's anomalies (model C)",
        "",
        f"{anomalies['day']}: {anomalies['scored_hours']} hours scored, "
        f"{anomalies['anomalous_hours']} anomalous, {anomalies['unscored_hours']} not scored.",
        "",
    ]
    for event in anomalies["events"]:
        flags = [
            name
            for name, is_set in (
                ("provisional", event["provisional"]),
                ("may have started before the window", event["may_have_started_earlier"]),
            )
            if is_set
        ]
        status = f", {', '.join(flags)}" if flags else ""
        lines.append(
            f"- Event {event['first_hour']} to {event['last_hour']} "
            f"({event['anomalous_hours']} anomalous hours{status}): peak {event['peak_hour']}, "
            f"price {_number(event['peak_price_eur_mwh'])} EUR/MWh (percentile "
            f"{_number(event['peak_price_percentile'])} of the {event['regime']} regime), "
            f"rarest input {event['top_driver']} (percentile "
            f"{_number(event['top_driver_percentile'])})"
        )
    return [*lines, ""] if anomalies["events"] else lines


def _track_record_lines(report: dict[str, Any]) -> list[str]:
    """Backtest savings of the last delivered days."""
    record = report["track_record"]
    savings = record["savings_eur"]
    shares = record["share_of_perfect_foresight"]
    return [
        f"## Track record ({record['first_day']} to {record['last_day']})",
        "",
        f"{record['days_evaluated']} of {record['days_requested']} days evaluated: model "
        f"{_number(savings['model'], 0)} EUR, naive {_number(savings['naive'], 0)} EUR, perfect "
        f"foresight {_number(savings['perfect_foresight'], 0)} EUR (model "
        f"{_number(shares['model'], 3)}, naive {_number(shares['naive'], 3)} of the ceiling); "
        f"{record['loss_days_model']} days with a loss.",
        "",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    """Markdown view of the morning report (built from the JSON content only)."""
    lines = [
        *_header_lines(report),
        *_market_lines(report),
        *_tomorrow_lines(report),
        *_schedule_lines(report),
        *_next_days_lines(report),
        *_anomaly_lines(report),
        *_track_record_lines(report),
        "## How to read this report",
        "",
        *(f"- {note}" for note in report["notes"]),
    ]
    return "\n".join(lines) + "\n"
