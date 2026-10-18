"""Anomaly pipeline nodes: hours unusual for their market regime, with Isolation Forests.

For each hour from the first scored month on, an anomaly score says how unusual
the hour is for the filtered regime of its day (decisions D21 to D24):

- one Isolation Forest per regime, refitted on the first day of each month in
  Brussels on the previous hours of that regime only (all of them, or those of
  the last training_window_months months), each hour with the filtered regime
  known on its day (never the smoothed regime);
- the threshold is the (1 - contamination) quantile of the training scores;
- a regime with too few training hours is scored by the closest regime in
  volatility order that has enough, the more volatile one on a tie;
- the driver of an anomaly is the feature whose value is the rarest among the
  training hours of its model (percentile), the first of the list on a tie.

The score of hour t only uses the hours before the first day of its month, the
values of hour t (known at the end of the hour) and the filtered regime of its
day (known at 12:45 the day before). Events group anomalous hours close in
time: they look at the following hours, so they are descriptive only.

Retrospective analysis (model C): nothing here is an input of models A and B.

Reference:
    Liu, F. T., Ting, K. M. and Zhou, Z.-H. (2008). Isolation forest. ICDM 2008, 413-422.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from energy_market_analyzer.pipelines.features.nodes import DAY_CALENDAR, HOUR, market_days
from energy_market_analyzer.pipelines.ingestion.nodes import PRICE, UTC_ZONE
from energy_market_analyzer.pipelines.regime.nodes import LABEL_DTYPE, REGIME_FILTERED

logger = logging.getLogger(__name__)

# --- Hourly anomalies table ---
ANOMALY_MODEL = "anomaly_model"
ANOMALY_SCORE = "anomaly_score"
ANOMALY_THRESHOLD = "anomaly_threshold"
IS_ANOMALY = "is_anomaly"
PRICE_PERCENTILE = "price_percentile"
TOP_DRIVER = "top_driver"
TOP_DRIVER_PERCENTILE = "top_driver_percentile"
EVENT_ID = "event_id"
SCORE_COLUMNS = (
    REGIME_FILTERED,
    ANOMALY_MODEL,
    ANOMALY_SCORE,
    ANOMALY_THRESHOLD,
    IS_ANOMALY,
    PRICE_PERCENTILE,
    TOP_DRIVER,
    TOP_DRIVER_PERCENTILE,
)
ANOMALY_COLUMNS = (*SCORE_COLUMNS, EVENT_ID)

# --- Events table: one row per event ---
FIRST_HOUR = "first_hour"
LAST_HOUR = "last_hour"
DURATION_HOURS = "duration_hours"
ANOMALOUS_HOURS = "anomalous_hours"
PEAK_HOUR = "peak_hour"
PEAK_SCORE = "peak_score"
PEAK_PRICE = "peak_price_eur_mwh"
PEAK_PRICE_PERCENTILE = "peak_price_percentile"
PEAK_REGIME = "peak_regime"
EVENT_COLUMNS = (
    FIRST_HOUR,
    LAST_HOUR,
    DURATION_HOURS,
    ANOMALOUS_HOURS,
    PEAK_HOUR,
    PEAK_SCORE,
    PEAK_PRICE,
    PEAK_PRICE_PERCENTILE,
    PEAK_REGIME,
    TOP_DRIVER,
    TOP_DRIVER_PERCENTILE,
)

# --- Types and units ---
FLAG_DTYPE = "boolean"
TEXT_DTYPE = "string"
EVENT_DTYPE = "Int32"
ONE_HOUR = pd.Timedelta(hours=1)
HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168
# Percentiles go from 0 (below every training value) to 100 (above every training value).
FULL_RANGE = 100.0
MEDIAN_PERCENTILE = 50.0
CALENDAR_COLUMNS = (HOUR, *DAY_CALENDAR)
GROUP = "model_group"  # column used while fitting: regime label, or one group for all regimes
MONTH_FORMAT = "%Y-%m"
EXPANDING = "expanding"  # report label of the window that keeps every past hour
LOCAL_TIME_FORMAT = "%Y-%m-%d %H:%M"

# --- Report ---
SHARE_DECIMALS = 4
SCORE_DECIMALS = 4
PERCENTILE_DECIMALS = 2
PRICE_DECIMALS = 2
DURATION_QUANTILE = 0.9
# Rules fixed before run 4 (decisions.md), read on the development period only:
# - one hour of the day holding more than this multiple of its fair share (1/24)
#   of the anomalies inside the training range means the anomalies follow the
#   daily shape: add the hour;
# - a regime-year flagged outside [contamination / factor, contamination x factor],
#   with at least min_training_hours scored hours, means the reference may drift:
#   a rolling training window is then compared on the development period, and the
#   window with the lower mean calibration error is kept.
HOUR_CONCENTRATION_FACTOR = 2.0
CALIBRATION_FACTOR = 2.0


@dataclass(frozen=True)
class Reference:
    """One fitted Isolation Forest and what scoring an hour needs from its training hours.

    Attributes:
        model: The fitted Isolation Forest.
        thresholds: Score above which an hour is flagged, per contamination level.
        sorted_values: Sorted training values of each input column, for the percentiles.
        training_hours: Number of training hours.
    """

    model: IsolationForest
    thresholds: dict[float, float]
    sorted_values: dict[str, np.ndarray]
    training_hours: int


# ==================================================
# Inputs
# ==================================================


def input_columns(features: list[str]) -> list[str]:
    """The features, plus the price when it is not one of them (its percentile is reported)."""
    return list(dict.fromkeys([*features, PRICE]))


def _require_table_c(features_model_c: pd.DataFrame, features: list[str]) -> None:
    """Raise ValueError unless table C is hourly in UTC and holds every feature, none a calendar."""
    index = features_model_c.index
    is_utc = isinstance(index, pd.DatetimeIndex) and str(index.tz) == UTC_ZONE
    if not is_utc:
        raise ValueError("features_model_c: index must be a DatetimeIndex in UTC")
    steps = index[1:] - index[:-1]
    is_continuous = bool((steps == ONE_HOUR).all())
    if not is_continuous:
        raise ValueError("features_model_c: index must be a continuous hourly grid")
    calendar_features = sorted(set(features) & set(CALENDAR_COLUMNS))
    if calendar_features:
        raise ValueError(
            f"anomaly.features: calendar columns are not Isolation Forest inputs "
            f"(hour 23 would be far from hour 0): {calendar_features}"
        )
    missing_columns = [name for name in input_columns(features) if name not in features_model_c]
    if missing_columns:
        raise ValueError(f"anomaly.features: columns {missing_columns} are not in features_model_c")


def build_anomaly_inputs(
    features_model_c: pd.DataFrame, regimes: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Hourly inputs of the Isolation Forests: the features and the filtered regime of the day.

    Known: the price of hour t the day before (12:45), its realized values at the
    end of the hour, the filtered regime of its day at 12:45 the day before. The
    smoothed regime is never read.

    Args:
        features_model_c: Hourly table C (features pipeline).
        regimes: Daily regimes table (regime pipeline).
        params: Anomaly parameters (features, market_timezone).

    Returns:
        One row per hour of table C (UTC index): the input columns, then regime_filtered.

    Raises:
        ValueError: If table C or the regimes table is not as expected.
    """
    features = params["features"]
    _require_table_c(features_model_c, features)
    if REGIME_FILTERED not in regimes.columns:
        raise ValueError(f"regimes: missing column {REGIME_FILTERED}, re-run the regime pipeline")
    days = market_days(features_model_c.index, params["market_timezone"])
    day_regimes = regimes[REGIME_FILTERED].reindex(days)
    hour_regimes = day_regimes.set_axis(features_model_c.index)
    inputs = features_model_c[input_columns(features)]
    labelled = inputs.assign(**{REGIME_FILTERED: hour_regimes})
    logger.info(
        "Anomaly inputs: %d hours, %d with a filtered regime",
        len(labelled),
        int(hour_regimes.notna().sum()),
    )
    return labelled


# ==================================================
# One model: fit, score, drivers
# ==================================================


def percentile(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value among the reference: 0 below all, 100 above all.

    Args:
        sorted_reference: Training values, sorted.
        values: Values to place.

    Returns:
        Percentiles in [0, 100]; exactly 0 or 100 only outside the training range.
    """
    below = np.searchsorted(sorted_reference, values, side="left")
    not_above = np.searchsorted(sorted_reference, values, side="right")
    return FULL_RANGE * (below + not_above) / (2 * len(sorted_reference))


def fit_reference(training: pd.DataFrame, params: dict[str, Any], seed: int) -> Reference:
    """Isolation Forest fitted on `training`, its thresholds and the sorted training values.

    No scaling: each split is drawn uniformly between the minimum and the maximum
    of a feature, so the forest does not depend on the scale of the features.

    Args:
        training: Complete training hours of one model (input columns).
        params: Anomaly parameters (features, n_estimators, max_samples, contamination_levels).
        seed: Random seed of the forest.

    Returns:
        The reference used to score the hours of the month.
    """
    features = params["features"]
    model = IsolationForest(
        n_estimators=params["n_estimators"], max_samples=params["max_samples"], random_state=seed
    )
    training_values = training[features].to_numpy()
    model.fit(training_values)
    # Anomaly score of Liu et al.: in (0, 1], higher = more anomalous.
    training_scores = -model.score_samples(training_values)
    thresholds = {
        level: float(np.quantile(training_scores, 1.0 - level))
        for level in params["contamination_levels"]
    }
    sorted_values = {
        column: np.sort(training[column].to_numpy()) for column in input_columns(features)
    }
    return Reference(model, thresholds, sorted_values, len(training))


def _top_drivers(
    percentiles: np.ndarray, features: list[str], is_anomaly: np.ndarray
) -> dict[str, Any]:
    """Rarest feature of each anomalous hour and its percentile; missing for normal hours.

    Rarity is the distance to the nearest end of the training distribution. On a
    tie (e.g. price and spread both above every training value), the first
    feature of the list wins: the list is ordered from the most readable.
    """
    rarity = np.minimum(percentiles, FULL_RANGE - percentiles)
    positions = rarity.argmin(axis=1)
    names = np.asarray(features, dtype=object)[positions]
    chosen_percentiles = percentiles[np.arange(len(positions)), positions]
    return {
        TOP_DRIVER: pd.array(np.where(is_anomaly, names, None), dtype=TEXT_DTYPE),
        TOP_DRIVER_PERCENTILE: np.where(is_anomaly, chosen_percentiles, np.nan),
    }


def score_hours(
    reference: Reference, hours: pd.DataFrame, model_group: int, params: dict[str, Any]
) -> pd.DataFrame:
    """Score, threshold, flag, price percentile and top driver of each hour, with one model.

    Each hour is scored on its own: nothing is computed over the hours scored together.

    Args:
        reference: Output of `fit_reference`.
        hours: Hours to score (input columns, no missing feature).
        model_group: Group (regime label) of the model, stored with each hour.
        params: Anomaly parameters (features, contamination).

    Returns:
        One row per hour, the score columns without regime_filtered.
    """
    features = params["features"]
    scores = -reference.model.score_samples(hours[features].to_numpy())
    threshold = reference.thresholds[params["contamination"]]
    is_anomaly = scores > threshold
    percentiles = np.column_stack(
        [percentile(reference.sorted_values[name], hours[name].to_numpy()) for name in features]
    )
    price_percentiles = percentile(reference.sorted_values[PRICE], hours[PRICE].to_numpy())
    columns = {
        ANOMALY_MODEL: model_group,
        ANOMALY_SCORE: scores,
        ANOMALY_THRESHOLD: threshold,
        IS_ANOMALY: is_anomaly,
        PRICE_PERCENTILE: price_percentiles,
        **_top_drivers(percentiles, features, is_anomaly),
    }
    return pd.DataFrame(columns, index=hours.index)


# ==================================================
# Monthly refit on the past
# ==================================================


def training_hours_mask(
    days: pd.DatetimeIndex, month_start: pd.Timestamp, window_months: int | None
) -> np.ndarray:
    """True for the hours of the days before `month_start`, within the window if one is set.

    Args:
        days: Calendar day (Brussels) of each complete hour.
        month_start: First day of the scored month.
        window_months: Months of past hours kept; None keeps every past hour.

    Returns:
        Boolean mask over `days`.
    """
    is_past = days < month_start
    if window_months is None:
        return np.asarray(is_past)
    window_start = month_start - pd.DateOffset(months=window_months)
    is_in_window = days >= window_start
    return np.asarray(is_past & is_in_window)


def scored_months(
    days: pd.DatetimeIndex, regime_labels: pd.Series, min_train_months: int
) -> list[pd.Timestamp]:
    """First day of each scored month: min_train_months after the month of the first labelled day.

    Raises:
        ValueError: If no hour has a filtered regime, or no month is left to score.
    """
    is_labelled = regime_labels.notna().to_numpy()
    if not is_labelled.any():
        raise ValueError("No hour has a filtered regime: run the regime pipeline first")
    first_month = days[is_labelled][0].to_period("M") + min_train_months
    months = days.to_period("M").unique()
    month_starts = [month.start_time for month in months if month >= first_month]
    if not month_starts:
        raise ValueError(f"No month left to score after {min_train_months} months of regimes")
    return month_starts


def closest_group(group: int, eligible: list[int]) -> int | None:
    """Group whose model scores `group`: itself if eligible, else the closest in volatility order.

    On a tie (a middle regime between two eligible ones), the more volatile one:
    its wider reference flags less, and in doubt the pipeline flags less.
    """
    if not eligible:
        return None
    return min(eligible, key=lambda candidate: (abs(candidate - group), -candidate))


def _models_of_groups(
    training: pd.DataFrame, n_groups: int, min_hours: int
) -> dict[int, int | None]:
    """Model group used for each group, from the number of training hours of each group."""
    counts = training[GROUP].value_counts()
    eligible = [group for group in range(n_groups) if counts.get(group, 0) >= min_hours]
    return {group: closest_group(group, eligible) for group in range(n_groups)}


def _model_record(
    model_group: int, reference: Reference, model_scores: pd.DataFrame
) -> dict[str, Any]:
    """One line of the refit log: training size, thresholds, hours scored and flagged."""
    return {
        "model": model_group,
        "training_hours": reference.training_hours,
        "thresholds": {str(level): value for level, value in reference.thresholds.items()},
        "scored_hours": len(model_scores),
        "flagged_hours": int(model_scores[IS_ANOMALY].sum()),
    }


def _score_month(
    training: pd.DataFrame,
    month_hours: pd.DataFrame,
    n_groups: int,
    params: dict[str, Any],
    seed: int,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    """Fit the models the month needs on the training hours, score the month's hours.

    The choice of each model only depends on the training hours, never on the
    hours of the month.
    """
    models_of = _models_of_groups(training, n_groups, params["min_training_hours"])
    scores_by_model: dict[int, list[pd.DataFrame]] = {}
    references: dict[int, Reference] = {}
    for group in sorted(int(group) for group in month_hours[GROUP].unique()):
        model = models_of[group]
        if model is None:
            continue
        if model not in references:
            model_training = training[training[GROUP] == model]
            references[model] = fit_reference(model_training, params, seed)
        group_hours = month_hours[month_hours[GROUP] == group]
        group_scores = score_hours(references[model], group_hours, model, params)
        scores_by_model.setdefault(model, []).append(group_scores)
    scores = [table for tables in scores_by_model.values() for table in tables]
    records = [
        _model_record(model, references[model], pd.concat(tables))
        for model, tables in sorted(scores_by_model.items())
    ]
    return scores, records


def score_months(
    inputs: pd.DataFrame,
    groups: pd.Series,
    n_groups: int,
    params: dict[str, Any],
    seed: int,
    window_months: int | None,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    """Monthly refit on the past and scoring, one Isolation Forest per group of hours.

    On the first day of each scored month (Brussels), each group with at least
    min_training_hours complete hours before that day (within the window) gets
    its model; the hours of the month are scored by the model of their group,
    or of the closest one.

    Args:
        inputs: Output of `build_anomaly_inputs`.
        groups: Group of each hour, 0 to n_groups - 1 (the regime label, or 0 for
            a single model); missing = not scored.
        n_groups: Number of groups.
        params: Anomaly parameters.
        seed: Random seed of the forests.
        window_months: Months of past hours used for training; None = all of them.

    Returns:
        Scores of each (month, group), and one record per fitted model.
    """
    columns = input_columns(params["features"])
    days = market_days(inputs.index, params["market_timezone"])
    is_complete = (inputs[columns].notna().all(axis=1) & groups.notna()).to_numpy()
    complete = inputs[is_complete].assign(**{GROUP: groups[is_complete].astype(int)})
    complete_days = days[is_complete]
    month_starts = scored_months(days, inputs[REGIME_FILTERED], params["min_train_months"])
    all_scores, records = [], []
    for month_start in month_starts:
        next_month_start = month_start + pd.offsets.MonthBegin(1)
        is_training = training_hours_mask(complete_days, month_start, window_months)
        is_month = np.asarray((complete_days >= month_start) & (complete_days < next_month_start))
        scores, month_records = _score_month(
            complete[is_training], complete[is_month], n_groups, params, seed
        )
        all_scores.extend(scores)
        month = month_start.strftime(MONTH_FORMAT)
        records.extend({"month": month, **record} for record in month_records)
    return all_scores, records


def score_table(inputs: pd.DataFrame, scores: list[pd.DataFrame]) -> pd.DataFrame:
    """All hours of `inputs` with their scores; missing values for the unscored hours."""
    if scores:
        scored = pd.concat(scores)
    else:
        scored = pd.DataFrame(columns=[name for name in SCORE_COLUMNS if name != REGIME_FILTERED])
    all_hours = scored.reindex(inputs.index)
    typed = all_hours.astype(
        {
            ANOMALY_MODEL: LABEL_DTYPE,
            ANOMALY_SCORE: float,
            ANOMALY_THRESHOLD: float,
            IS_ANOMALY: FLAG_DTYPE,
            PRICE_PERCENTILE: float,
            TOP_DRIVER: TEXT_DTYPE,
            TOP_DRIVER_PERCENTILE: float,
        }
    )
    with_regime = typed.assign(**{REGIME_FILTERED: inputs[REGIME_FILTERED]})
    return with_regime[list(SCORE_COLUMNS)]


def detect_anomalies(
    anomaly_inputs: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Anomaly score of each hour: Isolation Forest of its filtered regime, refitted each month.

    Known: at the end of the hour (its realized values); the models only use the
    hours before the first day of the month. The refits go on through the
    reserved period with the same settings (D9).

    Args:
        anomaly_inputs: Output of `build_anomaly_inputs`.
        params: Anomaly parameters.

    Returns:
        The hourly score table (SCORE_COLUMNS, missing for unscored hours) and one
        record per month and fitted model.
    """
    regime_names = params["regime_names"]
    regimes = anomaly_inputs[REGIME_FILTERED]
    scores, records = score_months(
        anomaly_inputs,
        regimes,
        len(regime_names),
        params,
        params["random_state"],
        params["training_window_months"],
    )
    table = score_table(anomaly_inputs, scores)
    named_records = [{**record, "model": regime_names[record["model"]]} for record in records]
    is_scored = table[IS_ANOMALY].notna()
    logger.info(
        "Anomalies: %d hours scored, %d flagged (%.2f %%), %d models fitted",
        int(is_scored.sum()),
        int(table.loc[is_scored, IS_ANOMALY].sum()),
        100 * float(table.loc[is_scored, IS_ANOMALY].mean()) if is_scored.any() else 0.0,
        len(records),
    )
    return table, named_records


# ==================================================
# Events
# ==================================================


def _holdout_instant(params: dict[str, Any]) -> pd.Timestamp:
    """Midnight of holdout_start in market time, as an instant."""
    return pd.Timestamp(params["holdout_start"], tz=params["market_timezone"])


def segments(index: pd.DatetimeIndex, params: dict[str, Any]) -> np.ndarray:
    """0 for the hours before holdout_start, 1 from it: an event never crosses it."""
    is_holdout = np.asarray(index >= _holdout_instant(params))
    return is_holdout.astype(int)


def assign_events(is_anomaly: pd.Series, segment: np.ndarray, max_gap_hours: int) -> pd.Series:
    """Event number of each anomalous hour; missing for the other hours.

    Two anomalous hours belong to the same event when at most max_gap_hours
    scored normal hours separate them, with no unscored hour between them (an
    unknown hour is not a normal hour) and both in the same segment.

    Args:
        is_anomaly: Flag of each hour of a continuous hourly grid; missing = unscored.
        segment: Segment of each hour (`segments`).
        max_gap_hours: Normal hours an event can bridge.

    Returns:
        Event numbers from 0, in time order (Int32).
    """
    flags = is_anomaly.to_numpy(dtype=float, na_value=np.nan)
    positions = np.flatnonzero(flags == 1)
    events = pd.Series(pd.NA, index=is_anomaly.index, dtype=EVENT_DTYPE)
    if len(positions) == 0:
        return events
    unscored_before = np.concatenate([[0], np.cumsum(np.isnan(flags))])
    gaps = np.diff(positions) - 1
    unscored_between = unscored_before[positions[1:]] - unscored_before[positions[:-1] + 1]
    same_segment = segment[positions[1:]] == segment[positions[:-1]]
    continues = (gaps <= max_gap_hours) & (unscored_between == 0) & same_segment
    starts_event = np.concatenate([[True], ~continues])
    events.iloc[positions] = np.cumsum(starts_event) - 1
    return events


def _empty_event_table() -> pd.DataFrame:
    """Event table without rows, with the column types of a full one."""
    empty = pd.DataFrame(columns=list(EVENT_COLUMNS))
    return empty.rename_axis(EVENT_ID)


def build_event_table(anomalies: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """One row per event: first and last anomalous hours, duration, and the peak hour.

    The duration counts real hours (UTC grid): the 25-hour night of the autumn
    clock change is counted right.

    Args:
        anomalies: Score table with event_id.
        prices: Hourly price (EUR/MWh), same index.

    Returns:
        One row per event_id, columns EVENT_COLUMNS.
    """
    in_event = anomalies[EVENT_ID].notna()
    if not in_event.any():
        return _empty_event_table()
    hours = anomalies[in_event].assign(**{PRICE: prices[in_event]})
    grouped = hours.groupby(EVENT_ID)
    hour_stamps = pd.Series(hours.index, index=hours.index).groupby(hours[EVENT_ID])
    first_hours = hour_stamps.min()
    last_hours = hour_stamps.max()
    peak_hours = grouped[ANOMALY_SCORE].idxmax()
    peaks = hours.loc[peak_hours.to_numpy()]
    durations = (last_hours - first_hours) / ONE_HOUR + 1
    columns = {
        FIRST_HOUR: first_hours.array,
        LAST_HOUR: last_hours.array,
        DURATION_HOURS: durations.astype(int).to_numpy(),
        ANOMALOUS_HOURS: grouped.size().to_numpy(),
        PEAK_HOUR: pd.DatetimeIndex(peak_hours.to_numpy()).array,
        PEAK_SCORE: peaks[ANOMALY_SCORE].to_numpy(),
        PEAK_PRICE: peaks[PRICE].to_numpy(),
        PEAK_PRICE_PERCENTILE: peaks[PRICE_PERCENTILE].to_numpy(),
        PEAK_REGIME: peaks[REGIME_FILTERED].array,
        TOP_DRIVER: peaks[TOP_DRIVER].array,
        TOP_DRIVER_PERCENTILE: peaks[TOP_DRIVER_PERCENTILE].to_numpy(),
    }
    event_index = pd.Index(first_hours.index, dtype=EVENT_DTYPE, name=EVENT_ID)
    return pd.DataFrame(columns, index=event_index)


def group_anomaly_events(
    anomaly_scores: pd.DataFrame, anomaly_inputs: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anomalies table with the event of each anomalous hour, and the events table.

    Grouping looks at the hours after each anomaly: descriptive only, never an
    input of a prediction. Events stop at holdout_start, so the development
    figures never see a reserved hour.

    Args:
        anomaly_scores: First output of `detect_anomalies`.
        anomaly_inputs: Output of `build_anomaly_inputs` (for the price).
        params: Anomaly parameters (event_max_gap_hours, holdout_start).

    Returns:
        The `anomalies` table (ANOMALY_COLUMNS) and the `anomaly_events` table.
    """
    segment = segments(anomaly_scores.index, params)
    events = assign_events(anomaly_scores[IS_ANOMALY], segment, params["event_max_gap_hours"])
    anomalies = anomaly_scores.assign(**{EVENT_ID: events})
    event_table = build_event_table(anomalies, anomaly_inputs[PRICE])
    logger.info(
        "Anomaly events: %d events from %d anomalous hours",
        len(event_table),
        int(events.notna().sum()),
    )
    return anomalies, event_table


# ==================================================
# Development period: checks that can inform a choice (D9)
# ==================================================


def _before_holdout(table: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Rows of an hourly table before holdout_start."""
    is_development = table.index < _holdout_instant(params)
    return table[is_development]


def _flags(table: pd.DataFrame) -> pd.Series:
    """Flags of the scored hours, as plain booleans."""
    is_scored = table[IS_ANOMALY].notna()
    return table.loc[is_scored, IS_ANOMALY].astype(bool)


def share_flagged_by_regime(
    table: pd.DataFrame, regime_names: list[str]
) -> dict[str, float | None]:
    """Share of the scored hours of each regime that are flagged; None without scored hours."""
    flags = _flags(table)
    regimes = table.loc[flags.index, REGIME_FILTERED]
    return {
        name: _share(int(flags[regimes == state].sum()), int((regimes == state).sum()))
        for state, name in enumerate(regime_names)
    }


def _ablation(
    per_regime: pd.DataFrame, one_model: pd.DataFrame, regime_names: list[str]
) -> dict[str, Any]:
    """Share flagged in each regime by the per-regime models and by one model for all regimes.

    Both on the hours scored by both, so the comparison is on the same hours.
    """
    is_common = per_regime[IS_ANOMALY].notna() & one_model[IS_ANOMALY].notna()
    return {
        "per_regime_models": share_flagged_by_regime(per_regime[is_common], regime_names),
        "one_model_for_all_regimes": share_flagged_by_regime(one_model[is_common], regime_names),
    }


def _events_found_by(events: pd.Series, other_flags: pd.Series) -> float | None:
    """Share of the events with at least one hour also flagged with the other seed."""
    in_event = events.notna()
    if not in_event.any():
        return None
    other_flagged = other_flags.reindex(events.index, fill_value=False)
    found = other_flagged[in_event].groupby(events[in_event]).any()
    return _rounded(found.mean())


def _seed_stability(
    first: pd.DataFrame, second: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Agreement of the flags, events and scores obtained with two seeds."""
    is_common = first[IS_ANOMALY].notna() & second[IS_ANOMALY].notna()
    first_flags = _flags(first[is_common])
    second_flags = _flags(second[is_common])
    union = int((first_flags | second_flags).sum())
    intersection = int((first_flags & second_flags).sum())
    segment = segments(first.index, params)
    max_gap = params["event_max_gap_hours"]
    first_events = assign_events(first[IS_ANOMALY], segment, max_gap)
    second_events = assign_events(second[IS_ANOMALY], segment, max_gap)
    score_correlation = first.loc[is_common, ANOMALY_SCORE].corr(
        second.loc[is_common, ANOMALY_SCORE], method="spearman"
    )
    return {
        "second_seed": params["stability_seed"],
        "anomalous_hours_jaccard": _share(intersection, union),
        "events_found_by_second_seed": _events_found_by(first_events, second_flags),
        "events_of_second_seed_found": _events_found_by(second_events, first_flags),
        "score_rank_correlation": _rounded(score_correlation),
    }


def window_label(window_months: int | None) -> str:
    """Report label of a training window: 'expanding' or e.g. '24_months'."""
    if window_months is None:
        return EXPANDING
    return f"{window_months}_months"


def _window_summary(table: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Calibration of one training window on the development hours."""
    scored = table[table[IS_ANOMALY].notna()]
    flags = scored[IS_ANOMALY].astype(bool)
    years = market_days(scored.index, params["market_timezone"]).year.to_numpy()
    by_year = _by_year_and_regime(scored, years, params["regime_names"])
    is_neighbour = (scored[ANOMALY_MODEL] != scored[REGIME_FILTERED]).to_numpy(dtype=bool)
    return {
        "scored_hours": len(scored),
        "share_flagged": _share(int(flags.sum()), len(scored)),
        "share_flagged_by_regime": share_flagged_by_regime(scored, params["regime_names"]),
        "mean_absolute_calibration_error": _calibration_error(by_year, params),
        "regime_years_outside_band": len(_calibration_outliers(by_year, params)),
        "hours_scored_by_neighbour_regime": int(is_neighbour.sum()),
    }


def _window_comparison(
    development: pd.DataFrame, main_scores: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Calibration of each candidate training window on the development hours (run 4 rule).

    The configured window reuses the main scores; the others are refitted the
    same way, on the development hours only.
    """
    regimes = development[REGIME_FILTERED]
    n_groups = len(params["regime_names"])
    summaries = {}
    for window_months in params["window_candidates_months"]:
        is_configured = window_months == params["training_window_months"]
        if is_configured:
            table = main_scores
        else:
            scores, _ = score_months(
                development, regimes, n_groups, params, params["random_state"], window_months
            )
            table = score_table(development, scores)
        summaries[window_label(window_months)] = _window_summary(table, params)
    errors = {
        label: summary["mean_absolute_calibration_error"]
        for label, summary in summaries.items()
        if summary["mean_absolute_calibration_error"] is not None
    }
    lowest = min(errors, key=errors.get) if errors else None
    return {
        "configured": window_label(params["training_window_months"]),
        "lowest_calibration_error": lowest,
        "candidates": summaries,
    }


def analyze_development_period(
    anomaly_inputs: pd.DataFrame, anomaly_scores: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Checks that can inform a choice, on the hours before holdout_start only (D9).

    - ablation: one Isolation Forest for all regimes, with the same monthly
      refit on the past; share of the hours of each regime it flags, next to
      the per-regime models;
    - stability: anomalous hours, events and scores with a second seed;
    - training windows: calibration of each candidate window.

    Args:
        anomaly_inputs: Output of `build_anomaly_inputs`.
        anomaly_scores: First output of `detect_anomalies`.
        params: Anomaly parameters.

    Returns:
        A JSON-ready part of the report.
    """
    development = _before_holdout(anomaly_inputs, params)
    main_scores = anomaly_scores.loc[development.index]
    regimes = development[REGIME_FILTERED]
    regime_names = params["regime_names"]
    window_months = params["training_window_months"]
    one_group = pd.Series(0, index=development.index).where(regimes.notna())
    one_model_scores, _ = score_months(
        development, one_group, 1, params, params["random_state"], window_months
    )
    one_model_table = score_table(development, one_model_scores)
    second_scores, _ = score_months(
        development, regimes, len(regime_names), params, params["stability_seed"], window_months
    )
    second_table = score_table(development, second_scores)
    return {
        "global_model_ablation": _ablation(main_scores, one_model_table, regime_names),
        "seed_stability": _seed_stability(main_scores, second_table, params),
        "training_window_comparison": _window_comparison(development, main_scores, params),
    }


# ==================================================
# Report
# ==================================================


def _share(count: int, total: int) -> float | None:
    """count / total rounded for the report; None when total is 0."""
    if total == 0:
        return None
    return _rounded(count / total)


def _rounded(value: Any, decimals: int = SHARE_DECIMALS) -> float | None:
    """Python float rounded for the JSON report; None if missing or infinite."""
    number = float(value)
    is_finite = bool(np.isfinite(number))
    return round(number, decimals) if is_finite else None


def _local_time(instant: pd.Timestamp, params: dict[str, Any]) -> str:
    """Instant written in market time, e.g. '2024-12-12 17:00'."""
    return instant.tz_convert(params["market_timezone"]).strftime(LOCAL_TIME_FORMAT)


def flags_by_level(
    anomalies: pd.DataFrame, refits: list[dict[str, Any]], params: dict[str, Any]
) -> pd.DataFrame:
    """Flag of each scored hour at each contamination level, from the thresholds of its model.

    Args:
        anomalies: Hourly anomalies table.
        refits: Second output of `detect_anomalies`.
        params: Anomaly parameters.

    Returns:
        One boolean column per level (missing for unscored hours); the column of
        the configured contamination equals is_anomaly.
    """
    levels = params["contamination_levels"]
    rows = [
        {"month": record["month"], "model": record["model"], **record["thresholds"]}
        for record in refits
    ]
    thresholds = pd.DataFrame(rows, columns=["month", "model", *map(str, levels)])
    by_model = thresholds.set_index(["month", "model"])
    months = market_days(anomalies.index, params["market_timezone"]).strftime(MONTH_FORMAT)
    names = anomalies[ANOMALY_MODEL].map(dict(enumerate(params["regime_names"])))
    keys = pd.MultiIndex.from_arrays([months, names.to_numpy(dtype=object)])
    hour_thresholds = by_model.reindex(keys).to_numpy(dtype=float)
    scores = anomalies[ANOMALY_SCORE].to_numpy()[:, None]
    flags = np.where(np.isnan(scores), np.nan, scores > hour_thresholds)
    return pd.DataFrame(flags, index=anomalies.index, columns=levels).astype(FLAG_DTYPE)


def _levels_summary(level_flags: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Share flagged and events per week at each contamination level (business budget)."""
    summary = {}
    for level in params["contamination_levels"]:
        flags = level_flags[level]
        scored_hours = int(flags.notna().sum())
        segment = segments(flags.index, params)
        events = assign_events(flags, segment, params["event_max_gap_hours"])
        weeks = scored_hours / HOURS_PER_WEEK
        summary[str(level)] = {
            "share_flagged": _share(int(flags.sum()), scored_hours),
            "events_per_week": _rounded(events.nunique() / weeks, 2) if weeks else None,
        }
    return summary


def outside_training_range(table: pd.DataFrame) -> np.ndarray:
    """True for the anomalies whose driver lies outside every training value (novelty)."""
    return table[TOP_DRIVER_PERCENTILE].isin([0.0, FULL_RANGE]).to_numpy()


def _by_year_and_regime(
    scored: pd.DataFrame, years: np.ndarray, regime_names: list[str]
) -> dict[str, dict[str, Any]]:
    """Scored hours, share flagged and share of novelty anomalies, per regime and year."""
    flags = scored[IS_ANOMALY].to_numpy(dtype=bool)
    is_novelty = outside_training_range(scored)
    regimes = scored[REGIME_FILTERED].to_numpy(dtype=float, na_value=np.nan)
    table = {}
    for year in sorted(set(years.tolist())):
        cells = {}
        for state, name in enumerate(regime_names):
            is_cell = (years == year) & (regimes == state)
            scored_hours = int(is_cell.sum())
            flagged_hours = int((flags & is_cell).sum())
            novelty_hours = int((is_novelty & is_cell).sum())
            cells[name] = {
                "scored_hours": scored_hours,
                "share_flagged": _share(flagged_hours, scored_hours),
                "share_of_anomalies_outside_training_range": _share(novelty_hours, flagged_hours),
            }
        table[str(year)] = cells
    return table


def _scored_by_neighbour(scored: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Hours scored by the model of another regime (too few hours in their own)."""
    is_neighbour = (scored[ANOMALY_MODEL] != scored[REGIME_FILTERED]).to_numpy(dtype=bool)
    days = market_days(scored.index[is_neighbour], params["market_timezone"])
    months = sorted(set(days.strftime(MONTH_FORMAT)))
    return {"hours": int(is_neighbour.sum()), "months": months}


def _events_summary(events: pd.DataFrame, scored_hours: int) -> dict[str, Any]:
    """Number of events, events per week, durations and anomalous hours per event."""
    weeks = scored_hours / HOURS_PER_WEEK
    if events.empty:
        return {"events": 0, "events_per_week": 0.0 if weeks else None}
    durations = events[DURATION_HOURS]
    return {
        "events": len(events),
        "events_per_week": _rounded(len(events) / weeks, 2) if weeks else None,
        "duration_hours": {
            "median": _rounded(durations.median(), 1),
            "p90": _rounded(durations.quantile(DURATION_QUANTILE), 1),
            "max": int(durations.max()),
        },
        "median_anomalous_hours_per_event": _rounded(events[ANOMALOUS_HOURS].median(), 1),
    }


def _drivers_summary(flagged: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """Share of the anomalies driven by each feature, above or below its usual values."""
    is_above = flagged[TOP_DRIVER_PERCENTILE] >= MEDIAN_PERCENTILE
    total = len(flagged)
    return {
        name: {
            "above": _share(int(((flagged[TOP_DRIVER] == name) & is_above).sum()), total),
            "below": _share(int(((flagged[TOP_DRIVER] == name) & ~is_above).sum()), total),
        }
        for name in features
    }


def _share_by_hour(flagged: pd.DataFrame, params: dict[str, Any]) -> dict[str, float | None]:
    """Share of the anomalies at each local hour of the day."""
    local_hours = flagged.index.tz_convert(params["market_timezone"]).hour
    hour_counts = pd.Series(local_hours).value_counts()
    total = len(flagged)
    return {
        str(hour): _share(int(hour_counts.get(hour, 0)), total) for hour in range(HOURS_PER_DAY)
    }


def _shape_of_anomalies(
    scored: pd.DataFrame, flagged: pd.DataFrame, switch_hours: pd.Series, params: dict[str, Any]
) -> dict[str, Any]:
    """Direction, hour of the day, novelty and regime-switch days of the anomalies.

    The hours of the day are also given without the novelty anomalies (driver
    outside the training range): a new market shape (e.g. solar dips) is not a
    daily pattern the features miss.
    """
    total = len(flagged)
    is_outside_range = outside_training_range(flagged)
    is_above = flagged[PRICE_PERCENTILE] >= MEDIAN_PERCENTILE
    on_switch_days = switch_hours.reindex(scored.index)
    return {
        "price_above_regime_median": _share(int(is_above.sum()), total),
        "share_by_hour_of_day": _share_by_hour(flagged, params),
        "share_by_hour_of_day_within_training_range": _share_by_hour(
            flagged[~is_outside_range], params
        ),
        "top_drivers": _drivers_summary(flagged, params["features"]),
        "driver_outside_training_range": _share(int(is_outside_range.sum()), total),
        "regime_switch_days": {
            "share_of_anomalies": _share(int(on_switch_days[flagged.index].sum()), total),
            "share_of_scored_hours": _share(int(on_switch_days.sum()), len(scored)),
        },
    }


def _top_events(events: pd.DataFrame, params: dict[str, Any]) -> list[dict[str, Any]]:
    """The strongest events (highest peak score), described in market time."""
    strongest = events.sort_values(PEAK_SCORE, ascending=False).head(params["top_events"])
    regime_names = params["regime_names"]
    return [
        {
            "first_hour": _local_time(event[FIRST_HOUR], params),
            "duration_hours": int(event[DURATION_HOURS]),
            "anomalous_hours": int(event[ANOMALOUS_HOURS]),
            "peak_hour": _local_time(event[PEAK_HOUR], params),
            "peak_score": _rounded(event[PEAK_SCORE], SCORE_DECIMALS),
            "peak_price_eur_mwh": _rounded(event[PEAK_PRICE], PRICE_DECIMALS),
            "peak_price_percentile": _rounded(event[PEAK_PRICE_PERCENTILE], PERCENTILE_DECIMALS),
            "regime": regime_names[int(event[PEAK_REGIME])],
            "top_driver": str(event[TOP_DRIVER]),
            "top_driver_percentile": _rounded(event[TOP_DRIVER_PERCENTILE], PERCENTILE_DECIMALS),
        }
        for _, event in strongest.iterrows()
    ]


def _period_summary(
    anomalies: pd.DataFrame,
    events: pd.DataFrame,
    level_flags: pd.DataFrame,
    switch_hours: pd.Series,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Everything the report says about one period (development or reserved)."""
    is_scored = anomalies[IS_ANOMALY].notna()
    scored = anomalies[is_scored]
    flagged = scored[scored[IS_ANOMALY].astype(bool)]
    years = market_days(scored.index, params["market_timezone"]).year.to_numpy()
    regime_names = params["regime_names"]
    return {
        "scored_hours": len(scored),
        "anomalous_hours": len(flagged),
        "share_flagged": _share(len(flagged), len(scored)),
        "share_flagged_by_regime": share_flagged_by_regime(scored, regime_names),
        "by_year_and_regime": _by_year_and_regime(scored, years, regime_names),
        "scored_by_neighbour_regime": _scored_by_neighbour(scored, params),
        "contamination_levels": _levels_summary(level_flags, params),
        "events": _events_summary(events, len(scored)),
        **_shape_of_anomalies(scored, flagged, switch_hours, params),
        "top_events": _top_events(events, params),
    }


def regime_switch_hours(anomalies: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """True for the hours of days whose filtered regime differs from the day before's.

    Only uses the labels of the day and the day before.
    """
    days = market_days(anomalies.index, params["market_timezone"])
    day_labels = anomalies[REGIME_FILTERED].groupby(days).first()
    previous_labels = day_labels.shift(1)
    is_switch_day = day_labels.ne(previous_labels).fillna(False).astype(bool)
    has_both = day_labels.notna() & previous_labels.notna()
    switch_days = (is_switch_day & has_both).reindex(days)
    return pd.Series(switch_days.to_numpy(dtype=bool), index=anomalies.index)


def _extreme_prices(
    scored: pd.DataFrame, prices: pd.Series, params: dict[str, Any]
) -> dict[str, Any]:
    """Share flagged among the scored hours at the highest and at the lowest price."""
    if scored.empty:
        return {}
    scored_prices = prices[scored.index]
    flags = scored[IS_ANOMALY].astype(bool)
    extremes = {}
    for name, value in (("highest", scored_prices.max()), ("lowest", scored_prices.min())):
        is_extreme = scored_prices == value
        extreme_hours = scored_prices.index[is_extreme.to_numpy()]
        extremes[name] = {
            "price_eur_mwh": _rounded(value, PRICE_DECIMALS),
            "hours": int(is_extreme.sum()),
            "first_hour": _local_time(extreme_hours[0], params),
            "share_flagged": _share(int(flags[is_extreme].sum()), int(is_extreme.sum())),
        }
    return extremes


def _calibration_outliers(
    by_year: dict[str, dict[str, Any]], params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Regime-years with enough scored hours flagged outside the calibration band."""
    contamination = params["contamination"]
    low = contamination / CALIBRATION_FACTOR
    high = contamination * CALIBRATION_FACTOR
    return [
        {"year": year, "regime": name, **cell}
        for year, name, cell in _calibration_cells(by_year, params)
        if not low <= cell["share_flagged"] <= high
    ]


def _calibration_cells(
    by_year: dict[str, dict[str, Any]], params: dict[str, Any]
) -> list[tuple[str, str, dict[str, Any]]]:
    """(year, regime, figures) of the regime-years with at least min_training_hours scored hours."""
    return [
        (year, name, cell)
        for year, regimes in by_year.items()
        for name, cell in regimes.items()
        if cell["scored_hours"] >= params["min_training_hours"]
    ]


def _calibration_error(by_year: dict[str, dict[str, Any]], params: dict[str, Any]) -> float | None:
    """Mean absolute gap between the share flagged and the contamination, over the regime-years.

    Compares training windows on the development period: lower is better calibrated.
    """
    gaps = [
        abs(cell["share_flagged"] - params["contamination"])
        for _, _, cell in _calibration_cells(by_year, params)
    ]
    return _rounded(np.mean(gaps)) if gaps else None


def _rules_for_run_4(
    summary: dict[str, Any],
    scored: pd.DataFrame,
    prices: pd.Series,
    params: dict[str, Any],
) -> dict[str, Any]:
    """The rules fixed before run 4, applied to the development period (decisions.md)."""
    within_range_shares = summary["share_by_hour_of_day_within_training_range"]
    hour_shares = {hour: share for hour, share in within_range_shares.items() if share is not None}
    busiest_hour = max(hour_shares, key=hour_shares.get, default=None)
    busiest_share = hour_shares.get(busiest_hour)
    hour_limit = HOUR_CONCENTRATION_FACTOR / HOURS_PER_DAY
    is_concentrated = busiest_share is not None and busiest_share > hour_limit
    contamination = params["contamination"]
    return {
        "hour_of_day": {
            "busiest_hour": busiest_hour,
            "share_within_training_range": busiest_share,
            "limit": _rounded(hour_limit),
            "add_hour_features": is_concentrated,
        },
        "calibration": {
            "band": [contamination / CALIBRATION_FACTOR, contamination * CALIBRATION_FACTOR],
            "min_scored_hours": params["min_training_hours"],
            "regime_years_outside_band": _calibration_outliers(
                summary["by_year_and_regime"], params
            ),
            "mean_absolute_error": _calibration_error(summary["by_year_and_regime"], params),
        },
        "extreme_prices_flagged": _extreme_prices(scored, prices, params),
    }


def _coverage(
    anomalies: pd.DataFrame, inputs: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Scored hours and the reason why the other hours are not scored."""
    days = market_days(inputs.index, params["market_timezone"])
    first_month = scored_months(days, inputs[REGIME_FILTERED], params["min_train_months"])[0]
    is_in_scored_months = np.asarray(days >= first_month)
    has_regime = inputs[REGIME_FILTERED].notna().to_numpy()
    columns = input_columns(params["features"])
    has_features = inputs[columns].notna().all(axis=1).to_numpy()
    is_scored = anomalies[IS_ANOMALY].notna().to_numpy()
    return {
        "hours": len(anomalies),
        "scored_hours": int(is_scored.sum()),
        "first_scored_month": first_month.strftime(MONTH_FORMAT),
        "unscored_hours": {
            "before_first_scored_month": int((~is_in_scored_months).sum()),
            "no_filtered_regime": int((is_in_scored_months & ~has_regime).sum()),
            "missing_feature": int((is_in_scored_months & has_regime & ~has_features).sum()),
            "no_model": int((is_in_scored_months & has_regime & has_features & ~is_scored).sum()),
        },
    }


def build_anomaly_report(
    anomalies: pd.DataFrame,
    anomaly_events: pd.DataFrame,
    anomaly_inputs: pd.DataFrame,
    anomaly_refits: list[dict[str, Any]],
    anomaly_development: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Report of the anomalies, in two sections and the refit log.

    - choices_before_holdout: the only section to read when choosing (features,
      contamination, event gap); nothing in it uses the reserved period.
    - holdout_descriptive_only: the same figures on the reserved period.

    Args:
        anomalies: First output of `group_anomaly_events`.
        anomaly_events: Second output of `group_anomaly_events`.
        anomaly_inputs: Output of `build_anomaly_inputs`.
        anomaly_refits: Second output of `detect_anomalies`.
        anomaly_development: Output of `analyze_development_period`.
        params: Anomaly parameters.

    Returns:
        A JSON-ready report.
    """
    holdout = _holdout_instant(params)
    is_development = np.asarray(anomalies.index < holdout)
    is_development_event = np.asarray(anomaly_events[FIRST_HOUR] < holdout, dtype=bool)
    level_flags = flags_by_level(anomalies, anomaly_refits, params)
    switch_hours = regime_switch_hours(anomalies, params)
    periods = {}
    for name, is_period, is_period_event in (
        ("development", is_development, is_development_event),
        ("holdout", ~is_development, ~is_development_event),
    ):
        periods[name] = _period_summary(
            anomalies[is_period],
            anomaly_events[is_period_event],
            level_flags[is_period],
            switch_hours,
            params,
        )
    development = anomalies[is_development]
    development_scored = development[development[IS_ANOMALY].notna()]
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "holdout_start": str(params["holdout_start"]),
        "regime_names": list(params["regime_names"]),
        "features": list(params["features"]),
        "contamination": params["contamination"],
        "training_window": window_label(params["training_window_months"]),
        "coverage": _coverage(anomalies, anomaly_inputs, params),
        "choices_before_holdout": {
            **periods["development"],
            **anomaly_development,
            "rules_for_run_4": _rules_for_run_4(
                periods["development"], development_scored, anomaly_inputs[PRICE], params
            ),
        },
        "holdout_descriptive_only": periods["holdout"],
        "refits": anomaly_refits,
    }
    _log_report(report)
    return report


def _log_report(report: dict[str, Any]) -> None:
    """Main figures in the log; warnings on the rules that ask for a decision."""
    choices = report["choices_before_holdout"]
    logger.info(
        "Before holdout: %s of scored hours flagged, %s events per week, by regime %s",
        choices["share_flagged"],
        choices["events"].get("events_per_week"),
        choices["share_flagged_by_regime"],
    )
    logger.info("One model for all regimes would flag: %s", choices["global_model_ablation"])
    logger.info("Second seed: %s", choices["seed_stability"])
    windows = choices["training_window_comparison"]
    calibration_errors = {
        label: summary["mean_absolute_calibration_error"]
        for label, summary in windows["candidates"].items()
    }
    logger.info("Training windows, mean calibration error: %s", calibration_errors)
    if windows["lowest_calibration_error"] != windows["configured"]:
        logger.warning(
            "Training window %s calibrates better than %s: decide before going on",
            windows["lowest_calibration_error"],
            windows["configured"],
        )
    rules = choices["rules_for_run_4"]
    if rules["hour_of_day"]["add_hour_features"]:
        logger.warning("Anomalies concentrate on one hour of the day: %s", rules["hour_of_day"])
    outliers = rules["calibration"]["regime_years_outside_band"]
    if outliers:
        logger.warning("Regime-years flagged outside the calibration band: %s", outliers)
    for name, extreme in rules["extreme_prices_flagged"].items():
        if extreme["share_flagged"] != 1.0:
            logger.warning(
                "The %s price of the development period is not flagged: %s", name, extreme
            )
