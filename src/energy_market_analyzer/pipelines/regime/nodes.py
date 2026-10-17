"""Regime pipeline nodes: market regime of each day with a Gaussian HMM.

One observation per calendar day in Brussels, built from table C (decision
D17): mean price, volatility (standard deviation of the hourly price changes
of the day) and mean clean spark spread, all in EUR/MWh. They are known before
the day starts: its day-ahead prices are published around 12:45 the day
before, and the fuel prices of table C are the last ones before the day. The
HMM sees the volatility as log(1 + value), closer to the Gaussian shape it
assumes; it is not a price return (D5).

Two outputs (decision D6):
- filtered regime: on the first day of each month the HMM is refitted on the
  previous days only, then the forward algorithm gives the probability of each
  regime on day d from the days up to d. Models A and B, the alerts and the
  Isolation Forest use it: the regime of the decision day J is known at 11:00.
- smoothed regime: one HMM fitted on every day, forward-backward posterior.
  Retrospective analysis only (model C, notebook 3a), never a prediction input.

States are ordered by the mean volatility of the fitted model: 0 is the calmest.
Everything that can inform a choice (number of states, stability, shape of the
distributions, seasonality) is computed on the days before holdout_start (D9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats
from scipy.special import logsumexp

from energy_market_analyzer.pipelines.features.nodes import (
    CLEAN_SPARK_SPREAD,
    DAY,
    PRICE_CHANGE,
    market_days,
)
from energy_market_analyzer.pipelines.ingestion.nodes import PRICE, UTC_ZONE

logger = logging.getLogger(__name__)

# hmmlearn logs a warning whenever an EM step lowers the log-likelihood a little
# (min_covar makes EM not strictly increasing). EM then stops; with 10 starts per
# monthly refit, the run log would fill up with these lines. Fits are compared on
# their likelihood, and the report counts the fits stopped by the iteration cap.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# --- Daily observations (EUR/MWh) and the hourly column of table C each one summarizes ---
PRICE_MEAN = "price_mean"
PRICE_CHANGE_STD = "price_change_std"
CLEAN_SPARK_SPREAD_MEAN = "clean_spark_spread_mean"
DAILY_STATISTICS = {
    PRICE_MEAN: (PRICE, "mean"),
    PRICE_CHANGE_STD: (PRICE_CHANGE, "std"),
    CLEAN_SPARK_SPREAD_MEAN: (CLEAN_SPARK_SPREAD, "mean"),
}
OBSERVATIONS = tuple(DAILY_STATISTICS)
# The HMM sees the volatility as log(1 + value); states are ordered by it.
VOLATILITY = PRICE_CHANGE_STD
VOLATILITY_POSITION = OBSERVATIONS.index(VOLATILITY)
ONE_HOUR = pd.Timedelta(hours=1)

# --- Regime columns ---
FILTERED = "filtered"
SMOOTHED = "smoothed"
REGIME_FILTERED = f"regime_{FILTERED}"
REGIME_SMOOTHED = f"regime_{SMOOTHED}"
LABEL_DTYPE = "Int8"

# --- HMM fitting ---
RANDOM_INIT = "stmc"  # hmmlearn draws start, transition, means and covariances
NO_INIT = ""  # hmmlearn keeps the parameters already set (warm start)
DIAGONAL = "diag"

# --- Report ---
SHARE_DECIMALS = 4
# Below this agreement, two regimes may have swapped labels (refit) or depend on the seed.
LABEL_CONTINUITY_WARNING = 0.9
STABILITY_WARNING = 0.9
# Skewness and kurtosis of fewer days say nothing.
MIN_DAYS_FOR_SHAPE = 30


@dataclass(frozen=True)
class FittedHmm:
    """Gaussian HMM in the units of `hmm_inputs`, states ordered by increasing volatility.

    Attributes:
        startprob: Probability of each state on the first day, shape (K,).
        transmat: Day-to-day transition probabilities, shape (K, K).
        means: Mean of each input in each state, shape (K, d).
        covars: Full covariance matrix of each state, shape (K, d, d).
        log_likelihood: Log-likelihood of the training days.
        hit_iteration_cap: EM stopped at n_iter instead of converging.
        from_previous: The best start was the previous month's model.
    """

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    log_likelihood: float
    hit_iteration_cap: bool
    from_previous: bool


# ==================================================
# Daily observations
# ==================================================


def _require_table_c(features_model_c: pd.DataFrame) -> None:
    """Raise ValueError unless `features_model_c` is the continuous hourly UTC table C."""
    index = features_model_c.index
    is_utc = isinstance(index, pd.DatetimeIndex) and str(index.tz) == UTC_ZONE
    if not is_utc:
        raise ValueError("features_model_c: index must be a DatetimeIndex in UTC")
    steps = index[1:] - index[:-1]
    is_continuous = bool((steps == ONE_HOUR).all())
    if not is_continuous:
        raise ValueError("features_model_c: index must be a continuous hourly grid")
    hourly_columns = [column for column, _ in DAILY_STATISTICS.values()]
    missing_columns = [name for name in hourly_columns if name not in features_model_c.columns]
    if missing_columns:
        raise ValueError(f"features_model_c: missing columns {missing_columns}, re-run features")


def _complete_daily_stat(values: pd.Series, days: pd.DatetimeIndex, statistic: str) -> pd.Series:
    """Statistic of each calendar day; NaN if one hour of the day is missing (V3-4)."""
    grouped = values.groupby(days)
    day_values = grouped.agg(statistic)
    is_complete = grouped.count() == grouped.size()
    return day_values.where(is_complete)


def build_regime_observations(
    features_model_c: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """One row per calendar day in Brussels: the three observations of the HMM.

    - price_mean: mean of the hourly day-ahead prices of the day;
    - price_change_std: standard deviation of the hourly price changes of the
      day (the first one goes from the last hour of the day before);
    - clean_spark_spread_mean: mean of the hourly clean spark spread.

    A day with a missing hour has NaN observations. A missing price also makes
    the next day incomplete: its first price change is unknown.

    Known: before the day starts. Day-ahead prices of day d are published around
    12:45 on d-1; the fuel prices in the spread are the last ones before d.

    Args:
        features_model_c: Hourly table C (features pipeline).
        params: Regime parameters (market_timezone).

    Returns:
        A table indexed by `day` (local date, without time zone), columns OBSERVATIONS.

    Raises:
        ValueError: If table C is not the expected hourly UTC table.
    """
    _require_table_c(features_model_c)
    days = market_days(features_model_c.index, params["market_timezone"])
    daily = {
        name: _complete_daily_stat(features_model_c[column], days, statistic)
        for name, (column, statistic) in DAILY_STATISTICS.items()
    }
    observations = pd.DataFrame(daily).rename_axis(DAY)
    complete_days = int(observations.notna().all(axis=1).sum())
    logger.info("Regime observations: %d days, %d complete", len(observations), complete_days)
    return observations


def hmm_inputs(observations: pd.DataFrame) -> np.ndarray:
    """Observations as the HMM sees them: the volatility as log(1 + value).

    The volatility is a standard deviation, never negative, and very skewed:
    its logarithm is close to the Gaussian shape the HMM assumes within a state.

    Args:
        observations: Output of `build_regime_observations` (or a slice of it).

    Returns:
        Array of shape (days, 3), NaN rows for incomplete days.
    """
    log_volatility = np.log1p(observations[VOLATILITY])
    inputs = observations.assign(**{VOLATILITY: log_volatility})
    return inputs[list(OBSERVATIONS)].to_numpy(dtype=float)


# ==================================================
# Fitting
# ==================================================


def _complete_runs(is_complete: np.ndarray) -> list[int]:
    """Lengths of the runs of consecutive complete days, in order.

    hmmlearn treats each run as its own sequence: an incomplete day cuts the
    chain instead of joining the days around it.
    """
    flags = np.concatenate([[0], is_complete.astype(int), [0]])
    edges = np.flatnonzero(np.diff(flags))
    run_starts = edges[::2]
    run_ends = edges[1::2]
    return (run_ends - run_starts).tolist()


def _covars_for(covars: np.ndarray, covariance_type: str) -> np.ndarray:
    """Full covariance matrices in the shape hmmlearn expects for `covariance_type`."""
    if covariance_type == DIAGONAL:
        return np.diagonal(covars, axis1=1, axis2=2).copy()
    return covars


def _new_model(n_states: int, params: dict[str, Any], seed: int, init_params: str) -> GaussianHMM:
    """Unfitted Gaussian HMM with the EM settings of `params`."""
    return GaussianHMM(
        n_components=n_states,
        covariance_type=params["covariance_type"],
        min_covar=params["min_covar"],
        n_iter=params["n_iter"],
        tol=params["tol"],
        random_state=seed,
        init_params=init_params,
    )


def _warm_start(
    previous: FittedHmm,
    params: dict[str, Any],
    center: np.ndarray,
    scale: np.ndarray,
) -> GaussianHMM:
    """Model starting from the previous month's parameters, rescaled to this month's inputs."""
    n_states = len(previous.startprob)
    model = _new_model(n_states, params, params["random_state"], NO_INIT)
    scaled_covars = previous.covars / np.outer(scale, scale)
    model.startprob_ = previous.startprob
    model.transmat_ = previous.transmat
    model.means_ = (previous.means - center) / scale
    model.covars_ = _covars_for(scaled_covars, params["covariance_type"])
    return model


def _starting_models(
    n_states: int,
    params: dict[str, Any],
    seed: int,
    previous: FittedHmm | None,
    scaling: tuple[np.ndarray, np.ndarray],
) -> list[tuple[GaussianHMM, bool]]:
    """`n_init` models to fit: the previous month's model first if any, random starts after.

    The previous model comes first so that it is kept when a random start
    reaches the same optimum: regime definitions then stay stable across refits.
    """
    has_previous = previous is not None
    random_count = params["n_init"] - int(has_previous)
    random_starts = [
        (_new_model(n_states, params, seed + offset, RANDOM_INIT), False)
        for offset in range(random_count)
    ]
    if not has_previous:
        return random_starts
    center, scale = scaling
    warm_start = (_warm_start(previous, params, center, scale), True)
    return [warm_start, *random_starts]


def _fit_score(model: GaussianHMM, scaled: np.ndarray, lengths: list[int]) -> float | None:
    """Fit `model` in place and return its log-likelihood; None if EM failed."""
    try:
        model.fit(scaled, lengths)
        score = model.score(scaled, lengths)
    except (ValueError, np.linalg.LinAlgError) as error:
        logger.debug("HMM start failed: %s", error)
        return None
    is_finite = bool(np.isfinite(score))
    return float(score) if is_finite else None


def _best_start(
    starts: list[tuple[GaussianHMM, bool]], scaled: np.ndarray, lengths: list[int]
) -> tuple[GaussianHMM, bool, float] | None:
    """Fitted start with the highest log-likelihood; None if every start failed."""
    best = None
    for model, from_previous in starts:
        score = _fit_score(model, scaled, lengths)
        is_better = score is not None and (best is None or score > best[2])
        if is_better:
            best = (model, from_previous, score)
    return best


def _hit_iteration_cap(model: GaussianHMM) -> bool:
    """True if EM stopped at n_iter rather than on `tol`.

    hmmlearn's `monitor_.converged` also returns True when EM merely reached
    n_iter, so the last gain of log-likelihood is checked here.
    """
    history = model.monitor_.history
    has_two_steps = len(history) >= 2
    last_gain = history[-1] - history[-2] if has_two_steps else np.inf
    return bool(last_gain >= model.tol)


def _ordered_by_volatility(
    model: GaussianHMM,
    scaling: tuple[np.ndarray, np.ndarray],
    scaled_log_likelihood: float,
    n_days: int,
    from_previous: bool,
) -> FittedHmm:
    """Parameters back in the units of the inputs, states sorted by increasing mean volatility.

    The order comes from the fitted means, which only depend on the training
    days: it never looks at the days being labelled.
    """
    center, scale = scaling
    means = model.means_ * scale + center
    covars = model.covars_ * np.outer(scale, scale)
    order = np.argsort(means[:, VOLATILITY_POSITION])
    # Change of variables x = center + scale * z: each day loses sum(log scale).
    log_likelihood = scaled_log_likelihood - n_days * float(np.log(scale).sum())
    return FittedHmm(
        startprob=model.startprob_[order],
        transmat=model.transmat_[np.ix_(order, order)],
        means=means[order],
        covars=covars[order],
        log_likelihood=log_likelihood,
        hit_iteration_cap=_hit_iteration_cap(model),
        from_previous=from_previous,
    )


def fit_hmm(
    inputs: np.ndarray,
    n_states: int,
    params: dict[str, Any],
    seed: int,
    previous: FittedHmm | None = None,
) -> FittedHmm:
    """Best of `n_init` EM fits of a Gaussian HMM on the complete days of `inputs`.

    The inputs are standardized for the fit (the random starts and min_covar
    then work on comparable scales); the parameters are brought back to the
    units of `inputs` afterwards.

    Args:
        inputs: Rows of `hmm_inputs` for consecutive days; NaN rows are skipped.
        n_states: Number of hidden states.
        params: Regime parameters (covariance_type, n_init, n_iter, tol, min_covar).
        seed: Seed of the first random start; start k uses seed + k.
        previous: Previous month's model, tried as one of the starts.

    Returns:
        The fitted model, states ordered by increasing volatility.

    Raises:
        ValueError: If there are too few complete days or an input is constant.
        RuntimeError: If every start failed.
    """
    is_complete = ~np.isnan(inputs).any(axis=1)
    complete_inputs = inputs[is_complete]
    if len(complete_inputs) <= n_states:
        raise ValueError(f"Only {len(complete_inputs)} complete days to fit {n_states} states")
    center = complete_inputs.mean(axis=0)
    scale = complete_inputs.std(axis=0)
    if (scale == 0).any():
        raise ValueError("An HMM input is constant over the training days")
    scaled = (complete_inputs - center) / scale
    lengths = _complete_runs(is_complete)
    starts = _starting_models(n_states, params, seed, previous, (center, scale))
    best = _best_start(starts, scaled, lengths)
    if best is None:
        raise RuntimeError(f"Every HMM start failed ({n_states} states, {len(scaled)} days)")
    model, from_previous, scaled_log_likelihood = best
    return _ordered_by_volatility(
        model, (center, scale), scaled_log_likelihood, len(scaled), from_previous
    )


def to_hmmlearn(fitted: FittedHmm, covariance_type: str) -> GaussianHMM:
    """hmmlearn model holding the parameters of `fitted`, for posteriors and the BIC."""
    n_states, n_features = fitted.means.shape
    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, init_params="")
    model.n_features = n_features
    model.startprob_ = fitted.startprob
    model.transmat_ = fitted.transmat
    model.means_ = fitted.means
    model.covars_ = _covars_for(fitted.covars, covariance_type)
    return model


# ==================================================
# Regime probabilities
# ==================================================


def _log_emissions(fitted: FittedHmm, inputs: np.ndarray) -> np.ndarray:
    """Log-density of each day's inputs in each state; 0 (no information) on incomplete days."""
    is_complete = ~np.isnan(inputs).any(axis=1)
    complete_inputs = inputs[is_complete]
    log_emissions = np.zeros((len(inputs), len(fitted.startprob)))
    for state, (mean, covar) in enumerate(zip(fitted.means, fitted.covars, strict=True)):
        log_emissions[is_complete, state] = stats.multivariate_normal.logpdf(
            complete_inputs, mean, covar
        )
    return log_emissions


def filter_probabilities(fitted: FittedHmm, inputs: np.ndarray) -> np.ndarray:
    """P(regime of day t | complete days up to t), by the forward algorithm.

    An incomplete day brings no observation: the filter only moves through the
    transition matrix, and the row of that day stays NaN.

    Args:
        fitted: The model.
        inputs: Rows of `hmm_inputs` for consecutive days, from the first day of the data.

    Returns:
        Array of shape (days, K); each known row sums to 1.
    """
    log_emissions = _log_emissions(fitted, inputs)
    is_complete = ~np.isnan(inputs).any(axis=1)
    with np.errstate(divide="ignore"):
        log_transmat = np.log(fitted.transmat)
        log_prior = np.log(fitted.startprob)
    log_filtered = np.empty_like(log_emissions)
    for position, log_emission in enumerate(log_emissions):
        is_first_day = position == 0
        if is_first_day:
            log_predicted = log_prior
        else:
            log_predicted = logsumexp(log_filtered[position - 1][:, None] + log_transmat, axis=0)
        log_joint = log_predicted + log_emission
        log_filtered[position] = log_joint - logsumexp(log_joint)
    probabilities = np.exp(log_filtered)
    probabilities[~is_complete] = np.nan
    return probabilities


def smoothed_probabilities(
    fitted: FittedHmm, inputs: np.ndarray, covariance_type: str
) -> np.ndarray:
    """P(regime of day t | all complete days), by the forward-backward algorithm.

    Uses the days after t: retrospective analysis only, never a prediction input.

    Args:
        fitted: The model.
        inputs: Rows of `hmm_inputs` for consecutive days.
        covariance_type: Covariance type the model was fitted with.

    Returns:
        Array of shape (days, K), NaN rows for incomplete days.
    """
    is_complete = ~np.isnan(inputs).any(axis=1)
    lengths = _complete_runs(is_complete)
    model = to_hmmlearn(fitted, covariance_type)
    posteriors = model.predict_proba(inputs[is_complete], lengths)
    probabilities = np.full((len(inputs), len(fitted.startprob)), np.nan)
    probabilities[is_complete] = posteriors
    return probabilities


def _most_probable(probabilities: np.ndarray) -> np.ndarray:
    """Most probable regime of each row, as floats; NaN where the row is unknown."""
    is_known = ~np.isnan(probabilities).any(axis=1)
    filled = np.nan_to_num(probabilities, nan=0.0)
    labels = filled.argmax(axis=1).astype(float)
    return np.where(is_known, labels, np.nan)


def regime_columns(kind: str, regime_names: list[str]) -> list[str]:
    """Label column then one probability column per regime.

    Args:
        kind: FILTERED or SMOOTHED.
        regime_names: Names in order of increasing volatility.

    Returns:
        e.g. ["regime_filtered", "prob_filtered_calm", "prob_filtered_volatile", ...].
    """
    probability_names = [f"prob_{kind}_{name}" for name in regime_names]
    return [f"regime_{kind}", *probability_names]


def _regime_table(
    probabilities: np.ndarray, days: pd.DatetimeIndex, kind: str, regime_names: list[str]
) -> pd.DataFrame:
    """Most probable regime (nullable integer) and the probability of each regime, per day."""
    label_column, *probability_columns = regime_columns(kind, regime_names)
    labels = pd.array(_most_probable(probabilities), dtype=LABEL_DTYPE)
    table = pd.DataFrame(probabilities, index=days, columns=probability_columns)
    labelled = table.assign(**{label_column: labels})
    return labelled[[label_column, *probability_columns]]


# ==================================================
# Filtered regime: monthly refit on the past, forward algorithm
# ==================================================


def _filtered_month_starts(days: pd.DatetimeIndex, min_train_months: int) -> list[pd.Timestamp]:
    """First day of each month that gets a filtered regime: after min_train_months of data."""
    months = days.to_period("M").unique()
    first_month = months[0] + min_train_months
    filtered_months = months[months >= first_month]
    return [month.start_time for month in filtered_months]


def _filter_month(
    fitted: FittedHmm, inputs: np.ndarray, days: pd.DatetimeIndex, month_start: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray]:
    """Filtered probabilities of the days of one month, with that month's model.

    The forward pass runs from the first day of the data to the end of the
    month: the probability of day d only uses the days up to d.

    Returns:
        Mask of the month's days over all days, and their probabilities.
    """
    next_month_start = month_start + pd.offsets.MonthBegin(1)
    is_until_month_end = np.asarray(days < next_month_start)
    is_month = is_until_month_end & np.asarray(days >= month_start)
    probabilities = filter_probabilities(fitted, inputs[is_until_month_end])
    month_probabilities = probabilities[is_month[is_until_month_end]]
    return is_month, month_probabilities


def _agreement(first: np.ndarray, second: np.ndarray) -> float | None:
    """Share of days with the same regime, among days labelled in both; None if none."""
    is_labelled = ~np.isnan(first) & ~np.isnan(second)
    if not is_labelled.any():
        return None
    is_same = first[is_labelled] == second[is_labelled]
    return _rounded(is_same.mean())


def _label_continuity(
    previous: FittedHmm | None,
    fitted: FittedHmm,
    training_inputs: np.ndarray,
    covariance_type: str,
) -> float | None:
    """Share of training days keeping their regime when this month's model replaces the last one.

    A low value flags two regimes swapping labels at a refit.
    """
    if previous is None:
        return None
    previous_probabilities = smoothed_probabilities(previous, training_inputs, covariance_type)
    new_probabilities = smoothed_probabilities(fitted, training_inputs, covariance_type)
    return _agreement(_most_probable(previous_probabilities), _most_probable(new_probabilities))


def _fit_record(
    month_start: pd.Timestamp,
    fitted: FittedHmm,
    previous: FittedHmm | None,
    training_inputs: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    """One line of the refit log: month, training size, fit quality, label continuity."""
    is_complete = ~np.isnan(training_inputs).any(axis=1)
    continuity = _label_continuity(previous, fitted, training_inputs, params["covariance_type"])
    return {
        "month": month_start.strftime("%Y-%m"),
        "training_days": int(is_complete.sum()),
        "log_likelihood": _rounded(fitted.log_likelihood, 1),
        "hit_iteration_cap": fitted.hit_iteration_cap,
        "previous_model_kept": fitted.from_previous,
        "label_continuity": continuity,
    }


def fit_filtered_regimes(
    regime_observations: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Filtered regime of each day: refit on the first day of each month, forward algorithm.

    Each month (from min_train_months after the first day of the data), the HMM
    is refitted on the previous days only, starting from last month's model and
    random starts. The forward algorithm then gives the probability of each
    regime on day d from the days up to d. Days before the first refit and
    incomplete days have no regime. The refits go on through the reserved
    period with the same settings (D9).

    Known: on day d, as soon as its observations are (12:45 on d-1).

    Args:
        regime_observations: Output of `build_regime_observations`.
        params: Regime parameters.

    Returns:
        The filtered regime table (regime_filtered, prob_filtered_*) and one
        record per monthly refit.
    """
    days = regime_observations.index
    inputs = hmm_inputs(regime_observations)
    n_states = len(params["regime_names"])
    probabilities = np.full((len(days), n_states), np.nan)
    fits = []
    previous = None
    for month_start in _filtered_month_starts(days, params["min_train_months"]):
        is_past = np.asarray(days < month_start)
        training_inputs = inputs[is_past]
        fitted = fit_hmm(training_inputs, n_states, params, params["random_state"], previous)
        is_month, month_probabilities = _filter_month(fitted, inputs, days, month_start)
        probabilities[is_month] = month_probabilities
        fits.append(_fit_record(month_start, fitted, previous, training_inputs, params))
        logger.debug("Refit %s: %s", fits[-1]["month"], fits[-1])
        previous = fitted
    table = _regime_table(probabilities, days, FILTERED, params["regime_names"])
    logger.info("Filtered regimes: %d monthly refits", len(fits))
    return table, fits


# ==================================================
# Smoothed regime and model summaries
# ==================================================


def _mean_duration_days(stay_probability: float) -> float | None:
    """Expected days in a regime once entered: 1 / (1 - probability of staying)."""
    leave_probability = 1.0 - float(stay_probability)
    if leave_probability <= 0:
        return None
    return _rounded(1.0 / leave_probability, 1)


def _model_summary(
    fitted: FittedHmm,
    observations: pd.DataFrame,
    labels: pd.Series,
    regime_names: list[str],
) -> dict[str, Any]:
    """Share of days, mean duration and mean observations (EUR/MWh) of each regime.

    The mean duration is None for a regime the model never leaves.
    """
    stay_probabilities = np.diag(fitted.transmat)
    labelled_days = int(labels.notna().sum())
    regimes = {}
    for state, name in enumerate(regime_names):
        is_state = labels.eq(state).fillna(False).to_numpy(dtype=bool)
        state_observations = observations[is_state]
        regimes[name] = {
            "share_of_days": _rounded(is_state.sum() / max(labelled_days, 1)),
            "mean_duration_days": _mean_duration_days(stay_probabilities[state]),
            "observation_means": {
                column: _rounded(state_observations[column].mean(), 2) for column in OBSERVATIONS
            },
        }
    transitions = {
        name: dict(zip(regime_names, map(_rounded, row), strict=True))
        for name, row in zip(regime_names, fitted.transmat, strict=True)
    }
    return {
        "regimes": regimes,
        "transition_matrix": transitions,
        "hit_iteration_cap": fitted.hit_iteration_cap,
    }


def fit_smoothed_regimes(
    regime_observations: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Smoothed regime of each day: one HMM on every day, forward-backward posterior.

    Uses the whole period, reserved period and future days included:
    retrospective analysis only (model C, notebook 3a), never for a prediction,
    an alert or the Isolation Forest.

    Args:
        regime_observations: Output of `build_regime_observations`.
        params: Regime parameters.

    Returns:
        The smoothed regime table (regime_smoothed, prob_smoothed_*) and a summary
        of the model.
    """
    days = regime_observations.index
    inputs = hmm_inputs(regime_observations)
    regime_names = params["regime_names"]
    fitted = fit_hmm(inputs, len(regime_names), params, params["random_state"])
    probabilities = smoothed_probabilities(fitted, inputs, params["covariance_type"])
    table = _regime_table(probabilities, days, SMOOTHED, regime_names)
    summary = _model_summary(fitted, regime_observations, table[REGIME_SMOOTHED], regime_names)
    logger.info("Smoothed regimes on %d days", int(table[REGIME_SMOOTHED].notna().sum()))
    return table, summary


# ==================================================
# Development period: everything that can inform a choice (D9)
# ==================================================


def _bic_selection(
    inputs: np.ndarray, params: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[int, FittedHmm]]:
    """BIC of each candidate number of states (lower is better), and the fitted models."""
    fits = {
        n_states: fit_hmm(inputs, n_states, params, params["random_state"])
        for n_states in params["bic_candidate_states"]
    }
    is_complete = ~np.isnan(inputs).any(axis=1)
    lengths = _complete_runs(is_complete)
    bic = {}
    for n_states, fitted in fits.items():
        model = to_hmmlearn(fitted, params["covariance_type"])
        bic[str(n_states)] = {
            "bic": _rounded(model.bic(inputs[is_complete], lengths), 1),
            "log_likelihood": _rounded(fitted.log_likelihood, 1),
            "hit_iteration_cap": fitted.hit_iteration_cap,
        }
    return bic, fits


def _stability(
    inputs: np.ndarray, labels: np.ndarray, params: dict[str, Any]
) -> dict[str, float | None]:
    """Agreement of the regimes with the same model fitted from other seeds."""
    n_states = len(params["regime_names"])
    agreements = {}
    for seed in params["stability_seeds"]:
        fitted = fit_hmm(inputs, n_states, params, seed)
        probabilities = smoothed_probabilities(fitted, inputs, params["covariance_type"])
        agreements[str(seed)] = _agreement(labels, _most_probable(probabilities))
    return agreements


def _moments(values: np.ndarray) -> dict[str, float | None]:
    """Skewness and excess kurtosis: both 0 for a Gaussian."""
    return {
        "skewness": _rounded(stats.skew(values), 2),
        "excess_kurtosis": _rounded(stats.kurtosis(values), 2),
    }


def _shape(
    inputs: np.ndarray, labels: np.ndarray, regime_names: list[str]
) -> dict[str, dict[str, Any] | None]:
    """Shape of each HMM input within each regime (volatility as log(1 + value))."""
    shape = {}
    for state, name in enumerate(regime_names):
        state_inputs = inputs[labels == state]
        has_enough_days = len(state_inputs) >= MIN_DAYS_FOR_SHAPE
        shape[name] = (
            {
                column: _moments(state_inputs[:, position])
                for position, column in enumerate(OBSERVATIONS)
            }
            if has_enough_days
            else None
        )
    return shape


def _share_by(
    labels: pd.Series, keys: pd.Index, regime_names: list[str]
) -> dict[str, dict[str, float | None]]:
    """Share of each regime among the labelled days of each group (year or calendar month)."""
    shares = pd.crosstab(keys, labels, normalize="index")
    all_states = shares.reindex(columns=range(len(regime_names)), fill_value=0.0)
    return {
        str(group): {name: _rounded(row[state]) for state, name in enumerate(regime_names)}
        for group, row in all_states.iterrows()
    }


def _filtered_vs_hindsight(filtered: pd.Series, hindsight: pd.Series) -> dict[str, Any]:
    """Share of days where the filtered regime matches the hindsight regime, overall and per year.

    Before the first crisis in the training days, the filtered regime can only
    be relative to the calm history seen so far: low early years show it.
    """
    years = filtered.index.year
    by_year = {
        str(year): _agreement(
            _labels_array(filtered[years == year]), _labels_array(hindsight[years == year])
        )
        for year in sorted(set(years))
    }
    return {
        "overall": _agreement(_labels_array(filtered), _labels_array(hindsight)),
        "by_year": by_year,
    }


def analyze_development_period(
    regime_observations: pd.DataFrame,
    filtered_regimes: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Everything that can inform a choice, on the days before holdout_start only (D9).

    - BIC of each candidate number of states (fitted on the development days);
    - the model with the configured number of states: regimes, transitions;
    - stability: agreement of its regimes with fits from other seeds;
    - shape of the HMM inputs within each regime (Gaussian = 0 and 0);
    - share of each regime per calendar month: are the regimes just seasons?
    - agreement of the filtered regime with this hindsight model, per year.

    Args:
        regime_observations: Output of `build_regime_observations`.
        filtered_regimes: First output of `fit_filtered_regimes`.
        params: Regime parameters.

    Returns:
        A JSON-ready section of the report.
    """
    holdout_start = pd.Timestamp(params["holdout_start"])
    development = regime_observations[regime_observations.index < holdout_start]
    inputs = hmm_inputs(development)
    regime_names = params["regime_names"]
    bic, fits = _bic_selection(inputs, params)
    configured = fits[len(regime_names)]
    probabilities = smoothed_probabilities(configured, inputs, params["covariance_type"])
    hindsight_table = _regime_table(probabilities, development.index, SMOOTHED, regime_names)
    smoothed_labels = hindsight_table[REGIME_SMOOTHED]
    label_values = _most_probable(probabilities)
    development_filtered = filtered_regimes.loc[development.index, REGIME_FILTERED]
    return {
        "last_day": str(development.index[-1].date()),
        "bic": bic,
        "bic_best_n_states": int(min(fits, key=lambda n_states: bic[str(n_states)]["bic"])),
        "configured_n_states": len(regime_names),
        "model": _model_summary(configured, development, smoothed_labels, regime_names),
        "stability_across_seeds": _stability(inputs, label_values, params),
        "shape_of_hmm_inputs": _shape(inputs, label_values, regime_names),
        "share_by_calendar_month": _share_by(
            smoothed_labels, smoothed_labels.index.month, regime_names
        ),
        "filtered_vs_hindsight": _filtered_vs_hindsight(development_filtered, smoothed_labels),
    }


# ==================================================
# Regimes table and report
# ==================================================


def build_regimes(
    regime_observations: pd.DataFrame,
    filtered_regimes: pd.DataFrame,
    smoothed_regimes: pd.DataFrame,
) -> pd.DataFrame:
    """Daily regimes table: the observations, the filtered regime, the smoothed regime.

    Args:
        regime_observations: Output of `build_regime_observations`.
        filtered_regimes: First output of `fit_filtered_regimes`.
        smoothed_regimes: First output of `fit_smoothed_regimes`.

    Returns:
        One row per calendar day (`day`, local date without time zone).
    """
    regimes = pd.concat([regime_observations, filtered_regimes, smoothed_regimes], axis=1)
    return regimes.rename_axis(DAY)


def _switches_per_year(labels: pd.Series) -> dict[str, int]:
    """Days whose regime differs from the day before (both labelled), per year."""
    previous_labels = labels.shift(1)
    is_switch = labels.ne(previous_labels).fillna(False).astype(bool)
    years = labels.index.year
    switches = is_switch.groupby(years).sum()
    has_labels = labels.notna().groupby(years).any()
    return {str(year): int(count) for year, count in switches[has_labels].items()}


def _filtered_summary(
    labels: pd.Series, fits: list[dict[str, Any]], regime_names: list[str]
) -> dict[str, Any]:
    """Shares, switches and refit quality of the filtered regime over one period."""
    labelled = labels.dropna()
    measured_fits = [fit for fit in fits if fit["label_continuity"] is not None]
    continuities = [fit["label_continuity"] for fit in measured_fits]
    low_continuity = [
        fit["month"] for fit in measured_fits if fit["label_continuity"] < LABEL_CONTINUITY_WARNING
    ]
    return {
        "first_day": str(labelled.index[0].date()) if len(labelled) else None,
        "labelled_days": len(labelled),
        "share_by_year": _share_by(labels, labels.index.year, regime_names),
        "switches_per_year": _switches_per_year(labels),
        "refits": len(fits),
        "refits_keeping_previous_model": sum(fit["previous_model_kept"] for fit in fits),
        "refits_at_iteration_cap": [fit["month"] for fit in fits if fit["hit_iteration_cap"]],
        "min_label_continuity": min(continuities, default=None),
        "months_below_label_continuity_warning": low_continuity,
    }


def _observations_summary(regimes: pd.DataFrame) -> dict[str, Any]:
    """Number of days, complete days, first and last complete day."""
    is_complete = regimes[list(OBSERVATIONS)].notna().all(axis=1)
    complete_days = regimes.index[is_complete]
    return {
        "days": len(regimes),
        "complete_days": len(complete_days),
        "first_complete_day": str(complete_days[0].date()),
        "last_complete_day": str(complete_days[-1].date()),
    }


def _labels_array(labels: pd.Series) -> np.ndarray:
    """Nullable integer labels as floats, NaN where unknown."""
    return labels.to_numpy(dtype=float, na_value=np.nan)


def _agreement_with_filtered(regimes: pd.DataFrame, is_development: np.ndarray) -> dict[str, Any]:
    """Share of days where the filtered and smoothed regimes agree, before and in the holdout."""
    filtered = _labels_array(regimes[REGIME_FILTERED])
    smoothed = _labels_array(regimes[REGIME_SMOOTHED])
    return {
        "before_holdout": _agreement(filtered[is_development], smoothed[is_development]),
        "holdout": _agreement(filtered[~is_development], smoothed[~is_development]),
    }


def _split_fits(
    fits: list[dict[str, Any]], holdout_start: pd.Timestamp
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refits of the months before the reserved period, and of the months inside it."""
    development_fits = [fit for fit in fits if pd.Timestamp(fit["month"]) < holdout_start]
    holdout_fits = [fit for fit in fits if pd.Timestamp(fit["month"]) >= holdout_start]
    return development_fits, holdout_fits


def build_regime_report(
    regimes: pd.DataFrame,
    regime_development: dict[str, Any],
    filtered_fits: list[dict[str, Any]],
    smoothed_model: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Report of the regimes, in three sections.

    - choices_before_holdout: the only section to look at when choosing
      (number of states, settings); nothing in it uses the reserved period.
    - holdout_descriptive_only: the filtered regime on the reserved period.
    - smoothed_descriptive_only: the hindsight model of the whole period.

    Args:
        regimes: Output of `build_regimes`.
        regime_development: Output of `analyze_development_period`.
        filtered_fits: Second output of `fit_filtered_regimes`.
        smoothed_model: Second output of `fit_smoothed_regimes`.
        params: Regime parameters.

    Returns:
        A JSON-ready report.
    """
    holdout_start = pd.Timestamp(params["holdout_start"])
    regime_names = params["regime_names"]
    is_development = np.asarray(regimes.index < holdout_start)
    development_fits, holdout_fits = _split_fits(filtered_fits, holdout_start)
    filtered = regimes[REGIME_FILTERED]
    smoothed = regimes[REGIME_SMOOTHED]
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "holdout_start": str(holdout_start.date()),
        "regime_names": list(regime_names),
        "observations": _observations_summary(regimes),
        "choices_before_holdout": {
            **regime_development,
            "filtered": _filtered_summary(filtered[is_development], development_fits, regime_names),
        },
        "holdout_descriptive_only": {
            "filtered": _filtered_summary(filtered[~is_development], holdout_fits, regime_names),
        },
        "smoothed_descriptive_only": {
            **smoothed_model,
            "share_by_year": _share_by(smoothed, smoothed.index.year, regime_names),
            "agreement_with_filtered": _agreement_with_filtered(regimes, is_development),
        },
        "filtered_refits": filtered_fits,
    }
    _log_report(report)
    return report


def _log_report(report: dict[str, Any]) -> None:
    """Main figures in the log; warnings on what needs a look before going on."""
    choices = report["choices_before_holdout"]
    bic_values = {n_states: entry["bic"] for n_states, entry in choices["bic"].items()}
    logger.info(
        "BIC before holdout %s: lowest at %d states (configured: %d)",
        bic_values,
        choices["bic_best_n_states"],
        choices["configured_n_states"],
    )
    logger.info(
        "Filtered vs hindsight regime before holdout: %s agreement",
        choices["filtered_vs_hindsight"]["overall"],
    )
    if choices["bic_best_n_states"] != choices["configured_n_states"]:
        logger.warning("The BIC prefers another number of states: decide before going on")
    stability = choices["stability_across_seeds"]
    logger.info("Agreement of the regimes across seeds: %s", stability)
    is_unstable = any(
        agreement is None or agreement < STABILITY_WARNING for agreement in stability.values()
    )
    if is_unstable:
        logger.warning("Regimes depend on the seed (agreement below %s)", STABILITY_WARNING)
    for section in ("choices_before_holdout", "holdout_descriptive_only"):
        low_months = report[section]["filtered"]["months_below_label_continuity_warning"]
        if low_months:
            logger.warning("%s: possible label swaps at the refits of %s", section, low_months)
    smoothed_shares = {
        name: regime["share_of_days"]
        for name, regime in report["smoothed_descriptive_only"]["regimes"].items()
    }
    logger.info("Smoothed regimes, share of days: %s", smoothed_shares)


def _rounded(value: Any, decimals: int = SHARE_DECIMALS) -> float | None:
    """Python float rounded for the JSON report; None if missing or infinite."""
    number = float(value)
    is_finite = bool(np.isfinite(number))
    return round(number, decimals) if is_finite else None
