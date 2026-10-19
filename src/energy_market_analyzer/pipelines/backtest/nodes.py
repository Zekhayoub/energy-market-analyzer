"""Backtest nodes: hour-by-hour savings of an industrial client that shifts load (D31, D32).

Decision simulated (decisions.md, section 6): at 11:00 on J the client fixes the
hourly schedule of its flexible consumption for J+1 from the predicted prices
(its day-ahead bids close at 12:00), then pays the realized day-ahead prices.

- Reference: a flat consumption. Each MWh of the flexible share can be consumed
  up to flexibility_window_hours earlier or later, within the delivered day
  (Brussels), without the flexible equipment exceeding its maximum power. Moving
  one MWh costs shift_cost_eur_mwh.
- Strategies, compared on the same days: the optimal schedule for the selected
  prediction, for the similar-day naive and for the realized prices (perfect
  foresight, an unreachable ceiling); and the rule a procurement team applies to
  the published alerts (energy of the most expensive hours to the cheapest ones),
  with the selected and with the naive ranks.
- Savings of a day: cost of the flat profile minus cost of the schedule at the
  realized prices, minus the cost of the moves. A wrong prediction loses money:
  losses are counted, never clipped.
- Only perfect foresight reads a realized price to decide; the other strategies
  read predictions made at 11:00 J. Development and reserved period are split by
  the delivered day; the headline results are those of the reserved period (D9).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from functools import cache
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from energy_market_analyzer.pipelines.features.nodes import DECISION_DAY, TARGET_PRICE
from energy_market_analyzer.pipelines.ingestion.nodes import ONE_DAY
from energy_market_analyzer.pipelines.prediction.evaluation import (
    DAY_FORMAT,
    MONTH_FORMAT,
    SHARE_DECIMALS,
    diebold_mariano,
    holdout_day,
    long_run_variance,
    rounded,
)
from energy_market_analyzer.pipelines.prediction.models import (
    IS_CHEAPEST_HOUR,
    IS_MOST_EXPENSIVE_HOUR,
    NAIVE_SIMILAR_DAY,
    SELECTED,
    TARGET_DAY,
    prediction_column,
)
from energy_market_analyzer.pipelines.prediction.nodes import daily_ranks
from energy_market_analyzer.pipelines.regime.nodes import LABEL_DTYPE, REGIME_FILTERED

logger = logging.getLogger(__name__)

# --- Calendar ---
HOURS_PER_YEAR = 8760  # 365 days: the energy of one hour is the annual energy / 8760
DAYS_PER_YEAR = 365
MONTHS_PER_YEAR = 12
ONE_HOUR = pd.Timedelta(hours=1)

# --- Columns ---
REALIZED = TARGET_PRICE
PRED_SELECTED = prediction_column(SELECTED)
PRED_NAIVE = prediction_column(NAIVE_SIMILAR_DAY)
NAIVE_SUFFIX = "_naive"
NAIVE_CHEAPEST = IS_CHEAPEST_HOUR + NAIVE_SUFFIX
NAIVE_MOST_EXPENSIVE = IS_MOST_EXPENSIVE_HOUR + NAIVE_SUFFIX
FLAG_COLUMNS = (IS_CHEAPEST_HOUR, IS_MOST_EXPENSIVE_HOUR, NAIVE_CHEAPEST, NAIVE_MOST_EXPENSIVE)
PRICE_COLUMNS = (REALIZED, PRED_SELECTED, PRED_NAIVE)
HOURS = "hours"
FLEXIBLE_ENERGY = "flexible_energy_mwh"
CLIENT_COST = "client_cost_eur"
SHIFT_COST = "shift_cost_eur_mwh"
HOURS_DTYPE = "int64"
FLAG_DTYPE = "boolean"

# --- Strategies ---
OPTIMIZED = "optimized"
ALERT_RULE = "alert_rule"
OPTIMIZED_SELECTED = f"{OPTIMIZED}_{SELECTED}"
OPTIMIZED_NAIVE = f"{OPTIMIZED}{NAIVE_SUFFIX}"
PERFECT_FORESIGHT = "perfect_foresight"
ALERT_RULE_SELECTED = f"{ALERT_RULE}_{SELECTED}"
ALERT_RULE_NAIVE = f"{ALERT_RULE}{NAIVE_SUFFIX}"

# --- Optimizer ---
# A tiny cost per MWh and per hour of shift: among moves of equal predicted gain the
# shortest wins, and nothing moves for a predicted gain of zero.
TIE_BREAK_EUR_MWH_PER_HOUR = 1e-6
SOLVER = "highs"
SOLVED = 0  # status of scipy's linprog when an optimum is found
MOVE_TOLERANCE_MWH = 1e-9  # solver noise below this is no move
ENERGY_TOLERANCE_MWH = 1e-6
SAVINGS_TOLERANCE_EUR = 0.01  # one cent: smaller losses are zero, smaller excesses solver noise

# --- Report ---
EUR_DECIMALS = 2
ENERGY_DECIMALS = 1
UNKNOWN_REGIME = "unknown"
# Rules fixed before run 6 (decisions.md), read on the development period only:
# - without wind and solar forecasts (D3), the selected schedule capturing more than
#   90 % of perfect foresight would be too good: look for a leak before going on;
# - an edge over the naive (share of perfect foresight of the selected schedule minus the
#   naive's) larger by more than 0.1 on the reserved period is an alarm (a bug to look
#   for), never a reason to change a setting.
LEAK_SHARE = 0.9
HOLDOUT_GAP_ALARM = 0.1


@dataclass(frozen=True)
class Strategy:
    """How one strategy decides the schedule of a day.

    Attributes:
        name: Name in the columns and the report.
        signal: Column of the prices that decide the schedule.
        alert_flags: Columns of the cheapest and most expensive hour flags for the
            alert rule; None for the optimizer.
        description: One line for the report.
    """

    name: str
    signal: str
    alert_flags: tuple[str, str] | None
    description: str


STRATEGIES = (
    Strategy(
        OPTIMIZED_SELECTED,
        PRED_SELECTED,
        None,
        "optimal schedule for the selected prediction",
    ),
    Strategy(
        OPTIMIZED_NAIVE,
        PRED_NAIVE,
        None,
        "optimal schedule for the similar-day naive (what the client does without the project)",
    ),
    Strategy(
        PERFECT_FORESIGHT,
        REALIZED,
        None,
        "optimal schedule for the realized prices: unreachable ceiling",
    ),
    Strategy(
        ALERT_RULE_SELECTED,
        PRED_SELECTED,
        (IS_CHEAPEST_HOUR, IS_MOST_EXPENSIVE_HOUR),
        "most expensive hours of the published alerts to their cheapest hours",
    ),
    Strategy(
        ALERT_RULE_NAIVE,
        PRED_NAIVE,
        (NAIVE_CHEAPEST, NAIVE_MOST_EXPENSIVE),
        "same rule with the ranks of the similar-day naive",
    ),
)
STRATEGY_NAMES = tuple(strategy.name for strategy in STRATEGIES)
# Pairs compared on their daily savings: the value of the model for each way of
# scheduling, then the optimizer against the simpler rule (rule 3 of run 6).
COMPARED_PAIRS = (
    (OPTIMIZED_SELECTED, OPTIMIZED_NAIVE),
    (ALERT_RULE_SELECTED, ALERT_RULE_NAIVE),
    (OPTIMIZED_SELECTED, ALERT_RULE_SELECTED),
)


def savings_column(strategy: str) -> str:
    """Column of a strategy's daily savings, e.g. 'savings_optimized_selected_eur'."""
    return f"savings_{strategy}_eur"


def moved_column(strategy: str) -> str:
    """Column of the energy a strategy moves in a day, e.g. 'moved_mwh_optimized_selected'."""
    return f"moved_mwh_{strategy}"


def pair_key(first: str, second: str) -> str:
    """Report key of a pairwise comparison."""
    return f"{first}_vs_{second}"


# ==================================================
# Client and schedules
# ==================================================


@dataclass(frozen=True)
class Client:
    """Hourly energies of the client (MWh in one hour, i.e. MW over the hour).

    Attributes:
        hourly_load_mwh: Whole consumption of one hour (flat profile).
        flexible_mwh: Flexible energy of one hour in the reference schedule.
        max_flexible_mwh: Most flexible energy one hour can take.
        window_hours: Longest shift of one MWh, in hours.
    """

    hourly_load_mwh: float
    flexible_mwh: float
    max_flexible_mwh: float
    window_hours: int


@dataclass(frozen=True)
class Schedule:
    """Flexible energy consumed in each hour of a day, and the energy moved to get there."""

    flexible_load: np.ndarray
    moved_mwh: float


def client_profile(params: dict[str, Any]) -> Client:
    """Hourly energies of the client from its annual consumption and flexibility."""
    hourly_load = params["annual_consumption_mwh"] / HOURS_PER_YEAR
    flexible = hourly_load * params["flexible_share"]
    return Client(
        hourly_load_mwh=hourly_load,
        flexible_mwh=flexible,
        max_flexible_mwh=flexible * params["max_flexible_power_factor"],
        window_hours=params["flexibility_window_hours"],
    )


@cache
def transport_problem(hours: int, window: int) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    """Possible moves of a day and the constraints on them (the same for every day of a length).

    A move takes energy from a source hour to a destination hour at most `window`
    hours away. Rows 0 to hours - 1: energy taken out of an hour (at most its
    flexible energy). Rows hours to 2 x hours - 1: energy in minus energy out of an
    hour (at most the room under the maximum power).

    Returns:
        Source and destination hour of each move, and the constraint matrix.
    """
    positions = np.arange(hours)
    source_grid, destination_grid = np.meshgrid(positions, positions, indexing="ij")
    distances = np.abs(destination_grid - source_grid)
    is_move = (distances >= 1) & (distances <= window)
    sources = source_grid[is_move]
    destinations = destination_grid[is_move]
    moves = np.arange(len(sources))
    rows = np.concatenate([sources, hours + destinations, hours + sources])
    columns = np.concatenate([moves, moves, moves])
    values = np.concatenate([np.ones(len(moves)), np.ones(len(moves)), -np.ones(len(moves))])
    constraints = csr_matrix((values, (rows, columns)), shape=(2 * hours, len(moves)))
    return sources, destinations, constraints


def _apply_moves(
    sources: np.ndarray, destinations: np.ndarray, moves: np.ndarray, client: Client, hours: int
) -> np.ndarray:
    """Flexible energy of each hour after the moves."""
    outgoing = np.bincount(sources, weights=moves, minlength=hours)
    incoming = np.bincount(destinations, weights=moves, minlength=hours)
    return client.flexible_mwh - outgoing + incoming


def optimal_schedule(prices: np.ndarray, client: Client, shift_cost: float) -> Schedule:
    """Cheapest schedule of a day for `prices` (a transport problem, solved exactly).

    Minimizes the cost of the moves at `prices` plus shift_cost per MWh moved. Each
    MWh moves at most window_hours within the day; no hour exceeds the maximum power.

    Raises:
        RuntimeError: If the solver finds no optimum (never expected: no move is feasible).
    """
    hours = len(prices)
    sources, destinations, constraints = transport_problem(hours, client.window_hours)
    distances = np.abs(destinations - sources)
    move_costs = (
        prices[destinations] - prices[sources] + shift_cost + TIE_BREAK_EUR_MWH_PER_HOUR * distances
    )
    room = client.max_flexible_mwh - client.flexible_mwh
    limits = np.concatenate([np.full(hours, client.flexible_mwh), np.full(hours, room)])
    result = linprog(move_costs, A_ub=constraints, b_ub=limits, bounds=(0, None), method=SOLVER)
    if result.status != SOLVED:
        raise RuntimeError(f"The schedule optimizer failed: {result.message}")
    is_move = result.x > MOVE_TOLERANCE_MWH
    moves = np.where(is_move, result.x, 0.0)
    flexible_load = _apply_moves(sources, destinations, moves, client, hours)
    return Schedule(flexible_load, float(moves.sum()))


def alert_rule_schedule(
    prices: np.ndarray,
    is_cheapest: np.ndarray,
    is_most_expensive: np.ndarray,
    client: Client,
    shift_cost: float,
) -> Schedule:
    """Schedule of the alert rule: the energy of each flagged expensive hour, most expensive
    first, goes to the flagged cheap hours within the window, cheapest (then closest) first,
    up to the maximum power and only while the predicted gain exceeds the shift cost."""
    flexible_load = np.full(len(prices), client.flexible_mwh)
    expensive_hours = np.flatnonzero(is_most_expensive)
    cheapest_hours = np.flatnonzero(is_cheapest)
    by_price = np.argsort(-prices[expensive_hours], kind="stable")
    moved = 0.0
    for source in expensive_hours[by_price]:
        moved += _move_out(source, cheapest_hours, prices, flexible_load, client, shift_cost)
    return Schedule(flexible_load, moved)


def _move_out(
    source: int,
    cheapest_hours: np.ndarray,
    prices: np.ndarray,
    flexible_load: np.ndarray,
    client: Client,
    shift_cost: float,
) -> float:
    """Move the flexible energy of `source` to reachable cheap hours (updates flexible_load).

    Returns:
        Energy moved (MWh).
    """
    distances = np.abs(cheapest_hours - source)
    reachable = cheapest_hours[distances <= client.window_hours]
    order = np.lexsort((np.abs(reachable - source), prices[reachable]))
    moved = 0.0
    for destination in reachable[order]:
        gain = prices[source] - prices[destination]
        if gain <= shift_cost:
            break
        room = client.max_flexible_mwh - flexible_load[destination]
        amount = min(flexible_load[source], room)
        flexible_load[destination] += amount
        flexible_load[source] -= amount
        moved += amount
    return moved


def check_schedule(schedule: Schedule, client: Client) -> None:
    """Raise RuntimeError if a schedule creates or loses energy, or breaks the power limits."""
    load = schedule.flexible_load
    energy_gap = abs(load.sum() - client.flexible_mwh * len(load))
    is_over = load.max() > client.max_flexible_mwh + ENERGY_TOLERANCE_MWH
    is_negative = load.min() < -ENERGY_TOLERANCE_MWH
    if energy_gap > ENERGY_TOLERANCE_MWH or is_over or is_negative:
        raise RuntimeError(
            f"Invalid schedule: energy gap {energy_gap:.6f} MWh, hourly load between "
            f"{load.min():.4f} and {load.max():.4f} MWh (limit {client.max_flexible_mwh:.4f})"
        )


def day_savings(
    schedule: Schedule, realized: np.ndarray, client: Client, shift_cost: float
) -> float:
    """Cost of the flat profile minus cost of the schedule at the realized prices, minus the
    cost of the moves (EUR). Negative when the moves cost more than they save."""
    energy_removed = client.flexible_mwh - schedule.flexible_load
    avoided_cost = float(energy_removed @ realized)
    return avoided_cost - shift_cost * schedule.moved_mwh


def strategy_schedule(
    strategy: Strategy, day: dict[str, np.ndarray], client: Client, shift_cost: float
) -> Schedule:
    """Schedule a strategy decides for a day, from its own signal only."""
    prices = day[strategy.signal]
    if strategy.alert_flags is None:
        return optimal_schedule(prices, client, shift_cost)
    cheapest, most_expensive = strategy.alert_flags
    return alert_rule_schedule(prices, day[cheapest], day[most_expensive], client, shift_cost)


def evaluate_day(day: dict[str, np.ndarray], client: Client, shift_cost: float) -> dict[str, float]:
    """Savings and moved energy of every strategy on one day.

    Raises:
        RuntimeError: If a schedule is invalid, or a strategy saves more than perfect
            foresight (impossible: perfect foresight is optimal for the realized prices).
    """
    results = {}
    for strategy in STRATEGIES:
        schedule = strategy_schedule(strategy, day, client, shift_cost)
        check_schedule(schedule, client)
        savings = day_savings(schedule, day[REALIZED], client, shift_cost)
        results[savings_column(strategy.name)] = savings
        results[moved_column(strategy.name)] = schedule.moved_mwh
    ceiling = results[savings_column(PERFECT_FORESIGHT)]
    above_ceiling = [
        name
        for name in STRATEGY_NAMES
        if results[savings_column(name)] > ceiling + SAVINGS_TOLERANCE_EUR
    ]
    if above_ceiling:
        raise RuntimeError(f"Strategies saving more than perfect foresight: {above_ceiling}")
    return results


# ==================================================
# Inputs: predictions known at 11:00 J and the realized prices
# ==================================================


def _rank_params(params: dict[str, Any]) -> dict[str, Any]:
    """Parameters `daily_ranks` reads: numbers of cheapest and most expensive hours."""
    counts = {
        "cheapest_hours": params["cheapest_hours"],
        "most_expensive_hours": params["most_expensive_hours"],
    }
    return {"alerts": counts}


def _check_alerts(predictions: pd.DataFrame, alerts: pd.DataFrame, params: dict[str, Any]) -> None:
    """Raise ValueError unless the alerts are those of these predictions (not a stale file)."""
    if not alerts.index.equals(predictions.index):
        raise ValueError(
            "alerts and predictions_model_a do not cover the same hours: rerun the prediction "
            "pipeline"
        )
    ranks = daily_ranks(predictions[PRED_SELECTED], predictions[TARGET_DAY], _rank_params(params))
    for column in (IS_CHEAPEST_HOUR, IS_MOST_EXPENSIVE_HOUR):
        recomputed = ranks[column].astype(FLAG_DTYPE)
        published = alerts[column].astype(FLAG_DTYPE)
        if not recomputed.equals(published):
            raise ValueError(
                f"alerts.{column} does not match the selected predictions (check "
                f"cheapest_hours and most_expensive_hours, or rerun the prediction pipeline)"
            )


def expected_hours(days: pd.DatetimeIndex, timezone: str) -> pd.Series:
    """Hours of each day in `timezone`: 23 on the spring clock change, 25 in autumn, else 24."""
    starts = days.tz_localize(timezone)
    ends = (days + ONE_DAY).tz_localize(timezone)
    durations = (ends - starts) / ONE_HOUR
    return pd.Series(durations.astype(int), index=days)


def exclusion_reasons(hours: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Why each predicted day cannot be scheduled; missing when it can (first reason wins).

    A day needs all its hours, a realized price for each (the evaluation) and both
    predictions (the selected and the naive strategies are compared on the same days).
    """
    grouped = hours.groupby(TARGET_DAY, sort=True)
    rows = grouped.size()
    expected = expected_hours(pd.DatetimeIndex(rows.index), params["market_timezone"])
    prediction_counts = grouped[[PRED_SELECTED, PRED_NAIVE]].count().min(axis=1)
    checks = [
        ("missing_hours", rows.to_numpy() == expected.to_numpy()),
        ("missing_realized_price", (grouped[REALIZED].count() == rows).to_numpy()),
        ("missing_prediction", (prediction_counts == rows).to_numpy()),
    ]
    reasons = pd.Series(pd.NA, index=rows.index, dtype="string")
    for reason, is_ok in reversed(checks):
        reasons = reasons.mask(~is_ok, reason)
    return reasons


def _coverage(reasons: pd.Series, params: dict[str, Any]) -> dict[str, Any]:
    """Predicted, evaluated and excluded days, by period and reason."""
    is_evaluated = reasons.isna().to_numpy()
    evaluated_days = reasons.index[is_evaluated]
    is_development_day = evaluated_days < holdout_day(params)
    excluded = reasons.dropna()
    return {
        "predicted_days": len(reasons),
        "evaluated_days": len(evaluated_days),
        "development_days": int(is_development_day.sum()),
        "holdout_days": int((~is_development_day).sum()),
        "first_evaluated_day": _day_text(evaluated_days.min()) if len(evaluated_days) else None,
        "last_evaluated_day": _day_text(evaluated_days.max()) if len(evaluated_days) else None,
        "excluded_days_by_reason": {str(key): int(n) for key, n in excluded.value_counts().items()},
        "excluded_days": {_day_text(day): str(reason) for day, reason in excluded.items()},
    }


def _day_text(day: pd.Timestamp) -> str:
    """Calendar day as text for the report."""
    return pd.Timestamp(day).strftime(DAY_FORMAT)


def build_backtest_inputs(
    predictions_model_a: pd.DataFrame, alerts: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Hours of the days every strategy can schedule, and why the other days are left out.

    Known at 11:00 J: the selected and the naive predictions, and the cheapest and
    most expensive hours that come from them (published alerts for the selected
    model, recomputed the same way for the naive). The realized price is for the
    evaluation, and for perfect foresight only. realized_level is never read.

    Args:
        predictions_model_a: Hourly predictions of model A (prediction pipeline).
        alerts: Hourly alerts of model A (prediction pipeline).
        params: Backtest parameters.

    Returns:
        One row per hour of the evaluated days (UTC index, in time order), and the coverage.

    Raises:
        ValueError: If the alerts do not come from these predictions.
    """
    predictions = predictions_model_a.sort_index()
    published = alerts.sort_index()
    _check_alerts(predictions, published, params)
    rank_params = _rank_params(params)
    naive_ranks = daily_ranks(predictions[PRED_NAIVE], predictions[TARGET_DAY], rank_params)
    columns = {
        TARGET_DAY: predictions[TARGET_DAY],
        DECISION_DAY: predictions[DECISION_DAY],
        REGIME_FILTERED: predictions[REGIME_FILTERED].astype(LABEL_DTYPE),
        **{name: predictions[name].astype(float) for name in PRICE_COLUMNS},
        IS_CHEAPEST_HOUR: published[IS_CHEAPEST_HOUR].astype(FLAG_DTYPE),
        IS_MOST_EXPENSIVE_HOUR: published[IS_MOST_EXPENSIVE_HOUR].astype(FLAG_DTYPE),
        NAIVE_CHEAPEST: naive_ranks[IS_CHEAPEST_HOUR].astype(FLAG_DTYPE),
        NAIVE_MOST_EXPENSIVE: naive_ranks[IS_MOST_EXPENSIVE_HOUR].astype(FLAG_DTYPE),
    }
    hours = pd.DataFrame(columns, index=predictions.index)
    reasons = exclusion_reasons(hours, params)
    is_evaluated_day = hours[TARGET_DAY].map(reasons.isna()).to_numpy(dtype=bool)
    evaluated = hours[is_evaluated_day]
    coverage = _coverage(reasons, params)
    logger.info(
        "Backtest inputs: %d evaluated days, excluded: %s",
        coverage["evaluated_days"],
        coverage["excluded_days_by_reason"],
    )
    return evaluated, coverage


# ==================================================
# Savings of every strategy, day by day
# ==================================================


def split_days(hours: pd.DataFrame) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    """Arrays of each day's hours in time order, keyed by delivered day.

    Raises:
        ValueError: If an evaluated hour has no rank flag (a complete day is always ranked).
    """
    has_missing_flag = hours[list(FLAG_COLUMNS)].isna().any(axis=None)
    if has_missing_flag:
        raise ValueError("An evaluated hour has no cheapest/most expensive flag")
    days = {}
    for day, rows in hours.groupby(TARGET_DAY, sort=True):
        ordered = rows.sort_index()
        prices = {name: ordered[name].to_numpy(dtype=float) for name in PRICE_COLUMNS}
        flags = {name: ordered[name].to_numpy(dtype=bool) for name in FLAG_COLUMNS}
        days[day] = {**prices, **flags}
    return days


def day_context(hours: pd.DataFrame, client: Client) -> pd.DataFrame:
    """Decision day, regime of J, hours, flexible energy and bill of each evaluated day."""
    grouped = hours.groupby(TARGET_DAY, sort=True)
    hour_counts = grouped.size()
    realized_sums = grouped[REALIZED].sum()
    columns = {
        DECISION_DAY: grouped[DECISION_DAY].first(),
        REGIME_FILTERED: grouped[REGIME_FILTERED].first().astype(LABEL_DTYPE),
        HOURS: hour_counts.astype(HOURS_DTYPE),
        FLEXIBLE_ENERGY: hour_counts * client.flexible_mwh,
        CLIENT_COST: realized_sums * client.hourly_load_mwh,
    }
    return pd.DataFrame(columns).rename_axis(TARGET_DAY)


def savings_table(
    days: dict[pd.Timestamp, dict[str, np.ndarray]], client: Client, shift_cost: float
) -> pd.DataFrame:
    """Savings and moved energy of every strategy on every day, at one shift cost."""
    records = {day: evaluate_day(arrays, client, shift_cost) for day, arrays in days.items()}
    table = pd.DataFrame.from_dict(records, orient="index")
    return table.rename_axis(TARGET_DAY)


def run_backtest(
    backtest_inputs: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Savings of every strategy on every evaluated day, at each shift cost of the list.

    Args:
        backtest_inputs: First output of `build_backtest_inputs`.
        params: Backtest parameters.

    Returns:
        backtest_daily: one row per delivered day (index target_day), its context and
        the savings and moved energy of each strategy at the configured shift cost;
        and the savings at every shift cost of the list (index shift cost, day).
    """
    client = client_profile(params)
    days = split_days(backtest_inputs)
    context = day_context(backtest_inputs, client)
    by_level = {}
    for level in params["shift_cost_levels_eur_mwh"]:
        started = time.perf_counter()
        by_level[float(level)] = savings_table(days, client, float(level))
        seconds = time.perf_counter() - started
        logger.info("Shift cost %s EUR/MWh: %d days scheduled in %.1f s", level, len(days), seconds)
    configured = by_level[float(params["shift_cost_eur_mwh"])]
    backtest_daily = pd.concat([context, configured], axis=1)
    sensitivity = pd.concat(by_level, names=[SHIFT_COST, TARGET_DAY])
    return backtest_daily, sensitivity


# ==================================================
# Report
# ==================================================


def _share(numerator: float, denominator: float) -> float | None:
    """Ratio for the report; None when the denominator is not positive."""
    if denominator <= 0:
        return None
    return rounded(numerator / denominator, SHARE_DECIMALS)


def strategy_summary(daily: pd.DataFrame, strategy: str) -> dict[str, Any]:
    """Savings of one strategy on the days of `daily`, in EUR and per flexible MWh."""
    savings = daily[savings_column(strategy)]
    total = float(savings.sum())
    flexible_energy = float(daily[FLEXIBLE_ENERGY].sum())
    ceiling = float(daily[savings_column(PERFECT_FORESIGHT)].sum())
    moved = float(daily[moved_column(strategy)].sum())
    is_loss = savings < -SAVINGS_TOLERANCE_EUR
    return {
        "total_eur": rounded(total, EUR_DECIMALS),
        "per_day_eur": rounded(savings.mean(), EUR_DECIMALS) if len(daily) else None,
        "per_flexible_mwh_eur": rounded(total / flexible_energy) if flexible_energy else None,
        "share_of_perfect_foresight": _share(total, ceiling),
        "moved_share_of_flexible_energy": _share(moved, flexible_energy),
        "loss_days_share": rounded(is_loss.mean(), SHARE_DECIMALS) if len(daily) else None,
    }


def compare_strategies(
    daily: pd.DataFrame, first: str, second: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Daily gain of `first` over `second`, per year with its confidence interval, and the
    Diebold-Mariano test on the daily savings (Newey-West variance, as in the prediction report).

    In the test, the loss of a strategy is minus its savings: a positive mean
    difference means `first` saves less.
    """
    gains = daily[savings_column(first)] - daily[savings_column(second)]
    values = gains.to_numpy(dtype=float)
    days = len(values)
    if days == 0:
        return {"days": 0}
    test = diebold_mariano(-gains, params)
    lag = test["lag_days"]
    gain_per_year = DAYS_PER_YEAR * values.mean()
    variance = long_run_variance(values, lag) if days > lag + 1 else math.nan
    is_estimable = math.isfinite(variance) and variance > 0
    standard_error = math.sqrt(variance / days) if is_estimable else math.nan
    confidence = 1 - params["significance"]
    quantile = NormalDist().inv_cdf(1 - params["significance"] / 2)
    half_width = DAYS_PER_YEAR * quantile * standard_error
    is_better = test["p_value"] is not None and test["p_value"] < params["significance"]
    return {
        "days": days,
        "gain_total_eur": rounded(values.sum(), EUR_DECIMALS),
        "gain_per_year_eur": rounded(gain_per_year, EUR_DECIMALS),
        "confidence_level": rounded(confidence, SHARE_DECIMALS),
        "gain_per_year_interval_eur": [
            rounded(gain_per_year - half_width, EUR_DECIMALS),
            rounded(gain_per_year + half_width, EUR_DECIMALS),
        ],
        "diebold_mariano": test,
        "first_significantly_better": bool(is_better and gain_per_year > 0),
    }


def _year_keys(daily: pd.DataFrame) -> pd.Series:
    """Calendar year of each delivered day."""
    return pd.Series(daily.index.year.astype(str), index=daily.index)


def _month_keys(daily: pd.DataFrame) -> pd.Series:
    """Month of each delivered day."""
    return pd.Series(daily.index.strftime(MONTH_FORMAT), index=daily.index)


def _regime_keys(daily: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Name of the filtered regime of J of each delivered day."""
    names = dict(enumerate(params["regime_names"]))
    labels = daily[REGIME_FILTERED].astype(float)
    return labels.map(names).fillna(UNKNOWN_REGIME)


def totals_by(daily: pd.DataFrame, keys: pd.Series) -> dict[str, dict[str, Any]]:
    """Days and total savings of every strategy in each group of days."""
    summary = {}
    for key, rows in daily.groupby(keys, sort=True):
        totals = {
            name: rounded(rows[savings_column(name)].sum(), EUR_DECIMALS) for name in STRATEGY_NAMES
        }
        summary[str(key)] = {"days": len(rows), **totals}
    return summary


def client_summary(daily: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """The client's bill without shifting ("cost of doing nothing") and the share saved."""
    client = client_profile(params)
    consumption = float((daily[HOURS] * client.hourly_load_mwh).sum())
    bill = float(daily[CLIENT_COST].sum())
    saved_share = {
        name: _share(daily[savings_column(name)].sum(), bill)
        for name in (OPTIMIZED_SELECTED, ALERT_RULE_SELECTED, PERFECT_FORESIGHT)
    }
    return {
        "consumption_mwh": rounded(consumption, ENERGY_DECIMALS),
        "flexible_energy_mwh": rounded(daily[FLEXIBLE_ENERGY].sum(), ENERGY_DECIMALS),
        "bill_eur": rounded(bill, EUR_DECIMALS),
        "mean_price_paid_eur_mwh": rounded(bill / consumption) if consumption else None,
        "savings_share_of_bill": saved_share,
    }


def _payback(gain_per_year: float, params: dict[str, Any]) -> dict[str, Any]:
    """Months for a yearly gain to repay the project; None if the gain is not positive."""
    monthly_gain = gain_per_year / MONTHS_PER_YEAR
    is_paid_back = math.isfinite(monthly_gain) and monthly_gain > 0
    months = params["project_cost_eur"] / monthly_gain if is_paid_back else None
    return {
        "gain_per_year_eur": rounded(gain_per_year, EUR_DECIMALS),
        "payback_months": rounded(months, ENERGY_DECIMALS) if months is not None else None,
    }


def payback_summary(daily: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Payback of the project for each way of using the predictions: against doing nothing
    (every saving counts) and against the naive (only what the model adds counts)."""
    if daily.empty:
        return {}
    summary = {}
    pairs = ((OPTIMIZED_SELECTED, OPTIMIZED_NAIVE), (ALERT_RULE_SELECTED, ALERT_RULE_NAIVE))
    for strategy, naive in pairs:
        savings = daily[savings_column(strategy)]
        gains = savings - daily[savings_column(naive)]
        summary[strategy] = {
            "vs_no_shift": _payback(DAYS_PER_YEAR * float(savings.mean()), params),
            "vs_naive": _payback(DAYS_PER_YEAR * float(gains.mean()), params),
        }
    return {"project_cost_eur": rounded(params["project_cost_eur"], EUR_DECIMALS), **summary}


def sensitivity_summary(sensitivity: pd.DataFrame) -> dict[str, Any]:
    """Total savings and share of perfect foresight of every strategy at each shift cost."""
    summary = {}
    for level, rows in sensitivity.groupby(level=SHIFT_COST, sort=True):
        totals = {name: float(rows[savings_column(name)].sum()) for name in STRATEGY_NAMES}
        ceiling = totals[PERFECT_FORESIGHT]
        summary[f"{level:g}"] = {
            "savings_eur": {name: rounded(value, EUR_DECIMALS) for name, value in totals.items()},
            "share_of_perfect_foresight": {
                name: _share(value, ceiling) for name, value in totals.items()
            },
            "optimized_gain_over_naive_eur": rounded(
                totals[OPTIMIZED_SELECTED] - totals[OPTIMIZED_NAIVE], EUR_DECIMALS
            ),
        }
    return summary


def worst_days(daily: pd.DataFrame, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Days where the selected optimized schedule lost the most (or saved the least)."""
    client = client_profile(params)
    worst = daily[savings_column(OPTIMIZED_SELECTED)].nsmallest(params["worst_days"])
    regimes = _regime_keys(daily, params)
    consumption = daily[HOURS] * client.hourly_load_mwh
    mean_prices = daily[CLIENT_COST] / consumption
    return [
        {
            "day": _day_text(day),
            "regime": regimes[day],
            "mean_price_eur_mwh": rounded(mean_prices[day]),
            **{
                name: rounded(daily.loc[day, savings_column(name)], EUR_DECIMALS)
                for name in STRATEGY_NAMES
            },
        }
        for day in worst.index
    ]


def period_section(
    daily: pd.DataFrame, sensitivity: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Everything the report says about one period."""
    return {
        "days": len(daily),
        "strategies": {name: strategy_summary(daily, name) for name in STRATEGY_NAMES},
        "comparisons": {
            pair_key(first, second): compare_strategies(daily, first, second, params)
            for first, second in COMPARED_PAIRS
        },
        "by_year": totals_by(daily, _year_keys(daily)),
        "by_regime": totals_by(daily, _regime_keys(daily, params)),
        "client": client_summary(daily, params),
        "payback": payback_summary(daily, params),
        "shift_cost_sensitivity": sensitivity_summary(sensitivity),
        "worst_days": worst_days(daily, params),
    }


def rules_for_run_6(development: dict[str, Any]) -> dict[str, Any]:
    """Rules fixed before run 6, computed on the development period only.

    1. Leak alarm: the selected optimized schedule captures more than LEAK_SHARE of
       perfect foresight.
    2. Value of the model: the selected optimized schedule saves significantly more
       than the naive one. If not, it is reported as is; nothing is tuned.
    3. Strategy of the morning report (commit 7): the alert rule, simpler, unless the
       optimized schedule saves significantly more.
    """
    share = development["strategies"][OPTIMIZED_SELECTED]["share_of_perfect_foresight"]
    model_value = development["comparisons"][pair_key(OPTIMIZED_SELECTED, OPTIMIZED_NAIVE)]
    against_rule = development["comparisons"][pair_key(OPTIMIZED_SELECTED, ALERT_RULE_SELECTED)]
    is_optimized_better = bool(against_rule.get("first_significantly_better", False))
    return {
        "1_leak_alarm": {
            "share_of_perfect_foresight": share,
            "limit": LEAK_SHARE,
            "investigate_leak": share is not None and share > LEAK_SHARE,
        },
        "2_model_value": {
            "gain_per_year_eur": model_value.get("gain_per_year_eur"),
            "p_value": model_value.get("diebold_mariano", {}).get("p_value"),
            "model_beats_naive": bool(model_value.get("first_significantly_better", False)),
        },
        "3_morning_report_strategy": {
            "gain_of_optimized_per_year_eur": against_rule.get("gain_per_year_eur"),
            "p_value": against_rule.get("diebold_mariano", {}).get("p_value"),
            "optimized_significantly_better": is_optimized_better,
            "strategy": OPTIMIZED if is_optimized_better else ALERT_RULE,
        },
    }


def _edge_over_naive(section: dict[str, Any]) -> float | None:
    """Share of perfect foresight the selected schedule captures beyond the naive one."""
    strategies = section["strategies"]
    selected = strategies[OPTIMIZED_SELECTED]["share_of_perfect_foresight"]
    naive = strategies[OPTIMIZED_NAIVE]["share_of_perfect_foresight"]
    if selected is None or naive is None:
        return None
    return selected - naive


def holdout_alarms(development: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    """Alarm (a bug to look for) when the selected schedule's edge over the naive one is much
    larger on the reserved period than on the development period. Never a reason to change
    a setting.

    The edge, not the share itself: a market that becomes easier to schedule (a deeper
    solar trough, less hourly noise) lifts the naive as much as the model, while a leak of
    the realized prices lifts the model only.
    """
    development_edge = _edge_over_naive(development)
    holdout_edge = _edge_over_naive(holdout)
    is_known = development_edge is not None and holdout_edge is not None
    gap = holdout_edge - development_edge if is_known else None
    shares = {
        name: {
            "development": development["strategies"][name]["share_of_perfect_foresight"],
            "holdout": holdout["strategies"][name]["share_of_perfect_foresight"],
        }
        for name in (OPTIMIZED_SELECTED, OPTIMIZED_NAIVE)
    }
    return {
        "share_of_perfect_foresight": shares,
        "edge_over_naive": {
            "development": rounded(development_edge, SHARE_DECIMALS) if is_known else None,
            "holdout": rounded(holdout_edge, SHARE_DECIMALS) if is_known else None,
        },
        "gap": rounded(gap, SHARE_DECIMALS) if gap is not None else None,
        "limit": HOLDOUT_GAP_ALARM,
        "investigate_bug": gap is not None and gap > HOLDOUT_GAP_ALARM,
    }


def _client_header(params: dict[str, Any]) -> dict[str, Any]:
    """Client assumptions, as configured and in hourly energies."""
    client = client_profile(params)
    flexible_per_year = client.flexible_mwh * HOURS_PER_YEAR
    return {
        "annual_consumption_mwh": params["annual_consumption_mwh"],
        "flexible_share": params["flexible_share"],
        "flexible_energy_mwh_per_year": rounded(flexible_per_year, ENERGY_DECIMALS),
        "average_load_mw": rounded(client.hourly_load_mwh),
        "average_flexible_power_mw": rounded(client.flexible_mwh),
        "max_flexible_power_mw": rounded(client.max_flexible_mwh),
        "flexibility_window_hours": client.window_hours,
        "shift_cost_eur_mwh": params["shift_cost_eur_mwh"],
        "project_cost_eur": params["project_cost_eur"],
    }


def build_backtest_report(
    backtest_daily: pd.DataFrame,
    backtest_sensitivity: pd.DataFrame,
    backtest_coverage: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Report of the backtest, in two sections.

    - choices_before_holdout: the only section to read when choosing (rules of run 6);
      nothing in it uses a delivered day of the reserved period.
    - holdout_descriptive_only: the headline results (12 months, no extrapolation),
      read after the choices.

    Args:
        backtest_daily: First output of `run_backtest`.
        backtest_sensitivity: Second output of `run_backtest`.
        backtest_coverage: Second output of `build_backtest_inputs`.
        params: Backtest parameters.

    Returns:
        A JSON-ready report.
    """
    first_holdout_day = holdout_day(params)
    is_development_day = backtest_daily.index < first_holdout_day
    sensitivity_days = backtest_sensitivity.index.get_level_values(TARGET_DAY)
    is_development_row = sensitivity_days < first_holdout_day
    development = period_section(
        backtest_daily[is_development_day], backtest_sensitivity[is_development_row], params
    )
    development["rules_for_run_6"] = rules_for_run_6(development)
    holdout_daily = backtest_daily[~is_development_day]
    holdout = period_section(holdout_daily, backtest_sensitivity[~is_development_row], params)
    holdout["by_month"] = totals_by(holdout_daily, _month_keys(holdout_daily))
    holdout["alarms"] = holdout_alarms(development, holdout)
    report = {
        "holdout_start": str(params["holdout_start"]),
        "client": _client_header(params),
        "strategies": {strategy.name: strategy.description for strategy in STRATEGIES},
        "coverage": backtest_coverage,
        "choices_before_holdout": development,
        "holdout_descriptive_only": holdout,
    }
    _log_report(report)
    return report


def _log_report(report: dict[str, Any]) -> None:
    """Main figures in the log; warnings on the rules of run 6 that ask for attention."""
    for section in ("choices_before_holdout", "holdout_descriptive_only"):
        strategies = report[section]["strategies"]
        logger.info(
            "%s: savings (EUR) %s",
            section,
            {name: values["total_eur"] for name, values in strategies.items()},
        )
    rules = report["choices_before_holdout"]["rules_for_run_6"]
    logger.info("Rules of run 6: %s", rules)
    if rules["1_leak_alarm"]["investigate_leak"]:
        logger.warning("Share of perfect foresight too high: look for a leak before going on")
    if not rules["2_model_value"]["model_beats_naive"]:
        logger.warning("The selected schedule does not save significantly more than the naive")
    alarms = report["holdout_descriptive_only"]["alarms"]
    if alarms["investigate_bug"]:
        logger.warning("Reserved period far better than development: look for a bug: %s", alarms)
