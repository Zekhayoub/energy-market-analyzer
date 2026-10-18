"""Synthetic data shared by the tests.

Every instant is in UTC with its time zone (conventions.md, V3-1). Nothing
here calls the network. Each pipeline adds the helpers it needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from energy_market_analyzer.pipelines.features.nodes import (
    CLEAN_SPARK_SPREAD,
    DAY,
    PRICE_CHANGE,
    RESIDUAL_LOAD,
    build_daily_market,
    build_features_model_a,
    build_features_model_b,
    market_days,
)
from energy_market_analyzer.pipelines.ingestion.nodes import (
    EUA,
    LOAD,
    LOAD_FORECAST,
    NET_IMPORT_TOTAL,
    ONE_DAY,
    PRICE,
    TIMESTAMP,
    TRADING_DATE,
    TTF,
    WEATHER_VARIABLES,
    build_master,
    delivery_period,
    flow_column,
)
from energy_market_analyzer.pipelines.regime.nodes import (
    CLEAN_SPARK_SPREAD_MEAN,
    FILTERED,
    LABEL_DTYPE,
    PRICE_CHANGE_STD,
    PRICE_MEAN,
    REGIME_FILTERED,
    REGIME_SMOOTHED,
    SMOOTHED,
    regime_columns,
)

GENERATION_COLUMNS = {
    "Fossil Gas": "gen_gas_mw",
    "Nuclear": "gen_nuclear_mw",
    "Wind Onshore": "gen_wind_onshore_mw",
    "Wind Offshore": "gen_wind_offshore_mw",
    "Solar": "gen_solar_mw",
}
QUARTER_HOUR = "15min"
RANDOM_SEED = 42
REGIME_NAMES = ["calm", "volatile", "crisis"]
# Daily observations of three synthetic regimes, in order of increasing volatility:
# (mean, standard deviation). The volatility is drawn on the log(1 + value) scale
# the HMM uses, so each regime is exactly Gaussian for the HMM.
SYNTHETIC_REGIMES = (
    {PRICE_MEAN: (45.0, 8.0), PRICE_CHANGE_STD: (6.0, 0.25), CLEAN_SPARK_SPREAD_MEAN: (12.0, 5.0)},
    {
        PRICE_MEAN: (85.0, 15.0),
        PRICE_CHANGE_STD: (20.0, 0.25),
        CLEAN_SPARK_SPREAD_MEAN: (25.0, 10.0),
    },
    {
        PRICE_MEAN: (230.0, 50.0),
        PRICE_CHANGE_STD: (60.0, 0.25),
        CLEAN_SPARK_SPREAD_MEAN: (70.0, 30.0),
    },
)
MARKET_TIMEZONE = "Europe/Brussels"
# Hourly market of three synthetic regimes, in order of increasing volatility (EUR/MWh):
# mean price, hourly noise, fuel cost. The price follows a daily shape around its mean.
HOURLY_REGIMES = (
    {"price": 45.0, "noise": 5.0, "fuel": 30.0},
    {"price": 85.0, "noise": 12.0, "fuel": 60.0},
    {"price": 230.0, "noise": 35.0, "fuel": 180.0},
)
DAILY_SHAPE_SHARE = 0.15  # amplitude of the daily price shape, as a share of the mean price
ANOMALY_FEATURES = [PRICE, PRICE_CHANGE, CLEAN_SPARK_SPREAD, RESIDUAL_LOAD, NET_IMPORT_TOTAL]
# Prediction tests: ten months of market, tested from May, reserved period from September.
PREDICTION_START = "2018-01-01"
PREDICTION_END = "2018-10-31"
PREDICTION_HOLDOUT = "2018-09-01"
# Synthetic filtered regimes: blocks of days, the label at this probability, the
# rest shared by the other regimes; no regime during the first days, as in the data.
REGIME_BLOCK_DAYS = 9
REGIME_PROBABILITY = 0.8
DAYS_WITHOUT_REGIME = 20


def make_ingestion_params(**overrides: Any) -> dict[str, Any]:
    """Ingestion parameters for tests: short period, fast retries, no real URL called.

    Args:
        **overrides: Keys to replace, e.g. start_date="2025-10-25".

    Returns:
        A dict shaped like the `ingestion` section of parameters.yml.
    """
    base_params = {
        "country": "BE",
        "market_timezone": "Europe/Brussels",
        "start_date": "2025-09-29",
        "end_date": "2025-10-02",
        "quarter_hour_prices_from": "2025-10-01",
        "borders": {"FR": None, "NL": None, "DE_LU": "2020-11-18", "GB": "2019-01-31"},
        "generation_columns": GENERATION_COLUMNS,
        "zero_when_unpublished": ["gen_wind_offshore_mw"],
        "ttf_ticker": "TTF=F",
        "eua_contract": "T3PA",
        "eua_archive_url": "https://example.test/eex-archive.zip",
        "eua_archive_last_year": 2025,
        "eua_yearly_url": "https://example.test/eex-{year}.xlsx",
        "max_history_gap_days": 7,
        "entsoe_window_months": 6,
        "weather": {
            "latitude": 50.85,
            "longitude": 4.35,
            "model": "era5",
            "archive_url": "https://example.test/archive",
        },
        "retry_max_attempts": 2,
        "retry_backoff_seconds": 0,
        "http_timeout_seconds": 5,
    }
    return {**base_params, **overrides}


def utc_index(start: str, periods: int, freq: str = "h") -> pd.DatetimeIndex:
    """Regular UTC index named `timestamp`."""
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC", name=TIMESTAMP)


def _period_index(params: dict[str, Any], freq: str) -> pd.DatetimeIndex:
    """Every time unit of the delivery period, in UTC."""
    period_start, period_end = delivery_period(params)
    return pd.date_range(
        period_start.tz_convert("UTC"),
        period_end.tz_convert("UTC"),
        freq=freq,
        inclusive="left",
        name=TIMESTAMP,
    )


def _raw_prices(params: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Hourly prices before the quarter-hour go-live, quarter-hour prices after."""
    quarter_hours = _period_index(params, QUARTER_HOUR)
    go_live = pd.Timestamp(params["quarter_hour_prices_from"], tz=params["market_timezone"])
    is_hourly_market = quarter_hours < go_live
    is_hour_start = quarter_hours.minute == 0
    is_published = ~is_hourly_market | is_hour_start
    published_units = quarter_hours[is_published]
    prices = rng.normal(80, 20, len(published_units))
    return pd.DataFrame({PRICE: prices}, index=published_units)


def _in_service(hours: pd.DatetimeIndex, in_service_from: str | None, timezone: str) -> np.ndarray:
    """True for the hours from the commissioning day of a border (market time)."""
    if in_service_from is None:
        return np.full(len(hours), True)
    service_start = pd.Timestamp(in_service_from, tz=timezone)
    return np.asarray(hours >= service_start)


def _raw_flows(params: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Hourly physical flows, each border only from its commissioning day."""
    hours = _period_index(params, "h")
    flows = {}
    for neighbor, in_service_from in params["borders"].items():
        is_in_service = _in_service(hours, in_service_from, params["market_timezone"])
        for exporter, importer in ((neighbor, params["country"]), (params["country"], neighbor)):
            flow = pd.Series(rng.uniform(0, 1000, len(hours)), index=hours)
            flows[flow_column(exporter, importer)] = flow.where(is_in_service)
    return pd.DataFrame(flows)


def _daily_prices(params: dict[str, Any], column: str, level: float) -> pd.DataFrame:
    """One price per weekday of the period, indexed by trading date."""
    days = pd.date_range(params["start_date"], params["end_date"], freq="D", name=TRADING_DATE)
    is_weekday = days.dayofweek < 5
    trading_days = days[is_weekday]
    prices = level + np.arange(len(trading_days), dtype=float)
    return pd.DataFrame({column: prices}, index=trading_days)


def make_raw_sources(params: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Raw tables shaped like the outputs of the 8 fetch nodes, for the period of `params`.

    Returns:
        A dict keyed like the arguments of `build_master` (without `params`).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    quarter_hours = _period_index(params, QUARTER_HOUR)
    hours = _period_index(params, "h")
    generation = {
        column: rng.uniform(0, 3000, len(quarter_hours))
        for column in params["generation_columns"].values()
    }
    # Like Open-Meteo: whole UTC days, one more day on each side of the period.
    weather_hours = pd.date_range(
        hours[0].floor("D") - ONE_DAY,
        hours[-1].floor("D") + 2 * ONE_DAY,
        freq="h",
        inclusive="left",
        name=TIMESTAMP,
    )
    weather = {name: rng.normal(10, 5, len(weather_hours)) for name in WEATHER_VARIABLES}
    return {
        "prices": _raw_prices(params, rng),
        "load": pd.DataFrame(
            {LOAD: rng.normal(10000, 1500, len(quarter_hours))}, index=quarter_hours
        ),
        "load_forecast": pd.DataFrame(
            {LOAD_FORECAST: rng.normal(10000, 1500, len(quarter_hours))}, index=quarter_hours
        ),
        "generation": pd.DataFrame(generation, index=quarter_hours),
        "flows": _raw_flows(params, rng),
        "ttf": _daily_prices(params, TTF, level=40.0),
        "eua": _daily_prices(params, EUA, level=70.0),
        "weather": pd.DataFrame(weather, index=weather_hours),
    }


def make_features_params(**overrides: Any) -> dict[str, Any]:
    """Features parameters for tests, shaped like the `features` section of parameters.yml.

    Args:
        **overrides: Keys to replace, e.g. eua_max_age_days=3.

    Returns:
        A dict of features parameters.
    """
    base_params = {
        "market_timezone": "Europe/Brussels",
        "holdout_start": "2025-10-01",
        "holiday_country": "BE",
        "gas_column": "gen_gas_mw",
        "nuclear_column": "gen_nuclear_mw",
        "wind_solar_columns": ["gen_wind_onshore_mw", "gen_wind_offshore_mw", "gen_solar_mw"],
        "ttf_max_age_days": 7,
        "eua_max_age_days": 50,
        "heat_rate": 2.0,
        "emission_factor_t_per_mwh_gas": 0.202,
        "volatility_window_hours": 24,
        "multi_day_horizons": [2, 3, 4],
    }
    return {**base_params, **overrides}


def make_master(start_date: str, end_date: str) -> pd.DataFrame:
    """Hourly master table of the period, built by the real `build_master` from synthetic sources.

    Args:
        start_date: First delivery day (Brussels), e.g. "2025-10-20".
        end_date: Last delivery day (Brussels), included.

    Returns:
        The master table, as the ingestion pipeline writes it.
    """
    params = make_ingestion_params(start_date=start_date, end_date=end_date)
    sources = make_raw_sources(params)
    return build_master(**sources, params=params)


def make_regime_params(**overrides: Any) -> dict[str, Any]:
    """Regime parameters for tests: few starts and short training, to stay fast.

    Args:
        **overrides: Keys to replace, e.g. random_state=50.

    Returns:
        A dict shaped like the `regime` section of parameters.yml.
    """
    base_params = {
        "market_timezone": "Europe/Brussels",
        "holdout_start": "2020-01-01",
        "regime_names": REGIME_NAMES,
        "bic_candidate_states": [2, 3, 4],
        "min_train_months": 6,
        "covariance_type": "full",
        "min_covar": 0.001,
        "n_init": 3,
        "n_iter": 200,
        "tol": 0.001,
        "random_state": 0,
        "stability_seeds": [100],
    }
    return {**base_params, **overrides}


def _draw_observation(column: str, true_states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One observation of each day, drawn from the regime of that day."""
    means = np.array([regime[column][0] for regime in SYNTHETIC_REGIMES])[true_states]
    deviations = np.array([regime[column][1] for regime in SYNTHETIC_REGIMES])[true_states]
    if column != PRICE_CHANGE_STD:
        return rng.normal(means, deviations)
    log_volatility = rng.normal(np.log1p(means), deviations)
    return np.expm1(log_volatility)


def make_regime_observations(
    states: list[int], block_days: int, start_day: str = "2019-01-01"
) -> tuple[pd.DataFrame, np.ndarray]:
    """Daily HMM observations drawn from known regimes, one block of days per entry of `states`.

    Args:
        states: Regime of each block (0 = calm, 1 = volatile, 2 = crisis).
        block_days: Days per block.
        start_day: First calendar day.

    Returns:
        The observations, shaped like the output of `build_regime_observations`,
        and the true regime of each day.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    true_states = np.repeat(states, block_days)
    days = pd.date_range(start_day, periods=len(true_states), freq="D", name=DAY)
    observations = {
        column: _draw_observation(column, true_states, rng) for column in SYNTHETIC_REGIMES[0]
    }
    return pd.DataFrame(observations, index=days), true_states


def make_anomaly_params(**overrides: Any) -> dict[str, Any]:
    """Anomaly parameters for tests: small forests and short training, to stay fast.

    Args:
        **overrides: Keys to replace, e.g. min_training_hours=10**5.

    Returns:
        A dict shaped like the `anomaly` section of parameters.yml.
    """
    base_params = {
        "market_timezone": MARKET_TIMEZONE,
        "holdout_start": "2019-07-01",
        "regime_names": REGIME_NAMES,
        "features": list(ANOMALY_FEATURES),
        "min_train_months": 2,
        "min_training_hours": 240,
        "training_window_months": None,
        "window_candidates_months": [None, 1],
        "contamination": 0.05,
        "contamination_levels": [0.02, 0.05, 0.1],
        "n_estimators": 50,
        "max_samples": 128,
        "random_state": 0,
        "stability_seed": 1,
        "event_max_gap_hours": 4,
        "top_events": 5,
    }
    return {**base_params, **overrides}


def make_hourly_market(
    day_states: list[int], start_day: str = "2019-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly table C columns of the anomaly features and a regimes table, from known regimes.

    Args:
        day_states: Regime of each calendar day (0 = calm, 1 = volatile, 2 = crisis).
        start_day: First calendar day in Brussels.

    Returns:
        Table C (UTC hourly index, the columns of ANOMALY_FEATURES) and the daily
        regimes table, whose filtered and smoothed regimes are the true ones.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    days = pd.date_range(start_day, periods=len(day_states), freq="D", name=DAY)
    first_hour = pd.Timestamp(start_day, tz=MARKET_TIMEZONE)
    end_hour = pd.Timestamp(days[-1] + ONE_DAY, tz=MARKET_TIMEZONE)
    local_hours = pd.date_range(first_hour, end_hour, freq="h", inclusive="left")
    hours = local_hours.tz_convert("UTC").rename(TIMESTAMP)
    states = pd.Series(day_states, index=days).reindex(market_days(hours, MARKET_TIMEZONE))
    regime_values = {
        name: np.array([regime[name] for regime in HOURLY_REGIMES])[states.to_numpy()]
        for name in HOURLY_REGIMES[0]
    }
    daily_shape = np.sin((local_hours.hour.to_numpy() - 6) / 24 * 2 * np.pi)
    noise = rng.normal(0, regime_values["noise"])
    price = regime_values["price"] * (1 + DAILY_SHAPE_SHARE * daily_shape) + noise
    fuel = regime_values["fuel"] + rng.normal(0, 1, len(hours))
    residual_load = 8000 + 1500 * daily_shape + rng.normal(0, 400, len(hours))
    table_c = pd.DataFrame(
        {
            PRICE: price,
            CLEAN_SPARK_SPREAD: price - fuel,
            RESIDUAL_LOAD: residual_load,
            NET_IMPORT_TOTAL: rng.normal(500, 800, len(hours)),
        },
        index=hours,
    )
    with_change = table_c.assign(**{PRICE_CHANGE: table_c[PRICE].diff()})
    labels = pd.array(day_states, dtype=LABEL_DTYPE)
    regimes = pd.DataFrame({REGIME_FILTERED: labels, REGIME_SMOOTHED: labels}, index=days)
    return with_change[ANOMALY_FEATURES], regimes


def make_prediction_params(**overrides: Any) -> dict[str, Any]:
    """Prediction parameters for tests: few small trees, short windows, to stay fast.

    Args:
        **overrides: Keys to replace, e.g. training_window_months=2.

    Returns:
        A dict shaped like the `prediction` section of parameters.yml.
    """
    xgboost = {
        "n_estimators": 10,
        "learning_rate": 0.3,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }
    lightgbm = {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    model = {"tree_checkpoint_step": 5, "xgboost": xgboost, "lightgbm": lightgbm}
    base_params = {
        "market_timezone": MARKET_TIMEZONE,
        "holdout_start": PREDICTION_HOLDOUT,
        "regime_names": REGIME_NAMES,
        "min_train_months": 4,
        "training_window_months": None,
        "window_candidates_months": [None, 2],
        "development_epochs": ["2018-07-01"],
        "min_regime_training_hours": 240,
        "model_a": model,
        "model_b": model,
        "ridge_alpha": 1.0,
        "random_state": 42,
        "stability_seed": 123,
        "significance": 0.05,
        "min_dm_lag_days": 7,
        "interval": {
            "lower_quantile": 0.1,
            "upper_quantile": 0.9,
            "calibration_days": 14,
            "calibration_candidates_days": [7, 14],
            "min_calibration_days": 7,
            "min_width_share": 0.2,
            "min_raw_width_eur_mwh": 1.0,
        },
        "alerts": {
            "high_quantile": 0.8,
            "critical_quantile": 0.95,
            "reference_window_months": 2,
            "min_reference_hours": 48,
            "confidence_probabilities": [0.7, 0.9],
            "cheapest_hours": 8,
            "most_expensive_hours": 4,
        },
        "worst_days": 3,
    }
    return {**base_params, **overrides}


def make_filtered_regimes(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily regimes table with a known filtered regime: blocks cycling calm, volatile, crisis.

    Args:
        days: Calendar days.

    Returns:
        regime_filtered (Int8) and one probability per regime: REGIME_PROBABILITY on
        the label, the rest shared; missing on the first DAYS_WITHOUT_REGIME days.
        The smoothed columns repeat the filtered ones (tests change them to check
        they are never read).
    """
    filtered = _known_regimes(days)
    smoothed_columns = regime_columns(SMOOTHED, REGIME_NAMES)
    smoothed = filtered.set_axis(smoothed_columns, axis=1)
    return pd.concat([filtered, smoothed], axis=1)


def _known_regimes(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Filtered regime columns of `make_filtered_regimes`."""
    label_column, *probability_columns = regime_columns(FILTERED, REGIME_NAMES)
    blocks = np.arange(len(days)) // REGIME_BLOCK_DAYS
    labels = blocks % len(REGIME_NAMES)
    other_share = (1 - REGIME_PROBABILITY) / (len(REGIME_NAMES) - 1)
    probabilities = np.full((len(days), len(REGIME_NAMES)), other_share)
    probabilities[np.arange(len(days)), labels] = REGIME_PROBABILITY
    probabilities[:DAYS_WITHOUT_REGIME] = np.nan
    known_labels = np.where(np.arange(len(days)) < DAYS_WITHOUT_REGIME, np.nan, labels)
    table = pd.DataFrame(probabilities, index=days.rename(DAY), columns=probability_columns)
    table.insert(0, label_column, pd.array(known_labels, dtype=LABEL_DTYPE))
    return table


def make_prediction_market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Feature tables A and B, built by the real feature nodes, and a filtered regimes table.

    Returns:
        features_model_a, features_model_b and regimes, from PREDICTION_START to
        PREDICTION_END.
    """
    master = make_master(PREDICTION_START, PREDICTION_END)
    params = make_features_params(holdout_start=PREDICTION_HOLDOUT)
    daily_market = build_daily_market(master, params)
    features_model_a = build_features_model_a(master, daily_market, params)
    features_model_b = build_features_model_b(daily_market, params)
    days = pd.date_range(PREDICTION_START, PREDICTION_END, freq="D")
    return features_model_a, features_model_b, make_filtered_regimes(days)
