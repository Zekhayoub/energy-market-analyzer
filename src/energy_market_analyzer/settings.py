"""Kedro project settings: parameter validation and the ENTSO-E key hook.

Kedro reads this module at startup. It defines:
1. Pydantic models that validate parameters.yml before any node runs.
   A wrong value or an unknown key (a typo) stops the run at once,
   with a message that names the faulty parameter.
2. Hooks that hand the ENTSO-E key to the pipelines as a masked value
   and ask for it only when the pipeline being run needs it.

Each pipeline adds the model of its own parameter section here.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import holidays
from kedro.framework.hooks import hook_impl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

# Name of the key in conf/local/credentials.yml and of the catalog entry fed to the nodes.
ENTSOE_API_KEY = "entsoe_api_key"
MISSING_KEY_MESSAGE = (
    "The ENTSO-E API key is missing. Create conf/local/credentials.yml containing the line "
    f"`{ENTSOE_API_KEY}: <your key>` (see README.md, section Installation)."
)

ENTSOE_ZONE_PATTERN = r"^[A-Z]{2}(_[A-Z]{2,3})*$"
GENERATION_COLUMN_PATTERN = r"^gen_[a-z0-9_]+_mw$"
HTTPS_URL_PATTERN = r"^https://"
# Model B predicts from J+2 on: J+1 is model A. Up to J+7: the price of the
# predicted day one week before must be known on the decision day.
MIN_MULTI_DAY_HORIZON = 2
MAX_MULTI_DAY_HORIZON = 7
# Regime names become column suffixes (prob_filtered_<name>); an HMM needs 2 states or more.
REGIME_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
MIN_REGIME_STATES = 2
MONTHS_PER_YEAR = 12
# Isolation Forest inputs are table C column names; one feature isolates nothing interesting.
COLUMN_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
MIN_ANOMALY_FEATURES = 2
# scikit-learn's Isolation Forest accepts a contamination up to 0.5.
MAX_CONTAMINATION = 0.5
# The point prediction is a median: the interval bounds lie on each side of it,
# the alert thresholds above it. The shortest day (spring clock change) has 23 hours.
MEDIAN = 0.5
SHORTEST_DAY_HOURS = 23
# A test needs at least two points on the tree curve (rule 3 of run 5).
MIN_TREE_CHECKPOINTS = 2
# Backtest: a shift stays inside the delivered day; with a maximum power equal to the
# average, no hour could take more energy and nothing could move.
MAX_FLEXIBILITY_WINDOW_HOURS = SHORTEST_DAY_HOURS - 1
MIN_FLEXIBLE_POWER_FACTOR = 1.0
# Interpretability: shap draws at most 100 background rows (its masker's default); the day of
# the method check exists in every month.
MAX_BACKGROUND_ROWS = 100
MIN_BACKGROUND_ROWS = 10
LAST_DAY_IN_EVERY_MONTH = 28


# ==================================================
# Parameter models
# ==================================================


class StrictModel(BaseModel):
    """Base of all parameter models: an unknown key is an error, not ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class WeatherConfig(StrictModel):
    """Point and model of the observed weather (Open-Meteo archive)."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    model: str = Field(..., min_length=1)
    archive_url: str = Field(..., pattern=HTTPS_URL_PATTERN)


class IngestionConfig(StrictModel):
    """Validates the `ingestion` section of parameters.yml."""

    country: str = Field(..., pattern=ENTSOE_ZONE_PATTERN)
    market_timezone: str
    start_date: date
    end_date: date
    quarter_hour_prices_from: date
    borders: dict[str, date | None] = Field(..., min_length=1)
    generation_columns: dict[str, str] = Field(..., min_length=1)
    zero_when_unpublished: list[str]
    ttf_ticker: str = Field(..., min_length=1)
    eua_contract: str = Field(..., min_length=1)
    eua_archive_url: str = Field(..., pattern=HTTPS_URL_PATTERN)
    eua_archive_last_year: int = Field(..., ge=2012)
    eua_yearly_url: str = Field(..., pattern=HTTPS_URL_PATTERN + r".*\{year\}")
    max_history_gap_days: int = Field(..., ge=0)
    entsoe_window_months: int = Field(..., ge=1, le=11)
    weather: WeatherConfig
    retry_max_attempts: int = Field(..., ge=1)
    retry_backoff_seconds: float = Field(..., ge=0)
    http_timeout_seconds: float = Field(..., gt=0)

    @field_validator("market_timezone")
    @classmethod
    def timezone_exists(cls, value: str) -> str:
        """Reject a time zone name that the tz database does not know."""
        return _known_timezone(value)

    @field_validator("borders")
    @classmethod
    def border_codes_are_entsoe_zones(cls, value: dict[str, date | None]) -> dict[str, date | None]:
        """Each neighbour must be written as an ENTSO-E zone code, e.g. 'FR' or 'DE_LU'."""
        invalid_codes = [code for code in value if not _is_entsoe_zone(code)]
        if invalid_codes:
            raise ValueError(f"Invalid ENTSO-E zone codes in borders: {invalid_codes}")
        return value

    @field_validator("generation_columns")
    @classmethod
    def generation_columns_are_unique(cls, value: dict[str, str]) -> dict[str, str]:
        """Column names must follow gen_<name>_mw and appear only once."""
        invalid_names = [name for name in value.values() if not _is_generation_column(name)]
        if invalid_names:
            raise ValueError(f"Generation columns must look like gen_<name>_mw: {invalid_names}")
        has_duplicates = len(set(value.values())) < len(value)
        if has_duplicates:
            raise ValueError("Two production types are mapped to the same generation column")
        return value

    @model_validator(mode="after")
    def dates_are_consistent(self) -> IngestionConfig:
        """Consistent period, borders in service before its end, known zero columns."""
        if self.end_date < self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must not be before start_date ({self.start_date})"
            )
        unknown_zero_columns = sorted(
            set(self.zero_when_unpublished) - set(self.generation_columns.values())
        )
        if unknown_zero_columns:
            raise ValueError(
                f"zero_when_unpublished lists columns not in generation_columns: "
                f"{unknown_zero_columns}"
            )
        if self.country in self.borders:
            raise ValueError(f"borders must not contain the country itself ({self.country})")
        late_borders = [
            zone
            for zone, since in self.borders.items()
            if since is not None and since > self.end_date
        ]
        if late_borders:
            raise ValueError(f"Borders put in service after end_date: {late_borders}")
        return self


class FeaturesConfig(StrictModel):
    """Validates the `features` section of parameters.yml."""

    market_timezone: str
    holdout_start: date
    holiday_country: str = Field(..., min_length=2)
    gas_column: str = Field(..., pattern=GENERATION_COLUMN_PATTERN)
    nuclear_column: str = Field(..., pattern=GENERATION_COLUMN_PATTERN)
    wind_solar_columns: list[str] = Field(..., min_length=1)
    ttf_max_age_days: int = Field(..., ge=1)
    eua_max_age_days: int = Field(..., ge=1)
    # Realistic gas plants: 1.6 (60 % efficiency) to 2.9 (35 %); catches typos like 20.
    heat_rate: float = Field(..., ge=1.5, le=3.5)
    emission_factor_t_per_mwh_gas: float = Field(..., gt=0, le=0.5)
    volatility_window_hours: int = Field(..., ge=2)
    multi_day_horizons: list[int] = Field(..., min_length=1)

    @field_validator("market_timezone")
    @classmethod
    def timezone_exists(cls, value: str) -> str:
        """Reject a time zone name that the tz database does not know."""
        return _known_timezone(value)

    @field_validator("holiday_country")
    @classmethod
    def holiday_calendar_exists(cls, value: str) -> str:
        """Reject a country code that the `holidays` package does not know."""
        try:
            holidays.country_holidays(value)
        except NotImplementedError as error:
            raise ValueError(f"No public holiday calendar for '{value}'") from error
        return value

    @field_validator("wind_solar_columns")
    @classmethod
    def wind_solar_columns_are_generation(cls, value: list[str]) -> list[str]:
        """Each column must follow gen_<name>_mw and appear only once."""
        invalid_names = [name for name in value if not _is_generation_column(name)]
        if invalid_names:
            raise ValueError(f"wind_solar_columns must look like gen_<name>_mw: {invalid_names}")
        if len(set(value)) < len(value):
            raise ValueError("wind_solar_columns lists a column twice")
        return value

    @field_validator("multi_day_horizons")
    @classmethod
    def horizons_fit_model_b(cls, value: list[int]) -> list[int]:
        """Model B covers J+2 to J+7 (J+1 is model A); each horizon appears once."""
        is_too_short = min(value) < MIN_MULTI_DAY_HORIZON
        is_too_long = max(value) > MAX_MULTI_DAY_HORIZON
        if is_too_short or is_too_long:
            raise ValueError(
                f"multi_day_horizons must lie between {MIN_MULTI_DAY_HORIZON} (J+1 is model A) "
                f"and {MAX_MULTI_DAY_HORIZON} (week-ago price known on the decision day): {value}"
            )
        if len(set(value)) < len(value):
            raise ValueError(f"multi_day_horizons lists a horizon twice: {value}")
        return value

    def generation_columns(self) -> list[str]:
        """Every master generation column the features read."""
        return [self.gas_column, self.nuclear_column, *self.wind_solar_columns]


class RegimeConfig(StrictModel):
    """Validates the `regime` section of parameters.yml."""

    market_timezone: str
    holdout_start: date
    regime_names: list[str] = Field(..., min_length=MIN_REGIME_STATES)
    bic_candidate_states: list[int] = Field(..., min_length=1)
    min_train_months: int = Field(..., ge=1)
    covariance_type: Literal["full", "diag"]
    min_covar: float = Field(..., gt=0)
    n_init: int = Field(..., ge=1)
    n_iter: int = Field(..., ge=1)
    tol: float = Field(..., gt=0)
    random_state: int = Field(..., ge=0)
    stability_seeds: list[int] = Field(..., min_length=1)

    @field_validator("market_timezone")
    @classmethod
    def timezone_exists(cls, value: str) -> str:
        """Reject a time zone name that the tz database does not know."""
        return _known_timezone(value)

    @field_validator("regime_names")
    @classmethod
    def names_are_column_suffixes(cls, value: list[str]) -> list[str]:
        """Each name ends a column name (prob_filtered_<name>): lowercase and unique."""
        invalid_names = [name for name in value if not re.fullmatch(REGIME_NAME_PATTERN, name)]
        if invalid_names:
            raise ValueError(f"regime_names must be lowercase identifiers: {invalid_names}")
        if len(set(value)) < len(value):
            raise ValueError(f"regime_names lists a name twice: {value}")
        return value

    @field_validator("bic_candidate_states")
    @classmethod
    def candidates_are_hmms(cls, value: list[int]) -> list[int]:
        """An HMM has at least 2 states; each candidate appears once."""
        if min(value) < MIN_REGIME_STATES:
            raise ValueError(f"bic_candidate_states must be at least {MIN_REGIME_STATES}: {value}")
        if len(set(value)) < len(value):
            raise ValueError(f"bic_candidate_states lists a number twice: {value}")
        return value

    @model_validator(mode="after")
    def states_and_seeds_agree(self) -> RegimeConfig:
        """The configured number of states has a BIC; random starts of the seeds never overlap."""
        n_states = len(self.regime_names)
        if n_states not in self.bic_candidate_states:
            raise ValueError(
                f"bic_candidate_states must include the number of regime_names ({n_states})"
            )
        seeds = sorted([self.random_state, *self.stability_seeds])
        gaps = [later - earlier for earlier, later in zip(seeds, seeds[1:], strict=False)]
        if min(gaps) < self.n_init:
            raise ValueError(
                f"random_state and stability_seeds must be at least n_init ({self.n_init}) "
                f"apart, else their random starts overlap: {seeds}"
            )
        return self


class AnomalyConfig(StrictModel):
    """Validates the `anomaly` section of parameters.yml.

    That each feature is a table C column, and not a calendar one, is checked by
    the first node of the pipeline, which reads table C.
    """

    market_timezone: str
    holdout_start: date
    regime_names: list[str] = Field(..., min_length=MIN_REGIME_STATES)
    features: list[str] = Field(..., min_length=MIN_ANOMALY_FEATURES)
    min_train_months: int = Field(..., ge=1)
    min_training_hours: int = Field(..., ge=1)
    training_window_months: int | None = Field(..., ge=1)
    window_candidates_months: list[int | None] = Field(..., min_length=1)
    contamination: float = Field(..., gt=0, lt=MAX_CONTAMINATION)
    contamination_levels: list[float] = Field(..., min_length=1)
    n_estimators: int = Field(..., ge=1)
    max_samples: int = Field(..., ge=2)
    random_state: int = Field(..., ge=0)
    stability_seed: int = Field(..., ge=0)
    event_max_gap_hours: int = Field(..., ge=0)
    top_events: int = Field(..., ge=1)

    @field_validator("market_timezone")
    @classmethod
    def timezone_exists(cls, value: str) -> str:
        """Reject a time zone name that the tz database does not know."""
        return _known_timezone(value)

    @field_validator("features")
    @classmethod
    def features_are_columns(cls, value: list[str]) -> list[str]:
        """Each feature is written like a column name and appears once."""
        invalid_names = [name for name in value if not re.fullmatch(COLUMN_NAME_PATTERN, name)]
        if invalid_names:
            raise ValueError(f"features must be table C column names: {invalid_names}")
        if len(set(value)) < len(value):
            raise ValueError(f"features lists a column twice: {value}")
        return value

    @field_validator("window_candidates_months")
    @classmethod
    def candidates_are_windows(cls, value: list[int | None]) -> list[int | None]:
        """Each candidate is null (every past hour) or at least 1 month, and appears once."""
        invalid_windows = [window for window in value if window is not None and window < 1]
        if invalid_windows:
            raise ValueError(f"window_candidates_months must be null or >= 1: {invalid_windows}")
        if len(set(value)) < len(value):
            raise ValueError(f"window_candidates_months lists a window twice: {value}")
        return value

    @field_validator("contamination_levels")
    @classmethod
    def levels_are_contaminations(cls, value: list[float]) -> list[float]:
        """Each level is a valid contamination and appears once."""
        invalid_levels = [level for level in value if not 0 < level < MAX_CONTAMINATION]
        if invalid_levels:
            raise ValueError(
                f"contamination_levels must lie strictly between 0 and {MAX_CONTAMINATION}: "
                f"{invalid_levels}"
            )
        if len(set(value)) < len(value):
            raise ValueError(f"contamination_levels lists a level twice: {value}")
        return value

    @model_validator(mode="after")
    def settings_agree(self) -> AnomalyConfig:
        """The report covers the configured level; each model has enough hours to fit and flag."""
        if self.contamination not in self.contamination_levels:
            raise ValueError(
                f"contamination_levels must include contamination ({self.contamination})"
            )
        if self.training_window_months not in self.window_candidates_months:
            raise ValueError(
                f"window_candidates_months must include training_window_months "
                f"({self.training_window_months})"
            )
        if self.max_samples > self.min_training_hours:
            raise ValueError(
                f"max_samples ({self.max_samples}) must not exceed min_training_hours "
                f"({self.min_training_hours}): each tree draws max_samples training hours"
            )
        smallest_flagged = self.min_training_hours * min(self.contamination_levels)
        if smallest_flagged < 1:
            raise ValueError(
                "min_training_hours x the smallest contamination level must be at least 1 hour"
            )
        if self.stability_seed == self.random_state:
            raise ValueError("stability_seed must differ from random_state")
        return self


class XGBoostConfig(StrictModel):
    """Fixed hyperparameters of the XGBoost models of one task (never tuned on tested months)."""

    n_estimators: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0, le=1)
    max_depth: int = Field(..., ge=1)
    min_child_weight: float = Field(..., ge=0)
    subsample: float = Field(..., gt=0, le=1)
    colsample_bytree: float = Field(..., gt=0, le=1)
    reg_lambda: float = Field(..., ge=0)


class LightGBMConfig(StrictModel):
    """Fixed hyperparameters of the LightGBM quantile models of one task."""

    n_estimators: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0, le=1)
    num_leaves: int = Field(..., ge=2)
    min_child_samples: int = Field(..., ge=1)
    subsample: float = Field(..., gt=0, le=1)
    colsample_bytree: float = Field(..., gt=0, le=1)


class ModelConfig(StrictModel):
    """Settings of model A or model B."""

    tree_checkpoint_step: int = Field(..., ge=1)
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig

    @model_validator(mode="after")
    def curve_has_points(self) -> ModelConfig:
        """The tree curve ends at n_estimators and has at least two points (rule 3 of run 5)."""
        n_estimators = self.xgboost.n_estimators
        step = self.tree_checkpoint_step
        if n_estimators % step != 0 or n_estimators // step < MIN_TREE_CHECKPOINTS:
            raise ValueError(
                f"tree_checkpoint_step ({step}) must divide n_estimators ({n_estimators}) "
                f"at least {MIN_TREE_CHECKPOINTS} times"
            )
        return self


class IntervalConfig(StrictModel):
    """Quantiles and daily conformal calibration of the prediction interval."""

    lower_quantile: float = Field(..., gt=0, lt=MEDIAN)
    upper_quantile: float = Field(..., gt=MEDIAN, lt=1)
    calibration_days: int = Field(..., ge=1)
    calibration_candidates_days: list[int] = Field(..., min_length=1)
    min_calibration_days: int = Field(..., ge=1)
    min_width_share: float = Field(..., gt=0, le=1)
    min_raw_width_eur_mwh: float = Field(..., gt=0)

    @model_validator(mode="after")
    def windows_agree(self) -> IntervalConfig:
        """The report covers the configured window; a calibration needs its minimum history."""
        candidates = self.calibration_candidates_days
        if self.calibration_days not in candidates or len(set(candidates)) < len(candidates):
            raise ValueError(
                f"calibration_candidates_days must list calibration_days "
                f"({self.calibration_days}) and each window once: {candidates}"
            )
        if self.min_calibration_days > min(candidates):
            raise ValueError(
                f"min_calibration_days ({self.min_calibration_days}) must not exceed the "
                f"shortest calibration window ({min(candidates)})"
            )
        return self


class AlertsConfig(StrictModel):
    """Alert thresholds, confidence classes and daily ranks."""

    high_quantile: float = Field(..., gt=MEDIAN, lt=1)
    critical_quantile: float = Field(..., gt=MEDIAN, lt=1)
    reference_window_months: int = Field(..., ge=1)
    min_reference_hours: int = Field(..., ge=1)
    confidence_probabilities: list[float] = Field(..., min_length=2, max_length=2)
    cheapest_hours: int = Field(..., ge=1)
    most_expensive_hours: int = Field(..., ge=1)

    @model_validator(mode="after")
    def levels_are_ordered(self) -> AlertsConfig:
        """high below critical, medium below high confidence, ranks inside the shortest day."""
        if self.high_quantile >= self.critical_quantile:
            raise ValueError("high_quantile must be below critical_quantile")
        medium, high = self.confidence_probabilities
        if not MEDIAN < medium < high <= 1:
            raise ValueError(
                f"confidence_probabilities must be increasing, above {MEDIAN}: "
                f"{self.confidence_probabilities}"
            )
        if self.cheapest_hours + self.most_expensive_hours > SHORTEST_DAY_HOURS:
            raise ValueError(
                f"cheapest_hours + most_expensive_hours must fit in a day of "
                f"{SHORTEST_DAY_HOURS} hours"
            )
        return self


class PredictionConfig(StrictModel):
    """Validates the `prediction` section of parameters.yml."""

    market_timezone: str
    holdout_start: date
    regime_names: list[str] = Field(..., min_length=MIN_REGIME_STATES)
    min_train_months: int = Field(..., ge=1)
    training_window_months: int | None = Field(..., ge=1)
    window_candidates_months: list[int | None] = Field(..., min_length=1)
    development_epochs: list[date]
    min_regime_training_hours: int = Field(..., ge=1)
    model_a: ModelConfig
    model_b: ModelConfig
    ridge_alpha: float = Field(..., gt=0)
    random_state: int = Field(..., ge=0)
    stability_seed: int = Field(..., ge=0)
    significance: float = Field(..., gt=0, lt=MEDIAN)
    min_dm_lag_days: int = Field(..., ge=1)
    interval: IntervalConfig
    alerts: AlertsConfig
    worst_days: int = Field(..., ge=1)

    @field_validator("market_timezone")
    @classmethod
    def timezone_exists(cls, value: str) -> str:
        """Reject a time zone name that the tz database does not know."""
        return _known_timezone(value)

    @field_validator("window_candidates_months")
    @classmethod
    def candidates_are_windows(cls, value: list[int | None]) -> list[int | None]:
        """Each candidate is null (every past row) or at least 1 month, and appears once."""
        invalid_windows = [window for window in value if window is not None and window < 1]
        if invalid_windows:
            raise ValueError(f"window_candidates_months must be null or >= 1: {invalid_windows}")
        if len(set(value)) < len(value):
            raise ValueError(f"window_candidates_months lists a window twice: {value}")
        return value

    @model_validator(mode="after")
    def settings_agree(self) -> PredictionConfig:
        """Reported window, distinct seeds, ordered epochs, reachable confidence."""
        if self.training_window_months not in self.window_candidates_months:
            raise ValueError(
                f"window_candidates_months must include training_window_months "
                f"({self.training_window_months})"
            )
        if self.stability_seed == self.random_state:
            raise ValueError("stability_seed must differ from random_state")
        epochs = self.development_epochs
        is_ordered = all(
            earlier < later for earlier, later in zip(epochs, epochs[1:], strict=False)
        )
        if not is_ordered or (epochs and epochs[-1] >= self.holdout_start):
            raise ValueError(
                f"development_epochs must be increasing and before holdout_start: {epochs}"
            )
        # Beyond its interval nothing is known: a probability never exceeds 1 - lower_quantile.
        highest_probability = 1 - self.interval.lower_quantile
        if self.alerts.confidence_probabilities[-1] > highest_probability:
            raise ValueError(
                f"the high confidence probability must not exceed {highest_probability:.2f} "
                f"(1 - lower_quantile), else no alert can reach it"
            )
        return self


class BacktestConfig(StrictModel):
    """Validates the `backtest` section of parameters.yml (client assumptions of D31)."""

    market_timezone: str
    holdout_start: date
    regime_names: list[str] = Field(..., min_length=MIN_REGIME_STATES)
    significance: float = Field(..., gt=0, lt=MEDIAN)
    min_dm_lag_days: int = Field(..., ge=1)
    cheapest_hours: int = Field(..., ge=1)
    most_expensive_hours: int = Field(..., ge=1)
    annual_consumption_mwh: float = Field(..., gt=0)
    flexible_share: float = Field(..., gt=0, le=1)
    flexibility_window_hours: int = Field(..., ge=1, le=MAX_FLEXIBILITY_WINDOW_HOURS)
    max_flexible_power_factor: float = Field(..., gt=MIN_FLEXIBLE_POWER_FACTOR)
    shift_cost_eur_mwh: float = Field(..., ge=0)
    shift_cost_levels_eur_mwh: list[float] = Field(..., min_length=1)
    project_cost_eur: float = Field(..., gt=0)
    worst_days: int = Field(..., ge=1)

    @field_validator("market_timezone")
    @classmethod
    def timezone_exists(cls, value: str) -> str:
        """Reject a time zone name that the tz database does not know."""
        return _known_timezone(value)

    @field_validator("shift_cost_levels_eur_mwh")
    @classmethod
    def levels_are_costs(cls, value: list[float]) -> list[float]:
        """Each level is a cost (0 or more) and appears once."""
        negative_levels = [level for level in value if level < 0]
        if negative_levels:
            raise ValueError(f"shift_cost_levels_eur_mwh must be 0 or more: {negative_levels}")
        if len(set(value)) < len(value):
            raise ValueError(f"shift_cost_levels_eur_mwh lists a level twice: {value}")
        return value

    @model_validator(mode="after")
    def settings_agree(self) -> BacktestConfig:
        """The daily table is computed at a level of the list; ranks fit in the shortest day."""
        if self.shift_cost_eur_mwh not in self.shift_cost_levels_eur_mwh:
            raise ValueError(
                f"shift_cost_levels_eur_mwh must include shift_cost_eur_mwh "
                f"({self.shift_cost_eur_mwh})"
            )
        if self.cheapest_hours + self.most_expensive_hours > SHORTEST_DAY_HOURS:
            raise ValueError(
                f"cheapest_hours + most_expensive_hours must fit in a day of "
                f"{SHORTEST_DAY_HOURS} hours"
            )
        return self


class DriversConfig(StrictModel):
    """Level of the drivers the morning report shows: features, or families of features."""

    day: Literal["feature", "family"]
    hour: Literal["feature", "family"]


class MorningReportConfig(StrictModel):
    """Decision day and content of the morning report."""

    decision_day: date | None
    track_record_days: int = Field(..., ge=1)
    anomaly_lookback_days: int = Field(..., ge=1)
    drivers: DriversConfig


class InterpretabilityConfig(StrictModel):
    """Validates the `interpretability` section of parameters.yml (decisions D33, D34)."""

    market_timezone: str
    holdout_start: date
    regime_names: list[str] = Field(..., min_length=MIN_REGIME_STATES)
    development_epochs: list[date]
    top_drivers: int = Field(..., ge=1)
    method_check_day_of_month: int = Field(..., ge=1, le=LAST_DAY_IN_EVERY_MONTH)
    method_check_background_rows: int = Field(..., ge=MIN_BACKGROUND_ROWS, le=MAX_BACKGROUND_ROWS)
    morning_report: MorningReportConfig

    @field_validator("market_timezone")
    @classmethod
    def timezone_exists(cls, value: str) -> str:
        """Reject a time zone name that the tz database does not know."""
        return _known_timezone(value)


class ProjectConfig(StrictModel):
    """All parameter sections. Each pipeline adds its own field in its commit."""

    ingestion: IngestionConfig
    features: FeaturesConfig
    regime: RegimeConfig
    anomaly: AnomalyConfig
    prediction: PredictionConfig
    backtest: BacktestConfig
    interpretability: InterpretabilityConfig

    @model_validator(mode="after")
    def sections_agree(self) -> ProjectConfig:
        """Later sections read what earlier ones produce, in the same time zone and period."""
        _features_agree(self.ingestion, self.features)
        _regime_agrees(self.ingestion, self.features, self.regime)
        _anomaly_agrees(self.ingestion, self.regime, self.anomaly)
        _prediction_agrees(self.ingestion, self.regime, self.prediction)
        _backtest_agrees(self.prediction, self.backtest)
        _interpretability_agrees(self.ingestion, self.prediction, self.interpretability)
        return self


def _features_agree(ingestion: IngestionConfig, features: FeaturesConfig) -> None:
    """Features read columns the ingestion produces, in the same time zone and period."""
    produced_columns = set(ingestion.generation_columns.values())
    unknown_columns = sorted(set(features.generation_columns()) - produced_columns)
    if unknown_columns:
        raise ValueError(
            f"features reads generation columns that ingestion does not produce: {unknown_columns}"
        )
    if features.market_timezone != ingestion.market_timezone:
        raise ValueError("features and ingestion must use the same market_timezone")
    holdout_start = features.holdout_start
    is_inside_period = ingestion.start_date < holdout_start <= ingestion.end_date
    if not is_inside_period:
        raise ValueError(
            f"holdout_start ({holdout_start}) must fall after start_date and not after end_date"
        )


def _regime_agrees(
    ingestion: IngestionConfig, features: FeaturesConfig, regime: RegimeConfig
) -> None:
    """Same time zone and reserved period as the features; a first refit before the holdout."""
    if regime.market_timezone != ingestion.market_timezone:
        raise ValueError("regime and ingestion must use the same market_timezone")
    if regime.holdout_start != features.holdout_start:
        raise ValueError(
            f"regime and features must use the same holdout_start "
            f"({regime.holdout_start} vs {features.holdout_start})"
        )
    months_before_holdout = _months_between(ingestion.start_date, regime.holdout_start)
    if regime.min_train_months >= months_before_holdout:
        raise ValueError(
            f"min_train_months ({regime.min_train_months}) leaves no filtered month before "
            f"holdout_start ({months_before_holdout} months of data before it)"
        )


def _anomaly_agrees(
    ingestion: IngestionConfig, regime: RegimeConfig, anomaly: AnomalyConfig
) -> None:
    """Same time zone, holdout and regimes as the regime section; a scored month before holdout."""
    if anomaly.market_timezone != ingestion.market_timezone:
        raise ValueError("anomaly and ingestion must use the same market_timezone")
    if anomaly.holdout_start != regime.holdout_start:
        raise ValueError(
            f"anomaly and regime must use the same holdout_start "
            f"({anomaly.holdout_start} vs {regime.holdout_start})"
        )
    if anomaly.regime_names != regime.regime_names:
        raise ValueError(
            f"anomaly and regime must use the same regime_names "
            f"({anomaly.regime_names} vs {regime.regime_names})"
        )
    months_before_holdout = _months_between(ingestion.start_date, anomaly.holdout_start)
    months_before_first_score = regime.min_train_months + anomaly.min_train_months
    if months_before_first_score >= months_before_holdout:
        raise ValueError(
            f"regime and anomaly min_train_months ({months_before_first_score} months in all) "
            f"leave no scored month before holdout_start"
        )


def _prediction_agrees(
    ingestion: IngestionConfig, regime: RegimeConfig, prediction: PredictionConfig
) -> None:
    """Same time zone, holdout and regimes as the regime section; tested months before the
    holdout, and every development epoch after the first tested month."""
    if prediction.market_timezone != ingestion.market_timezone:
        raise ValueError("prediction and ingestion must use the same market_timezone")
    if prediction.holdout_start != regime.holdout_start:
        raise ValueError(
            f"prediction and regime must use the same holdout_start "
            f"({prediction.holdout_start} vs {regime.holdout_start})"
        )
    if prediction.regime_names != regime.regime_names:
        raise ValueError(
            f"prediction and regime must use the same regime_names "
            f"({prediction.regime_names} vs {regime.regime_names})"
        )
    months_before_holdout = _months_between(ingestion.start_date, prediction.holdout_start)
    if prediction.min_train_months >= months_before_holdout:
        raise ValueError(
            f"prediction min_train_months ({prediction.min_train_months}) leaves no tested "
            f"month before holdout_start"
        )
    epochs = prediction.development_epochs
    if epochs:
        months_before_epoch = _months_between(ingestion.start_date, epochs[0])
        if months_before_epoch <= prediction.min_train_months:
            raise ValueError(
                f"development_epochs must start after the first tested month: {epochs[0]}"
            )


def _backtest_agrees(prediction: PredictionConfig, backtest: BacktestConfig) -> None:
    """The backtest reads the prediction tables: same time zone, reserved period, regimes,
    test and ranks of the hours (it checks the published alerts against the predictions)."""
    shared = {
        "market_timezone": (backtest.market_timezone, prediction.market_timezone),
        "holdout_start": (backtest.holdout_start, prediction.holdout_start),
        "regime_names": (backtest.regime_names, prediction.regime_names),
        "significance": (backtest.significance, prediction.significance),
        "min_dm_lag_days": (backtest.min_dm_lag_days, prediction.min_dm_lag_days),
        "cheapest_hours": (backtest.cheapest_hours, prediction.alerts.cheapest_hours),
        "most_expensive_hours": (
            backtest.most_expensive_hours,
            prediction.alerts.most_expensive_hours,
        ),
    }
    different = [name for name, (ours, theirs) in shared.items() if ours != theirs]
    if different:
        raise ValueError(f"backtest and prediction must use the same {different}")


def _interpretability_agrees(
    ingestion: IngestionConfig,
    prediction: PredictionConfig,
    interpretability: InterpretabilityConfig,
) -> None:
    """The explanations refit the prediction models: same time zone, reserved period, regimes
    and epochs; the morning report's decision day lies inside the delivery period."""
    shared = {
        "market_timezone": (interpretability.market_timezone, prediction.market_timezone),
        "holdout_start": (interpretability.holdout_start, prediction.holdout_start),
        "regime_names": (interpretability.regime_names, prediction.regime_names),
        "development_epochs": (interpretability.development_epochs, prediction.development_epochs),
    }
    different = [name for name, (ours, theirs) in shared.items() if ours != theirs]
    if different:
        raise ValueError(f"interpretability and prediction must use the same {different}")
    decision_day = interpretability.morning_report.decision_day
    is_inside = decision_day is None or ingestion.start_date <= decision_day < ingestion.end_date
    if not is_inside:
        raise ValueError(
            f"morning_report.decision_day ({decision_day}) must fall from start_date to the day "
            f"before end_date ({ingestion.end_date})"
        )


def _months_between(start: date, end: date) -> int:
    """Whole calendar months from the month of `start` to the month of `end`."""
    return (end.year - start.year) * MONTHS_PER_YEAR + end.month - start.month


def _known_timezone(value: str) -> str:
    """Return `value` if the tz database knows it, else raise ValueError."""
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown time zone '{value}'") from error
    return value


def _is_entsoe_zone(code: str) -> bool:
    """True if `code` is written like an ENTSO-E zone code."""
    return re.fullmatch(ENTSOE_ZONE_PATTERN, code) is not None


def _is_generation_column(name: str) -> bool:
    """True if `name` follows the gen_<name>_mw convention."""
    return re.fullmatch(GENERATION_COLUMN_PATTERN, name) is not None


# ==================================================
# Hooks
# ==================================================


class ProjectHooks:
    """Validates the parameters and provides the ENTSO-E key to the pipelines."""

    @hook_impl
    def after_context_created(self, context: Any) -> None:
        """Validate parameters.yml (with any --params overrides) before anything runs."""
        try:
            config = ProjectConfig(**context.params)
        except ValidationError as error:
            raise ValueError(f"Invalid parameters.yml, fix these errors:\n{error}") from None
        logger.info(
            "Parameters validated: zone %s, delivery days %s to %s, reserved period from %s",
            config.ingestion.country,
            config.ingestion.start_date,
            config.ingestion.end_date,
            config.features.holdout_start,
        )

    @hook_impl
    def after_catalog_created(self, catalog: Any, conf_creds: dict[str, Any]) -> None:
        """Add the ENTSO-E key to the catalog, masked, when credentials.yml provides it.

        `SecretStr` prints as '**********': if the key ever reaches a log or an
        error message, it stays hidden. Nodes unmask it only to create the client.
        """
        has_key = bool(conf_creds) and bool(conf_creds.get(ENTSOE_API_KEY))
        if not has_key:
            return
        masked_key = SecretStr(str(conf_creds[ENTSOE_API_KEY]))
        catalog.add_feed_dict({ENTSOE_API_KEY: masked_key}, replace=True)

    @hook_impl
    def before_pipeline_run(self, pipeline: Any, catalog: Any) -> None:
        """Stop with a clear message if the pipeline needs the key and none was found."""
        needs_key = ENTSOE_API_KEY in pipeline.inputs()
        has_key = ENTSOE_API_KEY in catalog
        if needs_key and not has_key:
            raise ValueError(MISSING_KEY_MESSAGE)


# ==================================================
# Kedro settings
# ==================================================

HOOKS = (ProjectHooks(),)

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}
