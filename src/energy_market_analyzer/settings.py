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
from typing import Any
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


class ProjectConfig(StrictModel):
    """All parameter sections. Each pipeline adds its own field in its commit."""

    ingestion: IngestionConfig
    features: FeaturesConfig

    @model_validator(mode="after")
    def sections_agree(self) -> ProjectConfig:
        """Features read columns the ingestion produces, in the same time zone and period."""
        produced_columns = set(self.ingestion.generation_columns.values())
        unknown_columns = sorted(set(self.features.generation_columns()) - produced_columns)
        if unknown_columns:
            raise ValueError(
                f"features reads generation columns that ingestion does not produce: "
                f"{unknown_columns}"
            )
        if self.features.market_timezone != self.ingestion.market_timezone:
            raise ValueError("features and ingestion must use the same market_timezone")
        holdout_start = self.features.holdout_start
        is_inside_period = self.ingestion.start_date < holdout_start <= self.ingestion.end_date
        if not is_inside_period:
            raise ValueError(
                f"holdout_start ({holdout_start}) must fall after start_date and not after end_date"
            )
        return self


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
