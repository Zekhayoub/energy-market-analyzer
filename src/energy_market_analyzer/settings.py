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
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown time zone '{value}'") from error
        return value

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


class ProjectConfig(StrictModel):
    """All parameter sections. Each pipeline adds its own field in its commit."""

    ingestion: IngestionConfig


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
            "Parameters validated: zone %s, delivery days %s to %s",
            config.ingestion.country,
            config.ingestion.start_date,
            config.ingestion.end_date,
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
