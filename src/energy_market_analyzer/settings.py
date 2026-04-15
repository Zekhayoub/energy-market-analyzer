"""Kedro project settings with pydantic validation hooks.

This module defines:
1. Pydantic models that validate parameters.yml at startup.
2. Kedro hooks that run validation before any pipeline node.

If a parameter is invalid (e.g., contamination=1.5), the
pipeline fails immediately with a clear error message instead
of crashing 3 hours later in scikit-learn.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Literal

from kedro.framework.hooks import hook_impl
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ==================================================
# Pydantic models for parameters.yml validation
# ==================================================


class IngestionConfig(BaseModel):
    """Validates the ingestion section of parameters.yml."""

    country: str = Field(..., pattern=r"^[A-Z]{2}$")
    start_date: date
    end_date: date
    # timezone removed
    neighbors: list[str]
    forward_fill_max_days: int = Field(..., ge=1, le=7)
    retry_max_attempts: int = Field(..., ge=1)
    retry_backoff_factor: int = Field(..., ge=1)

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info: Any) -> date:
        """Ensure end_date is after start_date."""
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError(f"end_date ({v}) must be after start_date ({info.data['start_date']})")
        return v

    @field_validator("neighbors")
    @classmethod
    def neighbors_valid(cls, v: list[str]) -> list[str]:
        """Ensure neighbors are valid ENTSO-E country codes."""
        if not v:
            raise ValueError("neighbors must contain at least one country code")
        pattern = re.compile(r"^[A-Z]{2}(_[A-Z]{2,3})*$")
        for code in v:
            if not pattern.match(code):
                raise ValueError(
                    f"Invalid ENTSO-E country code: '{code}'. "
                    f"Expected format: 'XX' or 'XX_YY' (e.g., 'FR', 'DE_LU')"
                )
        return v


class RollingWindowsConfig(BaseModel):
    """Validates rolling window sizes."""

    short: int = Field(..., ge=1)
    medium: int = Field(..., ge=1)
    long: int = Field(..., ge=1)

    @model_validator(mode="after")
    def short_lt_medium_lt_long(self) -> RollingWindowsConfig:
        """Ensure short < medium < long."""
        if not (self.short < self.medium < self.long):
            raise ValueError(
                f"Rolling windows must satisfy short < medium < long, "
                f"got short={self.short}, medium={self.medium}, long={self.long}"
            )
        return self


class SparkSpreadConfig(BaseModel):
    """Validates spark spread parameters."""

    heat_rate: float = Field(..., gt=0, lt=10)


class FeaturesConfig(BaseModel):
    """Validates the features section."""

    rolling_windows: RollingWindowsConfig
    spark_spread: SparkSpreadConfig


class HMMConfig(BaseModel):
    """Validates the HMM section.

    n_states must be between 2 and 10. BIC will select the
    optimal number, but we validate the search range here.
    """

    n_states: int = Field(..., ge=2, le=10)
    n_init: int = Field(..., ge=1)
    covariance_type: Literal["full", "diag", "tied", "spherical"]
    refit_frequency: str = Field(..., pattern=r"^(daily|weekly|monthly)$")
    random_state: int
    features: list[str]

    @field_validator("features")
    @classmethod
    def features_not_empty(cls, v: list[str]) -> list[str]:
        """HMM needs at least 2 features for multivariate analysis."""
        if len(v) < 2:
            raise ValueError("HMM requires at least 2 features")
        return v


class AnomalyConfig(BaseModel):
    """Validates the anomaly detection section.

    contamination=0.05 means 5% of hours are flagged as
    anomalies. This is a BUSINESS choice: the procurement
    team can handle ~5% alerts per day.
    """

    contamination: float = Field(..., gt=0, lt=0.5)
    n_estimators: int = Field(..., gt=0)
    min_samples_per_regime: int = Field(..., gt=0)
    event_grouping_hours: int = Field(..., gt=0)
    refit_frequency: str = Field(..., pattern=r"^(daily|weekly|monthly)$")
    random_state: int
    features: list[str]

    @field_validator("features")
    @classmethod
    def features_not_empty(cls, v: list[str]) -> list[str]:
        """Isolation Forest needs at least 1 feature."""
        if len(v) < 1:
            raise ValueError("anomaly_detection.features must not be empty")
        return v


class XGBoostConfig(BaseModel):
    """Validates XGBoost hyperparameters."""

    n_estimators: int = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0, lt=1)
    max_depth: int = Field(..., gt=0, le=20)
    early_stopping_rounds: int = Field(..., gt=0)
    eval_metric: Literal["mae", "rmse", "mape", "logloss"]
    random_state: int
    validation_split: float = Field(..., gt=0, lt=1)


class ModelBConfig(BaseModel):
    """Validates Model B horizons."""

    horizons: list[int]

    @field_validator("horizons")
    @classmethod
    def horizons_positive(cls, v: list[int]) -> list[int]:
        """All horizons must be positive and list must not be empty."""
        if not v:
            raise ValueError("horizons must not be empty")
        for h in v:
            if h < 1:
                raise ValueError(f"Horizon must be >= 1, got {h}")
        return v


class PredictionConfig(BaseModel):
    """Validates the prediction section."""

    xgboost: XGBoostConfig
    model_b: ModelBConfig


class LightGBMQuantileConfig(BaseModel):
    """Validates LightGBM quantile regression parameters."""

    n_estimators: int = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0, lt=1)
    num_leaves: int = Field(..., gt=1)
    quantiles: list[float]
    random_state: int

    @field_validator("quantiles")
    @classmethod
    def quantiles_valid(cls, v: list[float]) -> list[float]:
        """All quantiles must be between 0 and 1, list must not be empty."""
        if not v:
            raise ValueError("quantiles must not be empty")
        for q in v:
            if not 0 < q < 1:
                raise ValueError(f"Quantile must be between 0 and 1, got {q}")
        return v


class AlertsConfig(BaseModel):
    """Validates alert thresholds."""

    p80_threshold: float = Field(..., ge=0, le=1)
    p95_threshold: float = Field(..., ge=0, le=1)
    top_cheap_hours: int = Field(..., gt=0)
    top_expensive_hours: int = Field(..., gt=0)
    shap_top_drivers: int = Field(..., gt=0)

    @model_validator(mode="after")
    def p95_greater_than_p80(self) -> AlertsConfig:
        """p95 threshold must be strictly greater than p80."""
        if self.p95_threshold <= self.p80_threshold:
            raise ValueError(
                f"p95_threshold ({self.p95_threshold}) must be > "
                f"p80_threshold ({self.p80_threshold})"
            )
        return self


class BacktestConfig(BaseModel):
    """Validates backtest parameters."""

    client_consumption_gwh_year: float = Field(..., gt=0)
    flexible_share: float = Field(..., gt=0, le=1)
    flexibility_window_hours: int = Field(..., gt=0)
    assumed_project_cost_eur: float = Field(..., ge=0)
    period_start: date
    period_end: date

    @field_validator("period_end")
    @classmethod
    def end_after_start(cls, v: date, info: Any) -> date:
        """Ensure period_end is after period_start."""
        if "period_start" in info.data and v <= info.data["period_start"]:
            raise ValueError(
                f"period_end ({v}) must be after period_start ({info.data['period_start']})"
            )
        return v


class ProjectConfig(BaseModel):
    """Top-level config that assembles all sub-configs.

    This is the single entry point for validation. If ANY
    parameter is invalid, the entire pipeline fails at startup.
    """

    ingestion: IngestionConfig
    features: FeaturesConfig
    hmm: HMMConfig
    anomaly_detection: AnomalyConfig
    prediction: PredictionConfig
    lightgbm_quantile: LightGBMQuantileConfig
    alerts: AlertsConfig
    backtest: BacktestConfig


# ==================================================
# Kedro hooks
# ==================================================


class ValidateConfigHook:
    """Validates parameters and credentials at startup.

    Merged into a single hook to avoid creating a nested
    KedroSession. The config_loader captured in
    after_context_created is reused in before_pipeline_run.
    """

    def __init__(self) -> None:
        self._config_loader: Any = None

    @hook_impl
    def after_context_created(self, context: Any) -> None:
        """Validate parameters.yml against pydantic models."""
        self._config_loader = context.config_loader
        raw_params = self._config_loader["parameters"]

        try:
            validated = ProjectConfig(**raw_params)
            logger.info(
                "Parameters validated: country=%s, period=%s to %s, "
                "hmm_states=%d, contamination=%.2f",
                validated.ingestion.country,
                validated.ingestion.start_date,
                validated.ingestion.end_date,
                validated.hmm.n_states,
                validated.anomaly_detection.contamination,
            )
        except Exception as e:
            raise ValueError(
                f"Invalid parameters.yml — fix these errors before running:\n{e}"
            ) from e

        try:
            globals_conf = self._config_loader["globals"]
            globals_country = globals_conf.get("country")
            params_country = validated.ingestion.country
            if globals_country and globals_country != params_country:
                raise ValueError(
                    f"Dual source of truth: globals.yml has country='{globals_country}' "
                    f"but parameters.yml has ingestion.country='{params_country}'. "
                    f"These must match — globals.yml drives catalog paths, "
                    f"parameters.yml drives fetcher logic."
                )
        except KeyError:
            pass  # globals.yml not loaded yet — skip check

    @hook_impl
    def before_pipeline_run(self, run_params: dict[str, Any], pipeline: Any) -> None:
        """Check credentials only if ingestion nodes are in the pipeline."""
        node_names = [node.name for node in pipeline.nodes]

        ingestion_prefixes = ("fetch_", "ingest_")
        needs_credentials = any(
            node_name.startswith(ingestion_prefixes) for node_name in node_names
        )

        if not needs_credentials:
            logger.debug("No ingestion nodes — skipping credentials check")
            return

        try:
            credentials = self._config_loader["credentials"]
            api_key = credentials.get("entsoe_api_key", "")
            if not api_key or api_key == "YOUR_ENTSOE_API_KEY_HERE":
                raise ValueError(
                    "ENTSOE_API_KEY is missing or still set to placeholder. "
                    "Copy conf/local/credentials.yml.example to "
                    "conf/local/credentials.yml and add your key. "
                    "See: https://transparency.entsoe.eu/"
                )
            logger.info("ENTSO-E API key validated (length=%d)", len(api_key))
        except KeyError as e:
            raise ValueError(
                "conf/local/credentials.yml not found. "
                "Copy credentials.yml.example and add your ENTSO-E API key."
            ) from e


# ==================================================
# Register hooks with Kedro
# ==================================================

HOOKS = (ValidateConfigHook(),)

# CONFIG_LOADER_CLASS = OmegaConfigLoader

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # "config_patterns": {
    #     "spark" : ["spark*/"],
    #     "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
    # }
}
