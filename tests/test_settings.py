"""Tests for pydantic config validation.

These tests verify that:
1. Valid parameters pass validation
2. Invalid parameters fail with clear messages
3. Edge cases (composite ENTSO-E codes) are handled
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from energy_market_analyzer.settings import (
    AlertsConfig,
    AnomalyConfig,
    BacktestConfig,
    FeaturesConfig,
    HMMConfig,
    IngestionConfig,
    LightGBMQuantileConfig,
    ProjectConfig,
    XGBoostConfig,
)
from pydantic import ValidationError


class TestIngestionConfig:
    """Tests for ingestion parameter validation."""

    def test_valid_config(self) -> None:
        """A well-formed ingestion config passes validation."""
        config = IngestionConfig(
            country="BE",
            start_date="2015-01-01",
            end_date="2026-03-31",
            neighbors=["FR", "NL", "DE_LU"],
            forward_fill_max_days=3,
            retry_max_attempts=3,
            retry_backoff_factor=2,
        )
        assert config.country == "BE"

    def test_invalid_country_code_rejected(self) -> None:
        """A lowercase country code is rejected."""
        with pytest.raises(ValidationError):
            IngestionConfig(
                country="be",
                start_date="2015-01-01",
                end_date="2026-03-31",
                neighbors=["FR"],
                forward_fill_max_days=3,
                retry_max_attempts=3,
                retry_backoff_factor=2,
            )

    def test_end_before_start_rejected(self) -> None:
        """end_date before start_date is rejected."""
        with pytest.raises(ValidationError):
            IngestionConfig(
                country="BE",
                start_date="2026-01-01",
                end_date="2015-01-01",
                neighbors=["FR"],
                forward_fill_max_days=3,
                retry_max_attempts=3,
                retry_backoff_factor=2,
            )

    def test_empty_neighbors_rejected(self) -> None:
        """Empty neighbors list is rejected."""
        with pytest.raises(ValidationError):
            IngestionConfig(
                country="BE",
                start_date="2015-01-01",
                end_date="2026-03-31",
                neighbors=[],
                forward_fill_max_days=3,
                retry_max_attempts=3,
                retry_backoff_factor=2,
            )

    def test_composite_entsoe_code_accepted(self) -> None:
        """DE_LU (Germany-Luxembourg) is a valid ENTSO-E code."""
        config = IngestionConfig(
            country="BE",
            start_date="2015-01-01",
            end_date="2026-03-31",
            neighbors=["FR", "NL", "DE_LU"],
            forward_fill_max_days=3,
            retry_max_attempts=3,
            retry_backoff_factor=2,
        )
        assert "DE_LU" in config.neighbors

    def test_invalid_neighbor_code_rejected(self) -> None:
        """Lowercase or malformed neighbor codes are rejected."""
        with pytest.raises(ValidationError):
            IngestionConfig(
                country="BE",
                start_date="2015-01-01",
                end_date="2026-03-31",
                neighbors=["fr"],
                forward_fill_max_days=3,
                retry_max_attempts=3,
                retry_backoff_factor=2,
            )


class TestHMMConfig:
    """Tests for HMM parameter validation."""

    def test_valid_hmm_config(self) -> None:
        """A well-formed HMM config passes validation."""
        config = HMMConfig(
            n_states=3,
            n_init=50,
            covariance_type="full",
            refit_frequency="monthly",
            random_state=42,
            features=["price_return", "price_volatility_24h", "gas_share", "spark_spread"],
        )
        assert config.n_states == 3

    def test_n_states_too_high_rejected(self) -> None:
        """n_states > 10 is rejected."""
        with pytest.raises(ValidationError):
            HMMConfig(
                n_states=15,
                n_init=50,
                covariance_type="full",
                refit_frequency="monthly",
                random_state=42,
                features=["a", "b"],
            )

    def test_single_feature_rejected(self) -> None:
        """HMM with only 1 feature is rejected."""
        with pytest.raises(ValidationError):
            HMMConfig(
                n_states=3,
                n_init=50,
                covariance_type="full",
                refit_frequency="monthly",
                random_state=42,
                features=["price_return"],
            )

    def test_invalid_covariance_type_rejected(self) -> None:
        """Typo in covariance_type is rejected (Literal constraint)."""
        with pytest.raises(ValidationError):
            HMMConfig(
                n_states=3,
                n_init=50,
                covariance_type="ful",
                refit_frequency="monthly",
                random_state=42,
                features=["a", "b"],
            )


class TestAnomalyConfig:
    """Tests for Isolation Forest parameter validation."""

    def test_contamination_too_high_rejected(self) -> None:
        """contamination=1.5 is rejected (must be < 0.5)."""
        with pytest.raises(ValidationError):
            AnomalyConfig(
                contamination=1.5,
                n_estimators=200,
                min_samples_per_regime=500,
                event_grouping_hours=5,
                refit_frequency="monthly",
                random_state=42,
                features=["a", "b"],
            )

    def test_contamination_zero_rejected(self) -> None:
        """contamination=0 is rejected (must be > 0)."""
        with pytest.raises(ValidationError):
            AnomalyConfig(
                contamination=0,
                n_estimators=200,
                min_samples_per_regime=500,
                event_grouping_hours=5,
                refit_frequency="monthly",
                random_state=42,
                features=["a", "b"],
            )

    def test_empty_features_rejected(self) -> None:
        """Empty features list is rejected."""
        with pytest.raises(ValidationError):
            AnomalyConfig(
                contamination=0.05,
                n_estimators=200,
                min_samples_per_regime=500,
                event_grouping_hours=5,
                refit_frequency="monthly",
                random_state=42,
                features=[],
            )


class TestXGBoostConfig:
    """Tests for XGBoost parameter validation."""

    def test_learning_rate_too_high_rejected(self) -> None:
        """learning_rate >= 1 is rejected."""
        with pytest.raises(ValidationError):
            XGBoostConfig(
                n_estimators=1000,
                learning_rate=1.5,
                max_depth=6,
                early_stopping_rounds=50,
                eval_metric="mae",
                random_state=42,
                validation_split=0.8,
            )


class TestLightGBMConfig:
    """Tests for LightGBM quantile validation."""

    def test_invalid_quantile_rejected(self) -> None:
        """Quantile > 1 is rejected."""
        with pytest.raises(ValidationError):
            LightGBMQuantileConfig(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                quantiles=[0.1, 0.5, 1.5],
                random_state=42,
            )

    def test_empty_quantiles_rejected(self) -> None:
        """Empty quantiles list is rejected."""
        with pytest.raises(ValidationError):
            LightGBMQuantileConfig(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                quantiles=[],
                random_state=42,
            )


class TestRollingWindowsConfig:
    """Tests for rolling window order validation."""

    def test_wrong_order_rejected(self) -> None:
        """short >= medium is rejected."""
        with pytest.raises(ValidationError):
            FeaturesConfig(
                rolling_windows={"short": 720, "medium": 24, "long": 1},
                spark_spread={"heat_rate": 2.0},
            )


class TestAlertsConfig:
    """Tests for alert threshold validation."""

    def test_p95_less_than_p80_rejected(self) -> None:
        """p95_threshold <= p80_threshold is rejected."""
        with pytest.raises(ValidationError):
            AlertsConfig(
                p80_threshold=0.99,
                p95_threshold=0.50,
                top_cheap_hours=8,
                top_expensive_hours=4,
                shap_top_drivers=3,
            )


class TestBacktestConfig:
    """Tests for backtest parameter validation."""

    def test_end_before_start_rejected(self) -> None:
        """period_end before period_start is rejected."""
        with pytest.raises(ValidationError):
            BacktestConfig(
                client_consumption_gwh_year=100,
                flexible_share=0.10,
                flexibility_window_hours=4,
                assumed_project_cost_eur=30000,
                period_start="2026-01-01",
                period_end="2024-01-01",
            )


class TestFullProjectConfig:
    """Tests for the complete ProjectConfig validation."""

    def test_valid_full_config_from_yaml(self) -> None:
        """The actual parameters.yml passes validation."""
        config_path = Path(__file__).resolve().parent.parent / "conf" / "base" / "parameters.yml"
        with open(config_path) as f:
            raw_params = yaml.safe_load(f)

        config = ProjectConfig(**raw_params)
        assert config.ingestion.country == "BE"
        assert config.hmm.n_states == 3
        assert config.anomaly_detection.contamination == 0.05
        assert config.prediction.xgboost.learning_rate == 0.05
        assert config.lightgbm_quantile.quantiles == [0.1, 0.5, 0.9]
