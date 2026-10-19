"""Tests of the parameter validation in settings.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from kedro.config import OmegaConfigLoader
from pydantic import ValidationError

from energy_market_analyzer.settings import ProjectConfig

CONF_SOURCE = Path(__file__).resolve().parents[1] / "conf"


def _project_parameters() -> dict[str, Any]:
    """parameters.yml as Kedro loads it (globals resolved, local overrides applied)."""
    loader = OmegaConfigLoader(
        conf_source=str(CONF_SOURCE), base_env="base", default_run_env="local"
    )
    return loader["parameters"]


def _with_section(section: str, change: dict[str, Any]) -> dict[str, Any]:
    """The real parameters with `change` applied to one section."""
    parameters = _project_parameters()
    changed_section = {**parameters[section], **change}
    return {**parameters, section: changed_section}


def _assert_only_error(parameters: dict[str, Any], message: str) -> None:
    """Validation fails with exactly one error, matching `message`.

    One error only: the test cannot pass because of another problem.
    """
    with pytest.raises(ValidationError, match=message) as error:
        ProjectConfig(**parameters)
    assert error.value.error_count() == 1


def test_project_parameters_are_valid() -> None:
    """The real parameters.yml passes the validation run at every Kedro start."""
    config = ProjectConfig(**_project_parameters())

    assert config.ingestion.country == "BE"
    assert config.ingestion.start_date < config.ingestion.end_date
    assert config.ingestion.start_date < config.features.holdout_start
    assert config.regime.holdout_start == config.features.holdout_start
    assert len(config.regime.regime_names) in config.regime.bic_candidate_states
    assert config.anomaly.regime_names == config.regime.regime_names
    assert config.anomaly.contamination in config.anomaly.contamination_levels
    assert config.prediction.regime_names == config.regime.regime_names
    assert config.prediction.holdout_start == config.features.holdout_start
    assert config.prediction.training_window_months in config.prediction.window_candidates_months
    assert config.backtest.holdout_start == config.features.holdout_start
    assert config.backtest.cheapest_hours == config.prediction.alerts.cheapest_hours
    assert config.backtest.shift_cost_eur_mwh in config.backtest.shift_cost_levels_eur_mwh
    assert config.interpretability.regime_names == config.prediction.regime_names
    assert config.interpretability.development_epochs == config.prediction.development_epochs
    assert config.interpretability.morning_report.decision_day is None


def test_unknown_parameter_is_rejected() -> None:
    """A typo in a key name stops the run instead of being silently ignored."""
    parameters = _with_section("ingestion", {"start_dat": "2018-01-01"})

    _assert_only_error(parameters, "Extra inputs are not permitted")


def test_end_date_before_start_date_is_rejected() -> None:
    """A reversed period is rejected with a message naming both dates."""
    parameters = _with_section("ingestion", {"start_date": "2026-09-30", "end_date": "2018-01-01"})

    _assert_only_error(parameters, "end_date .* must not be before start_date")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"gas_column": "gen_fossil_gas_mw"}, "does not produce: \\['gen_fossil_gas_mw'\\]"),
        ({"multi_day_horizons": [1, 2]}, "multi_day_horizons must lie between 2"),
        ({"heat_rate": 20}, "less than or equal to 3.5"),
    ],
)
def test_inconsistent_features_section_is_rejected(change: dict[str, Any], message: str) -> None:
    """A column the ingestion does not produce, J+1 in model B or a typo in the heat rate."""
    parameters = _with_section("features", change)

    _assert_only_error(parameters, message)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"regime_names": ["calm", "volatile", "crisis", "panic", "extreme", "chaos"]},
            "bic_candidate_states must include the number of regime_names \\(6\\)",
        ),
        ({"holdout_start": "2025-07-01"}, "same holdout_start"),
        ({"stability_seeds": [45]}, "at least n_init \\(10\\) apart"),
    ],
)
def test_inconsistent_regime_section_is_rejected(change: dict[str, Any], message: str) -> None:
    """A number of states without a BIC, a different reserved period, overlapping seeds."""
    parameters = _with_section("regime", change)

    _assert_only_error(parameters, message)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"regime_names": ["low", "medium", "high"]}, "same regime_names"),
        ({"contamination": 0.03}, "contamination_levels must include contamination"),
        ({"max_samples": 1000}, "must not exceed min_training_hours"),
        ({"training_window_months": 36}, "must include training_window_months"),
    ],
)
def test_inconsistent_anomaly_section_is_rejected(change: dict[str, Any], message: str) -> None:
    """Other regimes than the regime section, an unreported contamination or window, big trees."""
    parameters = _with_section("anomaly", change)

    _assert_only_error(parameters, message)


def _with_prediction_setting(group: str | None, change: dict[str, Any]) -> dict[str, Any]:
    """The real parameters with `change` applied to the prediction section or one of its groups."""
    parameters = _project_parameters()
    prediction = parameters["prediction"]
    if group is None:
        return _with_section("prediction", change)
    changed_group = {**prediction[group], **change}
    return _with_section("prediction", {group: changed_group})


@pytest.mark.parametrize(
    ("group", "change", "message"),
    [
        (None, {"regime_names": ["low", "medium", "high"]}, "same regime_names"),
        (None, {"development_epochs": ["2023-01-01", "2021-07-01"]}, "must be increasing"),
        ("interval", {"calibration_days": 60}, "must list calibration_days \\(60\\)"),
        ("alerts", {"confidence_probabilities": [0.7, 0.95]}, "must not exceed 0.90"),
        ("model_a", {"tree_checkpoint_step": 300}, "must divide n_estimators \\(500\\)"),
    ],
)
def test_inconsistent_prediction_section_is_rejected(
    group: str | None, change: dict[str, Any], message: str
) -> None:
    """Other regimes, unordered epochs, an unreported calibration window, a confidence no
    alert can reach, a tree curve without two points."""
    parameters = _with_prediction_setting(group, change)

    _assert_only_error(parameters, message)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"cheapest_hours": 6}, "same \\['cheapest_hours'\\]"),
        ({"shift_cost_eur_mwh": 7.0}, "must include shift_cost_eur_mwh \\(7.0\\)"),
        ({"max_flexible_power_factor": 1.0}, "greater than 1"),
        ({"flexibility_window_hours": 24}, "less than or equal to 22"),
    ],
)
def test_inconsistent_backtest_section_is_rejected(change: dict[str, Any], message: str) -> None:
    """Other hour counts than the published alerts, a shift cost without its sensitivity, a
    maximum power at which nothing can move, a window longer than the shortest day."""
    parameters = _with_section("backtest", change)

    _assert_only_error(parameters, message)


def _with_morning_report(change: dict[str, Any]) -> dict[str, Any]:
    """The real parameters with `change` applied to interpretability.morning_report."""
    morning_report = _project_parameters()["interpretability"]["morning_report"]
    return _with_section("interpretability", {"morning_report": {**morning_report, **change}})


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (
            _with_section("interpretability", {"regime_names": ["low", "medium", "high"]}),
            "same \\['regime_names'\\]",
        ),
        (
            _with_section("interpretability", {"method_check_background_rows": 200}),
            "less than or equal to 100",
        ),
        (_with_morning_report({"decision_day": "2030-01-01"}), "must fall from start_date"),
        (
            _with_morning_report({"drivers": {"day": "features", "hour": "feature"}}),
            "Input should be 'feature' or 'family'",
        ),
    ],
)
def test_inconsistent_interpretability_section_is_rejected(
    parameters: dict[str, Any], message: str
) -> None:
    """Other regimes than the models explained, more background rows than shap uses, a
    decision day outside the data, a driver level that does not exist."""
    _assert_only_error(parameters, message)
