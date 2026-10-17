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
