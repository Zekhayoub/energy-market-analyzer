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


def test_project_parameters_are_valid() -> None:
    """The real parameters.yml passes the validation run at every Kedro start."""
    config = ProjectConfig(**_project_parameters())

    assert config.ingestion.country == "BE"
    assert config.ingestion.start_date < config.ingestion.end_date
    assert config.ingestion.start_date < config.features.holdout_start


def test_unknown_parameter_is_rejected() -> None:
    """A typo in a key name stops the run instead of being silently ignored."""
    parameters = _project_parameters()
    ingestion_with_typo = {**parameters["ingestion"], "start_dat": "2018-01-01"}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProjectConfig(ingestion=ingestion_with_typo, features=parameters["features"])


def test_end_date_before_start_date_is_rejected() -> None:
    """A reversed period is rejected with a message naming both dates."""
    parameters = _project_parameters()
    reversed_period = {
        **parameters["ingestion"],
        "start_date": "2026-09-30",
        "end_date": "2018-01-01",
    }

    with pytest.raises(ValidationError, match="end_date .* must not be before start_date"):
        ProjectConfig(ingestion=reversed_period, features=parameters["features"])


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
    parameters = _project_parameters()
    features = {**parameters["features"], **change}

    with pytest.raises(ValidationError, match=message):
        ProjectConfig(ingestion=parameters["ingestion"], features=features)
