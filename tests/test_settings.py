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


def test_unknown_parameter_is_rejected() -> None:
    """A typo in a key name stops the run instead of being silently ignored."""
    ingestion = _project_parameters()["ingestion"]
    ingestion_with_typo = {**ingestion, "start_dat": "2018-01-01"}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProjectConfig(ingestion=ingestion_with_typo)


def test_end_date_before_start_date_is_rejected() -> None:
    """A reversed period is rejected with a message naming both dates."""
    ingestion = _project_parameters()["ingestion"]
    reversed_period = {**ingestion, "start_date": "2026-09-30", "end_date": "2018-01-01"}

    with pytest.raises(ValidationError, match="end_date .* must not be before start_date"):
        ProjectConfig(ingestion=reversed_period)
