"""Regime pipeline: market regime of each day with a Gaussian HMM, filtered and smoothed.

The filtered regime only depends on the days up to the labelled day: models A
and B, the alerts and the Isolation Forest use it. The smoothed regime looks at
the whole period: retrospective analysis only. Choices (number of states,
settings) are informed by the development period only (decision D9).
"""

from __future__ import annotations

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from energy_market_analyzer.pipelines.regime.nodes import (
    analyze_development_period,
    build_regime_observations,
    build_regime_report,
    build_regimes,
    fit_filtered_regimes,
    fit_smoothed_regimes,
)

PARAMS = "params:regime"
# Not in the catalog: kept in memory between nodes.
OBSERVATIONS = "regime_observations"
DEVELOPMENT = "regime_development"
FILTERED = "filtered_regimes"
FILTERED_FITS = "filtered_fits"
SMOOTHED = "smoothed_regimes"
SMOOTHED_MODEL = "smoothed_model"


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Create the regime pipeline (6 nodes).

    Returns:
        Daily observations (in memory), the development-period analysis, the
        filtered and smoothed regimes, the `regimes` table (data/04_feature/)
        and `regime_report` (data/08_reporting/regime_report.json).
    """
    return pipeline(
        [
            node(
                build_regime_observations,
                inputs=["features_model_c", PARAMS],
                outputs=OBSERVATIONS,
                name="regime_observations_node",
            ),
            node(
                analyze_development_period,
                inputs=[OBSERVATIONS, FILTERED, PARAMS],
                outputs=DEVELOPMENT,
                name="regime_development_node",
            ),
            node(
                fit_filtered_regimes,
                inputs=[OBSERVATIONS, PARAMS],
                outputs=[FILTERED, FILTERED_FITS],
                name="filtered_regimes_node",
            ),
            node(
                fit_smoothed_regimes,
                inputs=[OBSERVATIONS, PARAMS],
                outputs=[SMOOTHED, SMOOTHED_MODEL],
                name="smoothed_regimes_node",
            ),
            node(
                build_regimes,
                inputs=[OBSERVATIONS, FILTERED, SMOOTHED],
                outputs="regimes",
                name="regimes_node",
            ),
            node(
                build_regime_report,
                inputs={
                    "regimes": "regimes",
                    "regime_development": DEVELOPMENT,
                    "filtered_fits": FILTERED_FITS,
                    "smoothed_model": SMOOTHED_MODEL,
                    "params": PARAMS,
                },
                outputs="regime_report",
                name="regime_report_node",
            ),
        ]
    )
