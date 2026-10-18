"""Anomaly pipeline: hours unusual for their filtered regime, with per-regime Isolation Forests.

Each month the forests are refitted on the previous hours only; the hours of
the month are scored by the forest of their regime. Anomalies are grouped into
events. Retrospective analysis (model C): never an input of models A and B.
Choices are informed by the development period only (decision D9).
"""

from __future__ import annotations

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from energy_market_analyzer.pipelines.anomaly.nodes import (
    analyze_development_period,
    build_anomaly_inputs,
    build_anomaly_report,
    detect_anomalies,
    group_anomaly_events,
)

PARAMS = "params:anomaly"
# Not in the catalog: kept in memory between nodes.
INPUTS = "anomaly_inputs"
SCORES = "anomaly_scores"
REFITS = "anomaly_refits"
DEVELOPMENT = "anomaly_development"


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Create the anomaly pipeline (5 nodes).

    Returns:
        The hourly `anomalies` table and the `anomaly_events` table
        (data/07_model_output/), and `anomaly_report` (data/08_reporting/).
    """
    return pipeline(
        [
            node(
                build_anomaly_inputs,
                inputs=["features_model_c", "regimes", PARAMS],
                outputs=INPUTS,
                name="anomaly_inputs_node",
            ),
            node(
                detect_anomalies,
                inputs=[INPUTS, PARAMS],
                outputs=[SCORES, REFITS],
                name="detect_anomalies_node",
            ),
            node(
                analyze_development_period,
                inputs=[INPUTS, SCORES, PARAMS],
                outputs=DEVELOPMENT,
                name="anomaly_development_node",
            ),
            node(
                group_anomaly_events,
                inputs=[SCORES, INPUTS, PARAMS],
                outputs=["anomalies", "anomaly_events"],
                name="anomaly_events_node",
            ),
            node(
                build_anomaly_report,
                inputs={
                    "anomalies": "anomalies",
                    "anomaly_events": "anomaly_events",
                    "anomaly_inputs": INPUTS,
                    "anomaly_refits": REFITS,
                    "anomaly_development": DEVELOPMENT,
                    "params": PARAMS,
                },
                outputs="anomaly_report",
                name="anomaly_report_node",
            ),
        ]
    )
