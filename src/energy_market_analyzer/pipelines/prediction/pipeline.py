"""Prediction pipeline: models A (J+1, hourly) and B (J+2 to J+4, daily mean), decided at 11:00 J.

Each month every model is refitted on the past only; the variant kept and the
interval calibration are chosen on the development period by rules fixed in
advance (decisions D25 to D30). Inputs are the feature tables and the regimes;
the outputs of the anomaly pipeline are never read (decision D24).
"""

from __future__ import annotations

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from energy_market_analyzer.pipelines.prediction.nodes import (
    build_alerts,
    build_model_a_inputs,
    build_model_b_inputs,
    build_prediction_report,
    finalize_predictions,
    walk_forward_model_a,
    walk_forward_model_b,
)

PARAMS = "params:prediction"
# Not in the catalog: kept in memory between nodes.
INPUTS_A = "prediction_inputs_a"
INPUTS_B = "prediction_inputs_b"
RAW_A = "raw_predictions_a"
RAW_B = "raw_predictions_b"
REFITS_A = "prediction_refits_a"
REFITS_B = "prediction_refits_b"
CHOICES = "prediction_choices"
THRESHOLDS = "alert_thresholds"


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Create the prediction pipeline (7 nodes).

    Returns:
        `predictions_model_a`, `predictions_model_b` and `alerts`
        (data/07_model_output/), and `prediction_report` (data/08_reporting/).
    """
    return pipeline(
        [
            node(
                build_model_a_inputs,
                inputs=["features_model_a", "regimes", PARAMS],
                outputs=INPUTS_A,
                name="model_a_inputs_node",
            ),
            node(
                build_model_b_inputs,
                inputs=["features_model_b", "regimes", PARAMS],
                outputs=INPUTS_B,
                name="model_b_inputs_node",
            ),
            node(
                walk_forward_model_a,
                inputs=[INPUTS_A, PARAMS],
                outputs=[RAW_A, REFITS_A],
                name="walk_forward_model_a_node",
            ),
            node(
                walk_forward_model_b,
                inputs=[INPUTS_B, PARAMS],
                outputs=[RAW_B, REFITS_B],
                name="walk_forward_model_b_node",
            ),
            node(
                finalize_predictions,
                inputs=[RAW_A, RAW_B, PARAMS],
                outputs=["predictions_model_a", "predictions_model_b", CHOICES],
                name="finalize_predictions_node",
            ),
            node(
                build_alerts,
                inputs=["predictions_model_a", INPUTS_A, "regimes", PARAMS],
                outputs=["alerts", THRESHOLDS],
                name="alerts_node",
            ),
            node(
                build_prediction_report,
                inputs={
                    "predictions_model_a": "predictions_model_a",
                    "predictions_model_b": "predictions_model_b",
                    "alerts": "alerts",
                    "prediction_choices": CHOICES,
                    "refits_a": REFITS_A,
                    "refits_b": REFITS_B,
                    "alert_thresholds": THRESHOLDS,
                    "params": PARAMS,
                },
                outputs="prediction_report",
                name="prediction_report_node",
            ),
        ]
    )
