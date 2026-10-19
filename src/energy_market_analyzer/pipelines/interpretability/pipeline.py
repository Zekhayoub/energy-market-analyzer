"""Interpretability pipeline: SHAP values of the published predictions and the morning report.

The selected variant of models A and B is refitted month by month exactly as in the
prediction pipeline and explained row by row (decision D33); the morning report of one
decision day gathers what the procurement team reads at 11:00 (decision D34). Its node
reads persisted tables only, so a past morning can be replayed on its own:
`kedro run --pipeline=interpretability --nodes=morning_report_node`.
"""

from __future__ import annotations

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from energy_market_analyzer.pipelines.interpretability.morning_report import (
    build_morning_report,
)
from energy_market_analyzer.pipelines.interpretability.nodes import (
    build_shap_report,
    explain_model_a,
    explain_model_b,
)
from energy_market_analyzer.pipelines.prediction.nodes import (
    build_model_a_inputs,
    build_model_b_inputs,
)

PARAMS = "params:interpretability"
PREDICTION_PARAMS = "params:prediction"
# Not in the catalog: kept in memory between nodes.
INPUTS_A = "explanation_inputs_a"
INPUTS_B = "explanation_inputs_b"
CHECKS_A = "shap_checks_a"
CHECKS_B = "shap_checks_b"


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Create the interpretability pipeline (6 nodes).

    Returns:
        `shap_model_a`, `shap_model_b` (data/07_model_output/), `shap_report`,
        `morning_report` and `morning_report_markdown` (data/08_reporting/).
    """
    return pipeline(
        [
            node(
                build_model_a_inputs,
                inputs=["features_model_a", "regimes", PREDICTION_PARAMS],
                outputs=INPUTS_A,
                name="explanation_inputs_a_node",
            ),
            node(
                build_model_b_inputs,
                inputs=["features_model_b", "regimes", PREDICTION_PARAMS],
                outputs=INPUTS_B,
                name="explanation_inputs_b_node",
            ),
            node(
                explain_model_a,
                inputs=[
                    INPUTS_A,
                    "predictions_model_a",
                    "prediction_report",
                    PREDICTION_PARAMS,
                    PARAMS,
                ],
                outputs=["shap_model_a", CHECKS_A],
                name="explain_model_a_node",
            ),
            node(
                explain_model_b,
                inputs=[
                    INPUTS_B,
                    "predictions_model_b",
                    "prediction_report",
                    PREDICTION_PARAMS,
                    PARAMS,
                ],
                outputs=["shap_model_b", CHECKS_B],
                name="explain_model_b_node",
            ),
            node(
                build_shap_report,
                inputs=[
                    "shap_model_a",
                    "shap_model_b",
                    INPUTS_A,
                    INPUTS_B,
                    CHECKS_A,
                    CHECKS_B,
                    PARAMS,
                ],
                outputs="shap_report",
                name="shap_report_node",
            ),
            node(
                build_morning_report,
                inputs=[
                    "predictions_model_a",
                    "predictions_model_b",
                    "alerts",
                    "shap_model_a",
                    "shap_model_b",
                    "features_model_a",
                    "features_model_b",
                    "regimes",
                    "anomalies",
                    "features_model_c",
                    "backtest_daily",
                    "prediction_report",
                    PARAMS,
                    "params:backtest",
                    "params:anomaly",
                ],
                outputs=["morning_report", "morning_report_markdown"],
                name="morning_report_node",
            ),
        ]
    )
