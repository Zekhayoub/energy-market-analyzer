"""Backtest pipeline: savings of an industrial client that shifts load with the predictions.

At 11:00 J the client fixes the hourly schedule of J+1 from the predictions of
model A, then pays the realized prices (decisions D31 and D32). Inputs are the
predictions and alerts of the prediction pipeline; nothing is refitted.
"""

from __future__ import annotations

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from energy_market_analyzer.pipelines.backtest.nodes import (
    build_backtest_inputs,
    build_backtest_report,
    run_backtest,
)

PARAMS = "params:backtest"
# Not in the catalog: kept in memory between nodes.
INPUTS = "backtest_inputs"
COVERAGE = "backtest_coverage"
SENSITIVITY = "backtest_sensitivity"


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Create the backtest pipeline (3 nodes).

    Returns:
        `backtest_daily` (data/07_model_output/) and `backtest_report` (data/08_reporting/).
    """
    return pipeline(
        [
            node(
                build_backtest_inputs,
                inputs=["predictions_model_a", "alerts", PARAMS],
                outputs=[INPUTS, COVERAGE],
                name="backtest_inputs_node",
            ),
            node(
                run_backtest,
                inputs=[INPUTS, PARAMS],
                outputs=["backtest_daily", SENSITIVITY],
                name="run_backtest_node",
            ),
            node(
                build_backtest_report,
                inputs=["backtest_daily", SENSITIVITY, COVERAGE, PARAMS],
                outputs="backtest_report",
                name="backtest_report_node",
            ),
        ]
    )
