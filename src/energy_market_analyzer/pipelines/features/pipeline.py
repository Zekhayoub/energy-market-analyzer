"""Features pipeline: one feature table per strategy, built from the master table.

Model A (hourly prices of J+1) and model B (daily means of J+2 to J+4) only see
what is known at 11:00 Brussels on the decision day J. Model C (regimes,
anomalies, retrospective analysis) is hourly and causal.
"""

from __future__ import annotations

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from energy_market_analyzer.pipelines.features.nodes import (
    build_daily_market,
    build_features_model_a,
    build_features_model_b,
    build_features_model_c,
    build_features_report,
)

PARAMS = "params:features"
# Not in the catalog: kept in memory between nodes.
DAILY_MARKET = "daily_market"


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Create the features pipeline (5 nodes).

    Returns:
        `build_daily_market_node` (in memory), the three feature tables
        (data/04_feature/) and `features_report_node`
        (data/08_reporting/features_report.json).
    """
    return pipeline(
        [
            node(
                build_daily_market,
                inputs=["master", PARAMS],
                outputs=DAILY_MARKET,
                name="build_daily_market_node",
            ),
            node(
                build_features_model_a,
                inputs=["master", DAILY_MARKET, PARAMS],
                outputs="features_model_a",
                name="features_model_a_node",
            ),
            node(
                build_features_model_b,
                inputs=[DAILY_MARKET, PARAMS],
                outputs="features_model_b",
                name="features_model_b_node",
            ),
            node(
                build_features_model_c,
                inputs=["master", DAILY_MARKET, PARAMS],
                outputs="features_model_c",
                name="features_model_c_node",
            ),
            node(
                build_features_report,
                inputs={
                    "features_model_a": "features_model_a",
                    "features_model_b": "features_model_b",
                    "features_model_c": "features_model_c",
                    "daily_market": DAILY_MARKET,
                    "params": PARAMS,
                },
                outputs="features_report",
                name="features_report_node",
            ),
        ]
    )
