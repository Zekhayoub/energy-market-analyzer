"""Ingestion pipeline: download every source, then build the hourly master table.

Run it with the default SequentialRunner. A parallel runner would send many
ENTSO-E requests at once and hit the API rate limit.
"""

from __future__ import annotations

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from energy_market_analyzer.pipelines.ingestion.nodes import (
    audit_quality,
    build_master,
    fetch_actual_load,
    fetch_crossborder_flows,
    fetch_day_ahead_prices,
    fetch_eua_prices,
    fetch_generation,
    fetch_load_forecast,
    fetch_ttf_prices,
    fetch_weather,
)

PARAMS = "params:ingestion"
# Added to the catalog by the hook in settings.py, masked (SecretStr).
API_KEY = "entsoe_api_key"


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Create the ingestion pipeline (10 nodes).

    Returns:
        8 download nodes writing data/01_raw/, then `build_master_node`
        (data/03_primary/master.parquet) and `audit_quality_node`
        (data/08_reporting/data_quality_report.json).
    """
    return pipeline(
        [
            node(
                fetch_day_ahead_prices,
                inputs=[PARAMS, API_KEY],
                outputs="entsoe_prices_raw",
                name="fetch_prices_node",
            ),
            node(
                fetch_actual_load,
                inputs=[PARAMS, API_KEY],
                outputs="entsoe_load_raw",
                name="fetch_load_node",
            ),
            node(
                fetch_load_forecast,
                inputs=[PARAMS, API_KEY],
                outputs="entsoe_load_forecast_raw",
                name="fetch_load_forecast_node",
            ),
            node(
                fetch_generation,
                inputs=[PARAMS, API_KEY],
                outputs="entsoe_generation_raw",
                name="fetch_generation_node",
            ),
            node(
                fetch_crossborder_flows,
                inputs=[PARAMS, API_KEY],
                outputs="entsoe_flows_raw",
                name="fetch_flows_node",
            ),
            node(fetch_ttf_prices, inputs=PARAMS, outputs="ttf_raw", name="fetch_ttf_node"),
            node(fetch_eua_prices, inputs=PARAMS, outputs="eua_raw", name="fetch_eua_node"),
            node(
                fetch_weather,
                inputs=PARAMS,
                outputs="weather_raw",
                name="fetch_weather_node",
            ),
            node(
                build_master,
                inputs={
                    "prices": "entsoe_prices_raw",
                    "load": "entsoe_load_raw",
                    "load_forecast": "entsoe_load_forecast_raw",
                    "generation": "entsoe_generation_raw",
                    "flows": "entsoe_flows_raw",
                    "ttf": "ttf_raw",
                    "eua": "eua_raw",
                    "weather": "weather_raw",
                    "params": PARAMS,
                },
                outputs="master",
                name="build_master_node",
            ),
            node(
                audit_quality,
                inputs={"master": "master", "prices": "entsoe_prices_raw", "params": PARAMS},
                outputs="data_quality_report",
                name="audit_quality_node",
            ),
        ]
    )
