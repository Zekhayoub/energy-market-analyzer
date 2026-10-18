"""Prediction pipeline: models A and B refitted each month on the past, intervals and alerts."""

from energy_market_analyzer.pipelines.prediction.pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
