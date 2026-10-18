"""Anomaly pipeline: hours unusual for their filtered regime, with per-regime Isolation Forests."""

from energy_market_analyzer.pipelines.anomaly.pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
