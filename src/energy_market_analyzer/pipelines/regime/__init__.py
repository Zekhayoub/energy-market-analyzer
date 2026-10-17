"""Regime pipeline: market regimes of each day (filtered and smoothed) with a Gaussian HMM."""

from energy_market_analyzer.pipelines.regime.pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
