"""Tests for ingestion pipeline nodes.

Tests use synthetic data and mocked API responses. They do
NOT call the real ENTSO-E, yfinance, or Open-Meteo APIs.
Tests that use real data are marked @pytest.mark.integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestFetchDayAheadPrices:
    """Tests for the ENTSO-E day-ahead price fetcher."""

    def test_empty_api_response_raises(self) -> None:
        """If the API returns no data, raise RuntimeError."""
        from energy_market_analyzer.pipelines.ingestion.nodes import (
            fetch_day_ahead_prices,
        )

        params = {
            "country": "BE",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "retry_max_attempts": 1,
            "retry_backoff_factor": 1,
        }

        with patch("entsoe.EntsoePandasClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.query_day_ahead_prices.return_value = None
            mock_client_class.return_value = mock_client

            with pytest.raises(RuntimeError, match="Failed to fetch"):
                fetch_day_ahead_prices(params, "fake_api_key")

    def test_prices_returned_as_utc_naive(self) -> None:
        """Prices must be in UTC naive (no timezone info).

        The ENTSO-E API returns CET/CEST timestamps. We convert
        to UTC and strip the timezone info.
        """
        from energy_market_analyzer.pipelines.ingestion.nodes import (
            fetch_day_ahead_prices,
        )

        params = {
            "country": "BE",
            "start_date": "2024-01-01",
            "end_date": "2024-01-05",
            "retry_max_attempts": 1,
            "retry_backoff_factor": 1,
        }

        # Create mock data with CET timezone (like real API)
        index = pd.date_range("2024-01-01", "2024-01-04 23:00", freq="h", tz="Europe/Brussels")
        mock_prices = pd.Series(
            np.random.default_rng(42).normal(50, 10, len(index)),
            index=index,
        )

        with patch("entsoe.EntsoePandasClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.query_day_ahead_prices.return_value = mock_prices
            mock_client_class.return_value = mock_client

            df = fetch_day_ahead_prices(params, "fake_api_key")

            assert df.index.tz is None
            assert "price_eur_mwh" in df.columns
            assert len(df) > 0

    def test_dst_march_23_hours(self) -> None:
        """CET->CEST transition (last Sunday of March): 23 hours."""
        from energy_market_analyzer.pipelines.ingestion.nodes import (
            fetch_day_ahead_prices,
        )

        params = {
            "country": "BE",
            "start_date": "2024-03-31",
            "end_date": "2024-04-01",
            "retry_max_attempts": 1,
            "retry_backoff_factor": 1,
        }

        index = pd.date_range("2024-03-31", "2024-03-31 23:00", freq="h", tz="Europe/Brussels")
        mock_prices = pd.Series(
            np.random.default_rng(42).normal(50, 10, len(index)),
            index=index,
        )

        with patch("entsoe.EntsoePandasClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.query_day_ahead_prices.return_value = mock_prices
            mock_client_class.return_value = mock_client

            df = fetch_day_ahead_prices(params, "fake_api_key")

            assert df.index.tz is None
            assert df.index.is_monotonic_increasing
            assert len(df) == 23

    def test_dst_october_25_hours(self) -> None:
        """CEST->CET transition (last Sunday of October): 25 hours."""
        from energy_market_analyzer.pipelines.ingestion.nodes import (
            fetch_day_ahead_prices,
        )

        params = {
            "country": "BE",
            "start_date": "2024-10-27",
            "end_date": "2024-10-28",
            "retry_max_attempts": 1,
            "retry_backoff_factor": 1,
        }

        index_utc = pd.date_range("2024-10-26 22:00", periods=25, freq="h", tz="UTC")
        index_brussels = index_utc.tz_convert("Europe/Brussels")
        mock_prices = pd.Series(
            np.random.default_rng(42).normal(50, 10, 25),
            index=index_brussels,
        )

        with patch("entsoe.EntsoePandasClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.query_day_ahead_prices.return_value = mock_prices
            mock_client_class.return_value = mock_client

            df = fetch_day_ahead_prices(params, "fake_api_key")

            assert df.index.tz is None
            assert df.index.is_monotonic_increasing
            assert not df.index.has_duplicates
            assert len(df) == 25
