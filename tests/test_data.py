"""Tests for M1 Data module."""

import os
import numpy as np
import pandas as pd
import pytest
from alphacheck.data import fetch_ticker_data, generate_data_quality_report, clean_and_align_universe


def test_fetch_ticker_data_caching(tmp_path):
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    mock_df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, 10),
            "high": np.linspace(102, 112, 10),
            "low": np.linspace(99, 109, 10),
            "close": np.linspace(101, 111, 10),
            "adj_close": np.linspace(101, 111, 10),
            "volume": [10000.0] * 10,
        },
        index=dates,
    )
    mock_df.index.name = "date"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cache_file = raw_dir / "TEST_TICKER.parquet"
    mock_df.to_parquet(cache_file)

    # Calling fetch should read directly from disk cache without web requests
    loaded = fetch_ticker_data("TEST_TICKER", "2023-01-01", "2023-01-15", cache_dir=str(raw_dir))
    assert len(loaded) == 10
    assert "adj_close" in loaded.columns
    assert list(loaded["adj_close"]) == list(mock_df["adj_close"])


def test_data_quality_report(tmp_path):
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    bench = pd.DataFrame({"adj_close": [100.0] * 5, "volume": [100.0] * 5}, index=dates)
    panel_records = []
    for t in ["TICK1", "TICK2"]:
        for d in dates:
            panel_records.append({"date": d, "ticker": t, "adj_close": 50.0, "volume": 10.0})
    panel = pd.DataFrame(panel_records).set_index(["date", "ticker"])

    rep_path = tmp_path / "dq.csv"
    rep = generate_data_quality_report(panel, bench, report_path=str(rep_path))
    assert len(rep) == 3  # BENCHMARK + 2 tickers
    assert (rep["missing_adj_close"] == 0).all()
    assert os.path.exists(str(rep_path))

