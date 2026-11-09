"""Tests for M3 Labels: hand-made exact numerical verification."""

import numpy as np
import pandas as pd
import pytest
from alphacheck.labels import generate_excess_labels


def test_hand_made_labels_example():
    """Verify exact numerical excess returns and binary targets on toy data."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")

    # Stock prices: day 0 is 100, day 5 is 110 (+10% gain)
    stock_prices = [100.0, 102.0, 101.0, 104.0, 105.0, 110.0, 108.0, 112.0, 115.0, 120.0]
    # Bench prices: day 0 is 1000, day 5 is 1050 (+5% gain)
    bench_prices = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0, 1060.0, 1070.0, 1080.0, 1090.0]

    df_stock = pd.DataFrame({"adj_close": stock_prices}, index=dates)
    df_bench = pd.DataFrame({"adj_close": bench_prices}, index=dates)

    labels = generate_excess_labels(df_stock, df_bench, horizon=5)

    # 1. Total rows must be len(dates) - 5 = 10 - 5 = 5
    assert len(labels) == 5

    # 2. At index 0 (day 0):
    # stock forward return: (110 - 100) / 100 = +0.10 (+10%)
    # bench forward return: (1050 - 1000) / 1000 = +0.05 (+5%)
    # excess return: 0.10 - 0.05 = +0.05 (+5%)
    # target: 1 (outperformed)
    assert pytest.approx(labels["fwd_ret_5d"].iloc[0], rel=1e-5) == 0.10
    assert pytest.approx(labels["bench_fwd_ret_5d"].iloc[0], rel=1e-5) == 0.05
    assert pytest.approx(labels["excess_ret_5d"].iloc[0], rel=1e-5) == 0.05
    assert labels["target"].iloc[0] == 1

    # At index 1:
    # stock forward return: (108 - 102) / 102 = 6 / 102 ~= 0.05882
    # bench forward return: (1060 - 1010) / 1010 = 50 / 1010 ~= 0.04950
    # excess return: positive -> target = 1
    assert labels["target"].iloc[1] == 1

    # No NaN values allowed in the output
    assert labels.isna().sum().sum() == 0
