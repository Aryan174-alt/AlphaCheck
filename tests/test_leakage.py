"""Leakage tests: proves features at time t never use information from t+1 or beyond."""

import numpy as np
import pandas as pd
import pytest
from alphacheck.features import build_stock_features


def test_feature_no_future_leakage():
    """Modifying future prices (t+1..T) must produce zero change in feature values at time t."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(42)
    base_prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    base_bench = 1000.0 * np.exp(np.cumsum(np.random.normal(0, 0.008, 100)))

    df_stock = pd.DataFrame(
        {"adj_close": base_prices, "volume": np.random.uniform(1000, 5000, 100)},
        index=dates,
    )
    df_bench = pd.DataFrame({"adj_close": base_bench}, index=dates)

    # Base features
    feat_base = build_stock_features(df_stock, df_bench)

    # Test at cutoff date t = index 70
    cutoff_idx = 70
    cutoff_date = dates[cutoff_idx]

    # Create mutated stock where future prices (t+1..end) are drastically altered (+500%)
    df_stock_mutated = df_stock.copy()
    df_stock_mutated.iloc[cutoff_idx + 1 :, df_stock_mutated.columns.get_loc("adj_close")] *= 5.0
    df_stock_mutated.iloc[cutoff_idx + 1 :, df_stock_mutated.columns.get_loc("volume")] *= 10.0

    df_bench_mutated = df_bench.copy()
    df_bench_mutated.iloc[cutoff_idx + 1 :, df_bench_mutated.columns.get_loc("adj_close")] *= 3.0

    feat_mutated = build_stock_features(df_stock_mutated, df_bench_mutated)

    # Check exact equality for all rows up to and including cutoff_date
    sub_base = feat_base.loc[:cutoff_date]
    sub_mutated = feat_mutated.loc[:cutoff_date]

    # Values must be identical up to machine epsilon
    diff = (sub_base - sub_mutated).abs().max().max()
    assert np.isnan(diff) or diff < 1e-10, f"Future leakage detected! Max difference at/before cutoff: {diff}"


def test_rolling_windows_are_past_only():
    """Ensure rolling operations do not have center=True or negative shifts."""
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    prices = np.arange(1.0, 31.0)
    df_stock = pd.DataFrame({"adj_close": prices, "volume": [100.0] * 30}, index=dates)
    df_bench = pd.DataFrame({"adj_close": prices * 2}, index=dates)

    feats = build_stock_features(df_stock, df_bench, return_windows=[5])
    # In pandas, pct_change(5) at index 5 is (prices[5] / prices[0]) - 1 = (6.0 / 1.0) - 1 = 5.0
    # and index 0..4 are NaN because 5 prior steps are required.
    assert pytest.approx(feats["ret_5d"].iloc[5], rel=1e-5) == (6.0 / 1.0) - 1.0
    assert np.isnan(feats["ret_5d"].iloc[4])
