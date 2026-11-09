"""M2 Feature Engineering: strictly backward-looking technical, momentum, and volatility features.

All features are calculated using past observations only and strictly lagged/shifted
so feature at time t never peeks at time t+1 or beyond.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(series: pd.Series, window: int) -> pd.Series:
    """Compute backward-looking percentage returns: (P_t / P_{t-window}) - 1."""
    return series.pct_change(window)


def compute_rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute backward-looking annualized rolling volatility of 1-day returns."""
    daily_ret = series.pct_change(1)
    return daily_ret.rolling(window=window, min_periods=window).std() * np.sqrt(252)


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute backward-looking Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Compute backward-looking MACD line and Signal line using exponential moving averages."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_rolling_beta(
    stock_returns: pd.Series,
    bench_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling capital asset pricing beta against benchmark returns."""
    cov = stock_returns.rolling(window=window, min_periods=window).cov(bench_returns)
    bench_var = bench_returns.rolling(window=window, min_periods=window).var()
    beta = cov / (bench_var + 1e-12)
    return beta


def build_stock_features(
    stock_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    return_windows: list[int] = [1, 5, 10, 20, 60],
    volatility_window: int = 20,
    rsi_window: int = 14,
    beta_window: int = 60,
) -> pd.DataFrame:
    """Build all technical features for a single stock DataFrame (sorted by date).

    Every feature is strictly computed on past adjusted prices and volumes.
    """
    df = stock_df.copy().sort_index()
    bench = bench_df.copy().sort_index()

    features = pd.DataFrame(index=df.index)
    price = df["adj_close"]
    volume = df["volume"]

    # 1. Momentum & Returns
    for w in return_windows:
        features[f"ret_{w}d"] = compute_returns(price, w)

    # 2. Volatility
    features[f"vol_{volatility_window}d"] = compute_rolling_volatility(price, volatility_window)

    # 3. Technical Oscillators
    features[f"rsi_{rsi_window}"] = compute_rsi(price, rsi_window)
    macd, signal = compute_macd(price)
    features["macd"] = macd
    features["macd_signal"] = signal
    features["macd_hist"] = macd - signal

    # 4. Volume change (5d and 20d moving average ratio)
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    features["vol_ratio_20d"] = volume / (vol_ma20 + 1e-12)

    # 5. Beta vs Benchmark
    stock_daily = price.pct_change(1)
    bench_daily = bench["adj_close"].pct_change(1).reindex(df.index)
    features[f"beta_{beta_window}d"] = compute_rolling_beta(stock_daily, bench_daily, beta_window)

    return features


def build_panel_features(
    panel_df: pd.DataFrame,
    bench_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute feature matrices across all stocks in the multi-index panel (date, ticker)."""
    tickers = panel_df.index.get_level_values("ticker").unique()
    feat_list = []

    for t in tickers:
        sub = panel_df.xs(t, level="ticker")
        f_sub = build_stock_features(sub, bench_df)
        f_sub["ticker"] = t
        f_sub = f_sub.reset_index()
        feat_list.append(f_sub)

    all_feat = pd.concat(feat_list, ignore_index=True)
    all_feat = all_feat.set_index(["date", "ticker"]).sort_index()
    return all_feat
