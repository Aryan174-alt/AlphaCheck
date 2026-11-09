"""M1 Data Pipeline: yfinance download, dividend/split adjustments, parquet caching, and quality report."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_ticker_data(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str = "data/raw",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download OHLCV data for a single ticker with parquet caching.

    Returns DataFrame with DatetimeIndex and standard lowercase columns:
    open, high, low, close, volume, dividends, stock_splits.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_ticker = ticker.replace("^", "IDX_").replace(".", "_")
    cache_path = Path(cache_dir) / f"{safe_ticker}.parquet"

    if not force_refresh and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return df

    # Download from yfinance (auto_adjust=False to keep unadjusted and splits/divs transparent)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        actions=True,
    )

    if raw.empty:
        raise ValueError(f"No price data returned for ticker: {ticker}")

    # Flatten multi-level columns if returned by yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"

    # Standardize column naming
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    }
    df = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})

    # Ensure required columns exist
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    df = df.sort_index()
    # Cache locally to parquet
    df.to_parquet(cache_path, index=True)
    return df


def clean_and_align_universe(
    tickers: List[str],
    benchmark_ticker: str,
    start: str,
    end: str,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download, clean, and align full universe against benchmark calendar.

    Returns:
        prices_df: MultiIndex (date, ticker) panel with clean adjusted prices and volume.
        bench_df: Benchmark index DataFrame indexed by date.
    """
    os.makedirs(processed_dir, exist_ok=True)
    panel_cache = Path(processed_dir) / "panel_clean.parquet"
    bench_cache = Path(processed_dir) / "bench_clean.parquet"

    if panel_cache.exists() and bench_cache.exists():
        return pd.read_parquet(panel_cache), pd.read_parquet(bench_cache)

    bench_df = fetch_ticker_data(benchmark_ticker, start, end, cache_dir=raw_dir)
    bench_calendar = bench_df.index.unique().sort_values()

    ticker_frames: List[pd.DataFrame] = []
    for t in tickers:
        try:
            tdf = fetch_ticker_data(t, start, end, cache_dir=raw_dir)
            # Reindex to benchmark calendar to avoid lookahead/missing date bias
            tdf = tdf.reindex(bench_calendar)
            tdf["adj_close"] = tdf["adj_close"].ffill().bfill()
            tdf["volume"] = tdf["volume"].fillna(0.0)
            tdf["ticker"] = t
            tdf = tdf.reset_index()
            ticker_frames.append(tdf)
        except Exception as e:
            continue

    if not ticker_frames:
        raise RuntimeError("No stock data could be downloaded or loaded.")

    panel = pd.concat(ticker_frames, ignore_index=True)
    panel = panel.set_index(["date", "ticker"]).sort_index()

    panel.to_parquet(panel_cache)
    bench_df.to_parquet(bench_cache)
    return panel, bench_df


def generate_data_quality_report(
    panel: pd.DataFrame,
    bench_df: pd.DataFrame,
    report_path: str = "reports/tables/data_quality.csv",
) -> pd.DataFrame:
    """Generate quality report checking missing values, zero volumes, and date coverage."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    records = []
    # Benchmark stats
    records.append({
        "ticker": "BENCHMARK",
        "rows": len(bench_df),
        "start_date": str(bench_df.index.min().date()),
        "end_date": str(bench_df.index.max().date()),
        "missing_adj_close": int(bench_df["adj_close"].isna().sum()),
        "zero_volume_days": int((bench_df["volume"] == 0).sum()),
    })

    tickers = panel.index.get_level_values("ticker").unique()
    for t in tickers:
        sub = panel.xs(t, level="ticker")
        records.append({
            "ticker": t,
            "rows": len(sub),
            "start_date": str(sub.index.min().date()),
            "end_date": str(sub.index.max().date()),
            "missing_adj_close": int(sub["adj_close"].isna().sum()),
            "zero_volume_days": int((sub["volume"] == 0).sum()),
        })

    report_df = pd.DataFrame(records)
    report_df.to_csv(report_path, index=False)
    return report_df
