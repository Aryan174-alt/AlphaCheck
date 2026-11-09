"""M4 Baselines: Benchmark and heuristic baseline strategies.

1. Buy-and-Hold Benchmark: passive index exposure (^NSEI).
2. Equal-Weight Basket: equal weight across all available universe stocks.
3. Cross-Sectional Momentum Rule: rank stocks by past 20-day return, hold top K.
4. Random Signal: Monte Carlo simulation across multiple seeds to test chance level.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def generate_buy_and_hold_weights(bench_df: pd.DataFrame) -> pd.DataFrame:
    """Generate weights for passive benchmark (100% in benchmark)."""
    weights = pd.DataFrame(1.0, index=bench_df.index, columns=["BENCHMARK"])
    return weights


def generate_equal_weight_basket_weights(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Generate equal-weight allocation across all available stocks on each date.

    Returns DataFrame with DatetimeIndex and ticker columns.
    """
    tickers = panel_df.index.get_level_values("ticker").unique()
    dates = panel_df.index.get_level_values("date").unique().sort_values()

    # Create unstacked presence matrix
    unstacked = panel_df["adj_close"].unstack(level="ticker").reindex(dates)
    valid_counts = unstacked.notna().sum(axis=1)

    weights = unstacked.notna().astype(float).div(valid_counts.replace(0, np.nan), axis=0)
    return weights.fillna(0.0)


def generate_momentum_rule_weights(
    panel_features: pd.DataFrame,
    momentum_col: str = "ret_20d",
    top_k: int = 10,
) -> pd.DataFrame:
    """Cross-sectional momentum rule: selects top_k stocks with highest past momentum."""
    # Pivot momentum values
    mom_matrix = panel_features[momentum_col].unstack(level="ticker")

    weights = pd.DataFrame(0.0, index=mom_matrix.index, columns=mom_matrix.columns)

    for dt, row in mom_matrix.iterrows():
        valid_row = row.dropna()
        if len(valid_row) == 0:
            continue
        k = min(top_k, len(valid_row))
        top_tickers = valid_row.nlargest(k).index
        weights.loc[dt, top_tickers] = 1.0 / k

    return weights


def generate_random_signals(
    panel_df: pd.DataFrame,
    top_k: int = 10,
    seeds: list[int] = [42, 101, 202, 303, 404],
) -> list[pd.DataFrame]:
    """Generate multiple random portfolio selections to establish stochastic chance performance."""
    price_matrix = panel_df["adj_close"].unstack(level="ticker")
    tickers = list(price_matrix.columns)
    random_weight_frames = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        weights = pd.DataFrame(0.0, index=price_matrix.index, columns=tickers)

        for dt, row in price_matrix.iterrows():
            valid_tickers = [t for t in tickers if pd.notna(row[t])]
            if len(valid_tickers) == 0:
                continue
            k = min(top_k, len(valid_tickers))
            chosen = rng.choice(valid_tickers, size=k, replace=False)
            weights.loc[dt, chosen] = 1.0 / k

        random_weight_frames.append(weights)

    return random_weight_frames
