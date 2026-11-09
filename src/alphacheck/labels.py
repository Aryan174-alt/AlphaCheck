"""M3 Labels: 5-day forward excess return binary targets vs Nifty 50 benchmark.

y_{i,t} = 1 if R_{i, t->t+5} > R_{bench, t->t+5} else 0
Rows without a full forward 5-day window are strictly dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forward_returns(series: pd.Series, horizon: int = 5) -> pd.Series:
    """Compute forward percentage return: (P_{t+horizon} / P_t) - 1.

    The last `horizon` rows will naturally be NaN because future prices do not exist yet.
    """
    future_price = series.shift(-horizon)
    forward_ret = (future_price / series) - 1.0
    return forward_ret


def generate_excess_labels(
    stock_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Generate forward excess return and binary classification target for a single stock.

    Returns:
        DataFrame with columns: ['fwd_ret_5d', 'bench_fwd_ret_5d', 'excess_ret_5d', 'target']
        Rows without the complete forward horizon are dropped.
    """
    df = stock_df.copy().sort_index()
    bench = bench_df.copy().sort_index()

    fwd_stock = compute_forward_returns(df["adj_close"], horizon=horizon)
    bench_reindexed = bench["adj_close"].reindex(df.index)
    fwd_bench = compute_forward_returns(bench_reindexed, horizon=horizon)

    excess = fwd_stock - fwd_bench
    # Binary label: 1 if stock beats Nifty over horizon, else 0
    target = (excess > 0.0).astype(int)

    out = pd.DataFrame(
        {
            "fwd_ret_5d": fwd_stock,
            "bench_fwd_ret_5d": fwd_bench,
            "excess_ret_5d": excess,
            "target": target,
        },
        index=df.index,
    )

    # Strictly drop rows without full forward window
    valid_mask = fwd_stock.notna() & fwd_bench.notna()
    return out.loc[valid_mask]


def generate_panel_labels(
    panel_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Generate forward labels across all stocks in the multi-index panel (date, ticker)."""
    tickers = panel_df.index.get_level_values("ticker").unique()
    label_list = []

    for t in tickers:
        sub = panel_df.xs(t, level="ticker")
        l_sub = generate_excess_labels(sub, bench_df, horizon=horizon)
        l_sub["ticker"] = t
        l_sub = l_sub.reset_index()
        label_list.append(l_sub)

    all_labels = pd.concat(label_list, ignore_index=True)
    all_labels = all_labels.set_index(["date", "ticker"]).sort_index()
    return all_labels

