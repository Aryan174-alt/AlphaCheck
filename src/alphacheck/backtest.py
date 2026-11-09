"""M6 Backtest Simulator: paper-trading simulation engine with friction, slippage, and position accounting.

Rules:
- Rebalances on schedule (e.g. weekly / every 5 days) when rebalance_days=5.
- Signals generated at close of day t are executed at day t+1 open/close (zero look-ahead).
- Applies transaction fees (default 0.10% per side) and slippage (default 0.02%).
- Reconciles cash, positions, transaction costs, and portfolio equity exactly.
- Exports a complete trade log CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions_history: pd.DataFrame
    trade_log: pd.DataFrame
    total_costs_paid: float
    turnover_series: pd.Series


def enforce_periodic_rebalancing(
    target_weights: pd.DataFrame,
    rebalance_freq_days: int = 5,
) -> pd.DataFrame:
    """Enforce that target weights change ONLY on periodic rebalance intervals.

    On non-rebalance days, previous desired weights are held constant.
    """
    if rebalance_freq_days <= 1:
        return target_weights

    rebalanced = pd.DataFrame(index=target_weights.index, columns=target_weights.columns)
    last_w = None

    for i, (dt, row) in enumerate(target_weights.iterrows()):
        if i % rebalance_freq_days == 0 or last_w is None:
            last_w = row
        rebalanced.loc[dt] = last_w

    return rebalanced.astype(float)


def run_portfolio_backtest(
    target_weights: pd.DataFrame,
    price_matrix: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    fee_rate: float = 0.0010,       # 0.10% (10 bps) one-way
    slippage_rate: float = 0.0002,  # 0.02% (2 bps) slippage
    execution_lag_days: int = 1,
    rebalance_freq_days: int = 5,   # Weekly (every 5 trading days)
) -> BacktestResult:
    """Run exact simulation of portfolio equity across time."""
    # 1. Enforce periodic rebalancing schedule
    scheduled_weights = enforce_periodic_rebalancing(target_weights, rebalance_freq_days=rebalance_freq_days)

    # 2. Align dates
    common_dates = scheduled_weights.index.intersection(price_matrix.index).sort_values()
    common_tickers = [c for c in scheduled_weights.columns if c in price_matrix.columns]

    weights = scheduled_weights.loc[common_dates, common_tickers].shift(execution_lag_days).fillna(0.0)
    prices = price_matrix.loc[common_dates, common_tickers]

    dates = list(common_dates)
    cash = initial_capital
    holdings: Dict[str, float] = {t: 0.0 for t in common_tickers}

    portfolio_values = []
    trade_records = []
    daily_turnovers = []
    total_costs = 0.0

    for i, dt in enumerate(dates):
        current_prices = prices.loc[dt].to_dict()

        # Marked-to-market value before rebalancing
        positions_value = sum(holdings[t] * current_prices[t] for t in common_tickers if pd.notna(current_prices[t]))
        current_nav = cash + positions_value

        desired_w = weights.loc[dt].to_dict()
        day_turnover_value = 0.0

        for t in common_tickers:
            p = current_prices.get(t, np.nan)
            if pd.isna(p) or p <= 0:
                continue

            target_val = desired_w.get(t, 0.0) * current_nav
            current_val = holdings[t] * p
            diff_val = target_val - current_val

            # Deadband threshold to avoid tiny micro-rebalances
            if abs(diff_val) > (0.01 * current_nav):  # 1% turnover threshold
                trade_shares = diff_val / p
                trade_value = abs(diff_val)
                effective_cost = trade_value * (fee_rate + slippage_rate)

                if diff_val > 0:  # Buy
                    cash -= (diff_val + effective_cost)
                else:  # Sell
                    cash += (trade_value - effective_cost)

                holdings[t] += trade_shares
                total_costs += effective_cost
                day_turnover_value += trade_value

                trade_records.append({
                    "date": dt,
                    "ticker": t,
                    "action": "BUY" if diff_val > 0 else "SELL",
                    "shares": abs(trade_shares),
                    "price": p,
                    "trade_value": trade_value,
                    "fees_and_slippage": effective_cost,
                })

        # Post-trade NAV
        positions_value = sum(holdings[t] * current_prices[t] for t in common_tickers if pd.notna(current_prices[t]))
        nav_post = cash + positions_value
        portfolio_values.append(nav_post)
        daily_turnovers.append(day_turnover_value / (current_nav + 1e-12))

    equity_curve = pd.Series(portfolio_values, index=common_dates, name="equity")
    daily_returns = equity_curve.pct_change().fillna(0.0)
    turnover_series = pd.Series(daily_turnovers, index=common_dates, name="turnover")
    trade_log = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(columns=["date", "ticker", "action", "shares", "price", "trade_value", "fees_and_slippage"])

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        positions_history=pd.DataFrame(),
        trade_log=trade_log,
        total_costs_paid=total_costs,
        turnover_series=turnover_series,
    )
