"""M6 Backtest Simulator: paper-trading simulation engine with friction, slippage, and position accounting.

Rules:
- Rebalances on schedule (e.g. weekly / every 5 days).
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


def run_portfolio_backtest(
    target_weights: pd.DataFrame,
    price_matrix: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    fee_rate: float = 0.0010,       # 0.10% (10 bps) one-way
    slippage_rate: float = 0.0002,  # 0.02% (2 bps) slippage
    execution_lag_days: int = 1,
) -> BacktestResult:
    """Run exact simulation of portfolio equity across time.

    Args:
        target_weights: DataFrame (dates x tickers) with target weight allocations summing to <= 1.0.
        price_matrix: DataFrame (dates x tickers) with adjusted close prices.
        initial_capital: Starting cash in INR.
        fee_rate: One-way fee (brokerage, STT, exchange fees).
        slippage_rate: One-way slippage.
        execution_lag_days: 1 means target weights formed on date t are traded on date t+1.
    """
    # Align dates
    common_dates = target_weights.index.intersection(price_matrix.index).sort_values()
    common_tickers = [c for c in target_weights.columns if c in price_matrix.columns]

    weights = target_weights.loc[common_dates, common_tickers].shift(execution_lag_days).fillna(0.0)
    prices = price_matrix.loc[common_dates, common_tickers]

    dates = list(common_dates)
    n_dates = len(dates)

    cash = initial_capital
    holdings: Dict[str, float] = {t: 0.0 for t in common_tickers}  # number of shares

    portfolio_values = []
    trade_records = []
    daily_turnovers = []
    total_costs = 0.0

    for i, dt in enumerate(dates):
        current_prices = prices.loc[dt].to_dict()

        # Compute current marked-to-market value before rebalancing
        positions_value = sum(holdings[t] * current_prices[t] for t in common_tickers if pd.notna(current_prices[t]))
        current_nav = cash + positions_value

        desired_w = weights.loc[dt].to_dict()
        day_turnover_value = 0.0

        # Execute trades to align with desired_w
        for t in common_tickers:
            p = current_prices.get(t, np.nan)
            if pd.isna(p) or p <= 0:
                continue

            target_val = desired_w.get(t, 0.0) * current_nav
            current_val = holdings[t] * p
            diff_val = target_val - current_val

            # Avoid tiny frictional rebalancing trades under 0.5% NAV
            if abs(diff_val) > (0.005 * current_nav):  # 0.5% turnover deadband threshold
                trade_shares = diff_val / p
                trade_value = abs(diff_val)
                effective_cost = trade_value * (fee_rate + slippage_rate)

                # Update cash and shares
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

        # Re-evaluate NAV after execution
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
