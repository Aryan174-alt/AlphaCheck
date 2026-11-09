"""Tests for M6 Backtest: friction deduction and cash reconciliation."""

import numpy as np
import pandas as pd
import pytest
from alphacheck.backtest import run_portfolio_backtest


def test_backtest_costs_and_reconciliation():
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    # Stock price flat at 100.0
    prices = pd.DataFrame({"STOCK_A": [100.0] * 5}, index=dates)

    # Day 0: 0% weight; Day 1: 100% weight; Day 2..4: 100% weight
    weights = pd.DataFrame({"STOCK_A": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)

    init_cap = 100_000.0
    fee_rate = 0.0010  # 10 bps
    slip_rate = 0.0002 # 2 bps (total 12 bps)

    res = run_portfolio_backtest(
        target_weights=weights,
        price_matrix=prices,
        initial_capital=init_cap,
        fee_rate=fee_rate,
        slippage_rate=slip_rate,
        execution_lag_days=1,
    )

    # Execution lag is 1 day, so the buy happens on Day 2 (index 2)
    # Trade value ~ 100,000 INR
    # Costs paid should be ~ 100,000 * 0.0012 = 120 INR
    assert len(res.trade_log) == 1
    assert pytest.approx(res.total_costs_paid, rel=1e-2) == 120.0

    # Ending equity must equal initial_capital - total_costs_paid because stock price never changed
    assert pytest.approx(res.equity_curve.iloc[-1], rel=1e-3) == init_cap - res.total_costs_paid

