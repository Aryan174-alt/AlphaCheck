"""Tests for M7 Metrics: exact hand-computed values on toy data."""

import numpy as np
import pandas as pd
import pytest
from alphacheck.metrics import (
    compute_annualized_return,
    compute_annualized_volatility,
    compute_max_drawdown,
    compute_sharpe_ratio,
    evaluate_strategy_performance,
)


def test_hand_computed_metrics():
    # 252 days: exactly doubles in 1 year (100% gain)
    daily_gain = (2.0 ** (1.0 / 252)) - 1.0
    daily_rets = pd.Series([daily_gain] * 252)
    equity = (1.0 + daily_rets).cumprod() * 100.0

    # 1. CAGR must equal exactly 1.0 (100%)
    cagr = compute_annualized_return(daily_rets)
    assert pytest.approx(cagr, rel=1e-3) == 1.0

    # 2. Drawdown on strictly monotonic series must be 0.0
    mdd_flat = compute_max_drawdown(equity)
    assert mdd_flat == 0.0

    # 3. Known drawdown test: peak 100 -> trough 70 -> recovery 90
    drawdown_equity = pd.Series([100.0, 105.0, 73.5, 90.0])  # Peak is 105.0, min after is 73.5 -> (73.5 - 105)/105 = -30%
    mdd = compute_max_drawdown(drawdown_equity)
    assert pytest.approx(mdd, rel=1e-5) == 0.30

    # 4. Scorecard structure
    scorecard = evaluate_strategy_performance(equity, daily_rets)
    assert "sharpe_ratio" in scorecard
    assert "sharpe_ci_lower_95" in scorecard
    assert "max_drawdown" in scorecard

