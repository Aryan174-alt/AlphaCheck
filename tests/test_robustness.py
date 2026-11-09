"""Tests for M9 Robustness."""

import numpy as np
import pandas as pd
import pytest
from alphacheck.robustness import evaluate_cost_sensitivity, evaluate_regimes


def test_cost_sensitivity_monotonic_decay(tmp_path):
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"STOCK": [100.0 + i for i in range(10)]}, index=dates)
    weights = pd.DataFrame({"STOCK": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]}, index=dates)

    out_csv = tmp_path / "costs.csv"
    df = evaluate_cost_sensitivity(
        weights=weights,
        prices=prices,
        fee_levels=[0.0, 0.0010, 0.0030],
        output_path=str(out_csv),
    )

    assert len(df) == 3
    # Higher friction must result in equal or lower CAGR / equity
    assert df.iloc[0]["cagr"] >= df.iloc[1]["cagr"] >= df.iloc[2]["cagr"]
    # Total costs paid must be monotonically increasing with fee level
    assert df.iloc[0]["total_costs_paid"] <= df.iloc[1]["total_costs_paid"] <= df.iloc[2]["total_costs_paid"]

