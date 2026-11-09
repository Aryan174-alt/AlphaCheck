"""Tests for M4 Baselines: weight allocations and sums."""

import numpy as np
import pandas as pd
import pytest
from alphacheck.baselines import (
    generate_equal_weight_basket_weights,
    generate_momentum_rule_weights,
    generate_random_signals,
)


def test_equal_weight_and_momentum_weights():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    tickers = ["A", "B", "C", "D", "E"]

    records = []
    feat_records = []
    for d in dates:
        for i, t in enumerate(tickers):
            records.append({"date": d, "ticker": t, "adj_close": 100.0 + i})
            feat_records.append({"date": d, "ticker": t, "ret_20d": float(i)})

    panel = pd.DataFrame(records).set_index(["date", "ticker"])
    feats = pd.DataFrame(feat_records).set_index(["date", "ticker"])

    # 1. Equal-weight weights must sum to 1.0 on every day
    eq_w = generate_equal_weight_basket_weights(panel)
    assert pytest.approx(eq_w.sum(axis=1).iloc[0], rel=1e-5) == 1.0
    assert pytest.approx(eq_w["A"].iloc[0], rel=1e-5) == 0.20

    # 2. Momentum weights for top_k=2 must pick highest ret_20d ('D' and 'E')
    mom_w = generate_momentum_rule_weights(feats, momentum_col="ret_20d", top_k=2)
    assert pytest.approx(mom_w.sum(axis=1).iloc[0], rel=1e-5) == 1.0
    assert mom_w["E"].iloc[0] == 0.5
    assert mom_w["D"].iloc[0] == 0.5
    assert mom_w["A"].iloc[0] == 0.0

    # 3. Random signal generates valid non-negative weights summing to 1.0
    rnd_list = generate_random_signals(panel, top_k=2, seeds=[42])
    assert len(rnd_list) == 1
    assert pytest.approx(rnd_list[0].sum(axis=1).iloc[0], rel=1e-5) == 1.0
