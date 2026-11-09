"""Tests for M5 validation and models."""

import numpy as np
import pandas as pd
import pytest
from alphacheck.models import build_gradient_boosting_model, build_logistic_model, get_feature_and_label_matrices
from alphacheck.validation import WalkForwardValidator


def test_walk_forward_embargo_gap():
    calendar = pd.date_range("2020-01-01", periods=1200, freq="B")
    validator = WalkForwardValidator(
        calendar=calendar,
        initial_train_days=500,
        step_days=100,
        test_days=100,
        embargo_days=5,
    )

    splits = list(validator.split())
    assert len(splits) > 0

    for s in splits:
        # Verify train ends strictly before test starts
        assert s.train_end < s.test_start

        # Calculate exact trading days difference between train_end and test_start
        train_end_loc = calendar.get_loc(s.train_end)
        test_start_loc = calendar.get_loc(s.test_start)
        gap = test_start_loc - train_end_loc - 1
        assert gap == s.embargo_days, f"Embargo gap violated! Expected {s.embargo_days}, got {gap}"


def test_models_fit_and_predict():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    idx = pd.MultiIndex.from_arrays([dates, ["STOCK"] * n], names=["date", "ticker"])

    X_df = pd.DataFrame({"feat1": np.random.randn(n), "feat2": np.random.randn(n)}, index=idx)
    y_df = pd.DataFrame({"target": np.random.choice([0, 1], size=n)}, index=idx)

    X, y = get_feature_and_label_matrices(X_df, y_df)
    assert len(X) == n

    # 1. Logistic
    lr = build_logistic_model()
    lr.fit(X, y)
    probs_lr = lr.predict_proba(X)[:, 1]
    assert len(probs_lr) == n
    assert (probs_lr >= 0.0).all() and (probs_lr <= 1.0).all()

    # 2. LightGBM
    lgbm = build_gradient_boosting_model(n_estimators=10)
    lgbm.fit(X, y)
    probs_lgb = lgbm.predict_proba(X)[:, 1]
    assert len(probs_lgb) == n
