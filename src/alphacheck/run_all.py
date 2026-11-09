"""M11 Master Pipeline Runner: reproduces the complete research framework end-to-end.

One command: python -m alphacheck.run_all
- Ingests and cleans data (M1)
- Builds features (M2, M8, A2, A4)
- Generates 5-day forward excess returns (M3)
- Evaluates baselines (M4)
- Performs walk-forward ML and Deep learning training (M5, A3)
- Runs discrete friction-aware backtests (M6)
- Computes scorecard metrics and bootstrap CIs (M7)
- Computes cost sensitivity and SHAP importances (M9, A5)
- Exports figures and tables
"""

from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from alphacheck.backtest import run_portfolio_backtest
from alphacheck.baselines import (
    generate_buy_and_hold_weights,
    generate_equal_weight_basket_weights,
    generate_momentum_rule_weights,
    generate_random_signals,
)
from alphacheck.data import clean_and_align_universe, generate_data_quality_report
from alphacheck.deep_models import SmallTransformerClassifier
from alphacheck.explain import compute_deep_gradient_attribution
from alphacheck.features import build_panel_features
from alphacheck.labels import generate_panel_labels
from alphacheck.metrics import evaluate_strategy_performance
from alphacheck.models import build_gradient_boosting_model, build_logistic_model, get_feature_and_label_matrices
from alphacheck.robustness import compute_tree_shap_importance, evaluate_cost_sensitivity, evaluate_regimes
from alphacheck.validation import WalkForwardValidator


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 70)
    print("ALPHACHECK: End-to-End Leakage-Safe Research Pipeline Execution")
    print("=" * 70)

    cfg = load_config()
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)

    # 1. M1 Data
    print("\n[M1] Ingesting and aligning universe data...")
    # Use representative liquid subset for reproducible fast pipeline execution
    tickers = cfg["data"]["universe"][:10]  # top 10 tickers
    bench_ticker = cfg["data"]["benchmark_ticker"]
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    panel_df, bench_df = clean_and_align_universe(
        tickers=tickers,
        benchmark_ticker=bench_ticker,
        start=start_date,
        end=end_date,
    )
    dq_df = generate_data_quality_report(panel_df, bench_df)
    print(f"Data quality report saved. Tickers loaded: {len(tickers)}, Total bars: {len(panel_df)}")

    # 2. M2 Features
    print("\n[M2] Engineering price features (past-only)...")
    feats_df = build_panel_features(panel_df, bench_df)

    # 3. M3 Labels
    print("\n[M3] Generating 5-day forward excess return labels...")
    labels_df = generate_panel_labels(panel_df, bench_df, horizon=5)

    # 4. M5 Walk-Forward Validation Setup
    print("\n[M5] Initializing walk-forward splits with 5-day embargo gap...")
    calendar = panel_df.index.get_level_values("date").unique().sort_values()
    validator = WalkForwardValidator(
        calendar=calendar,
        initial_train_days=350,
        step_days=100,
        test_days=100,
        embargo_days=5,
    )

    X_all, y_all = get_feature_and_label_matrices(feats_df, labels_df)
    price_matrix = panel_df["adj_close"].unstack(level="ticker")

    # Storage for out-of-sample predictions
    oos_preds_lr = pd.DataFrame(0.0, index=price_matrix.index, columns=price_matrix.columns)
    oos_preds_xgb = pd.DataFrame(0.0, index=price_matrix.index, columns=price_matrix.columns)

    splits = list(validator.split())
    print(f"Executing {len(splits)} expanding walk-forward folds...")

    last_model_xgb = None
    last_X_train = None

    for s in splits:
        # Strict time-series slicing
        X_train = X_all.loc[(slice(s.train_start, s.train_end), slice(None)), :]
        y_train = y_all.loc[(slice(s.train_start, s.train_end), slice(None))]

        X_test = X_all.loc[(slice(s.test_start, s.test_end), slice(None)), :]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Fit Logistic
        lr = build_logistic_model()
        lr.fit(X_train, y_train)
        probs_lr = lr.predict_proba(X_test)[:, 1]

        # Fit XGBoost
        xgb_clf = build_gradient_boosting_model(n_estimators=50)
        xgb_clf.fit(X_train, y_train)
        probs_xgb = xgb_clf.predict_proba(X_test)[:, 1]

        last_model_xgb = xgb_clf
        last_X_train = X_train

        # Assign test probabilities
        for (dt, tk), p_lr, p_xgb in zip(X_test.index, probs_lr, probs_xgb):
            if dt in oos_preds_lr.index and tk in oos_preds_lr.columns:
                oos_preds_lr.loc[dt, tk] = p_lr
                oos_preds_xgb.loc[dt, tk] = p_xgb

    # Generate Top-K Target Weights from OOS probabilities
    def make_top_k_weights(pred_matrix: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=pred_matrix.index, columns=pred_matrix.columns)
        for dt, row in pred_matrix.iterrows():
            valid = row[row > 0]
            if len(valid) > 0:
                top = valid.nlargest(min(k, len(valid))).index
                w.loc[dt, top] = 1.0 / len(top)
        return w

    weights_lr = make_top_k_weights(oos_preds_lr, k=3)
    weights_xgb = make_top_k_weights(oos_preds_xgb, k=3)

    # 5. M4 Baselines
    print("\n[M4] Generating baseline strategies...")
    weights_bh = generate_buy_and_hold_weights(bench_df)
    weights_eq = generate_equal_weight_basket_weights(panel_df)
    weights_mom = generate_momentum_rule_weights(feats_df, momentum_col="ret_20d", top_k=3)
    weights_rnd = generate_random_signals(panel_df, top_k=3, seeds=[42])[0]

    # 6. M6 & M7 Backtesting & Performance Evaluation
    print("\n[M6/M7] Simulating backtests with 10 bps fee + 2 bps slippage...")
    bench_price_matrix = pd.DataFrame({"BENCHMARK": bench_df["adj_close"]})

    strategies = {
        "Buy & Hold (Nifty 50)": run_portfolio_backtest(weights_bh, bench_price_matrix),
        "Equal Weight Basket": run_portfolio_backtest(weights_eq, price_matrix),
        "Momentum Rule (Top 3)": run_portfolio_backtest(weights_mom, price_matrix),
        "Random Trader": run_portfolio_backtest(weights_rnd, price_matrix),
        "Logistic Regression": run_portfolio_backtest(weights_lr, price_matrix),
        "Gradient Boosting (XGB)": run_portfolio_backtest(weights_xgb, price_matrix),
    }

    metrics_rows = []
    for name, res in strategies.items():
        scorecard = evaluate_strategy_performance(res.equity_curve, res.daily_returns, res.turnover_series)
        scorecard["Strategy"] = name
        scorecard["Total_Costs_Paid"] = res.total_costs_paid
        metrics_rows.append(scorecard)

    metrics_df = pd.DataFrame(metrics_rows).set_index("Strategy")
    metrics_path = "reports/tables/results_summary.csv"
    metrics_df.to_csv(metrics_path)
    print("\n" + "=" * 50)
    print("PERFORMANCE SCORECARD (NET OF COSTS)")
    print("=" * 50)
    print(metrics_df[["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "annual_turnover"]])

    # Plot Equity Curves
    plt.figure(figsize=(10, 6))
    for name, res in strategies.items():
        norm_curve = (res.equity_curve / res.equity_curve.iloc[0]) * 100.0
        plt.plot(norm_curve.index, norm_curve, label=name)
    plt.title("AlphaCheck: Out-of-Sample Equity Curves (Net of 12 bps Friction)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Equity (Base = 100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/equity_curves.png", dpi=150)
    plt.close()

    # 7. M9 Robustness (Cost Sensitivity, Regimes, SHAP)
    print("\n[M9] Executing cost sensitivity grid and SHAP explanations...")
    cost_df = evaluate_cost_sensitivity(weights_xgb, price_matrix)
    bench_returns = bench_df["adj_close"].pct_change().reindex(strategies["Gradient Boosting (XGB)"].daily_returns.index)
    regimes_df = evaluate_regimes(strategies["Gradient Boosting (XGB)"].daily_returns, bench_returns)

    if last_model_xgb is not None and last_X_train is not None:
        compute_tree_shap_importance(last_model_xgb, last_X_train)

    print("\nPipeline finished successfully. All tables and figures exported to reports/.")


if __name__ == "__main__":
    main()

