"""M9 Robustness: cost sensitivity grid, sub-period stress testing, and SHAP feature importance."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from alphacheck.backtest import run_portfolio_backtest
from alphacheck.metrics import evaluate_strategy_performance


def evaluate_cost_sensitivity(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    fee_levels: List[float] = [0.0, 0.0005, 0.0010, 0.0030],  # 0, 5, 10, 30 bps
    slippage: float = 0.0002,
    output_path: str = "reports/tables/cost_sensitivity.csv",
) -> pd.DataFrame:
    """Simulate strategy across varied transaction cost levels to measure friction degradation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []

    for fee in fee_levels:
        res = run_portfolio_backtest(
            target_weights=weights,
            price_matrix=prices,
            fee_rate=fee,
            slippage_rate=slippage,
        )
        perf = evaluate_strategy_performance(res.equity_curve, res.daily_returns, res.turnover_series)
        rows.append({
            "fee_bps": int(fee * 10000),
            "slippage_bps": int(slippage * 10000),
            "total_friction_bps": int((fee + slippage) * 10000),
            "cagr": perf["annualized_return"],
            "volatility": perf["annualized_volatility"],
            "sharpe": perf["sharpe_ratio"],
            "sortino": perf["sortino_ratio"],
            "max_drawdown": perf["max_drawdown"],
            "total_costs_paid": res.total_costs_paid,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def evaluate_regimes(
    daily_returns: pd.Series,
    bench_returns: pd.Series,
    output_path: str = "reports/tables/regime_breakdown.csv",
) -> pd.DataFrame:
    """Analyze strategy performance conditioned on market regime (Bull, Bear, Sideways)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aligned = pd.DataFrame({"strat": daily_returns, "bench": bench_returns}).dropna()

    # Define regimes by benchmark 60d rolling return
    bench_60d = aligned["bench"].rolling(60).sum()
    bull_mask = bench_60d > 0.05
    bear_mask = bench_60d < -0.05
    sideways_mask = (~bull_mask) & (~bear_mask)

    regimes = {"Bull (>+5%)": bull_mask, "Bear (<-5%)": bear_mask, "Sideways (±5%)": sideways_mask}
    results = []

    for name, mask in regimes.items():
        sub = aligned.loc[mask]
        if len(sub) < 10:
            continue
        cagr_strat = float(((1.0 + sub["strat"]).prod() ** (252.0 / len(sub))) - 1.0)
        cagr_bench = float(((1.0 + sub["bench"]).prod() ** (252.0 / len(sub))) - 1.0)
        sharpe_strat = float((sub["strat"].mean() / (sub["strat"].std() + 1e-12)) * np.sqrt(252))

        results.append({
            "regime": name,
            "trading_days": len(sub),
            "strategy_cagr": cagr_strat,
            "bench_cagr": cagr_bench,
            "strategy_sharpe": sharpe_strat,
            "excess_return": cagr_strat - cagr_bench,
        })

    out = pd.DataFrame(results)
    out.to_csv(output_path, index=False)
    return out


def compute_tree_shap_importance(
    model,
    X_train: pd.DataFrame,
    max_samples: int = 200,
    output_fig: str = "reports/figures/shap_summary.png",
) -> pd.DataFrame:
    """Compute Tree SHAP feature importances and export summary bar chart."""
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    sample = X_train.sample(min(len(X_train), max_samples), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # In binary classification, take positive class if list
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = shap_values

    mean_abs_shap = np.abs(vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": sample.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"][::-1], importance_df["mean_abs_shap"][::-1], color="#2b5c8f")
    plt.title("Tree SHAP Feature Importance (AlphaCheck M9)")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150)
    plt.close()

    return importance_df

