"""M7 Performance Metrics & Bootstrap Confidence Intervals.

Computes:
- Annualized Return, Annualized Volatility
- Sharpe Ratio, Sortino Ratio
- Maximum Drawdown, Calmar Ratio
- Hit Rate (fraction of positive excess return periods)
- Average Annual Turnover
- Stationary Bootstrap 95% Confidence Intervals for Sharpe and Excess Return
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def compute_annualized_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR computed from geometric compounding."""
    clean = daily_returns.dropna()
    if len(clean) == 0:
        return 0.0
    cum_ret = (1.0 + clean).prod()
    n_years = len(clean) / periods_per_year
    if n_years <= 0 or cum_ret <= 0:
        return 0.0
    return float(cum_ret ** (1.0 / n_years) - 1.0)


def compute_annualized_volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sample annualized standard deviation."""
    clean = daily_returns.dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std() * np.sqrt(periods_per_year))


def compute_sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.06,  # 6% annual risk-free rate for Indian G-Sec
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio over risk-free rate."""
    clean = daily_returns.dropna()
    if len(clean) < 2:
        return 0.0
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = clean - rf_daily
    std = clean.std()
    if std <= 1e-12:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def compute_sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.06,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio penalized only by downside deviation."""
    clean = daily_returns.dropna()
    if len(clean) < 2:
        return 0.0
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = clean - rf_daily
    downside = np.minimum(excess, 0.0)
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std <= 1e-12:
        return 0.0
    return float((excess.mean() / downside_std) * np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough decline as a positive percentage (0.0 to 1.0)."""
    clean = equity_curve.dropna()
    if len(clean) < 2:
        return 0.0
    running_max = clean.cummax()
    drawdown = (clean - running_max) / running_max
    return float(-drawdown.min())


def compute_bootstrap_ci(
    daily_returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """Stationary / block bootstrap for Sharpe ratio 95% confidence interval."""
    clean = daily_returns.dropna().values
    n = len(clean)
    if n < 10:
        return (0.0, 0.0)

    rng = np.random.default_rng(random_seed)
    sharpes = []
    n_blocks = max(1, n // block_size)

    for _ in range(n_bootstrap):
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled_indices = np.concatenate([np.arange(idx, idx + block_size) for idx in start_indices])
        sample = clean[sampled_indices]

        mean = np.mean(sample)
        std = np.std(sample, ddof=1)
        if std > 1e-12:
            sr = (mean / std) * np.sqrt(252)
            sharpes.append(sr)

    if len(sharpes) == 0:
        return (0.0, 0.0)

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(sharpes, 100.0 * alpha))
    upper = float(np.percentile(sharpes, 100.0 * (1.0 - alpha)))
    return lower, upper


def evaluate_strategy_performance(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    turnover_series: Optional[pd.Series] = None,
    risk_free_rate: float = 0.06,
) -> Dict[str, float]:
    """Compute comprehensive performance scorecard."""
    ann_ret = compute_annualized_return(daily_returns)
    ann_vol = compute_annualized_volatility(daily_returns)
    sharpe = compute_sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate)
    sortino = compute_sortino_ratio(daily_returns, risk_free_rate=risk_free_rate)
    mdd = compute_max_drawdown(equity_curve)
    calmar = (ann_ret / mdd) if mdd > 1e-6 else 0.0

    hit_rate = float((daily_returns > 0).mean()) if len(daily_returns) > 0 else 0.0
    avg_turnover = float(turnover_series.mean() * 252) if turnover_series is not None else 0.0
    lower_ci, upper_ci = compute_bootstrap_ci(daily_returns)

    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "hit_rate": hit_rate,
        "annual_turnover": avg_turnover,
        "sharpe_ci_lower_95": lower_ci,
        "sharpe_ci_upper_95": upper_ci,
    }
