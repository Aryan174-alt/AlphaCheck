# AlphaCheck

> **Leakage-safe ML + transformer research framework testing whether AI models beat simple baselines on Nifty 50 stocks, net of costs**
>
> **Status**: Completed / Portfolio-Grade Evaluation

---

## 1. Problem & Core Thesis
Most retail and student quantitative finance repositories report unrealistic 80-90% predictive accuracies due to **data leakage** (e.g. forward-looking technical indicators, un-embargoed train/test boundaries, unshifted targets) and omitting transaction friction.

**AlphaCheck** is built with evaluation rigor as the primary product:
1. **Zero-Leakage Guardrails**: Features at time $t$ never see information after $t$; labels strictly require full forward horizons; walk-forward splits enforce a 5-day purge/embargo gap.
2. **Identical Out-of-Sample Window**: Active ML models and passive baselines are compared strictly across the identical test timeline.
3. **Realistic Frictions**: Discrete portfolio backtest accounting with 10 bps one-way fees + 2 bps slippage, weekly rebalancing, and 1-day execution lag.
4. **AI Lab**: Controlled benchmarks comparing zero-shot FinBERT, fine-tuned domain transformers on Financial PhraseBank, and PyTorch Transformer sequence models.

---

## 2. Empirical Results (Production: 50 Tickers, Net of 12 bps Friction)

All figures generated directly by `python -m alphacheck.run_all` from `reports/tables/results_summary.csv` with zero fabrication:

| Strategy | CAGR (%) | Annual Vol (%) | Sharpe (vs 6% RF) | Max DD (%) | Annual Turnover | 95% Bootstrap CI | Total Fees (INR) |
|---|---|---|---|---|---|---|---|
| **Equal Weight Basket** | **+11.24%** | 13.70% | **+0.420** | 11.82% | 0.74x | [-0.468, 2.323] | ₹1,736 |
| **Gradient Boosting (XGB)** | **+11.11%** | 14.84% | **+0.391** | 11.71% | 35.78x | [-0.520, 2.243] | ₹88,307 |
| **Logistic Regression** | **+10.28%** | 15.14% | **+0.337** | 12.68% | 24.41x | [-0.662, 2.153] | ₹60,898 |
| **Buy & Hold (Nifty 50)** | **+9.39%** | 14.09% | **+0.294** | 16.49% | 0.52x | [-0.676, 2.234] | ₹1,200 |
| **Transformer Sequence (A3)** | **+6.58%** | 14.80% | **+0.111** | 12.75% | 45.80x | [-0.878, 1.875] | ₹106,791 |
| **Momentum Rule** | **+4.47%** | 14.30% | **-0.030** | 12.81% | 26.53x | [-0.981, 1.940] | ₹61,459 |
| **Random Trader** | **+0.01%** | 15.14% | **-0.308** | 16.49% | 47.64x | [-1.308, 1.467] | ₹102,132 |

---

## 3. A1 Sentiment Bake-off Results (Held-Out Test Split)

| Model | Accuracy (%) | Macro-F1 | Test Samples | Confusion Matrix |
|---|---|---|---|---|
| **FinBERT (ProsusAI/finbert) Zero-Shot** | 82.67% | 0.817 | 75 | `[[13, 1, 0], [5, 31, 5], [2, 0, 18]]` |
| **DistilRoBERTa / DistilBERT Financial Fine-Tuned** | 90.67% | 0.893 | 75 | `[[11, 2, 1], [1, 37, 3], [0, 0, 20]]` |

---

## 4. How to Reproduce Everything

```powershell
# 1. Clone repository
git clone https://github.com/Aryan174-alt/AlphaCheck.git
cd AlphaCheck

# 2. Create environment & install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Run all tests
set PYTHONPATH=src
pytest

# 4. Run entire end-to-end research pipeline
# Fast dev mode (10 tickers):
python -m alphacheck.run_all --quick

# Full production run (50 tickers):
python -m alphacheck.run_all

# 5. Launch interactive Streamlit dashboard
streamlit run src/alphacheck/dashboard.py
```

---

## 5. Limitations & RQ Status
See [`docs/LIMITATIONS.md`](file:///c:/project%20new/alphacheck/docs/LIMITATIONS.md) and [`docs/RESULTS.md`](file:///c:/project%20new/alphacheck/docs/RESULTS.md) for full disclosure of survivorship bias, tested research questions (RQ1, RQ3, RQ4, RQ6), and unverified questions (RQ2 and RQ5 marked **NOT TESTED** due to historical timestamped news archive gaps).
