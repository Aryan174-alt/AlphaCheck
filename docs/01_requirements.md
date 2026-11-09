# 01 Requirements: AlphaCheck

## 1. Problem Statement
Most retail and academic stock prediction projects report deceptively high accuracy (80-90%+) by committing silent data leakage: using centered rolling windows, unshifted forward returns, random time-series splits, ignoring corporate action adjustments, and omitting realistic transaction costs. When deployed, these strategies fail catastrophically.

**AlphaCheck** is an evaluation-first, leakage-safe machine learning and deep NLP research framework designed to honestly test whether ML models and transformer-based text/sequence models can beat simple market baselines on India's benchmark **Nifty 50** universe, net of realistic transaction costs and slippage.

---

## 2. Research Questions (RQs)
- **RQ1**: Do classical ML models (Logistic Regression, Gradient Boosting) beat a buy-and-hold index, an equal-weight basket, and simple momentum rules after transaction costs and slippage?
- **RQ2**: Does adding FinBERT news-sentiment features provide statistically significant predictive alpha over price-only features in identical walk-forward splits (ablation test)?
- **RQ3**: How robust are results across varied transaction cost assumptions (0 bps, 5 bps, 10 bps, 30 bps), distinct market regimes (bull, bear, sideways), and sub-periods?
- **RQ4**: In a sentiment bake-off, does a domain fine-tuned transformer (DistilBERT on Financial PhraseBank) outperform zero-shot FinBERT on text metrics (accuracy, macro-F1), and does higher text classification accuracy actually translate into improved downstream trading signal?
- **RQ5**: Do dense headline representations (Sentence-Transformers MiniLM embeddings reduced via PCA) or structured LLM event tags add predictive power beyond scalar polarity sentiment scores?
- **RQ6**: Does a deep sequence model (small Transformer encoder or Temporal Convolutional Network - TCN) on rolling historical feature windows outperform gradient boosted decision trees on identical expanding walk-forward splits?

---

## 3. Prediction Target & Universe
- **Universe**: Nifty 50 stocks (NSE India, 50 liquid large-cap equities) plus the benchmark index (`^NSEI`).
- **Time Horizon**: ~8 to 10 years of daily historical data.
- **Target Formulation**:
  $$y_{i,t} = \mathbf{1}\left( R_{i, t \to t+5} > R_{\text{Nifty}, t \to t+5} \right)$$
  Where $R_{i, t \to t+5}$ is the 5-trading-day forward cumulative return of stock $i$, and $R_{\text{Nifty}, t \to t+5}$ is the benchmark forward return. Binary classification: 1 if the stock outperforms Nifty 50 over the next 5 days, 0 otherwise.
- **Look-Ahead Truncation**: Rows without a full 5-day future window are strictly dropped during target generation.

---

## 4. Evaluation Protocol & Rigor Guardrails
1. **Walk-Forward Validation (Expanding Window)**:
   - Train on $[0, T_{\text{train}}]$.
   - **Purge / Embargo Gap**: Exactly 5 trading days between the end of train and the start of test, ensuring no overlap of 5-day forward target horizons.
   - Test on $[T_{\text{train}} + 5\text{ days}, T_{\text{test}}]$.
   - Hyperparameter tuning and feature scalers fitted *strictly* inside the training window.
2. **Backtest Accounting**:
   - Starting Capital: ₹1,000,000 (10 Lakhs INR).
   - Friction: 0.10% (10 bps) per side default transaction cost + 0.02% slippage.
   - Sizing: Top-K quintile / equal-weight long positions, zero look-ahead execution at $t+1$ open/close.
   - Exact reconciliation of cash, marked-to-market positions, realized costs, and total portfolio equity.
3. **Statistical Confidence**:
   - Annualized Return, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio, Hit Rate, Turnover.
   - Stationary Bootstrap confidence intervals (95% CI) computed for Sharpe and excess returns.

---

## 5. Success Criteria & Non-Goals
- **Success Criteria**:
  - Zero data leakage verified by automated test suites (`test_leakage.py`).
  - Completely reproducible via one command (`python -m alphacheck.run_all`).
  - Honest reporting: reporting underperformance against benchmark after transaction costs is considered a valid, rigorous result.
  - All AI Lab modules benchmarked directly against simpler baselines.
- **Non-Goals**:
  - Fabricating unrealistic Sharpe ratios or cherry-picking test periods.
  - High-frequency / intraday order-book trading.
  - Paid APIs, proprietary cloud infrastructure, or multi-GPU requirements.
  - Complex live order routing / broker API integrations.

