# Limitations & Integrity Declarations: AlphaCheck

Honest empirical science requires declaring exact scope, biases, and data gaps.

---

## 1. Survivorship Bias
- **Universe Definition**: We track the current (as of 2026) constituent list of the Nifty 50 index across the historical lookback.
- **Impact**: Companies that ran into distress, went bankrupt, or were removed from the Nifty 50 index during bi-annual index reconstitutions between 2016 and 2026 are not present in this dataset. This introduces positive survivorship selection bias.
- **Disclosure**: Results represent stock selection *within surviving liquid large-cap leaders* relative to the benchmark index (`^NSEI`).

## 2. Evaluation Window & Date Coverage
- **Historical Ingestion Window**: 2016-01-01 to 2026-01-01 (10 years of daily data).
- **Out-of-Sample Evaluation Protocol**: To prevent look-ahead bias and guarantee identical market conditions across all comparisons, **all strategies (both active ML models and passive baselines)** are evaluated strictly over the exact same out-of-sample window, beginning at `first_oos_date` (post-initial training and 5-day embargo gap) through the end of the timeline.

## 3. Financial Headline Data Gaps (RQ2 and RQ5 Status)
- **News Archive Gaps**: Free, cleanly timestamp-aligned historical news archives for domestic Indian equities are not available across the full 2016–2026 timeframe without un-timestamped aggregations.
- **Integrity Rule**: Rather than fabricating news sentiments or evaluating on an un-timestamped proxy, **RQ2 (FinBERT predictive alpha)** and **RQ5 (MiniLM headline embeddings & LLM event tags)** are explicitly marked as **NOT TESTED** in `docs/RESULTS.md`. No claims are made for experiments that could not be run with complete timestamp integrity.

## 4. Execution Frictions & Capacity
- **Transaction Costs**: Fixed at 10 bps (0.10%) one-way broker fees, STT, and exchange turnover charges, plus 2 bps (0.02%) slippage.
- **Turnover Drag**: Active models rebalancing frequently incur substantial cumulative drag. Weekly rebalancing limits churn but does not eliminate transaction cost sensitivity.
- **Cash Equity Long-Only**: Indian cash equity regulations prohibit overnight short positions without derivatives; all strategies operate strictly long-only or holding cash.
