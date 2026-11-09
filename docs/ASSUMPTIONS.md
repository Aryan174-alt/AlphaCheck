# Assumptions Log: AlphaCheck

This document logs all explicit engineering assumptions made during the design and execution of AlphaCheck to ensure full research transparency.

## 1. Data & Market Assumptions
- **Universe Definition**: We use the Nifty 50 constituent list as of the current index composition. Survivorship bias is acknowledged: delisted or historically replaced firms prior to the current index reconstitution are not retroactively re-introduced due to the lack of free historical point-in-time index composition datasets.
- **Price Adjustment**: We download split- and dividend-adjusted OHLCV daily data using `yfinance` with ticker suffix `.NS` (e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`) and `^NSEI` for Nifty 50 benchmark.
- **Calendar & Time Alignment**: Trading occurs on official NSE trading days. Stocks missing trading records on a given date due to corporate suspensions are forward-filled for positions or omitted from cross-sectional rankings for that timestamp.

## 2. Leakage & Embargo Assumptions
- **Label Horizon**: Label is strictly 5 trading days forward:
  $$y_{i,t} = \mathbf{1}\left( \frac{P_{i, t+5}}{P_{i, t}} - 1 > \frac{I_{t+5}}{I_t} - 1 \right)$$
- **Embargo Window**: Because labels extend 5 trading days into the future, any training set ending at index $t_{\text{end}}$ has labels computed up to $t_{\text{end}} + 5$. Therefore, test evaluation begins strictly at $t_{\text{end}} + 6$ (a minimum 5-trading-day embargo gap) so that training target returns never overlap test feature dates.
- **Lagged Execution**: Signals calculated using data up to close of date $t$ are assumed executed at date $t+1$ (open or close) to avoid same-day look-ahead execution bias.

## 3. Transaction Costs & Friction Assumptions
- **Brokerage & Taxes (STT, Exchange turnover fees, GST)**: Modelled as a default flat friction of 0.10% (10 bps) per one-way trade (buy and sell).
- **Slippage**: Assumed flat 0.02% (2 bps) slippage per trade for large-cap Nifty 50 liquid stocks.
- **Sensitivity Matrix**: Friction is tested across 0 bps (frictionless baseline), 5 bps (discount broker minimal), 10 bps (realistic institutional/retail average), and 30 bps (high friction/illiquid stress).

## 4. NLP & AI Lab Assumptions
- **Financial PhraseBank**: The sentence-level dataset is split sentence-by-sentence into 70% Train, 15% Validation, and 15% Test with a fixed random seed (42). It is solely used for sentiment model evaluation and fine-tuning; it is never mixed with market timeline splits.
- **News Timestamps**: Free public historical Indian financial headlines (from public archives / GDELT / Kaggle financial news datasets) are strictly timestamp-aligned. Only headlines published before 09:00 AM IST of day $t$ (or prior calendar day close) are eligible for day $t$ features.
- **Local LLM (Ollama)**: Optional module A4 checks if a local Ollama daemon is running with `llama3.2:3b` or `qwen2.5:3b`. If Ollama is not present or offline, A4 is gracefully skipped with mock fallback flags, ensuring the core pipeline is 100% self-contained on CPU.

