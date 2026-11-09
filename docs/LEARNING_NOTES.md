# Learning Notes: AlphaCheck

This document explains every module in simple terms, detailing why it was designed that way, common pitfalls avoided, and 3 interview questions with concise answers.

---

## M1: Data Pipeline (`data.py`)
- **What it does**: Ingests Nifty 50 and benchmark equity prices via `yfinance`, enforces consistent lowercase column naming, aligns all stock timestamps against the official exchange trading calendar, caches raw/processed frames to parquet, and outputs a data quality audit.
- **Why designed that way**: Local parquet caching prevents rate-limiting and makes the entire research pipeline offline-reproducible. Aligning against benchmark dates guarantees no gaps or look-ahead alignment issues across the 50 stocks.
- **Mistakes avoided**: Never downloading unadjusted prices without split adjustments; never allowing forward-fill across wide missing gaps without marking zero volume; never refetching data repeatedly over the internet.
- **Interview Questions**:
  1. *Q: How do you handle survivorship bias when fetching current Nifty 50 tickers?*
     *A: Acknowledge that using current constituents selects winners; note it as a declared limitation and benchmark against the exact same surviving basket or relative index return.*
  2. *Q: Why use Parquet instead of CSV for financial time series caching?*
     *A: Parquet preserves exact data types (datetime index, float64 precision), supports Snappy column compression, and loads 10-50x faster than parsing CSV strings.*
  3. *Q: How do corporate actions (dividends, splits) corrupt machine learning models if unadjusted?*
     *A: A2: A 2-for-1 stock split cuts the nominal price by 50% overnight. Without split adjustments, an unadjusted return calculation registers a false -50% crash, completely skewing loss functions.*

---

## M2: Feature Engineering (`features.py`)
- **What it does**: Computes strictly backward-looking multi-horizon returns (1/5/10/20/60d), rolling 20d volatility, RSI, MACD (12, 26, 9), volume ratios, and 60d rolling CAPM beta against Nifty 50.
- **Why designed that way**: Guarantees zero forward-looking bias. Rolling windows never use `center=True` and every statistical transformation uses past observations $[t-W, t]$ only.
- **Mistakes avoided**: Never computing indicators on unshifted future bars; never scaling features cross-sectionally using global statistics across all time.
- **Interview Questions**:
  1. *Q: How did you verify that features contain zero look-ahead leakage?*
     *A: Implemented an adversarial test (`test_leakage.py`): randomly altered all price/volume series from $t+1$ to $T$ by 500% and verified that features calculated at/before $t$ had an absolute change of zero.*
  2. *Q: Why include both short-term (5d) and intermediate (60d) momentum alongside volatility?*
     *A: Short momentum captures short-term mean-reversion/sentiment bursts, while 60d momentum captures institutional trend following, and volatility normalizes risk exposure.*
  3. *Q: Why is rolling CAPM beta against Nifty 50 useful as a feature?*
     *A: It measures systemic market sensitivity. During market drawdowns or rallies, high-beta stocks react non-linearly compared to low-beta defensives.*

