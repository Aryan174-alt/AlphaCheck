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
     *A: A 2-for-1 stock split cuts the nominal price by 50% overnight. Without split adjustments, an unadjusted return calculation registers a false -50% crash, completely skewing loss functions.*
