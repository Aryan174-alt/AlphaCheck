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

---

## M3: Label Generation (`labels.py`)
- **What it does**: Constructs 5-trading-day forward excess return binary labels: $y_{i,t} = \mathbf{1}(R_{i, t \to t+5} > R_{\text{bench}, t \to t+5})$, and strictly drops rows without the complete 5-day future horizon.
- **Why designed that way**: Classifying relative outperformance against the index removes macro beta (market-wide drift) and isolates idiosyncratic stock-picking ability.
- **Mistakes avoided**: Never keeping unfulfilled forward returns at the tail end of the dataset; never forward-filling missing labels.
- **Interview Questions**:
  1. *Q: Why formulate the target as excess return over Nifty 50 instead of raw direction (+/-)?*
     *A: Raw directional prediction mostly learns whether the broader market went up or down. Predicting excess return forces models to discover idiosyncratic stock-picking alpha.*
  2. *Q: Why 5 trading days horizon instead of 1-day?*
     *A: 1-day predictions are heavily dominated by bid-ask bounce and microstructure noise. A 5-day horizon provides enough time for fundamental/sentiment information to be absorbed.*
  3. *Q: What is the critical guardrail when generating forward labels?*
     *A: Dropping the last $H$ rows (where future data is not yet observed) and strictly embargoing the last $H$ training days from the test set.*


