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

---

## M4: Baselines (`baselines.py`)
- **What it does**: Establishes 4 rigorous reference benchmarks: passive buy-and-hold Nifty 50, equal-weight basket, cross-sectional top-K momentum rule, and Monte Carlo random trader (multi-seed average).
- **Why designed that way**: Complex ML and transformer models are pointless unless they can demonstrably beat simple, zero-parameter rules like cross-sectional momentum and market beta.
- **Mistakes avoided**: Never comparing ML solely against a risk-free rate; never omitting a random signal baseline to test chance distribution.
- **Interview Questions**:
  1. *Q: Why is an Equal-Weight benchmark often harder to beat than a market-cap index like Nifty 50?*
     *A: Equal-weighting assigns higher exposure to smaller, higher-beta constituents and rebalances by selling winners and buying losers (mean reversion bonus).*
  2. *Q: What purpose does the multi-seed random signal serve?*
     *A: It reveals the empirical distribution of random stock picking under the identical backtest engine, proving whether model alpha is statistically distinct from chance.*
  3. *Q: Why test cross-sectional momentum as a primary heuristic baseline?*
     *A: Cross-sectional momentum is one of the most persistent empirical market anomalies. Any ML model claiming predictive power must beat this simple sort.*

---

## M5: Walk-Forward Validation & ML Models (`validation.py`, `models.py`)
- **What it does**: Provides expanding-window walk-forward validation with an enforced purge/embargo gap equal to the label horizon (5 trading days). Encapsulates Logistic Regression (with StandardScaler) and Gradient Boosted Trees (XGBoost) with hyperparameter constraints.
- **Why designed that way**: Time-series cross-validation cannot shuffle data. Walk-forward mimics real out-of-sample trading where only the past is known. The embargo gap prevents the 5-day future return in the final training rows from leaking into the initial test rows.
- **Mistakes avoided**: Never tuning hyperparameters or fitting feature scalers across the full dataset; never using standard K-fold cross-validation.
- **Interview Questions**:
  1. *Q: What is the purpose of an embargo gap in walk-forward validation?*
     *A: When labels span $H$ future days, training labels near the split boundary use price movements that occur during the test period. The embargo gap purges $H$ days between train and test to guarantee zero label overlap.*
  2. *Q: Why expanding window instead of rolling window?*
     *A: Expanding window utilizes all historical regime information as time progresses, mirroring an institutional fund that accumulates more data each year.*
  3. *Q: Why prefer conservative tree depth (e.g. max_depth=3) in financial tabular ML?*
     *A: Financial data has extremely low signal-to-noise ratio. Deep trees rapidly memorize idiosyncratic noise, leading to severe test-set degradation.*

---

## M6: Backtest Simulator (`backtest.py`)
- **What it does**: Executes a discrete-event paper-trading portfolio simulator tracking exact cash, share positions, market-to-market valuations, turnover, and transaction friction (default 10 bps fee + 2 bps slippage). Enforces 1-day execution lag to eliminate look-ahead.
- **Why designed that way**: Vectorized backtests often hallucinate instant execution at the closing price used to form the signal, and ignore rebalancing costs. Our engine explicitly accounts for friction on every rupee traded.
- **Mistakes avoided**: Never trading at day $t$ close using signals computed on day $t$ close; never letting cash become silently negative; incorporating a deadband threshold to prevent micro-churn.
- **Interview Questions**:
  1. *Q: What is execution lag and why is it mandatory?*
     *A: If a model computes predictions after market close on Tuesday, trades can only be physically filled on Wednesday. Simulating Tuesday close execution is a fatal look-ahead flaw.*
  2. *Q: How does turnover affect realistic strategy performance?*
     *A: A strategy generating 500% annualized turnover pays $500\% \times 0.24\% = 1.20\%$ in friction annually. At higher frequency, turnover consumes 100% of gross alpha.*
  3. *Q: How do you verify portfolio accounting integrity?*
     *A: In `test_backtest.py`, we verify that on an asset with zero price change, ending equity matches initial capital minus the exact sum of broker fees and slippage.*

---

## M7: Performance Metrics & Bootstrap CIs (`metrics.py`)
- **What it does**: Calculates annualized return, annualized volatility, Sharpe ratio (vs 6% G-Sec risk-free rate), Sortino ratio, max drawdown, Calmar ratio, win rate, annual turnover, and stationary block bootstrap 95% confidence intervals for Sharpe.
- **Why designed that way**: Point-estimate Sharpe ratios are deeply noisy and easily overfitted. Bootstrap intervals reveal whether a Sharpe of 0.8 is statistically distinct from zero or negative return distributions.
- **Mistakes avoided**: Never computing annualized volatility by multiplying by 252 instead of $\sqrt{252}$; never reporting Sharpe without risk-free rate adjustment; never using IID bootstrap on serially correlated returns.
- **Interview Questions**:
  1. *Q: Why use a block bootstrap instead of standard IID bootstrap for financial returns?*
     *A: Financial returns exhibit volatility clustering and autocorrelation. Standard IID bootstrap destroys this time dependence; block bootstrap preserves local serial correlation.*
  2. *Q: Why is Sortino ratio often preferred over Sharpe ratio?*
     *A: Sharpe penalizes upside volatility equally with downside crashes. Sortino penalizes downside volatility only, rewarding large positive outliers.*
  3. *Q: Why is reporting confidence intervals for Sharpe essential in quantitative research?*
     *A: A sample Sharpe of 0.5 with a 95% CI of [-0.2, 1.2] across 200 trading days is statistically indistinguishable from noise. Reporting the CI prevents deceptive claims.*

---

## M8: NLP & Timestamp Alignment (`sentiment.py`)
- **What it does**: Parses financial news headlines, scores sentiment via FinBERT zero-shot ($P(\text{pos}) - P(\text{neg})$), and strictly maps news items to trading sessions using a 09:00 AM IST cutoff rule.
- **Why designed that way**: News breaking during the trading day (e.g. 11:30 AM) cannot be assumed traded at 09:15 AM open of that same day. Intraday news is strictly shifted to the next market open.
- **Mistakes avoided**: Never joining headlines on same-day calendar dates regardless of timestamps; never using post-market earnings reports for same-day predictions.
- **Interview Questions**:
  1. *Q: What is timestamp leakage in financial NLP and how did you prevent it?*
     *A: Using news published at 2:00 PM to predict the day's total return. We enforce a strict cutoff: news published after 09:00 AM IST is deferred to the subsequent trading day.*
  2. *Q: Why use $P(\text{positive}) - P(\text{negative})$ as a single scalar polarity metric?*
     *A: It spans $[-1.0, 1.0]$, inherently downweights ambiguous/neutral headlines (where probabilities are concentrated on neutral), and serves as an intuitive feature for linear and tree models.*
  3. *Q: How do you handle days where a company has no news?*
     *A: Impute sentiment with 0.0 (neutral), maintain an explicit binary feature `has_news=0`, and record `sentiment_count=0`.*







