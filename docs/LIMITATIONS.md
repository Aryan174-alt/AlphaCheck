# Limitations & Survivorship Bias: AlphaCheck

Honest science requires transparent disclosure of limitations. The following known limitations apply to AlphaCheck:

## 1. Survivorship Bias
- **Description**: We track the current (as-of 2026) constituent list of the Nifty 50 index over the entire 8-10 year historical lookback.
- **Impact**: Companies that deteriorated, went bankrupt, or were removed from the Nifty 50 index over the last decade (e.g. historical constituents removed during bi-annual rebalancing) are not present in this backtest. This introduces a positive selection (survivorship) bias into the equity universe.
- **Mitigation**: We document this explicitly. The research focus is on relative outperformance against the contemporary benchmark (`^NSEI`), which experienced the same historical index performance.

## 2. Market Impact & Capacity Constraints
- **Description**: The backtest uses an execution model assuming all rebalance orders are filled at the next day's open or close with a fixed 0.10% friction + 0.02% slippage.
- **Impact**: In reality, large order sizes relative to Average Daily Volume (ADV) cause non-linear market impact, adverse selection, and partial fills.
- **Mitigation**: The Nifty 50 universe is the most liquid equity universe in Indian markets, which minimizes slippage compared to mid- or small-caps. We test cost sensitivity up to 30 bps per trade.

## 3. Financial Headline Data Availability
- **Description**: Free, public, cleanly timestamp-aligned historical news archives for Indian domestic equities are sparse prior to 2020.
- **Impact**: While price series span 8-10 years, historical Indian news coverage may be patchy in earlier years.
- **Mitigation**: Sentiment and embedding ablations (M8, A1, A2, A4) are conducted across the largest contiguous window with verified high-density news coverage, rather than interpolating or forward-filling missing text.

## 4. Short-Selling Restrictions in Indian Equities
- **Description**: In the Indian cash equity market, retail and non-institutional participants cannot carry short equity positions overnight (intraday only; overnight shorting requires stock futures or options).
- **Impact**: AlphaCheck's simulated portfolios operate long-only (holding cash or top-ranked equities) and benchmarked against cash equity indices, avoiding impossible overnight physical short positions.

