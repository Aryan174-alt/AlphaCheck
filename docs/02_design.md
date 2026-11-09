# 02 Architecture & Design: AlphaCheck

## 1. System Architecture Diagram

```mermaid
flowchart TD
    subgraph S1["Data Ingestion & Hygiene (M1)"]
        YF["yfinance API (Nifty 50 + ^NSEI)"] --> CACHE["Parquet Local Cache (data/raw)"]
        NEWS["Financial News / PhraseBank"] --> NCACHE["Parquet News Cache"]
        CACHE --> ADJ["Split & Dividend Adjuster"]
        ADJ --> QC["Data Quality Checker"]
        QC --> PROC["Clean Daily Parquet (data/processed)"]
    end

    subgraph S2["Feature Engineering (M2, M8, A2, A4)"]
        PROC --> PF["Price Features (M2)\n• Returns (1/5/10/20/60d)\n• Volatility (20d)\n• RSI, MACD, Volume Delta\n• Rolling Beta vs ^NSEI\n*Strictly shifted t-1 to past*"]
        NCACHE --> FINBERT["FinBERT / Fine-Tuned NLP (M8, A1)"]
        NCACHE --> EMBED["MiniLM Embeddings + PCA (A2)"]
        NCACHE --> LLMTAG["Optional Ollama LLM Event Tags (A4)"]
        FINBERT --> FEAT_MERGE["Feature Set Combiner\n(Aligned strictly before 09:00 AM IST)"]
        EMBED --> FEAT_MERGE
        LLMTAG --> FEAT_MERGE
        PF --> FEAT_MERGE
    end

    subgraph S3["Label Generation & Leakage Guard (M3)"]
        PROC --> LBL["5-Day Forward Excess Return (M3)\ny = (R_stock > R_nifty)\n*Drop last 5 rows without future*"]
        LBL --> LEAK_TEST["Leakage Tests (tests/test_leakage.py)\nGuarantees Covariance(Feature_t, Return_t+k) = 0"]
    end

    subgraph S4["Validation Protocol & Model Zoo (M4, M5, A3)"]
        FEAT_MERGE --> WF["Expanding Walk-Forward Splitter\nTrain: [0, t]\nEmbargo Gap: 5 trading days\nTest: [t + 5, t + test_size]"]
        LBL --> WF
        WF --> BSL["Baselines (M4)\n• Buy & Hold Nifty\n• Equal Weight 50\n• Cross-sectional Momentum\n• Random Signal (Monte Carlo)"]
        WF --> ML["Classical ML (M5)\n• Logistic Regression\n• LightGBM / XGBoost\n*Scalers & Tuning strictly inside train*"]
        WF --> DEEP["Deep Sequence Model (A3)\n• PyTorch Transformer / TCN\n• Rolling sequences of features"]
    end

    subgraph S5["Realistic Backtest Engine (M6)"]
        BSL --> SIM["Paper Trading Simulator (M6)\n• Top-Quintile Portfolio\n• 0.10% friction per side\n• 0.02% slippage\n• t+1 execution\n• Cash & Position Reconciliation"]
        ML --> SIM
        DEEP --> SIM
    end

    subgraph S6["Metrics, Robustness & Explainability (M7, M9, A5)"]
        SIM --> MET["Performance Engine (M7)\n• Sharpe, Sortino, Max Drawdown\n• Calmar, Turnover, Hit Rate\n• Stationary Bootstrap 95% CIs"]
        SIM --> ROBUST["Stress & Sensitivity (M9)\n• Cost Scenarios: 0, 5, 10, 30 bps\n• Sub-period / Regime splits\n• Worst drawdown post-mortems"]
        ML --> EXP["Explainability (A5)\n• Tree SHAP values\n• Deep attention / gradient maps"]
        DEEP --> EXP
    end

    subgraph S7["Reporting & Orchestration (M11)"]
        MET --> REP["Reports & Artifacts (reports/)\n• figures/ (equity curves, drawdowns, SHAP)\n• tables/ (metrics summary CSVs)\n• docs/RESULTS.md & README.md"]
        ROBUST --> REP
        EXP --> REP
        REP --> STREAM["Optional Streamlit App (dashboard.py)"]
    end
```

---

## 2. Walk-Forward with Embargo Schema

Because labels represent 5-day future outperformance, each training label at day $t$ utilizes price information up to day $t+5$. An immediate adjacent test split would cause label information leakage. We enforce an **Embargo Gap** of at least 5 trading days:

```
Window 1:
[======= Train Set (e.g. 3 years) =======] [EMBARGO: 5 days] [=== Test Window (6 months) ===]

Window 2 (Expanding):
[============== Train Set (3.5 years) ==============] [EMBARGO: 5 days] [=== Test Window ===]
```

---

## 3. Module Inventory and Responsibilities

| Module | Source File | Core Responsibility |
|---|---|---|
| **M1** | `src/alphacheck/data.py` | Robust fetching of Nifty 50 and index data via yfinance, local parquet caching, adjustments, and DQ checks. |
| **M2** | `src/alphacheck/features.py` | Strictly backward-looking technical, momentum, and volatility features. Shifted to eliminate look-ahead. |
| **M3** | `src/alphacheck/labels.py` | 5-day forward excess return binary targets, dropping rows without complete forward horizons. |
| **M4** | `src/alphacheck/baselines.py` | Passive index benchmark, equal-weight basket, cross-sectional momentum rule, and Monte Carlo random trader. |
| **M5** | `src/alphacheck/models.py` | Classical ML pipelines (Logistic Regression, LightGBM/XGBoost) with clean fit/predict interfaces. |
| **M5v**| `src/alphacheck/validation.py` | Expanding-window walk-forward generator with enforced 5-day purge/embargo gap. |
| **M6** | `src/alphacheck/backtest.py` | Friction-aware paper-trading engine with trade-log export and cash accounting reconciliation. |
| **M7** | `src/alphacheck/metrics.py` | Sharpe, Sortino, Drawdown, Calmar, Hit Rate, Turnover, and stationary bootstrap confidence intervals. |
| **M8** | `src/alphacheck/sentiment.py` | FinBERT zero-shot sentiment inference aligned strictly before market open. |
| **M9** | `src/alphacheck/robustness.py` | Transaction cost sensitivity grid (0, 5, 10, 30 bps), market regime analysis, and SHAP feature importance. |
| **A1** | `src/alphacheck/sentiment_finetune.py` | Financial PhraseBank sentence fine-tuning of DistilBERT vs FinBERT zero-shot bake-off. |
| **A2** | `src/alphacheck/embeddings.py` | Sentence-Transformers (`all-MiniLM-L6-v2`) dense text embeddings with PCA dimensionality reduction. |
| **A3** | `src/alphacheck/deep_models.py` | PyTorch Transformer Encoder and TCN for sequence feature modeling. |
| **A4** | `src/alphacheck/llm_tagger.py` | Optional local Ollama LLM event classification (earnings, macro, regulatory, other) with cached disk store. |
| **A5** | `src/alphacheck/explain.py` | SHAP explanations for tree models and attention/gradient visualization for deep sequence models. |
| **M11**| `src/alphacheck/run_all.py` | One-command master pipeline orchestrator reproducing all experiments, tables, and figures. |
| **App**| `src/alphacheck/dashboard.py` | Lightweight interactive Streamlit app to explore equity curves and metrics. |

