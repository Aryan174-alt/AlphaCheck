"""M8 NLP: FinBERT Sentiment Inference & Pre-Market Timestamp Alignment.

Ensures headlines are filtered so ONLY news published strictly BEFORE 09:00 AM IST
of day t (or calendar day t-1 close) are included in day t features.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinBERTSentimentAnalyzer:
    """Inference engine for ProsusAI/finbert zero-shot financial sentiment."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def score_headlines(self, headlines: List[str], batch_size: int = 32) -> np.ndarray:
        """Score list of headlines returning net sentiment score in [-1.0, 1.0].

        Score = prob(positive) - prob(negative).
        """
        if not headlines:
            return np.array([])

        self.load_model()
        scores = []

        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # ProsusAI/finbert classes: 0: positive, 1: negative, 2: neutral
            pos = probs[:, 0]
            neg = probs[:, 1]
            net_sentiment = pos - neg
            scores.extend(net_sentiment)

        return np.array(scores)


def align_news_to_trading_days(
    news_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cutoff_hour: int = 9,  # 09:00 AM IST market open
) -> pd.DataFrame:
    """Strictly align headlines to the NEXT trading session if published at/after market open.

    Input news_df must have: ['timestamp', 'headline', 'ticker'].
    Timestamps are compared so no news after 09:00 AM on date t is used for date t features.
    """
    df = news_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    aligned_dates = []
    sorted_cal = sorted(calendar)

    for ts in df["timestamp"]:
        # If published after cutoff hour on date ts.date(), it can only be used on the subsequent trading date
        base_date = ts.floor("D")
        if ts.hour >= cutoff_hour or (ts.hour == cutoff_hour and ts.minute > 0):
            # Must shift to next available trading day
            eligible = [d for d in sorted_cal if d > base_date]
        else:
            eligible = [d for d in sorted_cal if d >= base_date]

        if eligible:
            aligned_dates.append(eligible[0])
        else:
            aligned_dates.append(pd.NaT)

    df["trading_date"] = aligned_dates
    valid = df.dropna(subset=["trading_date"]).copy()
    return valid


def aggregate_daily_sentiment(
    aligned_news_df: pd.DataFrame,
    cache_path: Optional[str] = "data/processed/sentiment_features.parquet",
) -> pd.DataFrame:
    """Aggregate headline sentiment per (trading_date, ticker).

    Computes: sentiment_mean, sentiment_count, sentiment_pos_ratio.
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_parquet(cache_path)

    grouped = (
        aligned_news_df.groupby(["trading_date", "ticker"])
        .agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_count=("sentiment_score", "count"),
            sentiment_pos_ratio=("sentiment_score", lambda s: float((s > 0).mean())),
        )
        .rename_axis(index={"trading_date": "date"})
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        grouped.to_parquet(cache_path)

    return grouped

