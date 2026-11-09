"""Tests for M8 news alignment and sentiment timestamp rules."""

import numpy as np
import pandas as pd
import pytest
from alphacheck.sentiment import align_news_to_trading_days


def test_news_timestamp_alignment():
    calendar = pd.date_range("2023-01-02", periods=5, freq="B")  # Mon Jan 2 to Fri Jan 6

    # News 1: Mon Jan 2 at 08:30 AM (before open) -> eligible for Mon Jan 2
    # News 2: Mon Jan 2 at 11:00 AM (during trading) -> must defer to Tue Jan 3
    # News 3: Fri Jan 6 at 02:00 PM -> after cutoff, cannot trade until next week
    raw_news = pd.DataFrame(
        {
            "timestamp": [
                "2023-01-02 08:30:00",
                "2023-01-02 11:00:00",
                "2023-01-06 14:00:00",
            ],
            "ticker": ["RELIANCE", "TCS", "INFY"],
            "headline": [
                "Reliance announces strong pre-market earnings",
                "TCS wins mid-day digital contract",
                "Infosys post-market board meeting",
            ],
        }
    )

    aligned = align_news_to_trading_days(raw_news, calendar, cutoff_hour=9)

    # 1. Mon 08:30 AM maps to Mon Jan 2
    assert aligned.iloc[0]["trading_date"] == pd.Timestamp("2023-01-02")

    # 2. Mon 11:00 AM maps to Tue Jan 3 (strictly no intraday look-ahead)
    assert aligned.iloc[1]["trading_date"] == pd.Timestamp("2023-01-03")

    # 3. Fri 02:00 PM has no subsequent date in 5-day calendar -> dropped or NaT
    assert len(aligned) == 2

