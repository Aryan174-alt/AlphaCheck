"""Expanding-window walk-forward validation with enforced purge / embargo gap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Tuple
import pandas as pd


@dataclass
class WalkForwardSplit:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_days: int


class WalkForwardValidator:
    """Generates expanding-window train/test date splits with a mandatory embargo gap.

    Because labels at time t require `embargo_days` future observations, any test set
    must start strictly `embargo_days` trading days after train_end.
    """

    def __init__(
        self,
        calendar: pd.DatetimeIndex,
        initial_train_days: int = 750,  # ~3 years of trading days (250/yr)
        step_days: int = 125,          # ~6 months step
        test_days: int = 125,          # ~6 months test
        embargo_days: int = 5,         # equal to forward label horizon
    ):
        self.calendar = pd.DatetimeIndex(sorted(calendar.unique()))
        self.initial_train_days = initial_train_days
        self.step_days = step_days
        self.test_days = test_days
        self.embargo_days = embargo_days

    def split(self) -> Generator[WalkForwardSplit, None, None]:
        n_days = len(self.calendar)
        train_len = self.initial_train_days
        fold_idx = 0

        while True:
            train_end_idx = train_len - 1
            test_start_idx = train_end_idx + self.embargo_days + 1
            test_end_idx = test_start_idx + self.test_days - 1

            if test_end_idx >= n_days:
                # If remaining window is at least half of test_days, yield final partial fold
                if n_days - test_start_idx >= max(20, self.test_days // 2):
                    test_end_idx = n_days - 1
                else:
                    break

            split_obj = WalkForwardSplit(
                fold_idx=fold_idx,
                train_start=self.calendar[0],
                train_end=self.calendar[train_end_idx],
                test_start=self.calendar[test_start_idx],
                test_end=self.calendar[test_end_idx],
                embargo_days=self.embargo_days,
            )
            yield split_obj

            if test_end_idx == n_days - 1:
                break

            train_len += self.step_days
            fold_idx += 1

