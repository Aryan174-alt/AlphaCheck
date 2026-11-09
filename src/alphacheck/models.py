"""M5 Machine Learning Models: Logistic Regression & Gradient Boosting pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def get_feature_and_label_matrices(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Align features and labels on (date, ticker) index, dropping NaNs."""
    if feature_cols is None:
        feature_cols = [c for c in features_df.columns if c not in ["date", "ticker"]]

    # Join on index
    merged = features_df[feature_cols].join(labels_df[["target"]], how="inner")
    clean = merged.dropna()

    X = clean[feature_cols]
    y = clean["target"].astype(int)
    return X, y


def build_logistic_model(C: float = 0.1, random_state: int = 42) -> Pipeline:
    """Logistic regression with robust standard scaling."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    random_state=random_state,
                    max_iter=1000,
                ),
            ),
        ]
    )


def build_gradient_boosting_model(
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """XGBoost classifier with conservative parameters to avoid overfitting."""
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )

