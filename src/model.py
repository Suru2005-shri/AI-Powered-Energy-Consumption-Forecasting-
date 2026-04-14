# src/model.py
# ─────────────────────────────────────────────────────────────
# Model Training Module
# ─────────────────────────────────────────────────────────────
# Trains a Random Forest Regressor using a CHRONOLOGICAL split.
#
# Why Random Forest?
# ──────────────────
# • Handles non-linear patterns (daily/weekly energy cycles)
# • No feature scaling required
# • Fast training on CPU
# • Built-in feature importance — great for explainability
# • Robust to outliers
#
# Why chronological split (NOT random)?
# ──────────────────────────────────────
# Random splitting on time-series data leaks future values into
# the training set — the model would 'cheat' and appear to
# perform much better than it would in production.
# Always split time-series data in time order.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import joblib
import time
import os

from sklearn.ensemble import RandomForestRegressor

MODEL_OUT = "models/rf_model.pkl"
TRAIN_RATIO = 0.80   # 80 % train, 20 % test


def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Chronological train/test split → Random Forest training → save.

    Returns
    -------
    model                           : fitted RandomForestRegressor
    X_train, X_test, y_train, y_test: split arrays
    """
    print("\n[3] MODEL TRAINING")

    # ── Chronological split ───────────────────────────────────
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    train_start = X_train.index[0].strftime("%Y-%m-%d")
    train_end   = X_train.index[-1].strftime("%Y-%m-%d")
    test_start  = X_test.index[0].strftime("%Y-%m-%d")
    test_end    = X_test.index[-1].strftime("%Y-%m-%d")

    print(f"  Train: {len(X_train):>8,} rows  ({train_start} → {train_end})")
    print(f"  Test : {len(X_test):>8,} rows  ({test_start}  → {test_end})")

    # ── Model definition ─────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=200,       # 200 trees — good accuracy/speed tradeoff
        max_depth=15,           # prevents overfitting
        min_samples_leaf=10,    # smooths predictions on hourly data
        max_features="sqrt",    # standard for regression
        n_jobs=-1,              # use all CPU cores
        random_state=42,        # reproducible results
    )

    # ── Training ─────────────────────────────────────────────
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  ✓ Training complete in {elapsed:.1f}s  "
          f"({model.n_estimators} trees × {X_train.shape[1]} features)")

    # ── Persist model ─────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"  ✓ Model saved → {MODEL_OUT}")

    return model, X_train, X_test, y_train, y_test


def load_model(path: str = MODEL_OUT) -> RandomForestRegressor:
    """Load a previously saved model."""
    return joblib.load(path)


# ── Quick self-test ───────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocess import load_and_clean
    from src.features   import build_features

    df = load_and_clean()
    X, y, _ = build_features(df)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    print(f"\nModel type : {type(model).__name__}")
    print(f"n_features : {model.n_features_in_}")
