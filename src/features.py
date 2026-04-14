# src/features.py
# ─────────────────────────────────────────────────────────────
# Feature Engineering Module
# ─────────────────────────────────────────────────────────────
# Creates temporal and statistical features that help the model
# learn daily / weekly consumption cycles.
#
# Feature categories
# ──────────────────
# Temporal:  hour, day_of_week, month, is_weekend
# Lag:       lag_1h  (1 hr ago)
#            lag_24h (same hour yesterday)
#            lag_168h(same hour last week)
# Rolling:   rolling_mean_24h, rolling_std_24h
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os

FEATURES_OUT = "data/processed/features.csv"

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h",
]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Add engineered features to the cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned hourly power data with column 'power_kw'.

    Returns
    -------
    X : pd.DataFrame  — feature matrix
    y : pd.Series     — target (power_kw)
    feature_cols : list[str]
    """
    print("\n[2] FEATURE ENGINEERING")
    df = df.copy()

    # ── Temporal features ─────────────────────────────────────
    df["hour"]       = df.index.hour          # 0-23
    df["day_of_week"]= df.index.dayofweek     # 0=Mon … 6=Sun
    df["month"]      = df.index.month         # 1-12
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # ── Lag features ──────────────────────────────────────────
    # These give the model 'memory' of recent consumption —
    # crucial for time-series forecasting.
    df["lag_1h"]   = df["power_kw"].shift(1)    # 1 hour prior
    df["lag_24h"]  = df["power_kw"].shift(24)   # same time yesterday
    df["lag_168h"] = df["power_kw"].shift(168)  # same time last week

    # ── Rolling statistics ────────────────────────────────────
    # Capture recent trend and variability.
    df["rolling_mean_24h"] = df["power_kw"].rolling(window=24, min_periods=1).mean()
    df["rolling_std_24h"]  = df["power_kw"].rolling(window=24, min_periods=1).std()

    # Drop rows with NaN (first 168 rows due to lag_168h)
    before = len(df)
    df.dropna(inplace=True)
    print(f"  Dropped {before - len(df)} rows due to lag NaN (first week)")

    X = df[FEATURE_COLS]
    y = df["power_kw"]

    print(f"  ✓ Features: {X.shape[1]} columns  |  Samples: {len(X):,}")

    # Save feature matrix for inspection
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(FEATURES_OUT)
    print(f"  ✓ Saved to {FEATURES_OUT}")

    return X, y, FEATURE_COLS


# ── Quick self-test ───────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocess import load_and_clean
    df = load_and_clean()
    X, y, cols = build_features(df)
    print(X.describe())
