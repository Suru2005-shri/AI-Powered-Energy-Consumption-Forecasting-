# src/forecast.py
# ─────────────────────────────────────────────────────────────
# Future Forecasting Module
# ─────────────────────────────────────────────────────────────
# Generates iterative multi-step forecasts.
#
# Strategy
# ─────────
# Because lag features reference PAST values, we build each
# future row using previously predicted values — exactly how
# a real grid operator would use this model in production.
#
#   Step 0 → predict hour+1  (uses last known lags)
#   Step 1 → predict hour+2  (uses step-0 prediction as lag_1h)
#   Step 2 → predict hour+3  (uses step-1 prediction as lag_1h)
#   …and so on for 24h or 168h (7-day) horizon.
#
# Industry parallel
# ─────────────────
# Day-ahead forecasting (24 h) is the standard scheduling
# window used by electricity transmission operators (e.g.,
# National Grid, POSOCO) to commit generation and price
# balancing energy markets.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os


FORECAST_OUT = "outputs/predictions.csv"


def forecast_future(
    model,
    df: pd.DataFrame,
    feature_cols: list,
    hours: int = 24,
) -> pd.DataFrame:
    """
    Iteratively predict `hours` future hourly values.

    Parameters
    ----------
    model        : fitted sklearn model
    df           : full cleaned DataFrame (power_kw column + datetime index)
    feature_cols : list of feature names expected by the model
    hours        : forecast horizon in hours (default 24)

    Returns
    -------
    forecast_df : pd.DataFrame with columns [predicted_kw]
                  indexed by future datetime
    """
    print(f"\n[5] FORECASTING  ({hours}h horizon)")

    # History buffer: we need up to 168 h of past values
    history = df["power_kw"].values.tolist()
    last_dt  = df.index[-1]

    predictions = []

    for step in range(hours):
        next_dt = last_dt + pd.Timedelta(hours=step + 1)

        # Resolve lags from history + already-predicted values
        def get_lag(n):
            idx = -(n - step)  # how far back in history
            combined = history + [p["predicted_kw"] for p in predictions]
            try:
                return combined[idx] if idx < 0 else combined[-1]
            except IndexError:
                return np.mean(history[-24:])

        lag_1h   = predictions[-1]["predicted_kw"] if predictions else history[-1]
        lag_24h  = get_lag(24)
        lag_168h = get_lag(168)

        # Rolling stats from last 24 known + predicted values
        recent = (history + [p["predicted_kw"] for p in predictions])[-24:]
        roll_mean = float(np.mean(recent))
        roll_std  = float(np.std(recent)) if len(recent) > 1 else 0.0

        row = {
            "hour":            next_dt.hour,
            "day_of_week":     next_dt.dayofweek,
            "month":           next_dt.month,
            "is_weekend":      int(next_dt.dayofweek >= 5),
            "lag_1h":          lag_1h,
            "lag_24h":         lag_24h,
            "lag_168h":        lag_168h,
            "rolling_mean_24h": roll_mean,
            "rolling_std_24h":  roll_std,
        }

        X_row = pd.DataFrame([row])[feature_cols]
        pred  = float(model.predict(X_row)[0])
        pred  = max(pred, 0.0)   # clamp: power can't be negative

        predictions.append({"datetime": next_dt, "predicted_kw": pred})

    forecast_df = pd.DataFrame(predictions).set_index("datetime")

    # ── Save ─────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    forecast_df.to_csv(FORECAST_OUT)
    print(f"  ✓ {hours}h forecast saved → {FORECAST_OUT}")
    print(f"  Min: {forecast_df['predicted_kw'].min():.3f} kWh  |  "
          f"Max: {forecast_df['predicted_kw'].max():.3f} kWh  |  "
          f"Mean: {forecast_df['predicted_kw'].mean():.3f} kWh")

    return forecast_df
