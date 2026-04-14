# src/generate_data.py
# Generates a realistic synthetic energy dataset that mimics the UCI
# Household Electric Power Consumption dataset structure.
# Use this if you have not downloaded the real dataset yet.

import numpy as np
import pandas as pd
import os

def generate_synthetic_data(years=4, seed=42):
    """
    Generate synthetic hourly energy data (2020-2024) that mimics real
    household consumption patterns:
      - Higher usage mornings (7-9 AM) and evenings (6-10 PM)
      - Lower usage overnight (midnight-6 AM)
      - Slightly higher weekday usage vs weekends
      - Seasonal variation (more in winter and summer, less in spring/fall)
      - Random noise to simulate real-world variation
    """
    np.random.seed(seed)
    start = "2020-01-01"
    end   = f"{2020 + years}-01-01"

    idx = pd.date_range(start=start, end=end, freq="1h")[:-1]
    n   = len(idx)

    # --- Base load: time-of-day pattern ---
    hour_pattern = np.array([
        0.35, 0.28, 0.25, 0.22, 0.22, 0.30,   # midnight–5 AM
        0.55, 0.90, 1.10, 0.85, 0.75, 0.80,   # 6–11 AM
        0.85, 0.80, 0.75, 0.78, 0.82, 1.00,   # noon–5 PM
        1.25, 1.40, 1.35, 1.20, 0.90, 0.60,   # 6–11 PM
    ])

    # --- Day-of-week factor (weekends slightly lower) ---
    dow_factor = np.where(idx.dayofweek >= 5, 0.90, 1.00)

    # --- Monthly/seasonal factor ---
    month_pattern = np.array([
        1.20, 1.15, 1.00, 0.85, 0.80, 0.85,   # Jan-Jun
        0.90, 0.95, 0.85, 0.90, 1.05, 1.20,   # Jul-Dec
    ])

    # Assemble base signal
    base = (
        hour_pattern[idx.hour]
        * dow_factor
        * month_pattern[idx.month - 1]
    )

    # Add Gaussian noise (±15 %)
    noise = np.random.normal(0, 0.08, n)

    # Add slow drift / multi-day correlation
    drift = pd.Series(np.random.normal(0, 0.03, n)).rolling(48, min_periods=1).mean().values

    power = np.clip(base + noise + drift, 0.05, 3.5)

    df = pd.DataFrame({"power_kw": power}, index=idx)
    df.index.name = "datetime"

    os.makedirs("data/raw", exist_ok=True)
    out = "data/raw/synthetic_power.csv"
    df.to_csv(out)
    print(f"✓ Synthetic dataset generated: {len(df):,} hourly rows → {out}")
    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    print(df.describe())
    print(df.head(10))
