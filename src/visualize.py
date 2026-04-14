# src/visualize.py
# ─────────────────────────────────────────────────────────────
# Visualisation Module
# ─────────────────────────────────────────────────────────────
# Generates 4 publication-quality PNG charts saved to outputs/images/
#
#  Chart 1: actual_vs_predicted.png  — overlaid line chart (2-week window)
#  Chart 2: feature_importance.png  — horizontal bar chart
#  Chart 3: residuals.png           — error distribution histogram
#  Chart 4: forecast_24h.png        — future demand with ±1 std band
#  Chart 5: eda_overview.png        — exploratory data analysis (bonus)
# ─────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all environments

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import os

IMG_DIR = "outputs/images"

# ── Shared style ─────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "figure.dpi":   100,
})

BLUE   = "#2563EB"
GREEN  = "#16A34A"
PURPLE = "#7C3AED"
AMBER  = "#D97706"
RED    = "#DC2626"
GRAY   = "#6B7280"

os.makedirs(IMG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. Actual vs Predicted
# ─────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    n_days: int = 14,
) -> None:
    """
    Overlaid line chart: actual (blue) vs predicted (green).
    Shows n_days days of test data so the chart isn't crowded.
    """
    n = n_days * 24
    actual = y_test.values[:n]
    pred   = y_pred[:n]
    x      = range(n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={"height_ratios": [3, 1]})

    # Top: overlaid lines
    ax = axes[0]
    ax.plot(x, actual, color=BLUE,  lw=1.3, label="Actual",    alpha=0.9)
    ax.plot(x, pred,   color=GREEN, lw=1.3, label="Predicted", alpha=0.85, linestyle="--")
    ax.fill_between(x, actual, pred, alpha=0.08, color=RED, label="Error")
    ax.set_title(f"Actual vs Predicted Energy Consumption  ({n_days}-day window)")
    ax.set_ylabel("Power (kWh)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, n)

    # Bottom: residual sparkline
    ax2 = axes[1]
    ax2.bar(x, actual - pred, color=AMBER, alpha=0.6, width=1)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.set_ylabel("Error (kWh)")
    ax2.set_xlabel("Hours")
    ax2.set_xlim(0, n)

    plt.tight_layout()
    path = f"{IMG_DIR}/actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")


# ─────────────────────────────────────────────────────────────
# 2. Feature Importance
# ─────────────────────────────────────────────────────────────
def plot_feature_importance(fi_sorted: pd.Series) -> None:
    """Horizontal bar chart of feature importances (all features)."""
    fi = fi_sorted.sort_values()   # ascending for barh
    colors = [PURPLE if v >= fi.median() else GRAY for v in fi]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(fi.index, fi.values, color=colors, edgecolor="white", height=0.65)

    # Value labels on bars
    for bar, val in zip(bars, fi.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=GRAY)

    ax.set_title("Feature Importance  (Random Forest)")
    ax.set_xlabel("Importance Score")
    ax.set_xlim(0, fi.max() * 1.18)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()

    path = f"{IMG_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")


# ─────────────────────────────────────────────────────────────
# 3. Residuals Distribution
# ─────────────────────────────────────────────────────────────
def plot_residuals(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Histogram of prediction errors + kernel density estimate.
    A well-fitted model shows a tight, symmetric distribution around 0.
    """
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax = axes[0]
    ax.hist(residuals, bins=80, color=AMBER, edgecolor="white", alpha=0.85)
    ax.axvline(0,               color=RED,  lw=2.0, linestyle="--", label="Zero error")
    ax.axvline(np.mean(residuals), color=BLUE, lw=1.5, linestyle="-",  label=f"Mean = {np.mean(residuals):.3f}")
    ax.set_title("Prediction Residuals — Histogram")
    ax.set_xlabel("Error (kWh)")
    ax.set_ylabel("Count")
    ax.legend()

    # Scatter: predicted vs actual (perfect model → diagonal)
    ax2 = axes[1]
    lim = [min(y_test.min(), y_pred.min()) * 0.95,
           max(y_test.max(), y_pred.max()) * 1.05]
    ax2.scatter(y_test.values, y_pred, alpha=0.2, s=5, color=PURPLE)
    ax2.plot(lim, lim, color=RED, lw=1.5, linestyle="--", label="Perfect fit")
    ax2.set_title("Predicted vs Actual  (scatter)")
    ax2.set_xlabel("Actual (kWh)")
    ax2.set_ylabel("Predicted (kWh)")
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.legend()

    plt.tight_layout()
    path = f"{IMG_DIR}/residuals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")


# ─────────────────────────────────────────────────────────────
# 4. 24-Hour Forecast
# ─────────────────────────────────────────────────────────────
def plot_forecast(forecast_df: pd.DataFrame) -> None:
    """
    Future demand forecast chart with ±1 std-dev confidence band.
    Mimics what a grid operator dashboard would display.
    """
    t   = forecast_df.index
    val = forecast_df["predicted_kw"]
    std = val.std()

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.fill_between(t, val - std, val + std,
                    alpha=0.15, color=BLUE, label="±1 std dev band")
    ax.plot(t, val, color=BLUE, lw=2.2, marker="o", ms=4, label="Forecast (kWh)")

    # Annotate peak
    peak_t   = val.idxmax()
    peak_val = val.max()
    ax.annotate(f"Peak\n{peak_val:.2f} kWh",
                xy=(peak_t, peak_val),
                xytext=(peak_t, peak_val + std * 0.8),
                ha="center", fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d %b"))
    ax.set_title(f"24-Hour Energy Demand Forecast  (horizon: {len(forecast_df)}h)")
    ax.set_ylabel("Power (kWh)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.set_xlim(t[0], t[-1])
    plt.tight_layout()

    path = f"{IMG_DIR}/forecast_24h.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")


# ─────────────────────────────────────────────────────────────
# 5. EDA Overview (bonus chart)
# ─────────────────────────────────────────────────────────────
def plot_eda_overview(df: pd.DataFrame) -> None:
    """
    3-panel exploratory data analysis chart:
      - Average consumption by hour of day
      - Average consumption by day of week
      - Monthly average consumption
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Hour of day
    hourly = df.groupby(df.index.hour)["power_kw"].mean()
    axes[0].bar(hourly.index, hourly.values, color=BLUE, alpha=0.85, edgecolor="white")
    axes[0].set_title("Avg Consumption by Hour")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Avg Power (kWh)")
    axes[0].set_xticks(range(0, 24, 2))

    # Day of week
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = df.groupby(df.index.dayofweek)["power_kw"].mean()
    colors = [AMBER if i >= 5 else GREEN for i in range(7)]
    axes[1].bar(days, daily.values, color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_title("Avg Consumption by Day")
    axes[1].set_xlabel("Day of Week")

    # Month
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = df.groupby(df.index.month)["power_kw"].mean()
    axes[2].plot(months, monthly.values, color=PURPLE, marker="o",
                 lw=2, ms=6)
    axes[2].fill_between(range(12), monthly.values,
                         alpha=0.15, color=PURPLE)
    axes[2].set_title("Avg Consumption by Month")
    axes[2].set_xlabel("Month")
    axes[2].set_xticks(range(12))
    axes[2].set_xticklabels(months, rotation=45, fontsize=8)

    plt.suptitle("Exploratory Data Analysis — Energy Consumption Patterns", y=1.02)
    plt.tight_layout()

    path = f"{IMG_DIR}/eda_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")
