# src/evaluate.py
# ─────────────────────────────────────────────────────────────
# Model Evaluation Module
# ─────────────────────────────────────────────────────────────
# Metrics used
# ─────────────
# RMSE  — Root Mean Squared Error
#         Penalises large errors more than small ones.
#         Unit: kWh (same as the target — easy to interpret).
#
# MAE   — Mean Absolute Error
#         Average absolute difference between actual & predicted.
#         Unit: kWh. More robust to outliers than RMSE.
#
# MAPE  — Mean Absolute Percentage Error
#         Relative error in %. Useful for comparing across scales.
#         (Skipped when actuals contain zeros to avoid division error)
#
# R²    — Coefficient of Determination
#         Proportion of variance explained by the model.
#         R² = 1.0 → perfect; R² = 0 → baseline mean model.
#         Industry target for smart-meter forecasting: R² > 0.85
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

METRICS_OUT = "outputs/metrics.txt"


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list,
) -> tuple[np.ndarray, pd.Series]:
    """
    Predict on test set, compute metrics, print results, save to file.

    Returns
    -------
    y_pred     : np.ndarray of predictions
    fi_sorted  : pd.Series of feature importances (desc.)
    """
    print("\n[4] EVALUATION")

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    # MAPE — safe calculation
    nonzero = y_test != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100
    else:
        mape = float("nan")

    metrics = {
        "RMSE  (kWh)": round(rmse, 4),
        "MAE   (kWh)": round(mae,  4),
        "MAPE  (%)  ": round(mape, 2) if not np.isnan(mape) else "N/A",
        "R²  Score  ": round(r2,   4),
    }

    # ── Pretty-print ─────────────────────────────────────────
    print("  ┌─────────────────────────────┐")
    print("  │     Evaluation Results      │")
    print("  ├─────────────────────────────┤")
    for k, v in metrics.items():
        print(f"  │  {k}: {str(v):>8}          │")
    print("  └─────────────────────────────┘")

    # ── Feature importances ───────────────────────────────────
    fi = pd.Series(model.feature_importances_, index=feature_cols)
    fi_sorted = fi.sort_values(ascending=False)
    print("\n  Feature Importances (top 5):")
    for fname, fval in fi_sorted.head(5).items():
        bar = "█" * int(fval * 40)
        print(f"    {fname:<22} {fval:.4f}  {bar}")

    # ── Persist metrics ───────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    with open(METRICS_OUT, "w") as fh:
        fh.write("AI Energy Forecasting — Evaluation Results\n")
        fh.write("=" * 45 + "\n")
        for k, v in metrics.items():
            fh.write(f"{k}: {v}\n")
        fh.write("\nFeature Importances\n" + "-" * 30 + "\n")
        for fname, fval in fi_sorted.items():
            fh.write(f"{fname:<22}: {fval:.6f}\n")
    print(f"\n  ✓ Metrics saved → {METRICS_OUT}")

    return y_pred, fi_sorted
