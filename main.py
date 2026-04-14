#!/usr/bin/env python3
# main.py
# ═══════════════════════════════════════════════════════════════
#  AI-Powered Energy Consumption Forecasting System
#  End-to-End Pipeline Runner
# ═══════════════════════════════════════════════════════════════
#
#  Usage
#  ─────
#    python main.py                 # full pipeline (auto-detects dataset)
#    python main.py --hours 168     # 7-day forecast instead of 24h
#    python main.py --no-plots      # skip chart generation
#
#  What this script does
#  ─────────────────────
#    Phase 1  Preprocessing  — load & clean raw data
#    Phase 2  Features       — engineer lag/rolling/temporal features
#    Phase 3  Training       — fit Random Forest on 80% of data
#    Phase 4  Evaluation     — RMSE, MAE, R², feature importance
#    Phase 5  Forecasting    — predict next N hours iteratively
#    Phase 6  Visualisation  — save 5 PNG charts to outputs/images/
# ═══════════════════════════════════════════════════════════════

import argparse
import sys
import time
import os

# ── Ensure project root is on PYTHONPATH ─────────────────────
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="AI Energy Forecasting Pipeline"
    )
    parser.add_argument("--hours",    type=int,  default=24,
                        help="Forecast horizon in hours (default: 24)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic dataset before running")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("  AI-Powered Energy Consumption Forecasting System")
    print("=" * 60)

    # ── Optional: generate synthetic data ────────────────────
    if args.generate:
        print("\n[0] GENERATING SYNTHETIC DATASET")
        from src.generate_data import generate_synthetic_data
        generate_synthetic_data()

    # ── Phase 1: Preprocessing ────────────────────────────────
    from src.preprocess import load_and_clean
    df = load_and_clean()

    # ── Phase 2: Feature Engineering ─────────────────────────
    from src.features import build_features
    X, y, feature_cols = build_features(df)

    # ── Phase 3: Model Training ───────────────────────────────
    from src.model import train_model
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # ── Phase 4: Evaluation ───────────────────────────────────
    from src.evaluate import evaluate_model
    y_pred, fi_sorted = evaluate_model(model, X_test, y_test, feature_cols)

    # ── Phase 5: Forecasting ──────────────────────────────────
    from src.forecast import forecast_future
    forecast_df = forecast_future(model, df, feature_cols, hours=args.hours)

    # ── Phase 6: Visualisation ────────────────────────────────
    if not args.no_plots:
        print("\n[6] VISUALISATION")
        from src.visualize import (
            plot_actual_vs_predicted,
            plot_feature_importance,
            plot_residuals,
            plot_forecast,
            plot_eda_overview,
        )
        plot_eda_overview(df)
        plot_actual_vs_predicted(y_test, y_pred)
        plot_feature_importance(fi_sorted)
        plot_residuals(y_test, y_pred)
        plot_forecast(forecast_df)

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  ✓ Pipeline complete in {elapsed:.1f}s")
    print("  Outputs:")
    print("    data/processed/clean_data.csv")
    print("    data/processed/features.csv")
    print("    models/rf_model.pkl")
    print("    outputs/metrics.txt")
    print("    outputs/predictions.csv")
    if not args.no_plots:
        print("    outputs/images/*.png  (5 charts)")
    print("=" * 60)


if __name__ == "__main__":
    main()
