#!/usr/bin/env python
"""
Standalone model training script — called by the web-app pipeline only.
Notebook runs are never tracked here.

Reads from  data/processed/
Writes to   models/
Records to  data/cgm_app.db  (model_runs table)

Usage:
    python train_models.py --models ridge
    python train_models.py --models ridge sarimax
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT          = os.environ.get("APP_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models")
DB_PATH       = os.environ.get("DB_PATH", os.path.join(ROOT, "data", "cgm_app.db"))

os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_HOLD_DAYS = 21


# ── helpers ────────────────────────────────────────────────────────────────────
def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))


def record(model_name, rmse_train, rmse_test, mae_test, r2_test, n_train, n_test, notes=""):
    """Insert one row into model_runs via raw sqlite3 (no ORM dependency)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO model_runs
                (run_ts, model_name, rmse_train, rmse_test, mae_test, r2_test,
                 n_train, n_test, notes, triggered_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'web_app')
            """,
            (
                datetime.utcnow().isoformat(),
                model_name,
                round(rmse_train, 4) if rmse_train is not None else None,
                round(rmse_test,  4),
                round(mae_test,   4),
                round(r2_test,    4),
                n_train,
                n_test,
                notes,
            ),
        )
        conn.commit()
    print(f"  Recorded: {model_name}  RMSE_test={rmse_test:.2f}  MAE={mae_test:.2f}  R2={r2_test:.3f}")


# ── Ridge per-meal ─────────────────────────────────────────────────────────────
def train_ridge():
    print("Training Ridge (per-meal)...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "per_meal_features.csv"), parse_dates=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    FEATURES = [
        "pre_meal_glucose", "glucose_velocity_30min", "glucose_60min_before_mean",
        "carbs_g_final", "hour_sin", "hour_cos", "is_weekend", "day_index",
        "walk_min_within_90", "fasted_meal", "prev_fast_hours",
        "minutes_since_last_glipizide",
    ]
    TARGET = "peak_delta"

    df = df.dropna(subset=[TARGET])
    split_date = df["ts"].max() - pd.Timedelta(days=TRAIN_HOLD_DAYS)
    train = df[df["ts"] <= split_date]
    test  = df[df["ts"] >  split_date]

    # Keep only features present in the file
    feats = [f for f in FEATURES if f in df.columns]

    X_train = train[feats].values
    y_train = np.log1p(train[TARGET].clip(lower=0).values)
    X_test  = test[feats].values
    y_test_raw = test[TARGET].clip(lower=0).values

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, y_train)

    pred_train = np.expm1(pipe.predict(X_train))
    pred_test  = np.expm1(pipe.predict(X_test))

    rmse_tr = rmse(np.expm1(y_train), pred_train)
    rmse_te = rmse(y_test_raw, pred_test)
    mae_te  = float(mean_absolute_error(y_test_raw, pred_test))
    r2_te   = float(r2_score(y_test_raw, pred_test))

    print(f"  Train RMSE={rmse_tr:.2f}  Test RMSE={rmse_te:.2f}  MAE={mae_te:.2f}  R2={r2_te:.3f}")
    print(f"  n_train={len(train)}  n_test={len(test)}")

    joblib.dump(pipe, os.path.join(MODELS_DIR, "per_meal_model.joblib"))
    record("ridge_per_meal", rmse_tr, rmse_te, mae_te, r2_te, len(train), len(test))


# ── SARIMAX hourly ─────────────────────────────────────────────────────────────
def train_sarimax():
    print("Training SARIMAX (hourly) — this may take several minutes...")
    try:
        import pickle
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("  statsmodels not available, skipping SARIMAX.")
        return

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "hourly_features.csv"), parse_dates=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    EXOG_COLS = ["carb_load_decayed", "walk_min_last_2h", "glipizide_active", "is_fasting", "hour_sin", "hour_cos"]
    split_ts  = df["ts"].iloc[-TRAIN_HOLD_DAYS * 24]
    train = df[df["ts"] <  split_ts].copy()
    test  = df[df["ts"] >= split_ts].copy()

    y_train = train.set_index("ts")["glucose_hourly_mean"].astype(float)
    y_test  = test.set_index("ts")["glucose_hourly_mean"].astype(float).dropna()
    X_train = train.set_index("ts")[EXOG_COLS].astype(float)
    X_test  = test.set_index("ts")[EXOG_COLS].astype(float)

    # Load best order from prior grid search if available, else use known best
    meta_path = os.path.join(MODELS_DIR, "sarimax_hourly_meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        order          = meta["order"]
        seasonal_order = meta["seasonal_order"]
        simple_diff    = meta.get("simple_differencing", False)
        print(f"  Using stored best order: {order} x {seasonal_order}")
    else:
        order          = (1, 0, 2)
        seasonal_order = (1, 0, 1, 24)
        simple_diff    = False
        print(f"  No meta file found, using default order: {order} x {seasonal_order}")

    model = SARIMAX(
        y_train, exog=X_train,
        order=order, seasonal_order=seasonal_order,
        enforce_stationarity=True, enforce_invertibility=True,
        simple_differencing=simple_diff,
    )
    result = model.fit(disp=False, maxiter=300)

    # 1-step-ahead in-sample
    fitted     = result.fittedvalues.dropna()
    y_tr_align = y_train.reindex(fitted.index).dropna()
    common     = fitted.index.intersection(y_tr_align.index)
    rmse_tr    = rmse(y_tr_align.loc[common].values, fitted.loc[common].values)

    # Multi-step test forecast
    fc      = result.get_forecast(steps=len(y_test), exog=X_test.iloc[:len(y_test)])
    fc_mean = fc.predicted_mean.reindex(y_test.index)
    valid   = y_test.notna() & fc_mean.notna()
    rmse_te = rmse(y_test[valid].values, fc_mean[valid].values)
    mae_te  = float(mean_absolute_error(y_test[valid].values, fc_mean[valid].values))
    r2_te   = float(r2_score(y_test[valid].values, fc_mean[valid].values))

    print(f"  Train RMSE={rmse_tr:.2f}  Test RMSE={rmse_te:.2f}  MAE={mae_te:.2f}  R2={r2_te:.3f}")

    result.save(os.path.join(MODELS_DIR, "sarimax_hourly.pkl"))
    record("sarimax_hourly", rmse_tr, rmse_te, mae_te, r2_te, len(y_train.dropna()), len(y_test))


# ── entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["ridge", "sarimax"],
                        default=["ridge"], help="Which models to train")
    args = parser.parse_args()

    if "ridge" in args.models:
        train_ridge()
    if "sarimax" in args.models:
        train_sarimax()

    print("Done.")
