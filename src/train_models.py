#!/usr/bin/env python
"""
Standalone model training script - called by the web-app pipeline only.
Notebook runs are never tracked here.

Reads from  data/processed/
Writes to   models/
Records to  data/cgm_app.db  (model_runs table)

Usage:
    python train_models.py --models ridge
    python train_models.py --models ridge sarimax
    python train_models.py --models ridge rf
    python train_models.py --models lstm
    python train_models.py --models ridge sarimax rf lstm
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

# Set by __main__ before any train_* call; shared across all models in one run.
RUN_TAG:   str | None = None
DATE_FROM: str | None = None   # "YYYY-MM-DD" inclusive; None = no start filter
DATE_TO:   str | None = None   # "YYYY-MM-DD" inclusive; None = no end filter

TRAIN_HOLD_DAYS = 21

# Shared per-meal feature list (used by Ridge, RF, and LSTM)
PER_MEAL_FEATURES = [
    "pre_meal_glucose", "glucose_velocity_30min", "glucose_60min_before_mean",
    "carbs_g_final", "hour_sin", "hour_cos", "is_weekend", "day_index",
    "walk_min_within_90", "fasted_meal", "prev_fast_hours",
    "minutes_since_last_glipizide",
]
TARGET = "peak_delta"

# Shared feature list for LSTM 5-min and LSTM 3h forecasters
LSTM_FEAT_COLS = [
    "glucose_mg_dl",
    "carb_load_decayed",
    "carbs_last_30min", "carbs_last_1h", "carbs_last_2h", "carbs_last_3h",
    "carbs_last_meal", "minutes_since_last_meal",
    "walk_min_last_1h", "walk_min_last_2h", "walk_min_last_3h",
    "minutes_since_last_walk",
    "minutes_since_last_glipizide", "glipizide_active",
    "is_fasting", "hours_into_fast",
    "hour_sin", "hour_cos", "is_weekend",
]


# ── helpers ────────────────────────────────────────────────────────────────────
def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))


def _apply_date_filter(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Restrict df to [DATE_FROM, DATE_TO] using the module-level globals.

    Both bounds are inclusive calendar-day boundaries in UTC.
    No-ops immediately if both globals are None (all-data run).
    Prints a summary line when filtering is active.
    """
    if not DATE_FROM and not DATE_TO:
        return df
    orig = len(df)
    if DATE_FROM:
        df = df[df[ts_col] >= DATE_FROM]
    if DATE_TO:
        # Include the full calendar day: ts < DATE_TO + 1 day
        dt_end = pd.Timestamp(DATE_TO) + pd.Timedelta(days=1)
        df = df[df[ts_col] < dt_end]
    df = df.reset_index(drop=True)
    span = ""
    if len(df) > 0:
        span = f"  ({df[ts_col].min().date()} → {df[ts_col].max().date()})"
    print(f"  Date filter: {len(df):,} of {orig:,} rows retained{span}")
    return df


def record(model_name, rmse_train, rmse_test, mae_test, r2_test, n_train, n_test, notes=""):
    """Insert one row into model_runs via raw sqlite3 (no ORM dependency).
    RUN_TAG (module-level global) is stamped on every row in a pipeline invocation.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO model_runs
                (run_ts, model_name, rmse_train, rmse_test, mae_test, r2_test,
                 n_train, n_test, notes, triggered_by, run_tag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'web_app', ?)
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
                RUN_TAG,
            ),
        )
        conn.commit()
    print(f"  Recorded: {model_name}  RMSE_test={rmse_test:.2f}  MAE={mae_test:.2f}  R2={r2_test:.3f}  run_tag={RUN_TAG}")


def _load_per_meal():
    """Load and temporally split per_meal_features.csv. Returns (df, train, test, feats)."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "per_meal_features.csv"), parse_dates=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df = df.dropna(subset=[TARGET])
    df = _apply_date_filter(df)
    feats = [f for f in PER_MEAL_FEATURES if f in df.columns]
    split_date = df["ts"].max() - pd.Timedelta(days=TRAIN_HOLD_DAYS)
    train = df[df["ts"] <= split_date]
    test  = df[df["ts"] >  split_date]
    return df, train, test, feats


# ── Ridge per-meal ─────────────────────────────────────────────────────────────
def train_ridge():
    print("Training Ridge (per-meal)...")
    print("  Loading per-meal features...")
    _, train, test, feats = _load_per_meal()
    print(f"  n_train={len(train)}  n_test={len(test)}  features={len(feats)}")

    X_train    = train[feats].values
    y_train    = np.log1p(train[TARGET].clip(lower=0).values)
    X_test     = test[feats].values
    y_test_raw = test[TARGET].clip(lower=0).values

    print("  Fitting Ridge pipeline (impute → scale → Ridge)...")
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


# ── Random Forest per-meal ─────────────────────────────────────────────────────
def train_rf():
    print("Training Random Forest (per-meal)...")
    from sklearn.ensemble import RandomForestRegressor

    print("  Loading per-meal features...")
    _, train, test, feats = _load_per_meal()
    print(f"  n_train={len(train)}  n_test={len(test)}  features={len(feats)}")

    X_train    = train[feats].values
    y_train    = np.log1p(train[TARGET].clip(lower=0).values)
    X_test     = test[feats].values
    y_test_raw = test[TARGET].clip(lower=0).values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    print(f"  Fitting RandomForest (n_estimators=200, n_jobs=-1) on {len(X_train_imp)} samples...")
    rf = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_imp, y_train)

    pred_train = np.expm1(rf.predict(X_train_imp))
    pred_test  = np.expm1(rf.predict(X_test_imp))

    rmse_tr = rmse(np.expm1(y_train), pred_train)
    rmse_te = rmse(y_test_raw, pred_test)
    mae_te  = float(mean_absolute_error(y_test_raw, pred_test))
    r2_te   = float(r2_score(y_test_raw, pred_test))

    print(f"  Train RMSE={rmse_tr:.2f}  Test RMSE={rmse_te:.2f}  MAE={mae_te:.2f}  R2={r2_te:.3f}")
    print(f"  n_train={len(train)}  n_test={len(test)}")

    joblib.dump(
        {"model": rf, "imputer": imputer, "features": feats},
        os.path.join(MODELS_DIR, "rf_per_meal.joblib"),
    )
    record("rf_per_meal", rmse_tr, rmse_te, mae_te, r2_te, len(train), len(test),
           notes="n_estimators=200 min_samples_leaf=5")


# ── LSTM per-meal ──────────────────────────────────────────────────────────────
def train_lstm():
    """
    LSTM using 24 five-minute CGM readings (2-hour window) before each meal as
    the input sequence.  Requires PyTorch; skips gracefully if not installed.
    Also skips if fewer than MIN_SAMPLES valid sequences can be built - the model
    is infrastructure for when more data is available.
    """
    MIN_SAMPLES = 150   # skip if not enough meals have dense pre-meal CGM coverage
    SEQ_LEN     = 24    # 24 × 5 min = 2-hour window
    EPOCHS      = 100
    HIDDEN      = 32
    LAYERS      = 2

    print("Training LSTM (per-meal)...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  PyTorch not installed - skipping LSTM.  Run: pip install torch")
        return

    # ── Load per-meal labels ───────────────────────────────────────────────────
    print("  Loading per-meal features and CGM readings...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "per_meal_features.csv"), parse_dates=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df = df.dropna(subset=[TARGET])

    # ── Load CGM readings from DB ──────────────────────────────────────────────
    with sqlite3.connect(DB_PATH) as conn:
        cgm_df = pd.read_sql_query(
            "SELECT ts, glucose_mg_dl FROM cgm_readings ORDER BY ts",
            conn,
            parse_dates=["ts"],
        )
    cgm_df["ts"] = pd.to_datetime(cgm_df["ts"])
    cgm_df = cgm_df.dropna(subset=["glucose_mg_dl"]).sort_values("ts").set_index("ts")
    print(f"  {len(df)} meals  {len(cgm_df):,} CGM readings — building 2h pre-meal sequences...")

    # ── Build sequences ────────────────────────────────────────────────────────
    sequences, targets, meal_timestamps = [], [], []

    for _, row in df.iterrows():
        meal_time    = pd.Timestamp(row["ts"])
        window_start = meal_time - pd.Timedelta(minutes=SEQ_LEN * 5)

        seg = cgm_df.loc[window_start:meal_time, "glucose_mg_dl"].dropna()
        if len(seg) < SEQ_LEN // 2:   # need at least 12 of 24 readings
            continue

        # Resample to exactly SEQ_LEN evenly-spaced readings via time interpolation
        idx_even      = pd.date_range(end=meal_time, periods=SEQ_LEN, freq="5min")
        seg_reindexed = (
            seg.reindex(seg.index.union(idx_even))
               .interpolate("time")
               .reindex(idx_even)
        )
        if seg_reindexed.isna().any():
            continue

        sequences.append(seg_reindexed.values.astype(np.float32))
        targets.append(float(row[TARGET]))
        meal_timestamps.append(meal_time)

    n = len(sequences)
    print(f"  {n} meals with valid CGM sequences (of {len(df)} total).")

    if n < MIN_SAMPLES:
        print(f"  Only {n} valid sequences - need ≥{MIN_SAMPLES} to train LSTM.")
        print("  Continue logging data and re-run when more meals are available.")
        return

    # ── Temporal train / test split ────────────────────────────────────────────
    split_ts   = max(meal_timestamps) - pd.Timedelta(days=TRAIN_HOLD_DAYS)
    train_mask = np.array([t <= split_ts for t in meal_timestamps])
    test_mask  = ~train_mask

    if train_mask.sum() < 20 or test_mask.sum() < 5:
        print("  Not enough samples after temporal split. Skipping.")
        return

    X = np.stack(sequences)                                     # (N, SEQ_LEN)
    y = np.log1p(np.clip(np.array(targets), 0, None)).astype(np.float32)
    y_raw = np.clip(np.array(targets), 0, None).astype(np.float32)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    y_test_raw      = y_raw[test_mask]

    # Normalise glucose using training-set statistics
    mu  = X_train.mean()
    sig = X_train.std() + 1e-6
    X_train_n = (X_train - mu) / sig
    X_test_n  = (X_test  - mu) / sig

    # ── Model definition ───────────────────────────────────────────────────────
    class LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1, hidden_size=HIDDEN, num_layers=LAYERS,
                batch_first=True,
                dropout=0.2 if LAYERS > 1 else 0.0,
            )
            self.fc = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            out, _ = self.lstm(x)          # (batch, seq_len, hidden)
            return self.fc(out[:, -1, :]).squeeze(-1)

    # ── Training ───────────────────────────────────────────────────────────────
    Xt       = torch.tensor(X_train_n[:, :, None])   # (N, 24, 1)
    yt       = torch.tensor(y_train)
    dataset  = TensorDataset(Xt, yt)
    loader   = DataLoader(dataset, batch_size=16, shuffle=True)

    model     = LSTMRegressor()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"  Training for {EPOCHS} epochs (batch_size=16, hidden={HIDDEN})...")
    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:>3}/{EPOCHS}  loss={loss.item():.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_train = np.expm1(model(Xt).numpy())
        Xte        = torch.tensor(X_test_n[:, :, None])
        pred_test  = np.expm1(model(Xte).numpy())

    rmse_tr = rmse(np.expm1(y_train.astype(float)), pred_train)
    rmse_te = rmse(y_test_raw, pred_test)
    mae_te  = float(mean_absolute_error(y_test_raw, pred_test))
    r2_te   = float(r2_score(y_test_raw, pred_test))

    print(f"  Train RMSE={rmse_tr:.2f}  Test RMSE={rmse_te:.2f}  MAE={mae_te:.2f}  R2={r2_te:.3f}")
    print(f"  n_train={train_mask.sum()}  n_test={test_mask.sum()}")

    # ── Save model + normalisation stats ──────────────────────────────────────
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mu":         float(mu),
            "sig":        float(sig),
            "seq_len":    SEQ_LEN,
            "hidden":     HIDDEN,
            "layers":     LAYERS,
        },
        os.path.join(MODELS_DIR, "lstm_per_meal.pt"),
    )
    record(
        "lstm_per_meal", rmse_tr, rmse_te, mae_te, r2_te,
        int(train_mask.sum()), int(test_mask.sum()),
        notes=f"SEQ_LEN={SEQ_LEN} hidden={HIDDEN} layers={LAYERS}",
    )


# ── SARIMAX hourly ─────────────────────────────────────────────────────────────
def train_sarimax():
    print("Training SARIMAX (hourly) - this may take several minutes...")
    try:
        import pickle
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("  statsmodels not available, skipping SARIMAX.")
        return

    print("  Loading hourly features...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "hourly_features.csv"), parse_dates=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df = _apply_date_filter(df)

    EXOG_COLS = ["carb_load_decayed", "walk_min_last_2h", "glipizide_active", "is_fasting", "hour_sin", "hour_cos"]
    split_ts  = df["ts"].iloc[-TRAIN_HOLD_DAYS * 24]
    train = df[df["ts"] <  split_ts].copy()
    test  = df[df["ts"] >= split_ts].copy()

    y_train = train.set_index("ts")["glucose_hourly_mean"].astype(float)
    y_test  = test.set_index("ts")["glucose_hourly_mean"].astype(float).dropna()
    X_train = train.set_index("ts")[EXOG_COLS].astype(float)
    X_test  = test.set_index("ts")[EXOG_COLS].astype(float)
    print(f"  n_train={len(y_train.dropna())} hourly obs  n_test={len(y_test)}")

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

    print(f"  Fitting SARIMAX{order}x{seasonal_order} — this takes 3–5 min...")
    model = SARIMAX(
        y_train, exog=X_train,
        order=order, seasonal_order=seasonal_order,
        enforce_stationarity=True, enforce_invertibility=True,
        simple_differencing=simple_diff,
    )
    result = model.fit(disp=False, maxiter=300)
    print("  Fit complete. Computing test forecast...")

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


# ── LSTM 5-min next-reading forecaster ────────────────────────────────────────
def train_lstm_5min():
    """
    LSTM trained on the 5-min feature matrix to predict the next CGM reading
    (5 minutes ahead).  Reads 5min_features.csv, builds 3-hour rolling windows,
    evaluates vs persistence baseline.  Requires PyTorch.
    """
    SEQ_LEN     = 36    # 36 × 5 min = 3-hour input window
    HIDDEN      = 64
    LAYERS      = 2
    DROPOUT     = 0.2
    EPOCHS      = 200
    PATIENCE    = 20
    BATCH_SIZE  = 64
    SEED        = 42

    FEAT_COLS     = LSTM_FEAT_COLS
    N_FEAT        = len(FEAT_COLS)
    MIN_TRAIN_SEQ = 500

    print("Training LSTM 5-min forecaster...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    except ImportError:
        print("  PyTorch not installed - skipping.  Run: pip install torch")
        return

    feat_path = os.path.join(PROCESSED_DIR, "5min_features.csv")
    if not os.path.exists(feat_path):
        print(f"  {feat_path} not found - run 02-build-5min.py first.  Skipping.")
        return

    print("  Loading 5-min feature matrix...")
    df = pd.read_csv(feat_path, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    df = _apply_date_filter(df)
    print(f"  {len(df):,} rows — building rolling sequences (SEQ_LEN={SEQ_LEN})...")

    missing_cols = [c for c in FEAT_COLS if c not in df.columns]
    if missing_cols:
        print(f"  Missing feature columns: {missing_cols}.  Skipping.")
        return

    # Fill NaN in all feature columns before building sequences.
    # "time since" and "amount in window" features are legitimately 0 when no
    # prior event exists (e.g. minutes_since_last_walk at the start of the day).
    # Leaving them NaN causes NaN loss on the first batch and breaks early stopping.
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    # ── Build rolling sequences ────────────────────────────────────────────────
    _5MIN = pd.Timedelta(minutes=5)
    sequences, targets, seq_ts = [], [], []

    for i in range(SEQ_LEN, len(df) - 1):
        window = df.iloc[i - SEQ_LEN : i + 2]
        if window["glucose_mg_dl"].isna().any():
            continue
        if not (window["ts"].diff().dropna() == _5MIN).all():
            continue
        sequences.append(window.iloc[:-1][FEAT_COLS].values.astype(np.float32))
        targets.append(float(window.iloc[-1]["glucose_mg_dl"]))
        seq_ts.append(window.iloc[-1]["ts"])

    n = len(sequences)
    print(f"  {n:,} valid sequences built.")
    if n < MIN_TRAIN_SEQ:
        print(f"  Need ≥{MIN_TRAIN_SEQ} sequences to train. Skipping.")
        return

    X_raw = np.stack(sequences)
    y_raw = np.array(targets, dtype=np.float32)

    # ── Persistence baseline on full set ──────────────────────────────────────
    glucose_idx   = FEAT_COLS.index("glucose_mg_dl")
    persist_pred  = X_raw[:, -1, glucose_idx]
    persist_rmse  = float(np.sqrt(np.mean((y_raw - persist_pred) ** 2)))
    print(f"  Persistence RMSE (all data): {persist_rmse:.2f} mg/dL")

    # ── Temporal train / test split ────────────────────────────────────────────
    split_ts   = max(seq_ts) - pd.Timedelta(days=TRAIN_HOLD_DAYS)
    train_mask = np.array([t <= split_ts for t in seq_ts])
    test_mask  = ~train_mask

    if train_mask.sum() < MIN_TRAIN_SEQ or test_mask.sum() < 50:
        print("  Not enough sequences after temporal split. Skipping.")
        return

    X_train_raw, X_test_raw = X_raw[train_mask], X_raw[test_mask]
    y_train_raw, y_test_raw = y_raw[train_mask], y_raw[test_mask]

    # Per-feature normalisation (training stats only)
    feat_mu  = X_train_raw.reshape(-1, N_FEAT).mean(axis=0)
    feat_sig = X_train_raw.reshape(-1, N_FEAT).std(axis=0) + 1e-6
    X_train  = (X_train_raw - feat_mu) / feat_sig
    X_test   = (X_test_raw  - feat_mu) / feat_sig

    y_mu  = float(y_train_raw.mean())
    y_sig = float(y_train_raw.std()) + 1e-6
    y_train_n = (y_train_raw - y_mu) / y_sig

    # ── Model ─────────────────────────────────────────────────────────────────
    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N_FEAT, HIDDEN, LAYERS, batch_first=True,
                                dropout=DROPOUT if LAYERS > 1 else 0.0)
            self.fc = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = LSTMForecaster()

    # ── Train / val split (last 15% of training by time) ──────────────────────
    val_n   = max(10, int(len(X_train) * 0.15))
    Xt_sub  = torch.tensor(X_train[:-val_n])
    yt_sub  = torch.tensor(y_train_n[:-val_n])
    Xv      = torch.tensor(X_train[-val_n:])
    yv      = torch.tensor(y_train_n[-val_n:])

    loader = DataLoader(
        TensorDataset(Xt_sub, yt_sub), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    # Initialise best_state to the model's starting weights so that if early
    # stopping fires before any epoch improves (e.g. NaN loss on epoch 1),
    # load_state_dict never receives None.
    best_val   = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    wait       = 0

    print(f"  Training for up to {EPOCHS} epochs (patience={PATIENCE}, "
          f"batch={BATCH_SIZE}, hidden={HIDDEN})...")
    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xv), yv).item()
        model.train()

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch + 1:>3}/{EPOCHS}  val={vl:.4f}")

    model.load_state_dict(best_state)

    # ── Evaluate on test set ───────────────────────────────────────────────────
    model.eval()
    Xte = torch.tensor(X_test)
    with torch.no_grad():
        pred_te = model(Xte).numpy() * y_sig + y_mu
        Xt_all  = torch.tensor(X_train)
        pred_tr = model(Xt_all).numpy() * y_sig + y_mu

    persist_test = X_test_raw[:, -1, glucose_idx]
    persist_rmse_test = float(np.sqrt(np.mean((y_test_raw - persist_test) ** 2)))

    rmse_tr = rmse(y_train_raw, pred_tr)
    rmse_te = rmse(y_test_raw, pred_te)
    mae_te  = float(mean_absolute_error(y_test_raw, pred_te))
    r2_te   = float(r2_score(y_test_raw, pred_te))

    print(f"  Train RMSE={rmse_tr:.2f}  Test RMSE={rmse_te:.2f}  "
          f"MAE={mae_te:.2f}  R2={r2_te:.3f}")
    print(f"  Persistence test RMSE={persist_rmse_test:.2f}  "
          f"Improvement={persist_rmse_test - rmse_te:+.2f} mg/dL")
    print(f"  n_train={train_mask.sum():,}  n_test={test_mask.sum():,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feat_mu":    feat_mu.tolist(),
            "feat_sig":   feat_sig.tolist(),
            "y_mu":       y_mu,
            "y_sig":      y_sig,
            "feat_cols":  FEAT_COLS,
            "seq_len":    SEQ_LEN,
            "hidden":     HIDDEN,
            "layers":     LAYERS,
        },
        os.path.join(MODELS_DIR, "lstm_5min.pt"),
    )
    record(
        "lstm_5min_forecaster", rmse_tr, rmse_te, mae_te, r2_te,
        int(train_mask.sum()), int(test_mask.sum()),
        notes=f"SEQ_LEN={SEQ_LEN} hidden={HIDDEN} layers={LAYERS} "
              f"persistence_test={persist_rmse_test:.2f}",
    )


# ── LSTM 3h direct forecaster ──────────────────────────────────────────────────
def train_lstm_3h():
    """
    LSTM that directly predicts glucose 3 hours ahead (no recursive chaining).
    Input: 3-hour window of 5-min CGM readings + behaviour features.
    Target: glucose_mg_dl at T+3h (absolute value; delta computed at inference time).
    Requires PyTorch.
    """
    HORIZON     = 36    # 36 × 5 min = 3 h ahead
    SEQ_LEN     = 36    # 36 × 5 min = 3-hour input window
    HIDDEN      = 64
    LAYERS      = 2
    DROPOUT     = 0.2
    EPOCHS      = 200
    PATIENCE    = 20
    BATCH_SIZE  = 64
    SEED        = 42
    MIN_TRAIN_SEQ = 500

    FEAT_COLS = LSTM_FEAT_COLS
    N_FEAT    = len(FEAT_COLS)

    print("Training LSTM 3h direct forecaster...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    except ImportError:
        print("  PyTorch not installed - skipping.  Run: pip install torch")
        return

    feat_path = os.path.join(PROCESSED_DIR, "5min_features.csv")
    if not os.path.exists(feat_path):
        print(f"  {feat_path} not found - run 02-build-5min.py first.  Skipping.")
        return

    print("  Loading 5-min feature matrix...")
    df = pd.read_csv(feat_path, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    df = _apply_date_filter(df)
    print(f"  {len(df):,} rows — building sequences (SEQ_LEN={SEQ_LEN}, HORIZON={HORIZON} steps)...")

    missing_cols = [c for c in FEAT_COLS if c not in df.columns]
    if missing_cols:
        print(f"  Missing feature columns: {missing_cols}.  Skipping.")
        return

    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    # ── Build sequences ────────────────────────────────────────────────────────
    _5MIN = pd.Timedelta(minutes=5)
    sequences, targets, seq_ts = [], [], []

    for i in range(SEQ_LEN, len(df) - HORIZON - 1):
        input_slice = df.iloc[i - SEQ_LEN : i]
        target_row  = df.iloc[i + HORIZON]

        # Input window must be gap-free
        if input_slice["glucose_mg_dl"].isna().any():
            continue
        if not (input_slice["ts"].diff().dropna() == _5MIN).all():
            continue

        # Target must exist and be ~3h after the window end (±15 min)
        if pd.isna(target_row["glucose_mg_dl"]):
            continue
        actual_gap   = target_row["ts"] - input_slice.iloc[-1]["ts"]
        expected_gap = pd.Timedelta(minutes=HORIZON * 5)
        if abs(actual_gap - expected_gap) > pd.Timedelta(minutes=15):
            continue

        sequences.append(input_slice[FEAT_COLS].values.astype(np.float32))
        targets.append(float(target_row["glucose_mg_dl"]))
        seq_ts.append(target_row["ts"])   # split on target timestamp

    n = len(sequences)
    print(f"  {n:,} valid sequences built.")
    if n < MIN_TRAIN_SEQ:
        print(f"  Need ≥{MIN_TRAIN_SEQ} sequences to train. Skipping.")
        return

    X_raw = np.stack(sequences)
    y_raw = np.array(targets, dtype=np.float32)

    # ── Baselines ─────────────────────────────────────────────────────────────
    glucose_idx       = FEAT_COLS.index("glucose_mg_dl")
    persist_pred      = X_raw[:, -1, glucose_idx]   # "glucose unchanged for 3h"
    persist_rmse_all  = float(np.sqrt(np.mean((y_raw - persist_pred) ** 2)))
    print(f"  Persistence (3h naive) RMSE (all data): {persist_rmse_all:.2f} mg/dL")

    # ── Temporal split ────────────────────────────────────────────────────────
    split_ts   = max(seq_ts) - pd.Timedelta(days=TRAIN_HOLD_DAYS)
    train_mask = np.array([t <= split_ts for t in seq_ts])
    test_mask  = ~train_mask

    if train_mask.sum() < MIN_TRAIN_SEQ or test_mask.sum() < 50:
        print("  Not enough sequences after temporal split. Skipping.")
        return

    X_train_raw, X_test_raw = X_raw[train_mask], X_raw[test_mask]
    y_train_raw, y_test_raw = y_raw[train_mask], y_raw[test_mask]

    # Per-feature normalisation (training stats only)
    feat_mu  = X_train_raw.reshape(-1, N_FEAT).mean(axis=0)
    feat_sig = X_train_raw.reshape(-1, N_FEAT).std(axis=0) + 1e-6
    X_train  = (X_train_raw - feat_mu) / feat_sig
    X_test   = (X_test_raw  - feat_mu) / feat_sig

    y_mu      = float(y_train_raw.mean())
    y_sig     = float(y_train_raw.std()) + 1e-6
    y_train_n = (y_train_raw - y_mu) / y_sig

    # ── Model ─────────────────────────────────────────────────────────────────
    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N_FEAT, HIDDEN, LAYERS, batch_first=True,
                                dropout=DROPOUT if LAYERS > 1 else 0.0)
            self.fc = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = LSTMForecaster()

    # ── Train / val split (last 15% of training by time) ──────────────────────
    val_n  = max(10, int(len(X_train) * 0.15))
    Xt_sub = torch.tensor(X_train[:-val_n])
    yt_sub = torch.tensor(y_train_n[:-val_n])
    Xv     = torch.tensor(X_train[-val_n:])
    yv     = torch.tensor(y_train_n[-val_n:])

    loader = DataLoader(
        TensorDataset(Xt_sub, yt_sub), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    optimiser  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion  = nn.MSELoss()
    best_val   = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    wait       = 0

    print(f"  Training for up to {EPOCHS} epochs (patience={PATIENCE}, "
          f"batch={BATCH_SIZE}, hidden={HIDDEN}, features={N_FEAT})...")
    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xv), yv).item()
        model.train()

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch + 1:>3}/{EPOCHS}  val={vl:.4f}")

    model.load_state_dict(best_state)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("  Evaluating on test set...")
    model.eval()
    Xte = torch.tensor(X_test)
    with torch.no_grad():
        pred_te = model(Xte).numpy() * y_sig + y_mu
        pred_tr = model(torch.tensor(X_train)).numpy() * y_sig + y_mu

    persist_test      = X_test_raw[:, -1, glucose_idx]
    persist_rmse_test = float(np.sqrt(np.mean((y_test_raw - persist_test) ** 2)))

    rmse_tr = rmse(y_train_raw, pred_tr)
    rmse_te = rmse(y_test_raw, pred_te)
    mae_te  = float(mean_absolute_error(y_test_raw, pred_te))
    r2_te   = float(r2_score(y_test_raw, pred_te))

    print(f"  Train RMSE={rmse_tr:.2f}  Test RMSE={rmse_te:.2f}  "
          f"MAE={mae_te:.2f}  R2={r2_te:.3f}")
    print(f"  Persistence (3h naive) test RMSE={persist_rmse_test:.2f}  "
          f"Improvement={persist_rmse_test - rmse_te:+.2f} mg/dL")
    print(f"  n_train={train_mask.sum():,}  n_test={test_mask.sum():,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feat_mu":    feat_mu.tolist(),
            "feat_sig":   feat_sig.tolist(),
            "y_mu":       y_mu,
            "y_sig":      y_sig,
            "feat_cols":  FEAT_COLS,
            "seq_len":    SEQ_LEN,
            "horizon":    HORIZON,
            "hidden":     HIDDEN,
            "layers":     LAYERS,
        },
        os.path.join(MODELS_DIR, "lstm_3h.pt"),
    )
    record(
        "lstm_3h_forecaster", rmse_tr, rmse_te, mae_te, r2_te,
        int(train_mask.sum()), int(test_mask.sum()),
        notes=f"horizon={HORIZON}steps SEQ_LEN={SEQ_LEN} hidden={HIDDEN} "
              f"layers={LAYERS} persistence_test={persist_rmse_test:.2f}",
    )


# ── LSTM Trajectory forecaster — 60-min lookback → full 3h curve ──────────────
# Shared exog column list (must stay in sync with predictor.py TRAJ_EXOG_COLS)
TRAJ_EXOG_COLS = [
    "carb_load_decayed", "walk_min_last_2h", "glipizide_active",
    "is_fasting", "hour_sin", "hour_cos",
]


def train_lstm_trajectory():
    """
    Direct multi-output LSTM: given the last 60 min of CGM history + current
    exog state, predict glucose at every 5-min interval for the next 3 hours
    (36 outputs, all in one forward pass — no autoregressive chaining).

    Input window:  SEQ_LEN=12 × 5-min features (same LSTM_FEAT_COLS as lstm_5min)
    Exog scalars:  TRAJ_EXOG_COLS (carb_load, walk, glip, fasting, hour sin/cos)
    Output:        36 absolute glucose values (t+5 min … t+3 h)
    Requires PyTorch.
    """
    SEQ_LEN       = 12    # 12 × 5 min = 60 min lookback
    HORIZON       = 36    # 36 × 5 min = 3 h trajectory
    HIDDEN        = 128
    LAYERS        = 2
    DROPOUT       = 0.2
    EPOCHS        = 200
    PATIENCE      = 20
    BATCH_SIZE    = 64
    SEED          = 42
    MIN_TRAIN_SEQ = 500

    FEAT_COLS = LSTM_FEAT_COLS
    EXOG_COLS = TRAJ_EXOG_COLS
    N_FEAT    = len(FEAT_COLS)
    N_EXOG    = len(EXOG_COLS)

    print("Training LSTM Trajectory (60-min lookback → 36-step 3h forecast)...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    except ImportError:
        print("  PyTorch not installed - skipping.  Run: pip install torch")
        return

    feat_path = os.path.join(PROCESSED_DIR, "5min_features.csv")
    if not os.path.exists(feat_path):
        print(f"  {feat_path} not found - run 02-build-5min.py first.  Skipping.")
        return

    print("  Loading 5-min feature matrix...")
    df = pd.read_csv(feat_path, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    df = _apply_date_filter(df)
    print(f"  {len(df):,} rows — building sliding windows (SEQ_LEN={SEQ_LEN} × HORIZON={HORIZON})...")

    missing = [c for c in FEAT_COLS + EXOG_COLS if c not in df.columns]
    if missing:
        print(f"  Missing columns: {missing}.  Skipping.")
        return

    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)
    df[EXOG_COLS] = df[EXOG_COLS].fillna(0.0)

    # ── Build sliding windows ──────────────────────────────────────────────────
    _5MIN = pd.Timedelta(minutes=5)
    sequences, exogs, targets, seq_ts = [], [], [], []

    for i in range(SEQ_LEN, len(df) - HORIZON):
        inp = df.iloc[i - SEQ_LEN : i]          # past 60 min
        out = df.iloc[i : i + HORIZON]           # next 3 h

        # Require gap-free consecutive 5-min readings in BOTH windows
        if inp["glucose_mg_dl"].isna().any():
            continue
        if not (inp["ts"].diff().dropna() == _5MIN).all():
            continue
        if out["glucose_mg_dl"].isna().any():
            continue
        if not (out["ts"].diff().dropna() == _5MIN).all():
            continue

        sequences.append(inp[FEAT_COLS].values.astype(np.float32))
        exogs.append(inp.iloc[-1][EXOG_COLS].values.astype(np.float32))  # state at anchor
        targets.append(out["glucose_mg_dl"].values.astype(np.float32))   # (36,)
        seq_ts.append(inp.iloc[-1]["ts"])

    n = len(sequences)
    print(f"  {n:,} valid windows built (SEQ_LEN={SEQ_LEN} × HORIZON={HORIZON}).")
    if n < MIN_TRAIN_SEQ:
        print(f"  Need ≥{MIN_TRAIN_SEQ} windows to train.  Skipping.")
        return

    X_raw = np.stack(sequences)   # (N, SEQ_LEN, N_FEAT)
    E_raw = np.stack(exogs)       # (N, N_EXOG)
    y_raw = np.stack(targets)     # (N, HORIZON)

    # ── Temporal train / test split ────────────────────────────────────────────
    split_ts   = max(seq_ts) - pd.Timedelta(days=TRAIN_HOLD_DAYS)
    train_mask = np.array([t <= split_ts for t in seq_ts])
    test_mask  = ~train_mask

    if train_mask.sum() < MIN_TRAIN_SEQ or test_mask.sum() < 50:
        print("  Not enough windows after temporal split.  Skipping.")
        return

    X_tr_r, X_te_r = X_raw[train_mask], X_raw[test_mask]
    E_tr_r, E_te_r = E_raw[train_mask], E_raw[test_mask]
    y_tr_r, y_te_r = y_raw[train_mask], y_raw[test_mask]

    # Per-feature normalisation (training stats only)
    feat_mu  = X_tr_r.reshape(-1, N_FEAT).mean(axis=0)
    feat_sig = X_tr_r.reshape(-1, N_FEAT).std(axis=0) + 1e-6
    X_tr = (X_tr_r - feat_mu) / feat_sig
    X_te = (X_te_r - feat_mu) / feat_sig

    exog_mu  = E_tr_r.mean(axis=0)
    exog_sig = E_tr_r.std(axis=0) + 1e-6
    E_tr = (E_tr_r - exog_mu) / exog_sig
    E_te = (E_te_r - exog_mu) / exog_sig

    # Output normalisation: use same stats as glucose_mg_dl input feature
    glucose_idx = FEAT_COLS.index("glucose_mg_dl")
    y_mu  = float(X_tr_r[:, :, glucose_idx].mean())
    y_sig = float(X_tr_r[:, :, glucose_idx].std()) + 1e-6
    y_tr_n = (y_tr_r - y_mu) / y_sig

    # ── Model definition ───────────────────────────────────────────────────────
    class LSTMTrajectory(nn.Module):
        """LSTM encoder (seq) + exog scalars → dense head → 36 glucose values."""
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N_FEAT, HIDDEN, LAYERS, batch_first=True,
                                dropout=DROPOUT if LAYERS > 1 else 0.0)
            self.head = nn.Linear(HIDDEN + N_EXOG, HORIZON)

        def forward(self, x_seq, x_exog):
            _, (h, _) = self.lstm(x_seq)
            return self.head(torch.cat([h[-1], x_exog], dim=1))

    model = LSTMTrajectory()

    # ── Train / val split (last 15% of training by time) ──────────────────────
    val_n  = max(10, int(len(X_tr) * 0.15))
    Xt_sub = torch.tensor(X_tr[:-val_n])
    Et_sub = torch.tensor(E_tr[:-val_n])
    yt_sub = torch.tensor(y_tr_n[:-val_n])
    Xv     = torch.tensor(X_tr[-val_n:])
    Ev     = torch.tensor(E_tr[-val_n:])
    yv     = torch.tensor(y_tr_n[-val_n:])

    loader = DataLoader(
        TensorDataset(Xt_sub, Et_sub, yt_sub), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimiser  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion  = nn.MSELoss()
    best_val   = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    wait       = 0

    print(f"  Training for up to {EPOCHS} epochs (patience={PATIENCE}, "
          f"batch={BATCH_SIZE}, hidden={HIDDEN}, seq={SEQ_LEN}→{HORIZON})...")
    model.train()
    for epoch in range(EPOCHS):
        for xb, eb, yb in loader:
            optimiser.zero_grad()
            criterion(model(xb, eb), yb).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xv, Ev), yv).item()
        model.train()

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch + 1:>3}/{EPOCHS}  val={vl:.4f}")

    model.load_state_dict(best_state)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_te_n = model(torch.tensor(X_te), torch.tensor(E_te)).numpy()
        pred_tr_n = model(torch.tensor(X_tr), torch.tensor(E_tr)).numpy()

    pred_te = pred_te_n * y_sig + y_mu   # (N_test, 36)
    pred_tr = pred_tr_n * y_sig + y_mu

    rmse_tr = float(np.sqrt(np.mean((y_tr_r - pred_tr) ** 2)))
    rmse_te = float(np.sqrt(np.mean((y_te_r - pred_te) ** 2)))
    mae_te  = float(np.mean(np.abs(y_te_r - pred_te)))
    r2_te   = float(r2_score(y_te_r.flatten(), pred_te.flatten()))

    # Per-step RMSE for diagnostics (shows how error grows with horizon)
    step_rmse = np.sqrt(np.mean((y_te_r - pred_te) ** 2, axis=0))
    print(f"  Per-step RMSE — step1={step_rmse[0]:.1f}  step18={step_rmse[17]:.1f}  "
          f"step36={step_rmse[-1]:.1f}  mean={step_rmse.mean():.1f} mg/dL")

    # Persistence baseline (flat line at last known glucose)
    persist   = np.stack([X_te_r[:, -1, glucose_idx]] * HORIZON, axis=1)
    p_rmse_te = float(np.sqrt(np.mean((y_te_r - persist) ** 2)))

    print(f"  Train RMSE={rmse_tr:.2f}  Test RMSE={rmse_te:.2f}  "
          f"MAE={mae_te:.2f}  R2={r2_te:.3f}")
    print(f"  Persistence (flat) test RMSE={p_rmse_te:.2f}  "
          f"Improvement={p_rmse_te - rmse_te:+.2f} mg/dL")
    print(f"  n_train={train_mask.sum():,}  n_test={test_mask.sum():,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feat_mu":    feat_mu.tolist(),
            "feat_sig":   feat_sig.tolist(),
            "exog_mu":    exog_mu.tolist(),
            "exog_sig":   exog_sig.tolist(),
            "y_mu":       y_mu,
            "y_sig":      y_sig,
            "feat_cols":  FEAT_COLS,
            "exog_cols":  EXOG_COLS,
            "seq_len":    SEQ_LEN,
            "horizon":    HORIZON,
            "hidden":     HIDDEN,
            "layers":     LAYERS,
        },
        os.path.join(MODELS_DIR, "lstm_trajectory.pt"),
    )
    record(
        "lstm_trajectory", rmse_tr, rmse_te, mae_te, r2_te,
        int(train_mask.sum()), int(test_mask.sum()),
        notes=f"SEQ_LEN={SEQ_LEN} HORIZON={HORIZON} hidden={HIDDEN} layers={LAYERS} "
              f"persistence_test={p_rmse_te:.2f}",
    )


# ── LSTM Trajectory V2 — per-step future exog encoder ────────────────────────
# Future exog: same 6 columns as TRAJ_EXOG_COLS but read from FUTURE rows so the
# model sees carb_load DECAYING, walk activity entering/leaving the window, etc.
FUTURE_EXOG_COLS = TRAJ_EXOG_COLS   # same list; "future" refers to which rows we read


def train_lstm_trajectory_v2():
    """
    Two-encoder LSTM: past CGM (12-step lookback) + future exog sequence (36 steps)
    → 36-step glucose trajectory.

    V1 feeds the exog at the anchor point only (static snapshot).
    V2 feeds the full 36×6 future exog matrix, giving the model explicit visibility
    into how carb load decays, when the walk effect dissipates, and how the time
    of day shifts across the 3-hour horizon — enabling rise-then-fall predictions.

    Saved as  models/lstm_trajectory_v2.pt  (separate from v1).
    Requires PyTorch.
    """
    SEQ_LEN        = 12    # 60 min past lookback
    HORIZON        = 36    # 3 h future trajectory
    HIDDEN_PAST    = 128
    HIDDEN_FUTURE  = 32
    LAYERS         = 2
    DROPOUT        = 0.2
    EPOCHS         = 200
    PATIENCE       = 20
    BATCH_SIZE     = 64
    SEED           = 42
    MIN_TRAIN_SEQ  = 500

    FEAT_COLS    = LSTM_FEAT_COLS
    F_EXOG_COLS  = FUTURE_EXOG_COLS
    N_FEAT       = len(FEAT_COLS)
    N_FUTURE_EXO = len(F_EXOG_COLS)

    print("Training LSTM Trajectory V2 (past encoder + future-exog encoder → 36-step trajectory)...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    except ImportError:
        print("  PyTorch not installed - skipping.  Run: pip install torch")
        return

    feat_path = os.path.join(PROCESSED_DIR, "5min_features.csv")
    if not os.path.exists(feat_path):
        print(f"  {feat_path} not found - run 02-build-5min.py first.  Skipping.")
        return

    print("  Loading 5-min feature matrix...")
    df = pd.read_csv(feat_path, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    df = _apply_date_filter(df)

    missing = [c for c in FEAT_COLS + F_EXOG_COLS if c not in df.columns]
    if missing:
        print(f"  Missing columns: {missing}.  Skipping.")
        return

    df[FEAT_COLS]   = df[FEAT_COLS].fillna(0.0)
    df[F_EXOG_COLS] = df[F_EXOG_COLS].fillna(0.0)
    print(f"  {len(df):,} rows — building sliding windows (SEQ_LEN={SEQ_LEN} × HORIZON={HORIZON})...")

    # ── Build sliding windows ──────────────────────────────────────────────────
    _5MIN = pd.Timedelta(minutes=5)
    sequences, future_exogs, targets, seq_ts = [], [], [], []

    for i in range(SEQ_LEN, len(df) - HORIZON):
        inp = df.iloc[i - SEQ_LEN : i]      # past 60 min
        out = df.iloc[i : i + HORIZON]       # next 3 h

        if inp["glucose_mg_dl"].isna().any():
            continue
        if not (inp["ts"].diff().dropna() == _5MIN).all():
            continue
        if out["glucose_mg_dl"].isna().any():
            continue
        if not (out["ts"].diff().dropna() == _5MIN).all():
            continue

        sequences.append(inp[FEAT_COLS].values.astype(np.float32))
        # Future exog: read from the FUTURE rows — carb load is already decayed at
        # each future timestamp, walk_min is correctly computed for that future window.
        future_exogs.append(out[F_EXOG_COLS].values.astype(np.float32))   # (HORIZON, 6)
        targets.append(out["glucose_mg_dl"].values.astype(np.float32))     # (HORIZON,)
        seq_ts.append(inp.iloc[-1]["ts"])

    n = len(sequences)
    print(f"  {n:,} valid windows built.")
    if n < MIN_TRAIN_SEQ:
        print(f"  Need ≥{MIN_TRAIN_SEQ} windows to train.  Skipping.")
        return

    X_raw = np.stack(sequences)      # (N, SEQ_LEN, N_FEAT)
    F_raw = np.stack(future_exogs)   # (N, HORIZON, N_FUTURE_EXO)
    y_raw = np.stack(targets)        # (N, HORIZON)

    # ── Temporal split ────────────────────────────────────────────────────────
    split_ts   = max(seq_ts) - pd.Timedelta(days=TRAIN_HOLD_DAYS)
    train_mask = np.array([t <= split_ts for t in seq_ts])
    test_mask  = ~train_mask

    if train_mask.sum() < MIN_TRAIN_SEQ or test_mask.sum() < 50:
        print("  Not enough windows after temporal split.  Skipping.")
        return

    X_tr_r, X_te_r = X_raw[train_mask], X_raw[test_mask]
    F_tr_r, F_te_r = F_raw[train_mask], F_raw[test_mask]
    y_tr_r, y_te_r = y_raw[train_mask], y_raw[test_mask]

    # Per-feature normalisation — past sequence
    feat_mu  = X_tr_r.reshape(-1, N_FEAT).mean(axis=0)
    feat_sig = X_tr_r.reshape(-1, N_FEAT).std(axis=0)  + 1e-6
    X_tr = (X_tr_r - feat_mu) / feat_sig
    X_te = (X_te_r - feat_mu) / feat_sig

    # Per-feature normalisation — future exog (pool across all time steps + windows)
    f_exog_mu  = F_tr_r.reshape(-1, N_FUTURE_EXO).mean(axis=0)
    f_exog_sig = F_tr_r.reshape(-1, N_FUTURE_EXO).std(axis=0) + 1e-6
    F_tr = (F_tr_r - f_exog_mu) / f_exog_sig
    F_te = (F_te_r - f_exog_mu) / f_exog_sig

    # Output normalisation: same glucose stats as past-sequence glucose feature
    glucose_idx = FEAT_COLS.index("glucose_mg_dl")
    y_mu  = float(X_tr_r[:, :, glucose_idx].mean())
    y_sig = float(X_tr_r[:, :, glucose_idx].std()) + 1e-6
    y_tr_n = (y_tr_r - y_mu) / y_sig

    # ── Model definition ───────────────────────────────────────────────────────
    class LSTMTrajectoryV2(nn.Module):
        """Past LSTM encoder + future-exog LSTM encoder → 36-step glucose head."""

        def __init__(self):
            super().__init__()
            self.past_lstm   = nn.LSTM(N_FEAT, HIDDEN_PAST, LAYERS, batch_first=True,
                                        dropout=DROPOUT if LAYERS > 1 else 0.0)
            self.future_lstm = nn.LSTM(N_FUTURE_EXO, HIDDEN_FUTURE, 1, batch_first=True)
            self.head        = nn.Linear(HIDDEN_PAST + HIDDEN_FUTURE, HORIZON)

        def forward(self, x_past, x_future):
            # x_past:   (B, SEQ_LEN, N_FEAT)
            # x_future: (B, HORIZON, N_FUTURE_EXO)
            _, (h_past,   _) = self.past_lstm(x_past)
            _, (h_future, _) = self.future_lstm(x_future)
            return self.head(torch.cat([h_past[-1], h_future[-1]], dim=1))

    model = LSTMTrajectoryV2()

    # ── Train / val split ─────────────────────────────────────────────────────
    val_n  = max(10, int(len(X_tr) * 0.15))
    Xt_sub = torch.tensor(X_tr[:-val_n])
    Ft_sub = torch.tensor(F_tr[:-val_n])
    yt_sub = torch.tensor(y_tr_n[:-val_n])
    Xv     = torch.tensor(X_tr[-val_n:])
    Fv     = torch.tensor(F_tr[-val_n:])
    yv     = torch.tensor(y_tr_n[-val_n:])

    loader = DataLoader(
        TensorDataset(Xt_sub, Ft_sub, yt_sub), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimiser  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion  = nn.MSELoss()
    best_val   = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    wait       = 0

    print(f"  Training for up to {EPOCHS} epochs (patience={PATIENCE}, "
          f"batch={BATCH_SIZE}, past_hidden={HIDDEN_PAST}, future_hidden={HIDDEN_FUTURE})...")
    model.train()
    for epoch in range(EPOCHS):
        for xb, fb, yb in loader:
            optimiser.zero_grad()
            criterion(model(xb, fb), yb).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xv, Fv), yv).item()
        model.train()

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch + 1:>3}/{EPOCHS}  val={vl:.4f}")

    model.load_state_dict(best_state)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("  Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        pred_te_n = model(torch.tensor(X_te), torch.tensor(F_te)).numpy()
        pred_tr_n = model(torch.tensor(X_tr), torch.tensor(F_tr)).numpy()

    pred_te = pred_te_n * y_sig + y_mu
    pred_tr = pred_tr_n * y_sig + y_mu

    rmse_tr = float(np.sqrt(np.mean((y_tr_r - pred_tr) ** 2)))
    rmse_te = float(np.sqrt(np.mean((y_te_r - pred_te) ** 2)))
    mae_te  = float(np.mean(np.abs(y_te_r - pred_te)))
    r2_te   = float(r2_score(y_te_r.flatten(), pred_te.flatten()))

    step_rmse = np.sqrt(np.mean((y_te_r - pred_te) ** 2, axis=0))
    print(f"  Per-step RMSE — step1={step_rmse[0]:.1f}  step18={step_rmse[17]:.1f}  "
          f"step36={step_rmse[-1]:.1f}  mean={step_rmse.mean():.1f} mg/dL")

    persist   = np.stack([X_te_r[:, -1, glucose_idx]] * HORIZON, axis=1)
    p_rmse_te = float(np.sqrt(np.mean((y_te_r - persist) ** 2)))

    print(f"  Train RMSE={rmse_tr:.2f}  Test RMSE={rmse_te:.2f}  "
          f"MAE={mae_te:.2f}  R2={r2_te:.3f}")
    print(f"  Persistence test RMSE={p_rmse_te:.2f}  "
          f"Improvement={p_rmse_te - rmse_te:+.2f} mg/dL")
    print(f"  n_train={train_mask.sum():,}  n_test={test_mask.sum():,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(
        {
            "state_dict":  model.state_dict(),
            "feat_mu":     feat_mu.tolist(),
            "feat_sig":    feat_sig.tolist(),
            "f_exog_mu":   f_exog_mu.tolist(),
            "f_exog_sig":  f_exog_sig.tolist(),
            "y_mu":        y_mu,
            "y_sig":       y_sig,
            "feat_cols":   FEAT_COLS,
            "f_exog_cols": F_EXOG_COLS,
            "seq_len":     SEQ_LEN,
            "horizon":     HORIZON,
            "hidden_past":   HIDDEN_PAST,
            "hidden_future": HIDDEN_FUTURE,
            "layers":        LAYERS,
        },
        os.path.join(MODELS_DIR, "lstm_trajectory_v2.pt"),
    )
    record(
        "lstm_trajectory_v2", rmse_tr, rmse_te, mae_te, r2_te,
        int(train_mask.sum()), int(test_mask.sum()),
        notes=f"SEQ_LEN={SEQ_LEN} HORIZON={HORIZON} hidden_past={HIDDEN_PAST} "
              f"hidden_future={HIDDEN_FUTURE} layers={LAYERS} "
              f"persistence_test={p_rmse_te:.2f}",
    )


# ── entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=["ridge", "sarimax", "rf", "lstm", "lstm5min", "lstm3h", "lstmtraj", "lstmtrajv2"],
        default=["ridge"],
        help="Which models to train",
    )
    parser.add_argument(
        "--run-tag", dest="run_tag", default=None,
        help="Short identifier shared across all models trained in this run "
             "(auto-generated as vYYYYMMDD-HHMM if omitted)",
    )
    parser.add_argument(
        "--date-from", dest="date_from", default=None,
        help="Restrict training data to rows on/after this date (YYYY-MM-DD, UTC)",
    )
    parser.add_argument(
        "--date-to", dest="date_to", default=None,
        help="Restrict training data to rows on/before this date (YYYY-MM-DD, UTC)",
    )
    args = parser.parse_args()

    # Set the module-level globals so all train_* functions share the same scope.
    if args.run_tag:
        RUN_TAG = args.run_tag
    else:
        RUN_TAG = "v" + datetime.utcnow().strftime("%Y%m%d-%H%M")
    print(f"Run tag: {RUN_TAG}")

    DATE_FROM = args.date_from or None
    DATE_TO   = args.date_to   or None
    if DATE_FROM or DATE_TO:
        print(f"Data filter: {DATE_FROM or 'beginning'} → {DATE_TO or 'latest'}")

    if "ridge"    in args.models:
        train_ridge()
    if "rf"       in args.models:
        train_rf()
    if "lstm"     in args.models:
        train_lstm()
    if "lstm5min" in args.models:
        train_lstm_5min()
    if "lstm3h"   in args.models:
        train_lstm_3h()
    if "lstmtraj"   in args.models:
        train_lstm_trajectory()
    if "lstmtrajv2" in args.models:
        train_lstm_trajectory_v2()
    if "sarimax"  in args.models:
        train_sarimax()

    print("Done.")
