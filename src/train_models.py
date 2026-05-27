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

TRAIN_HOLD_DAYS = 21

# Shared per-meal feature list (used by Ridge, RF, and LSTM)
PER_MEAL_FEATURES = [
    "pre_meal_glucose", "glucose_velocity_30min", "glucose_60min_before_mean",
    "carbs_g_final", "hour_sin", "hour_cos", "is_weekend", "day_index",
    "walk_min_within_90", "fasted_meal", "prev_fast_hours",
    "minutes_since_last_glipizide",
]
TARGET = "peak_delta"


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


def _load_per_meal():
    """Load and temporally split per_meal_features.csv. Returns (df, train, test, feats)."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "per_meal_features.csv"), parse_dates=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df = df.dropna(subset=[TARGET])
    feats = [f for f in PER_MEAL_FEATURES if f in df.columns]
    split_date = df["ts"].max() - pd.Timedelta(days=TRAIN_HOLD_DAYS)
    train = df[df["ts"] <= split_date]
    test  = df[df["ts"] >  split_date]
    return df, train, test, feats


# ── Ridge per-meal ─────────────────────────────────────────────────────────────
def train_ridge():
    print("Training Ridge (per-meal)...")
    _, train, test, feats = _load_per_meal()

    X_train    = train[feats].values
    y_train    = np.log1p(train[TARGET].clip(lower=0).values)
    X_test     = test[feats].values
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


# ── Random Forest per-meal ─────────────────────────────────────────────────────
def train_rf():
    print("Training Random Forest (per-meal)...")
    from sklearn.ensemble import RandomForestRegressor

    _, train, test, feats = _load_per_meal()

    X_train    = train[feats].values
    y_train    = np.log1p(train[TARGET].clip(lower=0).values)
    X_test     = test[feats].values
    y_test_raw = test[TARGET].clip(lower=0).values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

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

    FEAT_COLS = [
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
    N_FEAT = len(FEAT_COLS)
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

    df = pd.read_csv(feat_path, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    df["carbs_last_meal"] = df["carbs_last_meal"].fillna(0.0)

    missing_cols = [c for c in FEAT_COLS if c not in df.columns]
    if missing_cols:
        print(f"  Missing feature columns: {missing_cols}.  Skipping.")
        return

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
    best_val, best_state, wait = float("inf"), None, 0

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


# ── entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=["ridge", "sarimax", "rf", "lstm", "lstm5min"],
        default=["ridge"],
        help="Which models to train",
    )
    args = parser.parse_args()

    if "ridge"    in args.models:
        train_ridge()
    if "rf"       in args.models:
        train_rf()
    if "lstm"     in args.models:
        train_lstm()
    if "lstm5min" in args.models:
        train_lstm_5min()
    if "sarimax"  in args.models:
        train_sarimax()

    print("Done.")
