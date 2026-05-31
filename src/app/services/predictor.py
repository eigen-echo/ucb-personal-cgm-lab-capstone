"""
Inference service — loads trained models from models/ and runs all-model
3-hour glucose prediction from a feature dict.

Each model returns:
    {"predicted_delta": float, "predicted_glucose_3h": float}
or None  if the model file is absent
or {"error": str}  if loading / inference fails (shown as "—" in the UI).
"""
import math
import os
import pickle
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

ROOT          = os.environ.get("APP_ROOT", "/app")
MODELS_DIR    = os.path.join(ROOT, "models")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

# Must stay in sync with train_models.py
PER_MEAL_FEATURES = [
    "pre_meal_glucose", "glucose_velocity_30min", "glucose_60min_before_mean",
    "carbs_g_final", "hour_sin", "hour_cos", "is_weekend", "day_index",
    "walk_min_within_90", "fasted_meal", "prev_fast_hours",
    "minutes_since_last_glipizide",
]

SARIMAX_EXOG_COLS = [
    "carb_load_decayed", "walk_min_last_2h", "glipizide_active",
    "is_fasting", "hour_sin", "hour_cos",
]

# Same columns used as exog inputs for the LSTM trajectory model.
# Must stay in sync with train_models.py TRAJ_EXOG_COLS.
TRAJ_EXOG_COLS = SARIMAX_EXOG_COLS

_GAP_THRESHOLD_MIN = 30   # readings older than this → cold start

# Future-exog columns used by trajectory V2 (same list; "future" = read from future rows).
FUTURE_EXOG_COLS = TRAJ_EXOG_COLS


# ── Public API ─────────────────────────────────────────────────────────────────

def predict_all(
    db,
    baseline_glucose: float,
    carbs_g: float,
    meal_ts: datetime,
    fasted: bool = False,
    prev_fast_hours: float = 0.0,
    minutes_since_glipizide: float = 999.0,
    walk_min_after: float = 0.0,
) -> dict:
    """
    Run every available model.  Returns:
        {
          "ridge":   {predicted_delta, predicted_glucose_3h} | None | {error},
          "rf":      ...,
          "lstm3h":  ...,
          "sarimax": ...,
        }
    """
    hour       = meal_ts.hour + meal_ts.minute / 60.0
    hour_sin   = math.sin(2 * math.pi * hour / 24)
    hour_cos   = math.cos(2 * math.pi * hour / 24)
    is_weekend = int(meal_ts.weekday() >= 5)

    vel_30min, mean_60min = _glucose_context(db, meal_ts, baseline_glucose)
    day_idx = _day_index(db, meal_ts)

    per_meal_vec = [
        baseline_glucose, vel_30min, mean_60min,
        carbs_g,
        hour_sin, hour_cos, is_weekend, day_idx,
        walk_min_after, # walk_min_within_90 — user-supplied planned walk (default 0)
        float(fasted),
        prev_fast_hours,
        minutes_since_glipizide,
    ]

    results = {}

    # ── Ridge ─────────────────────────────────────────────────────────────────
    p = os.path.join(MODELS_DIR, "per_meal_model.joblib")
    if os.path.exists(p):
        try:
            pipe  = joblib.load(p)
            delta = float(np.expm1(pipe.predict([per_meal_vec])[0]))
            results["ridge"] = _pack(delta, baseline_glucose)
        except Exception as exc:
            results["ridge"] = {"error": str(exc)}
    else:
        results["ridge"] = None

    # ── Random Forest ─────────────────────────────────────────────────────────
    p = os.path.join(MODELS_DIR, "rf_per_meal.joblib")
    if os.path.exists(p):
        try:
            b     = joblib.load(p)
            vec   = b["imputer"].transform([per_meal_vec])
            delta = float(np.expm1(b["model"].predict(vec)[0]))
            results["rf"] = _pack(delta, baseline_glucose)
        except Exception as exc:
            results["rf"] = {"error": str(exc)}
    else:
        results["rf"] = None

    # ── LSTM 3h ────────────────────────────────────────────────────────────────
    p = os.path.join(MODELS_DIR, "lstm_3h.pt")
    if os.path.exists(p):
        results["lstm3h"] = _predict_lstm3h(p, baseline_glucose)
    else:
        results["lstm3h"] = None

    # ── SARIMAX ────────────────────────────────────────────────────────────────
    p = os.path.join(MODELS_DIR, "sarimax_hourly.pkl")
    if os.path.exists(p):
        results["sarimax"] = _predict_sarimax(
            p, meal_ts, baseline_glucose, carbs_g, minutes_since_glipizide,
            walk_min_after=walk_min_after,
        )
    else:
        results["sarimax"] = None

    return results


# ── Private helpers ────────────────────────────────────────────────────────────

def _pack(delta: float, baseline: float) -> dict:
    """Ridge & RF are trained on peak_delta, so predicted_delta == predicted_peak_delta."""
    return {
        "predicted_delta":      round(delta, 1),
        "predicted_glucose_3h": round(baseline + delta, 1),
        "predicted_peak_delta": round(delta, 1),  # same — these models predict the peak
    }


def _glucose_context(db, meal_ts: datetime, baseline: float):
    """Return (velocity_30min, mean_60min) from recent CGM readings."""
    from app.models.cgm import CGMReading
    rows = (
        db.query(CGMReading)
        .filter(CGMReading.ts >= meal_ts - timedelta(hours=1),
                CGMReading.ts <= meal_ts)
        .order_by(CGMReading.ts.asc())
        .all()
    )
    if len(rows) < 2:
        return 0.0, baseline
    vals = [r.glucose_mg_dl for r in rows]
    span = (rows[-1].ts - rows[0].ts).total_seconds() / 60
    vel  = (vals[-1] - vals[0]) / span * 30 if span > 0 else 0.0
    return round(vel, 2), round(float(np.mean(vals)), 1)


def _day_index(db, meal_ts: datetime) -> int:
    from app.models.cgm import CGMReading
    first = db.query(CGMReading).order_by(CGMReading.ts.asc()).first()
    return max(0, (meal_ts.date() - first.ts.date()).days) if first else 0


def _predict_lstm3h(model_path: str, baseline: float) -> dict:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"error": "PyTorch not installed"}

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"error": f"Cannot load model: {exc}"}

    feat_cols = ckpt["feat_cols"]
    seq_len   = ckpt["seq_len"]
    feat_mu   = np.array(ckpt["feat_mu"],  dtype=np.float32)
    feat_sig  = np.array(ckpt["feat_sig"], dtype=np.float32)
    y_mu, y_sig   = ckpt["y_mu"], ckpt["y_sig"]
    hidden, layers = ckpt["hidden"], ckpt["layers"]
    n_feat = len(feat_cols)

    feat_path = os.path.join(PROCESSED_DIR, "5min_features.csv")
    if not os.path.exists(feat_path):
        return {"error": "5min_features.csv missing — run training first"}

    df = pd.read_csv(feat_path, parse_dates=["ts"]).sort_values("ts")
    miss = [c for c in feat_cols if c not in df.columns]
    if miss:
        return {"error": f"Missing columns: {miss}"}

    df[feat_cols] = df[feat_cols].fillna(0.0)
    recent = df.tail(seq_len)
    if len(recent) < seq_len:
        return {"error": f"Need {seq_len} rows, have {len(recent)}"}

    X      = recent[feat_cols].values.astype(np.float32)
    X_norm = (X - feat_mu) / feat_sig

    class _LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                                dropout=0.2 if layers > 1 else 0.0)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    m = _LSTM()
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    with torch.no_grad():
        pred = m(torch.tensor(X_norm[None])).item() * y_sig + y_mu

    if not (40 <= pred <= 450):
        return {"error": f"Out-of-range prediction: {pred:.0f} mg/dL"}

    return {"predicted_glucose_3h": round(pred, 1),
            "predicted_delta":      round(pred - baseline, 1)}


def _predict_sarimax(
    model_path: str,
    meal_ts: datetime,
    baseline: float,
    carbs_g: float,
    minutes_since_glipizide: float,
    walk_min_after: float = 0.0,
) -> dict:
    try:
        with open(model_path, "rb") as f:
            result = pickle.load(f)
    except Exception as exc:
        return {"error": f"Cannot load SARIMAX: {exc}"}

    tau  = 75.0
    rows = []
    for step in range(1, 4):
        dt  = meal_ts + timedelta(hours=step)
        dm  = step * 60.0
        h   = dt.hour + dt.minute / 60.0
        # Walk planned within ~90 min of meal is visible in the 2h window at T+1h and T+2h.
        # At T+3h the walk has passed outside the lookback window, so use 0.
        walk_exog = walk_min_after if step <= 2 else 0.0
        rows.append({
            "carb_load_decayed": (carbs_g or 0) * math.exp(-dm / tau),
            "walk_min_last_2h":  walk_exog,
            "glipizide_active":  float(minutes_since_glipizide < 300),
            "is_fasting":        0.0,
            "hour_sin":          math.sin(2 * math.pi * h / 24),
            "hour_cos":          math.cos(2 * math.pi * h / 24),
        })

    exog = pd.DataFrame(rows)[SARIMAX_EXOG_COLS]
    try:
        fc    = result.get_forecast(steps=3, exog=exog)
        means = fc.predicted_mean  # index: T+1h, T+2h, T+3h
        pred  = float(means.iloc[-1])          # T+3h value
        peak  = float(means.iloc[:2].max())    # peak typically at T+1h or T+2h (90-150 min)
    except Exception as exc:
        return {"error": f"Forecast error: {exc}"}

    if not (40 <= pred <= 450) or np.isnan(pred):
        return {"error": f"Out-of-range prediction: {pred:.0f} mg/dL"}

    return {
        "predicted_glucose_3h":  round(pred, 1),
        "predicted_delta":       round(pred  - baseline, 1),
        "predicted_peak_delta":  round(peak  - baseline, 1),
        "predicted_peak_glucose": round(peak, 1),
    }


# ── Trajectory forecast (/forecast/ page) ─────────────────────────────────────

def forecast_trajectory(
    db,
    current_glucose: float,
    carbs_g: float,
    walk_min_after: float,
    meal_ts: datetime,
    minutes_since_glipizide: float = 999.0,
    is_fasting: bool = False,
) -> dict:
    """
    Run SARIMAX (3 hourly points + CI) and LSTM trajectory (36 × 5-min points),
    each for a "walk" and "no walk" scenario.

    Returns:
        {
          "cold_start": bool,
          "history":    [{"offset_min": float, "glucose": float}, ...],
          "sarimax":    {"walk": {t, mean, lower, upper}, "no_walk": ...} | None | {error},
          "lstm_trajectory": {"walk": {t, glucose}, "no_walk": ...}       | None | {error},
        }
    """
    history, cold_start = _get_cgm_history(db, meal_ts)

    out: dict = {
        "cold_start":          cold_start,
        "history":             history,
        "sarimax":             None,
        "lstm_trajectory":     None,
        "lstm_trajectory_v2":  None,
        "reference":           None,
        "reference_sensitivity": None,
    }

    sp = os.path.join(MODELS_DIR, "sarimax_hourly.pkl")
    if os.path.exists(sp):
        try:
            out["sarimax"] = _sarimax_trajectory(
                sp, meal_ts, current_glucose, carbs_g,
                minutes_since_glipizide, walk_min_after, is_fasting,
            )
        except Exception as exc:
            out["sarimax"] = {"error": str(exc)}

    lp = os.path.join(MODELS_DIR, "lstm_trajectory.pt")
    if os.path.exists(lp):
        try:
            out["lstm_trajectory"] = _lstm_trajectory(
                lp, current_glucose, carbs_g, walk_min_after,
                meal_ts, minutes_since_glipizide, is_fasting, cold_start,
            )
        except Exception as exc:
            out["lstm_trajectory"] = {"error": str(exc)}

    lp2 = os.path.join(MODELS_DIR, "lstm_trajectory_v2.pt")
    if os.path.exists(lp2):
        try:
            out["lstm_trajectory_v2"] = _lstm_trajectory_v2(
                lp2, current_glucose, carbs_g, walk_min_after,
                meal_ts, minutes_since_glipizide, is_fasting, cold_start,
            )
        except Exception as exc:
            out["lstm_trajectory_v2"] = {"error": str(exc)}

    # Parametric reference — always computed, no model file needed
    try:
        sens, t_peak = _estimate_physio_params()
        out["reference"] = _parametric_reference(
            current_glucose, carbs_g, walk_min_after,
            minutes_since_glipizide, sensitivity=sens, peak_time=t_peak,
        )
        out["reference_sensitivity"] = round(sens, 2)
    except Exception as exc:
        out["reference"] = {"error": str(exc)}

    return out


def _get_cgm_history(db, ref_ts: datetime, window_min: int = 65):
    """Return (history_list, cold_start).

    history_list — [{offset_min, glucose}] for the last window_min minutes,
    where offset_min is negative (past relative to ref_ts).
    cold_start is True when the most recent reading is > _GAP_THRESHOLD_MIN old,
    or when no readings exist at all.
    """
    from app.models.cgm import CGMReading
    cutoff = ref_ts - timedelta(minutes=window_min)
    rows = (
        db.query(CGMReading)
        .filter(CGMReading.ts >= cutoff, CGMReading.ts <= ref_ts + timedelta(minutes=5))
        .order_by(CGMReading.ts.asc())
        .all()
    )
    history = [
        {
            "offset_min": round((r.ts - ref_ts).total_seconds() / 60, 1),
            "glucose":    round(r.glucose_mg_dl, 1),
        }
        for r in rows
    ]
    cold_start = True
    if rows:
        newest_age = (ref_ts - rows[-1].ts).total_seconds() / 60
        cold_start = newest_age > _GAP_THRESHOLD_MIN
    return history, cold_start


def _sarimax_trajectory(
    model_path: str,
    meal_ts: datetime,
    current_glucose: float,
    carbs_g: float,
    minutes_since_glipizide: float,
    walk_min_after: float,
    is_fasting: bool,
) -> dict:
    """Return {"walk": {...}, "no_walk": {...}} — 3 hourly forecast points with 95% CI.

    The raw SARIMAX predictions are absolute glucose values anchored to the model's
    training end-state.  We apply a one-time offset so the T+1h point aligns with
    current_glucose, making the walk/no-walk delta visually meaningful regardless
    of how stale the model is.
    """
    with open(model_path, "rb") as f:
        result = pickle.load(f)

    tau = 75.0
    output: dict = {}

    for name, walk_val in [("walk", walk_min_after), ("no_walk", 0.0)]:
        rows = []
        for step in range(1, 4):
            dt = meal_ts + timedelta(hours=step)
            dm = step * 60.0
            h  = dt.hour + dt.minute / 60.0
            walk_exog = walk_val if step <= 2 else 0.0
            rows.append({
                "carb_load_decayed": (carbs_g or 0) * math.exp(-dm / tau),
                "walk_min_last_2h":  walk_exog,
                "glipizide_active":  float(minutes_since_glipizide < 300),
                "is_fasting":        float(is_fasting),
                "hour_sin":          math.sin(2 * math.pi * h / 24),
                "hour_cos":          math.cos(2 * math.pi * h / 24),
            })
        exog = pd.DataFrame(rows)[SARIMAX_EXOG_COLS]
        fc   = result.get_forecast(steps=3, exog=exog)
        means = fc.predicted_mean.values
        ci    = fc.conf_int()

        # Anchor: shift all predictions so T+1h = current_glucose.
        # This preserves the shape of the curve (inter-step deltas stay intact)
        # while making the chart meaningful when the model is months stale.
        offset = current_glucose - float(means[0])

        output[name] = {
            "t":     [60, 120, 180],
            "mean":  [round(float(v) + offset, 1) for v in means],
            "lower": [round(float(ci.iloc[i, 0]) + offset, 1) for i in range(3)],
            "upper": [round(float(ci.iloc[i, 1]) + offset, 1) for i in range(3)],
        }

    return output


def _lstm_trajectory(
    model_path: str,
    current_glucose: float,
    carbs_g: float,
    walk_min_after: float,
    meal_ts: datetime,
    minutes_since_glipizide: float,
    is_fasting: bool,
    cold_start: bool,
) -> dict:
    """Return {"walk": {t, glucose}, "no_walk": {t, glucose}} — 36 × 5-min predictions.

    Input sequence built from 5min_features.csv tail (same as lstm_3h).
    Cold-start: synthesise a flat window at current_glucose when real readings
    are unavailable.  Walk/no-walk differ only in the exog walk_min_last_2h scalar.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"error": "PyTorch not installed"}

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"error": f"Cannot load model: {exc}"}

    feat_cols = ckpt["feat_cols"]
    exog_cols = ckpt["exog_cols"]
    seq_len   = ckpt["seq_len"]
    HORIZON   = ckpt.get("horizon", 36)
    n_feat    = len(feat_cols)
    n_exog    = len(exog_cols)

    feat_mu  = np.array(ckpt["feat_mu"],  dtype=np.float32)
    feat_sig = np.array(ckpt["feat_sig"], dtype=np.float32)
    exog_mu  = np.array(ckpt["exog_mu"],  dtype=np.float32)
    exog_sig = np.array(ckpt["exog_sig"], dtype=np.float32)
    y_mu, y_sig = float(ckpt["y_mu"]), float(ckpt["y_sig"])
    hidden, layers = ckpt["hidden"], ckpt["layers"]

    # ── Build input sequence ───────────────────────────────────────────────────
    if cold_start:
        # Synthesise a flat baseline at current_glucose
        x_seq_raw = np.zeros((seq_len, n_feat), dtype=np.float32)
        if "glucose_mg_dl" in feat_cols:
            x_seq_raw[:, feat_cols.index("glucose_mg_dl")] = current_glucose
        h = meal_ts.hour + meal_ts.minute / 60.0
        for col, val in [
            ("hour_sin", math.sin(2 * math.pi * h / 24)),
            ("hour_cos", math.cos(2 * math.pi * h / 24)),
        ]:
            if col in feat_cols:
                x_seq_raw[:, feat_cols.index(col)] = val
    else:
        feat_path = os.path.join(PROCESSED_DIR, "5min_features.csv")
        if not os.path.exists(feat_path):
            return {"error": "5min_features.csv missing — run training pipeline first"}
        df = pd.read_csv(feat_path, parse_dates=["ts"]).sort_values("ts")
        df[feat_cols] = df[feat_cols].fillna(0.0)
        recent = df.tail(seq_len)
        if len(recent) == 0:
            return {"error": "5min_features.csv is empty"}
        if len(recent) < seq_len:
            # Pad from the front with the oldest available row
            pad_n = seq_len - len(recent)
            padded = pd.concat(
                [recent.iloc[[0]]] * pad_n + [recent], ignore_index=True
            )
            x_seq_raw = padded[feat_cols].values.astype(np.float32)
        else:
            x_seq_raw = recent[feat_cols].values.astype(np.float32)

    x_seq_norm = (x_seq_raw - feat_mu) / feat_sig

    # ── Rebuild model skeleton ─────────────────────────────────────────────────
    class _LSTMTraj(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                                dropout=0.2 if layers > 1 else 0.0)
            self.head = nn.Linear(hidden + n_exog, HORIZON)

        def forward(self, x_seq, x_exog):
            _, (h, _) = self.lstm(x_seq)
            return self.head(torch.cat([h[-1], x_exog], dim=1))

    m = _LSTMTraj()
    m.load_state_dict(ckpt["state_dict"])
    m.eval()

    h_angle = meal_ts.hour + meal_ts.minute / 60.0
    output: dict = {}

    for name, walk_val in [("walk", walk_min_after), ("no_walk", 0.0)]:
        # Exog at t=0: current meal/activity state
        # walk_min_after used as walk_min_last_2h proxy — treat planned walk as "recent"
        x_exog_raw = np.array([
            float(carbs_g or 0),                         # carb_load_decayed (just ate: exp(0)=1)
            float(walk_val),                             # walk_min_last_2h
            float(minutes_since_glipizide < 300),        # glipizide_active
            float(is_fasting),
            math.sin(2 * math.pi * h_angle / 24),
            math.cos(2 * math.pi * h_angle / 24),
        ], dtype=np.float32)
        x_exog_norm = (x_exog_raw - exog_mu) / exog_sig

        Xseq  = torch.tensor(x_seq_norm[None])    # (1, seq_len, n_feat)
        Xexog = torch.tensor(x_exog_norm[None])   # (1, n_exog)

        with torch.no_grad():
            pred_norm = m(Xseq, Xexog).numpy()[0]  # (HORIZON,)

        pred_glucose = np.clip(pred_norm * y_sig + y_mu, 40, 450)

        output[name] = {
            "t":       list(range(5, HORIZON * 5 + 1, 5)),  # [5, 10, ..., 180]
            "glucose": [round(float(v), 1) for v in pred_glucose],
        }

    return output


def _lstm_trajectory_v2(
    model_path: str,
    current_glucose: float,
    carbs_g: float,
    walk_min_after: float,
    meal_ts: datetime,
    minutes_since_glipizide: float,
    is_fasting: bool,
    cold_start: bool,
) -> dict:
    """Return {"walk": {t, glucose}, "no_walk": {t, glucose}} using the V2 architecture.

    The key difference from V1: a (36 × 6) future-exog matrix is built analytically
    for each scenario, giving the model step-by-step visibility into how carb load
    decays and when walk activity enters/exits the 2-hour lookback window.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"error": "PyTorch not installed"}

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"error": f"Cannot load model: {exc}"}

    feat_cols    = ckpt["feat_cols"]
    f_exog_cols  = ckpt["f_exog_cols"]
    seq_len      = ckpt["seq_len"]
    HORIZON      = ckpt.get("horizon", 36)
    n_feat       = len(feat_cols)
    n_future_exo = len(f_exog_cols)

    feat_mu    = np.array(ckpt["feat_mu"],    dtype=np.float32)
    feat_sig   = np.array(ckpt["feat_sig"],   dtype=np.float32)
    f_exog_mu  = np.array(ckpt["f_exog_mu"],  dtype=np.float32)
    f_exog_sig = np.array(ckpt["f_exog_sig"], dtype=np.float32)
    y_mu, y_sig    = float(ckpt["y_mu"]), float(ckpt["y_sig"])
    hidden_past    = ckpt["hidden_past"]
    hidden_future  = ckpt["hidden_future"]
    layers         = ckpt["layers"]

    # ── Past sequence (same as V1) ─────────────────────────────────────────────
    if cold_start:
        x_seq_raw = np.zeros((seq_len, n_feat), dtype=np.float32)
        if "glucose_mg_dl" in feat_cols:
            x_seq_raw[:, feat_cols.index("glucose_mg_dl")] = current_glucose
        h = meal_ts.hour + meal_ts.minute / 60.0
        for col, val in [
            ("hour_sin", math.sin(2 * math.pi * h / 24)),
            ("hour_cos", math.cos(2 * math.pi * h / 24)),
        ]:
            if col in feat_cols:
                x_seq_raw[:, feat_cols.index(col)] = val
    else:
        feat_path = os.path.join(PROCESSED_DIR, "5min_features.csv")
        if not os.path.exists(feat_path):
            return {"error": "5min_features.csv missing — run training pipeline first"}
        df = pd.read_csv(feat_path, parse_dates=["ts"]).sort_values("ts")
        df[feat_cols] = df[feat_cols].fillna(0.0)
        recent = df.tail(seq_len)
        if len(recent) == 0:
            return {"error": "5min_features.csv is empty"}
        if len(recent) < seq_len:
            pad_n  = seq_len - len(recent)
            padded = pd.concat([recent.iloc[[0]]] * pad_n + [recent], ignore_index=True)
            x_seq_raw = padded[feat_cols].values.astype(np.float32)
        else:
            x_seq_raw = recent[feat_cols].values.astype(np.float32)

    x_seq_norm = (x_seq_raw - feat_mu) / feat_sig

    # ── Model skeleton ────────────────────────────────────────────────────────
    class _LSTMTrajV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.past_lstm   = nn.LSTM(n_feat, hidden_past, layers, batch_first=True,
                                        dropout=0.2 if layers > 1 else 0.0)
            self.future_lstm = nn.LSTM(n_future_exo, hidden_future, 1, batch_first=True)
            self.head        = nn.Linear(hidden_past + hidden_future, HORIZON)

        def forward(self, x_past, x_future):
            _, (hp, _) = self.past_lstm(x_past)
            _, (hf, _) = self.future_lstm(x_future)
            return self.head(torch.cat([hp[-1], hf[-1]], dim=1))

    m = _LSTMTrajV2()
    m.load_state_dict(ckpt["state_dict"])
    m.eval()

    tau     = 75.0
    output: dict = {}

    for name, walk_val in [("walk", walk_min_after), ("no_walk", 0.0)]:
        # Build the 36 × 6 future exog matrix analytically
        future_exog_raw = np.zeros((HORIZON, n_future_exo), dtype=np.float32)

        for k, t_min in enumerate(range(5, HORIZON * 5 + 1, 5)):
            # carb_load_decayed: exponential decay from meal eaten at t=0
            carb_decay = float(carbs_g or 0) * math.exp(-t_min / tau)

            # walk_min_last_2h at this future step:
            # Walk happens from t=0 to t=walk_val minutes.
            # 2h lookback window at t_min: [t_min-120, t_min].
            # Overlap with walk interval [0, walk_val]:
            walk_in_win = max(
                0.0,
                min(float(walk_val), t_min) - max(0.0, t_min - 120.0)
            )

            # Glipizide still active at this future step?
            glip_active = float((minutes_since_glipizide + t_min) < 300)

            future_ts = meal_ts + timedelta(minutes=t_min)
            h_angle   = future_ts.hour + future_ts.minute / 60.0

            future_exog_raw[k] = [
                carb_decay,
                walk_in_win,
                glip_active,
                float(is_fasting),
                math.sin(2 * math.pi * h_angle / 24),
                math.cos(2 * math.pi * h_angle / 24),
            ]

        future_exog_norm = (future_exog_raw - f_exog_mu) / f_exog_sig

        Xseq  = torch.tensor(x_seq_norm[None])         # (1, seq_len, n_feat)
        Xfut  = torch.tensor(future_exog_norm[None])   # (1, HORIZON, n_future_exo)

        with torch.no_grad():
            pred_norm = m(Xseq, Xfut).numpy()[0]  # (HORIZON,)

        pred_glucose = np.clip(pred_norm * y_sig + y_mu, 40, 450)

        output[name] = {
            "t":       list(range(5, HORIZON * 5 + 1, 5)),
            "glucose": [round(float(v), 1) for v in pred_glucose],
        }

    return output


def _estimate_physio_params() -> tuple[float, float]:
    """Estimate personal carb sensitivity (mg/dL per gram) from historical meals.

    Returns (sensitivity, peak_time_min).  Falls back to (0.4, 60) when insufficient data.
    sensitivity = median(peak_delta / carbs_g_final) over meals with carbs > 20 g.
    peak_time   = fixed 60 min (per_meal_features.csv doesn't record timing of peak).
    """
    feat_path = os.path.join(PROCESSED_DIR, "per_meal_features.csv")
    if not os.path.exists(feat_path):
        return 0.4, 60.0
    try:
        df = pd.read_csv(feat_path)
        valid = df[(df.get("carbs_g_final", pd.Series(dtype=float)) > 20)
                   & (df.get("peak_delta",   pd.Series(dtype=float)) > 5)].dropna(
            subset=["carbs_g_final", "peak_delta"]
        )
        if len(valid) < 10:
            return 0.4, 60.0
        sensitivity = float(np.clip(
            np.median(valid["peak_delta"] / valid["carbs_g_final"]), 0.1, 2.0
        ))
        return sensitivity, 60.0
    except Exception:
        return 0.4, 60.0


def _parametric_reference(
    current_glucose: float,
    carbs_g: float,
    walk_min_after: float,
    minutes_since_glipizide: float,
    sensitivity: float = 0.4,
    peak_time: float   = 60.0,
) -> dict:
    """Deterministic physiological reference curve.

    Uses the log-normal kernel  f(t) = (t / T) × exp(1 − t / T)  which:
    • is exactly 0 at t = 0
    • peaks at exactly 1.0 when t = peak_time (T)
    • decays asymptotically thereafter (reaches ~40% of peak at t = 3 × T)

    Walk reduces the peak by ~0.5 % per minute of planned walking (capped at 25 %).
    Active Glipizide suppresses the peak by ~25 %.
    """
    glip_factor = 0.75 if minutes_since_glipizide < 300 else 1.0

    result: dict = {}
    for name, walk_val in [("walk", walk_min_after), ("no_walk", 0.0)]:
        walk_factor = max(0.75, 1.0 - 0.005 * min(float(walk_val), 50.0))
        peak_rise   = float(carbs_g or 0) * sensitivity * walk_factor * glip_factor

        glucose = []
        for t in range(5, 181, 5):
            u      = t / peak_time
            kernel = u * math.exp(1.0 - u)   # 0 at t=0, 1.0 at t=peak_time
            glucose.append(round(current_glucose + peak_rise * kernel, 1))

        result[name] = {"t": list(range(5, 181, 5)), "glucose": glucose}

    return result
