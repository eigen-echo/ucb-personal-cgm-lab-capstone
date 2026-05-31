"""Detect unattributed glucose spikes and query pending ones for the review page."""
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.activity import Activity
from app.models.app_config import AppConfig
from app.models.cgm import CGMReading
from app.models.meal import Meal
from app.models.spike_event import SpikeEvent


def _cfg(db: Session, key: str, default: str) -> float:
    row = db.query(AppConfig).filter(AppConfig.key == key).first()
    return float(row.value if row else default)


def detect_and_insert(db: Session, start_ts: datetime, end_ts: datetime) -> int:
    """
    Scan CGM readings in [start_ts, end_ts] for glucose spikes that have no
    matching meal/activity. Insert new SpikeEvent rows; return count inserted.
    """
    meal_thresh     = _cfg(db, "spike_threshold_meal",     "30")
    activity_thresh = _cfg(db, "spike_threshold_activity", "15")
    min_thresh      = min(meal_thresh, activity_thresh)

    # Extra 40-min buffer before start so first readings have a valid baseline
    buffer_start = start_ts - timedelta(minutes=40)
    readings = (
        db.query(CGMReading)
        .filter(CGMReading.ts >= buffer_start, CGMReading.ts <= end_ts)
        .order_by(CGMReading.ts)
        .all()
    )
    if len(readings) < 10:
        return 0

    df = pd.DataFrame(
        [(r.ts, r.glucose_mg_dl) for r in readings],
        columns=["ts", "glucose"],
    ).set_index("ts").sort_index()

    # Baseline: rolling min of the 7 readings *before* the current point
    df["baseline"] = df["glucose"].shift(1).rolling(window=7, min_periods=3).min()
    df["delta"]    = df["glucose"] - df["baseline"]
    df = df.dropna(subset=["baseline"])

    # Local maxima: glucose[i] must beat all neighbours within ±3 readings (±15 min)
    g = df["glucose"].values
    is_max = np.zeros(len(g), dtype=bool)
    W = 3
    for i in range(W, len(g) - W):
        hood = np.concatenate([g[i - W : i], g[i + 1 : i + W + 1]])
        if g[i] > hood.max():
            is_max[i] = True
    df["is_max"] = is_max

    candidates = df[
        df["is_max"] & (df["delta"] >= min_thresh) & (df.index >= start_ts)
    ].copy()

    if candidates.empty:
        return 0

    # Greedy cluster: within each 2-hour window keep only the highest-delta peak
    sorted_cands = candidates.sort_values("delta", ascending=False)
    selected: list[tuple] = []
    used: set = set()
    for ts, row in sorted_cands.iterrows():
        if ts in used:
            continue
        selected.append((ts, row))
        for other_ts in sorted_cands.index:
            if abs((other_ts - ts).total_seconds()) <= 7200:
                used.add(other_ts)

    # Fetch existing meals / activities to skip already-attributed spikes
    # Activity window is 120 min pre-spike: cover dinner→walk→late spike scenarios
    check_start = start_ts - timedelta(hours=2, minutes=30)
    meal_times = [
        m.ts for m in db.query(Meal)
        .filter(Meal.ts >= check_start, Meal.ts <= end_ts)
        .all()
    ]
    act_times = [
        a.ts for a in db.query(Activity)
        .filter(Activity.ts >= check_start, Activity.ts <= end_ts + timedelta(minutes=30))
        .all()
    ]

    inserted = 0
    for peak_ts, row in selected:
        if any(peak_ts - timedelta(minutes=90) <= mt <= peak_ts for mt in meal_times):
            continue
        if any(peak_ts - timedelta(minutes=120) <= at <= peak_ts + timedelta(minutes=30)
               for at in act_times):
            continue
        if db.query(SpikeEvent).filter(SpikeEvent.peak_ts == peak_ts).first():
            continue
        db.add(SpikeEvent(
            peak_ts=peak_ts,
            peak_glucose=float(row["glucose"]),
            baseline_glucose=float(row["baseline"]),
            peak_delta=float(row["delta"]),
        ))
        inserted += 1

    if inserted:
        db.commit()
    return inserted


def get_pending(db: Session) -> list[dict[str, Any]]:
    """
    Return all spikes that are neither manually attributed nor dynamically matched
    to a meal/activity. Each entry is a dict with spike fields + pre-fill timestamps.
    """
    spikes = db.query(SpikeEvent).order_by(SpikeEvent.peak_ts.desc()).all()
    if not spikes:
        return []

    earliest = spikes[-1].peak_ts
    meal_times = [
        m.ts for m in db.query(Meal)
        .filter(Meal.ts >= earliest - timedelta(hours=2))
        .all()
    ]
    act_times = [
        a.ts for a in db.query(Activity)
        .filter(Activity.ts >= earliest - timedelta(hours=2, minutes=30))
        .all()
    ]

    result = []
    for s in spikes:
        if s.attribution is not None:
            continue
        pt = s.peak_ts
        if any(pt - timedelta(minutes=90) <= mt <= pt for mt in meal_times):
            continue
        if any(pt - timedelta(minutes=120) <= at <= pt + timedelta(minutes=30) for at in act_times):
            continue
        from app.services.timezone import get_cached_tz, to_local
        tz = get_cached_tz()
        result.append({
            "id":               s.id,
            "peak_ts":          pt,
            "peak_glucose":     s.peak_glucose,
            "baseline_glucose": s.baseline_glucose,
            "peak_delta":       s.peak_delta,
            # Pre-fill times in user's local timezone for the log-meal / log-activity forms
            "meal_prefill":  to_local(pt - timedelta(minutes=45), tz, "%Y-%m-%dT%H:%M"),
            "act_prefill":   to_local(pt - timedelta(minutes=30), tz, "%Y-%m-%dT%H:%M"),
        })
    return result


def count_pending(db: Session) -> int:
    return len(get_pending(db))
