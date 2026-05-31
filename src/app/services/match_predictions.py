"""
Match pending glucose predictions to actual CGM readings.
Called automatically from cgm.py after each successful import.
"""
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.models.cgm import CGMReading
from app.models.glucose_prediction import GlucosePrediction

TOLERANCE = timedelta(minutes=15)


def match_predictions(db: Session) -> int:
    """
    For every GlucosePrediction where actual_glucose_3h IS NULL and
    predict_for_ts is at least 15 min in the past:
      1. Find the nearest CGM reading within ±15 min of predict_for_ts.
      2. Find the actual peak delta in the 3-hour window [predict_for_ts−3h, predict_for_ts].
    Returns the number of rows newly matched.
    """
    now    = datetime.utcnow()
    cutoff = now - TOLERANCE

    pending = (
        db.query(GlucosePrediction)
        .filter(
            GlucosePrediction.actual_glucose_3h.is_(None),
            GlucosePrediction.predict_for_ts <= cutoff,
        )
        .all()
    )

    matched = 0
    for pred in pending:
        lo = pred.predict_for_ts - TOLERANCE
        hi = pred.predict_for_ts + TOLERANCE

        candidates = (
            db.query(CGMReading)
            .filter(CGMReading.ts >= lo, CGMReading.ts <= hi,
                    CGMReading.glucose_mg_dl.isnot(None))
            .all()
        )
        if not candidates:
            continue

        nearest = min(
            candidates,
            key=lambda r: abs((r.ts - pred.predict_for_ts).total_seconds()),
        )

        # Peak delta over the 3-hour window ending at predict_for_ts
        window_readings = (
            db.query(CGMReading)
            .filter(
                CGMReading.ts >= pred.predict_for_ts - timedelta(hours=3),
                CGMReading.ts <= pred.predict_for_ts,
                CGMReading.glucose_mg_dl.isnot(None),
            )
            .all()
        )
        peak = max(
            (r.glucose_mg_dl for r in window_readings),
            default=nearest.glucose_mg_dl,
        )

        pred.actual_glucose_3h = round(nearest.glucose_mg_dl, 1)
        pred.actual_peak_delta = round(peak - pred.baseline_glucose, 1)
        pred.matched_cgm_ts    = nearest.ts
        pred.matched_at        = now
        matched += 1

    if matched:
        db.commit()

    return matched
