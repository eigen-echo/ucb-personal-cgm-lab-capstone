from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from app.database import Base


class GlucosePrediction(Base):
    """
    One row per saved prospective prediction.

    Workflow:
      1. User submits /predictions/new  → row created; actual_* columns NULL.
      2. CGM import runs match_predictions() → actual_* filled in for rows
         where predict_for_ts is now in the past and a nearby reading exists.
    """
    __tablename__ = "glucose_predictions"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    created_at       = Column(DateTime, default=datetime.utcnow, nullable=False)

    # When is the 3-hour mark we're predicting? (UTC)
    predict_for_ts   = Column(DateTime, nullable=False, index=True)

    # Source context
    meal_id          = Column(Integer, ForeignKey("meals.id"), nullable=True)
    meal_name        = Column(String,  nullable=True)   # free-form entry
    carbs_g          = Column(Float,   nullable=True)
    baseline_glucose = Column(Float,   nullable=False)  # last CGM reading at prediction time

    # ── Model predictions (NULL = model file not present at prediction time) ──
    # Ridge & RF: output is delta (rise above baseline) in mg/dL
    ridge_delta      = Column(Float, nullable=True)
    rf_delta         = Column(Float, nullable=True)

    # LSTM 3h: direct absolute glucose prediction at T+3h
    lstm3h_glucose   = Column(Float, nullable=True)   # predicted mg/dL at T+3h
    lstm3h_delta     = Column(Float, nullable=True)   # = lstm3h_glucose - baseline_glucose

    # SARIMAX: 3-step (3h) multi-step forecast from fitted hourly model
    sarimax3h_glucose  = Column(Float, nullable=True)
    sarimax3h_delta    = Column(Float, nullable=True)  # delta at T+3h
    sarimax_peak_delta = Column(Float, nullable=True)  # max rise at T+1h or T+2h (90-150 min window)

    # ── Actuals (filled in by match_predictions after CGM upload) ─────────────
    actual_glucose_3h  = Column(Float,    nullable=True)
    actual_peak_delta  = Column(Float,    nullable=True)  # max rise in full 3h window
    matched_cgm_ts     = Column(DateTime, nullable=True)  # which CGM reading was matched
    matched_at         = Column(DateTime, nullable=True)

    # Which model version produced this prediction (run_tag from model_runs at save time)
    model_run_tag    = Column(String, nullable=True)

    # Full snapshot of inputs used — lets us replay/debug a prediction later
    features_json    = Column(String, nullable=True)

    notes            = Column(String, nullable=True)
