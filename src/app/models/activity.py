from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from app.database import Base


class Activity(Base):
    __tablename__ = "activities"

    id            = Column(Integer, primary_key=True, index=True)
    ts            = Column(DateTime, nullable=False, index=True)
    activity_type = Column(String, default="walk")
    duration_min  = Column(Float, nullable=False)
    notes         = Column(String, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    # ── Fitness tracker fields (all optional) ────────────────────────────────
    calories_burned     = Column(Float,   nullable=True)   # kcal
    distance_km         = Column(Float,   nullable=True)   # km
    steps               = Column(Integer, nullable=True)
    avg_pace_min_per_km = Column(Float,   nullable=True)   # decimal minutes, e.g. 14'18" → 14.30
    avg_heart_rate_bpm  = Column(Integer, nullable=True)   # bpm
    cardio_load         = Column(Float,   nullable=True)   # device score
    elevation_gain_m    = Column(Float,   nullable=True)   # metres
    active_zone_min     = Column(Integer, nullable=True)   # minutes in active heart-rate zone
