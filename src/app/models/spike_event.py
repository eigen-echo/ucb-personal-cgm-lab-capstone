from sqlalchemy import Column, DateTime, Float, Integer, String
from app.database import Base


class SpikeEvent(Base):
    __tablename__ = "spike_events"
    id               = Column(Integer, primary_key=True, autoincrement=True)
    peak_ts          = Column(DateTime, unique=True, nullable=False)
    peak_glucose     = Column(Float)
    baseline_glucose = Column(Float)
    peak_delta       = Column(Float)
    # NULL = pending; "meal" | "activity" | "fasting" | "no_data" = resolved
    attribution      = Column(String, nullable=True)
    acknowledged_at  = Column(DateTime, nullable=True)
