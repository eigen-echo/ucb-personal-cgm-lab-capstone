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
