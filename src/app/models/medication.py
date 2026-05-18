from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from app.database import Base


class Medication(Base):
    __tablename__ = "medications"

    id           = Column(Integer, primary_key=True, index=True)
    scheduled_ts = Column(DateTime, nullable=False, index=True)
    drug         = Column(String, nullable=False)
    dose_mg      = Column(Float, nullable=True)
    taken        = Column(Boolean, default=True)
    notes        = Column(String, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
