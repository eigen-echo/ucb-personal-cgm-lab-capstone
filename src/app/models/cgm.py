from sqlalchemy import Column, Integer, Float, DateTime, UniqueConstraint
from app.database import Base


class CGMReading(Base):
    __tablename__ = "cgm_readings"

    id           = Column(Integer, primary_key=True, index=True)
    ts           = Column(DateTime, nullable=False, index=True)
    glucose_mg_dl = Column(Float, nullable=False)

    __table_args__ = (UniqueConstraint("ts", name="uq_cgm_ts"),)
