from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from app.database import Base


class FastingWindow(Base):
    __tablename__ = "fasting_windows"

    id          = Column(Integer, primary_key=True, index=True)
    start_ts    = Column(DateTime, nullable=False)
    end_ts      = Column(DateTime, nullable=True)   # null while still fasting
    window_type = Column(String, default="overnight")  # overnight | intermittent_skip_breakfast
    notes       = Column(String, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
