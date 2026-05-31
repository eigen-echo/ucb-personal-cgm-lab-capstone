from sqlalchemy import Column, DateTime, String
from datetime import datetime
from app.database import Base


class AppConfig(Base):
    __tablename__ = "app_config"
    key         = Column(String, primary_key=True)
    value       = Column(String, nullable=False)
    description = Column(String, nullable=True)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
