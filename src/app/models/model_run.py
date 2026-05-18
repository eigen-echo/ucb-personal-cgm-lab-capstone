from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from app.database import Base


class ModelRun(Base):
    __tablename__ = "model_runs"

    id           = Column(Integer, primary_key=True, index=True)
    run_ts       = Column(DateTime, default=datetime.utcnow, index=True)
    model_name   = Column(String, nullable=False)   # ridge_per_meal | sarimax_hourly
    rmse_train   = Column(Float, nullable=True)
    rmse_test    = Column(Float, nullable=True)
    mae_test     = Column(Float, nullable=True)
    r2_test      = Column(Float, nullable=True)
    n_train      = Column(Integer, nullable=True)
    n_test       = Column(Integer, nullable=True)
    notes        = Column(Text, nullable=True)
    triggered_by = Column(String, default="web_app")  # always web_app; notebook runs never recorded here
