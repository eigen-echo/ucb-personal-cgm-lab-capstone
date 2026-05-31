from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from app.database import Base


class Meal(Base):
    __tablename__ = "meals"

    id                    = Column(Integer, primary_key=True, index=True)
    ts                    = Column(DateTime, nullable=False, index=True)
    dish_names            = Column(String, nullable=True)
    meal_type             = Column(String, nullable=True)   # breakfast/lunch/dinner/snack
    carbs_grams_logged    = Column(Float, nullable=True)
    carbs_grams_estimated = Column(Float, nullable=True)
    fasted_meal           = Column(Boolean, default=False)
    notes                 = Column(String, nullable=True)
    created_at            = Column(DateTime, default=datetime.utcnow)
