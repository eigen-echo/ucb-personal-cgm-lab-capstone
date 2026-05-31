from sqlalchemy import Column, Integer, String, Float
from app.database import Base


class FoodItem(Base):
    __tablename__ = "food_items"

    id           = Column(Integer, primary_key=True, index=True)
    dish_key     = Column(String, unique=True, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    carbs_g      = Column(Float, nullable=True)
    protein_g    = Column(Float, nullable=True)
    fat_g        = Column(Float, nullable=True)
    gl           = Column(String, nullable=True)   # low | medium | high
    source       = Column(String, default="manual")  # manual | imported
