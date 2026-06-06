import os
import sqlite3
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.database import init_db, SessionLocal, DB_PATH
from app.models.app_config import AppConfig
from app.models.food_item import FoodItem
from app.routers import dashboard, meals, activities, cgm, food_lookup, training, spikes, config, predictions, medications, forecast, weekly

DATA_RAW = os.environ.get("DATA_RAW_DIR", "/app/data/raw")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _migrate_db()
    _seed_food_lookup()
    _seed_config()
    yield


def _migrate_db():
    """Add columns introduced after the initial schema. Safe to run every startup."""
    new_cols = [
        # (table, column, sqlite_type)
        ("activities",         "calories_burned",     "REAL"),
        ("activities",         "distance_km",         "REAL"),
        ("activities",         "steps",               "INTEGER"),
        ("activities",         "avg_pace_min_per_km", "REAL"),
        ("activities",         "avg_heart_rate_bpm",  "INTEGER"),
        ("activities",         "cardio_load",         "REAL"),
        ("activities",         "elevation_gain_m",    "REAL"),
        ("activities",         "active_zone_min",     "INTEGER"),
        ("glucose_predictions","sarimax_peak_delta",  "REAL"),
        ("glucose_predictions","model_run_tag",       "TEXT"),
        ("model_runs",         "run_tag",             "TEXT"),
    ]
    with sqlite3.connect(DB_PATH) as conn:
        for table, col, col_type in new_cols:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # column already exists


def _seed_food_lookup():
    """On first start, import food_lookup.csv into the food_items table if empty."""
    db = SessionLocal()
    try:
        if db.query(FoodItem).count() > 0:
            return
        csv_path = os.path.join(DATA_RAW, "food_lookup.csv")
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            item = FoodItem(
                dish_key     = str(row.get("dish_key", row.get("display_name", ""))),
                display_name = str(row.get("display_name", row.get("dish_key", ""))),
                carbs_g      = float(row["carbs_g"])    if pd.notna(row.get("carbs_g"))    else None,
                protein_g    = float(row["protein_g"])  if pd.notna(row.get("protein_g"))  else None,
                fat_g        = float(row["fat_g"])      if pd.notna(row.get("fat_g"))      else None,
                gl           = str(row["gl"])            if pd.notna(row.get("gl"))          else None,
                source       = "imported",
            )
            db.add(item)
        db.commit()
        print(f"Seeded {db.query(FoodItem).count()} food items from food_lookup.csv")
    finally:
        db.close()


def _seed_config():
    """Insert default config values if not already present."""
    defaults = [
        ("user_timezone",            "UTC", "IANA timezone name for display and input conversion"),
        ("spike_threshold_meal",     "30",  "Minimum glucose rise (mg/dL) to flag as a meal spike"),
        ("spike_threshold_activity", "15",  "Minimum glucose rise (mg/dL) to flag as an activity spike"),
    ]
    db = SessionLocal()
    try:
        for key, value, description in defaults:
            if not db.query(AppConfig).filter(AppConfig.key == key).first():
                db.add(AppConfig(key=key, value=value, description=description))
        db.commit()
    finally:
        db.close()


app = FastAPI(title="CGM Data Collector", lifespan=lifespan)

app.include_router(dashboard.router)
app.include_router(meals.router)
app.include_router(activities.router)
app.include_router(cgm.router)
app.include_router(food_lookup.router)
app.include_router(training.router)
app.include_router(spikes.router)
app.include_router(config.router)
app.include_router(predictions.router)
app.include_router(medications.router)
app.include_router(forecast.router)
app.include_router(weekly.router)
