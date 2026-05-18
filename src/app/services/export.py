"""Export SQLite tables to data/raw/ CSVs in the exact format the feature-build scripts expect."""
import os
import pandas as pd
from sqlalchemy.orm import Session

from app.models.cgm import CGMReading
from app.models.meal import Meal
from app.models.activity import Activity
from app.models.medication import Medication
from app.models.fasting_window import FastingWindow
from app.models.food_item import FoodItem

DATA_RAW = os.environ.get("DATA_RAW_DIR", "/app/data/raw")


def export_all(db: Session) -> dict:
    """Write all tables to data/raw/. Returns row counts per file."""
    os.makedirs(DATA_RAW, exist_ok=True)
    counts = {}
    counts["cgm_5min.csv"]         = _export_cgm(db)
    counts["meals_v2.csv"]         = _export_meals(db)
    counts["activities.csv"]       = _export_activities(db)
    counts["medications.csv"]      = _export_medications(db)
    counts["fasting_windows.csv"]  = _export_fasting_windows(db)
    counts["food_lookup.csv"]      = _export_food_lookup(db)
    return counts


def _save(df: pd.DataFrame, filename: str) -> int:
    df.to_csv(os.path.join(DATA_RAW, filename), index=False)
    return len(df)


def _export_cgm(db: Session) -> int:
    rows = db.query(CGMReading).order_by(CGMReading.ts).all()
    df = pd.DataFrame([{"ts": r.ts.isoformat(), "glucose_mg_dl": r.glucose_mg_dl} for r in rows])
    return _save(df, "cgm_5min.csv")


def _export_meals(db: Session) -> int:
    rows = db.query(Meal).order_by(Meal.ts).all()
    df = pd.DataFrame([{
        "ts":                    r.ts.isoformat(),
        "dish_names":            r.dish_names,
        "meal_type":             r.meal_type,
        "carbs_grams_logged":    r.carbs_grams_logged,
        "carbs_grams_estimated": r.carbs_grams_estimated,
        "fasted_meal":           r.fasted_meal,
        "notes":                 r.notes,
    } for r in rows])
    return _save(df, "meals_v2.csv")


def _export_activities(db: Session) -> int:
    rows = db.query(Activity).order_by(Activity.ts).all()
    df = pd.DataFrame([{
        "ts":           r.ts.isoformat(),
        "type":         r.activity_type,
        "duration_min": r.duration_min,
    } for r in rows])
    return _save(df, "activities.csv")


def _export_medications(db: Session) -> int:
    rows = db.query(Medication).order_by(Medication.scheduled_ts).all()
    df = pd.DataFrame([{
        "scheduled_ts": r.scheduled_ts.isoformat(),
        "drug":         r.drug,
        "dose_mg":      r.dose_mg,
        "taken":        r.taken,
    } for r in rows])
    return _save(df, "medications.csv")


def _export_fasting_windows(db: Session) -> int:
    rows = db.query(FastingWindow).order_by(FastingWindow.start_ts).all()
    df = pd.DataFrame([{
        "start_ts":    r.start_ts.isoformat(),
        "end_ts":      r.end_ts.isoformat() if r.end_ts else None,
        "window_type": r.window_type,
    } for r in rows])
    return _save(df, "fasting_windows.csv")


def _export_food_lookup(db: Session) -> int:
    rows = db.query(FoodItem).order_by(FoodItem.dish_key).all()
    df = pd.DataFrame([{
        "dish_key":     r.dish_key,
        "display_name": r.display_name,
        "carbs_g":      r.carbs_g,
        "protein_g":    r.protein_g,
        "fat_g":        r.fat_g,
        "gl":           r.gl,
    } for r in rows])
    return _save(df, "food_lookup.csv")
