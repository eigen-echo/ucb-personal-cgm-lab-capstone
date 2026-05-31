"""
/forecast/ — glucose trajectory visualisation.

GET  /forecast/        Form page: enter carbs + optional walk, see the full
                       3-hour trajectory (SARIMAX 3 hourly pts + LSTM 36 × 5-min pts),
                       with a walk-vs-no-walk overlay on a single Chart.js chart.

POST /forecast/preview  Returns JSON trajectory data; consumed by the page's JS.
                        No DB writes — this is pure visualisation.
"""
from datetime import datetime, timedelta

import pytz
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.cgm import CGMReading
from app.models.meal import Meal
from app.services import predictor
from app.services.timezone import get_cached_tz
from app.shared_templates import templates

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/", response_class=HTMLResponse)
def forecast_page(request: Request, db: Session = Depends(get_db)):
    last_cgm  = db.query(CGMReading).order_by(CGMReading.ts.desc()).first()
    last_meal = db.query(Meal).order_by(Meal.ts.desc()).first()

    now_utc         = datetime.utcnow()
    cgm_age_min     = None
    cold_start      = True
    current_glucose = ""

    if last_cgm:
        cgm_age_min     = round((now_utc - last_cgm.ts).total_seconds() / 60)
        cold_start      = cgm_age_min > 30
        current_glucose = int(round(last_cgm.glucose_mg_dl))

    # Pre-fill carbs from last meal if it was within 2 h
    last_meal_carbs = ""
    if last_meal:
        age_m = (now_utc - last_meal.ts).total_seconds() / 60
        if age_m < 120:
            c = last_meal.carbs_grams_logged or last_meal.carbs_grams_estimated
            if c:
                last_meal_carbs = int(round(float(c)))

    tz            = get_cached_tz()
    now_local_str = datetime.now(tz).strftime("%Y-%m-%dT%H:%M")

    return templates.TemplateResponse("forecast.html", {
        "request":         request,
        "current_glucose": current_glucose,
        "cold_start":      cold_start,
        "cgm_age_min":     cgm_age_min,
        "last_meal_carbs": last_meal_carbs,
        "now_local_str":   now_local_str,
    })


@router.post("/preview", response_class=JSONResponse)
async def forecast_preview(request: Request, db: Session = Depends(get_db)):
    form = await request.form()

    try:
        current_glucose = float(form.get("current_glucose") or 100)
        carbs_g         = float(form.get("carbs_g") or 0)
        walk_min_after  = float(form.get("walk_min_after") or 0)
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid numeric input"}, status_code=400)

    # Parse datetime-local (local time) → naive UTC
    tz          = get_cached_tz()
    meal_ts_str = (form.get("meal_ts") or "").strip()
    try:
        naive_local = datetime.fromisoformat(meal_ts_str)
        meal_ts_utc = (
            tz.localize(naive_local).astimezone(pytz.utc).replace(tzinfo=None)
        )
    except Exception:
        meal_ts_utc = datetime.utcnow()

    # Medication timing (re-use helper from predictions router)
    from app.routers.predictions import _minutes_since_glipizide
    min_glip = _minutes_since_glipizide(db)

    result = predictor.forecast_trajectory(
        db                      = db,
        current_glucose         = current_glucose,
        carbs_g                 = carbs_g,
        walk_min_after          = walk_min_after,
        meal_ts                 = meal_ts_utc,
        minutes_since_glipizide = min_glip,
    )
    return JSONResponse(result)
