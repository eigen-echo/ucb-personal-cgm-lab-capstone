from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models.cgm import CGMReading
from app.models.meal import Meal
from app.models.activity import Activity
from app.models.model_run import ModelRun
from app.shared_templates import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    now   = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    today    = now.date()

    # 7-day mean glucose
    recent_cgm = db.query(CGMReading).filter(CGMReading.ts >= week_ago).all()
    mean_glucose = (
        round(sum(r.glucose_mg_dl for r in recent_cgm) / len(recent_cgm), 1)
        if recent_cgm else None
    )

    # Time in range 70-180
    if recent_cgm:
        in_range = sum(1 for r in recent_cgm if 70 <= r.glucose_mg_dl <= 180)
        tir = round(in_range / len(recent_cgm) * 100, 1)
    else:
        tir = None

    # Last CGM reading
    last_cgm = db.query(CGMReading).order_by(CGMReading.ts.desc()).first()

    # Today's meals and activities
    today_start = datetime.combine(today, datetime.min.time())
    today_meals = (
        db.query(Meal)
        .filter(Meal.ts >= today_start)
        .order_by(Meal.ts.desc())
        .all()
    )
    today_acts = (
        db.query(Activity)
        .filter(Activity.ts >= today_start)
        .order_by(Activity.ts.desc())
        .all()
    )

    # Model run history for chart (last 30 runs per model)
    runs = (
        db.query(ModelRun)
        .order_by(ModelRun.run_ts.desc())
        .limit(60)
        .all()
    )
    runs_ridge   = [r for r in runs if r.model_name == "ridge_per_meal"][:30][::-1]
    runs_sarimax = [r for r in runs if r.model_name == "sarimax_hourly"][:30][::-1]

    last_run = runs[0] if runs else None

    return templates.TemplateResponse("dashboard.html", {
        "request":      request,
        "mean_glucose": mean_glucose,
        "tir":          tir,
        "last_cgm":     last_cgm,
        "last_run":     last_run,
        "today_meals":  today_meals,
        "today_acts":   today_acts,
        "runs_ridge":   runs_ridge,
        "runs_sarimax": runs_sarimax,
    })
