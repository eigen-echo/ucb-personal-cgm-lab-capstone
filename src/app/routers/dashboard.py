from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

import json
import pytz
from app.database import get_db
from app.models.activity import Activity
from app.models.cgm import CGMReading
from app.models.meal import Meal
from app.models.model_run import ModelRun
from app.services.timezone import get_cached_tz
from app.shared_templates import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    now      = datetime.utcnow()
    week_ago = now - timedelta(days=7)

    tz          = get_cached_tz()
    now_local   = datetime.now(tz)
    today_local = now_local.date()
    today_start = tz.localize(
        datetime(today_local.year, today_local.month, today_local.day)
    ).astimezone(pytz.utc).replace(tzinfo=None)

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

    last_cgm  = db.query(CGMReading).order_by(CGMReading.ts.desc()).first()
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
    last_run = db.query(ModelRun).order_by(ModelRun.run_ts.desc()).first()

    # ── Heatmap data: past 91 days aggregated by local calendar date ──────────
    heatmap_start = now - timedelta(days=91)

    meals_recent = db.query(Meal).filter(Meal.ts >= heatmap_start).all()
    carbs_by_date: dict[str, float] = {}
    for m in meals_recent:
        d = m.ts.replace(tzinfo=pytz.utc).astimezone(tz).date().isoformat()
        carbs_by_date[d] = carbs_by_date.get(d, 0) + (
            m.carbs_grams_logged or m.carbs_grams_estimated or 0
        )

    acts_recent = db.query(Activity).filter(Activity.ts >= heatmap_start).all()
    walk_by_date: dict[str, float] = {}
    for a in acts_recent:
        d = a.ts.replace(tzinfo=pytz.utc).astimezone(tz).date().isoformat()
        walk_by_date[d] = walk_by_date.get(d, 0) + (a.duration_min or 0)

    return templates.TemplateResponse("dashboard.html", {
        "request":       request,
        "mean_glucose":  mean_glucose,
        "tir":           tir,
        "last_cgm":      last_cgm,
        "last_run":      last_run,
        "today_meals":   today_meals,
        "today_acts":    today_acts,
        "carbs_by_date": json.dumps(carbs_by_date),
        "walk_by_date":  json.dumps(walk_by_date),
    })


@router.get("/api/daily-pattern", response_class=JSONResponse)
def daily_pattern_api(
    days: int = Query(default=14, ge=7, le=90),
    db: Session = Depends(get_db),
):
    """Return 15-min bucket statistics (avg, p15, p75) averaged over the last N days.

    Each bucket spans 15 minutes of the day (96 buckets total, 0 = midnight).
    Only buckets with at least 3 readings are returned.
    """
    since = datetime.utcnow() - timedelta(days=days)
    rows  = db.query(CGMReading).filter(CGMReading.ts >= since).all()

    if not rows:
        return {"days": days, "buckets": []}

    tz = get_cached_tz()

    # Accumulate values per 15-min bucket
    bucket_vals: dict[int, list] = {i: [] for i in range(96)}
    for r in rows:
        local_ts = r.ts.replace(tzinfo=pytz.utc).astimezone(tz)
        bucket   = (local_ts.hour * 60 + local_ts.minute) // 15
        bucket_vals[bucket].append(r.glucose_mg_dl)

    def _percentile(sorted_vals: list, p: float) -> float:
        """Linear interpolation percentile on a pre-sorted list."""
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        idx  = (n - 1) * p / 100
        lo   = int(idx)
        hi   = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    def _fmt(h: int, m: int) -> str:
        ampm = "AM" if h < 12 else "PM"
        h12  = h % 12 or 12
        return f"{h12}:{m:02d} {ampm}"

    buckets = []
    for i in range(96):
        vals = bucket_vals[i]
        if len(vals) < 3:          # skip buckets with too few readings
            continue
        sv       = sorted(vals)
        hour     = (i * 15) // 60
        minute   = (i * 15) % 60
        end_i    = (i + 1) % 96
        end_hour = (end_i * 15) // 60
        end_min  = (end_i * 15) % 60

        avg = round(sum(sv) / len(sv), 1)
        p15 = round(_percentile(sv, 15), 1)
        p75 = round(_percentile(sv, 75), 1)

        buckets.append({
            "bucket":     i,
            "time_label": _fmt(hour, minute),
            "time_range": f"{_fmt(hour, minute)} – {_fmt(end_hour, end_min)}",
            "avg":        avg,
            "p15":        p15,
            "p75":        p75,
            "n":          len(sv),
        })

    return {"days": days, "buckets": buckets}
