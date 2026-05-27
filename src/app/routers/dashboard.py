from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

import bisect
import json
import pytz
from app.database import get_db
from app.models.cgm import CGMReading
from app.models.meal import Meal
from app.models.activity import Activity
from app.models.model_run import ModelRun
from app.services.timezone import get_cached_tz
from app.shared_templates import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    now      = datetime.utcnow()
    week_ago = now - timedelta(days=7)

    # Determine "today" in the user's local timezone, then convert midnight back to UTC
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

    # Last CGM reading
    last_cgm = db.query(CGMReading).order_by(CGMReading.ts.desc()).first()

    # Today's meals and activities (filtered from local midnight, converted to UTC)
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
        .limit(150)
        .all()
    )
    runs_ridge    = [r for r in runs if r.model_name == "ridge_per_meal"][:30][::-1]
    runs_rf       = [r for r in runs if r.model_name == "rf_per_meal"][:30][::-1]
    runs_sarimax  = [r for r in runs if r.model_name == "sarimax_hourly"][:30][::-1]
    runs_lstm     = [r for r in runs if r.model_name == "lstm_per_meal"][:30][::-1]
    runs_lstm5min = [r for r in runs if r.model_name == "lstm_5min_forecaster"][:30][::-1]

    last_run = runs[0] if runs else None

    # ── Heatmap data: past 91 days aggregated by local calendar date ──────────
    heatmap_start = now - timedelta(days=91)

    meals_recent = (
        db.query(Meal).filter(Meal.ts >= heatmap_start).all()
    )
    carbs_by_date: dict[str, float] = {}
    for m in meals_recent:
        d = m.ts.replace(tzinfo=pytz.utc).astimezone(tz).date().isoformat()
        carbs_by_date[d] = carbs_by_date.get(d, 0) + (
            m.carbs_grams_logged or m.carbs_grams_estimated or 0
        )

    acts_recent = (
        db.query(Activity).filter(Activity.ts >= heatmap_start).all()
    )
    walk_by_date: dict[str, float] = {}
    for a in acts_recent:
        d = a.ts.replace(tzinfo=pytz.utc).astimezone(tz).date().isoformat()
        walk_by_date[d] = walk_by_date.get(d, 0) + (a.duration_min or 0)

    return templates.TemplateResponse("dashboard.html", {
        "request":      request,
        "mean_glucose": mean_glucose,
        "tir":          tir,
        "last_cgm":     last_cgm,
        "last_run":     last_run,
        "today_meals":  today_meals,
        "today_acts":   today_acts,
        "runs_ridge":      runs_ridge,
        "runs_rf":         runs_rf,
        "runs_sarimax":    runs_sarimax,
        "runs_lstm":       runs_lstm,
        "runs_lstm5min":   runs_lstm5min,
        "carbs_by_date":   json.dumps(carbs_by_date),
        "walk_by_date":    json.dumps(walk_by_date),
    })


@router.get("/api/weekly-cgm", response_class=JSONResponse)
def weekly_cgm_api(
    week_offset: int = Query(default=0, ge=-52, le=0),
    db: Session = Depends(get_db),
):
    """Return one week of CGM readings + meals + activities, grouped by day.

    week_offset=0  → current Mon–Sun week
    week_offset=-1 → previous week, etc.
    Times are expressed as fractional hours in the user's local timezone.
    Meal / activity markers include the nearest CGM glucose so they sit on the line.
    """
    tz          = get_cached_tz()
    today_local = datetime.now(tz).date()

    # Monday of the target week
    days_since_monday = today_local.weekday()          # 0=Mon … 6=Sun
    monday = today_local - timedelta(days=days_since_monday) + timedelta(weeks=week_offset)
    sunday = monday + timedelta(days=6)

    # UTC window for DB queries (add a 1-hour buffer each side for tz safety)
    utc_start = (
        tz.localize(datetime(monday.year, monday.month, monday.day))
        .astimezone(pytz.utc)
        .replace(tzinfo=None)
        - timedelta(hours=1)
    )
    utc_end = (
        tz.localize(datetime(sunday.year, sunday.month, sunday.day, 23, 59, 59))
        .astimezone(pytz.utc)
        .replace(tzinfo=None)
        + timedelta(hours=1)
    )

    # ── Fetch raw rows ────────────────────────────────────────────────────────
    cgm_rows  = (db.query(CGMReading)
                   .filter(CGMReading.ts >= utc_start, CGMReading.ts <= utc_end)
                   .order_by(CGMReading.ts).all())
    meal_rows = (db.query(Meal)
                   .filter(Meal.ts >= utc_start, Meal.ts <= utc_end)
                   .order_by(Meal.ts).all())
    act_rows  = (db.query(Activity)
                   .filter(Activity.ts >= utc_start, Activity.ts <= utc_end)
                   .order_by(Activity.ts).all())

    # ── Helper: UTC → local fractional hours ─────────────────────────────────
    def to_local(ts_utc: datetime):
        return ts_utc.replace(tzinfo=pytz.utc).astimezone(tz)

    def frac_h(ts_local) -> float:
        return round(ts_local.hour + ts_local.minute / 60 + ts_local.second / 3600, 4)

    # ── Group CGM by local date ───────────────────────────────────────────────
    cgm_by_date: dict = {}
    for r in cgm_rows:
        loc = to_local(r.ts)
        d   = loc.date()
        if d not in cgm_by_date:
            cgm_by_date[d] = {"t": [], "g": []}
        cgm_by_date[d]["t"].append(frac_h(loc))
        cgm_by_date[d]["g"].append(r.glucose_mg_dl)

    def nearest_glucose(d, t: float) -> float:
        """Return the CGM glucose value closest in time to fractional hour t."""
        pts = cgm_by_date.get(d)
        if not pts or not pts["t"]:
            return 150.0
        ts_list = pts["t"]
        idx = bisect.bisect_left(ts_list, t)
        if idx == 0:
            return pts["g"][0]
        if idx >= len(ts_list):
            return pts["g"][-1]
        # Pick whichever neighbour is closer
        if abs(ts_list[idx] - t) < abs(ts_list[idx - 1] - t):
            return pts["g"][idx]
        return pts["g"][idx - 1]

    # ── Build one entry per day (Mon … Sun) ──────────────────────────────────
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    days      = []

    for i in range(7):
        d    = monday + timedelta(days=i)
        pts  = cgm_by_date.get(d, {"t": [], "g": []})

        # CGM points - zip t and g into list of {t, g} dicts
        cgm_pts = [{"t": t, "g": round(g, 1)}
                   for t, g in zip(pts["t"], pts["g"])]

        # Meals for this day
        meals_out = []
        for m in meal_rows:
            loc = to_local(m.ts)
            if loc.date() != d:
                continue
            t    = frac_h(loc)
            name = (m.dish_names or "Meal").split(",")[0].strip()[:40]
            carbs = m.carbs_grams_logged or m.carbs_grams_estimated or 0
            meals_out.append({
                "t":     t,
                "g":     nearest_glucose(d, t),
                "name":  name,
                "carbs": round(float(carbs), 0) if carbs else 0,
            })

        # Activities for this day
        acts_out = []
        for a in act_rows:
            loc = to_local(a.ts)
            if loc.date() != d:
                continue
            t = frac_h(loc)
            acts_out.append({
                "t":        t,
                "g":        nearest_glucose(d, t),
                "type":     a.activity_type or "Activity",
                "duration": a.duration_min or 0,
            })

        days.append({
            "label":     day_names[i],
            "date":      d.isoformat(),
            "is_today":  d == today_local,
            "is_future": d > today_local,
            "cgm":       cgm_pts,
            "meals":     meals_out,
            "activities": acts_out,
        })

    week_label = f"{monday.strftime('%b %d')} – {sunday.strftime('%b %d, %Y')}"
    return {"week_label": week_label, "week_offset": week_offset, "days": days}
