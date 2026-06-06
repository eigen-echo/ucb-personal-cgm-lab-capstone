"""
/weekly/ — 7-day CGM overlay chart (moved from the dashboard).

GET  /weekly/        Full page with the day-by-day overlay chart.
GET  /weekly/data    JSON: one week of CGM + meals + activities (formerly /api/weekly-cgm).
"""
import bisect
from datetime import datetime, timedelta

import pytz
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.activity import Activity
from app.models.cgm import CGMReading
from app.models.meal import Meal
from app.services.timezone import get_cached_tz
from app.shared_templates import templates

router = APIRouter(prefix="/weekly", tags=["weekly"])


@router.get("/", response_class=HTMLResponse)
def weekly_page(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("weekly.html", {"request": request})


@router.get("/data", response_class=JSONResponse)
def weekly_cgm_data(
    week_offset: int = Query(default=0, ge=-52, le=0),
    db: Session = Depends(get_db),
):
    """Return one week of CGM readings + meals + activities, grouped by local day.

    week_offset=0  → current Mon–Sun week
    week_offset=-1 → previous week, etc.
    Times are expressed as fractional hours in the user's local timezone.
    Meal/activity markers include the nearest CGM glucose so they sit on the trace.
    """
    tz          = get_cached_tz()
    today_local = datetime.now(tz).date()

    days_since_monday = today_local.weekday()
    monday = today_local - timedelta(days=days_since_monday) + timedelta(weeks=week_offset)
    sunday = monday + timedelta(days=6)

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

    cgm_rows  = (db.query(CGMReading)
                   .filter(CGMReading.ts >= utc_start, CGMReading.ts <= utc_end)
                   .order_by(CGMReading.ts).all())
    meal_rows = (db.query(Meal)
                   .filter(Meal.ts >= utc_start, Meal.ts <= utc_end)
                   .order_by(Meal.ts).all())
    act_rows  = (db.query(Activity)
                   .filter(Activity.ts >= utc_start, Activity.ts <= utc_end)
                   .order_by(Activity.ts).all())

    def to_local(ts_utc):
        return ts_utc.replace(tzinfo=pytz.utc).astimezone(tz)

    def frac_h(ts_local) -> float:
        return round(ts_local.hour + ts_local.minute / 60 + ts_local.second / 3600, 4)

    cgm_by_date: dict = {}
    for r in cgm_rows:
        loc = to_local(r.ts)
        d   = loc.date()
        if d not in cgm_by_date:
            cgm_by_date[d] = {"t": [], "g": []}
        cgm_by_date[d]["t"].append(frac_h(loc))
        cgm_by_date[d]["g"].append(r.glucose_mg_dl)

    def nearest_glucose(d, t: float) -> float:
        pts = cgm_by_date.get(d)
        if not pts or not pts["t"]:
            return 150.0
        ts_list = pts["t"]
        idx = bisect.bisect_left(ts_list, t)
        if idx == 0:
            return pts["g"][0]
        if idx >= len(ts_list):
            return pts["g"][-1]
        if abs(ts_list[idx] - t) < abs(ts_list[idx - 1] - t):
            return pts["g"][idx]
        return pts["g"][idx - 1]

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    days      = []

    for i in range(7):
        d   = monday + timedelta(days=i)
        pts = cgm_by_date.get(d, {"t": [], "g": []})

        cgm_pts = [{"t": t, "g": round(g, 1)}
                   for t, g in zip(pts["t"], pts["g"])]

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
            "label":      day_names[i],
            "date":       d.isoformat(),
            "is_today":   d == today_local,
            "is_future":  d > today_local,
            "cgm":        cgm_pts,
            "meals":      meals_out,
            "activities": acts_out,
        })

    week_label = f"{monday.strftime('%b %d')} – {sunday.strftime('%b %d, %Y')}"
    return {"week_label": week_label, "week_offset": week_offset, "days": days}
