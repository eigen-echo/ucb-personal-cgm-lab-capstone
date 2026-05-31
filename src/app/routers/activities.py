import pytz
from datetime import date, datetime, timedelta
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.models.activity import Activity
from app.services.timezone import get_cached_tz, local_to_utc, now_local_str
from app.shared_templates import templates

router = APIRouter(prefix="/activities", tags=["activities"])


def _week_bounds_utc(week_param: Optional[str], tz):
    now_local      = datetime.now(tz)
    today_local    = now_local.date()
    current_monday = today_local - timedelta(days=today_local.weekday())

    if week_param:
        try:
            week_start_date = datetime.strptime(week_param, "%Y-%m-%d").date()
        except ValueError:
            week_start_date = current_monday
    else:
        week_start_date = current_monday

    def to_utc(d: date):
        return tz.localize(datetime(d.year, d.month, d.day)).astimezone(pytz.utc).replace(tzinfo=None)

    return to_utc(week_start_date), to_utc(week_start_date + timedelta(days=7)), week_start_date, week_start_date >= current_monday


@router.get("/", response_class=HTMLResponse)
def activities_index(request: Request, week: Optional[str] = None, db: Session = Depends(get_db)):
    tz = get_cached_tz()
    start_utc, end_utc, week_start_date, is_current = _week_bounds_utc(week, tz)
    week_end_date = week_start_date + timedelta(days=6)

    activities = (
        db.query(Activity)
        .filter(Activity.ts >= start_utc, Activity.ts < end_utc)
        .order_by(Activity.ts.desc())
        .all()
    )

    prev_week  = (week_start_date - timedelta(days=7)).isoformat()
    next_week  = (week_start_date + timedelta(days=7)).isoformat()
    week_label = f"{week_start_date.strftime('%b %d')} – {week_end_date.strftime('%b %d, %Y')}"

    return templates.TemplateResponse("activities/index.html", {
        "request":    request,
        "activities": activities,
        "week_label": week_label,
        "prev_week":  prev_week,
        "next_week":  next_week if not is_current else None,
        "week":       week_start_date.isoformat(),
    })


@router.get("/new", response_class=HTMLResponse)
def activity_form(request: Request, ts: Optional[str] = None):
    return templates.TemplateResponse("activities/form.html", {
        "request":  request,
        "now":      ts or now_local_str(get_cached_tz()),
        "activity": None,
    })


def _f(v: Optional[str]) -> Optional[float]:
    """Blank/missing form string → None, otherwise float."""
    return float(v) if v and v.strip() else None


def _i(v: Optional[str]) -> Optional[int]:
    """Blank/missing form string → None, otherwise int."""
    return int(float(v)) if v and v.strip() else None


def _pace(v: Optional[str]) -> Optional[float]:
    """Parse pace as 'mm:ss' or decimal minutes. '14:18' → 14.30, '14.5' → 14.5."""
    if not v or not v.strip():
        return None
    v = v.strip()
    if ":" in v:
        try:
            mm, ss = v.split(":", 1)
            return int(mm) + int(ss) / 60
        except ValueError:
            return None
    try:
        return float(v)
    except ValueError:
        return None


@router.post("/new")
def activity_create(
    ts:                  str           = Form(...),
    activity_type:       str           = Form("walk"),
    duration_min:        float         = Form(...),
    notes:               Optional[str] = Form(None),
    calories_burned:     Optional[str] = Form(None),
    distance_km:         Optional[str] = Form(None),
    steps:               Optional[str] = Form(None),
    avg_pace_min_per_km: Optional[str] = Form(None),
    avg_heart_rate_bpm:  Optional[str] = Form(None),
    cardio_load:         Optional[str] = Form(None),
    elevation_gain_m:    Optional[str] = Form(None),
    active_zone_min:     Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    act = Activity(
        ts=local_to_utc(ts, get_cached_tz()),
        activity_type=activity_type,
        duration_min=duration_min,
        notes=notes or None,
        calories_burned     = _f(calories_burned),
        distance_km         = _f(distance_km),
        steps               = _i(steps),
        avg_pace_min_per_km = _pace(avg_pace_min_per_km),
        avg_heart_rate_bpm  = _i(avg_heart_rate_bpm),
        cardio_load         = _f(cardio_load),
        elevation_gain_m    = _f(elevation_gain_m),
        active_zone_min     = _i(active_zone_min),
    )
    db.add(act)
    db.commit()
    return RedirectResponse("/activities/", status_code=303)


@router.post("/{act_id}/delete")
def activity_delete(act_id: int, db: Session = Depends(get_db)):
    act = db.query(Activity).filter(Activity.id == act_id).first()
    if act:
        db.delete(act)
        db.commit()
    return RedirectResponse("/activities/", status_code=303)
