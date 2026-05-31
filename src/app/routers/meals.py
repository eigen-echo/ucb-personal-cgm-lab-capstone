import json
import pytz
from datetime import date, datetime, timedelta
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.models.meal import Meal
from app.models.food_item import FoodItem
from app.services.timezone import get_cached_tz, local_to_utc, now_local_str, to_local
from app.shared_templates import templates

router = APIRouter(prefix="/meals", tags=["meals"])


def _week_bounds_utc(week_param: Optional[str], tz):
    """Return (week_start_utc, week_end_utc, week_start_date, is_current_or_future)."""
    now_local   = datetime.now(tz)
    today_local = now_local.date()
    current_monday = today_local - timedelta(days=today_local.weekday())

    if week_param:
        try:
            week_start_date = datetime.strptime(week_param, "%Y-%m-%d").date()
        except ValueError:
            week_start_date = current_monday
    else:
        week_start_date = current_monday

    week_end_date = week_start_date + timedelta(days=7)

    def to_utc(d: date):
        return tz.localize(datetime(d.year, d.month, d.day)).astimezone(pytz.utc).replace(tzinfo=None)

    return to_utc(week_start_date), to_utc(week_end_date), week_start_date, week_start_date >= current_monday


def _f(v: Optional[str]) -> Optional[float]:
    """Convert form string to float, treating blank as None."""
    if v is None or str(v).strip() == "":
        return None
    return float(v)


@router.get("/", response_class=HTMLResponse)
def meals_index(request: Request, week: Optional[str] = None, db: Session = Depends(get_db)):
    tz = get_cached_tz()
    start_utc, end_utc, week_start_date, is_current = _week_bounds_utc(week, tz)
    week_end_date = week_start_date + timedelta(days=6)

    meals = (
        db.query(Meal)
        .filter(Meal.ts >= start_utc, Meal.ts < end_utc)
        .order_by(Meal.ts.desc())
        .all()
    )

    prev_week = (week_start_date - timedelta(days=7)).isoformat()
    next_week = (week_start_date + timedelta(days=7)).isoformat()
    week_label = (
        f"{week_start_date.strftime('%b %d')} – {week_end_date.strftime('%b %d, %Y')}"
    )

    return templates.TemplateResponse("meals/index.html", {
        "request":    request,
        "meals":      meals,
        "week_label": week_label,
        "prev_week":  prev_week,
        "next_week":  next_week if not is_current else None,
        "week":       week_start_date.isoformat(),
    })


@router.get("/new", response_class=HTMLResponse)
def meal_form(request: Request, ts: Optional[str] = None, db: Session = Depends(get_db)):
    food_items = db.query(FoodItem).order_by(FoodItem.display_name).all()
    food_json  = json.dumps([
        {"display_name": f.display_name, "carbs_g": f.carbs_g}
        for f in food_items
    ])
    return templates.TemplateResponse("meals/form.html", {
        "request":    request,
        "food_items": food_items,
        "food_json":  food_json,
        "now":        ts or now_local_str(get_cached_tz()),
        "meal":       None,
    })


@router.post("/new")
def meal_create(
    request: Request,
    ts:                    str           = Form(...),
    dish_names:            Optional[str] = Form(None),
    meal_type:             Optional[str] = Form(None),
    carbs_grams_logged:    Optional[str] = Form(None),
    carbs_grams_estimated: Optional[str] = Form(None),
    fasted_meal:           bool          = Form(False),
    notes:                 Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    meal = Meal(
        ts=local_to_utc(ts, get_cached_tz()),
        dish_names=dish_names,
        meal_type=meal_type,
        carbs_grams_logged=_f(carbs_grams_logged),
        carbs_grams_estimated=_f(carbs_grams_estimated),
        fasted_meal=fasted_meal,
        notes=notes,
    )
    db.add(meal)
    db.commit()
    return RedirectResponse("/meals/", status_code=303)


@router.get("/{meal_id}/edit", response_class=HTMLResponse)
def meal_edit_form(meal_id: int, request: Request, db: Session = Depends(get_db)):
    meal       = db.query(Meal).filter(Meal.id == meal_id).first()
    food_items = db.query(FoodItem).order_by(FoodItem.display_name).all()
    food_json  = json.dumps([
        {"display_name": f.display_name, "carbs_g": f.carbs_g}
        for f in food_items
    ])
    return templates.TemplateResponse("meals/form.html", {
        "request":    request,
        "food_items": food_items,
        "food_json":  food_json,
        "now":        to_local(meal.ts, get_cached_tz(), "%Y-%m-%dT%H:%M"),
        "meal":       meal,
    })


@router.post("/{meal_id}/edit")
def meal_update(
    meal_id:               int,
    ts:                    str           = Form(...),
    dish_names:            Optional[str] = Form(None),
    meal_type:             Optional[str] = Form(None),
    carbs_grams_logged:    Optional[str] = Form(None),
    carbs_grams_estimated: Optional[str] = Form(None),
    fasted_meal:           bool          = Form(False),
    notes:                 Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    meal = db.query(Meal).filter(Meal.id == meal_id).first()
    if meal:
        meal.ts                    = local_to_utc(ts, get_cached_tz())
        meal.dish_names            = dish_names
        meal.meal_type             = meal_type
        meal.carbs_grams_logged    = _f(carbs_grams_logged)
        meal.carbs_grams_estimated = _f(carbs_grams_estimated)
        meal.fasted_meal           = fasted_meal
        meal.notes                 = notes
        db.commit()
    return RedirectResponse("/meals/", status_code=303)


@router.post("/{meal_id}/delete")
def meal_delete(meal_id: int, db: Session = Depends(get_db)):
    meal = db.query(Meal).filter(Meal.id == meal_id).first()
    if meal:
        db.delete(meal)
        db.commit()
    return RedirectResponse("/meals/", status_code=303)
