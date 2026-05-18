import json
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.models.meal import Meal
from app.models.food_item import FoodItem
from app.shared_templates import templates

router = APIRouter(prefix="/meals", tags=["meals"])


def _f(v: Optional[str]) -> Optional[float]:
    """Convert form string to float, treating blank as None."""
    if v is None or str(v).strip() == "":
        return None
    return float(v)


@router.get("/", response_class=HTMLResponse)
def meals_index(request: Request, db: Session = Depends(get_db)):
    week_ago = datetime.utcnow() - timedelta(days=7)
    meals = (
        db.query(Meal)
        .filter(Meal.ts >= week_ago)
        .order_by(Meal.ts.desc())
        .all()
    )
    return templates.TemplateResponse("meals/index.html", {
        "request": request, "meals": meals,
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
        "now":        ts or datetime.utcnow().strftime("%Y-%m-%dT%H:%M"),
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
        ts=datetime.fromisoformat(ts),
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
        "now":        meal.ts.strftime("%Y-%m-%dT%H:%M"),
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
        meal.ts                    = datetime.fromisoformat(ts)
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
