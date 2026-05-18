import os
import re
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.models.food_item import FoodItem
from app.shared_templates import templates

router   = APIRouter(prefix="/food", tags=["food"])
DATA_RAW = os.environ.get("DATA_RAW_DIR", "/app/data/raw")


def _make_key(display_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_")


def _f(v: Optional[str]) -> Optional[float]:
    if v is None or str(v).strip() == "":
        return None
    return float(v)


@router.get("/", response_class=HTMLResponse)
def food_index(request: Request, q: str = "", db: Session = Depends(get_db)):
    query = db.query(FoodItem).order_by(FoodItem.display_name)
    if q:
        query = query.filter(FoodItem.display_name.ilike(f"%{q}%"))
    items = query.all()
    return templates.TemplateResponse("food_lookup.html", {
        "request": request, "items": items, "q": q,
    })


@router.post("/add")
def food_add(
    display_name: str          = Form(...),
    carbs_g:      Optional[str] = Form(None),
    protein_g:    Optional[str] = Form(None),
    fat_g:        Optional[str] = Form(None),
    gl:           Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    key  = _make_key(display_name)
    item = FoodItem(
        dish_key=key,
        display_name=display_name,
        carbs_g=_f(carbs_g),
        protein_g=_f(protein_g),
        fat_g=_f(fat_g),
        gl=gl or None,
        source="manual",
    )
    db.add(item)
    db.commit()
    return RedirectResponse("/food/", status_code=303)


@router.post("/{item_id}/delete")
def food_delete(item_id: int, db: Session = Depends(get_db)):
    item = db.query(FoodItem).filter(FoodItem.id == item_id).first()
    if item:
        db.delete(item)
        db.commit()
    return RedirectResponse("/food/", status_code=303)
