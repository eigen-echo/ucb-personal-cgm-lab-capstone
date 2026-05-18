from datetime import datetime
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.app_config import AppConfig
from app.shared_templates import templates

router = APIRouter(prefix="/config", tags=["config"])

CONFIG_DEFS = [
    {
        "key":         "spike_threshold_meal",
        "label":       "Meal spike threshold (mg/dL)",
        "description": "Minimum glucose rise to flag a peak as a potential meal spike.",
        "default":     "30",
    },
    {
        "key":         "spike_threshold_activity",
        "label":       "Activity spike threshold (mg/dL)",
        "description": "Minimum glucose rise to flag a peak as a potential activity spike. Set lower than the meal threshold.",
        "default":     "15",
    },
]


@router.get("/", response_class=HTMLResponse)
def config_page(request: Request, saved: str = "", db: Session = Depends(get_db)):
    stored = {r.key: r.value for r in db.query(AppConfig).all()}
    items  = [{**d, "value": stored.get(d["key"], d["default"])} for d in CONFIG_DEFS]
    return templates.TemplateResponse("config.html", {
        "request": request,
        "items":   items,
        "saved":   saved == "1",
    })


@router.post("/save")
async def config_save(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    for defn in CONFIG_DEFS:
        key = defn["key"]
        val = str(form.get(key, defn["default"]))
        row = db.query(AppConfig).filter(AppConfig.key == key).first()
        if row:
            row.value      = val
            row.updated_at = datetime.utcnow()
        else:
            db.add(AppConfig(key=key, value=val, description=defn["description"]))
    db.commit()
    return RedirectResponse("/config/?saved=1", status_code=303)
