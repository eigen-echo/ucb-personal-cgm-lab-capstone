from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.models.activity import Activity
from app.shared_templates import templates

router = APIRouter(prefix="/activities", tags=["activities"])


@router.get("/", response_class=HTMLResponse)
def activities_index(request: Request, db: Session = Depends(get_db)):
    week_ago   = datetime.utcnow() - timedelta(days=7)
    activities = (
        db.query(Activity)
        .filter(Activity.ts >= week_ago)
        .order_by(Activity.ts.desc())
        .all()
    )
    return templates.TemplateResponse("activities/index.html", {
        "request": request, "activities": activities,
    })


@router.get("/new", response_class=HTMLResponse)
def activity_form(request: Request, ts: Optional[str] = None):
    return templates.TemplateResponse("activities/form.html", {
        "request":  request,
        "now":      ts or datetime.utcnow().strftime("%Y-%m-%dT%H:%M"),
        "activity": None,
    })


@router.post("/new")
def activity_create(
    ts:            str            = Form(...),
    activity_type: str            = Form("walk"),
    duration_min:  float          = Form(...),
    notes:         Optional[str]  = Form(None),
    db: Session = Depends(get_db),
):
    act = Activity(
        ts=datetime.fromisoformat(ts),
        activity_type=activity_type,
        duration_min=duration_min,
        notes=notes,
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
