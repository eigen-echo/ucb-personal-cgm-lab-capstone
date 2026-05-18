from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.model_run import ModelRun
from app.services import pipeline
from app.shared_templates import templates

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/", response_class=HTMLResponse)
def training_page(request: Request, db: Session = Depends(get_db)):
    runs = (
        db.query(ModelRun)
        .order_by(ModelRun.run_ts.desc())
        .limit(100)
        .all()
    )
    return templates.TemplateResponse("training.html", {
        "request": request,
        "runs":    runs,
        "status":  pipeline.status,
    })


@router.post("/run")
async def training_run(
    request: Request,
    db: Session = Depends(get_db),
):
    form    = await request.form()
    models  = form.getlist("models") or ["ridge"]

    if pipeline.status["running"]:
        return RedirectResponse("/training/?error=already_running", status_code=303)

    pipeline.run_in_background(db, models)
    return RedirectResponse("/training/", status_code=303)


@router.get("/status", response_class=JSONResponse)
def training_status():
    return pipeline.status
