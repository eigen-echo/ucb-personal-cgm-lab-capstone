import io
import os
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.model_run import ModelRun
from app.services import pipeline
from app.services.export import export_all, DATA_RAW
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


@router.get("/download-raw")
def download_raw(db: Session = Depends(get_db)):
    """Export all tables to data/raw/ CSVs and stream them as a single zip download."""
    export_all(db)

    buf = io.BytesIO()
    csv_files = [
        "cgm_5min.csv",
        "meals_v2.csv",
        "activities.csv",
        "medications.csv",
        "fasting_windows.csv",
        "food_lookup.csv",
    ]
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in csv_files:
            path = os.path.join(DATA_RAW, fname)
            if os.path.exists(path):
                zf.write(path, arcname=fname)
    buf.seek(0)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"cgm_raw_data_{stamp}.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
