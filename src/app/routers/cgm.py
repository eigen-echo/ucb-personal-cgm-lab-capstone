from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.cgm import CGMReading
from app.services.cgm_parser import parse
from app.services.spike_detector import detect_and_insert
from app.shared_templates import templates

router = APIRouter(prefix="/cgm", tags=["cgm"])


@router.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request):
    return templates.TemplateResponse("cgm_upload.html", {"request": request, "result": None})


@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    result   = {"filename": file.filename, "added": 0, "skipped": 0, "error": None}

    try:
        rows = parse(contents)
    except ValueError as exc:
        result["error"] = str(exc)
        return templates.TemplateResponse("cgm_upload.html", {"request": request, "result": result})

    for row in rows:
        try:
            db.add(CGMReading(ts=row["ts"], glucose_mg_dl=row["glucose_mg_dl"]))
            db.flush()
            result["added"] += 1
        except IntegrityError:
            db.rollback()
            result["skipped"] += 1

    db.commit()
    result["date_range"] = (
        f"{rows[0]['ts'].date()} → {rows[-1]['ts'].date()}" if rows else "—"
    )

    if rows:
        result["new_spikes"] = detect_and_insert(
            db, rows[0]["ts"], rows[-1]["ts"]
        )

    return templates.TemplateResponse("cgm_upload.html", {"request": request, "result": result})
