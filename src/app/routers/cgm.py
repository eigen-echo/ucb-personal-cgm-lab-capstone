import pytz

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.cgm import CGMReading
from app.models.spike_event import SpikeEvent
from app.services.cgm_parser import parse
from app.services.match_predictions import match_predictions
from app.services.spike_detector import detect_and_insert
from app.services.timezone import get_tz, TIMEZONE_CHOICES
from app.shared_templates import templates

router = APIRouter(prefix="/cgm", tags=["cgm"])


@router.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request, cleared: bool = False, db: Session = Depends(get_db)):
    tz = get_tz(db)
    return templates.TemplateResponse("cgm_upload.html", {
        "request":       request,
        "result":        None,
        "cleared":       cleared,
        "configured_tz": str(tz),
        "tz_choices":    TIMEZONE_CHOICES,
    })


@router.post("/upload", response_class=HTMLResponse)
async def upload_file(
    request:   Request,
    file:      UploadFile = File(...),
    source_tz: str        = Form("UTC"),
    db:        Session    = Depends(get_db),
):
    tz       = get_tz(db)
    contents = await file.read()
    result   = {"filename": file.filename, "added": 0, "skipped": 0, "error": None,
                "source_tz": source_tz}

    try:
        rows = parse(contents)
    except ValueError as exc:
        result["error"] = str(exc)
        return templates.TemplateResponse("cgm_upload.html", {
            "request": request, "result": result,
            "cleared": False, "configured_tz": str(tz), "tz_choices": TIMEZONE_CHOICES,
        })

    # Convert naive local timestamps (as exported by the device) to naive UTC for storage
    try:
        src_tz = pytz.timezone(source_tz)
    except pytz.exceptions.UnknownTimeZoneError:
        src_tz = pytz.utc

    first_utc = last_utc = None

    for row in rows:
        try:
            utc_ts = src_tz.localize(row["ts"], is_dst=None).astimezone(pytz.utc).replace(tzinfo=None)
        except Exception:
            # Ambiguous / non-existent wall-clock time (DST boundary) - use is_dst=False
            utc_ts = src_tz.localize(row["ts"], is_dst=False).astimezone(pytz.utc).replace(tzinfo=None)

        if first_utc is None:
            first_utc = utc_ts
        last_utc = utc_ts

        try:
            db.add(CGMReading(ts=utc_ts, glucose_mg_dl=row["glucose_mg_dl"]))
            db.flush()
            result["added"] += 1
        except IntegrityError:
            db.rollback()
            result["skipped"] += 1

    db.commit()

    if first_utc and last_utc:
        # Date range shown in local time for readability
        first_local = src_tz.localize(rows[0]["ts"], is_dst=False).date()
        last_local  = src_tz.localize(rows[-1]["ts"], is_dst=False).date()
        result["date_range"]       = f"{first_local} → {last_local}"
        result["new_spikes"]       = detect_and_insert(db, first_utc, last_utc)
        result["matched_preds"]    = match_predictions(db)
    else:
        result["date_range"] = "-"

    return templates.TemplateResponse("cgm_upload.html", {
        "request": request, "result": result,
        "cleared": False, "configured_tz": str(tz), "tz_choices": TIMEZONE_CHOICES,
    })


@router.post("/clear")
def clear_cgm(db: Session = Depends(get_db)):
    """Delete all CGM readings and spike events. Meals, activities, and all other data untouched."""
    db.query(SpikeEvent).delete()
    db.query(CGMReading).delete()
    db.commit()
    return RedirectResponse("/cgm/upload?cleared=1", status_code=303)
