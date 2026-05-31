"""
Medication log — Glipizide twice-daily tracker.

Doses are stored in the `medications` table:
  drug="Glipizide", taken=True/False, scheduled_ts=UTC.

Week view shows Mon–Sun; inline forms for edit / add / skip.
"""
from datetime import date, datetime, timedelta
from typing import Optional

import pytz
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.medication import Medication
from app.services.timezone import get_tz
from app.shared_templates import templates

router = APIRouter(prefix="/medications", tags=["medications"])

DRUG_NAME  = "Glipizide"
AM_HOUR    = 10    # default morning dose hour (local time)
PM_HOUR    = 16    # default evening dose hour (local time)
DEFAULT_MG = 5.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _week_start(d: date) -> date:
    """Return the Monday of the week containing d."""
    return d - timedelta(days=d.weekday())


def _to_utc(d: date, hour: int, minute: int, tz) -> datetime:
    """Localise a date+time to user's tz and return naive UTC for storage."""
    try:
        local_dt = tz.localize(datetime(d.year, d.month, d.day, hour, minute), is_dst=None)
    except Exception:
        local_dt = tz.localize(datetime(d.year, d.month, d.day, hour, minute), is_dst=False)
    return local_dt.astimezone(pytz.utc).replace(tzinfo=None)


def _parse_input(ts_str: str, tz) -> datetime:
    """Parse a datetime-local input value (YYYY-MM-DDTHH:MM) → naive UTC."""
    naive = datetime.fromisoformat(ts_str)
    return _to_utc(naive.date(), naive.hour, naive.minute, tz)


def _fmt_display(ldt: datetime) -> str:
    h = ldt.hour
    m = ldt.minute
    h12 = h % 12 or 12
    ampm = "AM" if h < 12 else "PM"
    return f"{h12}:{m:02d} {ampm}"


def _entry_dict(med: Medication, ldt: datetime) -> dict:
    return {
        "id":        med.id,
        "taken":     med.taken,
        "dose_mg":   med.dose_mg,
        "notes":     med.notes or "",
        "display":   _fmt_display(ldt),
        "local_str": ldt.strftime("%Y-%m-%dT%H:%M"),
    }


def _parse_time_str(t: str) -> tuple[int, int]:
    """Parse 'HH:MM' string → (hour, minute). Returns (0, 0) on error."""
    try:
        h, m = t.strip().split(":")
        return int(h), int(m)
    except Exception:
        return (0, 0)


def _build_week(db: Session, tz, ws: date) -> list[dict]:
    """Return one dict per day with dose1/dose2 slots (chronological order).

    Dose 1 = earlier entry, Dose 2 = later entry — no AM/PM boundary,
    so both-after-noon and both-before-noon are handled correctly.
    """
    we = ws + timedelta(days=7)
    # Generous UTC window to cover any timezone offset (±14 h)
    window_start = datetime.combine(ws, datetime.min.time()) - timedelta(hours=14)
    window_end   = datetime.combine(we, datetime.min.time()) + timedelta(hours=14)

    meds = (
        db.query(Medication)
        .filter(
            Medication.drug.ilike(f"%{DRUG_NAME}%"),
            Medication.scheduled_ts >= window_start,
            Medication.scheduled_ts <  window_end,
        )
        .order_by(Medication.scheduled_ts.asc())
        .all()
    )

    by_date: dict[date, list] = {}
    for med in meds:
        ldt = pytz.utc.localize(med.scheduled_ts).astimezone(tz)
        by_date.setdefault(ldt.date(), []).append((med, ldt))

    days = []
    for i in range(7):
        d       = ws + timedelta(days=i)
        entries = sorted(by_date.get(d, []), key=lambda x: x[1])  # chronological
        dose1_entry = entries[0] if len(entries) > 0 else None
        dose2_entry = entries[1] if len(entries) > 1 else None
        days.append({
            "date":     d,
            "date_str": d.strftime("%a, %b %-d") if hasattr(d, "strftime") else str(d),
            "dose1":    _entry_dict(*dose1_entry) if dose1_entry else None,
            "dose2":    _entry_dict(*dose2_entry) if dose2_entry else None,
        })

    return days


def _week_stats(days: list[dict]) -> dict:
    taken   = sum(1 for d in days for slot in (d["dose1"], d["dose2"]) if slot and slot["taken"])
    skipped = sum(1 for d in days for slot in (d["dose1"], d["dose2"]) if slot and not slot["taken"])
    missing = sum(1 for d in days for slot in (d["dose1"], d["dose2"]) if slot is None)
    return {"taken": taken, "skipped": skipped, "missing": missing, "total": 14}


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def medications_index(
    request: Request,
    week: Optional[str] = None,
    msg: Optional[str] = None,
    db: Session = Depends(get_db),
):
    tz    = get_tz(db)
    today = datetime.now(tz).date()

    try:
        ws = date.fromisoformat(week) if week else _week_start(today)
    except ValueError:
        ws = _week_start(today)

    days     = _build_week(db, tz, ws)
    stats    = _week_stats(days)
    week_end = ws + timedelta(days=6)

    return templates.TemplateResponse("medications/index.html", {
        "request":            request,
        "days":               days,
        "stats":              stats,
        "week_start":         ws,
        "week_end":           week_end,
        "prev_week":          (ws - timedelta(days=7)).isoformat(),
        "next_week":          (ws + timedelta(days=7)).isoformat(),
        "today":              today,
        "default_dose1_time": f"{AM_HOUR:02d}:00",   # "10:00" — browser override via sessionStorage
        "default_dose2_time": f"{PM_HOUR:02d}:00",   # "16:00"
        "msg":                msg,
    })


@router.post("/add")
async def medication_add(request: Request, db: Session = Depends(get_db)):
    form  = await request.form()
    tz    = get_tz(db)
    week  = form.get("week", _week_start(datetime.now(tz).date()).isoformat())
    ts_str = form.get("scheduled_ts", "")
    taken  = form.get("taken", "true") == "true"

    try:
        utc_ts = _parse_input(ts_str, tz)
    except Exception:
        return RedirectResponse(f"/medications/?week={week}", status_code=303)

    dose_mg = DEFAULT_MG
    try:
        dose_mg = float(form.get("dose_mg") or DEFAULT_MG)
    except ValueError:
        pass

    db.add(Medication(
        scheduled_ts = utc_ts,
        drug         = DRUG_NAME,
        dose_mg      = dose_mg,
        taken        = taken,
        notes        = form.get("notes") or None,
    ))
    db.commit()
    return RedirectResponse(f"/medications/?week={week}", status_code=303)


@router.post("/{med_id}/update")
async def medication_update(
    request: Request,
    med_id: int,
    db: Session = Depends(get_db),
):
    form = await request.form()
    tz   = get_tz(db)
    week = form.get("week", _week_start(datetime.now(tz).date()).isoformat())

    med = db.query(Medication).filter(Medication.id == med_id).first()
    if not med:
        return RedirectResponse(f"/medications/?week={week}", status_code=303)

    ts_str = form.get("scheduled_ts", "")
    if ts_str:
        try:
            med.scheduled_ts = _parse_input(ts_str, tz)
        except Exception:
            pass

    taken_val = form.get("taken")
    if taken_val is not None:
        med.taken = taken_val == "true"

    notes_val = form.get("notes")
    med.notes = (notes_val or None)

    dose_val = form.get("dose_mg")
    if dose_val:
        try:
            med.dose_mg = float(dose_val)
        except ValueError:
            pass

    db.commit()
    return RedirectResponse(f"/medications/?week={week}", status_code=303)


@router.post("/{med_id}/delete")
async def medication_delete(
    request: Request,
    med_id: int,
    db: Session = Depends(get_db),
):
    form = await request.form()
    tz   = get_tz(db)
    week = form.get("week", _week_start(datetime.now(tz).date()).isoformat())

    med = db.query(Medication).filter(Medication.id == med_id).first()
    if med:
        db.delete(med)
        db.commit()
    return RedirectResponse(f"/medications/?week={week}", status_code=303)


@router.post("/generate-defaults")
async def generate_defaults(request: Request, db: Session = Depends(get_db)):
    """Create Dose 1 + Dose 2 entries for a date, skipping slots that already have a record.

    dose1_time / dose2_time are HH:MM strings (browser time override, default 10:00 / 16:00).
    A ±4h proximity check prevents duplicates even if times differ slightly from existing records.
    """
    form       = await request.form()
    tz         = get_tz(db)
    week       = form.get("week", _week_start(datetime.now(tz).date()).isoformat())
    date_str   = form.get("date", "")
    dose1_time = form.get("dose1_time") or f"{AM_HOUR:02d}:00"
    dose2_time = form.get("dose2_time") or f"{PM_HOUR:02d}:00"

    try:
        d = date.fromisoformat(date_str)
    except Exception:
        return RedirectResponse(f"/medications/?week={week}", status_code=303)

    for time_str in [dose1_time, dose2_time]:
        hour, minute = _parse_time_str(time_str)
        utc_ts = _to_utc(d, hour, minute, tz)
        # Check for any existing entry within ±4 h of the target slot
        existing = (
            db.query(Medication)
            .filter(
                Medication.drug.ilike(f"%{DRUG_NAME}%"),
                Medication.scheduled_ts >= utc_ts - timedelta(hours=4),
                Medication.scheduled_ts <= utc_ts + timedelta(hours=4),
            )
            .first()
        )
        if not existing:
            db.add(Medication(
                scheduled_ts = utc_ts,
                drug         = DRUG_NAME,
                dose_mg      = DEFAULT_MG,
                taken        = True,
            ))

    db.commit()
    return RedirectResponse(f"/medications/?week={week}", status_code=303)


@router.post("/skip-day")
async def skip_day(request: Request, db: Session = Depends(get_db)):
    """Mark both Dose 1 + Dose 2 slots for a date as taken=False (creates entries if absent).

    dose1_time / dose2_time are HH:MM strings forwarded from the browser override inputs.
    """
    form       = await request.form()
    tz         = get_tz(db)
    week       = form.get("week", _week_start(datetime.now(tz).date()).isoformat())
    date_str   = form.get("date", "")
    dose1_time = form.get("dose1_time") or f"{AM_HOUR:02d}:00"
    dose2_time = form.get("dose2_time") or f"{PM_HOUR:02d}:00"

    try:
        d = date.fromisoformat(date_str)
    except Exception:
        return RedirectResponse(f"/medications/?week={week}", status_code=303)

    for time_str in [dose1_time, dose2_time]:
        hour, minute = _parse_time_str(time_str)
        utc_ts = _to_utc(d, hour, minute, tz)
        existing = (
            db.query(Medication)
            .filter(
                Medication.drug.ilike(f"%{DRUG_NAME}%"),
                Medication.scheduled_ts >= utc_ts - timedelta(hours=4),
                Medication.scheduled_ts <= utc_ts + timedelta(hours=4),
            )
            .first()
        )
        if existing:
            existing.taken = False
        else:
            db.add(Medication(
                scheduled_ts = utc_ts,
                drug         = DRUG_NAME,
                dose_mg      = DEFAULT_MG,
                taken        = False,
            ))

    db.commit()
    return RedirectResponse(f"/medications/?week={week}", status_code=303)
