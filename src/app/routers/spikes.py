from datetime import datetime
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.spike_event import SpikeEvent
from app.services.spike_detector import get_pending
from app.shared_templates import templates

router = APIRouter(prefix="/spikes", tags=["spikes"])


@router.get("/review", response_class=HTMLResponse)
def review(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("spikes/review.html", {
        "request": request,
        "spikes":  get_pending(db),
    })


@router.post("/{spike_id}/acknowledge")
def acknowledge(
    spike_id:    int,
    attribution: str = Form(...),
    db: Session = Depends(get_db),
):
    spike = db.query(SpikeEvent).filter(SpikeEvent.id == spike_id).first()
    if spike:
        spike.attribution    = attribution
        spike.acknowledged_at = datetime.utcnow()
        db.commit()
    return RedirectResponse("/spikes/review", status_code=303)
