"""Single Jinja2Templates instance shared by all routers.

A `pending_spikes` global callable is injected so the nav badge works on
every page without each router having to query and pass the count.
"""
import os
from fastapi.templating import Jinja2Templates
from app.database import SessionLocal

_TMPL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=_TMPL_DIR)


def _pending_spikes_count() -> int:
    from app.services.spike_detector import count_pending  # lazy — avoids circular at import
    db = SessionLocal()
    try:
        return count_pending(db)
    finally:
        db.close()


templates.env.globals["pending_spikes"] = _pending_spikes_count
