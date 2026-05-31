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
from app.services.timezone import get_cached_tz, to_local
from app.shared_templates import templates

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/", response_class=HTMLResponse)
def training_page(
    request: Request,
    db: Session = Depends(get_db),
    err: str = "",
):
    runs = (
        db.query(ModelRun)
        .order_by(ModelRun.run_ts.desc())
        .limit(100)
        .all()
    )

    tz = get_cached_tz()

    # Chart series: each model gets its own (x, y) pairs so the time axis is
    # accurate even when models were retrained on different dates.
    # x = local ISO timestamp truncated to the minute  (no seconds, no µs)
    # y = test RMSE
    _chart_models = [
        ("ridge_per_meal",       "Ridge",      "#1565c0"),
        ("rf_per_meal",          "RF",         "#2e7d32"),
        ("sarimax_hourly",       "SARIMAX",    "#e65100"),
        ("lstm_per_meal",        "LSTM",       "#6a1b9a"),
        ("lstm_5min_forecaster", "LSTM 5-min", "#0277bd"),
        ("lstm_3h",              "LSTM 3h",    "#880e4f"),
        ("lstm_trajectory",      "LSTM Traj",    "#00838f"),
        ("lstm_trajectory_v2",   "LSTM Traj V2", "#2e7d32"),
    ]
    chart_series = []
    for model_key, label, color in _chart_models:
        pts = sorted(
            [r for r in runs if r.model_name == model_key and r.rmse_test is not None],
            key=lambda r: r.run_ts,
        )
        if pts:
            chart_series.append({
                "label": label,
                "color": color,
                # Format as "YYYY-MM-DDTHH:MM" — Chart.js time scale parses this;
                # displayFormats in the template controls how it's shown on the axis.
                "data": [
                    {
                        "x": to_local(r.run_ts, tz, fmt="%Y-%m-%dT%H:%M"),
                        "y": round(r.rmse_test, 2),
                    }
                    for r in pts
                ],
            })

    return templates.TemplateResponse("training.html", {
        "request":      request,
        "runs":         runs,
        "status":       pipeline.status,
        "now_utc":      datetime.now(timezone.utc).strftime("%Y%m%d-%H%M"),
        "chart_series": chart_series,
        "err":          err,
    })


@router.post("/run")
async def training_run(
    request: Request,
    db: Session = Depends(get_db),
):
    form      = await request.form()
    models    = form.getlist("models") or ["ridge"]
    run_label = (form.get("run_label") or "").strip()

    # Sanitise label: keep only alphanumeric, dash, underscore; max 30 chars
    import re
    run_label = re.sub(r"[^A-Za-z0-9_\-]", "", run_label)[:30]
    run_tag   = run_label or None   # None → pipeline auto-generates vYYYYMMDD-HHMM

    # Date range filter — validate ISO format and minimum 14-day span
    _iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    date_from = (form.get("date_from") or "").strip()
    date_to   = (form.get("date_to")   or "").strip()
    date_from = date_from if _iso.match(date_from) else ""
    date_to   = date_to   if _iso.match(date_to)   else ""

    if date_from and date_to:
        from datetime import date as _date
        d1 = _date.fromisoformat(date_from)
        d2 = _date.fromisoformat(date_to)
        if (d2 - d1).days < 14:
            return RedirectResponse(
                "/training/?err=range_too_short", status_code=303
            )
        if d2 < d1:
            date_from, date_to = date_to, date_from  # swap if reversed

    if pipeline.status["running"]:
        return RedirectResponse("/training/?error=already_running", status_code=303)

    pipeline.run_in_background(
        db, models,
        run_tag   = run_tag,
        date_from = date_from or None,
        date_to   = date_to   or None,
    )
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
