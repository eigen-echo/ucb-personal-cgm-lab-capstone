import json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.cgm import CGMReading
from app.models.glucose_prediction import GlucosePrediction
from app.models.meal import Meal
from app.models.medication import Medication
from app.models.model_run import ModelRun
from app.services import predictor
from app.services.timezone import get_cached_tz, local_to_utc, now_local_str, to_local
from app.shared_templates import templates

router = APIRouter(prefix="/predictions", tags=["predictions"])


# ── List ───────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def predictions_list(request: Request, db: Session = Depends(get_db)):
    preds = (
        db.query(GlucosePrediction)
        .order_by(GlucosePrediction.created_at.desc())
        .limit(200)
        .all()
    )

    # Per-model MAE over matched rows
    matched = [p for p in preds if p.actual_glucose_3h is not None]
    stats   = _compute_stats(matched)

    return templates.TemplateResponse("predictions/list.html", {
        "request": request,
        "preds":   preds,
        "stats":   stats,
        "n_matched": len(matched),
    })


# ── New / preview / save ───────────────────────────────────────────────────────

@router.get("/new", response_class=HTMLResponse)
def prediction_form(
    request: Request,
    meal_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    tz = get_cached_tz()

    latest_cgm = (
        db.query(CGMReading)
        .order_by(CGMReading.ts.desc())
        .first()
    )

    # Pre-fill meal: from URL param first, otherwise last logged meal
    prefill_meal = None
    if meal_id:
        prefill_meal = db.query(Meal).filter(Meal.id == meal_id).first()
    if prefill_meal is None:
        prefill_meal = db.query(Meal).order_by(Meal.ts.desc()).first()

    prev_fast_hours, is_fasted = _fasting_context(db, prefill_meal)
    min_since_glip = _minutes_since_glipizide(db)

    return templates.TemplateResponse("predictions/new.html", {
        "request":               request,
        "baseline_glucose":      latest_cgm.glucose_mg_dl if latest_cgm else 120.0,
        "latest_cgm_ts":         to_local(latest_cgm.ts, tz) if latest_cgm else "—",
        "prefill_meal":          prefill_meal,
        "prefill_meal_id":       meal_id,
        "prev_fast_hours":       round(prev_fast_hours, 1),
        "is_fasted":             is_fasted,
        "min_since_glip":        round(min_since_glip, 0),
        "now_local":             now_local_str(tz),
    })


@router.post("/preview", response_class=JSONResponse)
async def prediction_preview(request: Request, db: Session = Depends(get_db)):
    """Compute all model predictions and return JSON. Called via fetch from the form."""
    form = await request.form()
    tz   = get_cached_tz()

    baseline          = float(form.get("baseline_glucose") or 120)
    carbs_g           = float(form.get("carbs_g") or 0)
    min_since_glip    = float(form.get("min_since_glip") or 999)
    prev_fast_hours   = float(form.get("prev_fast_hours") or 0)
    walk_min_after    = float(form.get("walk_min_after") or 0)
    fasted            = form.get("fasted") == "true"
    meal_ts_local_str = form.get("meal_ts") or now_local_str(tz)

    # Convert local → UTC
    try:
        meal_ts_utc = local_to_utc(meal_ts_local_str, tz)
    except Exception:
        meal_ts_utc = datetime.utcnow()

    preds = predictor.predict_all(
        db=db,
        baseline_glucose=baseline,
        carbs_g=carbs_g,
        meal_ts=meal_ts_utc,
        fasted=fasted,
        prev_fast_hours=prev_fast_hours,
        minutes_since_glipizide=min_since_glip,
        walk_min_after=walk_min_after,
    )

    predict_for_ts = meal_ts_utc + timedelta(hours=3)
    return JSONResponse({
        "ok":              True,
        "predictions":     preds,
        "predict_for_ts":  predict_for_ts.isoformat(),
        "predict_for_local": to_local(predict_for_ts, tz),
    })


@router.post("/save")
async def prediction_save(request: Request, db: Session = Depends(get_db)):
    """Persist a prediction row after the user approves the model outputs."""
    form = await request.form()
    tz   = get_cached_tz()

    baseline          = float(form.get("baseline_glucose") or 120)
    carbs_g           = float(form.get("carbs_g") or 0)
    meal_name         = form.get("meal_name") or None
    meal_id           = form.get("meal_id")
    meal_id           = int(meal_id) if meal_id else None
    min_since_glip    = float(form.get("min_since_glip") or 999)
    prev_fast_hours   = float(form.get("prev_fast_hours") or 0)
    walk_min_after    = float(form.get("walk_min_after") or 0)
    fasted            = form.get("fasted") == "on"
    meal_ts_local_str = form.get("meal_ts") or now_local_str(tz)
    predictions_json  = form.get("predictions_json") or "{}"

    try:
        meal_ts_utc = local_to_utc(meal_ts_local_str, tz)
    except Exception:
        meal_ts_utc = datetime.utcnow()

    predict_for_ts = meal_ts_utc + timedelta(hours=3)

    try:
        preds = json.loads(predictions_json)
    except Exception:
        preds = {}

    def _v(model, key):
        m = preds.get(model)
        return m.get(key) if isinstance(m, dict) and key in m else None

    # Stamp the most recent run_tag so predictions are traceable to a model version
    latest_run = (
        db.query(ModelRun)
        .filter(ModelRun.run_tag.isnot(None))
        .order_by(ModelRun.run_ts.desc())
        .first()
    )
    model_run_tag = latest_run.run_tag if latest_run else None

    row = GlucosePrediction(
        created_at         = datetime.utcnow(),
        predict_for_ts     = predict_for_ts,
        meal_id            = meal_id,
        meal_name          = meal_name,
        carbs_g            = carbs_g,
        baseline_glucose   = baseline,
        ridge_delta        = _v("ridge",   "predicted_delta"),       # = peak for Ridge
        rf_delta           = _v("rf",      "predicted_delta"),       # = peak for RF
        lstm3h_glucose     = _v("lstm3h",  "predicted_glucose_3h"),
        lstm3h_delta       = _v("lstm3h",  "predicted_delta"),
        sarimax3h_glucose  = _v("sarimax", "predicted_glucose_3h"),
        sarimax3h_delta    = _v("sarimax", "predicted_delta"),
        sarimax_peak_delta = _v("sarimax", "predicted_peak_delta"),
        model_run_tag      = model_run_tag,
        features_json     = json.dumps({
            "baseline_glucose":  baseline,
            "carbs_g":           carbs_g,
            "meal_ts":           meal_ts_local_str,
            "fasted":            fasted,
            "prev_fast_hours":   prev_fast_hours,
            "min_since_glip":    min_since_glip,
            "walk_min_after":    walk_min_after,
        }),
    )
    db.add(row)
    db.commit()
    return RedirectResponse("/predictions/", status_code=303)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _minutes_since_glipizide(db) -> float:
    """Return minutes since last *taken* Glipizide dose, or 999 if none on record."""
    last = (
        db.query(Medication)
        .filter(
            Medication.drug.ilike("%glipizide%"),
            Medication.taken == True,   # noqa: E712 — SQLAlchemy requires ==
        )
        .order_by(Medication.scheduled_ts.desc())
        .first()
    )
    if last is None:
        return 999.0
    return (datetime.utcnow() - last.scheduled_ts).total_seconds() / 60


def _fasting_context(db, last_meal) -> tuple:
    """Return (hours_since_last_meal, is_fasted)."""
    if last_meal is None:
        return 12.0, True
    hours = (datetime.utcnow() - last_meal.ts).total_seconds() / 3600
    return round(hours, 1), hours > 4


def _compute_stats(matched: list) -> dict:
    """MAE per model over matched predictions."""
    models = {
        "ridge":   ("ridge_delta",       "actual_peak_delta"),
        "rf":      ("rf_delta",          "actual_peak_delta"),
        "lstm3h":  ("lstm3h_delta",      "actual_peak_delta"),
        "sarimax": ("sarimax3h_delta",   "actual_peak_delta"),
    }
    stats = {}
    for name, (pred_col, actual_col) in models.items():
        errors = []
        for p in matched:
            pv = getattr(p, pred_col)
            av = getattr(p, actual_col)
            if pv is not None and av is not None:
                errors.append(abs(pv - av))
        stats[name] = round(sum(errors) / len(errors), 1) if errors else None
    return stats
