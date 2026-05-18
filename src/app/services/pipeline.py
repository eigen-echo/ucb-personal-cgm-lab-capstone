"""Orchestrate the full training pipeline: export → feature build → train → record."""
import os
import subprocess
import sys
import threading
from datetime import datetime
from sqlalchemy.orm import Session

from app.services.export import export_all

SCRIPTS_DIR = os.environ.get("SCRIPTS_DIR", "/app/scripts")
# pipeline.py lives at /app/app/services/pipeline.py → 3 dirnames → /app
SRC_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Shared in-memory status (single-user localhost — no need for Redis/DB)
status = {
    "running":    False,
    "started_at": None,
    "finished_at": None,
    "log":        [],
    "error":      None,
}


def _log(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    status["log"].append(f"[{ts}] {msg}")


def run(db: Session, models: list[str]):
    """Called as a background thread from the training router."""
    status["running"]    = True
    status["started_at"] = datetime.utcnow().isoformat()
    status["finished_at"] = None
    status["log"]        = []
    status["error"]      = None

    try:
        _log("Exporting SQLite tables to data/raw/ CSVs...")
        counts = export_all(db)
        for fname, n in counts.items():
            _log(f"  {fname}: {n} rows")

        _log("Running 00-build-features.py...")
        _run_script(os.path.join(SCRIPTS_DIR, "00-build-features.py"))

        _log("Running 01-build-hourly.py...")
        _run_script(os.path.join(SCRIPTS_DIR, "01-build-hourly.py"))

        train_script = os.path.join(SRC_DIR, "train_models.py")
        _log(f"Running train_models.py --models {' '.join(models)} ...")
        _run_script(train_script, extra_args=["--models"] + models)

        _log("Pipeline complete.")
    except subprocess.CalledProcessError as exc:
        status["error"] = f"Subprocess failed (exit {exc.returncode}): {exc.cmd}"
        _log(f"ERROR: {status['error']}")
    except Exception as exc:
        status["error"] = str(exc)
        _log(f"ERROR: {exc}")
    finally:
        status["running"]     = False
        status["finished_at"] = datetime.utcnow().isoformat()


def _run_script(path: str, extra_args: list[str] | None = None):
    cmd = [sys.executable, path] + (extra_args or [])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        _log(f"  {line}")


def run_in_background(db: Session, models: list[str]):
    """Launch the pipeline in a daemon thread so the HTTP response returns immediately."""
    t = threading.Thread(target=run, args=(db, models), daemon=True)
    t.start()
