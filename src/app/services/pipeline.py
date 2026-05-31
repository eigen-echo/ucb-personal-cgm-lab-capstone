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

# Shared in-memory status (single-user localhost - no need for Redis/DB)
status = {
    "running":    False,
    "started_at": None,
    "finished_at": None,
    "log":        [],
    "error":      None,
    "run_tag":    None,   # set at the start of each pipeline run
    "date_from":  None,   # ISO date string "YYYY-MM-DD" or None
    "date_to":    None,
}


def _log(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    status["log"].append(f"[{ts}] {msg}")


def run(
    db: Session,
    models: list[str],
    run_tag: str | None = None,
    date_from: str | None = None,
    date_to:   str | None = None,
):
    """Called as a background thread from the training router."""
    # Auto-generate tag if the user didn't supply one
    if not run_tag:
        run_tag = "v" + datetime.utcnow().strftime("%Y%m%d-%H%M")

    status["running"]     = True
    status["started_at"]  = datetime.utcnow().isoformat()
    status["finished_at"] = None
    status["log"]         = []
    status["error"]       = None
    status["run_tag"]     = run_tag
    status["date_from"]   = date_from
    status["date_to"]     = date_to

    try:
        _log(f"Run tag: {run_tag}")
        if date_from or date_to:
            _log(f"Data filter: {date_from or 'beginning'} → {date_to or 'latest'}")
        _log("Exporting SQLite tables to data/raw/ CSVs...")
        counts = export_all(db)
        for fname, n in counts.items():
            _log(f"  {fname}: {n} rows")

        _log("Running 00-build-features.py...")
        _run_script(os.path.join(SCRIPTS_DIR, "00-build-features.py"))

        _log("Running 01-build-hourly.py...")
        _run_script(os.path.join(SCRIPTS_DIR, "01-build-hourly.py"))

        _log("Running 02-build-5min.py...")
        _run_script(os.path.join(SCRIPTS_DIR, "02-build-5min.py"))

        train_args = ["--models"] + models + ["--run-tag", run_tag]
        if date_from:
            train_args += ["--date-from", date_from]
        if date_to:
            train_args += ["--date-to", date_to]

        train_script = os.path.join(SRC_DIR, "train_models.py")
        _log(f"Running train_models.py --models {' '.join(models)} --run-tag {run_tag}"
             + (f" --date-from {date_from}" if date_from else "")
             + (f" --date-to {date_to}"     if date_to   else "")
             + " ...")
        _run_script(train_script, extra_args=train_args)

        _log("Pipeline complete.")
    except subprocess.CalledProcessError as exc:
        # stdout+stderr were already streamed line-by-line into status["log"].
        # Pull the last non-empty log line as a quick summary for the banner.
        last_err = next(
            (l for l in reversed(status["log"]) if l.strip()),
            ""
        )
        status["error"] = (
            f"Subprocess failed (exit {exc.returncode}) — see log above."
            + (f"  Last line: {last_err}" if last_err else "")
        )
        _log(f"ERROR: Subprocess exit {exc.returncode}")
    except Exception as exc:
        status["error"] = str(exc)
        _log(f"ERROR: {exc}")
    finally:
        status["running"]     = False
        status["finished_at"] = datetime.utcnow().isoformat()


def _run_script(path: str, extra_args: list[str] | None = None):
    # -u forces Python's own stdout/stderr to be unbuffered so every print()
    # call appears in the log immediately instead of being held until the
    # process exits.  stderr=STDOUT merges both streams into one line iterator.
    cmd = [sys.executable, "-u", path] + (extra_args or [])
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr so nothing is lost
        text=True,
        bufsize=1,                  # line-buffered on our end
    )

    all_lines: list[str] = []
    for raw in proc.stdout:
        line = raw.rstrip("\n\r")
        _log(f"  {line}")
        all_lines.append(line)

    proc.wait()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, "\n".join(all_lines), ""
        )


def run_in_background(
    db: Session,
    models: list[str],
    run_tag:   str | None = None,
    date_from: str | None = None,
    date_to:   str | None = None,
):
    """Launch the pipeline in a daemon thread so the HTTP response returns immediately."""
    t = threading.Thread(
        target=run, args=(db, models, run_tag, date_from, date_to), daemon=True
    )
    t.start()
