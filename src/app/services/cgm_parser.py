"""Parse Stelo (and generic CGM) CSV uploads into (ts, glucose_mg_dl) rows."""
import io
from datetime import datetime
import pandas as pd

# Known column-name patterns across Stelo export versions
_TS_CANDIDATES = [
    "Timestamp (YYYY-MM-DDThh:mm:ss)",
    "timestamp",
    "Timestamp",
    "ts",
    "Time",
    "DateTime",
]
_GLUCOSE_CANDIDATES = [
    "Glucose Value (mg/dL)",
    "glucose_mg_dl",
    "Glucose",
    "glucose",
    "CGM Glucose Value (mmol/L)",   # handled separately — unit conversion
    "Value",
]


def parse(contents: bytes) -> list[dict]:
    """
    Parse raw CSV bytes from a Stelo export.

    Returns a list of dicts: [{"ts": datetime, "glucose_mg_dl": float}, ...]
    Rows with non-numeric glucose or unparseable timestamps are silently dropped.
    """
    df = pd.read_csv(io.BytesIO(contents))

    ts_col = _find_col(df, _TS_CANDIDATES)
    gl_col = _find_col(df, _GLUCOSE_CANDIDATES)

    if ts_col is None or gl_col is None:
        raise ValueError(
            f"Could not identify timestamp or glucose column. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[[ts_col, gl_col]].copy()
    df.columns = ["ts_raw", "glucose_raw"]

    df["ts"] = pd.to_datetime(df["ts_raw"], errors="coerce")
    df["glucose_mg_dl"] = pd.to_numeric(df["glucose_raw"], errors="coerce")

    # Unit conversion: mmol/L → mg/dL (multiply by 18.0182)
    if "mmol" in gl_col.lower():
        df["glucose_mg_dl"] = df["glucose_mg_dl"] * 18.0182

    df = df.dropna(subset=["ts", "glucose_mg_dl"])
    df = df.sort_values("ts")

    return [
        {"ts": row.ts.to_pydatetime(), "glucose_mg_dl": float(row.glucose_mg_dl)}
        for row in df.itertuples()
    ]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    # Case-insensitive fallback
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None
