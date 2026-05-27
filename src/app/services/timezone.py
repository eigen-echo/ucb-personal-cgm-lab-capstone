"""Timezone helpers: display conversion (UTC → local) and input parsing (local → UTC)."""
import time
from datetime import datetime

import pytz
from sqlalchemy.orm import Session

# Simple 5-second TTL cache so every template render doesn't hit the DB
_cache: dict = {"name": "UTC", "ts": 0.0}


def get_cached_tz() -> pytz.BaseTzInfo:
    now = time.monotonic()
    if now - _cache["ts"] > 5:
        from app.database import SessionLocal
        from app.models.app_config import AppConfig
        db = SessionLocal()
        try:
            row = db.query(AppConfig).filter(AppConfig.key == "user_timezone").first()
            _cache["name"] = row.value if row else "UTC"
        finally:
            db.close()
        _cache["ts"] = now
    return pytz.timezone(_cache["name"])


def get_tz(db: Session) -> pytz.BaseTzInfo:
    from app.models.app_config import AppConfig
    row = db.query(AppConfig).filter(AppConfig.key == "user_timezone").first()
    return pytz.timezone(row.value if row else "UTC")


def to_local(dt: datetime, tz: pytz.BaseTzInfo, fmt: str = "%b %d %H:%M") -> str:
    """Convert a naive UTC datetime (or ISO string) to a formatted local string."""
    if dt is None:
        return "-"
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.utc)
    return dt.astimezone(tz).strftime(fmt)


def local_to_utc(ts_str: str, tz: pytz.BaseTzInfo) -> datetime:
    """Parse a naive local datetime string and return a naive UTC datetime for storage."""
    naive = datetime.fromisoformat(ts_str)
    local_dt = tz.localize(naive, is_dst=None)
    return local_dt.astimezone(pytz.utc).replace(tzinfo=None)


def now_local_str(tz: pytz.BaseTzInfo, fmt: str = "%Y-%m-%dT%H:%M") -> str:
    """Current time formatted in the user's timezone (for datetime-local inputs)."""
    return datetime.now(tz).strftime(fmt)


# Common IANA timezone names for the settings dropdown
TIMEZONE_CHOICES = [
    "UTC",
    # Americas
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Anchorage",
    "America/Honolulu",
    "America/Toronto",
    "America/Vancouver",
    "America/Mexico_City",
    "America/Sao_Paulo",
    "America/Argentina/Buenos_Aires",
    # Europe
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Europe/Madrid",
    "Europe/Rome",
    "Europe/Amsterdam",
    "Europe/Stockholm",
    "Europe/Moscow",
    "Europe/Istanbul",
    # Africa
    "Africa/Cairo",
    "Africa/Johannesburg",
    # Middle East & South Asia
    "Asia/Dubai",
    "Asia/Karachi",
    "Asia/Kolkata",
    "Asia/Colombo",
    "Asia/Kathmandu",
    "Asia/Dhaka",
    # Southeast Asia
    "Asia/Yangon",
    "Asia/Bangkok",
    "Asia/Singapore",
    "Asia/Jakarta",
    # East Asia
    "Asia/Shanghai",
    "Asia/Taipei",
    "Asia/Seoul",
    "Asia/Tokyo",
    # Oceania
    "Australia/Perth",
    "Australia/Adelaide",
    "Australia/Sydney",
    "Pacific/Auckland",
]
