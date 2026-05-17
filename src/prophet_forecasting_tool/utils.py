"""Shared utilities: duration parsing, identifier validation, period counting."""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from dateutil.relativedelta import relativedelta


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

_DURATION_RE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)\s*$"
)


def validate_identifier(name: str, kind: str = "identifier") -> str:
    """Validate a SQL identifier (table or column name).

    Allows ASCII letters, digits and underscores, must start with a letter or
    underscore, max 63 chars (Postgres limit). Returns the validated name.
    """
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(
            f"Invalid {kind} name '{name}': must match [A-Za-z_][A-Za-z0-9_]*"
            f" and be 1..63 chars."
        )
    return name


def parse_duration(value: str) -> relativedelta:
    """Parse a human duration string into a calendar-aware ``relativedelta``.

    Supports e.g. ``"90D"``, ``"730 days"``, ``"2Y"``, ``"6M"``, ``"4W"``,
    ``"24H"``, ``"15min"``. Months and years are calendar-aware (not 30.44 days).
    """
    if not isinstance(value, str):
        raise ValueError(f"Duration must be a string, got {type(value).__name__}")

    # Allow pandas-style "730 days" / "365 days"
    spaced = value.strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(day|days|hour|hours|minute|minutes|week|weeks|month|months|year|years)$", spaced)
    if m:
        num = float(m.group(1))
        unit = m.group(2)
        if unit.startswith("day"):
            return relativedelta(days=int(num)) if num.is_integer() else relativedelta(hours=int(num * 24))
        if unit.startswith("hour"):
            return relativedelta(hours=int(num))
        if unit.startswith("minute"):
            return relativedelta(minutes=int(num))
        if unit.startswith("week"):
            return relativedelta(weeks=int(num))
        if unit.startswith("month"):
            return relativedelta(months=int(num))
        if unit.startswith("year"):
            return relativedelta(years=int(num))

    m = _DURATION_RE.match(value)
    if not m:
        raise ValueError(
            f"Could not parse duration '{value}'. Examples: '90D', '730 days', '2Y', '6M', '24H', '15min'."
        )
    num_str, unit = m.group("num"), m.group("unit")
    num = float(num_str)
    unit_upper = unit.upper()

    if unit_upper in ("D",):
        return relativedelta(days=int(num)) if num.is_integer() else relativedelta(hours=int(num * 24))
    if unit_upper in ("H",):
        return relativedelta(hours=int(num))
    if unit_upper in ("MIN",):
        return relativedelta(minutes=int(num))
    if unit_upper in ("W",):
        return relativedelta(weeks=int(num))
    if unit_upper in ("M",):
        return relativedelta(months=int(num))
    if unit_upper in ("Y",):
        return relativedelta(years=int(num))

    raise ValueError(
        f"Unrecognized duration unit '{unit}' in '{value}'. Supported: D, H, min, W, M, Y."
    )


_FREQ_MAP = {"M": "MS", "W": "W-MON"}


def duration_to_periods(value: str, freq: str) -> int:
    """Convert a duration string to an integer number of periods at ``freq``.

    Uses calendar-aware arithmetic: a 2Y horizon at freq='M' is 24 months,
    not 730.5/30 = 24.35. Always rounds down to a whole period count >= 1.
    """
    delta = parse_duration(value)
    anchor = pd.Timestamp("2024-01-01")
    target = anchor + delta
    # Map deprecated aliases. Prophet still accepts both 'M' and 'MS' for monthly.
    pandas_freq = _FREQ_MAP.get(freq, freq)
    rng = pd.date_range(start=anchor, end=target, freq=pandas_freq, inclusive="left")
    n = max(len(rng), 1)
    return n


def shift_timestamp(ts: pd.Timestamp, value: str, *, subtract: bool = False) -> pd.Timestamp:
    """Shift a ``pd.Timestamp`` by a parsed duration. Calendar-aware."""
    delta = parse_duration(value)
    py_ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    shifted = py_ts - delta if subtract else py_ts + delta
    return pd.Timestamp(shifted)


def normalize_tz_naive(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    """Drop tz info from a Timestamp, preserving wall-clock value."""
    if ts is None or pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts
