"""Manual field attributes: validation and normalisation.

Editable attributes feed the analysis context rules (planting_date, last_harvest_date) and are recorded with every
job, so a result can always be traced to the attribute values that were in force when the job was queued.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

EDITABLE = ["field_name", "crop", "variety", "ratoon_number", "planting_date", "last_harvest_date", "irrigation_system"]
LABELS = {"field_name": "Field name", "crop": "Crop", "variety": "Variety", "ratoon_number": "Ratoon number",
          "planting_date": "Planting / replanting date", "last_harvest_date": "Last harvest date",
          "irrigation_system": "Irrigation system"}
DATE_FIELDS = ("planting_date", "last_harvest_date")
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%d %b %Y", "%d %B %Y")
MIN_DATE = date(1990, 1, 1)


class AttributeError_(ValueError):
    pass


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v).strip() in ("", "None", "nan", "NaT")


def parse_date(v) -> date | None:
    if _blank(v):
        return None
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()[:32]
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s[:10] if fmt.startswith("%Y-") else s, fmt).date()
        except ValueError:
            continue
    raise AttributeError_(f"'{v}' is not a recognised date (use YYYY-MM-DD)")


def normalise_value(key: str, value, today: date | None = None):
    """Return the cleaned value (str or None). Raises AttributeError_ with a user-readable message."""
    if key not in EDITABLE:
        raise AttributeError_(f"'{key}' is not an editable attribute")
    if _blank(value):
        return None
    today = today or date.today()
    if key in DATE_FIELDS:
        d = parse_date(value)
        if d < MIN_DATE:
            raise AttributeError_(f"{LABELS[key]} {d.isoformat()} is before {MIN_DATE.year}")
        if d > today + timedelta(days=366):
            raise AttributeError_(f"{LABELS[key]} {d.isoformat()} is more than a year in the future")
        return d.isoformat()
    if key == "ratoon_number":
        try:
            f = float(str(value).strip())
        except ValueError as e:
            raise AttributeError_(f"Ratoon number '{value}' is not a number") from e
        if f != int(f) or not 0 <= f <= 15:
            raise AttributeError_("Ratoon number must be a whole number from 0 (plant cane) to 15")
        return str(int(f))
    s = str(value).strip()
    if len(s) > 60:
        raise AttributeError_(f"{LABELS[key]} is longer than 60 characters")
    return s


def clean_changes(changes: dict[str, dict], today: date | None = None) -> tuple[dict, list[str], list[str]]:
    """changes: {field_id: {attr: value}} -> (clean changes, errors, warnings). Nothing is applied if errors exist."""
    clean, errors, warnings = {}, [], []
    for fid, attrs in changes.items():
        row = {}
        for k, v in attrs.items():
            try:
                row[k] = normalise_value(k, v, today)
            except AttributeError_ as e:
                errors.append(f"{fid}: {e}")
        clean[fid] = row
        p, h = row.get("planting_date"), row.get("last_harvest_date")
        if p and h and h < p:
            warnings.append(f"{fid}: last harvest date ({h}) is before the planting date ({p}); check the dates.")
    return clean, errors, warnings


def changed_rows(before: pd.DataFrame, after: pd.DataFrame, key: str = "field_id", cols=EDITABLE) -> dict[str, dict]:
    """Rows of `after` whose editable values differ from `before` -> {field_id: {attr: new}} (only changed attrs)."""
    b = before.set_index(key)
    out = {}
    for r in after.itertuples(index=False):
        d = r._asdict()
        fid = d[key]
        if fid not in b.index:
            continue
        diff = {}
        for c in cols:
            if c not in d:
                continue
            old, new = b.at[fid, c], d[c]
            if _blank(old) and _blank(new):
                continue
            if str(old).strip() != str(new).strip():
                diff[c] = new
        if diff:
            out[str(fid)] = diff
    return out


def to_text(v):
    """Display/edit form of an attribute value: None for blanks, '2' for 2.0, ISO text for dates, else str."""
    if _blank(v):
        return None
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v).strip()


def text_frame(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Copy of df[cols] with every value converted by to_text (object dtype, None for blanks)."""
    out = df[list(cols)].copy().astype(object)
    for c in out.columns:
        out[c] = pd.Series([to_text(v) for v in out[c]], dtype=object, index=out.index)
    return out.reset_index(drop=True)
