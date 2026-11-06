"""Intervention planning helpers. Estimates (from user-entered rates) are kept apart from actual recorded values."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import db


@dataclass
class Rates:
    """All rates are entered by the user; defaults are 0 so nothing is invented."""
    labour_days_per_ha: float = 0.0
    labour_cost_per_day: float = 0.0
    material_units_per_ha: float = 0.0     # e.g. tonnes of seed cane per ha
    material_cost_per_unit: float = 0.0
    other_cost_per_ha: float = 0.0


def estimate(area_ha: float, rates: Rates) -> dict:
    if area_ha < 0:
        raise ValueError("area_ha must be >= 0")
    labour_days = area_ha * rates.labour_days_per_ha
    material = area_ha * rates.material_units_per_ha
    cost = (labour_days * rates.labour_cost_per_day + material * rates.material_cost_per_unit
            + area_ha * rates.other_cost_per_ha)
    return {"area_ha": area_ha, "labour_days": labour_days, "material_units": material, "est_cost": cost}


def suggest_priority(area_ha: float, confidence: float, high_ha: float = 0.25, medium_ha: float = 0.10,
                     min_conf: float = 0.5) -> str:
    """Inspection-priority heuristic (configurable). It ranks inspection effort, not a diagnosis."""
    if area_ha >= high_ha and confidence >= min_conf:
        return "High"
    if area_ha >= medium_ha:
        return "Medium"
    return "Low"


def worklist_from_gaps(conn, job_pk: int, min_area_ha: float = 0.05, min_confidence: float = 0.0,
                       assigned_to: str | None = None, target_date=None) -> tuple[int, list[str]]:
    """Create 'Pending Verification' interventions for a job's candidate gaps. Skips duplicates. Returns (n, skipped)."""
    job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_pk,)).fetchone()
    farm = conn.execute("SELECT * FROM farms WHERE id=?", (job["farm_pk"],)).fetchone()
    gaps = db.job_gaps(conn, job_pk)
    gaps = gaps[(gaps.area_ha >= min_area_ha) & (gaps.confidence >= min_confidence)
                & (gaps.verification_status != "Rejected")]
    n, skipped = 0, []
    for g in gaps.itertuples():
        action = g.proposed_action if g.proposed_action in db.ACTIONS else "Field verification"
        try:
            db.create_intervention(
                conn, farm_pk=job["farm_pk"], farm_id=farm["farm_id"], field_id=g.field_id, action=action,
                priority=suggest_priority(g.area_ha, g.confidence), gap_id=int(g.gap_id), job_pk=job_pk,
                version_pk=job["version_pk"], assigned_to=assigned_to, target_date=target_date,
                proposed_area_ha=float(g.area_ha), notes="Auto-created from candidate gap; cause unverified.")
            n += 1
        except ValueError as e:
            skipped.append(str(e))
    return n, skipped


def summary(df: pd.DataFrame) -> dict:
    """Keeps proposed, actual-treated and cost figures separate."""
    if df.empty:
        return {"n": 0}
    done = df[df.status == "Completed"]
    return {
        "n": int(len(df)),
        "by_status": df.status.value_counts().to_dict(),
        "proposed_area_ha": float(df.proposed_area_ha.fillna(0).sum()),
        "actual_treated_area_ha": float(done.actual_area_ha.fillna(0).sum()),
        "estimated_cost": float(df.est_cost.fillna(0).sum()),
        "actual_cost": float(done.actual_cost.fillna(0).sum()),
    }
