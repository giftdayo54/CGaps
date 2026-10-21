"""Roll-ups computed from independently analysed field results (area-weighted, no averaging of %)."""
from __future__ import annotations

import pandas as pd

from .areas import gap_percentage


def farm_summary(results: pd.DataFrame, overlap_policy: str = "first_wins", n_overlaps: int = 0) -> dict:
    ok = results[results["status"] == "analysed"]
    area = float(ok["analysed_area_ha"].sum())
    gap = float(ok["gap_area_ha"].sum())
    return {
        "n_fields": int(len(results)),
        "n_analysed": int(len(ok)),
        "n_insufficient_evidence": int((results["status"] == "insufficient_evidence").sum()),
        "n_context_excluded": int((results["status"] == "context_excluded").sum()),
        "n_unable": int((results["status"] == "unable").sum()),
        "analysed_area_ha": round(area, 3),
        "candidate_gap_area_ha": round(gap, 3),
        "gap_pct": round(gap_percentage(gap, area), 2),
        "n_gap_polygons": int(ok["n_gaps"].sum()),
        "double_count_risk": bool(overlap_policy == "flag_only" and n_overlaps > 0),
    }


def estate_summary(farm_rows: pd.DataFrame) -> pd.DataFrame:
    """farm_rows: columns estate, farm_id, analysed_area_ha, candidate_gap_area_ha."""
    g = farm_rows.groupby("estate", dropna=False)[["analysed_area_ha", "candidate_gap_area_ha"]].sum()
    g["gap_pct"] = [gap_percentage(a, b) for a, b in zip(g["candidate_gap_area_ha"], g["analysed_area_ha"])]
    return g.reset_index()


def rank_fields(results: pd.DataFrame, by: str = "gap_area_ha") -> pd.DataFrame:
    ok = results[results["status"] == "analysed"]
    return ok.sort_values(by, ascending=False)
