"""Estate -> farm -> field -> gap roll-ups and trends across assessment dates.

Everything is computed from stored field results (area-weighted); percentages are never averaged.
Each farm contributes the job you pick (default: its latest completed job); trends flag comparisons that may be
unreliable (different boundary version, method, source, parameters, observation coverage, field set, synthetic data).
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from . import db
from .areas import gap_percentage

NO_ESTATE = "(no estate)"


def _estate_label(e):
    return NO_ESTATE if (e is None or pd.isna(e) or str(e).strip() == "") else e


def latest_jobs(conn, farm_pks: list[int] | None = None, include_synthetic: bool = False) -> pd.DataFrame:
    """Latest completed job per farm (by imagery end date, then id)."""
    q = ("SELECT j.id job_pk, j.farm_pk, j.version_pk, j.date_to, j.source, j.synthetic, j.algo_version, j.status "
         "FROM jobs j WHERE j.status='completed'")
    df = pd.read_sql_query(q, conn)
    if farm_pks is not None:
        df = df[df.farm_pk.isin(farm_pks)]
    if not include_synthetic:
        df = df[df.synthetic == 0]
    return df.sort_values(["date_to", "job_pk"]).groupby("farm_pk").tail(1).reset_index(drop=True)


def _farm_rows(conn, job_pks: list[int]) -> pd.DataFrame:
    if not job_pks:
        return pd.DataFrame()
    marks = ",".join("?" * len(job_pks))
    q = (f"SELECT fr.*, j.farm_pk, j.date_to, j.version_pk, j.synthetic, f.farm_id, f.farm_name, f.estate "
         f"FROM field_results fr JOIN jobs j ON j.id=fr.job_pk JOIN farms f ON f.id=j.farm_pk WHERE fr.job_pk IN ({marks})")
    df = pd.read_sql_query(q, conn, params=job_pks)
    df["estate"] = df.estate.map(_estate_label)
    return df


def farm_table(conn, job_pks: list[int]) -> pd.DataFrame:
    df = _farm_rows(conn, job_pks)
    if df.empty:
        return df
    rows = []
    for (est, fpk), g in df.groupby(["estate", "farm_pk"]):
        ok = g[g.status == "analysed"]
        area, gap = float(ok.analysed_area_ha.sum()), float(ok.gap_area_ha.sum())
        rows.append({"estate": est, "farm_pk": int(fpk), "farm_id": g.farm_id.iloc[0], "farm_name": g.farm_name.iloc[0],
                     "job_pk": int(g.job_pk.iloc[0]), "assessment_date": g.date_to.iloc[0], "n_fields": len(g),
                     "n_analysed": len(ok), "n_not_assessed": len(g) - len(ok), "analysed_area_ha": round(area, 3),
                     "candidate_gap_ha": round(gap, 3), "gap_pct": round(gap_percentage(gap, area), 2),
                     "synthetic": bool(g.synthetic.iloc[0])})
    return pd.DataFrame(rows).sort_values(["estate", "candidate_gap_ha"], ascending=[True, False]).reset_index(drop=True)


def estate_table(farms: pd.DataFrame) -> pd.DataFrame:
    """Estate totals = sums over farms; % = total gap / total analysed area."""
    if farms.empty:
        return farms
    g = farms.groupby("estate").agg(n_farms=("farm_pk", "nunique"), n_fields=("n_fields", "sum"),
                                    n_analysed=("n_analysed", "sum"), analysed_area_ha=("analysed_area_ha", "sum"),
                                    candidate_gap_ha=("candidate_gap_ha", "sum")).reset_index()
    g["gap_pct"] = [round(gap_percentage(a, b), 2) for a, b in zip(g.candidate_gap_ha, g.analysed_area_ha)]
    g["mixed_assessment_dates"] = [farms[farms.estate == e].assessment_date.nunique() > 1 for e in g.estate]
    g["any_synthetic"] = [bool(farms[farms.estate == e].synthetic.any()) for e in g.estate]
    return g.sort_values("candidate_gap_ha", ascending=False).reset_index(drop=True)


def field_table(conn, job_pk: int) -> pd.DataFrame:
    df = db.job_results(conn, job_pk)
    return df[["field_id", "status", "field_area_ha", "analysed_area_ha", "gap_area_ha", "gap_pct", "n_gaps", "valid_fraction"]]


def _params_hash(params_json: str) -> str:
    p = json.loads(params_json or "{}")
    for k in ("n_overlaps", "overlaps", "field_attributes"):
        p.pop(k, None)
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:8]


def trend(conn, farm_pk: int, include_synthetic: bool = True) -> pd.DataFrame:
    """One row per completed job for a farm, ordered by imagery date, with comparability flags vs the previous row."""
    jobs = db.list_jobs(conn, farm_pk)
    jobs = jobs[jobs.status == "completed"].sort_values(["date_to", "id"])
    if not include_synthetic:
        jobs = jobs[jobs.synthetic == 0]
    rows, prev = [], None
    for j in jobs.itertuples():
        res = db.job_results(conn, int(j.id))
        ok = res[res.status == "analysed"]
        area, gap = float(ok.analysed_area_ha.sum()), float(ok.gap_area_ha.sum())
        row = {"job_pk": int(j.id), "assessment_date": j.date_to, "method": j.algo_version, "source": j.source,
               "boundary_version_pk": int(j.version_pk), "n_fields_analysed": len(ok), "analysed_area_ha": round(area, 2),
               "candidate_gap_ha": round(gap, 3), "gap_pct": round(gap_percentage(gap, area), 2),
               "mean_valid_fraction": round(float(ok.valid_fraction.mean()), 2) if len(ok) else None,
               "synthetic": bool(j.synthetic), "_params": _params_hash(j.params_json), "_fields": frozenset(ok.field_id),
               "_attrs": json.dumps(json.loads(j.params_json or "{}").get("field_attributes", {}), sort_keys=True)}
        flags = []
        if prev is not None:
            if row["boundary_version_pk"] != prev["boundary_version_pk"]:
                flags.append("boundary version changed")
            if row["method"] != prev["method"]:
                flags.append("method changed")
            if row["source"] != prev["source"]:
                flags.append("data source changed")
            if row["_params"] != prev["_params"]:
                flags.append("processing parameters changed")
            if row["_attrs"] != prev["_attrs"]:
                flags.append("field attributes (variety/dates) changed")
            if row["_fields"] != prev["_fields"]:
                flags.append("set of analysed fields changed")
            if (row["mean_valid_fraction"] is not None and prev["mean_valid_fraction"] is not None
                    and abs(row["mean_valid_fraction"] - prev["mean_valid_fraction"]) > 0.15):
                flags.append("observation coverage differs")
            if row["synthetic"] != prev["synthetic"]:
                flags.append("synthetic vs real data")
        if row["synthetic"]:
            flags.append("synthetic data")
        row["comparability_flags"] = "; ".join(flags)
        row["comparable_with_previous"] = prev is not None and not [f for f in flags if f != "synthetic data"]
        rows.append(row)
        prev = row
    out = pd.DataFrame(rows)
    return out.drop(columns=["_params", "_fields", "_attrs"], errors="ignore")


def estate_trend(conn, farm_pks: list[int]) -> pd.DataFrame:
    """Per-date estate totals using each farm's job on or before that date (carry-forward) - flagged as mixed when
    farms are not assessed on the same date."""
    t = pd.concat([trend(conn, f).assign(farm_pk=f) for f in farm_pks], ignore_index=True) if farm_pks else pd.DataFrame()
    if t.empty:
        return t
    rows = []
    for d in sorted(t.assessment_date.unique()):
        latest = t[t.assessment_date <= d].sort_values(["assessment_date", "job_pk"]).groupby("farm_pk").tail(1)
        area, gap = latest.analysed_area_ha.sum(), latest.candidate_gap_ha.sum()
        rows.append({"date": d, "farms_included": len(latest), "farms_total": len(farm_pks),
                     "analysed_area_ha": round(area, 2), "candidate_gap_ha": round(gap, 3),
                     "gap_pct": round(gap_percentage(gap, area), 2),
                     "carried_forward": bool((latest.assessment_date != d).any()), "any_synthetic": bool(latest.synthetic.any())})
    return pd.DataFrame(rows)
