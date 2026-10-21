"""Validation of predicted candidate gaps against field-verified reference polygons.

Three distinct levels are reported and must not be conflated:
  1. Pixel level   - 10 m cells inside each field's evaluation domain (confusion matrix, P/R/F1/IoU).
  2. Polygon level - object matching (is each predicted/reference gap found by the other?).
  3. Field level   - total gap hectares per field (absolute error, signed bias).

Rules enforced:
  * Only fields that were FULLY surveyed are evaluated; a surveyed field with no reference polygons means
    "no gaps found", an unsurveyed field is never treated as zero gaps.
  * Evaluation domain = field geometry minus the analysis edge buffer, so prediction and reference are judged
    on the same ground.
  * Reference polygons must come from independent observation (ground survey, drone interpretation), never
    from the model's own predictions.
  * With few fields the pooled numbers are very uncertain; a bootstrap-over-fields interval is reported
    (resampling fields because pixels inside a field are spatially correlated).
"""
from __future__ import annotations

import json
from datetime import date, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from .areas import M2_PER_HA, to_crs_geom, utm_crs_for
from .providers.base import make_grid


def _safe_div(a, b):
    return float(a) / float(b) if b else float("nan")


def _poly_only(g):
    if g is None or g.is_empty:
        return None
    if not g.is_valid:
        g = make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    if g.geom_type == "GeometryCollection":
        parts = [x for x in g.geoms if x.geom_type in ("Polygon", "MultiPolygon") and not x.is_empty]
        return unary_union(parts) if parts else None
    return None


def assign_reference_to_fields(ref: gpd.GeoDataFrame, fields: gpd.GeoDataFrame, id_col: str | None):
    """Return (assigned GeoDataFrame[field_id, geometry], issues list).

    Uses `id_col` when given and valid, else the field with the largest overlap. Polygons are clipped to their
    field, so reference gaps can never cross field boundaries."""
    ref = ref.to_crs("EPSG:4326")
    fmap = {r.field_id: r.geometry for r in fields.itertuples() if r.geometry is not None}
    rows, issues = [], []
    for i, r in enumerate(ref.itertuples()):
        g = _poly_only(r.geometry)
        if g is None:
            issues.append(f"Reference feature {i}: not a usable polygon (skipped)."); continue
        fid = str(getattr(r, id_col)) if id_col and getattr(r, id_col, None) is not None else None
        if fid is not None and fid not in fmap:
            issues.append(f"Reference feature {i}: field_id '{fid}' not in this boundary version (skipped)."); continue
        if fid is None:
            utm = utm_crs_for(g)
            gp = to_crs_geom(g, "EPSG:4326", utm)
            best, area = None, 0.0
            for k, fg in fmap.items():
                a = gp.intersection(to_crs_geom(fg, "EPSG:4326", utm)).area
                if a > area:
                    best, area = k, a
            if best is None:
                issues.append(f"Reference feature {i}: overlaps no field (skipped)."); continue
            fid = best
        clipped = g.intersection(fmap[fid])
        clipped = _poly_only(clipped)
        if clipped is None:
            issues.append(f"Reference feature {i}: empty after clipping to field '{fid}' (skipped)."); continue
        if clipped.area < g.area * 0.999:
            issues.append(f"Reference feature {i}: clipped to field '{fid}' boundary.")
        rows.append({"field_id": fid, "geometry": clipped})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326"), issues


def evaluate_field(pred_polys, ref_polys, field_geom, edge_buffer_m: float = 10.0, resolution: float = 10.0,
                   min_overlap: float = 0.5) -> dict:
    """All inputs EPSG:4326 shapely geometries (lists for polygons)."""
    utm = utm_crs_for(field_geom)
    fg = to_crs_geom(field_geom, "EPSG:4326", utm)
    dom = fg.buffer(-edge_buffer_m) if edge_buffer_m > 0 else fg
    note = ""
    if dom.is_empty:
        dom, note = fg, "field too small for edge buffer; buffer not applied"
    P = [p for p in (to_crs_geom(g, "EPSG:4326", utm).intersection(dom) for g in pred_polys) if not p.is_empty]
    R = [p for p in (to_crs_geom(g, "EPSG:4326", utm).intersection(dom) for g in ref_polys) if not p.is_empty]
    P = [p for g in P if (p := _poly_only(g)) is not None]
    R = [p for g in R if (p := _poly_only(g)) is not None]

    transform, h, w = make_grid(fg.bounds, resolution)
    domain = features.geometry_mask([dom], (h, w), transform, invert=True)
    def ras(polys):
        return features.rasterize([(g, 1) for g in polys], (h, w), transform=transform, fill=0,
                                  dtype="uint8").astype(bool) & domain if polys else np.zeros((h, w), bool)
    pm, rm = ras(P), ras(R)
    tp = int((pm & rm).sum()); fp = int((pm & ~rm).sum()); fn = int((~pm & rm).sum())
    tn = int((domain & ~pm & ~rm).sum())

    # polygon-level matching
    pu = unary_union(P) if P else None
    ru = unary_union(R) if R else None
    pred_hit = sum(1 for g in P if ru is not None and g.intersection(ru).area / g.area >= min_overlap - 1e-6)
    ref_found = sum(1 for g in R if pu is not None and g.intersection(pu).area / g.area >= min_overlap - 1e-6)

    pred_ha = sum(g.area for g in P) / M2_PER_HA
    ref_ha = sum(g.area for g in R) / M2_PER_HA
    return {
        "domain_ha": dom.area / M2_PER_HA, "tp_px": tp, "fp_px": fp, "fn_px": fn, "tn_px": tn,
        "precision": _safe_div(tp, tp + fp), "recall": _safe_div(tp, tp + fn), "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "n_pred": len(P), "n_ref": len(R), "pred_polys_matched": pred_hit, "ref_polys_found": ref_found,
        "poly_precision": _safe_div(pred_hit, len(P)), "poly_recall": _safe_div(ref_found, len(R)),
        "pred_gap_ha": pred_ha, "ref_gap_ha": ref_ha, "area_error_ha": pred_ha - ref_ha,
        "abs_area_error_ha": abs(pred_ha - ref_ha),
        "area_error_pct_of_field": 100 * (pred_ha - ref_ha) / (dom.area / M2_PER_HA),
        "note": note,
    }


def pooled_metrics(df: pd.DataFrame) -> dict:
    """Pool pixel counts across fields (micro-average) and summarise field-level area errors."""
    if df.empty:
        return {}
    tp, fp, fn, tn = (int(df[c].sum()) for c in ("tp_px", "fp_px", "fn_px", "tn_px"))
    n_pred, n_ref = int(df.n_pred.sum()), int(df.n_ref.sum())
    err = df.area_error_ha.to_numpy()
    out = {
        "n_fields": int(len(df)),
        "pixel": {
            "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
            "precision": _safe_div(tp, tp + fp), "recall": _safe_div(tp, tp + fn),
            "f1": _safe_div(2 * tp, 2 * tp + fp + fn), "iou": _safe_div(tp, tp + fp + fn),
            "commission_error": _safe_div(fp, tp + fp), "omission_error": _safe_div(fn, tp + fn),
            "overall_accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
        },
        "polygon": {
            "n_pred": n_pred, "n_ref": n_ref,
            "precision": _safe_div(df.pred_polys_matched.sum(), n_pred),
            "recall": _safe_div(df.ref_polys_found.sum(), n_ref),
        },
        "field_total_area": {
            "pred_total_ha": float(df.pred_gap_ha.sum()), "ref_total_ha": float(df.ref_gap_ha.sum()),
            "bias_ha_per_field": float(err.mean()), "mae_ha": float(np.abs(err).mean()),
            "rmse_ha": float(np.sqrt((err ** 2).mean())),
            "relative_bias_pct": 100 * _safe_div(df.pred_gap_ha.sum() - df.ref_gap_ha.sum(), df.ref_gap_ha.sum()),
            "pearson_r": (float(np.corrcoef(df.pred_gap_ha, df.ref_gap_ha)[0, 1])
                          if len(df) >= 3 and df.pred_gap_ha.std() > 0 and df.ref_gap_ha.std() > 0 else float("nan")),
        },
    }
    return out


def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile 95% intervals from resampling FIELDS (pixels within a field are not independent)."""
    if len(df) < 5:
        return {}
    rng = np.random.default_rng(seed)
    vals = {"precision": [], "recall": [], "f1": [], "mae_ha": [], "bias_ha_per_field": []}
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, len(df), len(df))]
        m = pooled_metrics(s)
        for k in ("precision", "recall", "f1"):
            vals[k].append(m["pixel"][k])
        vals["mae_ha"].append(m["field_total_area"]["mae_ha"])
        vals["bias_ha_per_field"].append(m["field_total_area"]["bias_ha_per_field"])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}


def _d(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def evaluate_job(conn, job_pk: int, min_overlap: float = 0.5, max_days_between: int = 30) -> dict:
    """Evaluate a job's predictions against the reference surveys of the same boundary version."""
    from . import db
    job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_pk,)).fetchone()
    params = json.loads(job["params_json"])
    edge, res_m = float(params.get("edge_buffer_m", 10.0)), float(params.get("resolution_m", 10.0))
    fields = db.get_fields(conn, job["version_pk"]).set_index("field_id")
    results = db.job_results(conn, job_pk).set_index("field_id")
    gaps = db.job_gaps(conn, job_pk)
    surveys = db.list_surveys(conn, job["version_pk"])
    warnings, rows, skipped = [], [], []
    if job["synthetic"]:
        warnings.append("This job was produced from SYNTHETIC demo imagery; metrics say nothing about real satellite "
                        "accuracy. Use only to test the workflow.")
    if int(gaps["edited"].sum()) if len(gaps) else 0:
        warnings.append(f"{int(gaps['edited'].sum())} predicted polygons were edited in review; metrics reflect "
                        f"post-review predictions. Deleted polygons are no longer counted as predictions.")
    end = _d(job["date_to"])
    for fid in results.index:
        r = results.loc[fid]
        sv = surveys[(surveys.field_id == fid) & (surveys.fully_surveyed == 1)]
        if sv.empty:
            skipped.append({"field_id": fid, "reason": "No fully surveyed reference for this field"}); continue
        if r.status != "analysed":
            skipped.append({"field_id": fid, "reason": f"Field not analysed ({r.status}); cannot be scored"}); continue
        sv = sv.sort_values("observed_at").iloc[-1]
        ref = [shape(json.loads(x)) for x in db.survey_polygon_json(conn, int(sv.survey_pk))]
        pred = list(gaps[gaps.field_id == fid].geometry)
        m = evaluate_field(pred, ref, fields.at[fid, "geometry"], edge, res_m, min_overlap)
        sd = _d(sv.observed_at)
        lag = abs((sd - end).days) if sd and end else None
        m.update(field_id=fid, survey_date=sv.observed_at, observer=sv.observer, method=sv.method,
                 days_between=lag, timing_flag=bool(lag is not None and lag > max_days_between))
        rows.append(m)
    df = pd.DataFrame(rows)
    if len(df) and df.timing_flag.any():
        warnings.append(f"{int(df.timing_flag.sum())} field(s) were surveyed more than {max_days_between} days from "
                        f"the imagery window end; real change on the ground can masquerade as error.")
    pooled = pooled_metrics(df) if len(df) else {}
    if pooled and pooled["n_fields"] < 5:
        warnings.append(f"Only {pooled['n_fields']} field(s) evaluated; treat metrics as anecdotal, not as a "
                        f"performance claim.")
    if pooled and pooled["polygon"]["n_ref"] == 0:
        warnings.append("No reference gap polygons in the evaluated fields: recall and polygon recall are undefined.")
    return {"fields": df, "skipped": pd.DataFrame(skipped), "pooled": pooled, "ci95": bootstrap_ci(df) if len(df) else {},
            "warnings": warnings, "params": {"min_overlap": min_overlap, "edge_buffer_m": edge, "resolution_m": res_m,
                                             "max_days_between": max_days_between}}
