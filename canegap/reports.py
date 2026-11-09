"""PDF reports (field and farm) built with ReportLab; maps drawn with matplotlib."""
from __future__ import annotations

import io
import json
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

from . import ALGO_VERSION, aggregate, db, interventions

SS = getSampleStyleSheet()
SMALL = SS["BodyText"].clone("small", fontSize=8, leading=10)
WARN = SS["BodyText"].clone("warn", textColor=colors.HexColor("#b71c1c"), fontSize=9, leading=11)

METHOD_NOTES = (
    f"Method ({ALGO_VERSION}): per-field masking of Sentinel-2 surface reflectance; a pixel is a candidate gap when NDVI is "
    "below an absolute threshold and well below the field's own median; very low NDVI = no detectable cane; low NDVI with high "
    "MNDWI = possible waterlogging. Edge pixels are excluded; polygons smaller than the minimum mapping unit are dropped; "
    "polygons are clipped to the parent field and measured in a local UTM projection.")
LIMITS = (
    "Limitations: unvalidated screening output unless a validation assessment is cited. Sentinel-2 pixels are 10 m (red-edge and "
    "SWIR bands native 20 m), so areas are quantised and edge pixels are mixed. Low vegetation index does not prove missing stools "
    "(weeds, stress, lodging, shadow, harvest trash and tracks can look similar). Confidence values are heuristic scores, not "
    "measured accuracy. Candidate gaps require field verification before intervention decisions.")


def _method_note(job) -> str:
    m = json.loads(job["params_json"] or "{}").get("model")
    if not m:
        return METHOD_NOTES
    return (f"Method ({m['label']}): Random Forest trained on field-verified polygons using the '{m['feature_set']}' feature set "
            f"(probability threshold {m['threshold']}). Scores are RF probabilities, not calibrated and not accuracy. "
            + ("Cross-validated internally by holding out whole fields; confirm on new fields/seasons. " if m.get("cv_validated")
               else "NOT cross-validated (too few surveyed fields): quality unknown. ")
            + "Polygons are clipped to the parent field and measured in a local UTM projection.")


def _p(text, style=None):
    return Paragraph(escape(str(text)), style or SS["BodyText"])


def _table(df: pd.DataFrame, col_widths=None, font=7.5):
    data = [list(df.columns)] + [[("" if (isinstance(v, float) and v != v) else v) for v in row] for row in df.values.tolist()]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), font), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f8e9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def _map_png(field_geom, gaps_gdf, title: str, width_in=6.0, height_in=4.2) -> bytes:
    import geopandas as gpd
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=150)
    fg = gpd.GeoSeries([field_geom], crs="EPSG:4326")
    utm = fg.estimate_utm_crs()
    fg.to_crs(utm).plot(ax=ax, facecolor="#e8f5e9", edgecolor="#1b5e20", linewidth=1.5)
    if gaps_gdf is not None and len(gaps_gdf):
        gaps_gdf.to_crs(utm).plot(ax=ax, facecolor="#ff1744", edgecolor="#b71c1c", alpha=0.65, linewidth=0.6)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6); ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_xlabel(f"Easting (m, {utm.to_string()})", fontsize=6); ax.set_ylabel("Northing (m)", fontsize=6)
    ax.grid(alpha=0.3, linewidth=0.3)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


def _header(story, conn, job, title):
    farm = conn.execute("SELECT * FROM farms WHERE id=?", (job["farm_pk"],)).fetchone()
    ver = conn.execute("SELECT * FROM boundary_versions WHERE id=?", (job["version_pk"],)).fetchone()
    story.append(Paragraph(escape(title), SS["Title"]))
    story.append(_p(f"{farm['farm_name']} ({farm['farm_id']})" + (f" - estate {farm['estate']}" if farm["estate"] else "")))
    story.append(_p(f"Job {job['id']} | source {job['source']} | imagery window {job['date_from']} to {job['date_to']} | "
                    f"algorithm {job['algo_version']} | boundary version {ver['version_no']} ({ver['filename']}) | "
                    f"generated {db.now()}", SMALL))
    if job["synthetic"]:
        story.append(Spacer(1, 4)); story.append(_p("SYNTHETIC DEMO DATA - NOT SATELLITE IMAGERY. Figures below are not real "
                                                     "observations.", WARN))
    story.append(Spacer(1, 4))
    story.append(_p("UNVALIDATED SCREENING OUTPUT: candidate gaps are model predictions, not confirmed missing cane.", WARN))
    story.append(Spacer(1, 6))
    return farm


def _validation_section(story, conn, job_pk):
    r = conn.execute("SELECT * FROM validations WHERE job_pk=? ORDER BY id DESC LIMIT 1", (job_pk,)).fetchone()
    story.append(Paragraph("Validation status", SS["Heading3"]))
    if not r:
        story.append(_p("No validation assessment against field-verified polygons has been run for this job; accuracy is unknown."))
        return
    pooled = json.loads(r["pooled_json"] or "{}")
    if not pooled:
        story.append(_p("A validation run exists but scored no fields.")); return
    px, fa = pooled["pixel"], pooled["field_total_area"]
    f = lambda x: "n/a" if x is None or x != x else f"{x:.2f}"  # noqa: E731
    story.append(_p(f"Latest assessment ({r['created_at'][:16]}) on {pooled['n_fields']} fully surveyed field(s): pixel precision "
                    f"{f(px['precision'])}, recall {f(px['recall'])}, F1 {f(px['f1'])}, IoU {f(px['iou'])}; total gap area bias "
                    f"{f(fa['bias_ha_per_field'])} ha/field, MAE {f(fa['mae_ha'])} ha/field. "
                    + ("Few fields: treat as indicative only." if pooled["n_fields"] < 5 else ""), SMALL))
    for w in json.loads(r["warnings_json"] or "[]"):
        story.append(_p("Note: " + w, SMALL))


def _build(story, title):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm, title=title, author="CaneGap Intelligence")
    doc.build(story)
    return buf.getvalue()


def _interv_table(conn, farm_pk, field_id=None):
    df = db.list_interventions(conn, farm_pk=farm_pk, field_id=field_id)
    if df.empty:
        return None, {}
    show = df[["id", "field_id", "gap_id", "action", "priority", "assigned_to", "target_date", "status",
               "proposed_area_ha", "actual_area_ha"]].round(3)
    return show, interventions.summary(df)


def field_report(conn, job_pk: int, field_id: str) -> bytes:
    job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_pk,)).fetchone()
    res = db.job_results(conn, job_pk)
    res = res[res.field_id == field_id]
    if res.empty:
        raise ValueError(f"Field {field_id} not in job {job_pk}")
    r = res.iloc[0]
    fields = db.get_fields(conn, int(job["version_pk"]))
    fg = fields[fields.field_id == field_id].iloc[0]
    gaps = db.job_gaps(conn, job_pk); gaps = gaps[gaps.field_id == field_id].sort_values("area_ha", ascending=False)
    story = []
    _header(story, conn, job, f"Field report - {field_id}")
    snap = json.loads(job["params_json"] or "{}").get("field_attributes", {}).get(field_id)
    src = {"field_name": fg["field_name"], **(snap if snap is not None else {k: fg[k] for k in ("crop", "variety", "ratoon_number", "last_harvest_date", "planting_date")})}
    attrs = {k: src[k] for k in ("field_name", "crop", "variety", "ratoon_number", "last_harvest_date", "planting_date")
             if k in src and src[k] is not None and not (isinstance(src[k], float) and src[k] != src[k]) and str(src[k]) != ""}
    story.append(_p("Field (attributes as used by this job): " + "; ".join(f"{k} {v}" for k, v in {"ID": field_id, **attrs}.items())))
    story.append(Spacer(1, 4))
    facts = pd.DataFrame([{
        "Status": r.status, "Field area (ha)": round(r.field_area_ha, 2), "Analysed area (ha)": round(r.analysed_area_ha or 0, 2),
        "Valid obs. fraction": round(r.valid_fraction or 0, 2), "Candidate gap (ha)": round(r.gap_area_ha or 0, 3),
        "Gap %": round(r.gap_pct or 0, 2), "Polygons": int(r.n_gaps or 0)}])
    story.append(_table(facts, font=8))
    if r.reason:
        story.append(Spacer(1, 4)); story.append(_p("Reason / note: " + r.reason, SMALL))
    story.append(Spacer(1, 6))
    story.append(Image(io.BytesIO(_map_png(fg.geometry, gaps, f"{field_id}: candidate gaps (model predictions)")),
                       width=15 * cm, height=10.5 * cm))
    if len(gaps):
        g = gaps[["gap_id", "area_ha", "pct_of_field", "category", "confidence", "change_status", "verification_status",
                  "verified_area_ha", "proposed_action"]].head(25).round(4)
        g.columns = ["Gap", "Area ha", "% field", "Category", "Score*", "Change", "Verification", "Verified ha", "Proposed action"]
        story.append(Paragraph("Candidate gap polygons (largest 25)", SS["Heading3"])); story.append(_table(g))
        story.append(_p("*Heuristic score, not measured accuracy. Predicted, field-verified, proposed and treated areas are separate.", SMALL))
    q = json.loads(r.quality_json or "{}")
    scenes = json.loads(r.scenes_json or "[]")
    story.append(Paragraph("Satellite observation metadata", SS["Heading3"]))
    for s_ in scenes:
        story.append(_p("; ".join(f"{k}: {v}" for k, v in s_.items()), SMALL))
    for k in ("pixel_size_m", "median_obs_count", "field_median_ndvi", "ndvi_threshold_used"):
        if k in q and q[k] is not None:
            story.append(_p(f"{k}: {q[k]}", SMALL))
    for n_ in q.get("notes", []):
        story.append(_p("Note: " + n_, SMALL))
    show, summ = _interv_table(conn, int(job["farm_pk"]), field_id)
    if show is not None:
        story.append(Paragraph("Interventions", SS["Heading3"])); story.append(_table(show))
        story.append(_p(f"Proposed area {summ['proposed_area_ha']:.3f} ha; actual treated {summ['actual_treated_area_ha']:.3f} ha.", SMALL))
    _validation_section(story, conn, job_pk)
    story += [Spacer(1, 8), _p(_method_note(job), SMALL), Spacer(1, 4), _p(LIMITS, SMALL)]
    return _build(story, f"Field report {field_id}")


def farm_report(conn, job_pk: int, top_n: int = 15) -> bytes:
    job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_pk,)).fetchone()
    res = db.job_results(conn, job_pk)
    gaps = db.job_gaps(conn, job_pk)
    params = json.loads(job["params_json"])
    s = aggregate.farm_summary(res, job["overlap_policy"], params.get("n_overlaps", 0))
    story = []
    _header(story, conn, job, "Farm gap assessment report")
    summ = pd.DataFrame([{"Fields": s["n_fields"], "Analysed": s["n_analysed"], "Insufficient evidence": s["n_insufficient_evidence"],
                          "Harvest/planting excl.": s["n_context_excluded"], "Unable": s["n_unable"],
                          "Analysed ha": s["analysed_area_ha"], "Candidate gap ha": s["candidate_gap_area_ha"],
                          "Farm gap %": s["gap_pct"], "Polygons": s["n_gap_polygons"]}])
    story.append(_table(summ, font=8))
    story.append(_p("Farm gap % = total candidate-gap ha / total analysed ha (area-weighted, not an average of field percentages).", SMALL))
    if s["double_count_risk"]:
        story.append(_p("Warning: overlapping fields were only flagged; totals may double count overlapping area.", WARN))
    story.append(Paragraph("Field-by-field results", SS["Heading3"]))
    ver = gaps.groupby("field_id").verification_status.apply(lambda x: f"{(x != 'Pending Verification').sum()}/{len(x)} reviewed").to_dict() if len(gaps) else {}
    ft = res[["field_id", "status", "field_area_ha", "gap_area_ha", "gap_pct", "n_gaps", "valid_fraction"]].copy()
    ft["verification"] = ft.field_id.map(ver).fillna("-")
    ft = ft.round({"field_area_ha": 2, "gap_area_ha": 3, "gap_pct": 2, "valid_fraction": 2}).sort_values("gap_area_ha", ascending=False)
    ft.columns = ["Field", "Status", "Area ha", "Gap ha", "Gap %", "Gaps", "Valid obs.", "Verification"]
    story.append(_table(ft))
    if len(gaps):
        top = gaps.sort_values("area_ha", ascending=False).head(top_n)[
            ["gap_id", "field_id", "area_ha", "pct_of_field", "category", "confidence", "change_status", "verification_status"]].round(4)
        top.columns = ["Gap", "Field", "Area ha", "% field", "Category", "Score*", "Change", "Verification"]
        story.append(Paragraph(f"Largest {len(top)} candidate gaps", SS["Heading3"])); story.append(_table(top))
        story.append(_p("*Heuristic score, not measured accuracy.", SMALL))
    show, isum = _interv_table(conn, int(job["farm_pk"]))
    if show is not None:
        story.append(Paragraph("Intervention workload", SS["Heading3"]))
        story.append(_p("Status counts: " + ", ".join(f"{k} {v}" for k, v in isum["by_status"].items())
                        + f". Proposed {isum['proposed_area_ha']:.2f} ha; actual treated {isum['actual_treated_area_ha']:.2f} ha; "
                          f"estimated cost {isum['estimated_cost']:,.0f}; actual cost {isum['actual_cost']:,.0f}.", SMALL))
    _validation_section(story, conn, job_pk)
    story += [Spacer(1, 8), _p(_method_note(job), SMALL), Spacer(1, 4), _p(LIMITS, SMALL)]
    return _build(story, "Farm gap assessment report")
