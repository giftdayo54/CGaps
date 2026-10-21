import json
import math

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box, mapping

from canegap import db, samples, validation
from canegap.geojson_import import validate_geojson
from canegap.pipeline import run_analysis
from canegap.providers import DemoProvider

X0, Y0 = 600_000.0, 8_230_000.0


def wgs(g):
    return gpd.GeoSeries([g], crs="EPSG:32736").to_crs("EPSG:4326").iloc[0]


FIELD = wgs(box(X0, Y0, X0 + 300, Y0 + 340))


def ev(pred, ref, **kw):
    return validation.evaluate_field([wgs(p) for p in pred], [wgs(r) for r in ref], FIELD, **kw)


def test_perfect_agreement():
    b = box(X0 + 100, Y0 + 100, X0 + 200, Y0 + 200)
    m = ev([b], [b])
    assert m["precision"] == m["recall"] == m["f1"] == m["iou"] == 1.0
    assert m["area_error_ha"] == pytest.approx(0, abs=1e-6) and m["pred_gap_ha"] == pytest.approx(1.0, rel=1e-3)
    assert m["tp_px"] == 100 and m["fp_px"] == 0 and m["fn_px"] == 0


def test_half_overlap_known_values():
    pred = box(X0 + 100, Y0 + 100, X0 + 200, Y0 + 200)
    ref = box(X0 + 150, Y0 + 100, X0 + 250, Y0 + 200)
    m = ev([pred], [ref])
    assert (m["tp_px"], m["fp_px"], m["fn_px"]) == (50, 50, 50)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["iou"] == pytest.approx(1 / 3)
    assert m["area_error_ha"] == pytest.approx(0, abs=1e-6)          # equal totals despite misplacement
    assert m["poly_precision"] == 1.0 and m["poly_recall"] == 1.0    # >=50% overlap counts as a match


def test_overestimate_gives_positive_bias_and_full_recall():
    ref = box(X0 + 100, Y0 + 100, X0 + 200, Y0 + 200)               # 1 ha
    pred = box(X0 + 100, Y0 + 100, X0 + 200, Y0 + 300)              # 2 ha, contains ref
    m = ev([pred], [ref])
    assert m["recall"] == 1.0 and m["precision"] == 0.5
    assert m["area_error_ha"] == pytest.approx(1.0, rel=1e-3)


def test_surveyed_clean_field_with_false_positive():
    m = ev([box(X0 + 100, Y0 + 100, X0 + 150, Y0 + 150)], [])
    assert m["precision"] == 0.0 and math.isnan(m["recall"]) and m["fp_px"] == 25 and m["tn_px"] > 0
    assert m["pred_gap_ha"] > 0 and m["ref_gap_ha"] == 0


def test_edge_buffer_excludes_boundary_strip():
    inside_edge = box(X0, Y0 + 100, X0 + 8, Y0 + 200)                # entirely within the 10 m buffer
    assert ev([inside_edge], [])["fp_px"] == 0
    assert ev([inside_edge], [], edge_buffer_m=0)["fp_px"] > 0


def test_pooling_is_micro_average_and_ci_needs_5_fields():
    df = pd.DataFrame([
        dict(tp_px=90, fp_px=10, fn_px=10, tn_px=900, n_pred=1, n_ref=1, pred_polys_matched=1, ref_polys_found=1,
             pred_gap_ha=1.0, ref_gap_ha=1.0, area_error_ha=0.0),
        dict(tp_px=0, fp_px=0, fn_px=10, tn_px=90, n_pred=0, n_ref=1, pred_polys_matched=0, ref_polys_found=0,
             pred_gap_ha=0.0, ref_gap_ha=0.1, area_error_ha=-0.1)])
    p = validation.pooled_metrics(df)
    assert p["pixel"]["recall"] == pytest.approx(90 / 110) and p["pixel"]["precision"] == pytest.approx(0.9)
    assert p["pixel"]["omission_error"] == pytest.approx(20 / 110)
    assert p["field_total_area"]["mae_ha"] == pytest.approx(0.05)
    assert validation.bootstrap_ci(df) == {}
    assert validation.bootstrap_ci(pd.concat([df] * 3)) != {}


def test_reference_assignment_clips_and_reports():
    vr = validate_geojson(samples.farm_with_fields(False))
    fields = vr.features
    f1 = fields.iloc[0].geometry
    stick_out = f1.buffer(0.0003)                                    # crosses into neighbours
    ref = gpd.GeoDataFrame({"fid": ["F01", "NOPE"]}, geometry=[stick_out, f1], crs="EPSG:4326")
    out, issues = validation.assign_reference_to_fields(ref, fields, "fid")
    assert list(out.field_id) == ["F01"] and any("NOPE" in i for i in issues) and any("clipped" in i for i in issues)
    assert fields.iloc[0].geometry.buffer(1e-9).contains(out.geometry.iloc[0])
    spatial, _ = validation.assign_reference_to_fields(
        gpd.GeoDataFrame(geometry=[fields.iloc[1].geometry.centroid.buffer(0.0002)], crs="EPSG:4326"), fields, None)
    assert list(spatial.field_id) == ["F02"]


def _job(conn):
    vr = validate_geojson(samples.farm_with_fields(True), outline_rows=[6])
    ids = db.save_import(conn, vr, "t")[0]
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01", "F02", "F05", "F06"], DemoProvider(),
                       "2026-08-01", "2026-10-15")
    return ids, job


def test_evaluate_job_scores_only_fully_surveyed_fields(conn):
    ids, job = _job(conn)
    gaps = db.job_gaps(conn, job)
    g2 = list(gaps[gaps.field_id == "F02"].geometry)
    # F02: independent reference identical to prediction (fully surveyed); F01: partial survey; F05: clean survey
    db.add_survey(conn, ids["farm_pk"], ids["version_pk"], "F02", "2026-10-10", "obs", "drone", True, "", g2)
    db.add_survey(conn, ids["farm_pk"], ids["version_pk"], "F01", "2026-10-10", "obs", "walk", False, "", [])
    db.add_survey(conn, ids["farm_pk"], ids["version_pk"], "F05", "2026-10-10", "obs", "walk", True, "", [])
    r = validation.evaluate_job(conn, job)
    f = r["fields"].set_index("field_id")
    assert set(f.index) == {"F02", "F05"}
    assert f.at["F02", "f1"] == pytest.approx(1.0) and f.at["F02", "area_error_ha"] == pytest.approx(0, abs=1e-6)
    assert f.at["F05", "ref_gap_ha"] == 0 and f.at["F05", "fp_px"] > 0           # clean field, model flagged gaps
    sk = r["skipped"].set_index("field_id")["reason"]
    assert "No fully surveyed" in sk["F06"] and "No fully surveyed" in sk["F01"]
    assert any("SYNTHETIC" in w for w in r["warnings"]) and any("Only 2 field" in w for w in r["warnings"])
    vid = db.save_validation(conn, job, r)
    assert len(db.list_validations(conn, job)) == 1 and vid


def test_field_not_analysed_is_not_scored(conn):
    vr = validate_geojson(samples.farm_with_fields(True), outline_rows=[6])
    ids = db.save_import(conn, vr, "t")[0]
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F03"], DemoProvider(), "2026-08-01", "2026-10-15")
    db.add_survey(conn, ids["farm_pk"], ids["version_pk"], "F03", "2026-10-10", "o", "walk", True, "", [])
    r = validation.evaluate_job(conn, job)
    assert r["fields"].empty and "not analysed" in r["skipped"].iloc[0].reason
