import json

import pandas as pd
import pytest

from canegap import dashboards as dsh, db, samples
from canegap.areas import gap_percentage
from canegap.geojson_import import validate_geojson
from canegap.pipeline import Params, run_analysis
from canegap.providers import DemoProvider


class Real(DemoProvider):
    """Pretends to be non-synthetic so dashboards (which exclude synthetic by default) include it. TEST ONLY."""
    synthetic = False
    name = "test-real"


def _farm(conn, farm_id, estate, n=2):
    fc = samples.farm_with_fields(False)
    for f in fc["features"]:
        f["properties"]["farm_id"] = farm_id; f["properties"]["farm_name"] = farm_id
    vr = validate_geojson(fc)
    return db.save_import(conn, vr, f"{farm_id}.geojson", estate=estate)[0]


def _job(conn, ids, prov, date_to, fields=("F01", "F02", "F05"), **kw):
    return run_analysis(conn, ids["farm_pk"], ids["version_pk"], list(fields), prov, "2026-08-01", date_to, **kw)


def test_estate_totals_are_area_weighted_sums(conn):
    a, b, c = _farm(conn, "A", "East"), _farm(conn, "B", "East"), _farm(conn, "C", None)
    ja = _job(conn, a, Real(gap_fraction=0.12), "2026-10-01")
    jb = _job(conn, b, Real(gap_fraction=0.02, seed=7), "2026-10-05", fields=("F01", "F02", "F05", "F06"))
    jc = _job(conn, c, Real(), "2026-10-05")
    farms = dsh.farm_table(conn, [ja, jb, jc])
    est = dsh.estate_table(farms).set_index("estate")
    east = farms[farms.estate == "East"]
    assert est.at["East", "candidate_gap_ha"] == pytest.approx(east.candidate_gap_ha.sum())
    assert est.at["East", "gap_pct"] == pytest.approx(gap_percentage(east.candidate_gap_ha.sum(), east.analysed_area_ha.sum()), abs=0.01)
    assert est.at["East", "gap_pct"] != pytest.approx(east.gap_pct.mean(), abs=1e-6)     # not a mean of percentages
    assert est.at["East", "mixed_assessment_dates"] and dsh.NO_ESTATE in est.index
    assert est.at["East", "n_farms"] == 2


def test_latest_job_selection_excludes_synthetic_by_default(conn):
    a = _farm(conn, "A", "East")
    _job(conn, a, DemoProvider(), "2026-10-01")
    assert dsh.latest_jobs(conn).empty and len(dsh.latest_jobs(conn, include_synthetic=True)) == 1
    j2 = _job(conn, a, Real(), "2026-10-02")
    _job(conn, a, Real(), "2026-09-01")                         # older imagery date, later id
    assert dsh.latest_jobs(conn).iloc[0].job_pk == j2


def test_trend_flags_unreliable_comparisons(conn):
    a = _farm(conn, "A", "East")
    _job(conn, a, Real(), "2026-08-31")
    _job(conn, a, Real(), "2026-09-30")                          # identical settings -> comparable
    p = Params(); p.ndvi_abs_threshold = 0.4
    _job(conn, a, Real(), "2026-10-31", params=p)                # parameters changed
    v2 = db.save_import(conn, validate_geojson(samples.farm_with_fields(False)), "v2", target_farm_pk=a["farm_pk"])[0]
    run_analysis(conn, a["farm_pk"], v2["version_pk"], ["F01", "F02", "F05"], Real(), "2026-11-01", "2026-11-30", params=p)
    t = dsh.trend(conn, a["farm_pk"])
    assert list(t.assessment_date) == sorted(t.assessment_date)
    assert not t.iloc[0].comparable_with_previous and t.iloc[1].comparable_with_previous
    assert "parameters" in t.iloc[2].comparability_flags and not t.iloc[2].comparable_with_previous
    assert "boundary version changed" in t.iloc[3].comparability_flags
    mixed = dsh.trend(conn, a["farm_pk"], include_synthetic=True)
    assert len(mixed) == 4


def test_estate_trend_carries_forward_and_marks_it(conn):
    a, b = _farm(conn, "A", "E"), _farm(conn, "B", "E")
    _job(conn, a, Real(), "2026-09-01"); _job(conn, b, Real(), "2026-10-01"); _job(conn, a, Real(), "2026-11-01")
    t = dsh.estate_trend(conn, [a["farm_pk"], b["farm_pk"]])
    assert list(t.farms_included) == [1, 2, 2] and list(t.carried_forward) == [False, True, True]
