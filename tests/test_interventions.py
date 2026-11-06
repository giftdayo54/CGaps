import pytest
from canegap import db, interventions as iv, samples
from canegap.geojson_import import validate_geojson
from canegap.pipeline import run_analysis
from canegap.providers import DemoProvider


def _job(conn):
    ids = db.save_import(conn, validate_geojson(samples.farm_with_fields(True), outline_rows=[6]), "t")[0]
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01", "F02", "F05"], DemoProvider(), "2026-08-01", "2026-10-15")
    return ids, job


def test_cost_estimate_uses_only_entered_rates():
    assert iv.estimate(2.0, iv.Rates())["est_cost"] == 0
    e = iv.estimate(2.0, iv.Rates(labour_days_per_ha=10, labour_cost_per_day=5, material_units_per_ha=3,
                                  material_cost_per_unit=100, other_cost_per_ha=7))
    assert e["labour_days"] == 20 and e["material_units"] == 6 and e["est_cost"] == 20 * 5 + 6 * 100 + 14
    with pytest.raises(ValueError):
        iv.estimate(-1, iv.Rates())


def test_priority_heuristic():
    assert iv.suggest_priority(0.3, 0.8) == "High" and iv.suggest_priority(0.3, 0.2) == "Medium"
    assert iv.suggest_priority(0.01, 0.9) == "Low"


def test_worklist_from_gaps_no_duplicates_and_links(conn):
    ids, job = _job(conn)
    n, sk = iv.worklist_from_gaps(conn, job, min_area_ha=0.0)
    gaps = db.job_gaps(conn, job)
    assert n == len(gaps) > 0 and not sk
    df = db.list_interventions(conn, farm_pk=ids["farm_pk"])
    assert set(df.gap_id) == set(gaps.gap_id) and (df.status == "Pending Verification").all()
    assert df.farm_id.eq("SYN_FARM_01").all() and df.version_pk.eq(ids["version_pk"]).all()
    n2, sk2 = iv.worklist_from_gaps(conn, job, min_area_ha=0.0)
    assert n2 == 0 and len(sk2) == n                                    # duplicates refused


def test_completion_requires_actuals_and_summary_separates_values(conn):
    ids, job = _job(conn)
    iv.worklist_from_gaps(conn, job, min_area_ha=0.0)
    df = db.list_interventions(conn)
    i0 = int(df.iloc[0].id)
    with pytest.raises(ValueError):
        db.update_intervention(conn, i0, status="Completed")
    db.update_intervention(conn, i0, status="Completed", treatment_date="2026-10-18", actual_area_ha=0.02,
                           actual_cost=150.0, verified_cause="Skip from planting")
    s = iv.summary(db.list_interventions(conn))
    assert s["actual_treated_area_ha"] == pytest.approx(0.02) and s["actual_cost"] == 150.0
    assert s["proposed_area_ha"] > s["actual_treated_area_ha"] and s["by_status"]["Completed"] == 1


def test_filters_and_validation(conn):
    ids, job = _job(conn)
    iv.worklist_from_gaps(conn, job, min_area_ha=0.0)
    fid = db.list_interventions(conn).iloc[0].field_id
    assert set(db.list_interventions(conn, field_id=fid).field_id) == {fid}
    assert db.list_interventions(conn, status="Scheduled").empty
    with pytest.raises(ValueError):
        db.create_intervention(conn, farm_pk=1, farm_id="x", field_id="F01", action="Do magic")
    with pytest.raises(ValueError):
        db.update_intervention(conn, int(db.list_interventions(conn).iloc[0].id), status="Bogus")


def test_pdf_reports_generate(conn):
    from canegap import reports, validation
    ids, job = _job(conn)
    iv.worklist_from_gaps(conn, job, min_area_ha=0.0)
    db.add_survey(conn, ids["farm_pk"], ids["version_pk"], "F05", "2026-10-10", "o", "walk", True, "", [])
    db.save_validation(conn, job, validation.evaluate_job(conn, job))
    farm = reports.farm_report(conn, job)
    fld = reports.field_report(conn, job, "F02")
    assert farm[:5] == b"%PDF-" and fld[:5] == b"%PDF-" and len(farm) > 4000 and len(fld) > 20000   # fld has map image
    with pytest.raises(ValueError):
        reports.field_report(conn, job, "NOPE")
