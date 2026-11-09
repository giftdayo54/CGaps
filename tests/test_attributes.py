from datetime import date

import pandas as pd
import pytest

from canegap import attributes as A, db, samples
from canegap.geojson_import import subset_validation, validate_geojson
from canegap.pipeline import Params, analyse_field, queue_analysis, execute_job, run_analysis
from canegap.providers import DemoProvider

TODAY = date(2026, 11, 6)


def test_normalisation_rules():
    n = lambda k, v: A.normalise_value(k, v, TODAY)  # noqa: E731
    assert n("planting_date", "15/08/2026") == "2026-08-15" and n("planting_date", "2026-8-5" if False else "2026-08-05") == "2026-08-05"
    assert n("last_harvest_date", date(2026, 7, 1)) == "2026-07-01" and n("planting_date", "  ") is None
    assert n("ratoon_number", "2.0") == "2" and n("ratoon_number", 0) == "0" and n("variety", "  N41 ") == "N41"
    for key, bad in (("planting_date", "yesterday"), ("planting_date", "1980-01-01"), ("planting_date", "2028-01-01"),
                     ("ratoon_number", "1.5"), ("ratoon_number", "20"), ("ratoon_number", "abc"), ("variety", "x" * 61),
                     ("geometry", "x")):
        with pytest.raises(A.AttributeError_):
            n(key, bad)


def test_clean_changes_collects_errors_and_warnings():
    clean, errors, warnings = A.clean_changes({"F01": {"planting_date": "2026-05-01", "last_harvest_date": "2026-01-01"},
                                               "F02": {"ratoon_number": "x"}}, TODAY)
    assert len(errors) == 1 and "F02" in errors[0] and len(warnings) == 1 and "before the planting date" in warnings[0]
    assert clean["F01"]["planting_date"] == "2026-05-01"


def test_changed_rows_only_reports_real_differences():
    before = pd.DataFrame({"field_id": ["A", "B"], "variety": ["N41", None], "planting_date": [None, "2026-01-01"], "ratoon_number": [None, None]})
    after = pd.DataFrame({"field_id": ["A", "B"], "variety": ["N41", "R579"], "planting_date": [None, "2026-01-01"], "ratoon_number": [None, None]})
    assert A.changed_rows(before, after) == {"B": {"variety": "R579"}}


def _farm(conn):
    ids = db.save_import(conn, validate_geojson(samples.farm_with_fields(True), outline_rows=[6]), "t")[0]
    return ids


def test_update_attributes_persists_logs_history_and_is_all_or_nothing(conn):
    ids = _farm(conn)
    r = db.update_field_attributes(conn, ids["version_pk"], {"F01": {"variety": "NCo376", "planting_date": "2026-03-01"},
                                                           "F02": {"ratoon_number": "3"}}, user="ana")
    assert r["changed"] == 3 and r["fields"] == 2
    f = db.get_fields(conn, ids["version_pk"]).set_index("field_id")
    assert f.at["F01", "variety"] == "NCo376" and f.at["F01", "planting_date"] == "2026-03-01" and f.at["F02", "ratoon_number"] == "3"
    h = db.attribute_history(conn, ids["version_pk"], "F01")
    assert set(h.attr) == {"variety", "planting_date"} and (h.changed_by == "ana").all() and len(h) == 2
    assert (db.attribute_history(conn, ids["version_pk"]).source == "manual").all()
    # one bad value -> nothing at all is written
    with pytest.raises(A.AttributeError_):
        db.update_field_attributes(conn, ids["version_pk"], {"F05": {"variety": "OK"}, "F06": {"planting_date": "nonsense"}})
    assert db.get_fields(conn, ids["version_pk"]).set_index("field_id").at["F05", "variety"] != "OK"
    with pytest.raises(A.AttributeError_, match="Unknown field"):
        db.update_field_attributes(conn, ids["version_pk"], {"NOPE": {"variety": "x"}})
    # no-op edits are not logged; geometry untouched
    n = len(db.attribute_history(conn, ids["version_pk"]))
    assert db.update_field_attributes(conn, ids["version_pk"], {"F01": {"variety": "NCo376"}})["changed"] == 0
    assert len(db.attribute_history(conn, ids["version_pk"])) == n


def test_manual_planting_date_changes_the_analysis(conn):
    """A field with no dates is analysed for gaps; adding a recent planting date flips it to 'context excluded'."""
    ids = _farm(conn)
    run = lambda: run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01"], DemoProvider(), "2026-08-01", "2026-10-15")  # noqa: E731
    db.update_field_attributes(conn, ids["version_pk"], {"F01": {"last_harvest_date": None, "planting_date": None}})
    j1 = run()
    r1 = db.job_results(conn, j1).iloc[0]
    assert r1.status == "analysed" and any("No planting or last-harvest date" in n for n in __import__("json").loads(r1.quality_json)["notes"])
    db.update_field_attributes(conn, ids["version_pk"], {"F01": {"planting_date": "2026-09-20", "variety": "R579"}})
    j2 = run()
    r2 = db.job_results(conn, j2).iloc[0]
    assert r2.status == "context_excluded" and "Newly planted" in r2.reason


def test_job_uses_attributes_frozen_at_queue_time(conn):
    import json
    ids = _farm(conn)
    db.update_field_attributes(conn, ids["version_pk"], {"F01": {"last_harvest_date": None, "planting_date": None, "variety": "A1"}})
    job = queue_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01"], DemoProvider(), "2026-08-01", "2026-10-15")
    db.update_field_attributes(conn, ids["version_pk"], {"F01": {"planting_date": "2026-09-20", "variety": "B2"}})   # edited while queued
    execute_job(conn, job, DemoProvider())
    r = db.job_results(conn, job).iloc[0]
    assert r.status == "analysed"                                                  # the queued snapshot had no planting date
    assert json.loads(r.quality_json)["attributes_used"]["variety"] == "A1"
    assert json.loads(db.get_job(conn, job)["params_json"])["field_attributes"]["F01"]["variety"] == "A1"


def test_attribute_change_makes_a_new_job_not_a_duplicate(conn):
    from canegap.pipeline import DuplicateJob
    ids = _farm(conn)
    q = lambda: queue_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01"], DemoProvider(), "2026-08-01", "2026-10-15")  # noqa: E731
    q()
    with pytest.raises(DuplicateJob):
        q()
    db.update_field_attributes(conn, ids["version_pk"], {"F01": {"variety": "ZZ9"}})
    assert q()


def test_trend_flags_attribute_changes(conn):
    from canegap import dashboards as dsh
    class Real(DemoProvider):
        synthetic = False; name = "real-test"
    ids = _farm(conn)
    run = lambda d: run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01", "F02"], Real(), "2026-08-01", d)  # noqa: E731
    run("2026-09-01")
    db.update_field_attributes(conn, ids["version_pk"], {"F02": {"variety": "NEW"}})
    run("2026-10-01")
    t = dsh.trend(conn, ids["farm_pk"])
    assert "field attributes" in t.iloc[1].comparability_flags and "parameters" not in t.iloc[1].comparability_flags


def test_subset_validation_keeps_selected_fields_and_outline_and_says_what_was_dropped():
    vr = validate_geojson(samples.farm_with_fields(True), outline_rows=[6])
    sub = subset_validation(vr, ["F02", "F05"])
    assert sorted(sub.features[~sub.features.is_outline].field_id) == ["F02", "F05"] and sub.features.is_outline.sum() == 1
    assert sub.summary["n_fields"] == 2 and sub.summary["not_imported"] == 4
    assert any("NOT imported by user selection" in i and "F01" in i for i in sub.dataset_issues)
    assert sub.summary["total_usable_area_ha"] == pytest.approx(20.4, abs=0.1)


def test_subset_drops_rejected_features_but_keeps_selected_valid_ones(conn):
    vr = validate_geojson(samples.problem_fixture())
    ok = vr.features[(~vr.features.is_outline) & (vr.features.status != "rejected")].field_id.tolist()
    sub = subset_validation(vr, ok[:2])
    assert (sub.features.status != "rejected").all() and len(sub.features) == 2
    ids = db.save_import(conn, sub, "p")[0]
    assert len(db.get_fields(conn, ids["version_pk"])) == 2


def test_text_frame_makes_numeric_geojson_attributes_editable_as_text():
    df = pd.DataFrame({"field_id": ["A", "B"], "ratoon_number": [2.0, float("nan")], "planting_date": [pd.Timestamp("2026-03-01"), None],
                       "variety": ["N41 ", None]})
    t = A.text_frame(df, ["field_id", "ratoon_number", "planting_date", "variety"])
    assert list(t.ratoon_number) == ["2", None] and list(t.planting_date) == ["2026-03-01", None] and list(t.variety) == ["N41", None]
    assert A.changed_rows(t, t) == {}                                           # no phantom edits from float/str differences
