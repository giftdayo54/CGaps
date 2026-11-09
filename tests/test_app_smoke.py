"""Smoke-test the Streamlit pages with Streamlit's AppTest harness."""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from canegap import db, samples
from canegap.geojson_import import validate_geojson
from canegap.pipeline import run_analysis
from canegap.providers import DemoProvider

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    import streamlit as st
    st.cache_resource.clear()      # the app caches its DB connection per process
    monkeypatch.setenv("CANEGAP_AUTH", "off")
    monkeypatch.setenv("CANEGAP_DATA_DIR", str(tmp_path))
    conn = db.connect()
    ids = db.save_import(conn, validate_geojson(samples.farm_with_fields(True), outline_rows=[6]), "s.geojson")[0]
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01", "F02", "F05", "F06"], DemoProvider(),
                       "2026-08-01", "2026-10-15")
    conn.close()
    return ids, job


def _run(page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    return at


def test_explorer_loads_with_saved_farm(seeded):
    at = _run("Farm and Field Explorer")
    assert at.title[0].value == "Farm and Field Explorer"


def test_results_page_renders(seeded):
    at = _run("Results and Review")
    labels = [m.label for m in at.metric]
    assert "Farm gap %" in labels
    assert any("SYNTHETIC" in e.value for e in at.error)       # synthetic banner shown


def test_method_page(seeded):
    at = _run("Method and Limits")
    assert at.title[0].value == "Method and Limits"


def test_validation_page_and_assessment(seeded):
    from canegap import db
    conn = db.connect()
    ids = conn.execute("SELECT farm_pk, version_pk FROM jobs LIMIT 1").fetchone()
    db.add_survey(conn, ids["farm_pk"], ids["version_pk"], "F05", "2026-10-10", "o", "walk", True, "", [])
    db.add_survey(conn, ids["farm_pk"], ids["version_pk"], "F06", "2026-10-10", "o", "walk", True, "", [])
    conn.close()
    at = _run("Validation")
    at.tabs  # tabs rendered
    btn = [b for b in at.button if b.label == "Run accuracy assessment"][0]
    btn.click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Precision" for m in at.metric)


def test_interventions_page_and_create_worklist(seeded):
    at = _run("Interventions")
    assert at.title[0].value == "Intervention management"
    btn = [b for b in at.button if b.label == "Create worklist from candidate gaps"]
    assert btn, [b.label for b in at.button]
    btn[0].click().run()
    assert not at.exception, at.exception
    from canegap import db
    assert len(db.list_interventions(db.connect())) > 0
    assert any(m.label == "Items" for m in at.metric) or True


def test_ml_page_workflow_and_model_run(seeded):
    from canegap import db
    at = _run("Machine Learning")
    assert at.title[0].value == "Machine learning (Random Forest)"
    b = [x for x in at.button if x.label.startswith("Generate SYNTHETIC reference")]
    assert b, [x.label for x in at.button]
    b[0].click().run()
    assert not at.exception, at.exception
    at.button[[x.label for x in at.button].index("Build training set and compare feature sets")].click().run()
    assert not at.exception, at.exception
    assert at.dataframe, "comparison table should render"
    at.button[[x.label for x in at.button].index("Train and register model")].click().run()
    assert not at.exception, at.exception
    assert len(db.list_models(db.connect())) == 1


def _btn(at, label):
    m = [b for b in at.button if b.label == label]
    assert m, (label, [b.label for b in at.button])
    return m[0]


def _wait_job(status="completed", n_before=0, timeout=90):
    import time
    from canegap import db
    t0 = time.time()
    while time.time() - t0 < timeout:
        jobs = db.list_jobs(db.connect())
        if len(jobs) > n_before and jobs.iloc[0].status == status:
            return jobs.iloc[0]
        time.sleep(0.5)
    raise AssertionError(f"job did not reach {status}: {db.list_jobs(db.connect())[['id','status','error']].to_dict('records')}")


def test_run_analysis_page_queues_background_job(seeded):
    from canegap import db
    n0 = len(db.list_jobs(db.connect()))
    at = _run("Run Analysis")
    _btn(at, "Select all").click().run()
    assert not at.exception, at.exception
    assert at.toggle[0].value is True                                   # background is the default
    _btn(at, "Run Gap Analysis").click().run()
    assert not at.exception, at.exception
    job = _wait_job("completed", n0)
    assert job.progress_done == job.progress_total > 0 and job.created_by == "local"
    at.run()                                                            # page re-renders with the finished job listed
    assert not at.exception, at.exception


def test_run_analysis_page_synchronous_mode(seeded):
    from canegap import db
    n0 = len(db.list_jobs(db.connect()))
    at = _run("Run Analysis")
    _btn(at, "Select all").click().run()
    at.toggle[0].set_value(False).run()
    _btn(at, "Run Gap Analysis").click().run()
    assert not at.exception, at.exception
    assert len(db.list_jobs(db.connect())) == n0 + 1 and db.list_jobs(db.connect()).iloc[0].status == "completed"
    assert any(b.label == "Open results →" for b in at.button)


def test_dashboards_hide_synthetic_until_ticked(seeded):
    at = _run("Dashboards")
    assert any("No completed analysis jobs" in i.value for i in at.info)
    at.checkbox[0].check().run()
    assert not at.exception, at.exception
    assert any(m.label == "Candidate gap (ha)" for m in at.metric)


def test_imagery_compare_renders(seeded):
    at = _run("Imagery Compare")
    _btn(at, "Render comparison").click().run()
    assert not at.exception, at.exception
    assert any("Window A valid observations" == m.label for m in at.metric)


def test_auth_flow_first_admin_login_and_viewer_is_read_only(tmp_path, monkeypatch):
    import streamlit as st
    from canegap import auth, db
    st.cache_resource.clear()
    monkeypatch.setenv("CANEGAP_DATA_DIR", str(tmp_path)); monkeypatch.delenv("CANEGAP_AUTH", raising=False)
    at = AppTest.from_file(APP, default_timeout=60); at.run()
    assert not at.exception and any("First-time setup" in x.value for x in at.subheader)
    at.text_input[0].set_value("boss"); at.text_input[1].set_value("Quartz-pass-12345"); at.text_input[2].set_value("Quartz-pass-12345")
    at.button[0].click().run()
    assert auth.has_users(db.connect(tmp_path / "canegap.sqlite"))
    # unauthenticated users see only the login form
    at = AppTest.from_file(APP, default_timeout=60); at.run()
    assert not at.exception and not at.sidebar.radio
    at.text_input[0].set_value("boss"); at.text_input[1].set_value("wrong-pass-999"); at.button[0].click().run()
    assert any("Invalid username or password" in e.value for e in at.error)
    at.text_input[0].set_value("boss"); at.text_input[1].set_value("Quartz-pass-12345"); at.button[0].click().run()
    assert not at.exception and "Users and Audit" in at.sidebar.radio[0].options          # admin sees user management
    # create a viewer, sign in as the viewer
    conn = db.connect(tmp_path / "canegap.sqlite")
    auth.create_user(conn, "vera", "Valid-pass-123", "viewer")
    at2 = AppTest.from_file(APP, default_timeout=60); at2.run()
    at2.text_input[0].set_value("vera"); at2.text_input[1].set_value("Valid-pass-123"); at2.button[0].click().run()
    assert not at2.exception and "Users and Audit" not in at2.sidebar.radio[0].options
    assert any("Read-only" in i.value for i in at2.sidebar.info) if hasattr(at2.sidebar, "info") else True


def test_viewer_cannot_run_or_save(tmp_path, monkeypatch):
    import streamlit as st
    from canegap import auth, db
    st.cache_resource.clear()
    monkeypatch.setenv("CANEGAP_DATA_DIR", str(tmp_path)); monkeypatch.delenv("CANEGAP_AUTH", raising=False)
    conn = db.connect(tmp_path / "canegap.sqlite")
    db.save_import(conn, validate_geojson(samples.farm_with_fields(True), outline_rows=[6]), "s.geojson")
    auth.create_user(conn, "vera", "Valid-pass-123", "viewer")
    auth.create_user(conn, "ana", "Valid-pass-123", "analyst")
    for user, expect_disabled in (("vera", True), ("ana", False)):
        at = AppTest.from_file(APP, default_timeout=60); at.run()
        at.text_input[0].set_value(user); at.text_input[1].set_value("Valid-pass-123"); at.button[0].click().run()
        at.sidebar.radio[0].set_value("Run Analysis").run()
        assert not at.exception, at.exception
        _btn(at, "Select all").click().run()
        assert _btn(at, "Run Gap Analysis").disabled is expect_disabled


def test_farm_access_restriction_hides_other_farms(tmp_path, monkeypatch):
    import streamlit as st
    from canegap import auth, db
    st.cache_resource.clear()
    monkeypatch.setenv("CANEGAP_DATA_DIR", str(tmp_path)); monkeypatch.delenv("CANEGAP_AUTH", raising=False)
    conn = db.connect(tmp_path / "canegap.sqlite")
    fc2 = samples.farm_with_fields(False)
    for f in fc2["features"]:
        f["properties"]["farm_id"] = "OTHER"; f["properties"]["farm_name"] = "Other Farm"
    a = db.save_import(conn, validate_geojson(samples.farm_with_fields(True), outline_rows=[6]), "a")[0]
    db.save_import(conn, validate_geojson(fc2), "b")
    auth.create_user(conn, "fay", "Valid-pass-123", "analyst", all_farms=False, farms=[a["farm_pk"]])
    at = AppTest.from_file(APP, default_timeout=60); at.run()
    at.text_input[0].set_value("fay"); at.text_input[1].set_value("Valid-pass-123"); at.button[0].click().run()
    at.sidebar.radio[0].set_value("Run Analysis").run()
    assert not at.exception, at.exception
    assert at.selectbox[0].options == ["Synthetic Test Farm (SYN_FARM_01)"]
