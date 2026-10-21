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
