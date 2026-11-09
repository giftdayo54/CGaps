"""End-to-end UI tests: upload -> validate -> select fields + type attributes -> save -> run analysis."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from canegap import db, samples

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def at(tmp_path, monkeypatch):
    import streamlit as st
    st.cache_resource.clear()
    monkeypatch.setenv("CANEGAP_DATA_DIR", str(tmp_path)); monkeypatch.setenv("CANEGAP_AUTH", "off")
    a = AppTest.from_file(APP, default_timeout=90); a.run()
    assert not a.exception, a.exception
    return a


def _btn(at, label):
    m = [b for b in at.button if b.label == label]
    assert m, (label, [b.label for b in at.button])
    return m[0]


def _upload_and_validate(at):
    at.file_uploader[0].upload("farm.geojson", json.dumps(samples.farm_with_fields(True)).encode()).run()
    assert not at.exception, at.exception
    # mark feature 6 as the farm outline (Arrangement B)
    ms = [m for m in at.multiselect if m.label.startswith("...or pick features by index")]
    ms[0].set_value([6]).run()
    _btn(at, "Validate").click().run()
    assert not at.exception, at.exception
    assert any(s.value == "Select fields and add attributes" for s in at.subheader)


def test_selection_appears_right_after_upload_and_all_fields_start_selected(at):
    _upload_and_validate(at)
    assert at.session_state["imp_sel"] == {f"F0{i}" for i in range(1, 7)}
    assert any("6 of 6 usable fields selected" in m.value for m in at.markdown)
    assert any(b.label == "Select all valid" for b in at.button) and any(b.label == "Clear selection" for b in at.button)


def test_import_only_selected_fields_with_manual_variety_and_planting_date(at):
    _upload_and_validate(at)
    # the user keeps F02 and F05 and types attributes (equivalent to typing into the table cells)
    attrs = at.session_state["imp_attrs"].copy()
    attrs.loc[attrs.field_id == "F02", ["variety", "planting_date"]] = ["NCo376", "15/03/2026"]       # tolerant date format
    attrs.loc[attrs.field_id == "F05", ["variety", "ratoon_number"]] = ["R579", "2"]
    at.session_state["imp_attrs"] = attrs
    at.session_state["imp_sel"] = {"F02", "F05"}
    at.session_state["imp_ver"] += 1
    at.run()
    assert not at.exception, at.exception
    assert any("2 of 6 usable fields selected" in m.value for m in at.markdown)
    at.radio[[r.label for r in at.radio].index("What should be saved to the library?")].set_value("Import only the selected fields").run()
    _btn(at, "Save to boundary library").click().run()
    assert not at.exception, at.exception
    conn = db.connect(); farm = db.list_farms(conn).iloc[0]
    f = db.get_fields(conn, db.current_version(conn, int(farm.farm_pk))).set_index("field_id")
    assert sorted(f.index) == ["F02", "F05"]                                         # only the selection was imported
    assert f.at["F02", "variety"] == "NCo376" and f.at["F02", "planting_date"] == "2026-03-15"     # normalised to ISO
    assert f.at["F05", "variety"] == "R579" and f.at["F05", "ratoon_number"] == "2"
    hist = db.attribute_history(conn, db.current_version(conn, int(farm.farm_pk)))
    assert len(hist) == 4 and (hist.source == "manual (at import)").all()
    assert at.session_state["sel_ids"] == {"F02", "F05"}                             # carried to Run Analysis


def test_import_all_but_analyse_selected_and_go_to_run_analysis(at):
    _upload_and_validate(at)
    at.session_state["imp_sel"] = {"F01", "F06"}; at.session_state["imp_ver"] += 1
    at.run()
    _btn(at, "Save and run analysis on the selected fields →").click().run()
    assert not at.exception, at.exception
    conn = db.connect(); farm = db.list_farms(conn).iloc[0]
    assert int(farm.n_fields) == 6                                                   # everything imported
    assert at.session_state["sel_ids"] == {"F01", "F06"}
    assert at.session_state["nav"] == "Run Analysis"                                 # landed on the run page
    assert any("2 of 6 fields selected" in m.value for m in at.markdown)


def test_invalid_attribute_blocks_save_with_a_clear_message(at):
    _upload_and_validate(at)
    attrs = at.session_state["imp_attrs"].copy()
    attrs.loc[attrs.field_id == "F01", "planting_date"] = "not a date"
    at.session_state["imp_attrs"] = attrs; at.session_state["imp_ver"] += 1; at.run()
    _btn(at, "Save to boundary library").click().run()
    assert any("Fix these attribute values first" in e.value and "F01" in e.value for e in at.error)
    assert db.list_farms(db.connect()).empty                                          # nothing was saved


def test_run_page_lets_you_add_planting_date_and_it_changes_the_job(at):
    _upload_and_validate(at)
    attrs = at.session_state["imp_attrs"].copy()
    attrs.loc[:, ["planting_date", "last_harvest_date"]] = None
    at.session_state["imp_attrs"] = attrs; at.session_state["imp_ver"] += 1; at.run()
    _btn(at, "Save and run analysis on the selected fields →").click().run()
    assert any("no planting or last-harvest date" in w.value for w in at.warning)     # nudge to add dates
    # add a recent planting date to F01 through the same panel code path used by the 'Save table edits' button
    conn = db.connect(); farm = db.list_farms(conn).iloc[0]; vpk = db.current_version(conn, int(farm.farm_pk))
    db.update_field_attributes(conn, vpk, {"F01": {"planting_date": "2026-10-01", "variety": "N41"}}, "tester")
    at.run()
    _btn(at, "Select all").click().run()
    at.toggle[0].set_value(False).run()                                               # synchronous for the test
    _btn(at, "Run Gap Analysis").click().run()
    assert not at.exception, at.exception
    jobs = db.list_jobs(db.connect()); res = db.job_results(db.connect(), int(jobs.iloc[0].id)).set_index("field_id")
    assert res.at["F01", "status"] == "context_excluded" and "Newly planted" in res.at["F01", "reason"]
    assert json.loads(jobs.iloc[0].params_json)["field_attributes"]["F01"]["variety"] == "N41"
