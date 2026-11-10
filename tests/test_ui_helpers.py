import pytest
from shapely.geometry import Point

from canegap import db, samples
from canegap.geojson_import import validate_geojson
from canegap.ui_helpers import field_at_point, toggle_selection


@pytest.fixture()
def fields(conn):
    ids = db.save_import(conn, validate_geojson(samples.farm_with_fields(True), outline_rows=[6]), "t")[0]
    return db.get_fields(conn, ids["version_pk"])


def test_click_resolves_to_the_field_under_the_point(fields):
    for r in fields.itertuples():
        c = r.geometry.centroid
        assert field_at_point(fields, c.y, c.x) == r.field_id
    assert field_at_point(fields, 10.0, 10.0) is None                      # click outside every field


def test_click_in_overlap_picks_smaller_polygon(conn):
    vr = validate_geojson(samples.problem_fixture())
    ids = db.save_import(conn, vr, "p")[0]
    f = db.get_fields(conn, ids["version_pk"])
    a, b = f[f.field_id == "P01"].iloc[0].geometry, f[f.field_id == "P02"].iloc[0].geometry
    pt = a.intersection(b).centroid
    assert field_at_point(f, pt.y, pt.x) in ("P01", "P02")


def test_toggle_selection_is_pure_and_respects_allowed():
    s = {"F01"}
    assert toggle_selection(s, "F02") == {"F01", "F02"} and s == {"F01"}
    assert toggle_selection({"F01", "F02"}, "F02") == {"F01"}
    assert toggle_selection({"F01"}, "F99", allowed={"F01", "F02"}) == {"F01"}      # rejected/unknown field ignored


def test_checked_ids_handles_odd_editor_dtypes():
    import numpy as np
    import pandas as pd
    from canegap.ui_helpers import checked_ids
    ids = ["A", "B", "C", "D", "E", "F"]
    for flags in ([True, False, True, False, False, False],
                  pd.array([True, False, True, None, pd.NA, False], dtype="boolean"),
                  np.array([1, 0, 1, 0, 0, 0], dtype=object),
                  ["True", "false", "yes", "0", "", "no"]):
        df = pd.DataFrame({"field_id": ids, "select": flags})
        assert checked_ids(df) == {"A", "C"}, flags
    assert checked_ids(pd.DataFrame({"field_id": [], "select": []})) == set()
    assert checked_ids(pd.DataFrame({"field_id": ["A"]})) == set()
    dup = pd.DataFrame([["A", True, False], ["B", False, True]], columns=["field_id", "select", "select"])   # duplicate name
    assert checked_ids(dup) == {"A"}
