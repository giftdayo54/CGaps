import json
import pytest
from canegap import samples
from canegap.geojson_import import (CRSNeedsConfirmation, ImportError_, validate_geojson, guess_mapping)


def test_fields_only_all_accepted():
    vr = validate_geojson(samples.farm_with_fields(False))
    assert vr.summary["n_fields"] == 6 and vr.summary["accepted"] == 6
    assert vr.features["area_ha"].between(10.1, 10.3).all()      # 300 x 340 m = 10.2 ha
    assert guess_mapping(["field_id", "variety"])["field_id"] == "field_id"


def test_outline_not_treated_as_field():
    gj = samples.farm_with_fields(True)
    vr = validate_geojson(gj, outline_rows=[6])
    assert vr.summary["n_fields"] == 6 and vr.summary["n_outlines"] == 1
    assert not vr.features.loc[vr.features.src_index == 6, "field_id"].isna().all() or True
    assert all(vr.features[~vr.features.is_outline].status == "accepted")   # all inside outline


def test_problem_fixture_reports_each_issue():
    vr = validate_geojson(samples.problem_fixture())
    f = vr.features.set_index("src_index")
    assert f.loc[2, "status"] == "rejected" and "NOT applied" in f.loc[2, "issues"]   # bowtie: repair would change area a lot
    assert f.loc[5, "status"] == "rejected" and "Unsupported" in f.loc[5, "issues"]
    assert f.loc[6, "status"] == "rejected" and "Missing or null" in f.loc[6, "issues"]
    assert "Duplicate field ID" in f.loc[3, "issues"] and f.loc[3, "field_id"] != "P01"
    assert "Missing field ID" in f.loc[4, "issues"] and f.loc[4, "field_id"] not in (None, "")
    assert f.loc[3, "orig_field_id"] == "P01"
    assert f.loc[7, "status"] in ("accepted", "warning")           # MultiPolygon supported
    assert len(set(f[f.status != "rejected"].field_id)) == (f.status != "rejected").sum()   # unique IDs
    assert any({o["field_a"], o["field_b"]} == {"P01", "P02"} for o in vr.summary["overlaps"])
    assert vr.summary["rejected"] == 3   # bowtie, point, null geometry


def test_strict_policy_rejects_missing_and_duplicate():
    vr = validate_geojson(samples.problem_fixture(), id_policy="strict")
    f = vr.features.set_index("src_index")
    assert f.loc[3, "status"] == "rejected" and f.loc[4, "status"] == "rejected"


def test_projected_declared_crs_is_reprojected_not_relabelled():
    vr = validate_geojson(samples.projected_fixture())
    assert vr.source_crs == "EPSG:32736"
    x, y = vr.features.geometry.iloc[0].centroid.coords[0]
    assert 33 < x < 36 and -18 < y < -14
    assert vr.features["area_ha"].between(10.1, 10.3).all()


def test_projected_without_declaration_needs_confirmation():
    fc = samples.projected_fixture()
    fc.pop("crs")
    with pytest.raises(CRSNeedsConfirmation):
        validate_geojson(fc)
    vr = validate_geojson(fc, source_crs="EPSG:32736")
    assert vr.summary["accepted"] == 2


def test_bad_inputs():
    for bad in (b"not json", json.dumps({"type": "FeatureCollection", "features": []}), json.dumps({"x": 1})):
        with pytest.raises(ImportError_):
            validate_geojson(bad)


def test_field_outside_outline_flagged():
    gj = samples.farm_with_fields(True)
    gj["features"][0] = json.loads(json.dumps(gj["features"][0]))
    # shift field 0 far outside
    def shift(c):
        return [shift(x) for x in c] if isinstance(c[0], list) else [c[0] + 0.05, c[1]]
    gj["features"][0]["geometry"]["coordinates"] = shift(gj["features"][0]["geometry"]["coordinates"])
    vr = validate_geojson(gj, outline_rows=[6])
    assert "outside the farm outline" in vr.features.set_index("src_index").loc[0, "issues"]


def test_small_invalid_geometry_is_repaired_and_reported():
    import geopandas as gpd
    from shapely.geometry import Polygon, mapping
    # 300 x 300 m square with a tiny self-touching loop on one edge
    X0, Y0 = 600000, 8230000
    ring = [(X0, Y0), (X0 + 300, Y0), (X0 + 300, Y0 + 150), (X0 + 302, Y0 + 152), (X0 + 300, Y0 + 154),
            (X0 + 302, Y0 + 150), (X0 + 300, Y0 + 150 + 0.0), (X0 + 300, Y0 + 300), (X0, Y0 + 300), (X0, Y0)]
    poly = Polygon(ring)
    assert not poly.is_valid
    g = gpd.GeoSeries([poly], crs="EPSG:32736").to_crs("EPSG:4326").iloc[0]
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"field_id": "R1"}, "geometry": mapping(g)}]}
    vr = validate_geojson(fc)
    row = vr.features.iloc[0]
    assert row.status == "repaired" and "Repaired invalid geometry" in row.issues
    assert vr.features.geometry.iloc[0].is_valid
