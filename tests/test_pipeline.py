import pandas as pd
import pytest
from shapely.ops import unary_union

from canegap import db, samples
from canegap.aggregate import farm_summary
from canegap.geojson_import import validate_geojson
from canegap.pipeline import Params, analyse_field, apply_overlap_policy, run_analysis
from canegap.providers import DemoProvider, Provider


def _import(conn, gj=None, **kw):
    vr = validate_geojson(gj or samples.farm_with_fields(True), outline_rows=[6], **kw)
    return db.save_import(conn, vr, "test.geojson")[0]


def test_end_to_end_independent_fields_and_clipping(conn):
    ids = _import(conn)
    fields = db.get_fields(conn, ids["version_pk"])
    assert len(fields) == 6                                          # outline not a field
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], list(fields.field_id), DemoProvider(),
                       "2026-08-01", "2026-10-15")
    res = db.job_results(conn, job)
    assert set(res.field_id) == set(fields.field_id)
    status = dict(zip(res.field_id, res.status))
    assert status["F03"] == "context_excluded"                      # recently harvested
    assert status["F04"] == "context_excluded"                      # newly planted
    assert status["F01"] == "analysed"
    gaps = db.job_gaps(conn, job)
    assert len(gaps) > 0 and gaps.farm_id.eq("SYN_FARM_01").all()
    fmap = fields.set_index("field_id").geometry
    for _, g in gaps.iterrows():                                     # every gap inside its own field only
        assert fmap[g.field_id].buffer(1e-7).contains(g.geometry)
        assert g.version_pk == ids["version_pk"]
        for other, geom in fmap.items():
            if other != g.field_id:
                assert g.geometry.intersection(geom).area < 1e-12
    # field totals equal the sum of their gap polygons
    for _, r in res[res.status == "analysed"].iterrows():
        assert gaps[gaps.field_id == r.field_id].area_ha.sum() == pytest.approx(r.gap_area_ha, rel=1e-6)
        assert r.gap_pct == pytest.approx(100 * r.gap_area_ha / r.analysed_area_ha, rel=1e-6)


def test_farm_aggregation_is_area_weighted_not_mean_of_pcts():
    df = pd.DataFrame({"status": ["analysed", "analysed", "insufficient_evidence"],
                       "analysed_area_ha": [10.0, 90.0, 0.0], "gap_area_ha": [5.0, 1.0, 0.0],
                       "n_gaps": [3, 1, 0]})
    s = farm_summary(df)
    assert s["candidate_gap_area_ha"] == 6.0 and s["analysed_area_ha"] == 100.0
    assert s["gap_pct"] == 6.0                                       # not (50% + 1.11%)/2
    assert s["n_insufficient_evidence"] == 1


def test_cloudy_field_is_insufficient_not_zero_gaps():
    gj = samples.farm_with_fields(False)
    vr = validate_geojson(gj)
    f = vr.features.iloc[0]
    r = analyse_field("F01", f.geometry, {"crop": "Sugarcane"}, DemoProvider(cloud_fraction=0.85), "2026-08-01", "2026-10-15")
    assert r.status == "insufficient_evidence" and r.gaps == [] and "valid observations" in r.reason


class _Failing(Provider):
    name = "failing"
    def get_composite(self, *a, **k):
        raise RuntimeError("HTTP 503")


def test_missing_imagery_reports_unable():
    vr = validate_geojson(samples.farm_with_fields(False))
    r = analyse_field("F01", vr.features.iloc[0].geometry, {}, _Failing(), "2026-08-01", "2026-10-15")
    assert r.status == "unable" and "HTTP 503" in r.reason


def test_overlap_policies_prevent_double_counting():
    vr = validate_geojson(samples.problem_fixture())
    f = vr.features[vr.features.field_id.isin(["P01", "P02"])]
    from canegap.areas import area_ha
    raw = sum(area_ha(g) for g in f.geometry)
    flag, ov = apply_overlap_policy(f, "flag_only")
    first, _ = apply_overlap_policy(f, "first_wins")
    excl, _ = apply_overlap_policy(f, "exclude")
    assert len(ov) == 1
    assert sum(area_ha(g) for g in flag.values()) == pytest.approx(raw, rel=1e-6)
    union_ha = area_ha(unary_union(list(f.geometry)))
    assert sum(area_ha(g) for g in first.values()) == pytest.approx(union_ha, rel=1e-3)
    assert sum(area_ha(g) for g in excl.values()) < union_ha


def test_boundary_versions_preserved_and_results_linked(conn):
    ids = _import(conn)
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01"], DemoProvider(), "2026-08-01", "2026-10-15")
    v2 = db.save_import(conn, validate_geojson(samples.farm_with_fields(True), outline_rows=[6]), "v2.geojson",
                        target_farm_pk=ids["farm_pk"])[0]
    assert v2["version_no"] == 2 and db.current_version(conn, ids["farm_pk"]) == v2["version_pk"]
    j = conn.execute("SELECT version_pk FROM jobs WHERE id=?", (job,)).fetchone()
    assert j["version_pk"] == ids["version_pk"]                      # old result still tied to v1
    assert len(db.get_fields(conn, ids["version_pk"])) == 6          # v1 geometry intact
    d = db.boundary_diff(conn, ids["version_pk"], v2["version_pk"])
    assert (d.change == "unchanged").all()


def test_library_search_and_rename(conn):
    _import(conn)
    assert len(db.list_farms(conn, "synthetic")) == 1 and len(db.list_farms(conn, "zzz")) == 0
    farm = db.list_farms(conn).iloc[0]
    db.rename_farm(conn, int(farm.farm_pk), "Renamed")
    assert db.list_farms(conn).iloc[0].farm_name == "Renamed"
    assert int(db.list_farms(conn).iloc[0].n_fields) == 6


def test_exports_carry_farm_and_field_ids(conn):
    from canegap import exports
    import geopandas as gpd, io, zipfile, json
    ids = _import(conn)
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01", "F02"], DemoProvider(), "2026-08-01", "2026-10-15")
    gaps = exports.gaps_with_flag(db.job_gaps(conn, job), True)
    gj = json.loads(exports.to_geojson(gaps))
    assert gj["features"] and all(f["properties"]["farm_id"] and f["properties"]["field_id"] for f in gj["features"])
    assert "SYNTHETIC" in gj["features"][0]["properties"]["synthetic"]
    z = zipfile.ZipFile(io.BytesIO(exports.to_shapefile_zip(gaps)))
    assert any(n.endswith(".prj") for n in z.namelist())
    assert exports.to_gpkg(gaps)[:6] == b"SQLite"


def test_excel_export_has_expected_sheets(conn):
    from canegap import exports
    import io, openpyxl
    ids = _import(conn)
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01"], DemoProvider(), "2026-08-01", "2026-10-15")
    res, gaps = db.job_results(conn, job), exports.gaps_with_flag(db.job_gaps(conn, job), True)
    wb = openpyxl.load_workbook(io.BytesIO(exports.to_xlsx(res, gaps, farm_summary(res), ["n"])))
    assert wb.sheetnames == ["README", "Farm summary", "Fields", "Gaps"]


def test_saved_boundary_with_rejected_features_loads(conn):
    """Regression: rejected features have no geometry; loading them back must not crash."""
    vr = validate_geojson(samples.problem_fixture())
    ids = db.save_import(conn, vr, "problem.geojson")[0]
    f = db.get_fields(conn, ids["version_pk"])
    assert f.geometry.isna().sum() >= 2 and (f.status == "rejected").sum() >= 2
    assert len(db.list_farms(conn)) == 1
