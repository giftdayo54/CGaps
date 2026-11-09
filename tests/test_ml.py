import numpy as np
import pandas as pd
import pytest

from canegap import db, ml, samples
from canegap.geojson_import import validate_geojson
from canegap.pipeline import Params, run_analysis
from canegap.providers import DemoProvider, make_grid
from canegap.providers.base import Composite

PROV = DemoProvider(gap_fraction=0.12)
ALL = ["F01", "F02", "F03", "F04", "F05", "F06"]


def _farm(conn, survey_fields=ALL, prov=PROV):
    vr = validate_geojson(samples.farm_with_fields(True), outline_rows=[6])
    ids = db.save_import(conn, vr, "t")[0]
    fields = db.get_fields(conn, ids["version_pk"]).set_index("field_id")
    for fid in survey_fields:
        truth = prov.truth_polygons(fields.at[fid, "geometry"], fid)
        db.add_survey(conn, ids["farm_pk"], ids["version_pk"], fid, "2026-10-10", "SYNTHETIC", "SYNTHETIC truth", True, "", truth)
    return ids


def _train(conn, ids, use_change=False, **kw):
    return ml.build_training_set(conn, PROV, ids["version_pk"], "2026-08-01", "2026-10-15", use_s1_change=use_change, **kw)


def test_baseline_window_is_equal_length_and_before():
    a, b = ml.baseline_window("2026-08-01", "2026-08-31")
    assert b == "2026-07-31" and a == "2026-07-01"


def test_nanmean_filter_works_in_linear_domain_and_keeps_nan():
    a = np.array([[1.0, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 100.0]], dtype="float32")
    out = ml._nanmean_filter(a, 3)
    assert np.isnan(out[1, 1]) and out[0, 0] == pytest.approx(1.0)
    assert out[2, 2] == pytest.approx((1 + 1 + 100) / 3, rel=1e-4)


def test_feature_stack_and_matrix():
    h = w = 30
    t, hh, ww = make_grid((0, 0, 280, 280), 10)
    s2 = PROV.get_composite((0, 0, 280, 280), "EPSG:32736", "a", "b", seed_key="x")
    s1 = PROV.get_s1_composite((0, 0, 280, 280), "EPSG:32736", "a", "b", seed_key="x")
    s1b = PROV.get_s1_composite((0, 0, 280, 280), "EPSG:32736", "c", "d", seed_key="x")
    st = ml.feature_stack(s2, s1, s1b)
    names = ml.feature_names("s1+s2", True)
    X, ok = ml.to_matrix(st, names)
    assert X.shape[-1] == len(names) and ok.mean() > 0.9
    assert -30 < np.nanmedian(st["VH_db"]) < -10 and np.nanmedian(st["VH_db"]) < np.nanmedian(st["VV_db"])
    with pytest.raises(ValueError):
        ml.to_matrix(ml.feature_stack(s2), ["VH_db"])
    with pytest.raises(ValueError):
        ml.feature_names("radar")


def test_training_set_weights_restore_population_and_ignore_band(conn):
    ids = _farm(conn)
    df, summ = _train(conn, ids, max_per_class=200)
    assert summ.used.all() and set(df.field_id) == set(ALL) and {"label", "w", "baseline_gap", "farm_id"} <= set(df.columns)
    for fid, g in df.groupby("field_id"):
        row = summ[summ.field_id == fid].iloc[0]
        assert g[g.label == 1].w.sum() == pytest.approx(row.pos_px, rel=1e-9)
        assert g[g.label == 0].w.sum() == pytest.approx(row.neg_px, rel=1e-9)
        assert len(g[g.label == 0]) <= 200
    _, s0 = _train(conn, ids, ignore_band_m=0.0)
    _, s20 = _train(conn, ids, ignore_band_m=20.0)
    assert (s0.valid_px.values > s20.valid_px.values).all()          # wider band drops more ambiguous pixels


def test_unsurveyed_and_partial_fields_are_not_used_for_training(conn):
    ids = _farm(conn, survey_fields=["F01", "F02"])
    db.add_survey(conn, ids["farm_pk"], ids["version_pk"], "F05", "2026-10-10", "o", "walk", False, "", [])
    df, summ = _train(conn, ids)
    assert set(df.field_id) == {"F01", "F02"} and set(summ.field_id) == {"F01", "F02"}


def test_cv_is_field_grouped_and_reports_baseline(conn):
    ids = _farm(conn)
    df, _ = _train(conn, ids, max_per_class=400)
    cols = ml.feature_names("s2")
    cv = ml.cross_validate(df, cols, ml.default_rf_params(n_estimators=60, n_jobs=1), n_splits=3)
    tests = [set(f["test_groups"]) for f in cv["folds"]]
    assert set().union(*tests) == set(ALL) and sum(len(t) for t in tests) == len(ALL)   # each field held out exactly once
    assert cv["rf"]["f1"] > 0.6 and "baseline" in cv and 0 <= cv["rf"]["auc"] <= 1
    tbl = ml.compare_feature_sets(df, rf_params=ml.default_rf_params(n_estimators=40, n_jobs=1), n_splits=3)
    assert list(tbl.method)[:2] == ["RF s2", "Threshold baseline (S2)"] and "RF s1+s2" in set(tbl.method)
    t = tbl.set_index("method")
    assert t.at["RF s1", "f1"] < t.at["RF s2", "f1"]          # synthetic radar is deliberately weakly informative


def test_too_few_fields_refuses_cv_but_does_not_claim_validation(conn):
    ids = _farm(conn, survey_fields=["F01", "F02"])
    df, _ = _train(conn, ids)
    with pytest.raises(ml.InsufficientData):
        ml.cross_validate(df, ml.feature_names("s2"))
    b = ml.train_model(df, "s2", rf_params=ml.default_rf_params(n_estimators=30, n_jobs=1), synthetic=True)
    assert b.meta["cv"] is None and "Cross-validation needs" in b.meta["cv_note"]


def test_register_load_roundtrip_and_registry(conn, tmp_path):
    ids = _farm(conn)
    df, _ = _train(conn, ids, max_per_class=300)
    b = ml.train_model(df, "s1+s2", rf_params=ml.default_rf_params(n_estimators=30, n_jobs=1), n_splits=3, synthetic=True)
    mid = ml.register_model(conn, b, "synthetic test model", tmp_path)
    rows = db.list_models(conn)
    assert rows.iloc[0].id == mid and rows.iloc[0].synthetic == 1 and rows.iloc[0].cv_validated == 1
    b2 = ml.load_bundle(db.get_model_row(conn, mid)["path"])
    X = df[b.feature_names].to_numpy("float32")[:50]
    np.testing.assert_allclose(b.predict_proba(X), b2.predict_proba(X))
    assert b2.label.startswith(f"rf#{mid}") and b2.needs_s1 and b2.needs_s2
    db.delete_model(conn, mid)
    assert db.list_models(conn).empty


def test_end_to_end_job_with_model_and_guards(conn, tmp_path):
    ids = _farm(conn, survey_fields=["F01", "F02", "F05", "F06"])
    df, _ = _train(conn, ids, max_per_class=300)
    b = ml.train_model(df, "s1+s2", rf_params=ml.default_rf_params(n_estimators=40, n_jobs=1), n_splits=3, synthetic=True)
    ml.register_model(conn, b, "m", tmp_path)
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01", "F02", "F05"], PROV, "2026-08-01", "2026-10-15", model=b)
    res, gaps = db.job_results(conn, job), db.job_gaps(conn, job)
    assert (res.status == "analysed").all() and len(gaps) > 0
    assert gaps.method.str.startswith("rf#").all() and gaps.confidence.between(0, 1).all()
    j = conn.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
    assert j["algo_version"] == b.label and '"model"' in j["params_json"]
    fmap = db.get_fields(conn, ids["version_pk"]).set_index("field_id").geometry
    assert all(fmap[g.field_id].buffer(1e-7).contains(g.geometry) for g in gaps.itertuples())   # still clipped per field
    from canegap import reports
    assert reports.farm_report(conn, job)[:5] == b"%PDF-" and "Random Forest" in reports._method_note(
        conn.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone())
    # synthetic model refuses real-imagery source
    class _Real(DemoProvider):
        synthetic = False
    with pytest.raises(ValueError, match="mismatch"):
        run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01"], _Real(), "2026-08-01", "2026-10-15", model=b)

    class _NoS1(DemoProvider):
        def get_s1_composite(self, *a, **k):
            raise RuntimeError("no radar scenes")
    job2 = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01"], _NoS1(gap_fraction=0.12), "2026-08-01", "2026-10-15", model=b)
    r2 = db.job_results(conn, job2).iloc[0]
    assert r2.status == "unable" and "Sentinel-1 unavailable" in r2.reason           # no silent fallback


def test_s2_only_model_needs_no_radar(conn):
    ids = _farm(conn, survey_fields=["F01", "F02", "F05", "F06"])
    df, _ = _train(conn, ids, max_per_class=300)
    b = ml.train_model(df, "s2", rf_params=ml.default_rf_params(n_estimators=30, n_jobs=1), n_splits=3, synthetic=True)
    class _NoS1(DemoProvider):
        def get_s1_composite(self, *a, **k):
            raise RuntimeError("no radar")
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01"], _NoS1(gap_fraction=0.12), "2026-08-01", "2026-10-15", model=b)
    assert db.job_results(conn, job).iloc[0].status == "analysed"


def test_cdse_s1_payload_and_parsing(monkeypatch):
    import io
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from canegap.providers import CDSEProvider, CDSEUnavailable
    from canegap.providers import cdse

    bounds = (600000, 8230000, 600200, 8230200)
    _, h, w = make_grid(bounds, 10)
    arr = np.stack([np.full((h, w), 0.1), np.full((h, w), 0.02), np.full((h, w), 3)]).astype("float32")
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=h, width=w, count=3, dtype="float32", crs="EPSG:32736",
                     transform=from_origin(0, 0, 10, 10)) as ds:
            ds.write(arr)
        tiff = mf.read()
    seen = {}

    class R:
        def __init__(self, code, content=b"", text=""):
            self.status_code, self.content, self.text = code, content, text

    def fake_post(url, json=None, headers=None, **k):
        seen["payload"] = json
        return R(200, tiff) if seen.get("ok", True) else R(500, text="boom")

    monkeypatch.setattr(cdse.requests, "post", fake_post)
    p = CDSEProvider("id", "secret")
    p._token, p._token_exp = "tok", 1e12
    p.orbit_direction = "DESCENDING"
    c = p.get_s1_composite(bounds, "EPSG:32736", "2026-08-01", "2026-09-01")
    d = seen["payload"]["input"]["data"][0]
    assert d["type"] == "sentinel-1-grd" and d["dataFilter"]["orbitDirection"] == "DESCENDING"
    assert d["processing"]["backCoeff"] == "SIGMA0_ELLIPSOID" and d["processing"]["orthorectify"] is True
    assert c.bands["VV"].mean() == pytest.approx(0.1) and c.bands["VH"].mean() == pytest.approx(0.02) and c.obs_count.max() == 3
    seen["ok"] = False
    with pytest.raises(CDSEUnavailable, match="500"):
        p.get_s1_composite(bounds, "EPSG:32736", "2026-08-01", "2026-09-01")


def _cal_data(rng, n=4000, groups=5, overconfident=True):
    g = rng.integers(0, groups, n)
    true_p = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    # a miscalibrated score: pushed towards 0/1
    raw = np.clip(0.5 + (true_p - 0.5) * (1.9 if overconfident else 1.0), 0.001, 0.999)
    return raw, y, np.ones(n), g


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_reduces_error_with_crossfitting(method):
    p, y, w, g = _cal_data(np.random.default_rng(1))
    rep, cal = ml.calibration_report(p, y, w, g, method)
    assert rep["ece_calibrated_crossfit"] < rep["ece_raw"] and rep["brier_calibrated_crossfit"] <= rep["brier_raw"]
    out = cal.predict(p)
    assert out.min() >= 0 and out.max() <= 1
    assert abs(np.average(out, weights=w) - y.mean()) < 0.03          # mean probability matches prevalence


def test_calibration_needs_groups_and_both_classes():
    p, y, w, g = _cal_data(np.random.default_rng(2), groups=2)
    with pytest.raises(ml.InsufficientData):
        ml.calibration_report(p, y, w, g, "isotonic")
    with pytest.raises(ValueError):
        ml.fit_calibrator(p, y, w, "magic")


def test_train_model_with_calibration_and_roundtrip(conn, tmp_path):
    ids = _farm(conn)
    df, _ = _train(conn, ids, max_per_class=300)
    b = ml.train_model(df, "s2", rf_params=ml.default_rf_params(n_estimators=30, n_jobs=1), n_splits=3, synthetic=True,
                       calibration="isotonic")
    assert b.calibrator is not None and b.meta["calibration"]["method"] == "isotonic" and "_oof" not in b.meta["cv"]
    mid = ml.register_model(conn, b, "cal", tmp_path)
    assert db.list_models(conn).iloc[0].calibrated == 1
    b2 = ml.load_bundle(db.get_model_row(conn, mid)["path"])
    X = df[b.feature_names].to_numpy("float32")[:40]
    np.testing.assert_allclose(b.predict_proba(X), b2.predict_proba(X))


def test_calibration_skipped_when_cv_impossible(conn):
    ids = _farm(conn, survey_fields=["F01", "F02"])
    df, _ = _train(conn, ids)
    b = ml.train_model(df, "s2", rf_params=ml.default_rf_params(n_estimators=20, n_jobs=1), synthetic=True, calibration="sigmoid")
    assert b.calibrator is None and "Not calibrated" in b.meta["calibration_note"]


def test_cdse_s2_payload_requests_blue_band_and_parses_seven_layers(monkeypatch):
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from canegap.providers import CDSEProvider, cdse
    bounds = (600000, 8230000, 600200, 8230200)
    _, h, w = make_grid(bounds, 10)
    vals = [0.03, 0.07, 0.04, 0.12, 0.42, 0.17, 5]               # B02,B03,B04,B05,B08,B11,count
    arr = np.stack([np.full((h, w), v) for v in vals]).astype("float32")
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=h, width=w, count=7, dtype="float32", crs="EPSG:32736",
                     transform=from_origin(0, 0, 10, 10)) as ds:
            ds.write(arr)
        tiff = mf.read()
    seen = {}

    class R:
        status_code, content, text = 200, tiff, ""

    monkeypatch.setattr(cdse.requests, "post", lambda url, json=None, headers=None, **k: (seen.update(p=json), R())[1])
    p = CDSEProvider("i", "s"); p._token, p._token_exp = "t", 1e12
    c = p.get_composite(bounds, "EPSG:32736", "2026-08-01", "2026-09-01")
    assert '"B02"' in seen["p"]["evalscript"] and "bands: 7" in seen["p"]["evalscript"]
    assert set(c.bands) == {"B02", "B03", "B04", "B05", "B08", "B11"} and c.bands["B08"].mean() == pytest.approx(0.42)
    assert c.obs_count.max() == 5
