import io

import numpy as np
import pytest
import rasterio

from canegap import db, imagery, rasters, samples
from canegap.areas import M2_PER_HA
from canegap.geojson_import import validate_geojson
from canegap.pipeline import run_analysis
from canegap.providers import DemoProvider

PROV = DemoProvider(gap_fraction=0.12)


def _job(conn):
    ids = db.save_import(conn, validate_geojson(samples.farm_with_fields(True), outline_rows=[6]), "t")[0]
    job = run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01", "F02", "F03", "F05"], PROV, "2026-08-01", "2026-10-15")
    return ids, job


def test_classified_geotiff_matches_polygons_and_statuses(conn):
    ids, job = _job(conn)
    fields = db.get_fields(conn, ids["version_pk"])
    res, gaps = db.job_results(conn, job), db.job_gaps(conn, job)
    data = rasters.gaps_to_geotiff(fields, res, gaps, tags={"JOB": str(job)})
    with rasterio.open(io.BytesIO(data)) as ds:
        arr = ds.read(1)
        assert ds.crs.is_projected and ds.nodata == 255 and ds.tags()["JOB"] == str(job) and "UNVALIDATED" in ds.tags()
        assert ds.colormap(1)[1][:3] == (213, 0, 0)
        px_ha = abs(ds.transform.a * ds.transform.e) / M2_PER_HA
    gap_ha_raster = (arr == 1).sum() * px_ha + (arr == 2).sum() * px_ha
    assert gap_ha_raster == pytest.approx(gaps.area_ha.sum(), rel=0.12)           # 10 m quantisation
    assert (arr == 3).sum() * px_ha == pytest.approx(10.2, rel=0.05)              # F03 recently harvested = 'not assessed'
    assert (arr == 0).any() and (arr == 255).any()
    # only chosen fields
    one = rasters.gaps_to_geotiff(fields, res, gaps, field_ids=["F02"])
    with rasterio.open(io.BytesIO(one)) as ds:
        a1 = ds.read(1)
    assert (a1 != 255).sum() < (arr != 255).sum() / 3
    with pytest.raises(ValueError):
        rasters.gaps_to_geotiff(fields, res, gaps, field_ids=["NOPE"])


def test_review_edits_flow_into_raster(conn):
    ids, job = _job(conn)
    fields = db.get_fields(conn, ids["version_pk"]); res = db.job_results(conn, job)
    before = db.job_gaps(conn, job)
    gid = int(before.sort_values("area_ha").iloc[-1].gap_id)
    db.delete_gap(conn, gid)
    after = db.job_gaps(conn, job)
    def n_gap(g):
        with rasterio.open(io.BytesIO(rasters.gaps_to_geotiff(fields, res, g))) as ds:
            a = ds.read(1)
        return int(((a == 1) | (a == 2)).sum())
    assert n_gap(after) < n_gap(before)


def test_composite_geotiff_layers():
    from canegap.areas import utm_crs_for
    vr = validate_geojson(samples.farm_with_fields(False))
    geom = vr.features.iloc[0].geometry
    utm = utm_crs_for(geom)
    from canegap.areas import to_crs_geom
    g = to_crs_geom(geom, "EPSG:4326", utm)
    comp = PROV.get_composite(g.bounds, str(utm), "2026-08-01", "2026-10-15", seed_key="F01")
    data = rasters.composite_to_geotiff(comp, geom, ["NDVI", "B08", "B02"])
    with rasterio.open(io.BytesIO(data)) as ds:
        assert ds.count == 3 and ds.descriptions == ("NDVI", "B08", "B02") and "SYNTHETIC" in ds.tags()["SYNTHETIC"]
        nd = ds.read(1)
    assert np.isnan(nd).any() and np.nanmedian(nd) > 0.6
    with pytest.raises(ValueError):
        rasters.composite_to_geotiff(comp, geom, ["BOGUS"])


@pytest.mark.parametrize("layer", imagery.LAYERS)
def test_every_layer_renders_and_swipe_html_is_self_contained(layer):
    vr = validate_geojson(samples.farm_with_fields(False))
    geom = vr.features.iloc[0].geometry
    pair = imagery.render_pair(PROV, geom, "F01", layer, ("2026-06-01", "2026-06-30"), ("2026-09-01", "2026-09-30"))
    a, b = pair["a"]["rgba"], pair["b"]["rgba"]
    assert a.shape == b.shape and a.shape[-1] == 4 and (a[..., 3] == 0).any() and (a[..., 3] == 255).any()
    assert pair["a"]["synthetic"] and pair["a"]["valid_fraction"] == pytest.approx(1.0)
    page, h = imagery.swipe_html(imagery.to_data_uri(a), imagery.to_data_uri(b), "Jun <b>", "Sep", a.shape[1] / a.shape[0])
    assert page.count("data:image/png;base64,") == 2 and "Jun &lt;b&gt;" in page and "http" not in page.replace("data:image", "")


def test_gap_overlay_draws_red_outlines():
    vr = validate_geojson(samples.farm_with_fields(False))
    geom = vr.features.iloc[0].geometry
    truth = PROV.truth_polygons(geom, "F01")
    pair = imagery.render_pair(PROV, geom, "F01", "NDVI", ("2026-06-01", "2026-06-30"), ("2026-09-01", "2026-09-30"), gaps_wgs=truth)
    red = lambda im: ((im[..., 0] == 213) & (im[..., 1] == 0)).sum()  # noqa: E731
    assert red(pair["b"]["rgba"]) > 0 and red(pair["a"]["rgba"]) == 0


def test_s1_layer_needs_provider_support():
    from canegap.providers import Provider
    class Only2(DemoProvider):
        def get_s1_composite(self, *a, **k):
            raise RuntimeError("no s1")
    geom = validate_geojson(samples.farm_with_fields(False)).features.iloc[0].geometry
    with pytest.raises(RuntimeError):
        imagery.render_pair(Only2(), geom, "F01", "Sentinel-1 VH (dB)", ("a", "b"), ("c", "d"))
