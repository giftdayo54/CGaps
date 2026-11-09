"""Raster export: classified GeoTIFF (derived from the stored, reviewed gap polygons) and imagery-layer GeoTIFFs."""
from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.shutil import copy as rio_copy

from .areas import utm_crs_for
from .providers.base import make_grid

CLASS_CODES = {0: "no candidate gap detected (field assessed)", 1: "candidate gap (model prediction, unverified)",
               2: "candidate gap - possible waterlogging", 3: "field not assessed (insufficient evidence / harvested / planting / unable)",
               255: "outside analysed fields (nodata)"}
CLASS_COLORS = {0: (200, 230, 201, 255), 1: (213, 0, 0, 255), 2: (21, 101, 192, 255), 3: (189, 189, 189, 255), 255: (0, 0, 0, 0)}


def _write_cog(arr: np.ndarray, transform, crs, nodata, tags: dict, descriptions=None, colormap=None) -> bytes:
    """Write a (bands,h,w) array as a Cloud-Optimized GeoTIFF (falls back to tiled GTiff + overviews)."""
    count, h, w = arr.shape
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "src.tif", Path(d) / "out.tif"
        with rasterio.open(src, "w", driver="GTiff", height=h, width=w, count=count, dtype=arr.dtype, crs=crs,
                           transform=transform, nodata=nodata, tiled=True, blockxsize=256, blockysize=256,
                           compress="deflate") as ds:
            ds.write(arr)
            ds.update_tags(**tags)
            if descriptions:
                for i, name in enumerate(descriptions, 1):
                    ds.set_band_description(i, name)
            if colormap:
                ds.write_colormap(1, colormap)
        try:
            rio_copy(src, dst, driver="COG", compress="DEFLATE")
            return dst.read_bytes()
        except Exception:  # noqa: BLE001  (GDAL without COG driver)
            with rasterio.open(src, "r+") as ds:
                ds.build_overviews([2, 4, 8], rasterio.enums.Resampling.nearest)
            return src.read_bytes()


def gaps_to_geotiff(fields: gpd.GeoDataFrame, results, gaps: gpd.GeoDataFrame, res: float = 10.0,
                    field_ids: list[str] | None = None, tags: dict | None = None) -> bytes:
    """Classified raster rasterised from the polygons currently stored (so review edits are included).

    Codes: 0 no gap, 1 candidate gap, 2 possible waterlogging, 3 field not assessed, 255 nodata. NOTE: this is a
    polygon-derived raster at `res` metres, not the model's internal pixel output."""
    f = fields.dropna(subset=["geometry"])
    f = f[f.field_id.isin(results.field_id)]
    if field_ids:
        f = f[f.field_id.isin(field_ids)]
    if f.empty:
        raise ValueError("No fields to export.")
    utm = utm_crs_for(f.geometry.union_all() if hasattr(f.geometry, "union_all") else f.geometry.unary_union)
    fu = f.to_crs(utm)
    status = dict(zip(results.field_id, results.status))
    transform, h, w = make_grid(tuple(fu.total_bounds), res)
    arr = np.full((h, w), 255, dtype="uint8")
    for row in fu.itertuples():
        code = 0 if status.get(row.field_id) == "analysed" else 3
        m = features.geometry_mask([row.geometry], (h, w), transform, invert=True)
        arr[m] = code
    if len(gaps):
        g = gaps[gaps.field_id.isin(set(f.field_id))].to_crs(utm)
        for cat, code in (("Candidate gap", 1), ("Candidate gap - possible waterlogging", 2)):
            sel = g[g.category == cat]
            if len(sel):
                m = features.rasterize([(x, 1) for x in sel.geometry], (h, w), transform=transform, fill=0, dtype="uint8").astype(bool)
                arr[m & (arr != 255)] = code
    t = {"DESCRIPTION": "CaneGap classified raster (derived from gap polygons)", "UNVALIDATED": "screening output; candidate gaps need field verification",
         "RESOLUTION_M": str(res), **{f"CLASS_{k}": v for k, v in CLASS_CODES.items()}, **(tags or {})}
    return _write_cog(arr[None], transform, utm, 255, t, ["class"], {k: v for k, v in CLASS_COLORS.items()})


def composite_to_geotiff(comp, field_geom_wgs, layers: list[str], tags: dict | None = None) -> bytes:
    """Float32 GeoTIFF of selected bands/indices for one field (pixels outside the field = NaN)."""
    from .areas import to_crs_geom
    from .pipeline import indices
    g = to_crs_geom(field_geom_wgs, "EPSG:4326", comp.crs)
    h, w = next(iter(comp.bands.values())).shape
    inside = features.geometry_mask([g], (h, w), comp.transform, invert=True)
    ix = indices(comp.bands)
    pool = {**comp.bands, **ix}
    miss = [x for x in layers if x not in pool]
    if miss:
        raise ValueError(f"Unknown layer(s): {miss}")
    arr = np.stack([np.where(inside, pool[x], np.nan).astype("float32") for x in layers])
    t = {"SYNTHETIC": "SYNTHETIC DEMO DATA - NOT SATELLITE IMAGERY" if comp.synthetic else "no", **(tags or {})}
    return _write_cog(arr, comp.transform, comp.crs, float("nan"), t, layers)
