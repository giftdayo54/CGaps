"""Synthetic imagery generator for running the app without satellite credentials.

OUTPUT IS NOT SATELLITE DATA. Everything produced here is flagged synthetic=True and the UI,
exports and reports carry that label. Synthetic Sentinel-1 backscatter is deliberately only
*weakly* informative so that S1-only models score worse than S2 ones; this is an arbitrary
design choice for testing, not a statement about real SAR behaviour.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import shape

from .base import Composite, Provider, make_grid


class DemoProvider(Provider):
    name = "demo-synthetic"
    synthetic = True

    def __init__(self, gap_fraction: float = 0.06, cloud_fraction: float = 0.0, seed: int = 42):
        self.gap_fraction = gap_fraction
        self.cloud_fraction = cloud_fraction
        self.seed = seed

    def _digest(self, seed_key):
        return int(hashlib.sha256(f"{self.seed}:{seed_key}".encode()).hexdigest()[:8], 16)

    def _cover(self, rng, h, w):
        vigour = gaussian_filter(rng.normal(size=(h, w)), sigma=6)
        vigour = vigour / (np.abs(vigour).max() + 1e-9) * 0.08
        blobs = gaussian_filter(rng.normal(size=(h, w)), sigma=3.5)
        gap = blobs > np.quantile(blobs, 1 - self.gap_fraction)
        cover = np.clip(0.95 + vigour, 0.6, 1.0)
        cover = np.where(gap, rng.uniform(0.0, 0.25, size=(h, w)), cover)
        return cover, gap

    def truth_polygons(self, field_geom_wgs84, seed_key, resolution=10.0):
        """The synthetic generator's true gap regions as WGS84 polygons (demo/testing only)."""
        from rasterio import features
        from shapely.ops import unary_union
        from ..areas import to_crs_geom, utm_crs_for
        utm = utm_crs_for(field_geom_wgs84)
        g = to_crs_geom(field_geom_wgs84, "EPSG:4326", utm)
        transform, h, w = make_grid(g.bounds, resolution)
        _, gap = self._cover(np.random.default_rng(self._digest(seed_key)), h, w)
        polys = [shape(d).intersection(g) for d, v in features.shapes(gap.astype("uint8"), mask=gap, transform=transform)]
        polys = [p for p in polys if not p.is_empty and p.area > 0 and p.geom_type in ("Polygon", "MultiPolygon")]
        return [to_crs_geom(p, utm, "EPSG:4326") for p in polys]

    def get_composite(self, bounds, crs, date_from, date_to, resolution=10.0, seed_key=""):
        transform, h, w = make_grid(bounds, resolution)
        rng = np.random.default_rng(self._digest(seed_key))
        cover, gap = self._cover(rng, h, w)

        def mix(veg, soil, noise=0.008):
            return (cover * veg + (1 - cover) * soil + rng.normal(0, noise, size=(h, w))).astype("float32")

        bands = {
            "B03": mix(0.07, 0.15), "B04": mix(0.04, 0.17), "B05": mix(0.12, 0.20),
            "B08": mix(0.42, 0.24), "B11": mix(0.17, 0.30),
        }
        bands["B02"] = mix(0.03, 0.12)       # added last so earlier bands keep their noise sequence
        valid = np.full((h, w), 3, dtype="uint8")
        if self.cloud_fraction > 0:
            noise = gaussian_filter(rng.normal(size=(h, w)), sigma=8)
            cl = noise > np.quantile(noise, 1 - self.cloud_fraction)
            for b in bands:
                bands[b][cl] = np.nan
            valid[cl] = 0
        return Composite(
            bands=bands, transform=transform, crs=crs, obs_count=valid,
            scenes=[{"id": "SYNTHETIC", "sensor": "none", "note": "synthetic demo data - not satellite imagery",
                     "date_from": str(date_from), "date_to": str(date_to)}],
            synthetic=True, notes=["SYNTHETIC DEMO DATA - NOT SATELLITE IMAGERY"], resolution_m=resolution)

    def get_s1_composite(self, bounds, crs, date_from, date_to, resolution=10.0, seed_key=""):
        transform, h, w = make_grid(bounds, resolution)
        cover, _ = self._cover(np.random.default_rng(self._digest(seed_key)), h, w)
        # window-specific speckle-like noise so a 'baseline' window differs from the current one
        r2 = np.random.default_rng(self._digest(seed_key) + 7919 + int(hashlib.sha256(str(date_from).encode()).hexdigest()[:4], 16) % 997)
        vh_db = -21.0 + 0.9 * cover + gaussian_filter(r2.normal(0, 3.0, size=(h, w)), 0.8)
        vv_db = -12.5 + 0.4 * cover + gaussian_filter(r2.normal(0, 3.0, size=(h, w)), 0.8)
        bands = {"VV": (10 ** (vv_db / 10)).astype("float32"), "VH": (10 ** (vh_db / 10)).astype("float32")}
        return Composite(
            bands=bands, transform=transform, crs=crs, obs_count=np.full((h, w), 4, dtype="uint8"),
            scenes=[{"id": "SYNTHETIC-S1", "sensor": "none", "note": "synthetic backscatter - not Sentinel-1",
                     "date_from": str(date_from), "date_to": str(date_to)}],
            synthetic=True, notes=["SYNTHETIC S1 DEMO DATA - NOT SENTINEL-1"], resolution_m=resolution)
