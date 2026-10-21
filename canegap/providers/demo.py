"""Synthetic imagery generator for running the app without satellite credentials.

OUTPUT IS NOT SATELLITE DATA. Everything produced here is flagged synthetic=True and the UI,
exports and reports carry that label.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter

from .base import Composite, Provider, make_grid


class DemoProvider(Provider):
    name = "demo-synthetic"
    synthetic = True

    def __init__(self, gap_fraction: float = 0.06, cloud_fraction: float = 0.0, seed: int = 42):
        self.gap_fraction = gap_fraction
        self.cloud_fraction = cloud_fraction
        self.seed = seed

    def get_composite(self, bounds, crs, date_from, date_to, resolution=10.0, seed_key=""):
        transform, h, w = make_grid(bounds, resolution)
        digest = int(hashlib.sha256(f"{self.seed}:{seed_key}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(digest)

        # smooth spatial vigour variation + a few blob-shaped gaps
        vigour = gaussian_filter(rng.normal(size=(h, w)), sigma=6)
        vigour = vigour / (np.abs(vigour).max() + 1e-9) * 0.08
        blobs = gaussian_filter(rng.normal(size=(h, w)), sigma=3.5)
        thr = np.quantile(blobs, 1 - self.gap_fraction)
        gap = blobs > thr
        cover = np.clip(0.95 + vigour, 0.6, 1.0)
        cover = np.where(gap, rng.uniform(0.0, 0.25, size=(h, w)), cover)

        def mix(veg, soil, noise=0.008):
            return (cover * veg + (1 - cover) * soil + rng.normal(0, noise, size=(h, w))).astype("float32")

        bands = {
            "B03": mix(0.07, 0.15), "B04": mix(0.04, 0.17), "B05": mix(0.12, 0.20),
            "B08": mix(0.42, 0.24), "B11": mix(0.17, 0.30),
        }
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
