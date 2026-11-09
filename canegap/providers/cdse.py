"""Sentinel-2 L2A via the Copernicus Data Space Ecosystem Sentinel Hub Process API.

Endpoints (CDSE documentation):
  token   https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
  process https://sh.dataspace.copernicus.eu/api/v1/process
Credentials: OAuth client created in the CDSE dashboard; supply CDSE_CLIENT_ID /
CDSE_CLIENT_SECRET as environment variables (backend only; never in code or git).

STATUS: written against the documented API but NOT tested against the live service in
the build environment (no credentials). Test with a small field first. Quotas and terms
are set by CDSE; check your account's Processing Unit allowance.
Note: B05 and B11 are native 20 m bands; the service resamples them to the requested
10 m grid, which does not add genuine spatial detail.
"""
from __future__ import annotations

import io
import os
import time

import numpy as np
import requests

from .base import BANDS, Composite, Provider, make_grid

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B02","B03","B04","B05","B08","B11","SCL","dataMask"]}],
    mosaicking: "ORBIT",
    output: [{id: "bands", bands: 7, sampleType: "FLOAT32"}]
  };
}
function median(a) {
  a.sort(function(x, y) { return x - y; });
  var n = a.length;
  return n % 2 ? a[(n - 1) / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}
function evaluatePixel(samples) {
  // valid = vegetation(4), not vegetated(5), water(6), unclassified(7); excludes shadow(3), cloud(8,9), cirrus(10), snow(11)
  var ok = [];
  for (var i = 0; i < samples.length; i++) {
    var s = samples[i];
    if (s.dataMask == 1 && (s.SCL == 4 || s.SCL == 5 || s.SCL == 6 || s.SCL == 7)) ok.push(s);
  }
  if (ok.length == 0) return {bands: [NaN, NaN, NaN, NaN, NaN, NaN, 0]};
  var out = [];
  var names = ["B02","B03","B04","B05","B08","B11"];
  for (var b = 0; b < names.length; b++) {
    var v = [];
    for (var k = 0; k < ok.length; k++) v.push(ok[k][names[b]]);
    out.push(median(v));
  }
  out.push(ok.length);
  return {bands: out};
}
"""


S1_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH", "dataMask"]}],
    mosaicking: "ORBIT",
    output: [{id: "bands", bands: 3, sampleType: "FLOAT32"}]
  };
}
function evaluatePixel(samples) {
  // temporal mean in the LINEAR domain (never average dB values)
  var sv = 0, sh = 0, n = 0;
  for (var i = 0; i < samples.length; i++) {
    var s = samples[i];
    if (s.dataMask == 1 && s.VV > 0 && s.VH > 0) { sv += s.VV; sh += s.VH; n++; }
  }
  if (n == 0) return {bands: [NaN, NaN, 0]};
  return {bands: [sv / n, sh / n, n]};
}
"""


class CDSEUnavailable(RuntimeError):
    pass


class CDSEProvider(Provider):
    name = "cdse-sentinel2-l2a"
    synthetic = False

    def __init__(self, client_id: str | None = None, client_secret: str | None = None,
                 max_cloud_cover: float = 60.0, timeout: int = 120):
        self.client_id = client_id or os.environ.get("CDSE_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("CDSE_CLIENT_SECRET")
        self.max_cloud_cover = max_cloud_cover
        self.timeout = timeout
        self._token = None
        self._token_exp = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _auth(self) -> str:
        if not self.configured:
            raise CDSEUnavailable("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are not set.")
        if self._token and time.time() < self._token_exp - 30:
            return self._token
        r = requests.post(TOKEN_URL, data={
            "grant_type": "client_credentials", "client_id": self.client_id,
            "client_secret": self.client_secret}, timeout=self.timeout)
        if r.status_code != 200:
            raise CDSEUnavailable(f"Authentication failed (HTTP {r.status_code}). Check client id/secret.")
        j = r.json()
        self._token, self._token_exp = j["access_token"], time.time() + j.get("expires_in", 300)
        return self._token

    def get_composite(self, bounds, crs, date_from, date_to, resolution=10.0, seed_key=""):
        import rasterio
        from rasterio.io import MemoryFile

        transform, h, w = make_grid(bounds, resolution)
        if h * w > 2500 * 2500:
            raise CDSEUnavailable("Field bounding box too large for a single request (max 2500x2500 px).")
        minx, maxy = transform.c, transform.f
        bbox = [minx, maxy - h * resolution, minx + w * resolution, maxy]
        epsg = crs.split(":")[-1]
        payload = {
            "input": {
                "bounds": {"bbox": bbox,
                           "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
                        "maxCloudCoverage": self.max_cloud_cover},
                }],
            },
            "output": {"width": w, "height": h,
                       "responses": [{"identifier": "bands", "format": {"type": "image/tiff"}}]},
            "evalscript": EVALSCRIPT,
        }
        headers = {"Authorization": f"Bearer {self._auth()}"}
        for attempt in range(3):
            r = requests.post(PROCESS_URL, json=payload, headers=headers, timeout=self.timeout)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2); continue
            break
        if r.status_code == 401:
            self._token = None
            raise CDSEUnavailable("Token rejected (HTTP 401).")
        if r.status_code == 429:
            raise CDSEUnavailable("Rate limited by CDSE (HTTP 429). Try again later.")
        if r.status_code != 200:
            raise CDSEUnavailable(f"Process API error HTTP {r.status_code}: {r.text[:300]}")
        with MemoryFile(r.content) as mf, mf.open() as ds:
            arr = ds.read().astype("float32")
        if arr.shape[0] != 7:
            raise CDSEUnavailable(f"Unexpected response with {arr.shape[0]} bands.")
        bands = {b: arr[i] for i, b in enumerate(BANDS)}
        return Composite(
            bands=bands, transform=transform, crs=crs, obs_count=arr[6].astype("uint8"),
            scenes=[{"sensor": "Sentinel-2 L2A", "composite": "per-pixel median of SCL-valid observations",
                     "date_from": str(date_from), "date_to": str(date_to),
                     "max_cloud_cover_filter": self.max_cloud_cover}],
            synthetic=False, resolution_m=resolution,
            notes=["B05/B11 are 20 m bands resampled to the 10 m grid by the service."])

    # ------------------------------------------------------------------ Sentinel-1
    orbit_direction: str | None = None   # "ASCENDING" / "DESCENDING"; mixing geometries adds noise, so set one

    def get_s1_composite(self, bounds, crs, date_from, date_to, resolution=10.0, seed_key=""):
        """Sentinel-1 IW GRD (VV+VH), orthorectified sigma0 (ellipsoid), temporal mean in linear power.

        UNTESTED against the live service (no credentials in the build environment). The processing block
        (backCoeff / demInstance / orthorectify) follows the Sentinel Hub Process API documentation; check it
        against the current CDSE docs. Speckle is reduced by the multi-date mean plus an optional spatial filter
        applied later in the feature stage; geometry (orbit direction) and soil-moisture/roughness effects still
        affect backscatter, and S1 alone cannot reliably separate cane from weeds or bare ground.
        """
        from rasterio.io import MemoryFile
        transform, h, w = make_grid(bounds, resolution)
        if h * w > 2500 * 2500:
            raise CDSEUnavailable("Field bounding box too large for a single request (max 2500x2500 px).")
        minx, maxy = transform.c, transform.f
        bbox = [minx, maxy - h * resolution, minx + w * resolution, maxy]
        flt = {"timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
               "acquisitionMode": "IW", "polarization": "DV", "resolution": "HIGH"}
        if self.orbit_direction:
            flt["orbitDirection"] = self.orbit_direction
        payload = {
            "input": {"bounds": {"bbox": bbox, "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{crs.split(':')[-1]}"}},
                      "data": [{"type": "sentinel-1-grd", "dataFilter": flt,
                                "processing": {"orthorectify": True, "backCoeff": "SIGMA0_ELLIPSOID",
                                               "demInstance": "COPERNICUS_30"}}]},
            "output": {"width": w, "height": h, "responses": [{"identifier": "bands", "format": {"type": "image/tiff"}}]},
            "evalscript": S1_EVALSCRIPT,
        }
        headers = {"Authorization": f"Bearer {self._auth()}"}
        for attempt in range(3):
            r = requests.post(PROCESS_URL, json=payload, headers=headers, timeout=self.timeout)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2); continue
            break
        if r.status_code != 200:
            raise CDSEUnavailable(f"Sentinel-1 Process API error HTTP {r.status_code}: {r.text[:300]}")
        with MemoryFile(r.content) as mf, mf.open() as ds:
            arr = ds.read().astype("float32")
        if arr.shape[0] != 3:
            raise CDSEUnavailable(f"Unexpected S1 response with {arr.shape[0]} bands.")
        return Composite(
            bands={"VV": arr[0], "VH": arr[1]}, transform=transform, crs=crs, obs_count=arr[2].astype("uint8"),
            scenes=[{"sensor": "Sentinel-1 IW GRD", "composite": "linear-power temporal mean, sigma0 ellipsoid, orthorectified",
                     "orbit_direction": self.orbit_direction or "both (mixed geometry)",
                     "date_from": str(date_from), "date_to": str(date_to)}],
            synthetic=False, resolution_m=resolution,
            notes=[] if self.orbit_direction else ["Ascending and descending passes were mixed; fix the orbit direction "
                                                   "for consistent viewing geometry."])
