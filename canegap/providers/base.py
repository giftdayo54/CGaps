from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from rasterio.transform import Affine, from_origin

BANDS = ["B03", "B04", "B05", "B08", "B11"]  # green, red, red-edge1, NIR, SWIR1 (surface reflectance 0-1)


@dataclass
class Composite:
    bands: dict[str, np.ndarray]          # float32, NaN = no valid observation
    transform: Affine
    crs: str                              # metric CRS of the grid (e.g. EPSG:32736)
    obs_count: np.ndarray | None = None   # number of valid observations per pixel
    scenes: list[dict] = field(default_factory=list)
    synthetic: bool = False
    notes: list[str] = field(default_factory=list)
    resolution_m: float = 10.0


def make_grid(bounds: tuple[float, float, float, float], res: float = 10.0, pad: int = 2):
    """Pixel grid covering `bounds` (minx, miny, maxx, maxy) in a metric CRS."""
    minx, miny, maxx, maxy = bounds
    minx = math.floor(minx / res) * res - pad * res
    maxy = math.ceil(maxy / res) * res + pad * res
    width = int(math.ceil((maxx - minx) / res)) + pad
    height = int(math.ceil((maxy - miny) / res)) + pad
    return from_origin(minx, maxy, res, res), height, width


class ProviderCapabilityError(RuntimeError):
    """The provider does not offer the requested data type."""


class Provider(ABC):
    name = "base"
    synthetic = False

    @abstractmethod
    def get_composite(self, bounds, crs: str, date_from, date_to, resolution: float = 10.0,
                      seed_key: str = "") -> Composite:
        """Return a multi-date composite on a grid covering `bounds` (in `crs`)."""

    def get_s1_composite(self, bounds, crs: str, date_from, date_to, resolution: float = 10.0,
                         seed_key: str = "") -> Composite:
        """Sentinel-1 composite: bands {'VV','VH'} as linear backscatter (sigma0), NaN = no data."""
        raise ProviderCapabilityError(f"{self.name} does not provide Sentinel-1 data.")
