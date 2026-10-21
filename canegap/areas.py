"""Area helpers. Areas are always computed in a local metric CRS (UTM), never in degrees."""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

M2_PER_HA = 10_000.0


def utm_crs_for(geom: BaseGeometry, src_crs="EPSG:4326"):
    """Local UTM CRS for a geometry given in `src_crs`."""
    return gpd.GeoSeries([geom], crs=src_crs).estimate_utm_crs()


def to_crs_geom(geom: BaseGeometry, src_crs, dst_crs) -> BaseGeometry:
    return gpd.GeoSeries([geom], crs=src_crs).to_crs(dst_crs).iloc[0]


def area_m2(geom: BaseGeometry, src_crs="EPSG:4326") -> float:
    if geom is None or geom.is_empty:
        return 0.0
    utm = utm_crs_for(geom, src_crs)
    return float(to_crs_geom(geom, src_crs, utm).area)


def area_ha(geom: BaseGeometry, src_crs="EPSG:4326") -> float:
    return area_m2(geom, src_crs) / M2_PER_HA


def gap_percentage(gap_area_ha: float, field_area_ha: float) -> float:
    """Field gap % = candidate-gap area / valid field area x 100."""
    if field_area_ha <= 0:
        return 0.0
    return 100.0 * gap_area_ha / field_area_ha
