import pytest
from shapely.geometry import box
import geopandas as gpd
from canegap.areas import area_ha, gap_percentage


def test_hectares_from_wgs84_square():
    utm_sq = box(600000, 8230000, 600100, 8230100)  # 100 m x 100 m = 1 ha
    g = gpd.GeoSeries([utm_sq], crs="EPSG:32736").to_crs("EPSG:4326").iloc[0]
    assert area_ha(g) == pytest.approx(1.0, rel=0.002)


def test_gap_percentage():
    assert gap_percentage(2.0, 40.0) == pytest.approx(5.0)
    assert gap_percentage(1.0, 0.0) == 0.0
