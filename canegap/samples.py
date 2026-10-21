"""Synthetic test/demo boundaries. NOT real farm boundaries."""
from __future__ import annotations

import json

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, box, mapping

# Synthetic location in UTM 36S (EPSG:32736); label only, no real farm.
X0, Y0 = 600_000.0, 8_230_000.0


def _fc(feats, name, crs=None):
    d = {"type": "FeatureCollection", "name": name, "features": feats}
    if crs:
        d["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs}"}}
    return d


def _to_wgs(geom):
    return gpd.GeoSeries([geom], crs="EPSG:32736").to_crs("EPSG:4326").iloc[0]


def farm_with_fields(parent_outline: bool = True) -> dict:
    """Arrangement A/B: 6 fields (3x2), optional parent outline. Attributes include harvest/planting."""
    feats = []
    attrs = [
        dict(variety="R579", ratoon_number=2, last_harvest_date="2025-08-15"),
        dict(variety="N41", ratoon_number=1, last_harvest_date="2025-07-20"),
        dict(variety="R579", ratoon_number=3, last_harvest_date="2026-09-25"),   # recently harvested
        dict(variety="N41", ratoon_number=0, planting_date="2026-09-10"),        # newly planted
        dict(variety="R570", ratoon_number=4, last_harvest_date="2025-06-01"),
        dict(variety="R579", ratoon_number=1, last_harvest_date="2025-09-02"),
    ]
    n = 0
    for r in range(2):
        for c in range(3):
            geom = box(X0 + c * 320, Y0 + r * 360, X0 + c * 320 + 300, Y0 + r * 360 + 340)
            props = {"farm_id": "SYN_FARM_01", "farm_name": "Synthetic Test Farm", "field_id": f"F{n + 1:02d}",
                     "field_name": f"Field {n + 1:02d}", "crop": "Sugarcane", **attrs[n]}
            feats.append({"type": "Feature", "properties": props, "geometry": mapping(_to_wgs(geom))})
            n += 1
    if parent_outline:
        outline = box(X0 - 10, Y0 - 10, X0 + 950, Y0 + 730)
        feats.append({"type": "Feature",
                      "properties": {"farm_id": "SYN_FARM_01", "farm_name": "Synthetic Test Farm", "feature_role": "farm_outline"},
                      "geometry": mapping(_to_wgs(outline))})
    return _fc(feats, "SYNTHETIC_Test_Farm_Fields")


def problem_fixture() -> dict:
    """Deliberate problems: bowtie polygon (repairable), missing ID, duplicate ID, overlap,
    unsupported geometry, null geometry, outside-outline field, MultiPolygon."""
    f = []
    def add(props, geom):
        f.append({"type": "Feature", "properties": props, "geometry": geom})
    base = dict(farm_id="SYN_PROB", farm_name="Synthetic Problem Farm", crop="Sugarcane")
    add({**base, "field_id": "P01"}, mapping(_to_wgs(box(X0, Y0, X0 + 200, Y0 + 200))))
    add({**base, "field_id": "P02"}, mapping(_to_wgs(box(X0 + 150, Y0, X0 + 350, Y0 + 200))))        # overlaps P01
    add({**base, "field_id": "P03"}, mapping(_to_wgs(Polygon([(X0 + 400, Y0), (X0 + 600, Y0 + 200),
                                                              (X0 + 600, Y0), (X0 + 400, Y0 + 200)]))))  # bowtie
    add({**base, "field_id": "P01"}, mapping(_to_wgs(box(X0, Y0 + 300, X0 + 200, Y0 + 500))))        # duplicate ID
    add({**base}, mapping(_to_wgs(box(X0 + 250, Y0 + 300, X0 + 450, Y0 + 500))))                       # missing ID
    add({**base, "field_id": "P06"}, {"type": "Point", "coordinates": [34.9, -16.1]})                  # unsupported
    add({**base, "field_id": "P07"}, None)                                                              # null geom
    mp = MultiPolygon([box(X0 + 500, Y0 + 300, X0 + 600, Y0 + 400), box(X0 + 620, Y0 + 300, X0 + 720, Y0 + 400)])
    add({**base, "field_id": "P08"}, mapping(_to_wgs(mp)))                                              # valid MultiPolygon
    return _fc(f, "SYNTHETIC_Problem_Fixture")


def projected_fixture() -> dict:
    """Same-style fields in EPSG:32736 metres (declared via crs member) - must be reprojected."""
    feats = []
    for i in range(2):
        g = box(X0 + i * 320, Y0, X0 + i * 320 + 300, Y0 + 340)
        feats.append({"type": "Feature", "properties": {"field_id": f"U{i + 1}", "farm_id": "SYN_UTM"},
                      "geometry": mapping(g)})
    return _fc(feats, "SYNTHETIC_Projected_Fixture", crs=32736)


def synthetic_reference() -> dict:
    """SYNTHETIC 'field-verified' polygons (arbitrary positions) for testing the validation workflow only."""
    feats = []
    for fid, (c, r), boxes in [("F02", (1, 0), [(60, 80, 110, 130), (200, 200, 240, 250)]),
                               ("F05", (1, 1), [(100, 100, 150, 140)])]:
        for (a, b, c2, d) in boxes:
            g = box(X0 + c * 320 + a, Y0 + r * 360 + b, X0 + c * 320 + c2, Y0 + r * 360 + d)
            feats.append({"type": "Feature", "properties": {"field_id": fid, "note": "SYNTHETIC reference"},
                          "geometry": mapping(_to_wgs(g))})
    return _fc(feats, "SYNTHETIC_reference_gaps")


def write_samples(folder="sample_data"):
    import pathlib
    p = pathlib.Path(folder); p.mkdir(exist_ok=True)
    for name, fc in [("SYNTHETIC_farm_with_outline.geojson", farm_with_fields(True)),
                     ("SYNTHETIC_fields_only.geojson", farm_with_fields(False)),
                     ("SYNTHETIC_problem_fixture.geojson", problem_fixture()),
                     ("SYNTHETIC_projected_utm36s.geojson", projected_fixture()),
                     ("SYNTHETIC_reference_gaps.geojson", synthetic_reference())]:
        (p / name).write_text(json.dumps(fc))


if __name__ == "__main__":
    write_samples()
