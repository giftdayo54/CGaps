from __future__ import annotations

import folium
import geopandas as gpd
import pandas as pd

BASEMAPS = {
    "OpenStreetMap": dict(tiles="OpenStreetMap"),
    "Esri World Imagery": dict(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics"),
}


def base_map(bounds, basemap="OpenStreetMap") -> folium.Map:
    m = folium.Map(location=[0, 0], zoom_start=2, tiles=None, control_scale=True)
    for name, kw in BASEMAPS.items():
        folium.TileLayer(name=name, show=(name == basemap), **kw).add_to(m)
    minx, miny, maxx, maxy = bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    return m


def selection_map(all_fields: gpd.GeoDataFrame, outlines: gpd.GeoDataFrame, selected: set[str],
                  basemap="OpenStreetMap") -> folium.Map:
    g = pd.concat([all_fields.geometry, outlines.geometry]) if len(outlines) else all_fields.geometry
    g = g.dropna()
    m = base_map(gpd.GeoSeries(g, crs="EPSG:4326").total_bounds, basemap)
    if len(outlines):
        folium.GeoJson(outlines[["geometry"]].to_json(), name="Farm outline",
                       style_function=lambda f: {"color": "#444", "weight": 2, "fill": False, "dashArray": "6"}).add_to(m)
    f = all_fields.dropna(subset=["geometry"]).copy()
    f["selected"] = f["field_id"].isin(selected)
    f["ha"] = f["area_ha"].round(2)
    folium.GeoJson(
        f[["field_id", "field_name", "ha", "status", "selected", "geometry"]].to_json(), name="Fields",
        style_function=lambda ft: {"color": "#1b5e20" if ft["properties"]["selected"] else "#607d8b",
                                   "weight": 3 if ft["properties"]["selected"] else 1.5,
                                   "fillColor": "#66bb6a" if ft["properties"]["selected"] else "#b0bec5",
                                   "fillOpacity": 0.45 if ft["properties"]["selected"] else 0.2},
        tooltip=folium.GeoJsonTooltip(["field_id", "field_name", "ha", "status"])).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def results_map(fields: gpd.GeoDataFrame, results: pd.DataFrame, gaps: gpd.GeoDataFrame,
                basemap="Esri World Imagery", focus_field: str | None = None) -> folium.Map:
    f = fields.dropna(subset=["geometry"]).merge(
        results[["field_id", "status", "gap_area_ha", "gap_pct"]], on="field_id", how="left", suffixes=("_v", ""))
    base = f if not focus_field else f[f.field_id == focus_field]
    m = base_map(base.total_bounds, basemap)
    def style(ft):
        st = ft["properties"].get("status")
        pct = ft["properties"].get("gap_pct") or 0
        col = {"analysed": "#2e7d32" if pct < 2 else "#f9a825" if pct < 5 else "#c62828",
               "insufficient_evidence": "#9e9e9e", "context_excluded": "#1976d2", "unable": "#000000"}.get(st, "#90a4ae")
        return {"color": col, "weight": 2, "fillColor": col, "fillOpacity": 0.15}
    f2 = f.assign(gap_area_ha=f.gap_area_ha.round(3), gap_pct=f.gap_pct.round(2))
    folium.GeoJson(f2[["field_id", "field_name", "status", "gap_area_ha", "gap_pct", "geometry"]].to_json(),
                   name="Fields (outline coloured by gap %)", style_function=style,
                   tooltip=folium.GeoJsonTooltip(["field_id", "status", "gap_area_ha", "gap_pct"])).add_to(m)
    if len(gaps):
        g = gaps if not focus_field else gaps[gaps.field_id == focus_field]
        g = g.assign(area_ha=g.area_ha.round(4), pct_of_field=g.pct_of_field.round(3))
        folium.GeoJson(
            g[["gap_id", "field_id", "area_ha", "pct_of_field", "category", "confidence", "change_status",
               "verification_status", "detected_at", "geometry"]].to_json(),
            name="Candidate gaps (model prediction)",
            style_function=lambda ft: {"color": "#d50000", "weight": 1.5, "fillColor": "#ff1744", "fillOpacity": 0.5},
            tooltip=folium.GeoJsonTooltip(["gap_id", "field_id", "area_ha", "pct_of_field", "category",
                                           "confidence", "change_status", "verification_status"])).add_to(m)
    folium.LayerControl().add_to(m)
    return m
