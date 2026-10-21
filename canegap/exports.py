from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

SHP_RENAME = {"gap_id": "gap_id", "farm_id": "farm_id", "field_id": "field_id", "version_pk": "bv_id",
              "area_m2": "area_m2", "area_ha": "area_ha", "pct_of_field": "pct_field", "category": "category",
              "confidence": "conf", "change_status": "change", "detected_at": "detected",
              "verification_status": "verif", "proposed_action": "action", "synthetic": "synthetic"}


def gaps_with_flag(gaps: gpd.GeoDataFrame, synthetic: bool) -> gpd.GeoDataFrame:
    g = gaps.copy()
    g["synthetic"] = "SYNTHETIC DEMO - NOT SATELLITE DATA" if synthetic else "no"
    return g


def to_geojson(gaps: gpd.GeoDataFrame) -> bytes:
    return gaps.to_json(drop_id=True).encode("utf-8")


def to_csv(df: pd.DataFrame) -> bytes:
    return df.drop(columns=[c for c in ("geometry",) if c in df.columns]).to_csv(index=False).encode("utf-8")


def to_gpkg(gaps: gpd.GeoDataFrame, fields: gpd.GeoDataFrame | None = None) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "canegap.gpkg"
        gaps.to_file(p, layer="candidate_gaps", driver="GPKG")
        if fields is not None and len(fields):
            fields[["field_id", "field_name", "area_ha", "geometry"]].to_file(p, layer="fields", driver="GPKG")
        return p.read_bytes()


def to_shapefile_zip(gaps: gpd.GeoDataFrame) -> bytes:
    g = gaps.rename(columns=SHP_RENAME)
    keep = [c for c in SHP_RENAME.values() if c in g.columns] + ["geometry"]
    g = g[keep]
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "candidate_gaps.shp"
        g.to_file(base, driver="ESRI Shapefile")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in Path(d).glob("candidate_gaps.*"):
                z.write(f, f.name)
        return buf.getvalue()


def to_xlsx(field_results: pd.DataFrame, gaps: pd.DataFrame, farm_summary: dict, notes: list[str]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"note": notes}).to_excel(xw, sheet_name="README", index=False)
        pd.DataFrame([farm_summary]).to_excel(xw, sheet_name="Farm summary", index=False)
        field_results.to_excel(xw, sheet_name="Fields", index=False)
        gaps.drop(columns=["geometry"], errors="ignore").to_excel(xw, sheet_name="Gaps", index=False)
    return buf.getvalue()
