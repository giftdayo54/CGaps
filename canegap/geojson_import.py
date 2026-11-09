"""GeoJSON parsing, CRS checks, geometry validation and attribute mapping.

Nothing is silently dropped: every feature gets a status (accepted / repaired /
warning / rejected) and a list of human-readable issues.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.validation import explain_validity, make_valid

from .areas import area_ha

MAPPING_KEYS = [
    "farm_id", "farm_name", "estate", "field_id", "field_name", "crop", "variety",
    "planting_date", "last_harvest_date", "ratoon_number", "irrigation_system",
]
REQUIRED_FOR_ANALYSIS = ["field_id"]
SUPPORTED = {"Polygon", "MultiPolygon"}
MAX_FEATURES = 20_000
MAX_REPAIR_AREA_CHANGE = 0.02  # repairs changing area by >2% are rejected, not applied
OVERLAP_WARN_M2 = 1.0

_GUESSES = {
    "farm_id": ["farm_id", "farmid", "farm_code", "farm"],
    "farm_name": ["farm_name", "farmname"],
    "estate": ["estate", "estate_name", "estate_id"],
    "field_id": ["field_id", "fieldid", "block_id", "blockid", "field_code", "block", "id"],
    "field_name": ["field_name", "fieldname", "block_name", "name"],
    "crop": ["crop", "crop_type"],
    "variety": ["variety", "cane_variety"],
    "planting_date": ["planting_date", "plant_date", "replant_date", "planted"],
    "last_harvest_date": ["last_harvest_date", "harvest_date", "last_harvest"],
    "ratoon_number": ["ratoon_number", "ratoon", "ratoon_no", "crop_class"],
    "irrigation_system": ["irrigation_system", "irrigation", "irrig_type"],
}


class ImportError_(ValueError):
    """Raised for problems that stop the import (bad JSON, unknown CRS, ...)."""


class CRSNeedsConfirmation(ImportError_):
    """Coordinates do not look like WGS84 lon/lat; the user must declare the source EPSG."""


@dataclass
class ValidationResult:
    features: gpd.GeoDataFrame            # all features incl. rejected (geometry may be None)
    summary: dict[str, Any]
    dataset_issues: list[str] = field(default_factory=list)
    mapping: dict[str, str | None] = field(default_factory=dict)
    source_crs: str = "EPSG:4326"

    @property
    def usable(self) -> gpd.GeoDataFrame:
        return self.features[self.features["status"] != "rejected"]

    def report(self) -> dict:
        return {
            "summary": self.summary,
            "dataset_issues": self.dataset_issues,
            "source_crs": self.source_crs,
            "mapping": self.mapping,
            "features": json.loads(
                self.features.drop(columns="geometry").to_json(orient="records", default_handler=str)
            ),
        }

    def report_csv(self) -> str:
        cols = ["src_index", "field_id", "field_name", "status", "area_ha", "issues"]
        df = self.features.drop(columns="geometry")
        return df[[c for c in cols if c in df.columns]].to_csv(index=False)


# ------------------------------------------------------------------ parsing
def parse_geojson(data: bytes | str | dict) -> dict:
    if isinstance(data, (bytes, bytearray)):
        try:
            data = data.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ImportError_(f"File is not valid UTF-8 text: {e}") from e
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ImportError_(f"File is not valid JSON: {e}") from e
    if not isinstance(data, dict) or "type" not in data:
        raise ImportError_("Not a GeoJSON object (missing 'type').")
    t = data["type"]
    if t == "Feature":
        data = {"type": "FeatureCollection", "features": [data]}
    elif t in ("Polygon", "MultiPolygon"):
        data = {"type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": data}]}
    elif t != "FeatureCollection":
        raise ImportError_(f"Unsupported GeoJSON type '{t}'. Expected a FeatureCollection.")
    feats = data.get("features")
    if not isinstance(feats, list) or not feats:
        raise ImportError_("FeatureCollection contains no features.")
    if len(feats) > MAX_FEATURES:
        raise ImportError_(f"Too many features ({len(feats)} > {MAX_FEATURES}).")
    return data


def declared_crs(gj: dict) -> str | None:
    """CRS from a legacy 'crs' member (e.g. QGIS/ogr output), if present."""
    name = (gj.get("crs") or {}).get("properties", {}).get("name")
    if not name:
        return None
    m = re.search(r"EPSG(?:::|:)(\d+)", name)
    if m:
        return f"EPSG:{m.group(1)}"
    if "CRS84" in name:
        return "OGC:CRS84"
    return None


def _iter_coords(c):
    if c and isinstance(c[0], (int, float)):
        yield c
    else:
        for sub in c or []:
            yield from _iter_coords(sub)


def looks_like_lonlat(gj: dict) -> bool:
    for f in gj["features"]:
        g = f.get("geometry")
        if not g or "coordinates" not in g:
            continue
        for x, y, *_ in _iter_coords(g["coordinates"]):
            if not (-180 <= x <= 180 and -90 <= y <= 90):
                return False
    return True


def guess_mapping(columns: list[str]) -> dict[str, str | None]:
    lower = {c.lower(): c for c in columns}
    out: dict[str, str | None] = {}
    for key in MAPPING_KEYS:
        out[key] = next((lower[g] for g in _GUESSES[key] if g in lower), None)
    return out


# --------------------------------------------------------------- validation
def _clean_poly(geom):
    """Keep only polygonal parts after make_valid."""
    if geom.geom_type in SUPPORTED:
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in SUPPORTED and not g.is_empty]
        if polys:
            from shapely.ops import unary_union
            return unary_union(polys)
    return None


def validate_geojson(
    data: bytes | str | dict,
    mapping: dict[str, str | None] | None = None,
    source_crs: str | None = None,
    outline_rows: list[int] | None = None,
    id_policy: str = "generate_missing",
    default_farm_id: str = "FARM_001",
    default_farm_name: str | None = None,
) -> ValidationResult:
    """Validate a GeoJSON and return a per-feature report.

    source_crs: EPSG string declared by the user. If None, a legacy 'crs' member is
    used, else WGS84 lon/lat is assumed *only if* coordinates are within lon/lat range.
    id_policy: 'generate_missing' (unique IDs created for missing/duplicate IDs, original
    kept in `orig_field_id`) or 'strict' (missing/duplicate IDs are rejected).
    outline_rows: source feature indices that are parent farm outlines (Arrangement B).
    """
    gj = parse_geojson(data)
    dataset_issues: list[str] = []

    crs = source_crs or declared_crs(gj)
    if crs in (None, "OGC:CRS84"):
        if not looks_like_lonlat(gj):
            raise CRSNeedsConfirmation(
                "Coordinates fall outside lon/lat range, so this file is probably in a projected "
                "CRS. Declare its EPSG code (e.g. EPSG:32736) so it can be reprojected correctly. "
                "Coordinates are never relabelled as WGS84."
            )
        crs = "EPSG:4326"
    else:
        # fail early if pyproj cannot understand the CRS
        try:
            import pyproj
            pyproj.CRS.from_user_input(crs)
        except Exception as e:  # noqa: BLE001
            raise ImportError_(f"Unknown CRS '{crs}': {e}") from e

    rows = []
    for i, f in enumerate(gj["features"]):
        props = dict(f.get("properties") or {})
        g = f.get("geometry")
        row = {"src_index": i, "_props": props, "_geom": None, "issues": [], "status": "accepted"}
        if not g:
            row["issues"].append("Missing or null geometry")
            row["status"] = "rejected"
        elif g.get("type") not in SUPPORTED:
            row["issues"].append(f"Unsupported geometry type '{g.get('type')}'")
            row["status"] = "rejected"
        elif not g.get("coordinates"):
            row["issues"].append("Empty coordinates")
            row["status"] = "rejected"
        else:
            try:
                geom = shape(g)
            except Exception as e:  # noqa: BLE001
                row["issues"].append(f"Cannot parse geometry: {e}")
                row["status"] = "rejected"
            else:
                row["_geom"] = geom
        rows.append(row)

    # reproject to WGS84 before repair/area so downstream everything is lon/lat
    ok = [r for r in rows if r["_geom"] is not None]
    if ok and crs != "EPSG:4326":
        ser = gpd.GeoSeries([r["_geom"] for r in ok], crs=crs).to_crs("EPSG:4326")
        for r, g in zip(ok, ser):
            r["_geom"] = g
        dataset_issues.append(f"Reprojected from {crs} to EPSG:4326.")

    for r in ok:
        geom = r["_geom"]
        if geom.is_empty:
            r["issues"].append("Empty geometry"); r["status"] = "rejected"; r["_geom"] = None
            continue
        if not geom.is_valid:
            reason = explain_validity(geom)
            fixed = _clean_poly(make_valid(geom))
            if fixed is None or fixed.is_empty:
                r["issues"].append(f"Invalid geometry, repair failed ({reason})")
                r["status"] = "rejected"; r["_geom"] = None
                continue
            a0, a1 = area_ha(geom), area_ha(fixed)
            change = abs(a1 - a0) / max(a0, 1e-9)
            if change > MAX_REPAIR_AREA_CHANGE:
                r["issues"].append(
                    f"Invalid geometry ({reason}); automatic repair would change area by "
                    f"{change:.1%}, so it was NOT applied")
                r["status"] = "rejected"; r["_geom"] = None
                continue
            r["_geom"] = fixed
            r["issues"].append(f"Repaired invalid geometry ({reason}); area change {change:.2%}")
            r["status"] = "repaired"

    outline_rows = set(outline_rows or [])
    mapping = mapping or guess_mapping(sorted({k for r in rows for k in r["_props"]}))
    mapping = {k: (mapping.get(k) or None) for k in MAPPING_KEYS}

    def get(r, key):
        col = mapping.get(key)
        v = r["_props"].get(col) if col else None
        return None if (v is None or (isinstance(v, str) and not v.strip())) else v

    recs = []
    for r in rows:
        rec = {
            "src_index": r["src_index"],
            "is_outline": r["src_index"] in outline_rows,
            "orig_field_id": get(r, "field_id"),
            "field_id": get(r, "field_id"),
            "field_name": get(r, "field_name"),
            "farm_id": get(r, "farm_id"),
            "farm_name": get(r, "farm_name"),
            "estate": get(r, "estate"),
            "crop": get(r, "crop"),
            "variety": get(r, "variety"),
            "planting_date": get(r, "planting_date"),
            "last_harvest_date": get(r, "last_harvest_date"),
            "ratoon_number": get(r, "ratoon_number"),
            "irrigation_system": get(r, "irrigation_system"),
            "status": r["status"],
            "issues": list(r["issues"]),
            "props": json.dumps(r["_props"], default=str),
            "geometry": r["_geom"],
        }
        recs.append(rec)
    gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")

    # ---- identifiers (fields only; outlines are not fields)
    is_field = ~gdf["is_outline"]
    used: set[str] = set()
    for idx in gdf.index[is_field]:
        fid = gdf.at[idx, "field_id"]
        problem = None
        if fid is None or pd.isna(fid):
            problem = "Missing field ID"
        elif str(fid) in used:
            problem = f"Duplicate field ID '{fid}'"
        if problem:
            if id_policy == "generate_missing":
                n = 1
                while f"FIELD_{n:03d}" in used or any(
                        str(x) == f"FIELD_{n:03d}" for x in gdf["field_id"].dropna()):
                    n += 1
                new = f"FIELD_{n:03d}"
                gdf.at[idx, "field_id"] = new
                gdf.at[idx, "issues"] = gdf.at[idx, "issues"] + [
                    f"{problem}; generated unique ID '{new}' (original kept in orig_field_id)"]
                if gdf.at[idx, "status"] == "accepted":
                    gdf.at[idx, "status"] = "warning"
            else:
                gdf.at[idx, "issues"] = gdf.at[idx, "issues"] + [problem]
                gdf.at[idx, "status"] = "rejected"
                gdf.at[idx, "geometry"] = gdf.at[idx, "geometry"]  # kept for display
                continue
        used.add(str(gdf.at[idx, "field_id"]))
        gdf.at[idx, "field_id"] = str(gdf.at[idx, "field_id"])

    if gdf["farm_id"].isna().all():
        gdf["farm_id"] = default_farm_id
        dataset_issues.append(f"No farm ID attribute mapped; using '{default_farm_id}'.")
    else:
        gdf["farm_id"] = gdf["farm_id"].fillna(default_farm_id).astype(str)
    if gdf["farm_name"].isna().all():
        gdf["farm_name"] = default_farm_name or gdf["farm_id"]
    gdf["farm_name"] = gdf["farm_name"].fillna(gdf["farm_id"])
    gdf["field_name"] = gdf["field_name"].fillna(gdf["field_id"])

    # ---- areas
    gdf["area_ha"] = [area_ha(g) if g is not None else None for g in gdf.geometry]

    # ---- duplicate geometries and overlaps among fields
    fidx = [i for i in gdf.index[is_field] if gdf.at[i, "status"] != "rejected" and gdf.at[i, "geometry"] is not None]
    geoms = [gdf.at[i, "geometry"] for i in fidx]
    wkbs: dict[bytes, int] = {}
    for i, g in zip(fidx, geoms):
        k = g.normalize().wkb
        if k in wkbs:
            other = gdf.at[wkbs[k], "field_id"]
            gdf.at[i, "issues"] = gdf.at[i, "issues"] + [f"Geometry duplicates field '{other}'"]
            if gdf.at[i, "status"] == "accepted":
                gdf.at[i, "status"] = "warning"
        else:
            wkbs[k] = i
    overlaps = []
    if len(geoms) > 1:
        from .areas import utm_crs_for
        utm = utm_crs_for(geoms[0])
        proj = gpd.GeoSeries(geoms, crs="EPSG:4326").to_crs(utm).tolist()
        tree = STRtree(proj)
        for a, pg in enumerate(proj):
            for b in tree.query(pg):
                if b <= a:
                    continue
                inter = pg.intersection(proj[b])
                if inter.area > OVERLAP_WARN_M2 and not inter.is_empty:
                    ia, ib = fidx[a], fidx[b]
                    overlaps.append({"field_a": gdf.at[ia, "field_id"], "field_b": gdf.at[ib, "field_id"],
                                     "overlap_m2": round(inter.area, 1)})
                    for i, other in ((ia, ib), (ib, ia)):
                        gdf.at[i, "issues"] = gdf.at[i, "issues"] + [
                            f"Overlaps field '{gdf.at[other, 'field_id']}' by {inter.area:,.0f} m²"]
                        if gdf.at[i, "status"] == "accepted":
                            gdf.at[i, "status"] = "warning"
    # ---- outline containment
    outlines = gdf[gdf["is_outline"] & gdf.geometry.notna()]
    if len(outlines):
        from shapely.ops import unary_union
        outline = unary_union(outlines.geometry.tolist()).buffer(1e-5)
        for i in fidx:
            if not outline.contains(gdf.at[i, "geometry"]):
                gdf.at[i, "issues"] = gdf.at[i, "issues"] + ["Extends outside the farm outline"]
                if gdf.at[i, "status"] == "accepted":
                    gdf.at[i, "status"] = "warning"
    # ---- missing optional attributes
    for key in ("crop", "variety", "planting_date", "last_harvest_date", "ratoon_number"):
        if gdf.loc[is_field, key].isna().all():
            dataset_issues.append(
                f"No '{key}' attribute available: context rules using it will be skipped in analysis.")

    gdf["issues"] = gdf["issues"].apply(lambda l: "; ".join(l))
    fields = gdf[is_field]
    summary = {
        "n_features": int(len(gdf)),
        "n_outlines": int(gdf["is_outline"].sum()),
        "n_fields": int(is_field.sum()),
        "accepted": int((fields.status == "accepted").sum()),
        "repaired": int((fields.status == "repaired").sum()),
        "warning": int((fields.status == "warning").sum()),
        "rejected": int((fields.status == "rejected").sum()),
        "overlaps": overlaps,
        "total_usable_area_ha": round(float(fields[fields.status != "rejected"]["area_ha"].sum()), 2),
    }
    return ValidationResult(gdf, summary, dataset_issues, mapping, crs)


def subset_validation(vr: ValidationResult, field_ids) -> ValidationResult:
    """Keep only the chosen fields (plus any farm outlines) from a validation result.

    Used when the user imports only part of a farm file. Features that are dropped are listed in the dataset issues and
    stay in the stored import report's source file name, so nothing disappears silently."""
    keep = {str(f) for f in field_ids}
    f = vr.features
    mask = f["is_outline"] | f["field_id"].astype(str).isin(keep)
    sub = f[mask & (f["status"] != "rejected")].copy() if not f["is_outline"].all() else f[mask].copy()
    dropped = f[~mask]
    issues = list(vr.dataset_issues)
    if len(dropped):
        names = ", ".join(map(str, dropped["field_id"].tolist()[:12])) + (" ..." if len(dropped) > 12 else "")
        issues.append(f"{len(dropped)} feature(s) NOT imported by user selection: {names}")
    fields = sub[~sub["is_outline"]]
    summary = dict(vr.summary)
    summary.update({"n_features": int(len(sub)), "n_fields": int(len(fields)),
                    "accepted": int((fields.status == "accepted").sum()), "repaired": int((fields.status == "repaired").sum()),
                    "warning": int((fields.status == "warning").sum()), "rejected": 0, "not_imported": int(len(dropped)),
                    "total_usable_area_ha": round(float(fields["area_ha"].sum()), 2)})
    return ValidationResult(sub, summary, issues, dict(vr.mapping), vr.source_crs)
