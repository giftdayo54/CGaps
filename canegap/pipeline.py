"""Field-specific candidate-gap detection (transparent threshold baseline).

IMPORTANT: this is an UNVALIDATED screening method. Confidence values are heuristic scores,
not measured accuracy. Candidate gaps are not confirmed missing cane; they require field
verification. Sentinel-2 pixels are 10 m (B05/B11 native 20 m), so mapped gaps are
quantised by pixel size and mixed pixels at gap/field edges.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import geopandas as gpd
import numpy as np
from rasterio import features
from scipy import ndimage as ndi
from shapely.geometry import shape
from shapely.ops import unary_union

from . import ALGO_VERSION
from .areas import M2_PER_HA, gap_percentage, to_crs_geom, utm_crs_for
from .providers import Provider


@dataclass
class Params:
    resolution_m: float = 10.0
    ndvi_abs_threshold: float = 0.50     # pixel must be below this absolute NDVI ...
    robust_z: float = 3.0                # ... and this many robust SDs below the field median
    bare_ndvi: float = 0.25              # below this => "no detectable cane"
    mndwi_water: float = 0.10            # waterlogging indicator (with low NDVI)
    min_mapping_unit_m2: float = 400.0   # 4 pixels at 10 m
    merge_distance_m: float = 10.0       # closing distance to join near-touching patches
    edge_buffer_m: float = 10.0          # exclude edge pixels (mixed pixels at boundaries)
    min_valid_fraction: float = 0.60     # below this => insufficient evidence
    harvest_window_days: int = 120       # recently harvested => not assessed
    planting_window_days: int = 90       # newly planted => not assessed
    persistence_overlap: float = 0.5


@dataclass
class FieldResult:
    field_id: str
    status: str                 # analysed | insufficient_evidence | context_excluded | unable
    reason: str = ""
    field_area_ha: float = 0.0
    analysed_area_ha: float = 0.0
    valid_fraction: float = 0.0
    gap_area_ha: float = 0.0
    gap_pct: float = 0.0
    gaps: list[dict] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    scenes: list[dict] = field(default_factory=list)


def _parse_date(v) -> date | None:
    if v is None or str(v).strip() in ("", "None", "nan", "NaT"):
        return None
    s = str(v).strip()[:10]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def indices(b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def nd(a, c):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (a - c) / (a + c)
    nir, red, g, re1, sw = b["B08"], b["B04"], b["B03"], b["B05"], b["B11"]
    out = {"NDVI": nd(nir, red), "NDRE": nd(nir, re1), "GNDVI": nd(nir, g),
           "NDMI": nd(nir, sw), "MNDWI": nd(g, sw)}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["SAVI"] = 1.5 * (nir - red) / (nir + red + 0.5)
        out["EVI"] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * b.get("B02", g) + 1) if "B02" in b else np.full_like(nir, np.nan)
    return out


def apply_overlap_policy(fields: gpd.GeoDataFrame, policy: str) -> tuple[dict[str, object], list[dict]]:
    """Return ({field_id: analysis geometry (EPSG:4326)}, overlaps).

    flag_only      geometries unchanged (aggregate totals may double count overlaps)
    first_wins     overlapping area is assigned to the earlier field (row order)
    exclude        overlapping area is removed from every field involved
    """
    ids = list(fields["field_id"])
    geoms = list(fields.geometry)
    utm = utm_crs_for(unary_union([g for g in geoms if g is not None]))
    proj = [to_crs_geom(g, "EPSG:4326", utm) if g is not None else None for g in geoms]
    overlaps = []
    out = list(proj)
    for i in range(len(proj)):
        for j in range(i + 1, len(proj)):
            if proj[i] is None or proj[j] is None or not proj[i].intersects(proj[j]):
                continue
            inter = proj[i].intersection(proj[j])
            if inter.area > 1.0:
                overlaps.append({"field_a": ids[i], "field_b": ids[j], "overlap_m2": round(inter.area, 1)})
                if policy == "first_wins":
                    out[j] = out[j].difference(inter)
                elif policy == "exclude":
                    out[i] = out[i].difference(inter); out[j] = out[j].difference(inter)
    res = {fid: (to_crs_geom(g, utm, "EPSG:4326") if g is not None and not g.is_empty else None)
           for fid, g in zip(ids, out)}
    return res, overlaps


def analyse_field(field_id: str, geom_wgs84, props: dict, provider: Provider, date_from, date_to,
                  params: Params | None = None, previous_gaps: list | None = None,
                  previous_date: str | None = None) -> FieldResult:
    p = params or Params()
    if geom_wgs84 is None or geom_wgs84.is_empty:
        return FieldResult(field_id, "unable", "No usable geometry (empty after overlap policy or invalid).")
    utm = utm_crs_for(geom_wgs84)
    g = to_crs_geom(geom_wgs84, "EPSG:4326", utm)
    field_ha = g.area / M2_PER_HA
    res = FieldResult(field_id, "unable", field_area_ha=field_ha)

    crop = props.get("crop")
    if crop and "cane" not in str(crop).lower():
        res.reason = f"Crop attribute is '{crop}', not sugarcane."
        return res

    end = _parse_date(date_to) or date.today()
    notes = []
    for key, window, label in (("last_harvest_date", p.harvest_window_days, "Recently harvested"),
                               ("planting_date", p.planting_window_days, "Newly planted")):
        d = _parse_date(props.get(key))
        if props.get(key) and d is None:
            notes.append(f"Could not parse {key}='{props.get(key)}'; context rule skipped.")
        elif d and d <= end and (end - d).days <= window:
            res.status = "context_excluded"
            res.reason = (f"{label} ({d.isoformat()}, {(end - d).days} d before assessment end); low vegetation is "
                          f"expected, so gaps were not assessed. Re-run later in the regrowth/establishment period.")
            res.quality = {"notes": notes}
            return res

    try:
        comp = provider.get_composite(g.bounds, str(utm), date_from, date_to, p.resolution_m, seed_key=field_id)
    except Exception as e:  # noqa: BLE001  (provider failures must not kill the job)
        res.reason = f"Imagery unavailable: {e}"
        return res
    res.scenes = comp.scenes
    notes += comp.notes

    h, w = next(iter(comp.bands.values())).shape
    inside = features.geometry_mask([g], out_shape=(h, w), transform=comp.transform, invert=True)
    core_geom = g.buffer(-p.edge_buffer_m) if p.edge_buffer_m > 0 else g
    if core_geom.is_empty:
        core_geom = g
        notes.append("Field too small for edge buffer; buffer not applied.")
    core = features.geometry_mask([core_geom], out_shape=(h, w), transform=comp.transform, invert=True)

    ix = indices(comp.bands)
    ndvi = ix["NDVI"]
    finite = np.isfinite(ndvi)
    n_core = int(core.sum())
    if n_core == 0:
        res.reason = "No pixels fall inside the field."
        return res
    valid_fraction = float((core & finite).sum() / n_core)
    res.valid_fraction = valid_fraction
    res.quality = {
        "valid_fraction": round(valid_fraction, 3), "pixel_size_m": comp.resolution_m,
        "median_obs_count": (float(np.median(comp.obs_count[core & finite]))
                             if comp.obs_count is not None and (core & finite).any() else None),
        "synthetic": comp.synthetic, "notes": notes,
        "limitations": ("Pixel size and mixed edge pixels limit area precision; low NDVI is not proof of "
                        "missing stools; confidence is heuristic, not accuracy."),
    }
    if valid_fraction < p.min_valid_fraction:
        res.status = "insufficient_evidence"
        res.reason = (f"Only {valid_fraction:.0%} of the field has valid observations "
                      f"(minimum {p.min_valid_fraction:.0%}); cloud/shadow/no-data.")
        return res

    vals = ndvi[core & finite]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    thr = min(p.ndvi_abs_threshold, med - p.robust_z * max(mad, 0.02))
    use = core & finite
    cand = use & (ndvi < thr)
    water = cand & (ix["MNDWI"] > p.mndwi_water) & (ndvi < 0.3)
    bare = cand & ~water & (ndvi < p.bare_ndvi)
    sparse = cand & ~water & ~bare
    res.quality.update({"field_median_ndvi": round(med, 3), "ndvi_threshold_used": round(thr, 3),
                        "sparse_cane_area_ha_not_in_gap_total": round(float(sparse.sum()) * comp.resolution_m ** 2 / M2_PER_HA, 3)})

    gap_mask_all = bare | water
    close_px = max(0, int(round(p.merge_distance_m / comp.resolution_m / 2)))
    if close_px:
        gap_mask_all = ndi.binary_closing(gap_mask_all, structure=np.ones((3, 3)), iterations=close_px) & use
    mmu_px = max(1, int(round(p.min_mapping_unit_m2 / comp.resolution_m ** 2)))
    lab, n = ndi.label(gap_mask_all, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(gap_mask_all, lab, index=np.arange(1, n + 1))
        keep = np.isin(lab, np.where(sizes >= mmu_px)[0] + 1)
        gap_mask_all &= keep
    water_frac_lab = water

    detected = end.isoformat()
    depth = np.clip((thr - np.nan_to_num(ndvi, nan=thr)) / max(thr, 1e-6), 0, 1)
    pieces = []
    for geom_dict, val in features.shapes(gap_mask_all.astype("uint8"), mask=gap_mask_all,
                                          transform=comp.transform, connectivity=8):
        poly = shape(geom_dict).intersection(g)         # clip to THIS field only
        if poly.is_empty:
            continue
        for part in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly]):
            if part.geom_type != "Polygon" or part.area < p.min_mapping_unit_m2 * 0.5:
                continue
            pieces.append(part)

    prev_union = None
    if previous_gaps:
        prev_union = unary_union([to_crs_geom(pg, "EPSG:4326", utm) for pg in previous_gaps])

    total_m2 = 0.0
    for part in pieces:
        pm = features.geometry_mask([part], out_shape=(h, w), transform=comp.transform, invert=True) & gap_mask_all
        npx = max(int(pm.sum()), 1)
        is_water = float(water_frac_lab[pm].mean()) > 0.5 if pm.any() else False
        category = "Candidate gap - possible waterlogging" if is_water else "Candidate gap"
        conf = 0.5 * float(depth[pm].mean() if pm.any() else 0) + 0.3 * min(1.0, npx / (4 * mmu_px)) + 0.2 * valid_fraction
        if prev_union is None:
            change, first = "Uncertain (no baseline)", detected
        else:
            ov = part.intersection(prev_union).area / max(part.area, 1e-9)
            change, first = (("Persistent", previous_date or detected) if ov >= p.persistence_overlap
                             else ("New", detected))
        a_m2 = float(part.area)
        total_m2 += a_m2
        res.gaps.append({
            "geometry": to_crs_geom(part, utm, "EPSG:4326"), "area_m2": a_m2, "area_ha": a_m2 / M2_PER_HA,
            "pct_of_field": gap_percentage(a_m2 / M2_PER_HA, field_ha), "category": category,
            "confidence": round(min(conf, 1.0), 2), "change_status": change, "detected_at": detected,
            "first_observed": first,
            "method": f"{ALGO_VERSION}{' [SYNTHETIC]' if comp.synthetic else ''}",
            "proposed_action": ("Drainage or waterlogging investigation" if is_water else "Field verification"),
        })
    res.gaps.sort(key=lambda d: -d["area_m2"])
    res.status = "analysed"
    res.gap_area_ha = total_m2 / M2_PER_HA
    res.analysed_area_ha = field_ha * valid_fraction
    res.gap_pct = gap_percentage(res.gap_area_ha, res.analysed_area_ha)
    return res


def run_analysis(conn, farm_pk: int, version_pk: int, field_ids: list[str], provider: Provider,
                 date_from, date_to, params: Params | None = None, overlap_policy: str = "flag_only",
                 progress: Callable[[int, int, str], None] | None = None) -> int:
    """Analyse each selected field independently and persist results. Returns job id."""
    from . import db
    p = params or Params()
    all_fields = db.get_fields(conn, version_pk)
    sel = all_fields[all_fields["field_id"].isin(field_ids) & (all_fields["status"] != "rejected")]
    farm = conn.execute("SELECT * FROM farms WHERE id=?", (farm_pk,)).fetchone()
    geoms, overlaps = apply_overlap_policy(sel, overlap_policy)
    pdict = asdict(p); pdict["n_overlaps"] = len(overlaps); pdict["overlaps"] = overlaps
    job = db.create_job(conn, farm_pk, version_pk, list(sel["field_id"]), provider.name, provider.synthetic,
                        date_from, date_to, pdict, ALGO_VERSION, overlap_policy)
    try:
        total = len(sel)
        for k, (_, row) in enumerate(sel.iterrows(), 1):
            if progress:
                progress(k - 1, total, row["field_id"])
            props = {c: row[c] for c in ("crop", "variety", "planting_date", "last_harvest_date",
                                         "ratoon_number")}
            prev, prev_date = ([], None) if provider.synthetic else db.previous_gaps(conn, farm_pk, row["field_id"], job)
            res = analyse_field(row["field_id"], geoms[row["field_id"]], props, provider, date_from, date_to,
                                p, prev, prev_date)
            db.save_field_result(conn, job, version_pk, farm["farm_id"], farm["estate"], res, int(row["row_pk"]))
        if progress:
            progress(total, total, "done")
        db.finish_job(conn, job, "completed")
    except Exception as e:  # noqa: BLE001
        db.finish_job(conn, job, "failed", str(e))
        raise
    return job
