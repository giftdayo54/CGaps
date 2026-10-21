"""SQLite persistence (Estate -> Farm -> boundary version -> field -> job -> result -> gap).

Geometries are stored as GeoJSON text in EPSG:4326. Boundary versions are immutable:
re-importing creates a new version; results keep a reference to the exact version used.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping, shape

from .areas import area_ha, to_crs_geom, utm_crs_for

SCHEMA = """
CREATE TABLE IF NOT EXISTS farms(
  id INTEGER PRIMARY KEY, farm_id TEXT NOT NULL, farm_name TEXT, estate TEXT,
  created_at TEXT, UNIQUE(farm_id, estate));
CREATE TABLE IF NOT EXISTS boundary_versions(
  id INTEGER PRIMARY KEY, farm_pk INTEGER NOT NULL REFERENCES farms(id),
  version_no INTEGER NOT NULL, filename TEXT, source_crs TEXT, mapping_json TEXT,
  report_json TEXT, is_current INTEGER DEFAULT 1, created_at TEXT,
  UNIQUE(farm_pk, version_no));
CREATE TABLE IF NOT EXISTS fields(
  id INTEGER PRIMARY KEY, version_pk INTEGER NOT NULL REFERENCES boundary_versions(id),
  field_id TEXT NOT NULL, field_name TEXT, orig_field_id TEXT, is_outline INTEGER DEFAULT 0,
  area_ha REAL, status TEXT, issues TEXT, crop TEXT, variety TEXT, planting_date TEXT,
  last_harvest_date TEXT, ratoon_number TEXT, irrigation_system TEXT, src_index INTEGER,
  props_json TEXT, geom_json TEXT, UNIQUE(version_pk, field_id));
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, farm_pk INTEGER, version_pk INTEGER, field_ids TEXT, source TEXT,
  synthetic INTEGER, date_from TEXT, date_to TEXT, params_json TEXT, algo_version TEXT,
  overlap_policy TEXT, status TEXT, error TEXT, created_at TEXT, finished_at TEXT);
CREATE TABLE IF NOT EXISTS field_results(
  id INTEGER PRIMARY KEY, job_pk INTEGER NOT NULL REFERENCES jobs(id),
  field_row_pk INTEGER REFERENCES fields(id), field_id TEXT, status TEXT, reason TEXT,
  field_area_ha REAL, analysed_area_ha REAL, valid_fraction REAL,
  gap_area_ha REAL, gap_pct REAL, n_gaps INTEGER, quality_json TEXT, scenes_json TEXT);
CREATE TABLE IF NOT EXISTS gaps(
  id INTEGER PRIMARY KEY, result_pk INTEGER NOT NULL REFERENCES field_results(id),
  job_pk INTEGER, farm_id TEXT, estate TEXT, field_id TEXT, version_pk INTEGER, geom_json TEXT,
  area_m2 REAL, area_ha REAL, pct_of_field REAL, category TEXT, confidence REAL,
  change_status TEXT, detected_at TEXT, first_observed TEXT, method TEXT,
  verification_status TEXT DEFAULT 'Pending Verification', verified_area_ha REAL,
  proposed_action TEXT, notes TEXT, edited INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS reference_surveys(
  id INTEGER PRIMARY KEY, farm_pk INTEGER, version_pk INTEGER NOT NULL REFERENCES boundary_versions(id),
  field_id TEXT NOT NULL, observed_at TEXT, observer TEXT, method TEXT, fully_surveyed INTEGER DEFAULT 1,
  notes TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS reference_polygons(
  id INTEGER PRIMARY KEY, survey_pk INTEGER NOT NULL REFERENCES reference_surveys(id) ON DELETE CASCADE,
  geom_json TEXT);
CREATE TABLE IF NOT EXISTS validations(
  id INTEGER PRIMARY KEY, job_pk INTEGER NOT NULL REFERENCES jobs(id), created_at TEXT, params_json TEXT,
  pooled_json TEXT, ci_json TEXT, fields_json TEXT, skipped_json TEXT, warnings_json TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def data_dir() -> Path:
    p = Path(os.environ.get("CANEGAP_DATA_DIR", "data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    path = path or (data_dir() / "canegap.sqlite")
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def _scalar(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


# ------------------------------------------------------------------ imports
def save_import(conn, vr, filename: str, estate: str | None = None,
                target_farm_pk: int | None = None, outline_override: list | None = None) -> list[dict]:
    """Persist a ValidationResult. Returns [{farm_pk, version_pk, version_no, farm_id}].

    Features are grouped by farm_id (one version per farm). If `target_farm_pk` is given
    all features are attached to that farm as a new version (replace / Arrangement C).
    Existing outlines of the target farm are carried into the new version when the new
    file contains none.
    """
    gdf = vr.features
    out = []
    groups = [(None, gdf)] if target_farm_pk else list(gdf.groupby("farm_id"))
    for farm_id, sub in groups:
        if target_farm_pk:
            farm_pk = target_farm_pk
            row = conn.execute("SELECT * FROM farms WHERE id=?", (farm_pk,)).fetchone()
            farm_id = row["farm_id"]
        else:
            name = sub["farm_name"].dropna().iloc[0] if sub["farm_name"].notna().any() else farm_id
            est = estate or (_scalar(sub["estate"].dropna().iloc[0]) if sub["estate"].notna().any() else None)
            cur = conn.execute("SELECT id FROM farms WHERE farm_id=? AND IFNULL(estate,'')=IFNULL(?,'')",
                               (farm_id, est))
            r = cur.fetchone()
            if r:
                farm_pk = r["id"]
            else:
                farm_pk = conn.execute(
                    "INSERT INTO farms(farm_id,farm_name,estate,created_at) VALUES(?,?,?,?)",
                    (farm_id, name, est, now())).lastrowid
        vno = (conn.execute("SELECT MAX(version_no) m FROM boundary_versions WHERE farm_pk=?",
                            (farm_pk,)).fetchone()["m"] or 0) + 1
        carried = pd.DataFrame()
        if target_farm_pk and not sub["is_outline"].any():
            prev = conn.execute("SELECT id FROM boundary_versions WHERE farm_pk=? AND is_current=1",
                                (farm_pk,)).fetchone()
            if prev:
                carried = pd.read_sql_query(
                    "SELECT * FROM fields WHERE version_pk=? AND is_outline=1", conn, params=(prev["id"],))
        conn.execute("UPDATE boundary_versions SET is_current=0 WHERE farm_pk=?", (farm_pk,))
        vpk = conn.execute(
            "INSERT INTO boundary_versions(farm_pk,version_no,filename,source_crs,mapping_json,report_json,"
            "is_current,created_at) VALUES(?,?,?,?,?,?,1,?)",
            (farm_pk, vno, filename, vr.source_crs, json.dumps(vr.mapping),
             json.dumps(vr.report(), default=str), now())).lastrowid
        for _, r in sub.iterrows():
            g = r.geometry
            conn.execute(
                "INSERT INTO fields(version_pk,field_id,field_name,orig_field_id,is_outline,area_ha,status,issues,"
                "crop,variety,planting_date,last_harvest_date,ratoon_number,irrigation_system,src_index,"
                "props_json,geom_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (vpk, _scalar(r.field_id) if not r.is_outline else f"OUTLINE_{r.src_index}",
                 _scalar(r.field_name), _scalar(r.orig_field_id), int(r.is_outline),
                 None if pd.isna(r.area_ha) else float(r.area_ha), r.status, r.issues,
                 _scalar(r.crop), _scalar(r.variety), _scalar(r.planting_date),
                 _scalar(r.last_harvest_date), _scalar(r.ratoon_number), _scalar(r.irrigation_system),
                 int(r.src_index), r.props, json.dumps(mapping(g)) if g is not None else None))
        for _, r in carried.iterrows():
            conn.execute(
                "INSERT INTO fields(version_pk,field_id,field_name,is_outline,area_ha,status,issues,src_index,"
                "props_json,geom_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (vpk, f"OUTLINE_C{r['id']}", r["field_name"], 1, r["area_ha"], r["status"], r["issues"],
                 r["src_index"], r["props_json"], r["geom_json"]))
        conn.commit()
        out.append({"farm_pk": farm_pk, "version_pk": vpk, "version_no": vno, "farm_id": farm_id})
    return out


def list_farms(conn, search: str = "") -> pd.DataFrame:
    q = ("SELECT f.id farm_pk, f.estate, f.farm_id, f.farm_name, f.created_at, "
         "(SELECT MAX(version_no) FROM boundary_versions b WHERE b.farm_pk=f.id) versions, "
         "(SELECT COUNT(*) FROM fields x JOIN boundary_versions b ON b.id=x.version_pk "
         " WHERE b.farm_pk=f.id AND b.is_current=1 AND x.is_outline=0) n_fields, "
         "(SELECT ROUND(SUM(x.area_ha),2) FROM fields x JOIN boundary_versions b ON b.id=x.version_pk "
         " WHERE b.farm_pk=f.id AND b.is_current=1 AND x.is_outline=0 AND x.status!='rejected') area_ha "
         "FROM farms f")
    df = pd.read_sql_query(q, conn)
    if search:
        s = search.lower()
        mask = df.apply(lambda r: s in " ".join(str(v).lower() for v in r.values), axis=1)
        df = df[mask]
    return df.reset_index(drop=True)


def rename_farm(conn, farm_pk: int, new_name: str):
    conn.execute("UPDATE farms SET farm_name=? WHERE id=?", (new_name, farm_pk)); conn.commit()


def list_versions(conn, farm_pk: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT id version_pk, version_no, filename, source_crs, is_current, created_at "
        "FROM boundary_versions WHERE farm_pk=? ORDER BY version_no DESC", conn, params=(farm_pk,))


def current_version(conn, farm_pk: int) -> int | None:
    r = conn.execute("SELECT id FROM boundary_versions WHERE farm_pk=? AND is_current=1", (farm_pk,)).fetchone()
    return r["id"] if r else None


def get_fields(conn, version_pk: int, include_outlines: bool = False) -> gpd.GeoDataFrame:
    q = "SELECT * FROM fields WHERE version_pk=?" + ("" if include_outlines else " AND is_outline=0")
    df = pd.read_sql_query(q, conn, params=(version_pk,))
    geoms = [shape(json.loads(g)) if g else None for g in df.pop("geom_json")]
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326").rename(columns={"id": "row_pk"})


def update_field_metadata(conn, row_pk: int, **kw):
    allowed = {"field_name", "crop", "variety", "planting_date", "last_harvest_date",
               "ratoon_number", "irrigation_system"}
    kw = {k: v for k, v in kw.items() if k in allowed}
    if kw:
        conn.execute(f"UPDATE fields SET {', '.join(k + '=?' for k in kw)} WHERE id=?", (*kw.values(), row_pk))
        conn.commit()


def boundary_diff(conn, v_from: int, v_to: int) -> pd.DataFrame:
    a = get_fields(conn, v_from).set_index("field_id")
    b = get_fields(conn, v_to).set_index("field_id")
    rows = []
    for fid in sorted(set(a.index) | set(b.index)):
        if fid not in a.index:
            rows.append({"field_id": fid, "change": "added", "area_from_ha": None, "area_to_ha": b.at[fid, "area_ha"], "changed_area_ha": b.at[fid, "area_ha"]}); continue
        if fid not in b.index:
            rows.append({"field_id": fid, "change": "removed", "area_from_ha": a.at[fid, "area_ha"], "area_to_ha": None, "changed_area_ha": a.at[fid, "area_ha"]}); continue
        ga, gb = a.at[fid, "geometry"], b.at[fid, "geometry"]
        if ga is None or gb is None:
            continue
        utm = utm_crs_for(ga)
        sd = to_crs_geom(ga, "EPSG:4326", utm).symmetric_difference(to_crs_geom(gb, "EPSG:4326", utm)).area / 1e4
        rows.append({"field_id": fid, "change": "modified" if sd > 1e-4 else "unchanged",
                     "area_from_ha": a.at[fid, "area_ha"], "area_to_ha": b.at[fid, "area_ha"],
                     "changed_area_ha": round(sd, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- jobs/results
def create_job(conn, farm_pk, version_pk, field_ids, source, synthetic, date_from, date_to, params,
               algo_version, overlap_policy) -> int:
    jid = conn.execute(
        "INSERT INTO jobs(farm_pk,version_pk,field_ids,source,synthetic,date_from,date_to,params_json,"
        "algo_version,overlap_policy,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (farm_pk, version_pk, json.dumps(field_ids), source, int(synthetic), str(date_from), str(date_to),
         json.dumps(params), algo_version, overlap_policy, "running", now())).lastrowid
    conn.commit()
    return jid


def finish_job(conn, job_pk, status="completed", error=None):
    conn.execute("UPDATE jobs SET status=?, error=?, finished_at=? WHERE id=?", (status, error, now(), job_pk))
    conn.commit()


def save_field_result(conn, job_pk, version_pk, farm_id, estate, res, row_pk) -> int:
    rpk = conn.execute(
        "INSERT INTO field_results(job_pk,field_row_pk,field_id,status,reason,field_area_ha,analysed_area_ha,"
        "valid_fraction,gap_area_ha,gap_pct,n_gaps,quality_json,scenes_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (job_pk, row_pk, res.field_id, res.status, res.reason, res.field_area_ha, res.analysed_area_ha,
         res.valid_fraction, res.gap_area_ha, res.gap_pct, len(res.gaps), json.dumps(res.quality, default=str),
         json.dumps(res.scenes, default=str))).lastrowid
    for g in res.gaps:
        conn.execute(
            "INSERT INTO gaps(result_pk,job_pk,farm_id,estate,field_id,version_pk,geom_json,area_m2,area_ha,"
            "pct_of_field,category,confidence,change_status,detected_at,first_observed,method,proposed_action) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rpk, job_pk, farm_id, estate, res.field_id, version_pk, json.dumps(mapping(g["geometry"])),
             g["area_m2"], g["area_ha"], g["pct_of_field"], g["category"], g["confidence"], g["change_status"],
             g["detected_at"], g["first_observed"], g["method"], g["proposed_action"]))
    conn.commit()
    return rpk


def list_jobs(conn, farm_pk: int | None = None) -> pd.DataFrame:
    q = "SELECT * FROM jobs" + (" WHERE farm_pk=?" if farm_pk else "") + " ORDER BY id DESC"
    return pd.read_sql_query(q, conn, params=(farm_pk,) if farm_pk else ())


def job_results(conn, job_pk: int) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM field_results WHERE job_pk=? ORDER BY field_id", conn, params=(job_pk,))


def job_gaps(conn, job_pk: int) -> gpd.GeoDataFrame:
    df = pd.read_sql_query("SELECT * FROM gaps WHERE job_pk=?", conn, params=(job_pk,))
    geoms = [shape(json.loads(g)) for g in df.pop("geom_json")]
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326").rename(columns={"id": "gap_id"})


def previous_gaps(conn, farm_pk: int, field_id: str, before_job: int) -> tuple[list, str | None]:
    """Gap geometries of the latest earlier completed job that analysed this field."""
    r = conn.execute(
        "SELECT j.id, j.date_to FROM jobs j JOIN field_results fr ON fr.job_pk=j.id "
        "WHERE j.farm_pk=? AND fr.field_id=? AND fr.status='analysed' AND j.id<? AND j.synthetic=0 "
        "ORDER BY j.id DESC LIMIT 1", (farm_pk, field_id, before_job)).fetchone()
    if not r:
        return [], None
    rows = conn.execute("SELECT geom_json FROM gaps WHERE job_pk=? AND field_id=?", (r["id"], field_id)).fetchall()
    return [shape(json.loads(x["geom_json"])) for x in rows], r["date_to"]


def update_gaps(conn, edits: pd.DataFrame):
    """edits columns: gap_id, verification_status, verified_area_ha, proposed_action, notes."""
    for _, r in edits.iterrows():
        conn.execute(
            "UPDATE gaps SET verification_status=?, verified_area_ha=?, proposed_action=?, notes=?, edited=1 "
            "WHERE id=?",
            (r["verification_status"], None if pd.isna(r["verified_area_ha"]) else float(r["verified_area_ha"]),
             r["proposed_action"], r["notes"], int(r["gap_id"])))
    conn.commit()


def delete_gap(conn, gap_id: int):
    conn.execute("DELETE FROM gaps WHERE id=?", (gap_id,)); conn.commit()


# ------------------------------------------------------------- validation data
def add_survey(conn, farm_pk, version_pk, field_id, observed_at, observer, method, fully_surveyed, notes, polygons) -> int:
    spk = conn.execute(
        "INSERT INTO reference_surveys(farm_pk,version_pk,field_id,observed_at,observer,method,fully_surveyed,notes,"
        "created_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (farm_pk, version_pk, field_id, str(observed_at), observer, method, int(fully_surveyed), notes, now())).lastrowid
    for g in polygons:
        conn.execute("INSERT INTO reference_polygons(survey_pk,geom_json) VALUES(?,?)", (spk, json.dumps(mapping(g))))
    conn.commit()
    return spk


def list_surveys(conn, version_pk: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT s.id survey_pk, s.version_pk, s.field_id, s.observed_at, s.observer, s.method, s.fully_surveyed, "
        "s.notes, (SELECT COUNT(*) FROM reference_polygons p WHERE p.survey_pk=s.id) n_polygons "
        "FROM reference_surveys s WHERE s.version_pk=? ORDER BY s.id DESC", conn, params=(version_pk,))


def survey_polygon_json(conn, survey_pk: int) -> list[str]:
    return [r["geom_json"] for r in conn.execute("SELECT geom_json FROM reference_polygons WHERE survey_pk=?", (survey_pk,))]


def survey_polygons_gdf(conn, version_pk: int) -> gpd.GeoDataFrame:
    df = pd.read_sql_query(
        "SELECT p.id, s.field_id, p.geom_json FROM reference_polygons p JOIN reference_surveys s ON s.id=p.survey_pk "
        "WHERE s.version_pk=?", conn, params=(version_pk,))
    geoms = [shape(json.loads(g)) for g in df.pop("geom_json")]
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")


def delete_survey(conn, survey_pk: int):
    conn.execute("DELETE FROM reference_polygons WHERE survey_pk=?", (survey_pk,))
    conn.execute("DELETE FROM reference_surveys WHERE id=?", (survey_pk,)); conn.commit()


def save_validation(conn, job_pk: int, ev: dict) -> int:
    f = ev["fields"]
    vid = conn.execute(
        "INSERT INTO validations(job_pk,created_at,params_json,pooled_json,ci_json,fields_json,skipped_json,warnings_json)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (job_pk, now(), json.dumps(ev["params"]), json.dumps(ev["pooled"], default=float), json.dumps(ev["ci95"]),
         f.to_json(orient="records"), ev["skipped"].to_json(orient="records"), json.dumps(ev["warnings"]))).lastrowid
    conn.commit()
    return vid


def list_validations(conn, job_pk: int) -> pd.DataFrame:
    return pd.read_sql_query("SELECT id, created_at, params_json, pooled_json, warnings_json FROM validations "
                             "WHERE job_pk=? ORDER BY id DESC", conn, params=(job_pk,))
