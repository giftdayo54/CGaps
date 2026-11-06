"""CaneGap Intelligence - Streamlit app.  Run:  streamlit run app.py"""
from __future__ import annotations

import io
import json
import os
from datetime import date, timedelta

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from canegap import ALGO_VERSION, aggregate, db, exports, interventions as iv, ml, reports, validation
from canegap.geojson_import import (MAPPING_KEYS, CRSNeedsConfirmation, ImportError_, guess_mapping,
                                    parse_geojson, validate_geojson)
from canegap.pipeline import Params, run_analysis
from canegap.providers import CDSEProvider, DemoProvider
from canegap.ui_helpers import base_map, selection_map, results_map

st.set_page_config(page_title="CaneGap Intelligence", page_icon="🌾", layout="wide")

VERIF = ["Pending Verification", "Inspected", "Approved", "Scheduled", "Completed", "Rejected"]


@st.cache_resource
def get_conn():
    return db.connect()


conn = get_conn()
S = st.session_state
S.setdefault("sel_ids", set())
S.setdefault("farm_pk", None)
S.setdefault("vr", None)


def banner_unvalidated():
    st.warning("**Unvalidated screening tool.** Candidate gaps are model predictions, not confirmed missing cane. "
               "No accuracy has been measured; verify in the field before operational decisions.", icon="⚠️")


# =========================================================================== sidebar
st.sidebar.title("🌾 CaneGap Intelligence")
page = st.sidebar.radio("Section", ["Farm and Field Explorer", "Results and Review", "Interventions", "Validation", "Machine Learning", "Method and Limits"])
basemap = st.sidebar.selectbox("Basemap", ["OpenStreetMap", "Esri World Imagery"])
st.sidebar.caption(f"Algorithm: {ALGO_VERSION}")


# =========================================================================== explorer
def import_panel():
    st.subheader("Browse and Import GeoJSON")
    st.caption("Opens your browser's normal file chooser (files on your own computer). "
               "Supports FeatureCollections of Polygon/MultiPolygon fields.")
    up = st.file_uploader("Browse and Import GeoJSON", type=["geojson", "json"], label_visibility="collapsed")
    if not up:
        st.info("Choose a .geojson or .json file. Synthetic samples are in the `sample_data/` folder.")
        return
    raw = up.getvalue()
    try:
        gj = parse_geojson(raw)
    except ImportError_ as e:
        st.error(f"Cannot read this file: {e}")
        return
    cols = sorted({k for f in gj["features"] for k in (f.get("properties") or {})})
    st.write(f"**{len(gj['features'])} features**, attributes: {', '.join(cols) or 'none'}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Coordinate system**")
        crs_in = st.text_input("Source CRS (leave blank for WGS84 lon/lat or a CRS declared in the file)",
                               placeholder="e.g. EPSG:32736")
        st.markdown("**Where should it go?**")
        farms = db.list_farms(conn)
        mode = st.radio("Import as", ["New farm(s) from this file", "New version of an existing farm",
                                      "Fields for an existing farm (outline imported earlier)"])
        target = None
        if mode != "New farm(s) from this file":
            if farms.empty:
                st.warning("No farms saved yet.")
            else:
                label = {f"{r.farm_name} ({r.farm_id})": int(r.farm_pk) for r in farms.itertuples()}
                target = label[st.selectbox("Existing farm", list(label))]
        estate = st.text_input("Estate (optional)")
        id_policy = st.radio("Missing / duplicate field IDs", ["generate_missing", "strict"],
                             format_func=lambda x: "Generate unique IDs (original kept)" if x == "generate_missing"
                             else "Reject those features")
    with c2:
        st.markdown("**Attribute mapping**")
        guess = guess_mapping(cols)
        mapping = {}
        for k in MAPPING_KEYS:
            opts = ["(none)"] + cols
            idx = opts.index(guess[k]) if guess[k] in opts else 0
            v = st.selectbox(k.replace("_", " ").title(), opts, index=idx, key=f"map_{k}")
            mapping[k] = None if v == "(none)" else v

    st.markdown("**Farm outline features (Arrangement B)** - features that are the parent farm boundary, not fields")
    o1, o2 = st.columns(2)
    outline_rows = []
    with o1:
        ocol = st.selectbox("Identify outlines by attribute", ["(none)"] + cols)
        if ocol != "(none)":
            vals = sorted({str((f.get("properties") or {}).get(ocol)) for f in gj["features"]})
            ov = st.multiselect("Values marking an outline", vals)
            outline_rows += [i for i, f in enumerate(gj["features"]) if str((f.get("properties") or {}).get(ocol)) in ov]
    with o2:
        manual = st.multiselect("...or pick features by index", list(range(len(gj["features"]))))
        outline_rows = sorted(set(outline_rows) | set(manual))
    if mode.startswith("Fields for") and not outline_rows:
        st.caption("The farm's existing outline is carried into the new version automatically.")

    if st.button("Validate", type="primary"):
        try:
            S.vr = validate_geojson(raw, mapping=mapping, source_crs=crs_in.strip() or None,
                                    outline_rows=outline_rows, id_policy=id_policy,
                                    default_farm_name=os.path.splitext(up.name)[0])
            S.vr_name, S.vr_target, S.vr_estate = up.name, target, estate.strip() or None
        except CRSNeedsConfirmation as e:
            S.vr = None
            st.error(str(e))
        except ImportError_ as e:
            S.vr = None
            st.error(f"Import failed: {e}")

    vr = S.vr
    if vr is not None and S.get("vr_name") == up.name:
        s = vr.summary
        a, b, c, d, e = st.columns(5)
        a.metric("Fields", s["n_fields"]); b.metric("Accepted", s["accepted"]); c.metric("Repaired", s["repaired"])
        d.metric("Warnings", s["warning"]); e.metric("Rejected", s["rejected"])
        st.caption(f"Source CRS: {vr.source_crs} | usable area {s['total_usable_area_ha']:,} ha | "
                   f"outlines: {s['n_outlines']}")
        for msg in vr.dataset_issues:
            st.info(msg)
        if s["overlaps"]:
            st.warning(f"{len(s['overlaps'])} overlapping field pair(s) - you will choose an overlap policy "
                       f"before analysis.")
        show = vr.features.drop(columns=["geometry", "props"])[
            ["src_index", "is_outline", "field_id", "field_name", "farm_id", "status", "area_ha", "issues"]]
        st.dataframe(show, width="stretch", hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("Validation report (JSON)", json.dumps(vr.report(), default=str, indent=1),
                           "validation_report.json", "application/json")
        d2.download_button("Validation report (CSV)", vr.report_csv(), "validation_report.csv", "text/csv")
        if s["rejected"]:
            st.caption("Rejected features are NOT discarded silently: they are listed above with reasons "
                       "and kept in the stored import report.")
        if vr.usable[~vr.usable.is_outline].empty and not outline_rows:
            st.error("No usable field geometries - nothing to save.")
        elif st.button("Save to boundary library"):
            res = db.save_import(conn, vr, S.vr_name, estate=S.vr_estate, target_farm_pk=S.vr_target)
            st.success("Saved: " + "; ".join(f"farm {r['farm_id']} version {r['version_no']}" for r in res))
            S.farm_pk = res[0]["farm_pk"]; S.sel_ids = set(); S.vr = None


def library_panel():
    st.subheader("Saved boundary library")
    q = st.text_input("Search farms (name, ID, estate, date)", key="farm_search")
    farms = db.list_farms(conn, q)
    if farms.empty:
        st.info("No saved farms yet. Import a GeoJSON on the 'Import' tab, or the sample in sample_data/.")
        return None
    st.dataframe(farms, width="stretch", hide_index=True)
    label = {f"{r.estate + ' / ' if r.estate else ''}{r.farm_name} ({r.farm_id})": int(r.farm_pk) for r in farms.itertuples()}
    keys = list(label)
    cur = next((i for i, k in enumerate(keys) if label[k] == S.farm_pk), 0)
    choice = st.selectbox("Selected farm", keys, index=cur)
    if label[choice] != S.farm_pk:
        S.farm_pk = label[choice]; S.sel_ids = set()
    farm_pk = S.farm_pk
    with st.expander("Rename farm, versions and import history"):
        new = st.text_input("Rename farm", value=farms[farms.farm_pk == farm_pk].iloc[0].farm_name)
        if st.button("Save name"):
            db.rename_farm(conn, farm_pk, new); st.rerun()
        vers = db.list_versions(conn, farm_pk)
        st.dataframe(vers, width="stretch", hide_index=True)
        if len(vers) > 1:
            a = st.selectbox("Compare version", vers.version_no.tolist(), index=1)
            b = st.selectbox("with version", vers.version_no.tolist(), index=0)
            if st.button("Show boundary differences"):
                pk = dict(zip(vers.version_no, vers.version_pk))
                st.dataframe(db.boundary_diff(conn, pk[a], pk[b]), width="stretch", hide_index=True)
    return farm_pk


def select_and_run(farm_pk: int):
    vpk = db.current_version(conn, farm_pk)
    fields = db.get_fields(conn, vpk)
    outlines = db.get_fields(conn, vpk, include_outlines=True)
    outlines = outlines[outlines.is_outline == 1]
    farm = conn.execute("SELECT * FROM farms WHERE id=?", (farm_pk,)).fetchone()
    st.subheader(f"{farm['farm_name']} - field selection")

    q = st.text_input("Filter fields (ID, name, variety, crop...)", key="field_filter")
    view = fields.drop(columns="geometry")
    if q:
        m = view.apply(lambda r: q.lower() in " ".join(str(v).lower() for v in r.values), axis=1)
        view = view[m]
    b1, b2, _ = st.columns([1, 1, 6])
    if b1.button("Select all"):
        S.sel_ids = set(fields[fields.status != "rejected"].field_id); S.editor_ver = S.get("editor_ver", 0) + 1
    if b2.button("Clear"):
        S.sel_ids = set(); S.editor_ver = S.get("editor_ver", 0) + 1

    tbl = view[["field_id", "field_name", "area_ha", "crop", "variety", "ratoon_number", "last_harvest_date",
                "planting_date", "status"]].copy()
    tbl.insert(0, "select", tbl.field_id.isin(S.sel_ids))
    edited = st.data_editor(
        tbl, hide_index=True, width="stretch", key=f"fields_{farm_pk}_{S.get('editor_ver', 0)}",
        disabled=[c for c in tbl.columns if c != "select"],
        column_config={"area_ha": st.column_config.NumberColumn("Area (ha)", format="%.2f"),
                       "status": st.column_config.TextColumn("Boundary validation")})
    shown = set(tbl.field_id)
    chosen = set(edited[edited.select].field_id)
    S.sel_ids = (S.sel_ids - shown) | chosen
    rejected = set(fields[fields.status == "rejected"].field_id)
    S.sel_ids -= rejected
    sel = fields[fields.field_id.isin(S.sel_ids)]
    st.markdown(f"**{len(sel)} of {len(fields)} fields selected - combined area {sel.area_ha.sum():,.2f} ha**")
    st_folium(selection_map(fields, outlines, set(S.sel_ids), basemap), height=430, width=None,
              returned_objects=[], key=f"selmap_{farm_pk}_{len(S.sel_ids)}")

    with st.expander("Edit field metadata"):
        fid = st.selectbox("Field", fields.field_id.tolist(), key="meta_field")
        r = fields[fields.field_id == fid].iloc[0]
        vals = {k: st.text_input(k, value=r[k] or "", key=f"meta_{k}_{fid}")
                for k in ["field_name", "crop", "variety", "ratoon_number", "last_harvest_date", "planting_date",
                          "irrigation_system"]}
        if st.button("Save metadata"):
            db.update_field_metadata(conn, int(r.row_pk), **{k: (v or None) for k, v in vals.items()})
            st.success("Saved."); st.rerun()

    # ---------------------------------------------------------------- run analysis
    st.divider()
    st.subheader("Run Gap Analysis")
    report = json.loads(conn.execute("SELECT report_json FROM boundary_versions WHERE id=?", (vpk,)).fetchone()[0])
    overlaps = report["summary"].get("overlaps", [])
    policy = "flag_only"
    if overlaps:
        st.warning(f"{len(overlaps)} overlapping field pair(s) in this boundary version. Choose how to handle them.")
        policy = st.radio("Overlap policy", ["first_wins", "exclude", "flag_only"],
                          format_func={"first_wins": "Assign overlap to the earlier field",
                                       "exclude": "Exclude overlap from all involved fields",
                                       "flag_only": "Flag only (totals may double count)"}.get)
    c1, c2, c3 = st.columns(3)
    src = c1.radio("Satellite source", ["Demo (synthetic data)", "Sentinel-2 L2A (Copernicus CDSE)"])
    today = date.today()
    d_from = c2.date_input("From", today - timedelta(days=60))
    d_to = c3.date_input("To", today)
    prov = DemoProvider()
    can_run = len(sel) > 0
    if src.startswith("Sentinel"):
        prov = CDSEProvider()
        if not prov.configured:
            st.error("CDSE credentials not found. Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (see README), "
                     "or use the demo source.")
            can_run = False
        else:
            st.caption("Real Sentinel-2 retrieval - provider code is untested against the live service.")
    else:
        st.info("Demo mode generates **synthetic** imagery for testing. Results are not real observations.")
    st.markdown("**Detection method**")
    models_df = db.list_models(conn)
    method_opts = ["Threshold baseline (transparent rules, Sentinel-2)"] + [
        f"Random Forest #{r.id}: {r.name} [{r.feature_set}]{' - SYNTHETIC' if r.synthetic else ''}{'' if r.cv_validated else ' - no cross-validation'}"
        for r in models_df.itertuples()]
    mchoice = st.selectbox("Method", method_opts, label_visibility="collapsed")
    bundle = None
    if mchoice.startswith("Random Forest"):
        mid = int(mchoice.split("#")[1].split(":")[0])
        row = db.get_model_row(conn, mid)
        bundle = ml.load_bundle(row["path"])
        if bool(bundle.synthetic) != bool(prov.synthetic):
            st.error("This model was trained on " + ("SYNTHETIC" if bundle.synthetic else "real") + " imagery and cannot be "
                     "used with the selected source.")
            can_run = False
        if not row["cv_validated"]:
            st.warning("This model has no cross-validation (too few surveyed fields): its quality is unknown.")
        else:
            st.caption("Cross-validated internally on held-out fields; still validate on new fields/seasons before relying on it.")
        if bundle.needs_s1 and src.startswith("Sentinel"):
            od = st.selectbox("Sentinel-1 orbit direction", ["(mixed - not recommended)", "ASCENDING", "DESCENDING"], index=1)
            prov.orbit_direction = None if od.startswith("(") else od
    with st.expander("Detection parameters (stored with every job)"):
        p = Params()
        p.ndvi_abs_threshold = st.slider("Absolute NDVI threshold", 0.2, 0.7, p.ndvi_abs_threshold, 0.01)
        p.robust_z = st.slider("Robust z below field median", 1.0, 6.0, p.robust_z, 0.5)
        p.min_mapping_unit_m2 = st.number_input("Minimum mapping unit (m²)", 100.0, 10000.0, p.min_mapping_unit_m2, 100.0)
        p.merge_distance_m = st.number_input("Merge distance (m)", 0.0, 50.0, p.merge_distance_m, 10.0)
        p.edge_buffer_m = st.number_input("Edge buffer (m)", 0.0, 50.0, p.edge_buffer_m, 5.0)
        p.min_valid_fraction = st.slider("Minimum valid-observation fraction", 0.1, 1.0, p.min_valid_fraction, 0.05)
        p.harvest_window_days = st.number_input("Recent-harvest window (days)", 0, 365, p.harvest_window_days)
        p.planting_window_days = st.number_input("New-planting window (days)", 0, 365, p.planting_window_days)
    if st.button("Run Gap Analysis", type="primary", disabled=not can_run):
        bar = st.progress(0.0, text="Starting...")
        def cb(i, n, fid):
            bar.progress(i / max(n, 1), text=f"Analysing field {fid} ({i}/{n})")
        try:
            job = run_analysis(conn, farm_pk, vpk, sorted(S.sel_ids), prov, d_from, d_to, p, policy, cb, model=bundle)
            S.job_id = job
            bar.progress(1.0, text="Done")
            st.success(f"Job {job} completed. Open 'Results and Review' in the sidebar.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Analysis failed: {e}")


def explorer():
    st.title("Farm and Field Explorer")
    t1, t2 = st.tabs(["Import", "Library, selection and analysis"])
    with t1:
        import_panel()
    with t2:
        farm_pk = library_panel()
        if farm_pk:
            st.divider()
            select_and_run(farm_pk)


# =========================================================================== results
def results():
    st.title("Results and Review")
    jobs = db.list_jobs(conn, S.farm_pk) if S.farm_pk else db.list_jobs(conn)
    if jobs.empty:
        st.info("No analysis jobs yet. Run one from the Explorer.")
        return
    labels = {f"Job {r.id} - {r.created_at[:16]} - {r.source} - {r.status}": int(r.id) for r in jobs.itertuples()}
    keys = list(labels)
    cur = next((i for i, k in enumerate(keys) if labels[k] == S.get("job_id")), 0)
    job_pk = labels[st.selectbox("Analysis job", keys, index=cur)]
    job = jobs[jobs.id == job_pk].iloc[0]
    synthetic = bool(job.synthetic)
    if synthetic:
        st.error("SYNTHETIC DEMO DATA - these are NOT satellite observations or real results.", icon="🧪")
    banner_unvalidated()
    if job.status != "completed":
        st.error(f"Job status: {job.status}. {job.error or ''}")
    res = db.job_results(conn, job_pk)
    gaps = db.job_gaps(conn, job_pk)
    fields = db.get_fields(conn, int(job.version_pk))
    params = json.loads(job.params_json)
    cur_v = db.current_version(conn, int(job.farm_pk))
    if cur_v != int(job.version_pk):
        st.info("This job used an earlier boundary version; the farm now has a newer one. "
                "Compare versions in the Explorer library.")

    s = aggregate.farm_summary(res, job.overlap_policy, params.get("n_overlaps", 0))
    cols = st.columns(5)
    cols[0].metric("Fields analysed", f"{s['n_analysed']} / {s['n_fields']}")
    cols[1].metric("Analysed area (ha)", f"{s['analysed_area_ha']:,.1f}")
    cols[2].metric("Candidate gap (ha)", f"{s['candidate_gap_area_ha']:,.2f}")
    cols[3].metric("Farm gap %", f"{s['gap_pct']:.2f}%")
    cols[4].metric("Gap polygons", s["n_gap_polygons"])
    st.caption(f"Method: **{job.algo_version}**" + (" (Random Forest; scores are probabilities, not accuracy)" if "model" in params else ""))
    if "model" in params and not params["model"].get("cv_validated"):
        st.warning("This job used a model with no cross-validation; its accuracy is unknown.")
    st.caption(f"Not assessed: {s['n_insufficient_evidence']} insufficient evidence, {s['n_context_excluded']} "
               f"recently harvested/newly planted, {s['n_unable']} unable. Farm % = total gap ha / total analysed ha "
               f"(area-weighted).")
    if s["double_count_risk"]:
        st.warning("Overlapping fields were only flagged, not resolved: farm totals may double count overlap.")

    st.subheader("Fields")
    show = res[["field_id", "status", "reason", "field_area_ha", "analysed_area_ha", "valid_fraction",
                "gap_area_ha", "gap_pct", "n_gaps"]].copy()
    st.dataframe(show.sort_values("gap_area_ha", ascending=False), width="stretch", hide_index=True,
                 column_config={"gap_pct": st.column_config.NumberColumn("Gap %", format="%.2f"),
                                "gap_area_ha": st.column_config.NumberColumn("Gap (ha)", format="%.3f"),
                                "field_area_ha": st.column_config.NumberColumn("Field (ha)", format="%.2f"),
                                "analysed_area_ha": st.column_config.NumberColumn("Analysed (ha)", format="%.2f"),
                                "valid_fraction": st.column_config.ProgressColumn("Valid obs.", min_value=0, max_value=1)})

    focus = st.selectbox("Map focus", ["All fields"] + res.field_id.tolist())
    fid = None if focus == "All fields" else focus
    st_folium(results_map(fields, res, gaps, basemap, fid), height=500, width=None, returned_objects=[],
              key=f"resmap_{job_pk}_{focus}_{len(gaps)}")
    st.caption("Layers: field outlines (green <2% / amber 2-5% / red >5% gap; grey = insufficient evidence; "
               "blue = harvested/new planting) and red candidate-gap polygons (model predictions, unverified).")

    if fid:
        r = res[res.field_id == fid].iloc[0]
        st.markdown(f"**{fid}** - {r.status}. {r.reason or ''}")
        q = json.loads(r.quality_json or "{}")
        if q:
            st.json(q, expanded=False)

    st.subheader("Review and verification")
    g = gaps if not fid else gaps[gaps.field_id == fid]
    if g.empty:
        st.write("No candidate gap polygons to review.")
    else:
        ed = g[["gap_id", "field_id", "area_ha", "pct_of_field", "category", "confidence", "change_status",
                "verification_status", "verified_area_ha", "proposed_action", "notes"]].copy()
        out = st.data_editor(
            ed, hide_index=True, width="stretch", key=f"gapedit_{job_pk}_{focus}",
            disabled=["gap_id", "field_id", "area_ha", "pct_of_field", "category", "confidence", "change_status"],
            column_config={
                "verification_status": st.column_config.SelectboxColumn("Verification", options=VERIF),
                "verified_area_ha": st.column_config.NumberColumn("Field-verified area (ha)", format="%.3f"),
                "confidence": st.column_config.NumberColumn("Heuristic score", format="%.2f",
                                                            help="Not a measured accuracy."),
                "area_ha": st.column_config.NumberColumn("Predicted area (ha)", format="%.4f")})
        c1, c2 = st.columns([1, 1])
        if c1.button("Save review edits"):
            db.update_gaps(conn, out); st.success("Saved."); st.rerun()
        del_id = c2.selectbox("Delete a false-positive polygon", [None] + g.gap_id.tolist())
        if del_id and c2.button("Delete polygon"):
            db.delete_gap(conn, int(del_id)); st.rerun()
        st.caption("Predicted, field-verified, proposed and treated areas are kept as separate fields. "
                   "Deleting removes the polygon from totals shown on re-run exports; the job's stored "
                   "field totals still reflect the original prediction.")

    st.subheader("Exports")
    gf = exports.gaps_with_flag(gaps, synthetic)
    e1, e2, e3, e4, e5, e6 = st.columns(6)
    fn = f"canegap_job{job_pk}"
    if len(gf):
        e1.download_button("GeoJSON", exports.to_geojson(gf), f"{fn}_gaps.geojson", "application/geo+json")
        e2.download_button("Shapefile (zip)", exports.to_shapefile_zip(gf), f"{fn}_gaps_shp.zip")
        e3.download_button("GeoPackage", exports.to_gpkg(gf, fields), f"{fn}.gpkg")
        e4.download_button("Gaps CSV", exports.to_csv(gf), f"{fn}_gaps.csv", "text/csv")
    st.session_state.setdefault("pdf_cache", {})
    p1, p2, _ = st.columns([1, 1, 4])
    if p1.button("Build farm PDF report"):
        S.pdf_cache[(job_pk, "farm")] = reports.farm_report(conn, job_pk)
    if (job_pk, "farm") in S.pdf_cache:
        p1.download_button("Download farm PDF", S.pdf_cache[(job_pk, "farm")], f"{fn}_farm_report.pdf", "application/pdf")
    if fid and p2.button(f"Build field PDF ({fid})"):
        S.pdf_cache[(job_pk, fid)] = reports.field_report(conn, job_pk, fid)
    if fid and (job_pk, fid) in S.pdf_cache:
        p2.download_button(f"Download field PDF ({fid})", S.pdf_cache[(job_pk, fid)], f"{fn}_{fid}_report.pdf", "application/pdf")
    e5.download_button("Fields CSV", exports.to_csv(res), f"{fn}_fields.csv", "text/csv")
    notes = ["CaneGap Intelligence export", f"Job {job_pk}; algorithm {job.algo_version}; boundary version id {job.version_pk}",
             "UNVALIDATED SCREENING OUTPUT - candidate gaps require field verification.",
             "SYNTHETIC DEMO DATA - NOT SATELLITE IMAGERY" if synthetic else f"Source: {job.source}",
             f"Dates: {job.date_from} to {job.date_to}", "Parameters: " + job.params_json]
    e6.download_button("Excel report", exports.to_xlsx(res, gf, s, notes), f"{fn}.xlsx")


# =========================================================================== method
def method():
    st.title("Method and Limits")
    banner_unvalidated()
    st.markdown(f"""
**Workflow.** Import boundaries -> validate -> select fields -> each selected field is processed on its own:
its geometry masks the imagery, features are computed, candidate gaps are polygonised, clipped to that field only,
and measured in hectares in a local UTM projection. Farm totals are summed from field results.

**Baseline classifier ({ALGO_VERSION}).** Within a field a pixel is a candidate gap when NDVI is below an absolute
threshold *and* well below that field's own median (robust z-score). Very low NDVI = "no detectable cane";
low NDVI with a high MNDWI = "possible waterlogging". Moderately low NDVI is reported as *sparse cane* and is
**not** counted in gap hectares. Fields harvested or planted within configurable windows are reported as
"context excluded" instead of being flagged for gaps. Edge pixels are excluded to reduce mixed-pixel false alarms.

**Definitions.** Field gap % = candidate-gap area / analysed field area x 100, where analysed area is field area
x fraction of pixels with valid observations. Farm gap % = total gap ha / total analysed ha (not an average of percentages).

**Limits you should know.** Sentinel-2 is 10 m (red-edge and SWIR bands are native 20 m); gap areas are quantised
by pixels. Low vegetation index does not prove missing stools (weeds, stress, lodging, shadow, harvest trash, tractor
tracks and drains can all look similar). The heuristic score is not accuracy. Until compared with field-verified
polygons this tool is a screening aid only. Random Forest models (Machine Learning section) can use Sentinel-2, Sentinel-1 (VV/VH dB, VH-VV, RVI, optional temporal change) or both; they are trained only on field-verified polygons and compared with this baseline under field-held-out cross-validation. Radar backscatter depends on moisture, roughness and viewing geometry, so S1 alone is not assumed to separate cane from weeds or bare ground.

**Demo mode.** Synthetic imagery is generated so you can test the workflow without credentials. It is not satellite data
and every output made from it is labelled as such.
""")


def interventions_page():
    st.title("Intervention management")
    st.caption("Plan follow-up for candidate gaps. When the cause is unknown the default action is field verification. "
               "Proposed, estimated and actual values are stored separately.")
    jobs = db.list_jobs(conn, S.farm_pk) if S.farm_pk else db.list_jobs(conn)
    farms = db.list_farms(conn)
    if farms.empty:
        st.info("Import a farm first."); return
    flabel = {f"{r.farm_name} ({r.farm_id})": int(r.farm_pk) for r in farms.itertuples()}
    fkeys = list(flabel)
    fcur = next((i for i, k in enumerate(fkeys) if flabel[k] == S.farm_pk), 0)
    farm_pk = flabel[st.selectbox("Farm", fkeys, index=fcur)]
    farm = conn.execute("SELECT * FROM farms WHERE id=?", (farm_pk,)).fetchone()
    t1, t2, t3, t4 = st.tabs(["Worklist", "Create from analysis", "Add / edit", "Cost estimate"])

    with t1:
        f = st.columns(4)
        fstat = f[0].selectbox("Status", [""] + db.STATUSES)
        fpri = f[1].selectbox("Priority", [""] + db.PRIORITIES)
        ffield = f[2].text_input("Field ID")
        dr = f[3].date_input("Target date range", value=(), help="Leave empty for all")
        d_from, d_to = (dr[0], dr[1]) if isinstance(dr, tuple) and len(dr) == 2 else (None, None)
        df = db.list_interventions(conn, farm_pk, ffield or None, fstat or None, fpri or None, d_from, d_to)
        if df.empty:
            st.info("No interventions match.")
        else:
            sm = iv.summary(df)
            a, b, c, d = st.columns(4)
            a.metric("Items", sm["n"]); b.metric("Proposed area (ha)", f"{sm['proposed_area_ha']:.2f}")
            c.metric("Actual treated (ha)", f"{sm['actual_treated_area_ha']:.2f}"); d.metric("Actual cost", f"{sm['actual_cost']:,.0f}")
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button("Worklist CSV", exports.to_csv(df), "worklist.csv", "text/csv")

    with t2:
        done = jobs[jobs.farm_pk == farm_pk] if len(jobs) else jobs
        if done.empty:
            st.info("Run an analysis for this farm first.")
        else:
            lab = {f"Job {r.id} - {r.created_at[:16]} - {r.source}": int(r.id) for r in done.itertuples()}
            jp = lab[st.selectbox("Analysis job", list(lab))]
            c1, c2, c3, c4 = st.columns(4)
            min_a = c1.number_input("Minimum gap area (ha)", 0.0, 10.0, 0.05, 0.01)
            min_c = c2.slider("Minimum heuristic score", 0.0, 1.0, 0.0, 0.05)
            who = c3.text_input("Assign to (team/supervisor)")
            tgt = c4.date_input("Target completion", date.today() + timedelta(days=14))
            if st.button("Create worklist from candidate gaps", type="primary"):
                n, sk = iv.worklist_from_gaps(conn, jp, min_a, min_c, who or None, tgt)
                st.success(f"Created {n} intervention(s); {len(sk)} skipped (already open or invalid).")
            st.caption("Priority is a configurable inspection-ranking heuristic (area and score), not a diagnosis.")

    with t3:
        gaps_df = pd.read_sql_query("SELECT id, field_id, area_ha FROM gaps WHERE farm_id=? ORDER BY id DESC LIMIT 500",
                                    conn, params=(farm["farm_id"],))
        vpk = db.current_version(conn, farm_pk)
        fields = db.get_fields(conn, vpk)
        with st.form("new_iv"):
            st.markdown("**New intervention**")
            c1, c2, c3 = st.columns(3)
            fid = c1.selectbox("Field", fields.field_id.tolist())
            gid = c2.selectbox("Gap (optional)", [None] + gaps_df[gaps_df.field_id == fid].id.tolist())
            action = c3.selectbox("Action", db.ACTIONS)
            c1, c2, c3 = st.columns(3)
            pri = c1.selectbox("Priority", db.PRIORITIES, index=1)
            who = c2.text_input("Assigned supervisor / team")
            tgt = c3.date_input("Target completion", date.today() + timedelta(days=14), key="nt")
            prop = st.number_input("Proposed treatment area (ha)", 0.0, 1000.0, 0.0, 0.01)
            notes = st.text_area("Notes")
            if st.form_submit_button("Create"):
                try:
                    jp = jobs[jobs.farm_pk == farm_pk].iloc[0].id if len(jobs) else None
                    db.create_intervention(conn, farm_pk=farm_pk, farm_id=farm["farm_id"], field_id=fid, action=action,
                                           priority=pri, gap_id=gid, job_pk=None if jp is None else int(jp), version_pk=vpk,
                                           assigned_to=who or None, target_date=tgt, proposed_area_ha=prop or None,
                                           notes=notes or None)
                    st.success("Created.")
                except ValueError as e:
                    st.error(str(e))
        df = db.list_interventions(conn, farm_pk)
        if not df.empty:
            st.markdown("**Update an intervention**")
            iid = st.selectbox("Intervention", df.id.tolist(),
                               format_func=lambda i: f"#{i} {df[df.id == i].iloc[0].field_id} - {df[df.id == i].iloc[0].action}")
            r = df[df.id == iid].iloc[0]
            with st.form("upd_iv"):
                c1, c2, c3 = st.columns(3)
                status = c1.selectbox("Status", db.STATUSES, index=db.STATUSES.index(r.status))
                pri = c2.selectbox("Priority", db.PRIORITIES, index=db.PRIORITIES.index(r.priority))
                who = c3.text_input("Assigned to", value=r.assigned_to or "")
                cause = st.text_input("Verified cause", value=r.verified_cause or "")
                c1, c2, c3 = st.columns(3)
                act_area = c1.number_input("Actual treated area (ha)", 0.0, 1000.0, float(r.actual_area_ha or 0.0), 0.01)
                tdate = c2.text_input("Treatment date (YYYY-MM-DD)", value=r.treatment_date or "")
                cost = c3.number_input("Actual cost", 0.0, 1e9, float(r.actual_cost or 0.0), 1.0)
                notes = st.text_area("Notes", value=r.notes or "")
                if st.form_submit_button("Save changes"):
                    try:
                        db.update_intervention(conn, int(iid), status=status, priority=pri, assigned_to=who or None,
                                               verified_cause=cause or None, actual_area_ha=act_area if (act_area or status == "Completed") else None,
                                               treatment_date=tdate or None, actual_cost=cost or None, notes=notes or None)
                        st.success("Saved."); st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            if st.button("Delete this intervention"):
                db.delete_intervention(conn, int(iid)); st.rerun()

    with t4:
        st.markdown("Enter your own rates; nothing is assumed. Estimates are never mixed with recorded actuals.")
        c = st.columns(5)
        rates = iv.Rates(c[0].number_input("Labour days / ha", 0.0, value=0.0), c[1].number_input("Cost / labour day", 0.0, value=0.0),
                         c[2].number_input("Material units / ha (e.g. t seed cane)", 0.0, value=0.0),
                         c[3].number_input("Cost / material unit", 0.0, value=0.0), c[4].number_input("Other cost / ha", 0.0, value=0.0))
        df = db.list_interventions(conn, farm_pk)
        area = float(df[df.status != "Rejected"].proposed_area_ha.fillna(0).sum()) if not df.empty else 0.0
        area = st.number_input("Treatment area to estimate (ha)", 0.0, value=area)
        e = iv.estimate(area, rates)
        a, b, c2, d = st.columns(4)
        a.metric("Area (ha)", f"{e['area_ha']:.2f}"); b.metric("Labour days (est.)", f"{e['labour_days']:.1f}")
        c2.metric("Material units (est.)", f"{e['material_units']:.1f}"); d.metric("Cost (est.)", f"{e['est_cost']:,.0f}")


def ml_page():
    st.title("Machine learning (Random Forest)")
    st.warning("Models learn from **field-verified polygons you supply**. Scores from few fields are anecdotal; a model is "
               "only as good as, and only valid for, the seasons/farms it was trained and tested on. Probabilities are "
               "not accuracy.", icon="⚠️")
    farms = db.list_farms(conn)
    if farms.empty:
        st.info("Import a farm first."); return
    flabel = {f"{r.farm_name} ({r.farm_id})": int(r.farm_pk) for r in farms.itertuples()}
    keys = list(flabel)
    cur = next((i for i, k in enumerate(keys) if flabel[k] == S.farm_pk), 0)
    farm_pk = flabel[st.selectbox("Farm (training fields come from its current boundary version)", keys, index=cur)]
    vpk = db.current_version(conn, farm_pk)
    fields = db.get_fields(conn, vpk)
    t1, t2, t3 = st.tabs(["1. Training data and comparison", "2. Train and register", "3. Model registry"])

    with t1:
        sv = db.list_surveys(conn, vpk)
        full = sv[sv.fully_surveyed == 1]
        st.write(f"**{full.field_id.nunique()}** fully surveyed field(s), **{int((full.n_polygons > 0).sum())}** with gap polygons "
                 f"(record surveys under Validation). Pixels near polygon boundaries are ignored to reduce label noise.")
        src = st.radio("Imagery source for training", ["Demo (synthetic)", "Sentinel-2 + Sentinel-1 (Copernicus CDSE)"], horizontal=True)
        prov = DemoProvider(gap_fraction=0.12) if src.startswith("Demo") else CDSEProvider()
        if not prov.synthetic and not prov.configured:
            st.error("CDSE credentials not set (see README)."); prov = None
        if prov is not None and prov.synthetic:
            st.info("Synthetic mode: you can generate SYNTHETIC reference surveys from the generator to test the workflow. "
                    "Results say nothing about real performance.")
            if st.button("Generate SYNTHETIC reference surveys for all fields"):
                for r in fields.itertuples():
                    db.add_survey(conn, farm_pk, vpk, r.field_id, date.today(), "SYNTHETIC", "SYNTHETIC truth (demo)", True,
                                  "generated by demo provider", prov.truth_polygons(r.geometry, r.field_id))
                st.success("Created."); st.rerun()
        c = st.columns(4)
        d_from = c[0].date_input("Imagery from", date.today() - timedelta(days=60), key="ml_from")
        d_to = c[1].date_input("Imagery to", date.today(), key="ml_to")
        use_change = c[2].checkbox("Add S1 change features (vs previous window)")
        ignore = c[3].number_input("Ignore band at polygon edges (m)", 0.0, 50.0, 10.0, 5.0)
        c = st.columns(3)
        per_class = c[0].number_input("Max samples per class per field", 100, 20000, 3000, 100)
        folds = c[1].number_input("CV folds", 2, 10, 5)
        grp = c[2].selectbox("Hold out whole", ["field_id", "farm_id"])
        if st.button("Build training set and compare feature sets", type="primary", disabled=prov is None):
            with st.spinner("Fetching imagery per surveyed field and sampling pixels..."):
                df, summ = ml.build_training_set(conn, prov, vpk, d_from, d_to, use_change, ignore_band_m=ignore, max_per_class=int(per_class))
            S.ml = {"df": df, "summ": summ, "farm_pk": farm_pk, "synthetic": prov.synthetic, "use_change": use_change,
                    "dates": (str(d_from), str(d_to)), "source": prov.name, "folds": int(folds), "grp": grp}
            if df.empty:
                st.error("No training samples: survey fields (with gap polygons) first and check imagery availability.")
            else:
                S.ml["cmp"] = ml.compare_feature_sets(df, use_change, n_splits=int(folds), group_col=grp)
        m = S.get("ml")
        if m and m["farm_pk"] == farm_pk and not m["df"].empty:
            st.subheader("Training fields")
            st.dataframe(m["summ"], hide_index=True, width="stretch")
            st.subheader("Cross-validation: held-out whole " + m["grp"].replace("_id", "s"))
            st.dataframe(m["cmp"].round(3), hide_index=True, width="stretch")
            st.caption("Pixel-level, population-weighted, before morphological filtering; baseline is the transparent threshold rule on the "
                       "same held-out pixels. Positive class = verified gap. Sentinel-1 alone is not expected to separate cane from weeds or bare soil.")
            if m["synthetic"]:
                st.error("SYNTHETIC data: these numbers only test the workflow.", icon="🧪")

    with t2:
        m = S.get("ml")
        if not m or m["farm_pk"] != farm_pk or m["df"].empty:
            st.info("Build a training set first."); return
        c = st.columns(4)
        fset = c[0].selectbox("Feature set", list(ml.FEATURE_SETS), index=2)
        trees = c[1].number_input("Trees", 50, 2000, 300, 50)
        leaf = c[2].number_input("Min samples per leaf", 1, 100, 5)
        thr = c[3].slider("Probability threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Choose before looking at test scores; tuning it on the evaluation fields inflates them.")
        name = st.text_input("Model name", f"RF {fset} {date.today()}")
        if fset != "s2" and "VH_db" not in m["df"].columns:
            st.error("Training set has no Sentinel-1 features."); return
        if st.button("Train and register model", type="primary"):
            try:
                with st.spinner("Training..."):
                    rfp = ml.default_rf_params(n_estimators=int(trees), min_samples_leaf=int(leaf))
                    b = ml.train_model(m["df"], fset, m["use_change"], rfp, thr, m["synthetic"], n_splits=m["folds"], group_col=m["grp"],
                                       provenance={"source": m["source"], "dates": m["dates"], "boundary_version": vpk,
                                                   "training_fields": sorted(m["df"].field_id.unique().tolist())})
                    mid = ml.register_model(conn, b, name, db.data_dir() / "models")
                st.success(f"Registered model #{mid}.")
                if b.meta["cv"]:
                    r = b.meta["cv"]["rf"]
                    st.write(f"Held-out CV: precision {r['precision']:.2f}, recall {r['recall']:.2f}, F1 {r['f1']:.2f}, IoU {r['iou']:.2f}, "
                             f"AUC {r['auc']:.2f} over {b.meta['cv']['n_groups']} groups.")
                else:
                    st.warning("Not cross-validated: " + b.meta["cv_note"])
                st.bar_chart(pd.Series(b.meta["importances"], name="importance"))
                st.caption("Impurity importances are indicative only and favour correlated/continuous features.")
            except ml.InsufficientData as e:
                st.error(str(e))

    with t3:
        mdf = db.list_models(conn)
        if mdf.empty:
            st.info("No models yet."); return
        st.dataframe(mdf, hide_index=True, width="stretch")
        mid = st.selectbox("Inspect model", mdf.id.tolist())
        row = db.get_model_row(conn, int(mid))
        st.json({"features": json.loads(row["features_json"]), "provenance": json.loads(row["provenance_json"] or "{}"),
                 "rf_params": json.loads(row["rf_params_json"] or "{}"), "sklearn": row["sklearn_version"],
                 "cv_note": row["cv_note"], "cv": json.loads(row["cv_json"]) if row["cv_json"] and row["cv_json"] != "null" else None},
                expanded=False)
        st.caption("Model files are pickles: only load files created by this app. Applying a model to a different season, region or "
                   "sensor geometry requires new validation.")
        if st.button("Delete this model"):
            db.delete_model(conn, int(mid)); st.rerun()


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def validation_page():
    st.title("Validation against field-verified polygons")
    st.markdown("Record independent field observations, then score a job's predictions. **Reference polygons must come "
                "from ground survey or drone interpretation, not from the model's own output.** Only fields marked "
                "*fully surveyed* are scored; a surveyed field with no polygons means *no gaps found*.")
    jobs = db.list_jobs(conn, S.farm_pk) if S.farm_pk else db.list_jobs(conn)
    if jobs.empty:
        st.info("Run an analysis first."); return
    labels = {f"Job {r.id} - {r.created_at[:16]} - {r.source}": int(r.id) for r in jobs.itertuples()}
    keys = list(labels)
    cur = next((i for i, k in enumerate(keys) if labels[k] == S.get("job_id")), 0)
    job_pk = labels[st.selectbox("Analysis job to validate", keys, index=cur)]
    job = jobs[jobs.id == job_pk].iloc[0]
    vpk, farm_pk = int(job.version_pk), int(job.farm_pk)
    if job.synthetic:
        st.error("This job used SYNTHETIC imagery: scores only test the workflow and say nothing about real accuracy.", icon="🧪")
    fields = db.get_fields(conn, vpk)

    t1, t2 = st.tabs(["1. Reference surveys", "2. Accuracy assessment"])
    with t1:
        up = st.file_uploader("Field-verified gap polygons (GeoJSON / GPKG / zipped SHP)",
                              type=["geojson", "json", "gpkg", "zip"], key="ref_up")
        c1, c2, c3 = st.columns(3)
        obs_date = c1.date_input("Survey date", date.today() - timedelta(days=7))
        observer = c2.text_input("Observer / team")
        method = c3.selectbox("Method", ["Ground survey (walked)", "Drone interpretation", "High-res imagery interpretation", "Other"])
        ref_assigned, issues, id_col = None, [], None
        if up is not None:
            try:
                raw = up.getvalue()
                ref_gdf = gpd.read_file(io.BytesIO(raw))
                if ref_gdf.crs is None:
                    ref_gdf = ref_gdf.set_crs("EPSG:4326")
                cols = [c for c in ref_gdf.columns if c != "geometry"]
                pick = st.selectbox("Attribute holding the field ID (or use spatial overlap)", ["(assign by spatial overlap)"] + cols)
                id_col = None if pick.startswith("(") else pick
                ref_assigned, issues = validation.assign_reference_to_fields(ref_gdf, fields, id_col)
            except Exception as e:  # noqa: BLE001
                st.error(f"Cannot read the reference file: {e}")
        if ref_assigned is not None:
            st.write(f"{len(ref_assigned)} reference polygon(s) assigned.")
            for i in issues:
                st.caption("• " + i)
            if len(ref_assigned):
                st.dataframe(ref_assigned.assign(ha=lambda d: d.to_crs(d.estimate_utm_crs()).area / 1e4).groupby("field_id").ha.agg(["count", "sum"]).rename(columns={"count": "polygons", "sum": "ha"}).round(3))
            full = st.multiselect("Fields FULLY surveyed (include fields where you found no gaps)",
                                  sorted(fields.field_id), default=sorted(set(ref_assigned.field_id)))
            notes = st.text_input("Notes (e.g. survey protocol)")
            if st.button("Save reference survey", disabled=not full):
                for fid in full:
                    polys = list(ref_assigned[ref_assigned.field_id == fid].geometry)
                    db.add_survey(conn, farm_pk, vpk, fid, obs_date, observer, method, True, notes, polys)
                st.success(f"Saved survey records for {len(full)} field(s)."); st.rerun()
        sv = db.list_surveys(conn, vpk)
        if len(sv):
            st.dataframe(sv, hide_index=True, width="stretch")
            d = st.selectbox("Delete a survey record", [None] + sv.survey_pk.tolist())
            if d and st.button("Delete survey"):
                db.delete_survey(conn, int(d)); st.rerun()
        else:
            st.info("No reference surveys recorded for this boundary version yet.")

    with t2:
        c1, c2 = st.columns(2)
        min_ov = c1.slider("Polygon match: minimum overlap fraction", 0.1, 1.0, 0.5, 0.05)
        max_days = c2.number_input("Flag surveys further than N days from imagery end", 1, 365, 30)
        if st.button("Run accuracy assessment", type="primary"):
            ev = validation.evaluate_job(conn, job_pk, min_ov, int(max_days))
            db.save_validation(conn, job_pk, ev); S.last_ev = (job_pk, ev)
        job_ev = S.get("last_ev")
        if not job_ev or job_ev[0] != job_pk:
            st.info("Run the assessment to see results (each run is stored with its settings)."); return
        ev = job_ev[1]
        for w in ev["warnings"]:
            st.warning(w)
        if ev["fields"].empty:
            st.error("No fields could be scored.")
            if len(ev["skipped"]):
                st.dataframe(ev["skipped"], hide_index=True)
            return
        P = ev["pooled"]; px, pg, fa = P["pixel"], P["polygon"], P["field_total_area"]
        st.caption(f"{P['n_fields']} field(s) scored. Pixel level = {ev['params']['resolution_m']:.0f} m cells inside "
                   f"each field minus the {ev['params']['edge_buffer_m']:.0f} m edge buffer.")
        st.subheader("Pixel level")
        a, b, c, d, e = st.columns(5)
        a.metric("Precision", fmt(px["precision"])); b.metric("Recall", fmt(px["recall"])); c.metric("F1", fmt(px["f1"]))
        d.metric("IoU", fmt(px["iou"])); e.metric("Overall accuracy", fmt(px["overall_accuracy"]))
        cm = px["confusion_matrix"]
        st.table(pd.DataFrame({"Reference: gap": [cm["TP"], cm["FN"]], "Reference: no gap": [cm["FP"], cm["TN"]]},
                              index=["Predicted: gap", "Predicted: no gap"]))
        st.caption(f"Commission error {fmt(px['commission_error'])} | omission error {fmt(px['omission_error'])}. "
                   f"Overall accuracy is dominated by non-gap cells; rely on precision/recall/IoU.")
        ci = ev["ci95"]
        if ci:
            st.caption("95% bootstrap intervals (fields resampled): " + "; ".join(
                f"{k} {fmt(v[0],2)}-{fmt(v[1],2)}" for k, v in ci.items()))
        st.subheader("Polygon level")
        a, b, c = st.columns(3)
        a.metric("Polygon precision", fmt(pg["precision"])); b.metric("Polygon recall", fmt(pg["recall"]))
        c.metric("Predicted / reference polygons", f"{pg['n_pred']} / {pg['n_ref']}")
        st.subheader("Field total area (hectares)")
        a, b, c, d = st.columns(4)
        a.metric("Predicted total (ha)", fmt(fa["pred_total_ha"], 2)); b.metric("Reference total (ha)", fmt(fa["ref_total_ha"], 2))
        c.metric("Bias per field (ha)", fmt(fa["bias_ha_per_field"], 3)); d.metric("MAE per field (ha)", fmt(fa["mae_ha"], 3))
        st.caption(f"RMSE {fmt(fa['rmse_ha'])} ha | relative bias {fmt(fa['relative_bias_pct'], 1)}% | Pearson r {fmt(fa['pearson_r'], 2)} "
                   f"(positive bias = model over-estimates gap area).")
        f = ev["fields"]
        st.dataframe(f[["field_id", "survey_date", "method", "days_between", "timing_flag", "pred_gap_ha", "ref_gap_ha",
                        "area_error_ha", "precision", "recall", "f1", "iou", "n_pred", "n_ref", "poly_precision",
                        "poly_recall"]], hide_index=True, width="stretch")
        if len(ev["skipped"]):
            with st.expander("Fields not scored"):
                st.dataframe(ev["skipped"], hide_index=True)
        fid = st.selectbox("Overlay predicted vs reference", f.field_id.tolist())
        fg = fields[fields.field_id == fid]
        m = base_map(fg.total_bounds, basemap)
        folium.GeoJson(fg[["field_id", "geometry"]].to_json(), name="Field", style_function=lambda x: {"color": "#333", "fill": False}).add_to(m)
        pr = db.job_gaps(conn, job_pk); pr = pr[pr.field_id == fid]
        rf = db.survey_polygons_gdf(conn, vpk); rf = rf[rf.field_id == fid]
        if len(pr):
            folium.GeoJson(pr[["geometry"]].to_json(), name="Predicted (red)", style_function=lambda x: {"color": "#d50000", "fillOpacity": 0.4}).add_to(m)
        if len(rf):
            folium.GeoJson(rf[["geometry"]].to_json(), name="Reference (blue)", style_function=lambda x: {"color": "#0d47a1", "fillOpacity": 0.3}).add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, height=420, width=None, returned_objects=[], key=f"valmap_{job_pk}_{fid}")
        st.download_button("Download field scores (CSV)", f.to_csv(index=False), f"validation_job{job_pk}.csv", "text/csv")
        st.download_button("Download full report (JSON)", json.dumps({"pooled": P, "ci95": ci, "params": ev["params"],
                           "warnings": ev["warnings"], "fields": json.loads(f.to_json(orient="records"))}, default=float, indent=1),
                           f"validation_job{job_pk}.json", "application/json")
        st.info("Until these numbers come from enough independent, representative fields (and not from the data used to "
                "tune thresholds), they should not be quoted as the tool's accuracy. Tune thresholds on some fields and "
                "score on different ones.")


{"Farm and Field Explorer": explorer, "Results and Review": results, "Interventions": interventions_page, "Validation": validation_page, "Machine Learning": ml_page,
 "Method and Limits": method}[page]()
