# CaneGap Intelligence (Streamlit)

Web GIS for screening **candidate sugarcane gaps** from Sentinel-2 imagery, field by field, with hectare
quantification, farm roll-ups, review/verification and GIS exports.

> **Unvalidated screening tool.** Candidate gaps are model predictions, not confirmed missing cane. No accuracy has been
> measured. Verify in the field before operational decisions. Demo mode uses **synthetic** imagery, never real data.

## What works now
- **Farm and Field Explorer**: browse/import GeoJSON (Polygon/MultiPolygon), CRS check + reprojection, validation
  (invalid/duplicate/overlapping geometries, duplicate/missing IDs, outside-outline, missing attributes), attribute mapping,
  farm-outline handling (Arrangements A, B, C), downloadable validation report, saved boundary library (SQLite) with
  search, rename, **immutable boundary versions** and version diff, single/multi/all field selection with map preview.
- **Analysis**: every selected field is processed independently (own mask, own composite, own result); gaps are clipped to the
  parent field; areas computed in local UTM; field and area-weighted farm gap %; overlap policies; context rules
  (recent harvest / new planting); insufficient-evidence handling; new/persistent comparison against an earlier real job.
- **Review**: edit verification status, field-verified area, action, notes; delete false positives.
- **Exports**: GeoJSON, zipped Shapefile, GeoPackage, CSV, Excel (all carry farm_id, field_id, boundary version id).
- **Validation** (sidebar -> Validation): record independent field-verified polygons (GeoJSON/GPKG/zipped SHP, assigned by
  field-ID attribute or spatial overlap, clipped to the field, stored per boundary version with date/observer/method), then score a
  job. Only **fully surveyed** fields are scored (a surveyed field with no polygons = no gaps found). Reported separately:
  *pixel level* (confusion matrix, precision, recall, F1, IoU, commission/omission error), *polygon level* (precision/recall by
  overlap matching), *field-total level* (predicted vs verified hectares: bias, MAE, RMSE, r). Evaluation excludes the analysis edge
  buffer, flags survey/imagery date gaps, warns about synthetic jobs, edited predictions and small sample sizes, gives bootstrap
  intervals (resampling fields) when >=5 fields, and stores every assessment with its settings.
- **Interventions** (sidebar -> Interventions): worklist filterable by status, priority, field and target date; bulk-create
  'Pending Verification' items from a job's candidate gaps (duplicates refused; priority is an area/score inspection heuristic);
  manual add/edit with supervisor, action, verified cause, treatment date, actual area and cost (a Completed item needs date and
  actual area); cost/labour/material estimator driven only by rates you enter. Proposed, estimated and actual values stay separate.
- **PDF reports** (Results and Review): farm report (summary, field table, largest gaps, intervention workload, validation status)
  and per-field report (map, polygon table, observation metadata, interventions, method and limitations). Synthetic and
  unvalidated banners are printed on every report.
- **Sentinel-1 + Random Forest** (sidebar -> Machine Learning): Sentinel-1 IW GRD retrieval (VV/VH sigma0, orthorectified, linear
  temporal mean, optional fixed orbit direction), light speckle reduction, features VV/VH dB, VH-VV, RVI, field-relative z-scores
  and optional temporal change vs the previous equal-length window. Models train **only on fully surveyed reference fields**
  (pixels near polygon edges ignored). Evaluation holds out **whole fields (or farms)**, is population-weighted, and compares
  **S2-only vs S1-only vs S1+S2 vs the threshold baseline** on the same held-out pixels. Models are registered with provenance
  (fields, dates, source, boundary version, parameters, CV or the reason CV was impossible), flagged if trained on synthetic
  data (refused for real imagery), and can be selected for any analysis run; job reports state the method. No fallback: if a
  model needs Sentinel-1 and it is unavailable for a field, that field is reported "unable", not silently analysed another way.
- **Run Analysis page** (sidebar, 2nd item): choose farm, tick fields (or *Select all*), pick method/source/dates, press **Run Gap Analysis**,
  then *Open results*. After importing a GeoJSON the Explorer now shows a *Select fields and run analysis* button.
- **Click-on-map selection**: click a field polygon in the Explorer map to add/remove it from the selection; click a field in the
  Results map to focus it (resolved by point-in-polygon, so it works for adjacent fields).
- **Dashboards**: estate -> farm -> field -> gap drill-down from each farm's latest completed job (synthetic jobs hidden unless
  ticked), area-weighted estate totals, warnings when farms were assessed on different dates, and **trends** across assessment dates
  with comparability flags (boundary version, method, source, parameters, observation coverage, field set, synthetic) so unreliable
  comparisons are called out.
- **Imagery swipe**: before/after slider (or side-by-side) for one field and two date windows: true colour, false colour, NDVI, NDRE,
  NDMI, Sentinel-1 VH/VV; optional overlay of predicted gaps; per-window valid-coverage warnings and scene metadata.
- **Raster export**: classified GeoTIFF (Cloud-Optimized when GDAL supports it) rasterised from the current *reviewed* gap polygons
  (0 no gap, 1 candidate gap, 2 possible waterlogging, 3 not assessed, 255 outside) and band/index GeoTIFFs of the composites.
- **Probability calibration**: sigmoid or isotonic, fitted on field-held-out predictions and evaluated by cross-fitting (Brier and
  expected-calibration-error before/after, reliability curve). Skipped, with the reason, if cross-validation was not possible.
- **Multi-user logins** (below): roles, per-farm access, lock-out, audit log.
- **Background jobs**: *Run Gap Analysis* queues the job and returns immediately (toggle off for a synchronous run). Jobs run in a
  worker thread pool with their own database connections; the **Jobs** panel on the Run Analysis page shows live progress,
  current field, who started it, and a **Cancel** button (own jobs; admins any). Results are saved field by field, so a cancelled,
  failed or interrupted job keeps the fields it finished. Identical jobs that are already queued/running are refused (no
  accidental double processing). Jobs left 'running' by an app restart are marked *interrupted* at startup. Cancellation takes
  effect between fields. This is an in-process queue: for several servers or very long runs, move `queue_analysis` /
  `execute_job` (which are separate, importable functions) behind Celery/RQ.
- **Select fields right after upload**: once a GeoJSON is validated, the Import tab shows a table with a tick-box per field plus a
  map (click polygons to toggle). Choose **Import all fields (analyse only the selected ones)** or **Import only the selected
  fields**, then *Save to boundary library* or *Save and run analysis on the selected fields →* (jumps to Run Analysis with them
  pre-selected). Dropped fields are listed in the import notes; rejected features stay visible in the validation report.
- **Manual field attributes** (variety, planting/replanting date, last harvest date, ratoon number, crop, irrigation system):
  type them into the same table at import, or later in *Explorer -> Field attributes* and in a panel on the **Run Analysis**
  page, with a bulk "set for all shown fields" option. Dates are accepted as YYYY-MM-DD, DD/MM/YYYY etc. and stored as ISO;
  ratoon must be a whole number 0-15; invalid values block the save (all-or-nothing). Every change is kept in an attribute history.
  Planting and harvest dates drive the recently-harvested / newly-planted rules; if a selected field has neither, Run Analysis
  warns you and the field's result carries a note. Each job **freezes the attribute values it used** (shown in the field PDF
  and the job record), so editing attributes later never rewrites past results, and a changed attribute makes a new job, not a duplicate.
- **Tests**: `pytest` (118 tests) incl. import validation, clipping/attribution, aggregation, versioning, exports, validation metrics
  with hand-computed values, app smoke tests.

## Not built yet (next stages)
Spatial-CV beyond field/farm blocks, deep-learning/UAV-trained models, multi-server job queue, per-field raster output of model probabilities, SSO/OIDC, a PostGIS backend.
The CDSE Sentinel-2 and Sentinel-1 providers are written to the documented Process API but **has not been tested against the live service**.

## Logins and roles
Authentication is **on by default**. On first start you create the administrator; afterwards everyone signs in.
Set `CANEGAP_AUTH=off` for a single-user local run with no login. For headless deployments seed the first admin with
`CANEGAP_ADMIN_USER` and `CANEGAP_ADMIN_PASSWORD` (used only when no users exist).

| Role | Can do |
|---|---|
| viewer | look at everything they may access; export; no changes |
| analyst | + import boundaries, run analyses, review gaps, record surveys, interventions, train models |
| admin | + manage users, farm access, delete models/surveys, read the audit log |

Admins can restrict a user to specific farms; restricted users never see other farms, their jobs or results.
Security: salted scrypt hashes, generic login errors, 5-failure lock-out (5 min), password policy (>=10 chars, letters+digits),
the last active admin cannot be removed, audit trail of logins and data-changing actions (Users and Audit page).
Limits: permissions are enforced in the application layer on one shared SQLite file; anyone with file-system access to the
server can read the data. Always serve over HTTPS (reverse proxy) and back up `data/`. For corporate SSO use Streamlit's OIDC
login (`st.login`) in front of the app.

## Windows setup
1. Install Python 3.10-3.12 (python.org) and tick "Add to PATH".
2. In PowerShell inside this folder:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
If `pip install` fails on `rasterio`/`pyogrio`, upgrade pip (`python -m pip install -U pip`); wheels exist for current Python versions.
Linux/macOS: use `source .venv/bin/activate`.

## Try it without satellite access
1. `streamlit run app.py` -> **Farm and Field Explorer -> Import**.
2. Upload `sample_data/SYNTHETIC_farm_with_outline.geojson`; in "Farm outline features" pick feature index **6**; Validate.
3. Tick the fields you want (or click them on the map), type a variety / planting date if you like, then **Save and run analysis on the selected fields →**; choose **Demo (synthetic data)** -> Run Gap Analysis.
4. Open **Results and Review**. Two fields are "context excluded" (recent harvest / new planting) by design.
Other fixtures: `SYNTHETIC_problem_fixture.geojson` (invalid/duplicate/overlap/missing IDs), `SYNTHETIC_projected_utm36s.geojson`
(projected CRS declared in file), `SYNTHETIC_fields_only.geojson`. Regenerate with `python -m canegap.samples`.

## Try the validation workflow (synthetic)
After the demo run: Validation -> Reference surveys -> upload `sample_data/SYNTHETIC_reference_gaps.geojson` (field ID attribute
`field_id`), mark F02 and F05 plus any other fully surveyed fields, save; then Accuracy assessment -> Run. These numbers test the
workflow only. For real use: survey independently of the model, on fields not used to tune thresholds, close in time to the imagery.

## Try the ML workflow (synthetic)
Machine Learning -> Training data -> source Demo -> *Generate SYNTHETIC reference surveys* -> *Build training set and compare
feature sets* -> Train and register -> run an analysis choosing the Random Forest method. In demo mode the synthetic radar is
deliberately weak, so S1-only scores poorly; that is a property of the generator, not of real Sentinel-1.
**Real use:** record independent field surveys first (Validation section), train on some fields/farms, and validate on others.

## Where do I run an analysis?
Sidebar -> **Run Analysis**. Steps: import/select a farm in *Farm and Field Explorer* -> (optionally click fields on the map) ->
**Run Analysis** -> choose source (Demo works without credentials) -> *Run Gap Analysis* -> *Open results*.

## Real Sentinel-2 data (Copernicus Data Space Ecosystem)
1. Register at dataspace.copernicus.eu, then create an OAuth client in the Sentinel Hub dashboard.
2. Copy `.env.example` to `.env` values into your environment (PowerShell: `$env:CDSE_CLIENT_ID="..."; $env:CDSE_CLIENT_SECRET="..."`)
   before `streamlit run app.py`. Never commit credentials. On Streamlit Community Cloud use app Secrets exposed as env vars.
3. Choose "Sentinel-2 L2A (Copernicus CDSE)". Usage is limited by your CDSE account quotas (Processing Units) and terms; check them.
Per-pixel median of SCL-valid observations (cloud, shadow, cirrus, snow excluded) is built by the service for each field's bounding box.

## Data and deployment
- Data lives in `data/canegap.sqlite` (override with `CANEGAP_DATA_DIR`). Back it up; it is the boundary library and audit trail.
- Streamlit Community Cloud's disk is ephemeral: for persistent use deploy on a VM/container with a mounted volume
  (e.g. Docker with `-v canegap-data:/app/data`) behind HTTPS (Caddy/nginx), or swap `db.py` for PostgreSQL/PostGIS.
- Dockerfile sketch: `FROM python:3.12-slim` + `pip install -r requirements.txt` + `CMD streamlit run app.py --server.port 8501`
  (the GDAL-based wheels bundle their libraries; no apt GDAL needed).

## Layout
`app.py` UI | `canegap/geojson_import.py` validation | `db.py` persistence | `providers/` demo + CDSE |
`pipeline.py` per-field detection | `aggregate.py` roll-ups | `exports.py` | `samples.py` | `tests/`
