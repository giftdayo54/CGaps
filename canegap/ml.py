"""Random Forest framework: Sentinel-1/Sentinel-2 features, training from field-verified polygons,
field-grouped cross-validation and model bundles.

Honesty rules built in
  * Training labels come ONLY from fully surveyed reference fields (independent observation).
  * Cross-validation holds out whole FIELDS (GroupKFold), because pixels inside a field are spatially
    correlated; random pixel splits would leak and inflate scores. With several farms you can hold out farms.
  * CV metrics are population-weighted (class subsampling is undone with sample weights) and computed
    before morphological filtering, side by side with the threshold baseline on the same held-out pixels.
  * A model is flagged `cv_validated` only if cross-validation could actually be run (>= 3 groups, positives in
    >= 2 groups). Even then CV scores are internal estimates; apply to new seasons/farms only after new validation.
  * Model scores are RF probabilities, not calibrated and not accuracy.
  * Models trained on SYNTHETIC imagery are marked and refused for real imagery.
  * Model files are joblib/pickle: load only files this app created.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio import features
from scipy.ndimage import uniform_filter

from . import db
from .areas import M2_PER_HA, to_crs_geom, utm_crs_for
from .pipeline import Params, indices, threshold_masks

S2_FEATURES = ["B03", "B04", "B05", "B08", "B11", "NDVI", "NDRE", "GNDVI", "NDMI", "MNDWI", "SAVI", "NDVI_z"]
S1_FEATURES = ["VV_db", "VH_db", "VHVV_db", "RVI", "VH_db_z"]
S1_CHANGE_FEATURES = ["dVV_db", "dVH_db"]
FEATURE_SETS = ("s2", "s1", "s1+s2")


class InsufficientData(ValueError):
    pass


def feature_names(feature_set: str, use_s1_change: bool = False) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    names = []
    if "s2" in feature_set:
        names += S2_FEATURES
    if "s1" in feature_set:
        names += S1_FEATURES + (S1_CHANGE_FEATURES if use_s1_change else [])
    return names


def baseline_window(date_from, date_to):
    """Equal-length window immediately before [date_from, date_to] for S1 change features."""
    a = datetime.strptime(str(date_from)[:10], "%Y-%m-%d").date()
    b = datetime.strptime(str(date_to)[:10], "%Y-%m-%d").date()
    span = max((b - a).days, 1)
    return (a - timedelta(days=span + 1)).isoformat(), (a - timedelta(days=1)).isoformat()


# ----------------------------------------------------------------- features
def _db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.where(x > 0, x, np.nan))


def _nanmean_filter(a, size):
    """Spatial mean in the LINEAR domain, ignoring NaN (light speckle reduction)."""
    if size <= 1:
        return a
    ok = np.isfinite(a)
    num = uniform_filter(np.where(ok, a, 0.0), size=size, mode="constant", cval=0.0)
    den = uniform_filter(ok.astype("float32"), size=size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~ok] = np.nan
    return out


def _robust_z(a, mask):
    v = a[mask & np.isfinite(a)]
    if v.size < 10:
        return np.full_like(a, np.nan)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return (a - med) / max(mad, 1e-3)


def feature_stack(comp_s2, comp_s1=None, comp_s1_base=None, core=None, s1_smooth_px: int = 3) -> dict:
    """All candidate features on the field grid. Field-relative z-scores use the field's core pixels."""
    ix = indices(comp_s2.bands)
    b = comp_s2.bands
    out = {k: b[k] for k in ("B03", "B04", "B05", "B08", "B11")}
    out.update({k: ix[k] for k in ("NDVI", "NDRE", "GNDVI", "NDMI", "MNDWI", "SAVI")})
    core = np.ones_like(ix["NDVI"], dtype=bool) if core is None else core
    out["NDVI_z"] = _robust_z(ix["NDVI"], core)
    if comp_s1 is not None:
        vv = _nanmean_filter(comp_s1.bands["VV"].astype("float32"), s1_smooth_px)
        vh = _nanmean_filter(comp_s1.bands["VH"].astype("float32"), s1_smooth_px)
        out["VV_db"], out["VH_db"] = _db(vv), _db(vh)
        out["VHVV_db"] = out["VH_db"] - out["VV_db"]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["RVI"] = 4 * vh / (vv + vh)
        out["VH_db_z"] = _robust_z(out["VH_db"], core)
        if comp_s1_base is not None:
            bvv = _nanmean_filter(comp_s1_base.bands["VV"].astype("float32"), s1_smooth_px)
            bvh = _nanmean_filter(comp_s1_base.bands["VH"].astype("float32"), s1_smooth_px)
            out["dVV_db"], out["dVH_db"] = out["VV_db"] - _db(bvv), out["VH_db"] - _db(bvh)
    return out


def to_matrix(stack: dict, names: list[str]):
    """(h, w, n) float32 matrix and the mask of pixels where every feature is finite."""
    missing = [n for n in names if n not in stack]
    if missing:
        raise ValueError(f"Features not available from the imagery: {missing}")
    X = np.stack([np.asarray(stack[n], dtype="float32") for n in names], axis=-1)
    return X, np.isfinite(X).all(axis=-1)


# -------------------------------------------------------------- model bundle
@dataclass
class ModelBundle:
    model: object
    feature_names: list[str]
    feature_set: str
    use_s1_change: bool = False
    s1_smooth_px: int = 3
    threshold: float = 0.5
    synthetic: bool = False
    meta: dict = field(default_factory=dict)
    calibrator: object = None            # optional probability calibrator (see fit_calibrator)

    @property
    def needs_s1(self) -> bool:
        return "s1" in self.feature_set

    @property
    def needs_s2(self) -> bool:
        return "s2" in self.feature_set

    @property
    def label(self) -> str:
        i = self.meta.get("id")
        return f"rf{'#' + str(i) if i else ''}-{self.feature_set}"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(X)[:, list(self.model.classes_).index(1)].astype("float32")
        cal = getattr(self, "calibrator", None)
        return cal.predict(p).astype("float32") if cal is not None else p


def save_bundle(bundle: ModelBundle, path) -> None:
    import joblib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path) -> ModelBundle:
    import joblib
    return joblib.load(path)       # only load files created by this application (pickle)


# ------------------------------------------------------------- training data
def build_training_set(conn, provider, version_pk: int, date_from, date_to, use_s1_change: bool = False,
                       params: Params | None = None, ignore_band_m: float = 10.0, max_per_class: int = 3000,
                       s1_smooth_px: int = 3, seed: int = 0, include_s1: bool = True):
    """Sample labelled pixels from fully surveyed reference fields (all features; subset later by feature set).

    Pixels within `ignore_band_m` of a reference polygon boundary are dropped (label ambiguity at mixed pixels).
    Class subsampling is recorded as weight = (valid pixels of class in field) / (sampled), so metrics can
    estimate the full population. Returns (DataFrame, per-field summary DataFrame)."""
    p = params or Params()
    rng = np.random.default_rng(seed)
    fields = db.get_fields(conn, version_pk).set_index("field_id")
    farm = conn.execute("SELECT f.farm_id FROM boundary_versions b JOIN farms f ON f.id=b.farm_pk WHERE b.id=?",
                        (version_pk,)).fetchone()["farm_id"]
    surveys = db.list_surveys(conn, version_pk)
    surveys = surveys[surveys.fully_surveyed == 1].sort_values("observed_at").groupby("field_id").tail(1)
    names_all = feature_names("s1+s2" if include_s1 else "s2", use_s1_change and include_s1)
    rows, summary = [], []
    for sv in surveys.itertuples():
        fid = sv.field_id
        if fid not in fields.index or fields.at[fid, "geometry"] is None:
            summary.append({"field_id": fid, "used": False, "reason": "field geometry missing"}); continue
        utm = utm_crs_for(fields.at[fid, "geometry"])
        g = to_crs_geom(fields.at[fid, "geometry"], "EPSG:4326", utm)
        try:
            s2 = provider.get_composite(g.bounds, str(utm), date_from, date_to, p.resolution_m, seed_key=fid)
            s1 = s1b = None
            if include_s1:
                s1 = provider.get_s1_composite(g.bounds, str(utm), date_from, date_to, p.resolution_m, seed_key=fid)
                if use_s1_change:
                    bf, bt = baseline_window(date_from, date_to)
                    s1b = provider.get_s1_composite(g.bounds, str(utm), bf, bt, p.resolution_m, seed_key=fid)
        except Exception as e:  # noqa: BLE001
            summary.append({"field_id": fid, "used": False, "reason": f"imagery unavailable: {e}"}); continue
        h, w = next(iter(s2.bands.values())).shape
        core_geom = g.buffer(-p.edge_buffer_m) if p.edge_buffer_m > 0 else g
        core_geom = g if core_geom.is_empty else core_geom
        core = features.geometry_mask([core_geom], (h, w), s2.transform, invert=True)
        stack = feature_stack(s2, s1, s1b, core, s1_smooth_px)
        X, finite = to_matrix(stack, names_all)
        from shapely.geometry import shape
        from shapely.ops import unary_union
        ref = [to_crs_geom(shape(json.loads(j)), "EPSG:4326", utm).intersection(g)
               for j in db.survey_polygon_json(conn, int(sv.survey_pk))]
        ref = [r for r in ref if not r.is_empty]
        if ref:
            ru = unary_union(ref)
            y = features.rasterize([(ru, 1)], (h, w), transform=s2.transform, fill=0, dtype="uint8").astype(bool)
            band = features.rasterize([(ru.boundary.buffer(ignore_band_m), 1)], (h, w), transform=s2.transform,
                                      fill=0, dtype="uint8").astype(bool) if ignore_band_m > 0 else np.zeros((h, w), bool)
        else:
            y, band = np.zeros((h, w), bool), np.zeros((h, w), bool)
        valid = core & finite & ~band
        # baseline threshold prediction on the same pixels (needs S2 finite)
        ix = indices(s2.bands)
        base = np.zeros((h, w), bool)
        sfin = core & np.isfinite(ix["NDVI"])
        if sfin.sum() > 10:
            tm = threshold_masks(ix, core, np.isfinite(ix["NDVI"]), p)
            base = tm["bare"] | tm["water"]
        info = {"field_id": fid, "used": True, "valid_px": int(valid.sum()), "pos_px": int((valid & y).sum()),
                "neg_px": int((valid & ~y).sum()), "survey_date": sv.observed_at}
        for cls, mask in ((1, valid & y), (0, valid & ~y)):
            idx = np.flatnonzero(mask.ravel())
            if idx.size == 0:
                continue
            sel = rng.choice(idx, size=min(idx.size, max_per_class), replace=False)
            part = pd.DataFrame(X.reshape(-1, X.shape[-1])[sel], columns=names_all)
            part["label"] = cls
            part["w"] = idx.size / sel.size
            part["field_id"], part["farm_id"] = fid, farm
            part["baseline_gap"] = base.ravel()[sel]
            rows.append(part)
        summary.append(info)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return df, pd.DataFrame(summary)


# ---------------------------------------------------------------- evaluation
def _prf(y, pred, w):
    tp = float(w[(pred == 1) & (y == 1)].sum()); fp = float(w[(pred == 1) & (y == 0)].sum())
    fn = float(w[(pred == 0) & (y == 1)].sum()); tn = float(w[(pred == 0) & (y == 0)].sum())
    d = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"precision": d(tp, tp + fp), "recall": d(tp, tp + fn), "f1": d(2 * tp, 2 * tp + fp + fn),
            "iou": d(tp, tp + fp + fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def default_rf_params(**kw) -> dict:
    d = dict(n_estimators=300, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample",
             n_jobs=-1, random_state=0)
    d.update(kw)
    return d


def _check_groups(df, group_col, min_groups=3):
    g = df.groupby(group_col).label.agg(["sum", "count"])
    n_groups, n_pos_groups = len(g), int((g["sum"] > 0).sum())
    if n_groups < min_groups or n_pos_groups < 2:
        raise InsufficientData(
            f"Cross-validation needs >= {min_groups} {group_col}s and positives in >= 2 of them "
            f"(have {n_groups} with positives in {n_pos_groups}). Survey more fields before trusting any score.")
    return n_groups


def cross_validate(df: pd.DataFrame, feature_cols: list[str], rf_params: dict | None = None, n_splits: int = 5,
                   group_col: str = "field_id", threshold: float = 0.5) -> dict:
    """Group-held-out CV. Returns pooled weighted metrics for the RF and for the threshold baseline."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    n_groups = _check_groups(df, group_col)
    X, y, w, grp = df[feature_cols].to_numpy("float32"), df.label.to_numpy(), df.w.to_numpy(), df[group_col].to_numpy()
    proba = np.full(len(df), np.nan)
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=min(n_splits, n_groups)).split(X, y, grp)):
        if len(np.unique(y[tr])) < 2:
            folds.append({"fold": k, "skipped": "training split has one class", "test_groups": sorted(set(grp[te]))}); continue
        m = RandomForestClassifier(**(rf_params or default_rf_params())).fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, list(m.classes_).index(1)]
        f = _prf(y[te], (proba[te] >= threshold).astype(int), w[te])
        f.update(fold=k, test_groups=sorted(set(grp[te])))
        folds.append(f)
    ok = np.isfinite(proba)
    if not ok.any():
        raise InsufficientData("No fold could be evaluated.")
    pooled = _prf(y[ok], (proba[ok] >= threshold).astype(int), w[ok])
    pooled["auc"] = (float(roc_auc_score(y[ok], proba[ok], sample_weight=w[ok])) if len(np.unique(y[ok])) == 2 else float("nan"))
    out = {"group_col": group_col, "n_groups": int(n_groups), "n_samples": int(ok.sum()), "threshold": threshold,
           "rf": pooled, "folds": folds, "_oof": proba}
    if "baseline_gap" in df:
        out["baseline"] = _prf(y[ok], df.baseline_gap.to_numpy()[ok].astype(int), w[ok])
    return out


# --------------------------------------------------------------- calibration
class Calibrator:
    """Maps raw RF probabilities to calibrated ones. method: 'sigmoid' (Platt on logit) or 'isotonic'."""

    def __init__(self, method: str, model):
        self.method, self.model = method, model

    def predict(self, p):
        p = np.asarray(p, dtype="float64")
        if self.method == "isotonic":
            return self.model.predict(p)
        z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        return self.model.predict_proba(z.reshape(-1, 1))[:, 1]


def fit_calibrator(p, y, w, method: str) -> Calibrator:
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        return Calibrator(method, IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y, sample_weight=w))
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression
        pp = np.clip(p, 1e-4, 1 - 1e-4)
        z = np.log(pp / (1 - pp)).reshape(-1, 1)
        return Calibrator(method, LogisticRegression(C=1e6).fit(z, y, sample_weight=w))
    raise ValueError("calibration must be 'sigmoid' or 'isotonic'")


def reliability_table(p, y, w, n_bins: int = 10) -> pd.DataFrame:
    """Weighted reliability: mean predicted probability vs observed gap fraction per probability bin."""
    bins = np.minimum((np.asarray(p) * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        ww = w[m]
        rows.append({"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "mean_pred": float(np.average(p[m], weights=ww)),
                     "observed": float(np.average(y[m], weights=ww)), "weight": float(ww.sum())})
    return pd.DataFrame(rows)


def _brier_ece(p, y, w, n_bins=10):
    brier = float(np.average((p - y) ** 2, weights=w))
    t = reliability_table(p, y, w, n_bins)
    ece = float((t.weight / t.weight.sum() * (t.mean_pred - t.observed).abs()).sum()) if len(t) else float("nan")
    return brier, ece


def calibration_report(p, y, w, groups, method: str, threshold: float = 0.5, n_splits: int = 5) -> tuple[dict, Calibrator]:
    """Raw vs calibrated reliability using CROSS-FITTING (calibrator fit on other groups, applied to the held-out
    group) so the reported improvement is not simply an in-sample fit. Returns (report, calibrator fit on all)."""
    from sklearn.model_selection import GroupKFold
    ok = np.isfinite(p)
    p, y, w, groups = p[ok], y[ok], w[ok], np.asarray(groups)[ok]
    n_groups = len(np.unique(groups))
    if n_groups < 3 or len(np.unique(y)) < 2:
        raise InsufficientData("Calibration needs out-of-fold predictions from >= 3 groups with both classes.")
    pcal = np.full(len(p), np.nan)
    for tr, te in GroupKFold(n_splits=min(n_splits, n_groups)).split(p, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pcal[te] = fit_calibrator(p[tr], y[tr], w[tr], method).predict(p[te])
    use = np.isfinite(pcal)
    if not use.any():
        raise InsufficientData("Calibration cross-fitting produced no predictions.")
    b0, e0 = _brier_ece(p[use], y[use], w[use])
    b1, e1 = _brier_ece(pcal[use], y[use], w[use])
    report = {
        "method": method, "n_samples": int(use.sum()),
        "brier_raw": b0, "brier_calibrated_crossfit": b1, "ece_raw": e0, "ece_calibrated_crossfit": e1,
        "reliability_raw": reliability_table(p[use], y[use], w[use]).to_dict("records"),
        "reliability_calibrated": reliability_table(pcal[use], y[use], w[use]).to_dict("records"),
        "metrics_at_threshold_calibrated": _prf(y[use], (pcal[use] >= threshold).astype(int), w[use]),
        "note": ("Calibration is estimated against the TRAINING population's gap prevalence (sample weights undo class "
                 "subsampling); fields with a very different gap rate, a new season or a new sensor setup need re-checking. "
                 "A calibrated probability is a better-behaved score, not a guarantee of accuracy."),
    }
    return report, fit_calibrator(p, y, w, method)


def compare_feature_sets(df, use_s1_change=False, sets=FEATURE_SETS, **kw) -> pd.DataFrame:
    """Sentinel-2-only vs Sentinel-1-only vs combined RF (+ threshold baseline), same held-out folds."""
    rows, base_done = [], False
    for fs in sets:
        cols = [c for c in feature_names(fs, use_s1_change) if c in df.columns]
        if len(cols) < len(feature_names(fs, use_s1_change)):
            rows.append({"method": f"RF {fs}", "note": "features missing from training set"}); continue
        try:
            r = cross_validate(df, cols, **kw)
        except InsufficientData as e:
            rows.append({"method": f"RF {fs}", "note": str(e)}); continue
        rows.append({"method": f"RF {fs}", **{k: r["rf"][k] for k in ("precision", "recall", "f1", "iou", "auc")},
                     "n_groups": r["n_groups"], "n_samples": r["n_samples"]})
        if "baseline" in r and not base_done:
            rows.append({"method": "Threshold baseline (S2)", **{k: r["baseline"][k] for k in ("precision", "recall", "f1", "iou")},
                         "n_groups": r["n_groups"], "n_samples": r["n_samples"]})
            base_done = True
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ training
def train_model(df: pd.DataFrame, feature_set: str, use_s1_change: bool = False, rf_params: dict | None = None,
                threshold: float = 0.5, synthetic: bool = False, s1_smooth_px: int = 3, n_splits: int = 5,
                group_col: str = "field_id", provenance: dict | None = None, calibration: str = "none") -> ModelBundle:
    """Fit on ALL samples; attach group-CV metrics if (and only if) CV could be run."""
    from sklearn.ensemble import RandomForestClassifier
    cols = feature_names(feature_set, use_s1_change)
    if df.empty or df.label.nunique() < 2:
        raise InsufficientData("Training needs both gap and non-gap pixels from surveyed fields.")
    meta = {"n_samples": int(len(df)), "n_fields": int(df.field_id.nunique()),
            "n_pos_samples": int((df.label == 1).sum()), "n_neg_samples": int((df.label == 0).sum()),
            "field_ids": sorted(df.field_id.unique().tolist()), "provenance": provenance or {}, "cv": None,
            "cv_note": ""}
    oof, calibrator = None, None
    try:
        cv = cross_validate(df, cols, rf_params, n_splits, group_col, threshold)
        oof = cv.pop("_oof")
        meta["cv"] = cv
    except InsufficientData as e:
        meta["cv_note"] = str(e)
    meta["calibration"] = None
    if calibration != "none":
        if oof is None:
            meta["calibration_note"] = "Not calibrated: calibration needs cross-validated out-of-fold predictions."
        else:
            try:
                rep, calibrator = calibration_report(oof, df.label.to_numpy(), df.w.to_numpy(), df[group_col].to_numpy(),
                                                     calibration, threshold, n_splits)
                meta["calibration"] = rep
            except InsufficientData as e:
                meta["calibration_note"] = f"Not calibrated: {e}"
    params = rf_params or default_rf_params()
    m = RandomForestClassifier(**params).fit(df[cols].to_numpy("float32"), df.label.to_numpy())
    meta["importances"] = dict(sorted(zip(cols, map(float, m.feature_importances_)), key=lambda t: -t[1]))
    meta["rf_params"] = {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in params.items()}
    return ModelBundle(m, cols, feature_set, use_s1_change, s1_smooth_px, threshold, synthetic, meta, calibrator)


def register_model(conn, bundle: ModelBundle, name: str, folder, created_by: str | None = None) -> int:
    """Persist bundle to disk + registry. Returns model id and sets bundle.meta['id']."""
    mid = db.add_model_row(conn, name, bundle)
    bundle.meta["id"] = mid
    path = Path(folder) / f"model_{mid}.joblib"
    save_bundle(bundle, path)
    db.set_model_path(conn, mid, str(path))
    return mid
