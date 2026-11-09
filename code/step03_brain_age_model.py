# -*- coding: utf-8 -*-
"""
Step 3 — Brain age prediction model.

Trains and evaluates:
  • SVR + FBTS (main model)
  • Ridge + FBTS (linear baseline)
  • SVR + PSD  (spectral baseline)
  • SVR + DE   (entropy baseline)
  • SVR + single-band (ablation, per band)

Cross-validation: 5-fold stratified (outer) × 3-fold (inner for tuning).
Also runs LOSO for the main model.

Saves per-subject predictions and aggregate metrics.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_predict, GridSearchCV,
    LeaveOneOut,
)

from config import *
from utils import setup_logging

from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore", category=FutureWarning)
logger = setup_logging("step03_model")


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Return COR, RMSE, MAE."""
    if len(y_true) >= 2:
        r, _ = pearsonr(y_true, y_pred)
    else:
        r = np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return dict(cor=r, rmse=rmse, mae=mae)


# ── Age-stratified folds ────────────────────────────────────────────────

def make_age_strata(ages, n_folds=CV_FOLDS):
    """Create age-stratified labels for StratifiedKFold.
    
    Adapts the number of bins to sample size so no stratum has
    fewer than n_folds members.  Returns at least 2 strata.
    """
    n = len(ages)
    # ensure each stratum has at least n_folds members
    max_bins = max(2, n // n_folds)
    n_bins = min(n_folds, max_bins)
    try:
        bins = pd.qcut(ages, n_bins, labels=False, duplicates="drop")
    except Exception:
        # edge case: all ages identical or too few unique values
        bins = np.zeros(n, dtype=int)
    # ensure at least 2 distinct strata for StratifiedKFold
    if len(np.unique(bins)) < 2:
        bins = np.zeros(n, dtype=int)
        bins[n // 2:] = 1
    return bins


# ── Model builders ──────────────────────────────────────────────────────

def build_svr_pipeline():
    """SVR pipeline: SelectKBest → StandardScaler → PCA → SVR.
    
    PCA after feature selection reduces dimensionality to 30-120,
    addressing the curse of dimensionality with small samples.
    """
    return Pipeline([
        ("fs", SelectKBest(f_regression)),
        ("sc", StandardScaler()),
        ("pca", PCA()),
        ("svr", SVR(kernel="rbf")),
    ])


def build_ridge_pipeline():
    """Ridge pipeline: SelectKBest → StandardScaler → PCA → Ridge.
    
    Ridge with PCA is more stable than SVR for high-dimensional
    features with small samples (fewer hyperparameters, convex optimization).
    """
    return Pipeline([
        ("fs", SelectKBest(f_regression)),
        ("sc", StandardScaler()),
        ("pca", PCA()),
        ("ridge", Ridge()),
    ])


def build_gbr_pipeline():
    """Gradient Boosting pipeline: SelectKBest → StandardScaler → PCA → GBR.

    GBR can capture non-linear age patterns that SVR's RBF kernel
    may miss, especially at the age extremes where regression to
    the mean is most severe.
    """
    return Pipeline([
        ("fs", SelectKBest(f_regression)),
        ("sc", StandardScaler()),
        ("pca", PCA()),
        ("gbr", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])


# ── Hyperparameter grids ────────────────────────────────────────────────

SVR_GRID = {
    "fs__k": FS_K_SEARCH,
    "pca__n_components": PCA_N_SEARCH,
    "svr__C": SVR_PARAMS["C"],
    "svr__gamma": SVR_PARAMS["gamma"],
    "svr__epsilon": SVR_PARAMS["epsilon"],
}

RIDGE_GRID = {
    "fs__k": FS_K_SEARCH,
    "pca__n_components": PCA_N_SEARCH,
    "ridge__alpha": RIDGE_PARAMS["alpha"],
}

GBR_GRID = {
    "fs__k": [100, 200, 500],
    "pca__n_components": PCA_N_SEARCH,
    "gbr__n_estimators": [100, 200],
    "gbr__max_depth": [3, 5],
    "gbr__learning_rate": [0.05, 0.1],
}

# For PSD / DE (lower-dimensional, no feature selection needed)
SVR_GRID_PSD = {
    "svr__C": SVR_PARAMS["C"],
    "svr__gamma": SVR_PARAMS["gamma"],
    "svr__epsilon": SVR_PARAMS["epsilon"],
}


def build_svr_psd_pipeline():
    """SVR for PSD/DE features (640-dim): StandardScaler → SVR."""
    return Pipeline([
        ("sc", StandardScaler()),
        ("svr", SVR(kernel="rbf")),
    ])


# ── Cross-validation ────────────────────────────────────────────────────

def run_cv_model(X, y, model, param_grid, strata, model_name):
    """
    Run stratified CV with inner tuning.

    Returns (predictions, per_fold_metrics, best_params).
    """
    # adapt outer folds to the smallest stratum
    min_class = pd.Series(strata).value_counts().min()
    n_outer = min(CV_FOLDS, min_class)
    n_outer = max(2, n_outer)  # at least 2-fold
    # inner folds also adapted
    n_inner = min(INNER_CV_FOLDS, max(2, len(y) * (n_outer - 1) // n_outer // 2))

    if n_outer < CV_FOLDS:
        logger.info("  [%s] adapting CV: %d outer folds (min stratum=%d)",
                    model_name, n_outer, min_class)

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True,
                               random_state=RANDOM_STATE)
    inner_cv = KFold(n_splits=n_inner, shuffle=True,
                     random_state=RANDOM_STATE)

    preds = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, strata)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        gs = GridSearchCV(model, param_grid, cv=inner_cv,
                          scoring="neg_root_mean_squared_error",
                          n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        fold_preds = gs.predict(X_te)
        # clip predictions to plausible age range to prevent divergence
        fold_preds = np.clip(fold_preds, 3.0, 22.0)
        preds[test_idx] = fold_preds

        m = compute_metrics(y_te, preds[test_idx])
        fold_metrics.append(m)
        logger.info("  [%s] fold %d: COR=%.3f RMSE=%.3f MAE=%.3f  best_k=%s",
                    model_name, fold + 1, m["cor"], m["rmse"], m["mae"],
                    gs.best_params_.get("fs__k", "N/A"))

    overall = compute_metrics(y, preds)
    fold_df = pd.DataFrame(fold_metrics)
    logger.info("  [%s] OVERALL: COR=%.3f±%.3f RMSE=%.3f±%.3f MAE=%.3f±%.3f",
                model_name,
                overall["cor"], fold_df["cor"].std(),
                overall["rmse"], fold_df["rmse"].std(),
                overall["mae"], fold_df["mae"].std())
    return preds, fold_df, overall


# ── LOSO ────────────────────────────────────────────────────────────────

def run_loso(X, y, model, param_grid, model_name="FBTS_SVR_LOSO"):
    """Leave-One-Subject-Out validation."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    n = len(y)

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        # Inner CV folds must not exceed n_samples-1; for very small
        # training sets (n < 6), fall back to 2-fold to avoid crashes.
        n_inner = min(INNER_CV_FOLDS, max(2, len(X_tr) - 1))
        gs = GridSearchCV(model, param_grid, cv=n_inner,
                          scoring="neg_root_mean_squared_error",
                          n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        preds[test_idx] = np.clip(gs.predict(X_te), 3.0, 22.0)

        if (i + 1) % 50 == 0:
            r_partial, _ = pearsonr(y[:i + 1], preds[:i + 1])
            logger.info("  [%s] %d/%d done (partial COR=%.3f)",
                        model_name, i + 1, n, r_partial)

    metrics = compute_metrics(y, preds)
    logger.info("  [%s] LOSO: COR=%.3f RMSE=%.3f MAE=%.3f",
                model_name, metrics["cor"], metrics["rmse"], metrics["mae"])
    return preds, metrics


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HBN brain age model")
    parser.add_argument("--loso", action="store_true",
                        help="Also run LOSO validation (slow)")
    parser.add_argument("--merge-releases", action="store_true",
                        help="Merge features from all releases in FEATURE_DIR")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step 3: Brain Age Prediction Model")
    logger.info("=" * 60)

    # load features
    if args.merge_releases:
        # Find all release feature files (R1, R2, ..., R10)
        release_files = sorted(FEATURE_DIR.glob(f"{REST_TASK}_features_R*.npz"))
        if not release_files:
            # No release-specific files — use single base file
            release_files = [FEATURE_DIR / f"{REST_TASK}_features.npz"]
            logger.info("No release-specific files found, using single feature file")
        else:
            logger.info("Merging %d release feature files: %s",
                        len(release_files), [f.name for f in release_files])

        all_data = []
        for rf in release_files:
            d = np.load(rf, allow_pickle=True)
            all_data.append(d)
            logger.info("  %s: %d subjects", rf.name, len(d["ages"]))

        # Concatenate all arrays
        def _cat(key):
            return np.concatenate([d[key] for d in all_data])

        ages = _cat("ages")
        sids = _cat("subject_ids")
        psd = _cat("psd")
        de = _cat("de")
        covmats_merged = _cat("covmats")  # (n_total, 5, 128, 128)
        if "alpha_peak" in all_data[0].files:
            alpha_peak = _cat("alpha_peak")
        else:
            alpha_peak = None

        # HBN releases may contain repeated subjects across releases
        # (longitudinal re-recording of the same participant).  Keep the
        # first occurrence (earliest release) to avoid counting the same
        # subject twice and to prevent information leakage between
        # train/test folds during cross-validation.
        unique_sids, unique_idx = np.unique(sids, return_index=True)
        if len(unique_idx) < len(sids):
            n_dup = len(sids) - len(unique_idx)
            logger.warning("  Removed %d duplicate subjects across releases "
                           "(keeping first occurrence)", n_dup)
            keep = np.sort(unique_idx)  # preserve original order
            ages = ages[keep]
            sids = sids[keep]
            psd = psd[keep]
            de = de[keep]
            covmats_merged = covmats_merged[keep]
            if alpha_peak is not None:
                alpha_peak = alpha_peak[keep]

        n_subjects = len(ages)
        logger.info("Merged total: %d subjects", n_subjects)

        # ── Re-compute tangent space on merged covariances ──
        # Each release's tangent vectors were projected relative to that
        # release's own Riemannian mean.  Direct concatenation would mix
        # different coordinate systems.  Re-project using a global mean
        # computed from ALL merged covariance matrices.
        n_bands = len(BAND_ORDER)
        n_ch = covmats_merged.shape[2]
        ts_dim = n_ch * (n_ch + 1) // 2
        tangent = np.zeros((n_subjects, n_bands, ts_dim))
        for b, band_name in enumerate(BAND_ORDER):
            cov_band = covmats_merged[:, b, :, :]
            ts = TangentSpace(metric="riemann")
            tangent[:, b, :] = ts.fit_transform(cov_band)
            logger.info("  Re-projected tangent space (%s): %s",
                        band_name, tangent[:, b, :].shape)
        fbts = tangent.reshape(n_subjects, -1)
        # free memory — all_data still holds covmats references
        del covmats_merged, all_data
    else:
        feat_path = FEATURE_DIR / f"{REST_TASK}_features.npz"
        if not feat_path.exists():
            logger.error("Feature file not found: %s — run step02 first", feat_path)
            return

        data = np.load(feat_path, allow_pickle=True)
        fbts = data["fbts"]
        psd = data["psd"]
        de = data["de"]
        tangent = data["tangent"]
        ages = data["ages"]
        sids = data["subject_ids"]
        alpha_peak = data["alpha_peak"] if "alpha_peak" in data.files else None
        n_subjects = len(ages)

    # replace NaN / Inf
    for arr in [fbts, psd, de]:
        arr[np.isnan(arr)] = 0.0
        arr[np.isinf(arr)] = 0.0

    logger.info("Loaded: %d subjects, FBTS %s, PSD %s, DE %s",
                n_subjects, fbts.shape, psd.shape, de.shape)
    if alpha_peak is not None:
        logger.info("  Alpha peak freq: mean=%.2f Hz, range [%.2f, %.2f]",
                    alpha_peak.mean(), alpha_peak.min(), alpha_peak.max())

    strata = make_age_strata(ages)

    all_preds = pd.DataFrame({"subject": sids, "age": ages})
    all_metrics = []
    all_fold_metrics = []

    # ── Model 1: SVR + FBTS (main) ──
    logger.info("\n--- Model 1: SVR + FBTS (main) ---")
    preds, folds, overall = run_cv_model(
        fbts, ages, build_svr_pipeline(), SVR_GRID, strata, "FBTS_SVR")
    all_preds["pred_fbts_svr"] = preds
    all_preds["bag_fbts"] = preds - ages
    all_metrics.append({"model": "FBTS_SVR", **overall})
    for i, fm in enumerate(folds.iterrows()):
        all_fold_metrics.append({"model": "FBTS_SVR", "fold": i, **fm[1].to_dict()})

    # ── Model 2: Ridge + FBTS ──
    logger.info("\n--- Model 2: Ridge + FBTS ---")
    preds_ridge, folds, overall = run_cv_model(
        fbts, ages, build_ridge_pipeline(), RIDGE_GRID, strata, "Ridge_FBTS")
    all_preds["pred_ridge_fbts"] = preds_ridge
    all_metrics.append({"model": "Ridge_FBTS", **overall})
    for i, fm in enumerate(folds.iterrows()):
        all_fold_metrics.append({"model": "Ridge_FBTS", "fold": i, **fm[1].to_dict()})

    # ── Model 3: Gradient Boosting + FBTS ──
    logger.info("\n--- Model 3: GBR + FBTS ---")
    preds_gbr, folds, overall = run_cv_model(
        fbts, ages, build_gbr_pipeline(), GBR_GRID, strata, "GBR_FBTS")
    all_preds["pred_gbr_fbts"] = preds_gbr
    all_metrics.append({"model": "GBR_FBTS", **overall})
    for i, fm in enumerate(folds.iterrows()):
        all_fold_metrics.append({"model": "GBR_FBTS", "fold": i, **fm[1].to_dict()})

    # ── Model 4: SVR + PSD ──
    logger.info("\n--- Model 4: SVR + PSD ---")
    preds, folds, overall = run_cv_model(
        psd, ages, build_svr_psd_pipeline(), SVR_GRID_PSD, strata, "SVR_PSD")
    all_preds["pred_svr_psd"] = preds
    all_metrics.append({"model": "SVR_PSD", **overall})
    for i, fm in enumerate(folds.iterrows()):
        all_fold_metrics.append({"model": "SVR_PSD", "fold": i, **fm[1].to_dict()})

    # ── Model 5: SVR + DE ──
    logger.info("\n--- Model 5: SVR + DE ---")
    preds, folds, overall = run_cv_model(
        de, ages, build_svr_psd_pipeline(), SVR_GRID_PSD, strata, "SVR_DE")
    all_preds["pred_svr_de"] = preds
    all_metrics.append({"model": "SVR_DE", **overall})
    for i, fm in enumerate(folds.iterrows()):
        all_fold_metrics.append({"model": "SVR_DE", "fold": i, **fm[1].to_dict()})

    # ── Model 6: Alpha peak frequency (simple age marker) ──
    if alpha_peak is not None:
        logger.info("\n--- Model 6: Alpha Peak Frequency ---")
        X_ap = alpha_peak.reshape(-1, 1)
        preds_ap, folds, overall = run_cv_model(
            X_ap, ages, build_svr_psd_pipeline(), SVR_GRID_PSD, strata,
            "AlphaPeak")
        all_preds["pred_alpha_peak"] = preds_ap
        all_metrics.append({"model": "AlphaPeak", **overall})
        for i, fm in enumerate(folds.iterrows()):
            all_fold_metrics.append(
                {"model": "AlphaPeak", "fold": i, **fm[1].to_dict()})

    # ── Model 7-11: SVR + single-band (ablation) ──
    for b, band_name in enumerate(BAND_ORDER):
        logger.info("\n--- Model %d: SVR + %s band ---", 7 + b, band_name)
        X_band = tangent[:, b, :]
        X_band[np.isnan(X_band)] = 0.0
        X_band[np.isinf(X_band)] = 0.0
        preds, folds, overall = run_cv_model(
            X_band, ages, build_svr_pipeline(), SVR_GRID, strata,
            f"SVR_{band_name}")
        all_preds[f"pred_svr_{band_name}"] = preds
        all_metrics.append({"model": f"SVR_{band_name}", **overall})
        for i, fm in enumerate(folds.iterrows()):
            all_fold_metrics.append(
                {"model": f"SVR_{band_name}", "fold": i, **fm[1].to_dict()})

    # ── Model 12: Ensemble (weighted average of FBTS_SVR + GBR + AlphaPeak) ──
    logger.info("\n--- Model 12: Ensemble ---")
    # simple average of FBTS_SVR + GBR, then blend with alpha peak
    if "pred_gbr_fbts" not in all_preds.columns:
        logger.warning("  GBR predictions not available, skipping ensemble")
    else:
        ens_preds = (all_preds["pred_fbts_svr"].values +
                     all_preds["pred_gbr_fbts"].values) / 2
        if "pred_alpha_peak" in all_preds.columns:
            # blend 70% FBTS avg + 30% alpha peak
            ens_preds = 0.7 * ens_preds + 0.3 * all_preds["pred_alpha_peak"].values
        all_preds["pred_ensemble"] = ens_preds
        # Create ensemble BAG for consistent downstream usage (figures,
        # statistics).  Without this, step05 would fall back to bag_fbts
        # even when the primary model is the ensemble.
        all_preds["bag_ensemble"] = ens_preds - ages
        ens_metrics = compute_metrics(ages, ens_preds)
        all_metrics.append({"model": "Ensemble", **ens_metrics})
        logger.info("  [Ensemble] COR=%.3f RMSE=%.3f MAE=%.3f",
                    ens_metrics["cor"], ens_metrics["rmse"], ens_metrics["mae"])

        # Compute per-fold ensemble metrics using the same CV split
        min_class = pd.Series(strata).value_counts().min()
        n_outer = min(CV_FOLDS, min_class)
        n_outer = max(2, n_outer)
        outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True,
                                   random_state=RANDOM_STATE)
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(fbts, strata)):
            fm = compute_metrics(ages[test_idx], ens_preds[test_idx])
            all_fold_metrics.append({"model": "Ensemble", "fold": fold, **fm})

    # ── Save CV results ──
    all_preds.to_csv(RESULT_DIR / "cv5_predictions.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(RESULT_DIR / "cv5_metrics.csv", index=False)
    pd.DataFrame(all_fold_metrics).to_csv(
        RESULT_DIR / "cv5_fold_metrics.csv", index=False)
    logger.info("\nCV5 results saved to %s", RESULT_DIR)

    # ── LOSO (optional) ──
    if args.loso:
        logger.info("\n--- LOSO: SVR + FBTS ---")
        loso_preds, loso_metrics = run_loso(
            fbts, ages, build_svr_pipeline(), SVR_GRID)
        loso_df = pd.DataFrame({
            "subject": sids, "age": ages,
            "pred": loso_preds, "bag": loso_preds - ages,
        })
        loso_df.to_csv(RESULT_DIR / "loso_predictions.csv", index=False)
        pd.DataFrame([{"model": "FBTS_SVR_LOSO", **loso_metrics}]).to_csv(
            RESULT_DIR / "loso_metrics.csv", index=False)
        logger.info("LOSO results saved.")

    # ── Feature importance (for Figure 9) ──
    logger.info("\n--- Feature importance analysis ---")
    # fit on full data to get selected features
    gs_full = GridSearchCV(build_svr_pipeline(), SVR_GRID, cv=INNER_CV_FOLDS,
                           scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs_full.fit(fbts, ages)
    fs = gs_full.best_estimator_.named_steps["fs"]
    selected_mask = fs.get_support()
    ts_dim = tangent.shape[2]
    band_counts = {}
    for b, band_name in enumerate(BAND_ORDER):
        start = b * ts_dim
        end = (b + 1) * ts_dim
        n_sel = selected_mask[start:end].sum()
        band_counts[band_name] = int(n_sel)
        logger.info("  %s: %d / %d features selected", band_name, n_sel, ts_dim)
    pd.DataFrame([
        {"band": k, "n_selected": v} for k, v in band_counts.items()
    ]).to_csv(RESULT_DIR / "feature_importance.csv", index=False)

    logger.info("\nStep 3 complete.")


if __name__ == "__main__":
    main()
