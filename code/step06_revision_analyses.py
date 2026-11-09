# -*- coding: utf-8 -*-
"""Revision analyses for the Frontiers in Neuroinformatics re-review.

A. SVR-PSD with log10 transform + unified full pipeline (SelectKBest -> Scaler -> PCA -> SVR)
B. SVR-DE with unified full pipeline
C. SelectKBest cross-fold stability + band-wise permutation importance (FBTS main model)
D. Exclusion sensitivity: retained (status=ok) vs excluded (failed) — age / sex / p-factor
E. Broadband-amplitude control for the alpha-theta MGD correlation

Outputs -> results/analysis/revision_*.csv
Run:  python3 code/step06_revision_analyses.py [--parts A,B,C,D,E]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mannwhitneyu, chi2_contingency, ttest_ind

from config import (FEATURE_DIR, RESULT_DIR, META_DIR, RAW_BASE, RELEASES,
                    REST_TASK, BAND_ORDER, CV_FOLDS, RANDOM_STATE)
from step03_brain_age_model import (make_age_strata, build_svr_pipeline,
                                    SVR_GRID, run_cv_model, compute_metrics,
                                    INNER_CV_FOLDS)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger("step06_revision")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(h)

TANGENT_DIM = 8256          # 128*129/2 per band
FBTS_DIM = TANGENT_DIM * 5  # 41280
N_PERM_IMPORTANCE = 20      # permutations per band per fold


# ── data loading ─────────────────────────────────────────────────────────

def load_merged_features():
    files = sorted(FEATURE_DIR.glob(f"{REST_TASK}_features_R*.npz"))
    if not files:
        files = [FEATURE_DIR / f"{REST_TASK}_features_merged.npz"]
    logger.info("Merging %d feature files", len(files))
    datas = [np.load(f, allow_pickle=True) for f in files]
    def cat(key):
        return np.concatenate([d[key] for d in datas])
    out = {
        "ages": cat("ages").astype(float),
        "sids": cat("subject_ids").astype(str),
        "psd": cat("psd").astype(float),
        "de": cat("de").astype(float),
        "fbts": cat("fbts").astype(float),
    }
    logger.info("merged n=%d, psd min=%.3e, psd max=%.3e",
                len(out["ages"]), out["psd"].min(), out["psd"].max())
    return out


# ── A/B: unified-pipeline reruns ─────────────────────────────────────────

def run_part_A_B(feat):
    ages = feat["ages"]
    strata = make_age_strata(ages, CV_FOLDS)
    rows = []
    for name, X in (("SVR_PSD_LOG_FULL", np.log10(np.maximum(feat["psd"], 1e-12))),
                    ("SVR_DE_FULL", feat["de"])):
        logger.info("[%s] unified pipeline rerun", name)
        preds, fold_df, overall = run_cv_model(X, ages, build_svr_pipeline(),
                                               SVR_GRID, strata, name)
        for fi, fr in enumerate(fold_df.itertuples(), 1):
            rows.append({"model": name, "fold": fi,
                         "cor": fr.cor, "rmse": fr.rmse, "mae": fr.mae})
        rows.append({"model": name, "fold": -1, "cor": overall["cor"],
                     "rmse": overall["rmse"], "mae": overall["mae"]})
    pd.DataFrame(rows).to_csv(RESULT_DIR / "revision_psd_de_full.csv", index=False)
    logger.info("A/B done -> revision_psd_de_full.csv")


# ── C: SelectKBest stability + band permutation importance ──────────────

def run_part_C(feat):
    X, y = feat["fbts"], feat["ages"]
    strata = make_age_strata(y, CV_FOLDS)
    # replicate run_cv_model fold geometry exactly
    min_class = pd.Series(strata).value_counts().min()
    n_outer = max(2, min(CV_FOLDS, min_class))
    n_inner = min(INNER_CV_FOLDS, max(2, len(y) * (n_outer - 1) // n_outer // 2))
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True,
                               random_state=RANDOM_STATE)
    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_STATE)
    from step03_brain_age_model import build_svr_pipeline as bsp
    from sklearn.feature_selection import SelectKBest
    from step03_brain_age_model import SVR_GRID as grid

    stab_rows, imp_rows = [], []
    for fold, (tr, te) in enumerate(outer_cv.split(X, strata), 1):
        logger.info("[C] fold %d: grid search", fold)
        gs = GridSearchCV_local(bsp(), grid, inner_cv, X[tr], y[tr])
        best = gs.best_estimator_
        support = best.named_steps["fs"].get_support()
        k_best = gs.best_params_.get("fs__k", int(support.sum()))
        for bi, band in enumerate(BAND_ORDER):
            n_sel = int(support[bi * TANGENT_DIM:(bi + 1) * TANGENT_DIM].sum())
            stab_rows.append({"fold": fold, "band": band,
                              "n_selected": n_sel, "k_best": k_best})
        # permutation importance on the held-out fold
        pred_full = np.clip(best.predict(X[te]), 3.0, 22.0)
        r_full = pearsonr(y[te], pred_full)[0]
        rng = np.random.default_rng(1000 + fold)
        band_cols_all = np.arange(FBTS_DIM)
        for bi, band in enumerate(BAND_ORDER):
            cols = band_cols_all[bi * TANGENT_DIM:(bi + 1) * TANGENT_DIM]
            drops = []
            for _ in range(N_PERM_IMPORTANCE):
                perm = rng.permutation(cols)
                Xp = X[te].copy()
                Xp[:, cols] = Xp[:, perm]
                pred_p = np.clip(best.predict(Xp), 3.0, 22.0)
                drops.append(r_full - pearsonr(y[te], pred_p)[0])
            imp_rows.append({"fold": fold, "band": band,
                             "r_full": r_full,
                             "mean_r_drop": float(np.mean(drops)),
                             "sd_r_drop": float(np.std(drops))})
            logger.info("[C] fold %d band %s: mean r-drop %.4f",
                        fold, band, np.mean(drops))
    pd.DataFrame(stab_rows).to_csv(RESULT_DIR / "revision_kbest_stability.csv",
                                   index=False)
    pd.DataFrame(imp_rows).to_csv(RESULT_DIR / "revision_band_perm_importance.csv",
                                  index=False)
    logger.info("C done -> revision_kbest_stability.csv, revision_band_perm_importance.csv")


def GridSearchCV_local(model, grid, inner_cv, X_tr, y_tr):
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model, grid, cv=inner_cv,
                      scoring="neg_root_mean_squared_error",
                      n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    return gs


# ── D: exclusion sensitivity ─────────────────────────────────────────────

def load_pfactor_map():
    pmap = {}
    for r, info in RELEASES.items():
        p = RAW_BASE / f"HBN-Release{r}-{info['dataset']}" / "participants.tsv"
        if not p.exists():
            logger.warning("missing %s", p)
            continue
        df = pd.read_csv(p, sep="\t", low_memory=False)
        sid_col = ("participant_id" if "participant_id" in df.columns
                   else df.columns[0])
        for col in ("p_factor",):
            if col not in df.columns:
                logger.warning("no %s column in %s", col, p.name)
                continue
        for sid, pf in zip(df[sid_col].astype(str), df.get("p_factor")):
            try:
                v = float(pf)
            except (TypeError, ValueError):
                v = np.nan
            if sid not in pmap or (np.isnan(pmap[sid]) and not np.isnan(v)):
                pmap[sid] = v
    logger.info("p_factor map: %d subjects", len(pmap))
    return pmap


def run_part_D():
    qc = pd.read_csv(META_DIR / "qc_preprocessing.csv")
    qc = qc[qc["task"] == REST_TASK].copy()
    pmap = load_pfactor_map()
    qc["p_factor"] = qc["subject"].astype(str).map(pmap)
    ret = qc[qc["status"] == "ok"]
    exc = qc[qc["status"] != "ok"]
    logger.info("retained n=%d, excluded n=%d", len(ret), len(exc))

    rows = []
    # age
    a1, a0 = ret["age"].values, exc["age"].values
    t_p = ttest_ind(a1, a0, equal_var=False, nan_policy="omit").pvalue
    u_p = mannwhitneyu(a1, a0).pvalue
    rows.append({"metric": "age_years", "stat": "mean_sd",
                 "retained": f"{np.nanmean(a1):.2f}±{np.nanstd(a1):.2f}",
                 "excluded": f"{np.nanmean(a0):.2f}±{np.nanstd(a0):.2f}",
                 "pWelch": t_p, "pMWU": u_p})
    # sex (proportion male)
    m1 = (ret["sex"] == "M").mean()
    m0 = (exc["sex"] == "M").mean()
    tab = pd.crosstab(qc["status"], qc["sex"]).values
    chi_p = chi2_contingency(tab).pvalue
    rows.append({"metric": "sex_prop_male", "stat": "proportion",
                 "retained": f"{m1:.3f}", "excluded": f"{m0:.3f}",
                 "pWelch": np.nan, "pMWU": np.nan, "pChi2": chi_p})
    # p-factor
    p1 = ret["p_factor"].values
    p0 = exc["p_factor"].values
    p1v, p0v = p1[~np.isnan(p1)], p0[~np.isnan(p0)]
    u_p2 = mannwhitneyu(p1v, p0v).pvalue
    t_p2 = ttest_ind(p1v, p0v, equal_var=False).pvalue
    rows.append({"metric": "cbcl_p_factor", "stat": "mean_sd",
                 "retained": f"{np.mean(p1v):.3f}±{np.std(p1v):.3f}",
                 "excluded": f"{np.mean(p0v):.3f}±{np.std(p0v):.3f}",
                 "pWelch": t_p2, "pMWU": u_p2,
                 "n_valid_retained": len(p1v), "n_valid_excluded": len(p0v),
                 "miss_rate_retained": 1 - len(p1v) / len(ret),
                 "miss_rate_excluded": 1 - len(p0v) / len(exc)})
    pd.DataFrame(rows).to_csv(RESULT_DIR / "revision_exclusion_sensitivity.csv",
                              index=False)
    logger.info("D done -> revision_exclusion_sensitivity.csv")


# ── E: broadband control ────────────────────────────────────────────────

def run_part_E(feat):
    bb = np.log10(np.maximum(feat["psd"], 1e-12)).mean(axis=1)
    bb_df = pd.DataFrame({"subject": feat["sids"], "broadband": bb})
    mgd = pd.read_csv(RESULT_DIR / "mgd_results.csv")
    m = mgd.merge(bb_df, on="subject", how="inner")
    logger.info("E: merged n=%d", len(m))
    def resid(v, z):
        b = np.polyfit(z, v, 1)
        return v - np.polyval(b, z)
    ok = m[["mgd_z_alpha", "mgd_z_theta", "broadband"]].notna().all(axis=1).values
    ra = resid(m.loc[ok, "mgd_z_alpha"].values, m.loc[ok, "broadband"].values)
    rt = resid(m.loc[ok, "mgd_z_theta"].values, m.loc[ok, "broadband"].values)
    pr_r, pr_p = pearsonr(ra, rt)
    r_bb_a, p_bb_a = pearsonr(m.loc[ok, "broadband"], m.loc[ok, "mgd_z_alpha"])
    r_bb_t, p_bb_t = pearsonr(m.loc[ok, "broadband"], m.loc[ok, "mgd_z_theta"])
    r_ab, p_ab = pearsonr(m.loc[ok, "mgd_z_alpha"], m.loc[ok, "mgd_z_theta"])
    rows = [{"pair": "alpha_theta_mgd_z", "n": int(ok.sum()),
             "r_raw": float(r_ab), "p_raw": float(p_ab),
             "r_partial_broadband": float(pr_r), "p_partial": float(pr_p),
             "r_broadband_alpha": float(r_bb_a), "p_broadband_alpha": float(p_bb_a),
             "r_broadband_theta": float(r_bb_t), "p_broadband_theta": float(p_bb_t)}]
    pd.DataFrame(rows).to_csv(RESULT_DIR / "revision_broadband_control.csv",
                              index=False)
    logger.info("E done: raw r=%.3f -> partial r=%.3f | bb~alpha %.3f, bb~theta %.3f",
                r_ab, pr_r, r_bb_a, r_bb_t)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="D,E,A,B,C",
                    help="order-sensitive list, e.g. D,E then A,B,C separately")
    args = ap.parse_args()
    parts = [p.strip().upper() for p in args.parts.split(",") if p.strip()]

    feat = None
    for p in parts:
        logger.info("=" * 60)
        logger.info("PART %s", p)
        if p in ("A", "B", "C", "E") and feat is None:
            feat = load_merged_features()
        if p in ("A", "B"):
            run_part_A_B(feat)
        elif p == "C":
            run_part_C(feat)
        elif p == "D":
            run_part_D()
        elif p == "E":
            run_part_E(feat)
    logger.info("step06 complete.")


if __name__ == "__main__":
    main()
