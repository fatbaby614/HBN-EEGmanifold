#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4b: Manifold Geometry Deviation (MGD) Analysis

Computes individual deviation from the age-normative trajectory on the
Riemannian SPD manifold and correlates it with CBCL psychopathology scores.

This is a more direct biomarker than BAG (Brain Age Gap) because:
1. No intermediate brain-age model (no error propagation)
2. No regression-to-the-mean (normative trajectory is age-aware by design)
3. Leave-one-out estimation avoids circular bias
"""

import numpy as np
import pandas as pd
from scipy.linalg import logm, expm, sqrtm
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, norm
from sklearn.metrics import roc_curve, roc_auc_score
import argparse

try:
    from pyriemann.geometry.distance import distance_riemann
except ImportError:
    from pyriemann.utils.distance import distance_riemann

from config import (
    RESULT_DIR, FEATURE_DIR, BAND_ORDER, REST_TASK,
    MGD_BANDWIDTH, MGD_N_AGE_BINS, MGD_N_PERMUTATIONS, MGD_N_BOOTSTRAP,
    RISK_PERCENTILE_HIGH, RISK_PERCENTILE_LOW, RANDOM_STATE,
)
from utils import setup_logging
from step04_statistics import load_merged_features, fdr_correct, bootstrap_auc_ci, partial_corr

logger = setup_logging("step04b_mgd")

CBCL_VARS = ["p_factor", "attention", "internalizing", "externalizing"]


# ═══════════════════════════════════════════════════════════════════════════
#  Riemannian Geometry Utilities
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_kernel_weights(target_age, ages, bandwidth):
    """Gaussian kernel weights, normalized to sum=1."""
    w = np.exp(-0.5 * ((ages - target_age) / bandwidth) ** 2)
    w /= w.sum()
    return w


def weighted_log_euclidean_mean(covmats, weights, log_covs=None):
    """Weighted log-Euclidean mean on SPD manifold (fast approximation).

    C_mean = expm( sum_j w_j * logm(C_j) )

    If log_covs are pre-computed, pass them to avoid redundant logm calls.
    """
    if log_covs is not None:
        weighted_log = np.average(log_covs, axis=0, weights=weights)
    else:
        log_covs = np.array([logm(C) for C in covmats])
        weighted_log = np.average(log_covs, axis=0, weights=weights)
    C = expm(weighted_log)
    return (C + C.T) / 2  # ensure symmetric


# ═══════════════════════════════════════════════════════════════════════════
#  MGD Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_mgd_for_band(covmats_band, ages, bandwidth=MGD_BANDWIDTH):
    """Compute MGD for one frequency band using leave-one-out (LOO).

    Parameters
    ----------
    covmats_band : np.ndarray, shape (n, 128, 128)
    ages : np.ndarray, shape (n,)
    bandwidth : float, Gaussian kernel bandwidth in years

    Returns
    -------
    mgd : np.ndarray, shape (n,), Riemannian distances
    normative_covs : np.ndarray, shape (n, 128, 128), LOO normative covariance
    """
    n = len(ages)

    # Pre-compute logm for all subjects (major speedup)
    logger.info("  Pre-computing logm for %d matrices...", n)
    log_covs = np.array([logm(C) for C in covmats_band])

    mgd = np.zeros(n)
    normative_covs = np.zeros((n, *covmats_band.shape[1:]))

    for i in range(n):
        if (i + 1) % 100 == 0:
            logger.info("  MGD: %d/%d", i + 1, n)

        # LOO: exclude subject i
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        weights = gaussian_kernel_weights(ages[i], ages[mask], bandwidth)

        # Weighted log-Euclidean mean (fast, using pre-computed logm)
        normative_cov = weighted_log_euclidean_mean(
            covmats_band[mask], weights, log_covs=log_covs[mask])

        normative_covs[i] = normative_cov

        # Full Riemannian distance (more accurate than log-Euclidean)
        mgd[i] = distance_riemann(covmats_band[i], normative_cov)

    return mgd, normative_covs


def zscore_by_age(mgd, ages, n_bins=MGD_N_AGE_BINS):
    """Z-score MGD within age bins to remove age-varying dispersion."""
    # Degenerate-input guard: if ages have < 2 unique values (e.g. all
    # subjects are 10 years old), pd.qcut returns an all-NaN array and
    # every mgd_z entry stays NaN — silently breaking all downstream
    # analyses.  Return z = 0 (subjects are "average") in that case.
    ages_arr = np.asarray(ages, dtype=float)
    if len(np.unique(ages_arr)) < 2 or n_bins < 2:
        logger.warning("  zscore_by_age: ages have < 2 unique values, "
                       "returning zeros")
        return np.zeros_like(mgd, dtype=float)
    try:
        age_bins = pd.qcut(ages_arr, n_bins, labels=False, duplicates='drop')
    except ValueError as exc:
        logger.warning("  zscore_by_age: pd.qcut failed (%s), "
                       "falling back to zeros", exc)
        return np.zeros_like(mgd, dtype=float)
    # Initialise with NaN so that subjects in degenerate bins (sigma == 0
    # or all-NaN inputs) are flagged as missing rather than being silently
    # treated as "average" (z = 0).
    mgd_z = np.full_like(mgd, np.nan, dtype=float)
    # Skip NaN bins (occurs when pd.qcut emits NaN for duplicates)
    for b in np.unique(age_bins[~np.isnan(age_bins)]):
        mask = age_bins == b
        if not mask.any():
            continue
        mu = mgd[mask].mean()
        sigma = mgd[mask].std()
        if sigma > 0:
            mgd_z[mask] = (mgd[mask] - mu) / sigma
    return mgd_z


def compute_all_mgd(covmats, ages, bandwidth=MGD_BANDWIDTH):
    """Compute MGD for all frequency bands + combined.

    Returns
    -------
    mgd_raw : dict[str, np.ndarray]
    mgd_z : dict[str, np.ndarray]
    normative_covs : dict[str, np.ndarray]
    """
    mgd_raw = {}
    mgd_z = {}
    normative_covs = {}

    for b, band_name in enumerate(BAND_ORDER):
        logger.info("Computing MGD for %s band...", band_name)
        mgd, ncovs = compute_mgd_for_band(
            covmats[:, b, :, :], ages, bandwidth=bandwidth)
        mgd_raw[band_name] = mgd
        mgd_z[band_name] = zscore_by_age(mgd, ages)
        normative_covs[band_name] = ncovs
        logger.info("  %s: MGD=%.4f±%.4f, z=%.4f±%.4f",
                    band_name, mgd.mean(), mgd.std(),
                    mgd_z[band_name].mean(), mgd_z[band_name].std())

    # Combined MGD (equal-weight average of z-scored band MGDs)
    # Use nanmean so that any band that has NaN z-scores (e.g. due to a
    # degenerate age bin with sigma == 0) does not contaminate every
    # subject in the combined band with NaN.
    mgd_z["combined"] = np.nanmean(
        [mgd_z[b] for b in BAND_ORDER], axis=0)
    logger.info("  combined: z=%.4f±%.4f (nanmean across bands)",
                np.nanmean(mgd_z["combined"]),
                np.nanstd(mgd_z["combined"]))

    return mgd_raw, mgd_z, normative_covs


# ═══════════════════════════════════════════════════════════════════════════
#  Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════════

def _residualize(x, covariates):
    """Regress out covariates from x (linear)."""
    from numpy.linalg import lstsq
    X = np.column_stack([covariates, np.ones_like(covariates[:, 0])])
    coef, *_ = lstsq(X, x, rcond=None)
    return x - X @ coef


def analyse_mgd_correlations(mgd_z, ages, sexes, cbcl_data):
    """Analysis A: Partial correlation of MGD with CBCL (controlling age, sex)."""
    band_names = BAND_ORDER + ["combined"]
    sex_numeric = (sexes == "M").astype(float)
    covariates = np.column_stack([ages, sex_numeric])

    rows = []
    for band_name in band_names:
        mgd = mgd_z[band_name]
        for var in CBCL_VARS:
            vals = cbcl_data[var]
            mask = ~np.isnan(vals) & ~np.isnan(mgd)
            if mask.sum() < 20:
                continue

            # Use partial_corr for correct p-value degrees of freedom
            # (n-k-2 where k=n_covariates, not n-2 from pearsonr)
            r_p, p_p = partial_corr(mgd[mask], vals[mask], covariates[mask])
            rho_s, p_s = spearmanr(mgd[mask], vals[mask])

            rows.append({
                "band": band_name, "variable": var,
                "r_pearson": r_p, "p_pearson": p_p,
                "r_spearman": rho_s, "p_spearman": p_s,
                "n": int(mask.sum()),
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["p_fdr"] = fdr_correct(result["p_pearson"].values)
        result["sig"] = result["p_fdr"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01
            else "*" if p < 0.05 else "ns")

        for _, row in result.iterrows():
            logger.info("  MGD(%s) ~ %s: r=%.3f, p=%.4f, fdr=%.4f %s (n=%d)",
                        row["band"], row["variable"],
                        row["r_pearson"], row["p_pearson"],
                        row["p_fdr"], row["sig"], row["n"])
    return result


def analyse_mgd_roc(mgd_z, p_factor,
                    n_perm=MGD_N_PERMUTATIONS, n_boot=MGD_N_BOOTSTRAP):
    """Analysis B: ROC classification (high-risk vs low-risk)."""
    band_names = BAND_ORDER + ["combined"]
    mask = ~np.isnan(p_factor)
    pf = p_factor[mask]

    hi_thr = np.percentile(pf, RISK_PERCENTILE_HIGH)
    lo_thr = np.percentile(pf, RISK_PERCENTILE_LOW)
    hi_mask = pf > hi_thr
    lo_mask = pf < lo_thr

    if hi_mask.sum() < 5 or lo_mask.sum() < 5:
        logger.warning("  Insufficient risk groups for ROC")
        return pd.DataFrame()

    roc_labels = np.concatenate([np.ones(hi_mask.sum()),
                                 np.zeros(lo_mask.sum())])
    rows = []

    for band_name in band_names:
        mgd = mgd_z[band_name][mask]
        # MGD z-scores can be NaN when an age bin has sigma==0 (degenerate
        # bin).  Filter those subjects out of BOTH the risk groups and the
        # labels so roc_auc_score does not receive NaN scores.
        valid = ~np.isnan(mgd)
        hi_v = hi_mask & valid
        lo_v = lo_mask & valid
        if hi_v.sum() < 5 or lo_v.sum() < 5:
            logger.warning("  ROC (%s): insufficient valid samples after NaN "
                           "removal (hi=%d, lo=%d), skipping",
                           band_name, hi_v.sum(), lo_v.sum())
            continue
        roc_labels_b = np.concatenate([np.ones(hi_v.sum()),
                                       np.zeros(lo_v.sum())])
        roc_scores = np.concatenate([mgd[hi_v], mgd[lo_v]])

        auc = roc_auc_score(roc_labels_b, roc_scores)
        ci_lo, ci_hi = bootstrap_auc_ci(roc_scores, roc_labels_b, n_boot=n_boot)

        # Permutation test (two-sided).  Use the proportion of permutations with
        # AUC at least as extreme as the observed value (above OR below the
        # chance level).  Without the two-sided correction an AUC < 0.5
        # would be incorrectly classified as "non-significant" simply
        # because the chance-level AUC = 0.5 is symmetric.
        rng = np.random.RandomState(RANDOM_STATE)
        perm_aucs = []
        for _ in range(n_perm):
            perm_labels = rng.permutation(roc_labels_b)
            if len(np.unique(perm_labels)) < 2:
                continue
            perm_aucs.append(roc_auc_score(perm_labels, roc_scores))
        perm_arr = np.array(perm_aucs)
        # Distance from 0.5 (chance level) for observed and each permutation
        d_obs = abs(auc - 0.5)
        d_perm = np.abs(perm_arr - 0.5)
        p_perm = np.mean(d_perm >= d_obs)

        fpr, tpr, thresholds = roc_curve(roc_labels_b, roc_scores)
        best_idx = np.argmax(tpr - fpr)

        rows.append({
            "band": band_name, "auc": auc,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_perm": p_perm,
            "sensitivity": tpr[best_idx],
            "specificity": 1 - fpr[best_idx],
            "best_cutoff": thresholds[best_idx],
        })
        logger.info("  ROC (%s): AUC=%.3f (95%%CI %.3f-%.3f), p_perm=%.4f",
                    band_name, auc, ci_lo, ci_hi, p_perm)

    result = pd.DataFrame(rows)
    result["p_fdr"] = fdr_correct(result["p_perm"].values)

    # Save best ROC curve data
    if len(result) > 0:
        best_band = result.loc[result["auc"].idxmax(), "band"]
        mgd_best = mgd_z[best_band][mask]
        # Re-apply NaN filter for the best band (may differ from the band
        # used in the last loop iteration).
        valid_best = ~np.isnan(mgd_best)
        hi_b = hi_mask & valid_best
        lo_b = lo_mask & valid_best
        roc_labels_best = np.concatenate([np.ones(hi_b.sum()),
                                          np.zeros(lo_b.sum())])
        roc_scores_best = np.concatenate([mgd_best[hi_b],
                                          mgd_best[lo_b]])
        fpr, tpr, thresholds = roc_curve(roc_labels_best, roc_scores_best)
        pd.DataFrame({"fpr": fpr, "tpr": tpr,
                      "thresholds": thresholds,
                      "band": best_band}).to_csv(
            RESULT_DIR / "mgd_roc_curve.csv", index=False)

    return result


def analyse_mgd_age_stratified(mgd_z, ages, cbcl_data):
    """Analysis C: Age-stratified MGD-CBCL correlations."""
    age_bins = [(5, 8), (9, 12), (13, 16), (17, 21)]
    band_names = ["alpha", "combined"]

    rows = []
    for band_name in band_names:
        mgd = mgd_z[band_name]
        for lo_age, hi_age in age_bins:
            age_mask = (ages >= lo_age) & (ages < hi_age)
            for var in CBCL_VARS:
                vals = cbcl_data[var]
                mask = age_mask & ~np.isnan(vals) & ~np.isnan(mgd)
                if mask.sum() > 15:
                    r, p = pearsonr(mgd[mask], vals[mask])
                    rows.append({
                        "band": band_name,
                        "age_group": f"{lo_age}-{hi_age}",
                        "variable": var,
                        "r": r, "p": p, "n": int(mask.sum()),
                    })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["p_fdr"] = fdr_correct(result["p"].values)
    return result


def analyse_mgd_sex(mgd_z, ages, sexes, cbcl_data):
    """Analysis D: Sex differences in MGD and MGD-CBCL associations."""
    band_names = ["alpha", "combined"]
    male_mask = sexes == "M"
    female_mask = sexes == "F"

    rows = []
    for band_name in band_names:
        mgd = mgd_z[band_name]

        # 1. Distribution difference
        if male_mask.sum() > 10 and female_mask.sum() > 10:
            u, p = mannwhitneyu(mgd[male_mask], mgd[female_mask])
            # Use the standard Mann-Whitney Z = (U - mu_U) / sigma_U formula
            # rather than `norm.ppf(1 - p/2)`, which loses information when
            # p is very small and breaks the symmetry between two-sided and
            # one-sided tests.
            n_m = int(male_mask.sum())
            n_f = int(female_mask.sum())
            mu_u = n_m * n_f / 2.0
            sigma_u = np.sqrt(n_m * n_f * (n_m + n_f + 1) / 12.0)
            z_val = (u - mu_u) / sigma_u if sigma_u > 0 else 0.0
            # Cohen's d ≈ 2*z / sqrt(N) for Mann-Whitney; use the standard
            # Mann-Whitney → Cohen's d conversion (2 * Z / sqrt(N)).
            d = 2.0 * z_val / np.sqrt(n_m + n_f)
            rows.append({
                "analysis": "distribution",
                "band": band_name,
                "male_mean": mgd[male_mask].mean(),
                "female_mean": mgd[female_mask].mean(),
                "U": u, "p": p, "cohen_d": d,
            })

        # 2. MGD-CBCL sex difference (Fisher z)
        for var in CBCL_VARS:
            vals = cbcl_data[var]
            for sex_label, sex_mask in [("M", male_mask), ("F", female_mask)]:
                mask = sex_mask & ~np.isnan(vals) & ~np.isnan(mgd)
                if mask.sum() > 20:
                    r, p = pearsonr(mgd[mask], vals[mask])
                    rows.append({
                        "analysis": "correlation",
                        "band": band_name, "sex": sex_label,
                        "variable": var, "r": r, "p": p,
                        "n": int(mask.sum()),
                    })

            m_mask = male_mask & ~np.isnan(vals) & ~np.isnan(mgd)
            f_mask = female_mask & ~np.isnan(vals) & ~np.isnan(mgd)
            if m_mask.sum() > 20 and f_mask.sum() > 20:
                r_m, _ = pearsonr(mgd[m_mask], vals[m_mask])
                r_f, _ = pearsonr(mgd[f_mask], vals[f_mask])
                # Clamp to (-1, 1) — Fisher z diverges as |r| → 1
                r_m_c = max(min(r_m, 0.9999), -0.9999)
                r_f_c = max(min(r_f, 0.9999), -0.9999)
                z_m = 0.5 * np.log((1 + r_m_c) / (1 - r_m_c))
                z_f = 0.5 * np.log((1 + r_f_c) / (1 - r_f_c))
                se = np.sqrt(1.0 / (m_mask.sum() - 3) +
                             1.0 / (f_mask.sum() - 3))
                z_diff = (z_m - z_f) / se
                p_diff = 2 * (1 - norm.cdf(abs(z_diff)))
                rows.append({
                    "analysis": "sex_difference",
                    "band": band_name, "variable": var,
                    "r_male": r_m, "r_female": r_f,
                    "z_diff": z_diff, "p_diff": p_diff,
                })

    return pd.DataFrame(rows)


def analyse_mgd_vs_bag(mgd_z, bag, ages, cbcl_data):
    """Supplementary: Compare MGD and BAG as biomarkers."""
    mask = ~np.isnan(bag)
    rows = []
    for var in CBCL_VARS:
        vals = cbcl_data[var]
        m = mask & ~np.isnan(vals)
        if m.sum() < 2:
            logger.warning("  analyse_mgd_vs_bag: insufficient samples for %s (n=%d)",
                           var, m.sum())
            rows.append({"variable": var, "r_mgd_alpha": np.nan,
                         "p_mgd_alpha": np.nan, "r_bag_raw": np.nan,
                         "p_bag_raw": np.nan, "n": int(m.sum())})
            continue

        # MGD vs CBCL (alpha)
        if "alpha" in mgd_z:
            r_mgd, p_mgd = pearsonr(mgd_z["alpha"][m], vals[m])
        else:
            r_mgd, p_mgd = np.nan, np.nan

        # BAG vs CBCL (raw)
        r_bag, p_bag = pearsonr(bag[m], vals[m])

        rows.append({
            "variable": var,
            "r_mgd_alpha": r_mgd, "p_mgd_alpha": p_mgd,
            "r_bag_raw": r_bag, "p_bag_raw": p_bag,
            "n": int(m.sum()),
        })

    # MGD-BAG correlation
    m = mask & ~np.isnan(mgd_z["alpha"])
    if m.sum() >= 2:
        r_mgd_bag, p_mgd_bag = pearsonr(mgd_z["alpha"][m], bag[m])
    else:
        r_mgd_bag, p_mgd_bag = np.nan, np.nan

    return pd.DataFrame(rows), r_mgd_bag, p_mgd_bag


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("Step 4b: Manifold Geometry Deviation (MGD) Analysis")
    logger.info("=" * 60)

    # Load features
    data = load_merged_features(REST_TASK)
    covmats = data["covmats"]
    ages = data["ages"]
    sexes = data["sexes"]
    sids = data["subject_ids"]
    n = len(ages)

    # Load CBCL data from feature file
    cbcl_data = {}
    for var in CBCL_VARS:
        if var in data.files:
            cbcl_data[var] = np.array(data[var], dtype=float)
        else:
            logger.warning("  %s not in feature file, filling NaN", var)
            cbcl_data[var] = np.full(n, np.nan)

    # Try to supplement CBCL from participants.tsv if missing
    n_valid = sum(~np.isnan(cbcl_data.get("p_factor", np.full(n, np.nan))))
    if n_valid < n * 0.5:
        logger.info("  Loading CBCL from participants.tsv files...")
        cbcl_data = _load_cbcl_from_tsv(sids, cbcl_data, n)

    logger.info("Loaded %d subjects, ages %.1f-%.1f", n, ages.min(), ages.max())
    for var in CBCL_VARS:
        nv = sum(~np.isnan(cbcl_data.get(var, np.full(n, np.nan))))
        logger.info("  %s: %d valid", var, nv)

    # ── 1. Compute MGD for all bands ──
    bandwidth = MGD_BANDWIDTH
    logger.info("\n=== Computing MGD (bandwidth=%.1f) ===", bandwidth)
    mgd_raw, mgd_z, normative_covs = compute_all_mgd(covmats, ages, bandwidth)

    # Save MGD results
    mgd_df = pd.DataFrame({"subject": sids, "age": ages, "sex": sexes})
    for band_name in BAND_ORDER + ["combined"]:
        if band_name in mgd_z:
            mgd_df[f"mgd_z_{band_name}"] = mgd_z[band_name]
            mgd_df[f"mgd_raw_{band_name}"] = mgd_raw.get(band_name, np.nan)
    for var in CBCL_VARS:
        mgd_df[var] = cbcl_data.get(var, np.full(n, np.nan))
    mgd_df.to_csv(RESULT_DIR / "mgd_results.csv", index=False)
    logger.info("Saved mgd_results.csv")

    # ── 2. Correlation analysis ──
    logger.info("\n=== MGD-CBCL Correlation Analysis ===")
    corr_result = analyse_mgd_correlations(mgd_z, ages, sexes, cbcl_data)
    corr_result.to_csv(RESULT_DIR / "mgd_correlations.csv", index=False)

    # ── 3. ROC analysis ──
    logger.info("\n=== MGD ROC Analysis ===")
    roc_result = analyse_mgd_roc(mgd_z, cbcl_data["p_factor"])
    roc_result.to_csv(RESULT_DIR / "mgd_roc_summary.csv", index=False)

    # ── 4. Age-stratified analysis ──
    logger.info("\n=== MGD Age-Stratified Analysis ===")
    strat_result = analyse_mgd_age_stratified(mgd_z, ages, cbcl_data)
    strat_result.to_csv(RESULT_DIR / "mgd_age_stratified.csv", index=False)

    # ── 5. Sex differences ──
    logger.info("\n=== MGD Sex Differences ===")
    sex_result = analyse_mgd_sex(mgd_z, ages, sexes, cbcl_data)
    sex_result.to_csv(RESULT_DIR / "mgd_sex_differences.csv", index=False)

    # ── 6. MGD vs BAG comparison ──
    logger.info("\n=== MGD vs BAG Comparison ===")
    bag_path = RESULT_DIR / "cv5_predictions.csv"
    if bag_path.exists():
        pred_df = pd.read_csv(bag_path)
        # Match subjects
        bag_dict = dict(zip(pred_df["subject"], pred_df["bag_fbts"]))
        bag = np.array([bag_dict.get(s, np.nan) for s in sids])
        bag_result, r_mgd_bag, p_mgd_bag = analyse_mgd_vs_bag(
            mgd_z, bag, ages, cbcl_data)
        bag_result.to_csv(RESULT_DIR / "mgd_vs_bag.csv", index=False)
        logger.info("  MGD-BAG correlation (alpha): r=%.3f, p=%.4f",
                    r_mgd_bag, p_mgd_bag)

    # ── 7. Save normative covariances for Figure 6 ──
    np.savez_compressed(
        RESULT_DIR / "mgd_normative_covs.npz",
        **{f"normative_{b}": normative_covs[b] for b in BAND_ORDER},
        ages=ages, sids=sids,
    )

    # ── 8. Cross-band MGD correlation matrix ──
    cross_corr = np.zeros((len(BAND_ORDER), len(BAND_ORDER)))
    for i, bi in enumerate(BAND_ORDER):
        for j, bj in enumerate(BAND_ORDER):
            mask = ~np.isnan(mgd_z[bi]) & ~np.isnan(mgd_z[bj])
            if mask.sum() >= 2:
                cross_corr[i, j] = pearsonr(
                    mgd_z[bi][mask], mgd_z[bj][mask])[0]
            else:
                cross_corr[i, j] = np.nan
    pd.DataFrame(cross_corr, index=BAND_ORDER,
                 columns=BAND_ORDER).to_csv(RESULT_DIR / "mgd_cross_band.csv")

    # ── 9. Bandwidth sensitivity analysis (optional but recommended) ──
    # Computes MGD-age correlations and MGD-CBCL partial correlations
    # across a sweep of Gaussian kernel bandwidths, plus cross-band
    # MGD correlations. Output feeds Figure S1 in step05c.
    try:
        analyse_bandwidth_sensitivity(covmats, ages, sexes, cbcl_data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Bandwidth sensitivity analysis failed (%s); "
                       "Figure S1 will be skipped", exc)

    logger.info("\nStep 4b complete.")


def analyse_bandwidth_sensitivity(covmats, ages, sexes, cbcl_data):
    """Sweep MGD over a range of Gaussian kernel bandwidths and save
    the per-band MGD-age correlations, MGD-CBCL partial correlations,
    and cross-band MGD correlations to CSV.

    The output is consumed by ``step05c_supplementary_figures.figureS1``
    to render the bandwidth-sensitivity panel. If the merge fails or
    CBCL data is missing the function still writes the age-independence
    rows so Panel~A can be rendered.

    Notes
    -----
    Runs 6 bandwidth values centred around the default MGD_BANDWIDTH.
    Each pass requires a full Riemannian mean computation; cost is
    roughly 6× the single-bandwidth step04b MGD cost. For very large
    cohorts consider reducing the sweep to 4 bandwidths.
    """
    from step04_statistics import partial_corr as _partial_corr
    bandwidths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    rows_indep = []
    rows_cbcl = []
    rows_cross = []

    n = len(ages)
    sex_numeric = (sexes == "M").astype(float)
    covar_2d = np.column_stack([ages, sex_numeric])

    for bw in bandwidths:
        logger.info("  Bandwidth sweep: h=%.1f years", bw)
        mgd_raw_b, _, _ = compute_all_mgd(covmats, ages, bandwidth=bw)
        mgd_z_b = {}
        for b in BAND_ORDER + ["combined"]:
            if b in mgd_raw_b:
                mgd_z_b[b] = zscore_by_age(mgd_raw_b[b], ages)

        # Panel A: |r(MGD, age)|
        for b, vals in mgd_z_b.items():
            mask = ~np.isnan(vals)
            if mask.sum() < 10:
                continue
            r, _ = pearsonr(vals[mask], ages[mask])
            rows_indep.append({"bandwidth": bw, "band": b,
                               "r": r, "abs_r": abs(r), "n": int(mask.sum())})

        # Panel B: partial corr with CBCL (alpha + combined)
        for b in ["alpha", "combined"]:
            if b not in mgd_z_b:
                continue
            mgd_vals = mgd_z_b[b]
            for var in CBCL_VARS:
                y = cbcl_data[var]
                mask = ~np.isnan(mgd_vals) & ~np.isnan(y)
                if mask.sum() < 20:
                    continue
                r_p, _ = _partial_corr(mgd_vals[mask], y[mask],
                                       covar_2d[mask])
                rows_cbcl.append({"bandwidth": bw, "band": b,
                                  "variable": var, "r": r_p,
                                  "abs_r": abs(r_p),
                                  "n": int(mask.sum())})

        # Panel C: cross-band alpha-theta and beta-mean specificity
        if "alpha" in mgd_z_b and "theta" in mgd_z_b:
            mask = ~np.isnan(mgd_z_b["alpha"]) & ~np.isnan(mgd_z_b["theta"])
            if mask.sum() >= 10:
                r_at, _ = pearsonr(mgd_z_b["alpha"][mask],
                                   mgd_z_b["theta"][mask])
                rows_cross.append({"bandwidth": bw, "band_pair": "theta-alpha",
                                   "r": r_at})
        if "beta" in mgd_z_b:
            mask = ~np.isnan(mgd_z_b["beta"])
            if mask.sum() >= 10:
                other_bands = [b for b in BAND_ORDER if b != "beta"
                               and b in mgd_z_b]
                cross = []
                for o in other_bands:
                    m2 = mask & ~np.isnan(mgd_z_b[o])
                    if m2.sum() < 10:
                        continue
                    r_b, _ = pearsonr(mgd_z_b["beta"][m2], mgd_z_b[o][m2])
                    cross.append(abs(r_b))
                if cross:
                    rows_cross.append(
                        {"bandwidth": bw, "band_pair": "beta-mean",
                         "r": float(np.mean(cross))})

    df_indep = pd.DataFrame(rows_indep)
    df_cbcl = pd.DataFrame(rows_cbcl)
    df_cross = pd.DataFrame(rows_cross)

    # Attach metric name so step05c can pivot
    if len(df_indep):
        df_indep["metric"] = "age_independence"
    if len(df_cbcl):
        df_cbcl["metric"] = "cbcl_partial_corr"

    # Merge age-independence + cbcl into a single DataFrame
    parts = [d for d in (df_indep, df_cbcl) if len(d)]
    bw_all = pd.concat(parts, ignore_index=True, sort=False)
    if len(bw_all):
        bw_all.to_csv(RESULT_DIR / "bandwidth_sensitivity.csv", index=False)
    df_cross.to_csv(RESULT_DIR / "bandwidth_cross_band.csv", index=False)
    logger.info("Saved bandwidth_sensitivity.csv (%d rows) and "
                "bandwidth_cross_band.csv (%d rows)",
                len(bw_all), len(df_cross))


def _load_cbcl_from_tsv(sids, existing, n):
    """Load CBCL scores from participants.tsv files as fallback."""
    from config import RELEASES_TO_PROCESS, RELEASES, RAW_BASE
    import os

    # Build subject -> CBCL mapping from all available releases
    cbcl_map = {}
    for r in RELEASES_TO_PROCESS:
        info = RELEASES[r]
        tsv_path = RAW_BASE / f"HBN-Release{r}-{info['dataset']}" / "participants.tsv"
        if tsv_path.exists():
            try:
                df = pd.read_csv(tsv_path, sep="\t")
                for _, row in df.iterrows():
                    sid = row.get("participant_id", "")
                    if not sid:
                        continue
                    new_record = {
                        "p_factor": float(row.get("p_factor", np.nan)),
                        "attention": float(row.get("attention", np.nan)),
                        "internalizing": float(row.get("internalizing", np.nan)),
                        "externalizing": float(row.get("externalizing", np.nan)),
                    }
                    # Don't overwrite existing records that already have
                    # non-NaN values with NaN values from another release.
                    if sid in cbcl_map:
                        for k, v in new_record.items():
                            if np.isnan(v) and not np.isnan(cbcl_map[sid].get(k, np.nan)):
                                continue
                            cbcl_map[sid][k] = v
                    else:
                        cbcl_map[sid] = new_record
            except Exception as e:
                logger.warning("  Failed to read %s: %s", tsv_path, e)

    logger.info("  Loaded CBCL for %d subjects from TSV files", len(cbcl_map))

    for var in CBCL_VARS:
        vals = np.array([cbcl_map.get(s, {}).get(var, np.nan) for s in sids])
        # Only fill where existing is NaN
        if var in existing:
            mask = np.isnan(existing[var])
            existing[var][mask] = vals[mask]
        else:
            existing[var] = vals

    return existing


if __name__ == "__main__":
    main()
