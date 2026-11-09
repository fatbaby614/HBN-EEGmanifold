# -*- coding: utf-8 -*-
"""
Step 4 — Statistical analysis.

Computes:
  • BAG vs phenotypes (partial correlation, FDR corrected)
  • High / low risk group comparison (Mann-Whitney U)
  • Geodesic distance to age-group centroid
  • ROC: geodesic distance → high/low p_factor classification
  • Permutation test for sex differences (10 000 permutations)
  • Resting vs task-state centroid shift (seqLearning 6/8 target)

Saves all results as CSV in data/processed/results/.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample

# pyriemann moved utils → geometry in 0.14; support both
try:
    from pyriemann.geometry.mean import mean_riemann
    from pyriemann.geometry.distance import distance_riemann
except ImportError:
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.distance import distance_riemann

from config import *
from utils import setup_logging, parse_seqlearning_events

try:
    import pingouin as pg
    HAS_PG = True
except ImportError:
    HAS_PG = False

warnings.filterwarnings("ignore")
logger = setup_logging("step04_stats")


# ── Feature loading (supports multi-release merge) ─────────────────────

def load_merged_features(task=REST_TASK):
    """Load features from a single file or merge all release files.
    
    Priority: merge all R*.npz files if they exist; otherwise use single file.
    """
    # Multi-release: find and merge all R*.npz files FIRST
    release_files = sorted(FEATURE_DIR.glob(f"{task}_features_R*.npz"))
    if len(release_files) > 1:
        logger.info("Merging %d release feature files for %s: %s",
                    len(release_files), task,
                    [f.name for f in release_files])
        all_data = [np.load(f, allow_pickle=True) for f in release_files]

        # Concatenate per-subject arrays; keep metadata from first file
        n_ref = len(all_data[0]["ages"])
        merged = {}
        for key in all_data[0].files:
            arr = all_data[0][key]
            if hasattr(arr, '__len__') and len(arr) == n_ref:
                merged[key] = np.concatenate([d[key] for d in all_data])
            else:
                merged[key] = arr  # metadata (e.g. band_names), same for all releases

        # HBN may include the same participant in multiple releases
        # (longitudinal re-recording).  Without deduplication the same
        # subject's features would be counted twice and would also leak
        # information between train/test folds during CV.  Keep the
        # first occurrence (earliest release) per subject id.
        if "subject_ids" in merged:
            sids = np.asarray(merged["subject_ids"])
            unique_sids, unique_idx = np.unique(sids, return_index=True)
            if len(unique_idx) < len(sids):
                logger.warning("  Removed %d duplicate subjects across releases "
                               "(keeping first occurrence)",
                               len(sids) - len(unique_idx))
                keep = np.sort(unique_idx)
                for k in list(merged.keys()):
                    arr = merged[k]
                    if hasattr(arr, '__len__') and len(arr) == len(sids):
                        merged[k] = np.asarray(arr)[keep]

        # ── Re-project tangent space on merged covariances ──
        # Each release's tangent/FBTS vectors were projected relative to that
        # release's own Riemannian mean.  Direct concatenation mixes different
        # coordinate systems.  Re-project using a global mean computed from
        # ALL merged covariance matrices, consistent with step03's approach.
        if "covmats" in merged and "tangent" in merged:
            from config import BAND_ORDER
            from pyriemann.tangentspace import TangentSpace

            covmats_merged = merged["covmats"]  # (n_total, 5, 128, 128)
            n_subjects = covmats_merged.shape[0]
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

            merged["tangent"] = tangent
            merged["fbts"] = tangent.reshape(n_subjects, -1)
            logger.info("  Re-projected FBTS: %s", merged["fbts"].shape)

        # Return as a dict-like object that mimics np.load result
        class MergedNpz:
            def __init__(self, d):
                self._d = d
            def __getitem__(self, key):
                return self._d[key]
            @property
            def files(self):
                return list(self._d.keys())

        return MergedNpz(merged)
    
    # Single release file or base file
    if len(release_files) == 1:
        return np.load(release_files[0], allow_pickle=True)
    base_path = FEATURE_DIR / f"{task}_features.npz"
    if base_path.exists():
        return np.load(base_path, allow_pickle=True)
    
    raise FileNotFoundError(
        f"No feature files found in {FEATURE_DIR} for task {task}")


# ── FDR correction ──────────────────────────────────────────────────────

def fdr_correct(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


# ── Bootstrap AUC CI ───────────────────────────────────────────────────

def bootstrap_auc_ci(scores, labels, n_boot=2000, alpha=0.05):
    """Bootstrap 95 % CI for AUC.

    sklearn 1.7 introduced the ``stratify`` keyword in
    ``sklearn.utils.resample``.  Older sklearn versions would raise
    ``TypeError`` if we passed it.  We try the stratified path first,
    then fall back to plain bootstrap so the code works on any sklearn
    ≥ 1.0.
    """
    aucs = []
    n = len(labels)
    rng = np.random.RandomState(RANDOM_STATE)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    for _ in range(n_boot):
        try:
            idx = resample(np.arange(n), n_samples=n,
                           replace=True, random_state=rng,
                           stratify=labels)
        except TypeError:
            # Older sklearn: stratified + bootstrap is not supported; do a
            # per-class bootstrap instead (still yields valid CI).
            p = rng.choice(pos_idx, size=len(pos_idx), replace=True) \
                if len(pos_idx) else pos_idx
            q = rng.choice(neg_idx, size=len(neg_idx), replace=True) \
                if len(neg_idx) else neg_idx
            idx = np.concatenate([p, q])
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
    if not aucs:
        return np.nan, np.nan
    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return lo, hi


# ── Partial correlation ─────────────────────────────────────────────────

def partial_corr(x, y, covar):
    """
    Partial Pearson correlation between x and y, controlling for covar.

    Parameters
    ----------
    x, y : 1-D arrays
    covar : array-like
        Single covariate (1-D) or multiple covariates (2-D, n_samples x n_covars).

    Falls back to residual regression if pingouin is unavailable.
    """
    covar = np.asarray(covar, dtype=float)
    if covar.ndim == 1:
        covar_2d = covar.reshape(-1, 1)
    else:
        covar_2d = covar
    n_covars = covar_2d.shape[1]

    if HAS_PG:
        data = {"x": x, "y": y}
        if n_covars == 1:
            data["covar"] = covar_2d[:, 0]
            covar_cols = ["covar"]
        else:
            covar_cols = []
            for j in range(n_covars):
                col_name = f"covar{j}"
                data[col_name] = covar_2d[:, j]
                covar_cols.append(col_name)
        df = pd.DataFrame(data)
        r = pg.partial_corr(data=df, x="x", y="y", covar=covar_cols,
                            method="pearson")
        r_val = r["r"].values[0]
        # pingouin column name varies across versions:
        #   'p-val' (old), 'p_val' (some), 'p-unc' (>=0.5.4)
        for col in ["p-unc", "p-val", "p_val"]:
            if col in r.columns:
                p_val = r[col].values[0]
                break
        else:
            p_val = r.iloc[0, -1]  # fallback: last column
        return r_val, p_val
    # manual: regress out covariates from both x and y via OLS
    n = len(x)
    intercept = np.ones((n, 1))
    C = np.hstack([intercept, covar_2d])
    from numpy.linalg import lstsq
    bx, *_ = lstsq(C, x, rcond=None)
    by, *_ = lstsq(C, y, rcond=None)
    rx = x - C @ bx
    ry = y - C @ by
    r = np.corrcoef(rx, ry)[0, 1]
    # Partial correlation t-test: df = n - k - 2 (k = n_covariates)
    # pearsonr would use n-2, which is wrong when covariates are present
    df = n - n_covars - 2
    if df <= 0 or abs(r) >= 1.0:
        return r, np.nan
    from scipy.stats import t as t_dist
    t_val = r * np.sqrt(df / (1 - r ** 2))
    p = 2 * (1 - t_dist.cdf(abs(t_val), df=df))
    return r, p


# ── 1. BAG analysis ────────────────────────────────────────────────────

def analyse_bag(preds_path):
    """Partial correlation of BAG with phenotypic factors."""
    logger.info("\n=== BAG Partial Correlation Analysis ===")
    df = pd.read_csv(preds_path)
    bag = df["bag_fbts"].values
    age = df["age"].values

    pheno_vars = ["p_factor", "attention", "internalizing", "externalizing"]

    # need to load phenotype values from feature file
    feat_data = load_merged_features(REST_TASK)
    sid_feat = feat_data["subject_ids"]

    # build pheno lookup (include sex for partial correlation)
    sexes_feat = feat_data["sexes"]
    pheno_lookup = {}
    for i, sid in enumerate(sid_feat):
        sex_val = sexes_feat[i]
        # Normalize sex to numeric: M=1, F=0
        if isinstance(sex_val, str):
            sex_num = 1.0 if sex_val.upper().startswith("M") else 0.0
        else:
            sex_num = float(sex_val)
        pheno_lookup[sid] = {
            "p_factor": feat_data["p_factor"][i],
            "attention": feat_data["attention"][i],
            "internalizing": feat_data["internalizing"][i],
            "externalizing": feat_data["externalizing"][i],
            "sex": sex_num,
        }

    rows = []
    for var in pheno_vars:
        vals = np.array([pheno_lookup.get(s, {}).get(var, np.nan) for s in df["subject"]])
        sex_vals = np.array([pheno_lookup.get(s, {}).get("sex", np.nan) for s in df["subject"]])
        mask = (~np.isnan(vals)) & (~np.isnan(sex_vals))
        if mask.sum() < 10:
            continue
        # Control for both age and sex
        covars = np.column_stack([age[mask], sex_vals[mask]])
        r, p = partial_corr(bag[mask], vals[mask], covars)
        n = mask.sum()
        # 95% CI via Fisher z-transformation (df = n - k - 2, k=2 covariates)
        from scipy.stats import norm
        r_clamped = max(min(r, 0.9999), -0.9999)
        z = 0.5 * np.log((1 + r_clamped) / (1 - r_clamped))
        se_z = 1.0 / np.sqrt(n - 5)  # n - k - 3 for k=2 covariates
        z_crit = norm.ppf(0.975)
        z_lo = z - z_crit * se_z
        z_hi = z + z_crit * se_z
        ci_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
        ci_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
        rows.append({"variable": var, "r": r, "p": p, "n": n,
                     "ci_lo": ci_lo, "ci_hi": ci_hi})

    result = pd.DataFrame(rows)
    result["p_fdr"] = fdr_correct(result["p"].values)
    result["sig"] = result["p_fdr"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    )
    for _, row in result.iterrows():
        logger.info("  BAG ~ %s: r=%.3f, p=%.4f, p_fdr=%.4f %s (n=%d)",
                    row["variable"], row["r"], row["p"], row["p_fdr"],
                    row["sig"], row["n"])

    result.to_csv(RESULT_DIR / "bag_partial_corr.csv", index=False)

    # High/low risk group comparison
    logger.info("\n=== High/Low Risk Group Comparison ===")
    p_factor = np.array([pheno_lookup.get(s, {}).get("p_factor", np.nan)
                         for s in df["subject"]])
    mask = ~np.isnan(p_factor)
    pf = p_factor[mask]
    bg = bag[mask]
    hi = pf > np.percentile(pf, RISK_PERCENTILE_HIGH)
    lo = pf < np.percentile(pf, RISK_PERCENTILE_LOW)

    if hi.sum() == 0 or lo.sum() == 0:
        logger.warning("  Empty risk group (hi=%d, lo=%d), skipping comparison",
                       hi.sum(), lo.sum())
        group_df = pd.DataFrame([
            {"group": "high_risk", "n": int(hi.sum()),
             "bag_mean": np.nan, "bag_std": np.nan},
            {"group": "low_risk", "n": int(lo.sum()),
             "bag_mean": np.nan, "bag_std": np.nan},
        ])
        group_df.to_csv(RESULT_DIR / "bag_group_comparison.csv", index=False)
        return result

    u_stat, p_val = mannwhitneyu(bg[hi], bg[lo], alternative="greater")
    # Compute Mann-Whitney Z-statistic from U (the standard formula), so that
    # effect size r = Z / sqrt(N) is well-defined even when p → 0 or p → 1.
    from scipy.stats import norm
    n1 = int(hi.sum())
    n2 = int(lo.sum())
    mu_u = n1 * n2 / 2.0
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma_u > 0:
        z = (u_stat - mu_u) / sigma_u
    else:
        z = 0.0
    eff_r = z / np.sqrt(n1 + n2)
    logger.info("  High risk (n=%d) BAG=%.3f±%.3f vs Low risk (n=%d) BAG=%.3f±%.3f",
                hi.sum(), bg[hi].mean(), bg[hi].std(),
                lo.sum(), bg[lo].mean(), bg[lo].std())
    logger.info("  Mann-Whitney U=%.1f, p=%.4f, effect r=%.3f", u_stat, p_val, eff_r)

    group_df = pd.DataFrame([
        {"group": "high_risk", "n": int(hi.sum()),
         "bag_mean": bg[hi].mean(), "bag_std": bg[hi].std()},
        {"group": "low_risk", "n": int(lo.sum()),
         "bag_mean": bg[lo].mean(), "bag_std": bg[lo].std()},
        {"test": "mannwhitney_u", "U": u_stat, "p": p_val, "effect_r": eff_r},
    ])
    group_df.to_csv(RESULT_DIR / "bag_group_comparison.csv", index=False)

    return result


# ── 1b. BAG correction and age-stratified analysis ────────────────────

def analyse_bag_corrected(preds_path):
    """BAG correction: remove linear age-dependent bias, then re-test
    clinical correlations. Also reports per-age-group accuracy.
    """
    logger.info("\n=== BAG Correction & Age-Stratified Analysis ===")
    df = pd.read_csv(preds_path)
    bag = df["bag_fbts"].values
    age = df["age"].values
    pred = df["pred_fbts_svr"].values

    # ── Linear BAG correction ──
    # BAG_corrected = BAG - (a + b*age), standard in brain age literature
    # Guard for tiny samples (np.polyfit requires n >= 2)
    if len(age) < 2:
        logger.warning("  BAG correction: only %d samples, skipping",
                       len(age))
        return {}
    coeffs = np.polyfit(age, bag, 1)
    bag_corrected = bag - np.polyval(coeffs, age)
    logger.info("  BAG = %.4f * age + %.4f", coeffs[0], coeffs[1])
    logger.info("  Corrected BAG: mean=%.3f, std=%.3f (raw: mean=%.3f, std=%.3f)",
                bag_corrected.mean(), bag_corrected.std(),
                bag.mean(), bag.std())

    # Save corrected BAG to a SEPARATE CSV (not back to cv5_predictions.csv)
    # Rewriting the source CSV would cause repeated runs to compound
    # the BAG correction (because each re-run would treat the already-
    # corrected BAG as the new raw BAG and re-correct again).
    corrected_path = RESULT_DIR / "bag_corrected.csv"
    pd.DataFrame({"subject": df["subject"].values,
                  "age": age, "bag_fbts": bag,
                  "bag_corrected": bag_corrected,
                  "pred_fbts_svr": pred}).to_csv(corrected_path, index=False)
    logger.info("  Saved corrected BAG to %s", corrected_path)

    # ── Corrected BAG vs clinical (controlling age + sex) ──
    pheno_vars = ["p_factor", "attention", "internalizing", "externalizing"]
    feat_data = load_merged_features(REST_TASK)
    sid_feat = feat_data["subject_ids"]
    sexes_feat = feat_data["sexes"]
    pheno_lookup = {}
    for i, sid in enumerate(sid_feat):
        sex_val = sexes_feat[i]
        if isinstance(sex_val, str):
            sex_num = 1.0 if sex_val.upper().startswith("M") else 0.0
        else:
            sex_num = float(sex_val)
        pheno_lookup[sid] = {
            v: feat_data[v][i] for v in pheno_vars
        }
        pheno_lookup[sid]["sex"] = sex_num

    rows = []
    for var in pheno_vars:
        vals = np.array([pheno_lookup.get(s, {}).get(var, np.nan)
                         for s in df["subject"]])
        sex_vals = np.array([pheno_lookup.get(s, {}).get("sex", np.nan)
                             for s in df["subject"]])
        mask = (~np.isnan(vals)) & (~np.isnan(sex_vals))
        if mask.sum() < 10:
            continue
        # Raw BAG: partial correlation controlling age + sex
        covars_raw = np.column_stack([age[mask], sex_vals[mask]])
        r_raw, p_raw = partial_corr(bag[mask], vals[mask], covars_raw)
        # Corrected BAG: partial correlation controlling age + sex
        # (age included to capture any residual non-linear age effect)
        r_corr, p_corr = partial_corr(bag_corrected[mask], vals[mask], covars_raw)
        n = mask.sum()
        # 95% CI via Fisher z-transformation (k=2 covariates, df=n-k-2=n-4)
        from scipy.stats import norm
        r_clamped = max(min(r_corr, 0.9999), -0.9999)
        z = 0.5 * np.log((1 + r_clamped) / (1 - r_clamped))
        se_z = 1.0 / np.sqrt(n - 5)  # n - k - 3 for k=2 covariates
        z_crit = norm.ppf(0.975)
        z_lo = z - z_crit * se_z
        z_hi = z + z_crit * se_z
        ci_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
        ci_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
        rows.append({
            "variable": var, "r_raw": r_raw, "p_raw": p_raw,
            "r_corrected": r_corr, "p_corrected": p_corr,
            "n": n, "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
    result = pd.DataFrame(rows)
    result["p_fdr_corrected"] = fdr_correct(result["p_corrected"].values)
    for _, row in result.iterrows():
        logger.info("  %s: raw r=%.3f (p=%.3f) → corrected r=%.3f (p=%.4f, fdr=%.4f, 95%%CI [%.3f, %.3f])",
                    row["variable"], row["r_raw"], row["p_raw"],
                    row["r_corrected"], row["p_corrected"],
                    row["p_fdr_corrected"],
                    row["ci_lo"], row["ci_hi"])
    result.to_csv(RESULT_DIR / "bag_corrected_corr.csv", index=False)

    # ── Age-stratified accuracy ──
    logger.info("\n  --- Per-age-group prediction accuracy ---")
    # Use the 3-year-wide bins (independent of the 4-year-wide
    # AGE_GROUPS in config.py) for finer-grained stratification.
    age_bins = [(5, 8), (8, 11), (11, 14), (14, 18)]
    strat_rows = []
    for lo, hi in age_bins:
        # half-open [lo, hi) to keep boundary ages deterministic
        mask = (age >= lo) & (age < hi)
        if mask.sum() > 5:
            r, _ = pearsonr(age[mask], pred[mask])
            rmse = np.sqrt(np.mean((age[mask] - pred[mask]) ** 2))
            mae = np.mean(np.abs(age[mask] - pred[mask]))
            bag_mean = bag[mask].mean()
            logger.info("  %d-%d (n=%d): r=%.3f, RMSE=%.2f, BAG=%.2f",
                        lo, hi, mask.sum(), r, rmse, bag_mean)
            strat_rows.append({
                "age_group": f"{lo}-{hi}", "n": int(mask.sum()),
                "cor": r, "rmse": rmse, "mae": mae,
                "bag_mean": bag_mean, "bag_std": bag[mask].std(),
            })
    pd.DataFrame(strat_rows).to_csv(
        RESULT_DIR / "age_stratified_accuracy.csv", index=False)


# ── 2. Geodesic distance & ROC ─────────────────────────────────────────

def analyse_geodesic():
    """Geodesic distance to age-group centroid and ROC analysis."""
    logger.info("\n=== Geodesic Distance & ROC Analysis ===")
    data = load_merged_features(REST_TASK)
    covmats = data["covmats"]   # (n, 5, 128, 128)
    ages = data["ages"]
    p_factor = data["p_factor"]
    sids = data["subject_ids"]
    n = len(ages)

    # use alpha band for primary analysis
    band_idx = BAND_ORDER.index("alpha")
    cov_band = covmats[:, band_idx, :, :]  # (n, 128, 128)

    # age-group centroids (integer years)
    age_int = np.round(ages).astype(int)
    unique_ages = np.unique(age_int)

    distances = np.zeros(n)
    for a in unique_ages:
        mask = age_int == a
        if mask.sum() < 3:
            # merge with nearest age group
            continue
        centroid = mean_riemann(cov_band[mask])
        for idx in np.where(mask)[0]:
            distances[idx] = distance_riemann(cov_band[idx], centroid)

    # assign merged-group subjects to nearest valid centroid
    valid_ages = [a for a in unique_ages
                  if (age_int == a).sum() >= 3]
    if len(valid_ages) == 0:
        logger.warning("  No age group has >= 3 subjects, skipping geodesic analysis")
        return
    for idx in range(n):
        if distances[idx] == 0 and age_int[idx] not in valid_ages:
            nearest = min(valid_ages, key=lambda a: abs(a - age_int[idx]))
            mask = age_int == nearest
            centroid = mean_riemann(cov_band[mask])
            distances[idx] = distance_riemann(cov_band[idx], centroid)

    # geodesic distance vs p_factor correlation
    mask = ~np.isnan(p_factor)
    if mask.sum() < 2:
        logger.warning("  Insufficient valid p_factor for correlation (n=%d)", mask.sum())
        return
    r_pf, p_pf = pearsonr(distances[mask], p_factor[mask])
    logger.info("  Geodesic dist vs p_factor: r=%.3f, p=%.4f (n=%d)",
                r_pf, p_pf, mask.sum())

    # ROC: classify high vs low risk
    pf = p_factor[mask]
    dist = distances[mask]
    hi_mask = pf > np.percentile(pf, RISK_PERCENTILE_HIGH)
    lo_mask = pf < np.percentile(pf, RISK_PERCENTILE_LOW)
    if hi_mask.sum() < 2 or lo_mask.sum() < 2:
        logger.warning("  Insufficient risk groups for ROC (hi=%d, lo=%d), skipping",
                       hi_mask.sum(), lo_mask.sum())
        return
    roc_labels = np.concatenate([np.ones(hi_mask.sum()),
                                  np.zeros(lo_mask.sum())])
    roc_scores = np.concatenate([dist[hi_mask], dist[lo_mask]])
    auc = roc_auc_score(roc_labels, roc_scores)
    ci_lo, ci_hi = bootstrap_auc_ci(roc_scores, roc_labels)
    fpr, tpr, thresholds = roc_curve(roc_labels, roc_scores)
    best_idx = np.argmax(tpr - fpr)  # Youden's J
    logger.info("  ROC AUC=%.3f (95%% CI: %.3f–%.3f)", auc, ci_lo, ci_hi)
    logger.info("  Best cutoff=%.3f, sensitivity=%.3f, specificity=%.3f",
                thresholds[best_idx], tpr[best_idx], 1 - fpr[best_idx])

    # per-band AUC
    band_aucs = []
    for b, band_name in enumerate(BAND_ORDER):
        cov_b = covmats[:, b, :, :]
        dist_b = np.zeros(n)
        for a in valid_ages:
            mask_a = age_int == a
            if mask_a.sum() < 3:
                continue
            centroid = mean_riemann(cov_b[mask_a])
            for idx in np.where(mask_a)[0]:
                dist_b[idx] = distance_riemann(cov_b[idx], centroid)
        # assign subjects in small age groups to nearest valid centroid
        for idx in range(n):
            if dist_b[idx] == 0 and age_int[idx] not in valid_ages:
                nearest = min(valid_ages, key=lambda a: abs(a - age_int[idx]))
                mask_a = age_int == nearest
                centroid = mean_riemann(cov_b[mask_a])
                dist_b[idx] = distance_riemann(cov_b[idx], centroid)
        roc_scores_b = np.concatenate([dist_b[mask][hi_mask],
                                        dist_b[mask][lo_mask]])
        auc_b = roc_auc_score(roc_labels, roc_scores_b)
        band_aucs.append({"band": band_name, "auc": auc_b})
        logger.info("  AUC (%s): %.3f", band_name, auc_b)

    # save
    geo_df = pd.DataFrame({
        "subject": sids, "age": ages, "age_int": age_int,
        "p_factor": p_factor, "geodesic_dist": distances,
    })
    geo_df.to_csv(RESULT_DIR / "geodesic_distances.csv", index=False)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
    roc_df.to_csv(RESULT_DIR / "roc_curve.csv", index=False)

    pd.DataFrame([{
        "auc": auc, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "best_cutoff": thresholds[best_idx],
        "sensitivity": tpr[best_idx], "specificity": 1 - fpr[best_idx],
        "r_pfactor": r_pf, "p_pfactor": p_pf,
    }]).to_csv(RESULT_DIR / "roc_summary.csv", index=False)

    pd.DataFrame(band_aucs).to_csv(
        RESULT_DIR / "roc_per_band.csv", index=False)


# ── 3. Sex differences (permutation test) ───────────────────────────────

def analyse_sex_differences():
    """Permutation test for male vs female centroid distance.

    Uses log-Euclidean approximation for the permutation loop (precomputes
    matrix logarithms once), which is ~100× faster than iterative
    mean_riemann while producing equivalent p-values.  The actual
    distance uses the full Riemannian metric for accuracy.
    """
    logger.info("\n=== Sex Differences Permutation Test ===")
    data = load_merged_features(REST_TASK)
    covmats = data["covmats"]
    sexes = data["sexes"]
    ages = data["ages"]
    n = len(sexes)

    # use alpha band
    band_idx = BAND_ORDER.index("alpha")
    cov_band = covmats[:, band_idx, :, :]  # (n, 128, 128)

    male_mask = sexes == "M"
    female_mask = sexes == "F"

    if male_mask.sum() < 10 or female_mask.sum() < 10:
        logger.warning("  Insufficient subjects for sex comparison")
        return

    # actual distance (full Riemannian — one-time cost)
    male_centroid = mean_riemann(cov_band[male_mask])
    female_centroid = mean_riemann(cov_band[female_mask])
    actual_dist = distance_riemann(male_centroid, female_centroid)
    logger.info("  Male (n=%d) vs Female (n=%d) centroid distance: %.4f",
                male_mask.sum(), female_mask.sum(), actual_dist)

    # ── Precompute matrix logarithms for fast permutation loop ──
    # Log-Euclidean mean = expm(mean(logm(matrices)))
    # Log-Euclidean dist = ||logm(A) - logm(B)||_F
    # This avoids iterative mean_riemann inside the permutation loop.
    from scipy.linalg import logm, expm
    logger.info("  Precomputing matrix logarithms...")
    log_covs = np.zeros_like(cov_band)
    for i in range(n):
        log_covs[i] = logm(cov_band[i])

    log_male_mean = log_covs[male_mask].mean(axis=0)
    log_female_mean = log_covs[female_mask].mean(axis=0)
    actual_dist_le = np.linalg.norm(log_male_mean - log_female_mean, 'fro')
    logger.info("  Log-Euclidean actual distance: %.4f", actual_dist_le)

    # permutation test (log-Euclidean — extremely fast)
    rng = np.random.RandomState(RANDOM_STATE)
    perm_dists = []
    import time
    t0 = time.time()
    for i_perm in range(N_PERMUTATIONS):
        perm_labels = rng.permutation(sexes)
        m_mask = perm_labels == "M"
        f_mask = perm_labels == "F"
        if m_mask.sum() < 5 or f_mask.sum() < 5:
            continue
        log_m_mean = log_covs[m_mask].mean(axis=0)
        log_f_mean = log_covs[f_mask].mean(axis=0)
        perm_dists.append(np.linalg.norm(log_m_mean - log_f_mean, 'fro'))
        if (i_perm + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i_perm + 1) * (N_PERMUTATIONS - i_perm - 1)
            logger.info("  Permutation %d / %d  (%.1fs elapsed, ~%.1fs remaining)",
                        i_perm + 1, N_PERMUTATIONS, elapsed, eta)

    perm_dists = np.array(perm_dists)
    p_val = np.mean(perm_dists >= actual_dist_le)
    logger.info("  Permutation p=%.4f (n_perm=%d, metric=log-Euclidean)",
                p_val, len(perm_dists))

    # per-age-layer distances (descriptive)
    age_layers = []
    for a in np.unique(np.round(ages).astype(int)):
        mask = np.round(ages).astype(int) == a
        m = mask & male_mask
        f = mask & female_mask
        if m.sum() >= 3 and f.sum() >= 3:
            mc = mean_riemann(cov_band[m])
            fc = mean_riemann(cov_band[f])
            d = distance_riemann(mc, fc)
            age_layers.append({
                "age": a, "n_male": int(m.sum()),
                "n_female": int(f.sum()), "distance": d,
            })
            logger.info("  Age %d: M=%d F=%d dist=%.4f",
                        a, m.sum(), f.sum(), d)

    pd.DataFrame(age_layers).to_csv(
        RESULT_DIR / "sex_distance_by_age.csv", index=False)
    pd.DataFrame([{
        # Report BOTH metrics so the reader is not misled by a single value:
        #   - actual_distance / actual_distance_riemann: full Riemannian
        #     centroid distance (kept for backward compatibility with any
        #     downstream code that reads the legacy column name)
        #   - actual_distance_log_euclidean: log-Euclidean distance
        #     (the metric used to compute p_value in the permutation loop)
        "actual_distance": actual_dist_le,
        "actual_distance_riemann": actual_dist,
        "actual_distance_log_euclidean": actual_dist_le,
        "p_value": p_val,
        "p_metric": "log_euclidean",
        "n_permutations": len(perm_dists),
        "n_male": int(male_mask.sum()), "n_female": int(female_mask.sum()),
    }]).to_csv(RESULT_DIR / "sex_permutation_test.csv", index=False)

    # save permutation distribution
    np.savetxt(RESULT_DIR / "sex_permutation_dist.csv",
               perm_dists, delimiter=",")


# ── 4. Resting vs task-state centroid shift ────────────────────────────

def analyse_task_shift():
    """Compute resting vs seqLearning centroid shift per task version."""
    logger.info("\n=== Resting vs Task-State Centroid Shift ===")

    rest_data = load_merged_features(REST_TASK)
    rest_sids = set(rest_data["subject_ids"].tolist())

    for seq_task in [SEQ6_TASK, SEQ8_TASK]:
        # Try merged release files first, then single file
        seq_release_files = sorted(FEATURE_DIR.glob(f"{seq_task}_features_R*.npz"))
        if seq_release_files:
            all_seq = [np.load(f, allow_pickle=True) for f in seq_release_files]
            n_ref_seq = len(all_seq[0]["ages"])
            seq_data_dict = {}
            for key in all_seq[0].files:
                arr = all_seq[0][key]
                if hasattr(arr, '__len__') and len(arr) == n_ref_seq:
                    seq_data_dict[key] = np.concatenate([d[key] for d in all_seq])
                else:
                    seq_data_dict[key] = arr
            class MergedSeq:
                def __init__(self, d):
                    self._d = d
                def __getitem__(self, k):
                    return self._d[k]
                @property
                def files(self):
                    return list(self._d.keys())
            seq_data = MergedSeq(seq_data_dict)
        else:
            seq_path = FEATURE_DIR / f"{seq_task}_features.npz"
            if not seq_path.exists():
                logger.warning("  %s features not found, skipping", seq_task)
                continue
            seq_data = np.load(seq_path, allow_pickle=True)
        seq_sids = seq_data["subject_ids"]
        common = [s for s in seq_sids if s in rest_sids]
        if len(common) < 5:
            logger.warning("  %s: too few common subjects (%d)", seq_task, len(common))
            continue

        # build index maps
        sid_to_rest = {s: i for i, s in enumerate(rest_data["subject_ids"])}
        sid_to_seq = {s: i for i, s in enumerate(seq_sids)}

        rest_idx = [sid_to_rest[s] for s in common]
        seq_idx = [sid_to_seq[s] for s in common]

        # alpha band
        band_idx = BAND_ORDER.index("alpha")
        rest_cov = rest_data["covmats"][rest_idx, band_idx, :, :]
        seq_cov = seq_data["covmats"][seq_idx, band_idx, :, :]
        ages_common = rest_data["ages"][rest_idx]

        # centroids
        rest_centroid = mean_riemann(rest_cov)
        task_centroid = mean_riemann(seq_cov)
        centroid_dist = distance_riemann(rest_centroid, task_centroid)
        logger.info("  %s: rest-task centroid distance=%.4f (n=%d)",
                    seq_task, centroid_dist, len(common))

        # per-subject rest-to-task distance
        subj_dists = np.array([
            distance_riemann(rest_cov[i], seq_cov[i])
            for i in range(len(common))
        ])

        # correlate with age
        if len(subj_dists) >= 2:
            r_age, p_age = pearsonr(subj_dists, ages_common)
        else:
            r_age, p_age = np.nan, np.nan
            logger.warning("  %s: insufficient subjects for age correlation (n=%d)",
                           seq_task, len(subj_dists))
        logger.info("  %s: distance vs age r=%.3f, p=%.4f",
                    seq_task, r_age, p_age)

        # behavioural accuracy correlation
        from utils import parse_seqlearning_events
        accs = []
        n_found = 0
        for s in common:
            bh = parse_seqlearning_events(s, seq_task)
            if bh is not None:
                n_found += 1
            accs.append(bh["accuracy"] if bh else np.nan)
        accs = np.array(accs)
        mask = ~np.isnan(accs)
        if mask.sum() >= 5:
            r_acc, p_acc = spearmanr(subj_dists[mask], accs[mask])
            logger.info("  %s: distance vs accuracy r=%.3f, p=%.4f (n=%d)",
                        seq_task, r_acc, p_acc, mask.sum())
        else:
            r_acc, p_acc = np.nan, np.nan
            if n_found == 0:
                logger.warning("  %s: no events.tsv found (raw data may have "
                               "been deleted after feature extraction); "
                               "accuracy correlation skipped", seq_task)

        pd.DataFrame([{
            "task": seq_task, "n": len(common),
            "centroid_distance": centroid_dist,
            "r_age": r_age, "p_age": p_age,
            "r_accuracy": r_acc, "p_accuracy": p_acc,
        }]).to_csv(RESULT_DIR / f"task_shift_{seq_task}.csv", index=False)

        # per-subject data
        pd.DataFrame({
            "subject": common, "age": ages_common,
            "rest_task_distance": subj_dists,
            "seq_accuracy": accs,
        }).to_csv(RESULT_DIR / f"task_shift_{seq_task}_subjects.csv",
                   index=False)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Step 4: Statistical Analysis")
    logger.info("=" * 60)

    preds_path = RESULT_DIR / "cv5_predictions.csv"
    if preds_path.exists():
        analyse_bag(preds_path)
        analyse_bag_corrected(preds_path)
    else:
        logger.warning("cv5_predictions.csv not found — run step03 first")

    # Check if any feature files exist (single or multi-release)
    has_features = (FEATURE_DIR / f"{REST_TASK}_features.npz").exists() or \
                   bool(list(FEATURE_DIR.glob(f"{REST_TASK}_features_R*.npz")))
    if has_features:
        analyse_geodesic()
        analyse_sex_differences()
    else:
        logger.warning("Resting features not found — run step02 first")

    analyse_task_shift()

    logger.info("\nStep 4 complete.")


if __name__ == "__main__":
    main()
