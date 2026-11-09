# -*- coding: utf-8 -*-
"""
Step 5 — Generate all publication figures (F1–F10).

Output: PNG files at 300 dpi in drafts/figures/.

Each function is independent — if one fails the rest still run.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from sklearn.manifold import TSNE
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

from config import *
from utils import setup_logging, age_group

from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore")
logger = setup_logging("step05_figures")

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif",
                   "DejaVu Serif", "serif"],
})

AGE_COLORS = {"5-8": "#2196F3", "9-12": "#4CAF50",
              "13-16": "#FF9800", "17-21": "#F44336"}
BAND_COLORS = {"delta": "#9C27B0", "theta": "#2196F3", "alpha": "#4CAF50",
               "beta": "#FF9800", "gamma": "#F44336"}
RISK_COLORS = {"high": "#F44336", "low": "#2196F3"}


def _save(fig, name):
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


class MergedNpz:
    """Wrapper that mimics np.load() output for merged release files."""
    def __init__(self, d):
        self._d = d
    def __getitem__(self, k):
        return self._d[k]
    @property
    def files(self):
        return list(self._d.keys())


def _merge_release_files(release_files):
    """
    Merge multiple per-release .npz feature files into one.

    Re-computes tangent space using a global Riemannian mean so that
    features from different releases share the same coordinate system.
    Direct concatenation of per-release tangent vectors would mix
    different reference points and corrupt t-SNE / model visualizations.
    """
    all_data = [np.load(f, allow_pickle=True) for f in release_files]
    n_ref = len(all_data[0]["ages"])

    merged = {}
    for key in all_data[0].files:
        arr = all_data[0][key]
        if hasattr(arr, '__len__') and len(arr) == n_ref:
            merged[key] = np.concatenate([d[key] for d in all_data])
        else:
            merged[key] = arr  # metadata (e.g. band_names)

    # Deduplicate subjects across releases (HBN longitudinal re-recording).
    if "subject_ids" in merged:
        sids = np.asarray(merged["subject_ids"])
        unique_sids, unique_idx = np.unique(sids, return_index=True)
        if len(unique_idx) < len(sids):
            logger.info("  Deduplicated: removed %d duplicate subjects",
                        len(sids) - len(unique_idx))
            keep = np.sort(unique_idx)
            for k in list(merged.keys()):
                arr = merged[k]
                if hasattr(arr, '__len__') and len(arr) == len(sids):
                    merged[k] = np.asarray(arr)[keep]

    # Re-compute tangent space on merged covariances
    if "covmats" in merged and "tangent" in merged:
        covmats = merged["covmats"]  # (n_total, 5, 128, 128)
        n_total = covmats.shape[0]
        n_bands = len(BAND_ORDER)
        n_ch = covmats.shape[2]
        ts_dim = n_ch * (n_ch + 1) // 2
        tangent = np.zeros((n_total, n_bands, ts_dim))
        for b, band_name in enumerate(BAND_ORDER):
            cov_band = covmats[:, b, :, :]
            ts = TangentSpace(metric="riemann")
            tangent[:, b, :] = ts.fit_transform(cov_band)
        merged["tangent"] = tangent
        merged["fbts"] = tangent.reshape(n_total, -1)
        logger.info("  Re-projected tangent space for %d merged subjects",
                     n_total)

    return MergedNpz(merged)


def _load_rest_features():
    """Load resting features: merge all release files if >1, else single file.

    Disk cache: when merging multiple release files, the merged result
    (including the recomputed tangent-space projection) is cached at
    ``RestingState_features_merged.npz``. Subsequent calls load the
    cache in <1 s instead of recomputing the projection for 1642
    subjects × 5 bands, which otherwise takes ~7 min.
    """
    release_files = sorted(FEATURE_DIR.glob(f"{REST_TASK}_features_R*.npz"))

    # Disk-cache the merged result so re-running step05 is fast.
    if len(release_files) > 1:
        cache_path = FEATURE_DIR / f"{REST_TASK}_features_merged.npz"
        fingerprint = "|".join(
            f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}"
            for f in release_files
        )
        # Cache the fingerprint as a Python object array (dtype=object) so
        # that np.load round-trips it as a single string and str() ==
        # comparison works.  Saving with np.array(fingerprint) would
        # produce a 0-D numpy str array whose str() yields the broken
        # "['...']" representation, defeating the cache.
        if cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=True)
                fp_arr = cached.get("_fingerprint", None)
                if fp_arr is not None:
                    cached_fp = (fp_arr.item()
                                 if fp_arr.dtype == object
                                 else str(fp_arr))
                    if cached_fp == fingerprint:
                        logger.info("  Using cached merged features: %s",
                                    cache_path)
                        return cached
            except Exception:
                logger.warning("  Cache %s is unreadable; re-merging",
                               cache_path)

        merged_obj = _merge_release_files(release_files)
        try:
            cache_payload = dict(merged_obj._d)  # MergedNpz wraps _d
            cache_payload["_fingerprint"] = np.array(fingerprint,
                                                    dtype=object)
            np.savez(cache_path, **cache_payload)
            logger.info("  Cached merged features → %s", cache_path)
        except Exception as e:
            logger.warning("  Failed to write merged-feature cache: %s", e)
        return merged_obj

    # Single release file or base file
    if len(release_files) == 1:
        return np.load(release_files[0], allow_pickle=True)
    p = FEATURE_DIR / f"{REST_TASK}_features.npz"
    if p.exists():
        return np.load(p, allow_pickle=True)
    return None


# ── F1: Demographics ────────────────────────────────────────────────────

def figure01():
    logger.info("Figure 1: Demographics")
    data = _load_rest_features()
    if data is None:
        return
    ages = data["ages"]
    sexes = data["sexes"]
    pf = data["p_factor"]
    att = data["attention"]
    intl = data["internalizing"]
    ext = data["externalizing"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    # A: age histogram by sex
    ax = axes[0]
    for s, color, label in [("M", "#2196F3", "Male"), ("F", "#E91E63", "Female")]:
        ax.hist(ages[sexes == s], bins=20, alpha=0.6, color=color, label=label)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.set_title("A. Age Distribution")
    ax.legend()

    # B: phenotype boxplots by age group
    ax = axes[1]
    groups = [age_group(a) for a in ages]
    pheno_df = pd.DataFrame({
        "group": groups,
        "p_factor": pf, "attention": att,
        "internalizing": intl, "externalizing": ext,
    })
    pheno_long = pheno_df.melt(id_vars="group",
                               var_name="measure", value_name="score")
    box_data = [pheno_df.loc[pheno_df["group"] == g, "p_factor"].dropna()
                for g in ["5-8", "9-12", "13-16", "17-21"]]
    bp = ax.boxplot(box_data, labels=["5-8", "9-12", "13-16", "17-21"],
                    patch_artist=True)
    for patch, g in zip(bp["boxes"], ["5-8", "9-12", "13-16", "17-21"]):
        patch.set_facecolor(AGE_COLORS[g])
        patch.set_alpha(0.6)
    ax.set_ylabel("p_factor")
    ax.set_title("B. P-factor by Age Group")

    # C: sex × age heatmap
    ax = axes[2]
    age_bins = np.arange(5, 22)
    counts = np.zeros((2, len(age_bins)))
    for j, a in enumerate(age_bins):
        counts[0, j] = np.sum((ages >= a) & (ages < a + 1) & (sexes == "M"))
        counts[1, j] = np.sum((ages >= a) & (ages < a + 1) & (sexes == "F"))
    im = ax.imshow(counts, aspect="auto", cmap="YlOrRd")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Male", "Female"])
    ax.set_xticks(range(0, len(age_bins), 2))
    ax.set_xticklabels(age_bins[::2])
    ax.set_xlabel("Age")
    ax.set_title("C. Sample Counts")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    _save(fig, "figure01_demographics.png")


# ── F2: t-SNE visualization ─────────────────────────────────────────────

def figure02():
    logger.info("Figure 2: t-SNE Visualization")
    data = _load_rest_features()
    if data is None:
        return
    fbts = data["fbts"]
    tangent = data["tangent"]
    ages = data["ages"]
    fbts[np.isnan(fbts)] = 0

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    panels = [
        ("A. All Bands (FBTS)", fbts),
        ("B. Theta Band", tangent[:, BAND_ORDER.index("theta"), :]),
        ("C. Alpha Band", tangent[:, BAND_ORDER.index("alpha"), :]),
    ]
    for ax, (title, X) in zip(axes.flat[:3], panels):
        X[np.isnan(X)] = 0
        ts = TSNE(n_components=2, perplexity=min(30, len(X) - 1),
                  random_state=RANDOM_STATE).fit_transform(X)
        sc = ax.scatter(ts[:, 0], ts[:, 1], c=ages, cmap="viridis", s=12, alpha=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, label="Age", shrink=0.7)

    # D: age group centroids with ellipses
    ax = axes[1, 1]
    ts_full = TSNE(n_components=2, perplexity=min(30, len(fbts) - 1),
                   random_state=RANDOM_STATE).fit_transform(fbts)
    for g in ["5-8", "9-12", "13-16", "17-21"]:
        mask = np.array([age_group(a) == g for a in ages])
        if mask.sum() < 3:
            continue
        pts = ts_full[mask]
        cx, cy = pts.mean(axis=0)
        ax.scatter(cx, cy, c=AGE_COLORS[g], s=50, zorder=5, label=g)
        # 95% confidence ellipse
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        w, h = 2 * np.sqrt(np.maximum(eigvals, 0) * 5.991)  # chi2 95% for 2 df
        ell = Ellipse((cx, cy), w, h, angle=angle, fill=False,
                      edgecolor=AGE_COLORS[g], linewidth=1.5)
        ax.add_patch(ell)
    ax.set_title("D. Age Group Centroids (95% CI)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, "figure02_tsne.png")


# ── F3: Model performance ───────────────────────────────────────────────

def figure03():
    logger.info("Figure 3: Model Performance")
    preds_path = RESULT_DIR / "cv5_predictions.csv"
    metrics_path = RESULT_DIR / "cv5_metrics.csv"
    if not preds_path.exists() or not metrics_path.exists():
        logger.warning("  Model results not found, skipping")
        return

    preds = pd.read_csv(preds_path)
    metrics = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # A: predicted vs actual
    ax = axes[0]
    ax.scatter(preds["age"], preds["pred_fbts_svr"], s=12, alpha=0.5, c="#2196F3")
    lim = [min(preds["age"].min(), preds["pred_fbts_svr"].min()) - 1,
           max(preds["age"].max(), preds["pred_fbts_svr"].max()) + 1]
    ax.plot(lim, lim, "k--", lw=0.8)
    r, _ = pearsonr(preds["age"], preds["pred_fbts_svr"]) if len(preds) >= 2 else (np.nan, np.nan)
    ax.set_xlabel("Actual Age")
    ax.set_ylabel("Predicted Age")
    ax.set_title(f"A. FBTS+SVR (r={r:.3f})")

    # B: COR and RMSE bar chart
    ax = axes[1]
    model_order = ["FBTS_SVR", "Ridge_FBTS", "SVR_PSD", "SVR_DE"]
    labels = ["FBTS+SVR", "FBTS+Ridge", "PSD+SVR", "DE+SVR"]
    cors = [metrics.loc[metrics["model"] == m, "cor"].values[0] for m in model_order]

    # Fold-level variability: SD of the outer-fold CORs per model.
    # FBTS_SVR / Ridge_FBTS come from cv5_fold_metrics.csv; the PSD and DE
    # bars are the revision re-runs (revision_psd_de_full.csv), so their
    # error bars must come from the fold rows of the same re-run.
    fold_sds = []
    fold_metrics = pd.read_csv(RESULT_DIR / "cv5_fold_metrics.csv")
    rev_full = pd.read_csv(RESULT_DIR / "revision_psd_de_full.csv")
    for m in model_order:
        if m == "SVR_PSD":
            fcor = rev_full.loc[(rev_full["model"] == "SVR_PSD_LOG_FULL")
                                & (rev_full["fold"] > 0), "cor"].values
        elif m == "SVR_DE":
            fcor = rev_full.loc[(rev_full["model"] == "SVR_DE_FULL")
                                & (rev_full["fold"] > 0), "cor"].values
        else:
            fcor = fold_metrics.loc[fold_metrics["model"] == m, "cor"].values
        fold_sds.append(float(np.std(fcor, ddof=1)))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    bars = ax.bar(labels, cors, yerr=fold_sds, capsize=4,
                  error_kw={"lw": 1, "ecolor": "#555555"},
                  color=colors, alpha=0.8, width=0.65)
    ax.set_ylabel("COR")
    ax.set_title("B. Prediction Accuracy")
    ax.set_ylim(0, max(cors) * 1.2)
    ax.tick_params(axis="x", labelsize=8, pad=4)
    for bar, v, sd in zip(bars, cors, fold_sds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sd + 0.015,
                f"{v:.3f}", ha="center", fontsize=8)

    # C: BAG distribution by age group
    ax = axes[2]
    bag = preds["bag_fbts"].values
    groups = [age_group(a) for a in preds["age"]]
    box_data = [bag[np.array(groups) == g] for g in ["5-8", "9-12", "13-16", "17-21"]]
    bp = ax.boxplot(box_data, labels=["5-8", "9-12", "13-16", "17-21"],
                    patch_artist=True)
    for patch, g in zip(bp["boxes"], ["5-8", "9-12", "13-16", "17-21"]):
        patch.set_facecolor(AGE_COLORS[g])
        patch.set_alpha(0.6)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Brain Age Gap")
    ax.set_title("C. BAG Distribution")

    fig.tight_layout()
    _save(fig, "figure03_model_performance.png")


# ── F4: BAG clinical ────────────────────────────────────────────────────

def figure04():
    logger.info("Figure 4: BAG Clinical")
    bag_path = RESULT_DIR / "bag_partial_corr.csv"
    group_path = RESULT_DIR / "bag_group_comparison.csv"
    preds_path = RESULT_DIR / "cv5_predictions.csv"
    if not bag_path.exists() or not preds_path.exists():
        logger.warning("  BAG results not found, skipping")
        return

    bag_df = pd.read_csv(bag_path)
    preds = pd.read_csv(preds_path)
    data = _load_rest_features()
    if data is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # A: BAG distribution
    ax = axes[0, 0]
    ax.hist(preds["bag_fbts"], bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Brain Age Gap")
    ax.set_ylabel("Count")
    ax.set_title(f"A. BAG (μ={preds['bag_fbts'].mean():.2f}, σ={preds['bag_fbts'].std():.2f})")

    # B-D: partial regression scatter plots
    sid_map = {s: i for i, s in enumerate(data["subject_ids"])}
    pheno = {"p_factor": data["p_factor"], "attention": data["attention"],
             "internalizing": data["internalizing"]}
    titles = ["B. BAG vs P-factor", "C. BAG vs Attention"]
    keys = ["p_factor", "attention"]

    for ax, title, key in zip([axes[0, 1], axes[1, 0]], titles, keys):
        vals = np.array([pheno[key][sid_map[s]] if s in sid_map else np.nan
                         for s in preds["subject"]])
        age = preds["age"].values
        bag = preds["bag_fbts"].values
        # Sex covariate aligned with the prediction table; residualising on
        # age AND sex makes the scatter reproduce the age+sex partial
        # correlations reported in Table 2 (bag_partial_corr.csv).
        sex_raw = data["sexes"]
        sex_num = np.array([{"M": 1.0, "F": 0.0}.get(sex_raw[sid_map[s]], np.nan)
                            if s in sid_map else np.nan
                            for s in preds["subject"]])
        mask = ~np.isnan(vals) & ~np.isnan(sex_num)
        if mask.sum() < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        def _resid(y, cov):
            X = np.column_stack([cov, np.ones(len(cov))])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            return y - X @ coef

        cov = np.column_stack([age[mask], sex_num[mask]])
        bag_res = _resid(bag[mask], cov)
        val_res = _resid(vals[mask], cov)
        ax.scatter(val_res, bag_res, s=12, alpha=0.5, c="#4CAF50")
        if len(val_res) >= 2:
            m, b = np.polyfit(val_res, bag_res, 1)
            ax.plot(val_res, m * val_res + b, "r--", lw=1)
            r, p = pearsonr(val_res, bag_res)
        else:
            r, p = np.nan, np.nan
        ax.set_xlabel(f"{key} (age- and sex-adjusted)")
        ax.set_ylabel("BAG (age- and sex-adjusted)")
        ax.set_title(f"{title}\nr={r:.3f}, p={p:.4f}")

    # D: high vs low risk boxplot
    ax = axes[1, 1]
    if group_path.exists():
        group_df = pd.read_csv(group_path)
        hi = group_df[group_df["group"] == "high_risk"]
        lo = group_df[group_df["group"] == "low_risk"]
        if len(hi) > 0 and len(lo) > 0:
            ax.bar(["High Risk", "Low Risk"],
                   [hi["bag_mean"].values[0], lo["bag_mean"].values[0]],
                   yerr=[hi["bag_std"].values[0], lo["bag_std"].values[0]],
                   color=[RISK_COLORS["high"], RISK_COLORS["low"]], alpha=0.8, capsize=5)
    else:
        ax.text(0.5, 0.5, "Group comparison\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_ylabel("Mean BAG")
    ax.set_title("D. Risk Group Comparison")
    ax.axhline(0, color="k", ls="--", lw=0.8)

    fig.tight_layout()
    _save(fig, "figure04_bag_clinical.png")


# ── F5: Geodesic ROC ───────────────────────────────────────────────────

def figure05():
    """Figure 5: Manifold Geometry Deviation as Psychopathology Biomarker."""
    logger.info("Figure 5: MGD Biomarker")
    mgd_path = RESULT_DIR / "mgd_results.csv"
    if not mgd_path.exists():
        logger.warning("  MGD results not found (run step04b first), skipping")
        return

    mgd_df = pd.read_csv(mgd_path)
    corr_df = pd.read_csv(RESULT_DIR / "mgd_correlations.csv") if \
        (RESULT_DIR / "mgd_correlations.csv").exists() else None
    roc_df = pd.read_csv(RESULT_DIR / "mgd_roc_summary.csv") if \
        (RESULT_DIR / "mgd_roc_summary.csv").exists() else None
    roc_curve_df = pd.read_csv(RESULT_DIR / "mgd_roc_curve.csv") if \
        (RESULT_DIR / "mgd_roc_curve.csv").exists() else None
    strat_df = pd.read_csv(RESULT_DIR / "mgd_age_stratified.csv") if \
        (RESULT_DIR / "mgd_age_stratified.csv").exists() else None

    ages = mgd_df["age"].values
    mgd_alpha = mgd_df["mgd_z_alpha"].values
    p_factor = mgd_df["p_factor"].values
    internalizing = mgd_df["internalizing"].values

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # ── Panel A: MGD across development, colored by p_factor ──
    ax = axes[0, 0]
    mask = ~np.isnan(p_factor) & ~np.isnan(mgd_alpha)
    scatter = ax.scatter(ages[mask], mgd_alpha[mask],
                         c=p_factor[mask], cmap="RdBu_r", s=15,
                         alpha=0.7, edgecolors="none")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("MGD z-score (alpha)")
    ax.set_title("A. MGD Across Development")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    plt.colorbar(scatter, ax=ax, label="P-factor", shrink=0.8)

    # ── Panel B: MGD vs p_factor (age-residualized) ──
    ax = axes[0, 1]
    mask = ~np.isnan(p_factor) & ~np.isnan(mgd_alpha)
    if mask.sum() < 3:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("B. MGD vs P-factor")
    else:
        a = ages[mask]; m = mgd_alpha[mask]; pf = p_factor[mask]
        m_res = m - np.polyval(np.polyfit(a, m, 1), a)
        pf_res = pf - np.polyval(np.polyfit(a, pf, 1), a)
        ax.scatter(pf_res, m_res, s=12, alpha=0.5, c="#4CAF50", edgecolors="none")
        if len(pf_res) >= 2:
            slope, intercept = np.polyfit(pf_res, m_res, 1)
            ax.plot(pf_res, slope * pf_res + intercept, "r--", lw=1.5)
            r, p = pearsonr(pf_res, m_res)
        else:
            r, p = np.nan, np.nan
    p_fdr, sig = p, ""
    if corr_df is not None:
        row = corr_df[(corr_df["band"] == "alpha") &
                      (corr_df["variable"] == "p_factor")]
        if len(row) > 0:
            p_fdr = row["p_fdr"].values[0]
            sig = row["sig"].values[0]
    ax.set_xlabel("P-factor (age-residualized)")
    ax.set_ylabel("MGD z-score (age-residualized)")
    ax.set_title(f"B. MGD vs P-factor\nr={r:.3f}, p_fdr={p_fdr:.4f} {sig}")

    # ── Panel C: MGD vs internalizing ──
    ax = axes[0, 2]
    mask = ~np.isnan(internalizing) & ~np.isnan(mgd_alpha)
    if mask.sum() < 3:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("C. MGD vs Internalizing")
    else:
        a = ages[mask]; m = mgd_alpha[mask]; intl = internalizing[mask]
        m_res = m - np.polyval(np.polyfit(a, m, 1), a)
        intl_res = intl - np.polyval(np.polyfit(a, intl, 1), a)
        ax.scatter(intl_res, m_res, s=12, alpha=0.5, c="#FF9800", edgecolors="none")
        if len(intl_res) >= 2:
            slope, intercept = np.polyfit(intl_res, m_res, 1)
            ax.plot(intl_res, slope * intl_res + intercept, "r--", lw=1.5)
            r, p = pearsonr(intl_res, m_res)
        else:
            r, p = np.nan, np.nan
    p_fdr, sig = p, ""
    if corr_df is not None:
        row = corr_df[(corr_df["band"] == "alpha") &
                      (corr_df["variable"] == "internalizing")]
        if len(row) > 0:
            p_fdr = row["p_fdr"].values[0]
            sig = row["sig"].values[0]
    ax.set_xlabel("Internalizing (age-residualized)")
    ax.set_ylabel("MGD z-score (age-residualized)")
    ax.set_title(f"C. MGD vs Internalizing\nr={r:.3f}, p_fdr={p_fdr:.4f} {sig}")

    # ── Panel D: ROC curves ──
    ax = axes[1, 0]
    if roc_curve_df is not None and roc_df is not None:
        # Plot best band ROC
        best_band = roc_df.loc[roc_df["auc"].idxmax(), "band"]
        best_data = roc_curve_df[roc_curve_df["band"] == best_band] if \
            "band" in roc_curve_df.columns else roc_curve_df
        ax.plot(best_data["fpr"], best_data["tpr"],
                color=BAND_COLORS.get(best_band, "#212121"), lw=2,
                label=f"{best_band.capitalize()} "
                      f"(AUC={roc_df[roc_df['band']==best_band]['auc'].values[0]:.3f})")

        # Plot combined if available
        if "combined" in roc_df["band"].values:
            # The combined band's ROC curve is stored in the same
            # mgd_roc_curve.csv file (with band=="combined"), not in a
            # separate mgd_roc_curve_combined.csv file (which was never
            # created by step04b).
            if roc_curve_df is not None and "band" in roc_curve_df.columns and \
                    "combined" in roc_curve_df["band"].values:
                combined_curve = roc_curve_df[roc_curve_df["band"] == "combined"]
                ax.plot(combined_curve["fpr"], combined_curve["tpr"],
                        color="#212121", lw=2,
                        label=f"Combined "
                              f"(AUC={roc_df[roc_df['band']=='combined']['auc'].values[0]:.3f})")
            else:
                # Fall back to a legend-only marker so the AUC is still
                # communicated to the reader (e.g. when no curve CSV exists)
                ax.plot([], [], color="#212121", lw=2,
                        label=f"Combined "
                              f"(AUC={roc_df[roc_df['band']=='combined']['auc'].values[0]:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("D. ROC: MGD Risk Classification")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No ROC data", ha="center", va="center")
        ax.set_title("D. ROC")

    # ── Panel E: Per-band AUC bar chart ──
    ax = axes[1, 1]
    if roc_df is not None:
        bands_list = roc_df["band"].tolist()
        aucs = roc_df["auc"].values
        ci_los = roc_df["ci_lo"].values
        ci_his = roc_df["ci_hi"].values
        colors = [BAND_COLORS.get(b, "#212121") for b in bands_list]
        ax.bar(range(len(bands_list)), aucs,
               yerr=[aucs - ci_los, ci_his - aucs],
               color=colors, alpha=0.8, capsize=3)
        ax.set_xticks(range(len(bands_list)))
        ax.set_xticklabels(bands_list, rotation=45, ha="right")
        ax.axhline(0.5, color="k", ls="--", lw=0.8, label="Chance")
        ax.set_ylabel("AUC")
        ax.set_title("E. AUC by Frequency Band")
        ax.set_ylim(0.3, 0.9)
        for i, row in enumerate(roc_df.itertuples(index=False)):
            if getattr(row, "p_fdr", 1) < 0.05:
                ax.text(i, row.ci_hi + 0.02, "*",
                        ha="center", fontsize=12, fontweight="bold")
    else:
        ax.set_title("E. AUC (no data)")

    # ── Panel F: Age-stratified heatmap ──
    ax = axes[1, 2]
    if strat_df is not None and len(strat_df) > 0:
        strat_alpha = strat_df[strat_df["band"] == "alpha"]
        if len(strat_alpha) > 0:
            pivot = strat_alpha.pivot_table(
                index="variable", columns="age_group",
                values="r", aggfunc="first")
            n_pivot = strat_alpha.pivot_table(
                index="variable", columns="age_group",
                values="n", aggfunc="first")
            im = ax.imshow(pivot.values, cmap="RdBu_r",
                           vmin=-0.3, vmax=0.3, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        n_cell = n_pivot.values[i, j]
                        ax.text(j, i, f"{val:.2f}\nn={int(n_cell)}",
                                ha="center", va="center", fontsize=7,
                                color="white" if abs(val) > 0.15
                                else "black")
            plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
            ax.set_title("F. Age-Stratified MGD-CBCL r "
                         "(alpha band)")
    else:
        ax.set_title("F. Age-Stratified (no data)")

    fig.suptitle("Manifold Geometry Deviation as a Biomarker "
                 "for Psychopathology Risk", y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, "figure05_mgd_biomarker.png")


# ── F6: Normative trajectory and deviation topography ───────────────────

def figure06():
    """Figure 6: Normative Trajectory and Deviation Topography."""
    logger.info("Figure 6: Normative Trajectory & Deviation")
    data = _load_rest_features()
    if data is None:
        return

    mgd_path = RESULT_DIR / "mgd_results.csv"
    if not mgd_path.exists():
        logger.warning("  MGD results not found (run step04b first), skipping")
        return

    mgd_df = pd.read_csv(mgd_path)
    covmats = data["covmats"]
    ages = data["ages"]
    p_factor = data["p_factor"]
    bidx = BAND_ORDER.index("alpha")

    try:
        from pyriemann.geometry.mean import mean_riemann
    except ImportError:
        from pyriemann.utils.mean import mean_riemann
    from scipy.linalg import logm as scipy_logm

    from step04b_mgd_analysis import (
        weighted_log_euclidean_mean, gaussian_kernel_weights)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # ── Panel A: Normative SPD matrices for 3 age groups ──
    group_centers = {"5-8": 6.5, "9-12": 10.5, "13-16": 14.5}
    all_corrs = {}
    for g, center_age in group_centers.items():
        weights = gaussian_kernel_weights(center_age, ages, bandwidth=1.5)
        mean_cov = weighted_log_euclidean_mean(
            covmats[:, bidx, :, :], weights)
        diag = np.sqrt(np.diag(mean_cov))
        corr = mean_cov / np.outer(diag, diag)
        all_corrs[g] = corr

    all_off_diag = np.concatenate(
        [c[np.triu_indices(c.shape[0], k=1)] for c in all_corrs.values()])
    vmax = max(np.percentile(np.abs(all_off_diag), 99), 1e-6)

    for i, (g, corr) in enumerate(all_corrs.items()):
        ax = axes[0, i]
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"A{i+1}. Normative (age {g})")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)

    # ── Panel B: Deviation matrix (high MGD - normative) ──
    ax = axes[1, 0]
    mgd_alpha = mgd_df["mgd_z_alpha"].values
    valid_mgd = mgd_alpha[~np.isnan(mgd_alpha)]
    high_mask = mgd_alpha > np.percentile(valid_mgd, 75)
    high_mask = high_mask & ~np.isnan(mgd_alpha)

    if high_mask.sum() > 3:
        mean_high = mean_riemann(covmats[high_mask, bidx, :, :])
        avg_age_high = ages[high_mask].mean()
        weights_norm = gaussian_kernel_weights(avg_age_high, ages, 1.5)
        mean_norm = weighted_log_euclidean_mean(
            covmats[:, bidx, :, :], weights_norm)

        deviation = scipy_logm(mean_high) - scipy_logm(mean_norm)
        dev_max = np.percentile(np.abs(deviation), 95)
        im_dev = ax.imshow(deviation, cmap="RdBu_r", vmin=-dev_max, vmax=dev_max)
        ax.set_title(f"B. Deviation (High MGD, n={high_mask.sum()})")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im_dev, ax=ax, shrink=0.8)
    else:
        ax.set_title("B. Deviation (insufficient data)")

    # ── Panel C: MGD by age group x risk group ──
    ax = axes[1, 1]
    mask = ~np.isnan(p_factor)
    pf = p_factor[mask]
    mgd = mgd_alpha[mask]
    ag = np.array([age_group(a) for a in ages[mask]])

    hi_mask = pf > np.percentile(pf, RISK_PERCENTILE_HIGH)
    lo_mask = pf < np.percentile(pf, RISK_PERCENTILE_LOW)

    group_order = ["5-8", "9-12", "13-16", "17-21"]
    hi_means = []; lo_means = []; hi_sems = []; lo_sems = []
    for g in group_order:
        g_mask = ag == g
        hi_g = g_mask & hi_mask
        lo_g = g_mask & lo_mask
        hi_means.append(mgd[hi_g].mean() if hi_g.sum() > 2 else np.nan)
        lo_means.append(mgd[lo_g].mean() if lo_g.sum() > 2 else np.nan)
        hi_sems.append(mgd[hi_g].std()/np.sqrt(hi_g.sum()) if hi_g.sum() > 2 else 0)
        lo_sems.append(mgd[lo_g].std()/np.sqrt(lo_g.sum()) if lo_g.sum() > 2 else 0)

    x = np.arange(len(group_order))
    width = 0.35
    ax.bar(x - width/2, hi_means, width, yerr=hi_sems,
           color=RISK_COLORS["high"], alpha=0.8, capsize=3, label="High risk")
    ax.bar(x + width/2, lo_means, width, yerr=lo_sems,
           color=RISK_COLORS["low"], alpha=0.8, capsize=3, label="Low risk")
    ax.set_xticks(x)
    ax.set_xticklabels(group_order)
    ax.set_ylabel("MGD z-score (alpha)")
    ax.set_title("C. MGD by Age x Risk Group")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", ls="--", lw=0.5)

    # ── Panel D: Developmental trajectory by risk group ──
    ax = axes[1, 2]
    mask = ~np.isnan(p_factor)
    a = ages[mask]; m = mgd_alpha[mask]; pf = p_factor[mask]
    hi = pf > np.percentile(pf, RISK_PERCENTILE_HIGH)
    lo = pf < np.percentile(pf, RISK_PERCENTILE_LOW)

    ax.scatter(a[hi], m[hi], c=RISK_COLORS["high"], s=12,
               alpha=0.4, label="High risk")
    ax.scatter(a[lo], m[lo], c=RISK_COLORS["low"], s=12,
               alpha=0.4, label="Low risk")

    # LOESS-like smoothing
    from scipy.interpolate import UnivariateSpline
    for mask_risk, color in [(hi, RISK_COLORS["high"]),
                              (lo, RISK_COLORS["low"])]:
        if mask_risk.sum() > 10:
            a_r = a[mask_risk]; m_r = m[mask_risk]
            order = np.argsort(a_r)
            try:
                spline = UnivariateSpline(a_r[order], m_r[order],
                                          k=3, s=len(a_r))
                a_grid = np.linspace(a_r.min(), a_r.max(), 100)
                ax.plot(a_grid, spline(a_grid), color=color, lw=2)
            except Exception:
                pass

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("MGD z-score (alpha)")
    ax.set_title("D. Developmental Trajectory by Risk")
    ax.legend(fontsize=8)

    fig.suptitle("Age-Normative Trajectory on the Riemannian Manifold "
                 "and Deviation Topography", y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, "figure06_mgd_trajectory.png")


# ── F7: Cortical mapping ────────────────────────────────────────────────

def figure07():
    logger.info("Figure 7: Cortical Mapping")
    data = _load_rest_features()
    if data is None:
        return
    import mne
    mne.set_log_level("ERROR")

    covmats = data["covmats"]
    ages = data["ages"]
    groups = [age_group(a) for a in ages]
    bidx = BAND_ORDER.index("alpha")

    # montage for topomap
    montage = mne.channels.make_standard_montage(MONTAGE)
    info = mne.create_info(montage.ch_names[:N_CHANNELS], SFREQ, "eeg")
    info.set_montage(montage, on_missing='ignore')

    # Get 2D electrode positions from the montage
    pos_2d = np.array([info.get_montage().get_positions()["ch_pos"][ch][:2]
                       for ch in info.ch_names])

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for col, g in enumerate(["5-8", "9-12", "13-16", "17-21"]):
        ax = axes[col]
        mask = np.array([gr == g for gr in groups])
        n = mask.sum()
        if n < 3:
            ax.set_title(f"{g} (n={n})")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        mean_cov = np.mean(covmats[mask, bidx, :, :], axis=0)

        # Plot top 5% strongest connections
        iu = np.triu_indices(N_CHANNELS, k=1)
        conn_vals = np.abs(mean_cov[iu])
        threshold = np.percentile(conn_vals, 95)  # top 5%

        # Blank white scalp with anatomical landmarks (head circle, nose,
        # ears).  We use MNE's plot_topomap on a zero-valued vector so the
        # background is uniform white but the head outline is preserved.
        mne.viz.plot_topomap(np.zeros(N_CHANNELS), info, axes=ax,
                             show=False, cmap=None, contours=0,
                             outlines='head', sensors=False)

        # Draw connection lines for top 5% (zorder 4, on top of outline)
        drawn = 0
        for i in range(N_CHANNELS):
            for j in range(i + 1, N_CHANNELS):
                if abs(mean_cov[i, j]) >= threshold:
                    ax.plot([pos_2d[i, 0], pos_2d[j, 0]],
                            [pos_2d[i, 1], pos_2d[j, 1]],
                            color='crimson', alpha=0.55, linewidth=0.6,
                            zorder=4)
                    drawn += 1

        # Draw electrode positions (small, don't obscure connections)
        ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c='black', s=6, zorder=5)

        ax.set_title(f"{g} years (n={n}, {drawn} links)")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Alpha Band — Top 5% Channel-Pair Covariance by Age Group",
                 y=1.02)
    fig.tight_layout()
    _save(fig, "figure07_cortical_mapping.png")


# ── F8: Rest-task shift ────────────────────────────────────────────────

def figure08():
    logger.info("Figure 8: Rest-Task Shift")
    rest_data = _load_rest_features()
    if rest_data is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for row, (seq_task, title) in enumerate(
        [(SEQ6_TASK, "6-target (≤8y)"), (SEQ8_TASK, "8-target (≥9y)")]
    ):
        # Try merged release files first, then single file
        seq_release_files = sorted(FEATURE_DIR.glob(f"{seq_task}_features_R*.npz"))
        if seq_release_files:
            seq_data = _merge_release_files(seq_release_files)
        else:
            seq_path = FEATURE_DIR / f"{seq_task}_features.npz"
            if not seq_path.exists():
                axes[row, 0].text(0.5, 0.5, f"{seq_task}\nnot found",
                                  ha="center", va="center")
                continue
            seq_data = np.load(seq_path, allow_pickle=True)
        common = [s for s in seq_data["subject_ids"]
                  if s in rest_data["subject_ids"]]
        if len(common) < 5:
            continue

        rest_map = {s: i for i, s in enumerate(rest_data["subject_ids"])}
        seq_map = {s: i for i, s in enumerate(seq_data["subject_ids"])}
        ridx = [rest_map[s] for s in common]
        sidx = [seq_map[s] for s in common]
        ages_common = rest_data["ages"][ridx]

        bidx = BAND_ORDER.index("alpha")
        # Compute joint tangent space for rest + task data so they share
        # the same reference point.  Using separately-projected tangent
        # vectors would mix different coordinate systems and produce
        # artificial separation in the t-SNE.
        rest_cov = rest_data["covmats"][ridx, bidx, :, :]
        seq_cov = seq_data["covmats"][sidx, bidx, :, :]
        joint_cov = np.vstack([rest_cov, seq_cov])
        ts_joint = TangentSpace(metric="riemann")
        joint_tangent = ts_joint.fit_transform(joint_cov)
        rest_t = joint_tangent[:len(common)]
        seq_t = joint_tangent[len(common):]
        combined = np.vstack([rest_t, seq_t])
        combined[np.isnan(combined)] = 0
        labels = np.array(["Rest"] * len(common) + ["Task"] * len(common))

        ts = TSNE(n_components=2, perplexity=min(30, len(combined) - 1),
                  random_state=RANDOM_STATE).fit_transform(combined)

        # A/B: t-SNE scatter
        ax = axes[row, 0]
        for lbl, color in [("Rest", "#2196F3"), ("Task", "#F44336")]:
            m = labels == lbl
            ax.scatter(ts[m, 0], ts[m, 1], c=color, s=12, alpha=0.6, label=lbl)
        ax.set_title(f"{'A' if row==0 else 'B'}. {title}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()

        # C: centroid distance (log-spaced bins for highly skewed data)
        ax = axes[row, 1]
        shift_path = RESULT_DIR / f"task_shift_{seq_task}_subjects.csv"
        if shift_path.exists():
            shift = pd.read_csv(shift_path)
            dist = shift["rest_task_distance"].values
            dist_pos = dist[dist > 0]  # exclude exact zeros for log binning
            if len(dist_pos) > 10:
                log_edges = np.logspace(
                    np.log10(max(dist_pos.min(), 1e-6)),
                    np.log10(dist_pos.max()),
                    21  # 20 bins
                )
                counts, edges = np.histogram(dist_pos, bins=log_edges)
                ax.bar(edges[:-1], counts, width=np.diff(edges),
                       align='edge', color="#4CAF50", alpha=0.7, edgecolor='white',
                       linewidth=0.3)
                ax.set_xscale('log')
            else:
                ax.hist(dist, bins=20, color="#4CAF50", alpha=0.7)
            ax.set_xlabel("Rest-Task Distance (log scale)")
            ax.set_ylabel("Count")
            summary = pd.read_csv(RESULT_DIR / f"task_shift_{seq_task}.csv")
            r_age = summary["r_age"].values[0]
            ax.set_title(f"{'C' if row==0 else 'D'}. Shift (r_age={r_age:.3f})")

        # D/E: distance vs age and accuracy
        ax = axes[row, 2]
        if shift_path.exists():
            shift = pd.read_csv(shift_path)
            ax.scatter(shift["age"], shift["rest_task_distance"],
                       s=12, alpha=0.5, c="#FF9800")
            if len(shift) >= 2:
                m, b = np.polyfit(shift["age"], shift["rest_task_distance"], 1)
                ax.plot(shift["age"], m * shift["age"] + b, "r--", lw=1)
            ax.set_xlabel("Age")
            ax.set_ylabel("Rest-Task Distance")
            ax.set_title(f"{'E' if row==0 else 'F'}. Distance vs Age")

    fig.tight_layout()
    _save(fig, "figure08_rest_task_shift.png")


# ── F9: LOSO and ablation ───────────────────────────────────────────────

def figure09():
    logger.info("Figure 9: LOSO and Ablation")
    metrics_path = RESULT_DIR / "cv5_metrics.csv"
    fi_path = RESULT_DIR / "feature_importance.csv"
    if not metrics_path.exists():
        logger.warning("  Metrics not found, skipping")
        return

    metrics = pd.read_csv(metrics_path)
    fi = pd.read_csv(fi_path) if fi_path.exists() else None

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # A: single-band COR bar chart
    ax = axes[0]
    band_models = [f"SVR_{b}" for b in BAND_ORDER]
    band_cors = []
    for m in band_models:
        row = metrics[metrics["model"] == m]
        band_cors.append(row["cor"].values[0] if len(row) > 0 else 0)
    fbts_cor = metrics.loc[metrics["model"] == "FBTS_SVR", "cor"].values
    fbts_cor = fbts_cor[0] if len(fbts_cor) > 0 else 0

    all_labels = BAND_ORDER + ["FBTS"]
    all_cors = band_cors + [fbts_cor]
    colors = [BAND_COLORS[b] for b in BAND_ORDER] + ["#212121"]
    ax.bar(all_labels, all_cors, color=colors, alpha=0.8)
    ax.set_ylabel("COR")
    ax.set_title("A. Single-Band vs Full FBTS")
    ax.set_ylim(0, max(all_cors) * 1.2)

    # B: feature selection proportion (horizontal bar, clearer than pie for extreme distributions)
    ax = axes[1]
    if fi is not None:
        total = fi["n_selected"].sum()
        pcts = fi["n_selected"] / total * 100
        bar_colors = [BAND_COLORS[b] for b in fi["band"]]
        bars = ax.barh(fi["band"], pcts, color=bar_colors, alpha=0.8,
                       edgecolor='white', linewidth=0.5)
        for bar, pct, nsel in zip(bars, pcts, fi["n_selected"]):
            label = f"{pct:.0f}% (n={nsel})" if nsel > 0 else f"{pct:.0f}%"
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                    label, va='center', fontsize=7)
        ax.set_xlabel("Selected features (%)")
        ax.set_xlim(0, 115)
        ax.set_title(f"B. Feature Selection (n={total})")

    # C: performance drop (FBTS - single-band)
    ax = axes[2]
    drops = [fbts_cor - c for c in band_cors]
    ax.bar(BAND_ORDER, drops, color=[BAND_COLORS[b] for b in BAND_ORDER], alpha=0.8)
    ax.set_ylabel("COR Drop (FBTS − band)")
    ax.set_title("C. Band Contribution")
    ax.axhline(0, color="k", ls="--", lw=0.8)

    fig.tight_layout()
    _save(fig, "figure09_loso_ablation.png")


# ── F10: Sex differences ────────────────────────────────────────────────

def figure10():
    logger.info("Figure 10: Sex Differences")
    data = _load_rest_features()
    if data is None:
        return
    fbts = data["fbts"]
    ages = data["ages"]
    sexes = data["sexes"]
    fbts[np.isnan(fbts)] = 0

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # A: t-SNE by sex
    ax = axes[0, 0]
    ts = TSNE(n_components=2, perplexity=min(30, len(fbts) - 1),
              random_state=RANDOM_STATE).fit_transform(fbts)
    for s, color, label in [("M", "#2196F3", "Male"), ("F", "#E91E63", "Female")]:
        m = sexes == s
        ax.scatter(ts[m, 0], ts[m, 1], c=color, s=12, alpha=0.5, label=label)
    ax.set_title("A. t-SNE by Sex")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()

    # B: sex distance by age
    ax = axes[0, 1]
    sex_age_path = RESULT_DIR / "sex_distance_by_age.csv"
    if sex_age_path.exists():
        sad = pd.read_csv(sex_age_path)
        ax.bar(sad["age"], sad["distance"], color="#9C27B0", alpha=0.8)
        ax.set_xlabel("Age")
        ax.set_ylabel("Male-Female Distance")
        ax.set_title("B. Sex Distance by Age")

    # C: permutation test histogram
    ax = axes[1, 0]
    perm_path = RESULT_DIR / "sex_permutation_dist.csv"
    test_path = RESULT_DIR / "sex_permutation_test.csv"
    if perm_path.exists() and test_path.exists():
        perm = np.loadtxt(perm_path, delimiter=",")
        test = pd.read_csv(test_path)
        actual = test["actual_distance"].values[0]
        p_val = test["p_value"].values[0]
        ax.hist(perm, bins=50, color="#607D8B", alpha=0.7, edgecolor="white")
        ax.axvline(actual, color="#F44336", lw=2, ls="--",
                   label=f"Actual={actual:.3f}\np={p_val:.4f}")
        ax.set_xlabel("Permutation Distance")
        ax.set_ylabel("Count")
        ax.set_title("C. Permutation Test")
        ax.legend()

    # D: cross-prediction
    ax = axes[1, 1]
    male_mask = sexes == "M"
    female_mask = sexes == "F"
    if male_mask.sum() > 20 and female_mask.sum() > 20:
        from step03_brain_age_model import build_svr_pipeline, SVR_GRID, compute_metrics
        results = []
        for train_mask, test_mask, label in [
            (male_mask, female_mask, "M→F"),
            (female_mask, male_mask, "F→M"),
        ]:
            X_tr, y_tr = fbts[train_mask], ages[train_mask]
            X_te, y_te = fbts[test_mask], ages[test_mask]
            gs = GridSearchCV(build_svr_pipeline(), SVR_GRID, cv=3,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(X_tr, y_tr)
            pred = gs.predict(X_te)
            m = compute_metrics(y_te, pred)
            results.append({"label": label, "cor": m["cor"], "rmse": m["rmse"]})
            logger.info("  Cross-pred %s: COR=%.3f RMSE=%.3f", label, m["cor"], m["rmse"])
        res_df = pd.DataFrame(results)
        ax.bar(res_df["label"], res_df["cor"], color=["#2196F3", "#E91E63"], alpha=0.8)
        ax.set_ylabel("COR")
        ax.set_title("D. Cross-Sex Prediction")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")

    fig.tight_layout()
    _save(fig, "figure10_sex_differences.png")


# ── Main ────────────────────────────────────────────────────────────────

FIGURES = [
    ("F1", figure01), ("F2", figure02), ("F3", figure03),
    ("F4", figure04), ("F5", figure05), ("F6", figure06),
    ("F7", figure07), ("F8", figure08), ("F9", figure09),
    ("F10", figure10),
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--only", default=None,
                        help="Generate only one figure (e.g. F1)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step 5: Figure Generation")
    logger.info("Output: %s", FIGURE_DIR)
    logger.info("=" * 60)

    for name, func in FIGURES:
        if args.only and args.only.upper() != name:
            continue
        try:
            func()
        except Exception as exc:
            logger.error("%s failed: %s", name, exc)
            import traceback
            logger.debug(traceback.format_exc())

    logger.info("\nFigure generation complete.")


if __name__ == "__main__":
    main()
