# -*- coding: utf-8 -*-
"""
Step 5 (Nature) — Redraw all publication figures in Nature style.

Output: SVG (editable text) + PNG (300 dpi) to paper/JNER/ and results/figures/.
Only redraws figures with available data (F1–F4, F7–F10).
F5/F6 require MGD results from step04b (not yet available).
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.manifold import TSNE

from config import *
from utils import setup_logging, age_group
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore")
logger = setup_logging("step05_nature")

# ── Nature style configuration ──────────────────────────────────────────
plt.rcParams.update({
    # Mandatory: editable SVG text
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    # Compact journal sizing
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    # Clean spines
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    # Output
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Palette (from Nature figure skill API) ──────────────────────────────
P = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
    "neutral_black":  "#272727",
    "gold":           "#FFD700",
}

# Age group colors — restrained clinical palette
AGE_COLORS = {
    "5-8":   P["blue_main"],
    "9-12":  P["teal"],
    "13-16": "#D24B40",
    "17-21": P["violet"],
}

# Band colors — coherent hue family
BAND_COLORS = {
    "delta": P["violet"],
    "theta": P["blue_main"],
    "alpha": P["teal"],
    "beta":  "#D28E2C",
    "gamma": P["red_strong"],
}

RISK_COLORS = {"high": P["red_strong"], "low": P["blue_main"]}
SEX_COLORS = {"M": P["blue_main"], "F": P["red_strong"]}

# Output directories
PAPER_DIR = PROJECT_ROOT / "paper" / "HumanBrainMapping"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    """Save figure as SVG + PNG to both paper dir and results/figures."""
    stem = name.replace(".png", "").replace(".svg", "")
    for out_dir in [PAPER_DIR, FIGURE_DIR]:
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", stem)


def _panel_label(ax, label, x=-0.12, y=1.05):
    """Place a Nature-style bold lowercase panel label."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="black",
            ha="left", va="bottom")


def _load_rest_features():
    """Load resting features: merge all release files if >1, else single file.
    
    Re-projects tangent space on merged covariances to ensure all release
    features share the same Riemannian reference point.
    """
    release_files = sorted(FEATURE_DIR.glob(f"{REST_TASK}_features_R*.npz"))
    if len(release_files) > 1:
        all_data = [np.load(f, allow_pickle=True) for f in release_files]
        n_ref = len(all_data[0]["ages"])
        merged = {}
        for key in all_data[0].files:
            arr = all_data[0][key]
            if hasattr(arr, '__len__') and len(arr) == n_ref:
                merged[key] = np.concatenate([d[key] for d in all_data])
            else:
                merged[key] = arr

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

        # Re-project tangent space on merged covariances
        if "covmats" in merged and "tangent" in merged:
            covmats = merged["covmats"]
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

        class MergedNpz:
            def __init__(self, d):
                self._d = d
            def __getitem__(self, k):
                return self._d[k]
            @property
            def files(self):
                return list(self._d.keys())
        return MergedNpz(merged)
    if len(release_files) == 1:
        return np.load(release_files[0], allow_pickle=True)
    p = FEATURE_DIR / f"{REST_TASK}_features.npz"
    if p.exists():
        return np.load(p, allow_pickle=True)
    return None


def _load_seq_features(seq_task):
    """Load sequence learning task features (merged releases).
    
    Re-projects tangent space on merged covariances for consistent
    coordinate system across releases.
    """
    release_files = sorted(FEATURE_DIR.glob(f"{seq_task}_features_R*.npz"))
    if release_files:
        all_data = [np.load(f, allow_pickle=True) for f in release_files]
        n_ref = len(all_data[0]["ages"])
        merged = {}
        for key in all_data[0].files:
            arr = all_data[0][key]
            if hasattr(arr, '__len__') and len(arr) == n_ref:
                merged[key] = np.concatenate([d[key] for d in all_data])
            else:
                merged[key] = arr

        # Deduplicate subjects across releases
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

        # Re-project tangent space on merged covariances
        if "covmats" in merged and "tangent" in merged:
            covmats = merged["covmats"]
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
            logger.info("  Re-projected tangent space for %d merged %s subjects",
                        n_total, seq_task)

        class MergedSeq:
            def __init__(self, d):
                self._d = d
            def __getitem__(self, k):
                return self._d[k]
            @property
            def files(self):
                return list(self._d.keys())
        return MergedSeq(merged)
    p = FEATURE_DIR / f"{seq_task}_features.npz"
    if p.exists():
        return np.load(p, allow_pickle=True)
    return None


# ═════════════════════════════════════════════════════════════════════════
# F1: Demographics
# ═════════════════════════════════════════════════════════════════════════

def figure01():
    """Demographics: age distribution, p-factor by age group, sex × age counts."""
    logger.info("Figure 1: Demographics (Nature)")
    data = _load_rest_features()
    if data is None:
        return
    ages = data["ages"]
    sexes = data["sexes"]
    pf = data["p_factor"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))

    # ── Panel a: Age histogram by sex (overlaid) ──
    ax = axes[0]
    bins = np.arange(5, 22, 1)
    for s, color, label in [("M", P["blue_main"], "Male"),
                            ("F", P["red_strong"], "Female")]:
        ax.hist(ages[sexes == s], bins=bins, alpha=0.55,
                color=color, edgecolor="white", linewidth=0.3, label=label)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    _panel_label(ax, "a")

    # ── Panel b: P-factor boxplot by age group ──
    ax = axes[1]
    groups = [age_group(a) for a in ages]
    group_order = ["5-8", "9-12", "13-16", "17-21"]
    box_data = [pf[np.array(groups) == g] for g in group_order]
    box_data = [d[~np.isnan(d)] for d in box_data]
    bp = ax.boxplot(box_data, labels=group_order, patch_artist=True,
                    widths=0.6, showfliers=False,
                    medianprops=dict(color="black", linewidth=0.8),
                    whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6),
                    flierprops=dict(marker=".", markersize=1.5))
    for patch, g in zip(bp["boxes"], group_order):
        patch.set_facecolor(AGE_COLORS[g])
        patch.set_alpha(0.5)
        patch.set_edgecolor(AGE_COLORS[g])
        patch.set_linewidth(0.6)
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel("P-factor")
    _panel_label(ax, "b")

    # ── Panel c: Sex × age heatmap ──
    ax = axes[2]
    age_bins = np.arange(5, 22)
    counts = np.zeros((2, len(age_bins)))
    for j, a in enumerate(age_bins):
        counts[0, j] = np.sum((ages >= a) & (ages < a + 1) & (sexes == "M"))
        counts[1, j] = np.sum((ages >= a) & (ages < a + 1) & (sexes == "F"))
    im = ax.imshow(counts, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Male", "Female"])
    ax.set_xticks(range(0, len(age_bins), 2))
    ax.set_xticklabels(age_bins[::2])
    ax.set_xlabel("Age (years)")
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Count", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "c")

    fig.tight_layout(w_pad=1.5)
    _save(fig, "figure01_demographics.png")


# ═════════════════════════════════════════════════════════════════════════
# F2: t-SNE visualization
# ═════════════════════════════════════════════════════════════════════════

def figure02():
    """t-SNE of FBTS and single-band tangent vectors, colored by age."""
    logger.info("Figure 2: t-SNE (Nature)")
    data = _load_rest_features()
    if data is None:
        return
    fbts = data["fbts"].copy()
    tangent = data["tangent"]
    ages = data["ages"]
    fbts[np.isnan(fbts)] = 0

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))

    panels = [
        ("a", "All bands (FBTS)", fbts),
        ("b", "Theta band", tangent[:, BAND_ORDER.index("theta"), :]),
        ("c", "Alpha band", tangent[:, BAND_ORDER.index("alpha"), :]),
    ]
    for ax, (label, _, X) in zip(axes.flat[:3], panels):
        X = X.copy()
        X[np.isnan(X)] = 0
        ts = TSNE(n_components=2, perplexity=min(30, len(X) - 1),
                  random_state=RANDOM_STATE).fit_transform(X)
        sc = ax.scatter(ts[:, 0], ts[:, 1], c=ages, cmap="viridis",
                        s=8, alpha=0.6, edgecolors="none", rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        _panel_label(ax, label)
        cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label("Age (years)", fontsize=6)
        cb.ax.tick_params(labelsize=5)

    # ── Panel d: Age group centroids with 95% CI ellipses ──
    ax = axes[1, 1]
    ts_full = TSNE(n_components=2, perplexity=min(30, len(fbts) - 1),
                   random_state=RANDOM_STATE).fit_transform(fbts)
    for g in ["5-8", "9-12", "13-16", "17-21"]:
        mask = np.array([age_group(a) == g for a in ages])
        if mask.sum() < 3:
            continue
        pts = ts_full[mask]
        cx, cy = pts.mean(axis=0)
        ax.scatter(cx, cy, c=AGE_COLORS[g], s=20, zorder=5, label=g,
                   edgecolors="white", linewidths=0.3)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        w, h = 2 * np.sqrt(np.maximum(eigvals, 0) * 5.991)
        ell = Ellipse((cx, cy), max(w, 0.1), max(h, 0.1), angle=angle,
                      fill=False, edgecolor=AGE_COLORS[g], linewidth=1.0)
        ax.add_patch(ell)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=5, loc="upper left", markerscale=0.6,
              handletextpad=0.2, borderpad=0.2)
    _panel_label(ax, "d")

    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    _save(fig, "figure02_tsne.png")


# ═════════════════════════════════════════════════════════════════════════
# F3: Model performance
# ═════════════════════════════════════════════════════════════════════════

def figure03():
    """Model performance: predicted vs actual, model comparison, BAG by age."""
    logger.info("Figure 3: Model Performance (Nature)")
    preds_path = RESULT_DIR / "cv5_predictions.csv"
    metrics_path = RESULT_DIR / "cv5_metrics.csv"
    if not preds_path.exists() or not metrics_path.exists():
        logger.warning("  Model results not found, skipping")
        return

    preds = pd.read_csv(preds_path)
    metrics = pd.read_csv(metrics_path)

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = gridspec.GridSpec(1, 3, wspace=0.4)

    # ── Panel a: Predicted vs actual (hero panel) ──
    ax = fig.add_subplot(gs[0])
    # Use ensemble predictions if available, else FBTS-SVR
    pred_col = "pred_ensemble" if "pred_ensemble" in preds.columns else "pred_fbts_svr"
    label_text = "Ensemble" if pred_col == "pred_ensemble" else "FBTS+SVR"
    # Use ensemble BAG if available for consistency with pred_col
    bag_col = "bag_ensemble" if "bag_ensemble" in preds.columns else "bag_fbts"
    ax.scatter(preds["age"], preds[pred_col], s=6, alpha=0.35,
               c=P["blue_main"], edgecolors="none", rasterized=True)
    lim = [min(preds["age"].min(), preds[pred_col].min()) - 1,
           max(preds["age"].max(), preds[pred_col].max()) + 1]
    ax.plot(lim, lim, color=P["neutral_mid"], ls="--", lw=0.6)
    if len(preds) >= 2:
        r, _ = pearsonr(preds["age"], preds[pred_col])
    else:
        r = np.nan
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Predicted age (years)")
    ax.text(0.05, 0.95, f"{label_text}\n$r={r:.3f}$",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.8, edgecolor="none"))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    _panel_label(ax, "a")

    # ── Panel b: Model comparison bar chart ──
    ax = fig.add_subplot(gs[1])
    model_map = [
        ("Ensemble", "Ensemble", P["neutral_black"]),
        ("GBR_FBTS", "GBR", "#D28E2C"),
        ("FBTS_SVR", "FBTS+SVR", P["blue_main"]),
        ("SVR_DE", "DE+SVR", P["teal"]),
        ("SVR_PSD", "PSD+SVR", P["red_strong"]),
        ("Ridge_FBTS", "Ridge", P["neutral_light"]),
    ]
    labels, cors, colors = [], [], []
    for m_name, l_name, color in model_map:
        row = metrics[metrics["model"] == m_name]
        if len(row) > 0:
            labels.append(l_name)
            cors.append(row["cor"].values[0])
            colors.append(color)
    bars = ax.barh(range(len(labels)), cors, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.3, height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("COR")
    ax.invert_yaxis()
    for i, v in enumerate(cors):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=5)
    _panel_label(ax, "b")

    # ── Panel c: BAG distribution by age group ──
    ax = fig.add_subplot(gs[2])
    bag = preds[bag_col].values
    groups = [age_group(a) for a in preds["age"]]
    group_order = ["5-8", "9-12", "13-16", "17-21"]
    box_data = [bag[np.array(groups) == g] for g in group_order]
    bp = ax.boxplot(box_data, labels=group_order, patch_artist=True,
                    widths=0.55, showfliers=False,
                    medianprops=dict(color="black", linewidth=0.7),
                    whiskerprops=dict(linewidth=0.5),
                    capprops=dict(linewidth=0.5))
    for patch, g in zip(bp["boxes"], group_order):
        patch.set_facecolor(AGE_COLORS[g])
        patch.set_alpha(0.5)
        patch.set_edgecolor(AGE_COLORS[g])
        patch.set_linewidth(0.5)
    ax.axhline(0, color=P["neutral_mid"], ls="--", lw=0.5)
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel("BAG (years)")
    _panel_label(ax, "c")

    fig.tight_layout()
    _save(fig, "figure03_model_performance.png")


# ═════════════════════════════════════════════════════════════════════════
# F4: BAG clinical
# ═════════════════════════════════════════════════════════════════════════

def figure04():
    """BAG clinical: distribution, partial correlations, risk group comparison."""
    logger.info("Figure 4: BAG Clinical (Nature)")
    bag_path = RESULT_DIR / "bag_partial_corr.csv"
    group_path = RESULT_DIR / "bag_group_comparison.csv"
    preds_path = RESULT_DIR / "cv5_predictions.csv"
    if not bag_path.exists() or not preds_path.exists():
        logger.warning("  BAG results not found, skipping")
        return

    preds = pd.read_csv(preds_path)
    bag_df = pd.read_csv(bag_path)
    data = _load_rest_features()
    if data is None:
        return

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── Panel a: BAG distribution ──
    ax = fig.add_subplot(gs[0, 0])
    # Use ensemble BAG if available for consistency with Figure 3
    bag_col = "bag_ensemble" if "bag_ensemble" in preds.columns else "bag_fbts"
    ax.hist(preds[bag_col], bins=35, color=P["blue_main"], alpha=0.6,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", ls="--", lw=0.6)
    mu, sigma = preds[bag_col].mean(), preds[bag_col].std()
    ax.set_xlabel("BAG (years)")
    ax.set_ylabel("Count")
    ax.text(0.95, 0.95, f"$\\mu={mu:.2f}$\n$\\sigma={sigma:.2f}$\n$n={len(preds)}$",
            transform=ax.transAxes, fontsize=5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.8, edgecolor="none"))
    _panel_label(ax, "a")

    # ── Panels b, c: Partial regression scatter plots ──
    sid_map = {s: i for i, s in enumerate(data["subject_ids"])}
    pheno = {"p_factor": data["p_factor"], "attention": data["attention"]}
    panel_labels = ["b", "c"]
    keys = ["p_factor", "attention"]
    ylabels = ["P-factor", "Attention"]
    for idx, (pk, yl) in enumerate(zip(keys, ylabels)):
        ax = fig.add_subplot(gs[0, 1] if idx == 0 else gs[1, 0])
        vals = np.array([pheno[pk][sid_map[s]] if s in sid_map else np.nan
                         for s in preds["subject"]])
        age = preds["age"].values
        bag = preds[bag_col].values
        mask = ~np.isnan(vals) & ~np.isnan(bag)
        if mask.sum() < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6)
            _panel_label(ax, panel_labels[idx])
            continue
        bag_res = bag[mask] - np.polyval(np.polyfit(age[mask], bag[mask], 1), age[mask])
        val_res = vals[mask] - np.polyval(np.polyfit(age[mask], vals[mask], 1), age[mask])
        ax.scatter(val_res, bag_res, s=5, alpha=0.35, c=P["teal"],
                   edgecolors="none", rasterized=True)
        if len(val_res) >= 2:
            m_slope, b_int = np.polyfit(val_res, bag_res, 1)
            x_fit = np.linspace(val_res.min(), val_res.max(), 100)
            ax.plot(x_fit, m_slope * x_fit + b_int, color=P["red_strong"],
                    lw=0.8, ls="--")
            r, p = pearsonr(val_res, bag_res)
        else:
            r, p = np.nan, np.nan
        ax.set_xlabel(f"{yl} (age-residualized)")
        ax.set_ylabel("BAG (age-residualized)")
        ax.text(0.05, 0.95, f"$r={r:.3f}$\n$p={p:.3f}$",
                transform=ax.transAxes, fontsize=5, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor="none"))
        _panel_label(ax, panel_labels[idx])

    # ── Panel d: High vs low risk comparison ──
    ax = fig.add_subplot(gs[1, 1])
    group_df = pd.read_csv(group_path)
    hi = group_df[group_df["group"] == "high_risk"]
    lo = group_df[group_df["group"] == "low_risk"]
    if len(hi) > 0 and len(lo) > 0:
        x_pos = [0, 1]
        means = [hi["bag_mean"].values[0], lo["bag_mean"].values[0]]
        stds = [hi["bag_std"].values[0], lo["bag_std"].values[0]]
        ax.bar(x_pos, means, yerr=stds,
               color=[RISK_COLORS["high"], RISK_COLORS["low"]],
               alpha=0.7, capsize=3, width=0.5,
               edgecolor="white", linewidth=0.3,
               error_kw=dict(elinewidth=0.5, capthick=0.5))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["High risk", "Low risk"])
        # Read U statistic and p-value from CSV produced by step04_statistics
        # (bag_group_comparison.csv). The Mann-Whitney row is the third one
        # (test="mannwhitney_u").  Falls back gracefully if columns missing.
        u_val = None
        p_val = None
        test_rows = group_df[group_df["test"] == "mannwhitney_u"] \
            if "test" in group_df.columns else group_df.iloc[0:0]
        if len(test_rows) > 0 and "U" in test_rows.columns:
            u_val = test_rows["U"].values[0]
            p_val = test_rows["p"].values[0]
        if u_val is not None and not pd.isna(u_val):
            ax.text(0.5, 0.95, f"$U={u_val:.0f}$\n$p={p_val:.3f}$",
                    transform=ax.transAxes, fontsize=5, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.8, edgecolor="none"))
        else:
            ax.text(0.5, 0.95, "Mann-Whitney\n$p=$-",
                    transform=ax.transAxes, fontsize=5, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.8, edgecolor="none"))
    ax.axhline(0, color=P["neutral_mid"], ls="--", lw=0.5)
    ax.set_ylabel("Mean BAG (years)")
    _panel_label(ax, "d")

    fig.tight_layout()
    _save(fig, "figure04_bag_clinical.png")


# ═════════════════════════════════════════════════════════════════════════
# F7: Cortical mapping
# ═════════════════════════════════════════════════════════════════════════

def figure07():
    """Cortical topographic mapping of alpha-band functional connectivity."""
    logger.info("Figure 7: Cortical Mapping (Nature)")
    data = _load_rest_features()
    if data is None:
        return
    import mne
    mne.set_log_level("ERROR")

    covmats = data["covmats"]
    ages = data["ages"]
    groups = [age_group(a) for a in ages]
    bidx = BAND_ORDER.index("alpha")

    montage = mne.channels.make_standard_montage(MONTAGE)
    info = mne.create_info(montage.ch_names[:N_CHANNELS], SFREQ, "eeg")
    info.set_montage(montage, on_missing="ignore")
    pos_2d = np.array([info.get_montage().get_positions()["ch_pos"][ch][:2]
                       for ch in info.ch_names])

    group_order = ["5-8", "9-12", "13-16", "17-21"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))

    for col, g in enumerate(group_order):
        ax = axes[col]
        mask = np.array([gr == g for gr in groups])
        n = mask.sum()
        if n < 3:
            ax.text(0.5, 0.5, f"n={n}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        mean_cov = np.mean(covmats[mask, bidx, :, :], axis=0)
        iu = np.triu_indices(N_CHANNELS, k=1)
        conn_vals = np.abs(mean_cov[iu])
        threshold = np.percentile(conn_vals, 95)

        # Topomap background (node degree)
        degree = np.zeros(N_CHANNELS)
        for i in range(N_CHANNELS):
            degree[i] = np.mean(np.abs(np.delete(mean_cov[i, :], i)))
        mne.viz.plot_topomap(degree, info, axes=ax, show=False,
                             cmap="Grays", contours=0)

        # Draw top 5% connection lines
        drawn = 0
        for i in range(N_CHANNELS):
            for j in range(i + 1, N_CHANNELS):
                if abs(mean_cov[i, j]) >= threshold:
                    ax.plot([pos_2d[i, 0], pos_2d[j, 0]],
                            [pos_2d[i, 1], pos_2d[j, 1]],
                            color=P["red_strong"], alpha=0.25, linewidth=0.4)
                    drawn += 1

        ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c="black", s=3, zorder=5)
        ax.set_xticks([])
        ax.set_yticks([])
        # Panel label inside since topomap fills the axes
        ax.text(-0.05, 1.02, chr(ord("a") + col),
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                ha="left", va="bottom")
        ax.set_title(f"{g} years\n($n={n}$, {drawn} links)", fontsize=6,
                     pad=2)

    fig.tight_layout(w_pad=0.5)
    _save(fig, "figure07_cortical_mapping.png")


# ═════════════════════════════════════════════════════════════════════════
# F8: Rest-task shift
# ═════════════════════════════════════════════════════════════════════════

def figure08():
    """Rest-to-task manifold shift for sequence learning tasks."""
    logger.info("Figure 8: Rest-Task Shift (Nature)")
    rest_data = _load_rest_features()
    if rest_data is None:
        logger.warning("  RestingState features not found, skipping Figure 8")
        return

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    for row, (seq_task, title) in enumerate(
        [(SEQ6_TASK, "6-target"), (SEQ8_TASK, "8-target")]
    ):
        shift_path = RESULT_DIR / f"task_shift_{seq_task}_subjects.csv"
        summary_path = RESULT_DIR / f"task_shift_{seq_task}.csv"
        seq_data = _load_seq_features(seq_task) if rest_data is not None else None

        if seq_data is not None:
            # ── Full mode: t-SNE + histogram + scatter ──
            common = [s for s in seq_data["subject_ids"]
                      if s in rest_data["subject_ids"]]
            if len(common) < 5:
                continue

            rest_map = {s: i for i, s in enumerate(rest_data["subject_ids"])}
            seq_map = {s: i for i, s in enumerate(seq_data["subject_ids"])}
            ridx = [rest_map[s] for s in common]
            sidx = [seq_map[s] for s in common]

            bidx = BAND_ORDER.index("alpha")
            rest_t = rest_data["tangent"][ridx, bidx, :]
            seq_t = seq_data["tangent"][sidx, bidx, :]
            combined = np.vstack([rest_t, seq_t])
            combined[np.isnan(combined)] = 0
            labels = np.array(["Rest"] * len(common) + ["Task"] * len(common))

            ts = TSNE(n_components=2, perplexity=min(30, len(combined) - 1),
                      random_state=RANDOM_STATE).fit_transform(combined)

            # ── Panel: t-SNE scatter ──
            ax = fig.add_subplot(gs[row, 0])
            for lbl, color in [("Rest", P["blue_main"]), ("Task", P["red_strong"])]:
                m = labels == lbl
                ax.scatter(ts[m, 0], ts[m, 1], c=color, s=5, alpha=0.45,
                           label=lbl, edgecolors="none", rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize=5, loc="upper left", markerscale=0.5,
                      handletextpad=0.2, borderpad=0.2)
            _panel_label(ax, chr(ord("a") + row * 3))
            ax.set_title(title, fontsize=6, pad=2)
        elif shift_path.exists():
            # ── Fallback mode: CSV-only (feature .npz cleaned up) ──
            ax = fig.add_subplot(gs[row, 0])
            ax.text(0.5, 0.5, f"{title}\n($n$ from CSV)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=6, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            _panel_label(ax, chr(ord("a") + row * 3))
            ax.set_title(title, fontsize=6, pad=2)
        else:
            # ── No data at all ──
            for col in range(3):
                ax = fig.add_subplot(gs[row, col])
                ax.text(0.5, 0.5, f"{seq_task}\nnot found",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
            continue

        # ── Panel: centroid distance histogram (from CSV) ──
        ax = fig.add_subplot(gs[row, 1])
        if shift_path.exists():
            shift = pd.read_csv(shift_path)
            ax.hist(shift["rest_task_distance"], bins=20,
                    color=P["teal"], alpha=0.6, edgecolor="white",
                    linewidth=0.3)
            ax.set_xlabel("Rest–task distance")
            ax.set_ylabel("Count")
            if summary_path.exists():
                summary = pd.read_csv(summary_path)
                r_age = summary["r_age"].values[0]
                p_age = summary["p_age"].values[0]
                n_subj = summary["n"].values[0]
                ax.text(0.95, 0.95,
                        f"$r_{{\\mathrm{{age}}}}={r_age:.3f}$\n$p={p_age:.3f}$\n$n={n_subj}$",
                        transform=ax.transAxes, fontsize=5, ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  alpha=0.8, edgecolor="none"))
        _panel_label(ax, chr(ord("b") + row * 3))

        # ── Panel: distance vs age scatter (from CSV) ──
        ax = fig.add_subplot(gs[row, 2])
        if shift_path.exists():
            shift = pd.read_csv(shift_path)
            ax.scatter(shift["age"], shift["rest_task_distance"],
                       s=5, alpha=0.4, c="#D28E2C", edgecolors="none",
                       rasterized=True)
            if len(shift) >= 2:
                m_sl, b_int = np.polyfit(shift["age"], shift["rest_task_distance"], 1)
                x_fit = np.linspace(shift["age"].min(), shift["age"].max(), 100)
                ax.plot(x_fit, m_sl * x_fit + b_int,
                        color=P["red_strong"], lw=0.8, ls="--")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Rest–task distance")
        _panel_label(ax, chr(ord("c") + row * 3))

    fig.tight_layout()
    _save(fig, "figure08_rest_task_shift.png")


# ═════════════════════════════════════════════════════════════════════════
# F9: LOSO and ablation
# ═════════════════════════════════════════════════════════════════════════

def figure09():
    """Single-band ablation: band vs FBTS performance, feature selection, contribution."""
    logger.info("Figure 9: Ablation (Nature)")
    metrics_path = RESULT_DIR / "cv5_metrics.csv"
    fi_path = RESULT_DIR / "feature_importance.csv"
    if not metrics_path.exists():
        logger.warning("  Metrics not found, skipping")
        return

    metrics = pd.read_csv(metrics_path)
    fi = pd.read_csv(fi_path) if fi_path.exists() else None

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = gridspec.GridSpec(1, 3, wspace=0.4)

    # ── Panel a: Single-band vs FBTS bar chart ──
    ax = fig.add_subplot(gs[0])
    band_models = [f"SVR_{b}" for b in BAND_ORDER]
    band_cors = []
    for m in band_models:
        row = metrics[metrics["model"] == m]
        band_cors.append(row["cor"].values[0] if len(row) > 0 else 0)
    fbts_row = metrics[metrics["model"] == "FBTS_SVR"]
    fbts_cor = fbts_row["cor"].values[0] if len(fbts_row) > 0 else 0

    all_labels = BAND_ORDER + ["FBTS"]
    all_cors = band_cors + [fbts_cor]
    colors = [BAND_COLORS[b] for b in BAND_ORDER] + [P["neutral_black"]]
    bars = ax.bar(range(len(all_labels)), all_cors, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.3, width=0.7)
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=30, ha="right")
    ax.set_ylabel("COR")
    for i, v in enumerate(all_cors):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=5)
    _panel_label(ax, "a")

    # ── Panel b: Feature selection donut ──
    ax = fig.add_subplot(gs[1])
    if fi is not None:
        total = fi["n_selected"].sum()
        wedges, texts, autotexts = ax.pie(
            fi["n_selected"], labels=fi["band"],
            colors=[BAND_COLORS.get(b, P["neutral_mid"]) for b in fi["band"]],
            autopct="%1.1f%%", startangle=90,
            pctdistance=0.78, textprops=dict(fontsize=5),
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=0.5))
        for at in autotexts:
            at.set_fontsize(5)
        ax.text(0, 0, f"$n={total}$", ha="center", va="center", fontsize=6)
    _panel_label(ax, "b")

    # ── Panel c: Band contribution (FBTS - single-band) ──
    ax = fig.add_subplot(gs[2])
    drops = [fbts_cor - c for c in band_cors]
    colors_drop = [BAND_COLORS[b] for b in BAND_ORDER]
    bars = ax.bar(BAND_ORDER, drops, color=colors_drop, alpha=0.85,
                  edgecolor="white", linewidth=0.3, width=0.7)
    ax.set_ylabel("$\\Delta$COR (FBTS $-$ band)")
    ax.axhline(0, color=P["neutral_mid"], ls="--", lw=0.5)
    for i, v in enumerate(drops):
        y_pos = v + 0.002 if v >= 0 else v - 0.006
        va = "bottom" if v >= 0 else "top"
        ax.text(i, y_pos, f"{v:.3f}", ha="center", va=va, fontsize=5)
    ax.set_xticklabels(BAND_ORDER, rotation=30, ha="right")
    _panel_label(ax, "c")

    fig.tight_layout()
    _save(fig, "figure09_loso_ablation.png")


# ═════════════════════════════════════════════════════════════════════════
# F10: Sex differences
# ═════════════════════════════════════════════════════════════════════════

def figure10():
    """Sex differences: t-SNE, distance by age, permutation test, cross-prediction."""
    logger.info("Figure 10: Sex Differences (Nature)")
    data = _load_rest_features()
    if data is None:
        return
    fbts = data["fbts"].copy()
    ages = data["ages"]
    sexes = data["sexes"]
    fbts[np.isnan(fbts)] = 0

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── Panel a: t-SNE by sex ──
    ax = fig.add_subplot(gs[0, 0])
    ts = TSNE(n_components=2, perplexity=min(30, len(fbts) - 1),
              random_state=RANDOM_STATE).fit_transform(fbts)
    for s, color, label in [("M", P["blue_main"], "Male"),
                            ("F", P["red_strong"], "Female")]:
        m = sexes == s
        ax.scatter(ts[m, 0], ts[m, 1], c=color, s=5, alpha=0.4,
                   label=f"{label} (n={m.sum()})", edgecolors="none",
                   rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=5, loc="upper left", markerscale=0.5,
              handletextpad=0.2, borderpad=0.2)
    _panel_label(ax, "a")

    # ── Panel b: Sex distance by age ──
    ax = fig.add_subplot(gs[0, 1])
    sex_age_path = RESULT_DIR / "sex_distance_by_age.csv"
    if sex_age_path.exists():
        sad = pd.read_csv(sex_age_path)
        ax.bar(sad["age"], sad["distance"], color=P["violet"], alpha=0.6,
               edgecolor="white", linewidth=0.2, width=0.8)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Male–female distance")
    _panel_label(ax, "b")

    # ── Panel c: Permutation test ──
    ax = fig.add_subplot(gs[1, 0])
    perm_path = RESULT_DIR / "sex_permutation_dist.csv"
    test_path = RESULT_DIR / "sex_permutation_test.csv"
    if perm_path.exists() and test_path.exists():
        perm = np.loadtxt(perm_path, delimiter=",")
        test = pd.read_csv(test_path)
        actual = test["actual_distance"].values[0]
        p_val = test["p_value"].values[0]
        ax.hist(perm, bins=40, color=P["neutral_mid"], alpha=0.5,
                edgecolor="white", linewidth=0.3)
        ax.axvline(actual, color=P["red_strong"], lw=1.2, ls="--",
                   label=f"Observed = {actual:.3f}\n$p = {p_val:.3f}$")
        ax.set_xlabel("Permutation distance")
        ax.set_ylabel("Count")
        ax.legend(fontsize=5, loc="upper right",
                  handletextpad=0.2, borderpad=0.2)
    _panel_label(ax, "c")

    # ── Panel d: Cross-sex prediction ──
    ax = fig.add_subplot(gs[1, 1])
    male_mask = sexes == "M"
    female_mask = sexes == "F"
    if male_mask.sum() > 20 and female_mask.sum() > 20:
        from step03_brain_age_model import build_svr_pipeline, SVR_GRID, compute_metrics
        from sklearn.model_selection import GridSearchCV
        results = []
        for train_mask, test_mask, label in [
            (male_mask, female_mask, "M$\\to$F"),
            (female_mask, male_mask, "F$\\to$M"),
        ]:
            X_tr, y_tr = fbts[train_mask], ages[train_mask]
            X_te, y_te = fbts[test_mask], ages[test_mask]
            gs_cv = GridSearchCV(build_svr_pipeline(), SVR_GRID, cv=3,
                                 scoring="neg_root_mean_squared_error",
                                 n_jobs=-1)
            gs_cv.fit(X_tr, y_tr)
            pred = gs_cv.predict(X_te)
            m_res = compute_metrics(y_te, pred)
            results.append({"label": label, "cor": m_res["cor"], "rmse": m_res["rmse"]})
            logger.info("  Cross-pred %s: COR=%.3f RMSE=%.3f",
                        label, m_res["cor"], m_res["rmse"])
        res_df = pd.DataFrame(results)
        bars = ax.bar(range(len(res_df)), res_df["cor"],
                      color=[P["blue_main"], P["red_strong"]], alpha=0.7,
                      edgecolor="white", linewidth=0.3, width=0.5)
        ax.set_xticks(range(len(res_df)))
        ax.set_xticklabels(res_df["label"])
        ax.set_ylabel("COR")
        for i, row in res_df.iterrows():
            ax.text(i, row["cor"] + 0.01, f"{row['cor']:.3f}",
                    ha="center", fontsize=5)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=6)
    _panel_label(ax, "d")

    fig.tight_layout()
    _save(fig, "figure10_sex_differences.png")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

FIGURES = [
    ("F1", figure01), ("F2", figure02), ("F3", figure03),
    ("F4", figure04), ("F7", figure07), ("F8", figure08),
    ("F9", figure09), ("F10", figure10),
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Nature-style figures")
    parser.add_argument("--only", default=None,
                        help="Generate only one figure (e.g. F1)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step 5 (Nature): Figure Generation")
    logger.info("Output: %s + %s", PAPER_DIR, FIGURE_DIR)
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

    logger.info("Figure generation complete.")


if __name__ == "__main__":
    main()
