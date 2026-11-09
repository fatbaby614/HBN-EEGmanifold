# -*- coding: utf-8 -*-
"""
Step 5b (Nature) — Generate MGD figures (F5, F6) from step04b results.

F5: MGD as Psychopathology Biomarker (2x3)
F6: Normative Trajectory and Deviation Topography (2x3)
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config import *
from utils import setup_logging, age_group

warnings.filterwarnings("ignore")
logger = setup_logging("step05b_mgd_nature")

# ── Nature style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

P = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "teal": "#42949E", "red_strong": "#B64342",
    "violet": "#9A4D8E", "neutral_light": "#CFCECE",
    "neutral_mid": "#767676", "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}

BAND_COLORS = {
    "delta": P["violet"], "theta": P["blue_main"], "alpha": P["teal"],
    "beta": "#D28E2C", "gamma": P["red_strong"], "combined": P["neutral_black"],
}

RISK_COLORS = {"high": P["red_strong"], "low": P["blue_main"]}

PAPER_DIR = PROJECT_ROOT / "paper" / "HumanBrainMapping"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    stem = name.replace(".png", "").replace(".svg", "")
    for out_dir in [PAPER_DIR, FIGURE_DIR]:
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", stem)


def _panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="black",
            ha="left", va="bottom")


# ═════════════════════════════════════════════════════════════════════════
# F5: MGD as Psychopathology Biomarker
# ═════════════════════════════════════════════════════════════════════════

def figure05():
    """MGD as Psychopathology Biomarker: scatter, ROC, bar chart, heatmap."""
    logger.info("Figure 5: MGD Biomarker (Nature)")
    mgd_path = RESULT_DIR / "mgd_results.csv"
    if not mgd_path.exists():
        logger.warning("  mgd_results.csv not found, skipping")
        return

    mgd_df = pd.read_csv(mgd_path)
    roc_path = RESULT_DIR / "mgd_roc_summary.csv"
    roc_curve_path = RESULT_DIR / "mgd_roc_curve.csv"
    strat_path = RESULT_DIR / "mgd_age_stratified.csv"
    if not roc_path.exists():
        logger.warning("  mgd_roc_summary.csv not found, panels d/e will be empty")
        roc_summary = pd.DataFrame()
    else:
        roc_summary = pd.read_csv(roc_path)
    roc_curve_df = pd.read_csv(roc_curve_path) if roc_curve_path.exists() \
        else pd.DataFrame(columns=["band", "fpr", "tpr"])
    strat_df = pd.read_csv(strat_path) if strat_path.exists() \
        else pd.DataFrame(columns=["band", "age_group", "variable", "r"])

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    pf = mgd_df["p_factor"].values
    mgd_alpha = mgd_df["mgd_z_alpha"].values
    ages = mgd_df["age"].values
    mask = ~np.isnan(pf) & ~np.isnan(mgd_alpha)
    sc = ax.scatter(ages[mask], mgd_alpha[mask], c=pf[mask], cmap="RdBu_r",
                    s=5, alpha=0.5, edgecolors="none", rasterized=True,
                    vmin=-2, vmax=2)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("MGD (alpha, z-scored)")
    cb = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.04, fraction=0.046)
    cb.set_label("P-factor", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    _panel_label(ax, "a")

    # ── Panel b: MGD vs p_factor (age-residualized) ──
    ax = fig.add_subplot(gs[0, 1])
    for band, color in [("alpha", P["teal"]), ("combined", P["neutral_black"])]:
        mgd_vals = mgd_df["mgd_z_" + band].values
        pf_vals = mgd_df["p_factor"].values
        age_vals = mgd_df["age"].values
        m = ~np.isnan(mgd_vals) & ~np.isnan(pf_vals)
        if m.sum() < 3:
            logger.warning("  Panel b: insufficient samples for %s (n=%d), skipping",
                           band, m.sum())
            continue
        mgd_res = mgd_vals[m] - np.polyval(np.polyfit(age_vals[m], mgd_vals[m], 1), age_vals[m])
        pf_res = pf_vals[m] - np.polyval(np.polyfit(age_vals[m], pf_vals[m], 1), age_vals[m])
        ax.scatter(pf_res, mgd_res, s=4, alpha=0.3, c=color, edgecolors="none",
                   rasterized=True, label=band)
        if len(pf_res) >= 2:
            slope, intercept = np.polyfit(pf_res, mgd_res, 1)
            x_fit = np.linspace(pf_res.min(), pf_res.max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, color=color, lw=0.8, ls="--")
    ax.set_xlabel("P-factor (age-residualized)")
    ax.set_ylabel("MGD (age-residualized)")
    ax.legend(fontsize=5, markerscale=0.5, loc="upper right")
    _panel_label(ax, "b")

    # ── Panel c: MGD vs internalizing (age-residualized) ──
    ax = fig.add_subplot(gs[0, 2])
    mgd_vals = mgd_df["mgd_z_combined"].values
    intl_vals = mgd_df["internalizing"].values
    age_vals = mgd_df["age"].values
    m = ~np.isnan(mgd_vals) & ~np.isnan(intl_vals)
    if m.sum() < 3:
        logger.warning("  Panel c: insufficient samples (n=%d), skipping scatter",
                       m.sum())
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=6)
        _panel_label(ax, "c")
    else:
        mgd_res = mgd_vals[m] - np.polyval(np.polyfit(age_vals[m], mgd_vals[m], 1), age_vals[m])
        intl_res = intl_vals[m] - np.polyval(np.polyfit(age_vals[m], intl_vals[m], 1), age_vals[m])
        ax.scatter(intl_res, mgd_res, s=4, alpha=0.3, c=P["violet"], edgecolors="none",
                   rasterized=True)
        if len(intl_res) >= 2:
            slope, intercept = np.polyfit(intl_res, mgd_res, 1)
            x_fit = np.linspace(intl_res.min(), intl_res.max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, color=P["red_strong"], lw=0.8, ls="--")
            r, p = pearsonr(intl_res, mgd_res)
            ax.text(0.05, 0.95, "r=%.3f\np=%.3f" % (r, p), transform=ax.transAxes,
                    fontsize=5, va="top", bbox=dict(boxstyle="round,pad=0.2",
                    facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set_xlabel("Internalizing (age-residualized)")
        ax.set_ylabel("MGD combined (age-residualized)")
        _panel_label(ax, "c")

    # ── Panel d: ROC curves (one per band) ──
    ax = fig.add_subplot(gs[1, 0])
    if len(roc_curve_df) > 0 and "band" in roc_curve_df.columns:
        # Plot a ROC curve for every band present in the CSV
        for band_name in roc_curve_df["band"].unique():
            sub = roc_curve_df[roc_curve_df["band"] == band_name]
            color = BAND_COLORS.get(band_name, P["neutral_mid"])
            ax.plot(sub["fpr"], sub["tpr"], color=color, lw=1.0,
                    label=band_name)
    ax.plot([0, 1], [0, 1], color=P["neutral_mid"], ls="--", lw=0.5)
    if len(roc_summary) > 0:
        best_row = roc_summary.loc[roc_summary["auc"].idxmax()]
        ax.text(0.6, 0.15, "AUC=%.3f\np=%.3f" % (best_row["auc"], best_row["p_perm"]),
                transform=ax.transAxes, fontsize=5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor="none"))
    else:
        ax.text(0.5, 0.5, "No ROC data", ha="center", va="center",
                transform=ax.transAxes, fontsize=6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=4, loc="upper left", ncol=2)
    _panel_label(ax, "d")

    # ── Panel e: Per-band AUC bar chart ──
    ax = fig.add_subplot(gs[1, 1])
    band_order = ["delta", "theta", "alpha", "beta", "gamma", "combined"]
    aucs, ci_los, ci_his, p_perms, colors = [], [], [], [], []
    for b in band_order:
        row = roc_summary[roc_summary["band"] == b]
        if len(row) > 0:
            aucs.append(row["auc"].values[0])
            ci_los.append(row["ci_lo"].values[0])
            ci_his.append(row["ci_hi"].values[0])
            p_perms.append(row["p_perm"].values[0])
        else:
            aucs.append(0.5); ci_los.append(0.5); ci_his.append(0.5); p_perms.append(1.0)
        colors.append(BAND_COLORS.get(b, P["neutral_mid"]))

    x_pos = np.arange(len(band_order))
    yerr = np.array([[aucs[i] - ci_los[i] for i in range(len(aucs))],
                     [ci_his[i] - aucs[i] for i in range(len(aucs))]])
    ax.bar(x_pos, aucs, yerr=yerr, color=colors, alpha=0.8,
           edgecolor="white", linewidth=0.3, width=0.6,
           error_kw=dict(elinewidth=0.5, capthick=0.5))
    ax.axhline(0.5, color=P["neutral_mid"], ls="--", lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_order, rotation=30, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 0.6)
    for i, p_val in enumerate(p_perms):
        if p_val < 0.05:
            ax.text(i, ci_his[i] + 0.005, "*", ha="center", fontsize=6)
    _panel_label(ax, "e")

    # ── Panel f: Age-stratified heatmap ──
    ax = fig.add_subplot(gs[1, 2])
    age_groups = ["5-8", "9-12", "13-16", "17-21"]
    cbcl_vars = ["p_factor", "attention", "internalizing", "externalizing"]
    mat = np.full((len(cbcl_vars), len(age_groups)), np.nan)
    for j, ag in enumerate(age_groups):
        for i, var in enumerate(cbcl_vars):
            row = strat_df[(strat_df["band"] == "alpha") &
                           (strat_df["age_group"] == ag) &
                           (strat_df["variable"] == var)]
            if len(row) > 0:
                mat[i, j] = row["r"].values[0]
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(age_groups)))
    ax.set_xticklabels(age_groups)
    ax.set_yticks(range(len(cbcl_vars)))
    ax.set_yticklabels(["P-factor", "Attention", "Internal.", "External."])
    ax.set_xlabel("Age group (years)")
    for i in range(len(cbcl_vars)):
        for j in range(len(age_groups)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, "%.2f" % mat[i, j], ha="center", va="center",
                        fontsize=5, color="white" if abs(mat[i, j]) > 0.2 else "black")
    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("r (MGD-CBCL)", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    _panel_label(ax, "f")

    _save(fig, "figure05_mgd_biomarker.png")


# ═════════════════════════════════════════════════════════════════════════
# F6: Normative Trajectory and Deviation Topography
# ═════════════════════════════════════════════════════════════════════════

def figure06():
    """Normative trajectory and deviation topography using SPD matrices."""
    logger.info("Figure 6: MGD Trajectory (Nature)")
    mgd_path = RESULT_DIR / "mgd_results.csv"
    if not mgd_path.exists():
        logger.warning("  mgd_results.csv not found, skipping")
        return

    mgd_df = pd.read_csv(mgd_path)

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # ── Load covmats for SPD visualization ──
    from step04_statistics import load_merged_features
    # pyriemann moved utils → geometry in 0.14; support both
    try:
        from pyriemann.geometry.mean import mean_riemann
    except ImportError:
        from pyriemann.utils.mean import mean_riemann
    data = load_merged_features(REST_TASK)
    if data is None:
        logger.warning("  RestingState features not found, skipping Figure 6")
        plt.close(fig)
        return
    covmats = data["covmats"]
    data_ages = data["ages"]
    bidx = BAND_ORDER.index("alpha")

    young_idx = np.where((data_ages >= 5) & (data_ages < 9))[0]
    old_idx = np.where((data_ages >= 13) & (data_ages < 21))[0]
    # Use Riemannian mean instead of Euclidean mean — the latter does not
    # preserve the SPD property, so logm() downstream may fail or yield
    # non-symmetric / complex results on degenerate inputs.
    if len(young_idx) >= 2 and len(old_idx) >= 2:
        young_cov = mean_riemann(covmats[young_idx, bidx])
        old_cov = mean_riemann(covmats[old_idx, bidx])
    else:
        logger.warning("  Insufficient subjects in young/old age groups, "
                       "falling back to Euclidean mean (non-SPD)")
        young_cov = np.mean(covmats[young_idx, bidx], axis=0)
        old_cov = np.mean(covmats[old_idx, bidx], axis=0)

    def to_corr(cov):
        d = np.sqrt(np.diag(cov))
        return cov / np.outer(d, d)

    young_corr = to_corr(young_cov)
    old_corr = to_corr(old_cov)
    sub = slice(0, 128, 4)
    vmax = max(np.percentile(np.abs(young_corr[sub, sub][np.triu_indices(32, k=1)]), 95),
               np.percentile(np.abs(old_corr[sub, sub][np.triu_indices(32, k=1)]), 95))

    # ── Panel d: MGD distribution by risk group (bottom-left) ──
    ax = fig.add_subplot(gs[1, 0])
    pf = mgd_df["p_factor"].values
    mgd_combined = mgd_df["mgd_z_combined"].values
    hi_thr = np.nanpercentile(pf, 75)
    lo_thr = np.nanpercentile(pf, 25)
    hi_mask = pf > hi_thr
    lo_mask = pf < lo_thr
    parts = ax.boxplot(
        [mgd_combined[lo_mask], mgd_combined[hi_mask]],
        labels=["Low risk", "High risk"], patch_artist=True,
        widths=0.5, showfliers=False,
        medianprops=dict(color="black", linewidth=0.7),
        whiskerprops=dict(linewidth=0.5),
        capprops=dict(linewidth=0.5))
    for patch, color in zip(parts["boxes"], [RISK_COLORS["low"], RISK_COLORS["high"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.5)
    ax.set_ylabel("MGD (combined, z)")
    ax.axhline(0, color=P["neutral_mid"], ls="--", lw=0.5)
    _panel_label(ax, "d")

    # ── Panel a: Normative SPD for young (alpha band) ──
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(young_corr[sub, sub], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_title("Age 5-8 (alpha)", fontsize=6, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    _panel_label(ax, "a")

    # ── Panel b: Normative SPD for old ──
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(old_corr[sub, sub], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_title("Age 13-21 (alpha)", fontsize=6, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Correlation", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    _panel_label(ax, "b")

    # ── Panel c: Deviation matrix (old - young, log-space) ──
    ax = fig.add_subplot(gs[0, 2])
    from scipy.linalg import logm
    # logm on near-singular SPD can yield tiny imaginary parts due to
    # numerical noise; truncate them.  If either covariance is non-SPD
    # (Euclidean-mean fallback) logm may raise — degrade gracefully.
    try:
        log_old = np.real_if_close(logm(old_cov), tol=1e6)
        log_young = np.real_if_close(logm(young_cov), tol=1e6)
        diff_log = log_old - log_young
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("  logm failed (%s); falling back to Euclidean diff", exc)
        diff_log = old_cov - young_cov
    diff_sub = diff_log[sub, sub]
    vmax_d = np.percentile(np.abs(diff_sub[np.triu_indices(32, k=1)]), 95)
    im = ax.imshow(diff_sub, cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d,
                   interpolation="nearest")
    ax.set_title("Deviation (log-space)", fontsize=6, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.08, fraction=0.038)
    cb.set_label("Log-difference", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    _panel_label(ax, "c")

    # ── Panel e: MGD by age group x risk group ──
    ax = fig.add_subplot(gs[1, 1])
    age_group_order = ["5-8", "9-12", "13-16", "17-21"]
    groups = [age_group(a) for a in mgd_df["age"].values]
    mgd_df["_ag"] = groups
    x_pos = np.arange(len(age_group_order))
    width = 0.35
    for i, (risk_label, risk_mask, color) in enumerate([
        ("Low risk", lo_mask, RISK_COLORS["low"]),
        ("High risk", hi_mask, RISK_COLORS["high"])
    ]):
        means, sems = [], []
        for ag in age_group_order:
            mask = (mgd_df["_ag"] == ag) & risk_mask & ~np.isnan(mgd_combined)
            vals = mgd_combined[mask]
            means.append(vals.mean() if len(vals) > 0 else 0)
            sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax.bar(x_pos + (i - 0.5) * width, means, width, yerr=sems,
               color=color, alpha=0.7, edgecolor="white", linewidth=0.3,
               capsize=2, error_kw=dict(elinewidth=0.5, capthick=0.5),
               label=risk_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(age_group_order)
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel("MGD (combined, z)")
    ax.legend(fontsize=5, loc="upper left")
    ax.axhline(0, color=P["neutral_mid"], ls="--", lw=0.5)
    _panel_label(ax, "e")

    # ── Panel f: Developmental trajectory by risk group ──
    ax = fig.add_subplot(gs[1, 2])
    for risk_label, risk_mask, color in [
        ("Low risk", lo_mask, RISK_COLORS["low"]),
        ("High risk", hi_mask, RISK_COLORS["high"])
    ]:
        mask = risk_mask & ~np.isnan(mgd_combined)
        if mask.sum() > 5:
            ax.scatter(mgd_df.loc[mask, "age"], mgd_df.loc[mask, "mgd_z_combined"],
                       s=3, alpha=0.2, c=color, edgecolors="none", rasterized=True)
            age_bins = np.arange(5, 21, 1)
            bin_means, bin_centers = [], []
            for bi in range(len(age_bins) - 1):
                bm = mask & (mgd_df["age"] >= age_bins[bi]) & (mgd_df["age"] < age_bins[bi + 1])
                if bm.sum() > 3:
                    bin_means.append(mgd_df.loc[bm, "mgd_z_combined"].mean())
                    bin_centers.append((age_bins[bi] + age_bins[bi + 1]) / 2)
            if bin_centers:
                ax.plot(bin_centers, bin_means, color=color, lw=1.2, label=risk_label)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("MGD (combined, z)")
    ax.legend(fontsize=5, loc="upper left")
    ax.axhline(0, color=P["neutral_mid"], ls="--", lw=0.5)
    _panel_label(ax, "f")

    _save(fig, "figure06_mgd_trajectory.png")


if __name__ == "__main__":
    figure05()
    figure06()
