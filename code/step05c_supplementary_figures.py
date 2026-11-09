# -*- coding: utf-8 -*-
"""Regenerate Figure S1: Bandwidth Sensitivity Analysis with adjusted Y-axis ranges.

Panel A: Y-axis 0~0.02 (reduce blank space)
Panel B: Y-axis 0~0.06 (prevent legend from blocking curves)
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import *
from utils import setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging("step05c_supp")

# ── Nature style configuration ──
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
    "legend.fontsize": 5.5,
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

PAPER_DIR = PROJECT_ROOT / "paper" / "HumanBrainMapping"
PAPER_DIR.mkdir(parents=True, exist_ok=True)


def _panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="black",
            ha="left", va="bottom")


def _save(fig, name):
    stem = name.replace(".png", "").replace(".svg", "")
    for out_dir in [PAPER_DIR, FIGURE_DIR]:
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", stem)


def figureS1():
    """Bandwidth sensitivity analysis with adjusted Y-axis ranges.

    Note: depends on ``bandwidth_sensitivity.csv`` and
    ``bandwidth_cross_band.csv`` which must be produced by
    ``step04b_mgd_analysis.analyse_bandwidth_sensitivity``. If the
    required CSV files are missing (e.g. on a partial pipeline rerun
    that only ran the single-bandwidth analyses) we skip Figure S1
    gracefully so the rest of the pipeline can continue.
    """
    bw_csv = RESULT_DIR / "bandwidth_sensitivity.csv"
    bw_cb_csv = RESULT_DIR / "bandwidth_cross_band.csv"
    if not (bw_csv.exists() and bw_cb_csv.exists()):
        logger.warning(
            "figureS1 skipped: required CSVs are missing. "
            "Run step04b's bandwidth sensitivity analysis to generate "
            "%s and %s. Existing Figure S1 (if any) will be kept.",
            bw_csv.name, bw_cb_csv.name,
        )
        return

    df = pd.read_csv(bw_csv)
    df_cb = pd.read_csv(bw_cb_csv)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    # bandwidth values are 0.5..3.0 in 0.5 steps; label every point so the
    # sparse default ticks (1, 2, 3) are not misread as the grid
    BW_TICKS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # ── Panel A: Age Association ──
    ax = axes[0]
    ai = df[df["metric"] == "age_independence"]
    bandwidths = sorted(ai["bandwidth"].unique())
    for band in ["delta", "theta", "alpha", "beta", "gamma", "combined"]:
        sub = ai[ai["band"] == band].sort_values("bandwidth")
        ax.plot(sub["bandwidth"], sub["r"].abs(), marker="o", markersize=3,
                lw=1.0, color=BAND_COLORS.get(band, P["neutral_mid"]),
                label=band)
    ax.axhline(0.05, color=P["neutral_mid"], ls="--", lw=0.5)
    ax.set_xlabel("Bandwidth $h$ (years)")
    ax.set_ylabel("$|r(\\mathrm{MGD}, \\mathrm{age})|$")
    ax.set_title("Age Association", fontsize=7, pad=28)
    ax.set_ylim(0, 0.02)
    ax.set_xticks(BW_TICKS)
    ax.legend(fontsize=5, loc="upper right", ncol=1)
    _panel_label(ax, "a")

    # ── Panel B: MGD-CBCL Partial Correlations ──
    ax = axes[1]
    cb = df[df["metric"] == "cbcl_partial_corr"]
    # Show alpha and combined bands with CBCL variables
    cbcl_vars = ["p_factor", "attention", "internalizing", "externalizing"]
    band_var_colors = {
        ("alpha", "p_factor"): P["teal"],
        ("alpha", "attention"): P["blue_main"],
        ("alpha", "internalizing"): P["violet"],
        ("alpha", "externalizing"): P["red_strong"],
        ("combined", "p_factor"): P["neutral_dark"],
        ("combined", "attention"): "#D28E2C",
        ("combined", "internalizing"): P["neutral_mid"],
        ("combined", "externalizing"): P["neutral_black"],
    }
    for band in ["alpha", "combined"]:
        for var in cbcl_vars:
            sub = cb[(cb["band"] == band) & (cb["variable"] == var)].sort_values("bandwidth")
            if len(sub) == 0:
                continue
            color = band_var_colors.get((band, var), P["neutral_mid"])
            label = f"{band}/{var}"
            ax.plot(sub["bandwidth"], sub["r"].abs(), marker="o", markersize=2.5,
                    lw=0.8, color=color, label=label)
    ax.axhline(0.05, color=P["neutral_mid"], ls="--", lw=0.5)
    ax.set_xlabel("Bandwidth $h$ (years)")
    ax.set_ylabel("$|r(\\mathrm{MGD}, \\mathrm{CBCL})|$")
    ax.set_title("MGD-CBCL Partial Correlations", fontsize=7, pad=3)
    ax.set_ylim(0, 0.07)
    ax.set_xticks(BW_TICKS)
    ax.legend(fontsize=4.5, loc="upper right", ncol=2,
              columnspacing=0.6, handlelength=1.0, handletextpad=0.3,
              labelspacing=0.3, frameon=True, framealpha=0.85,
              edgecolor='none')
    _panel_label(ax, "b")

    # ── Panel C: Cross-Band Structure ──
    ax = axes[2]
    # Alpha-theta correlation
    at = df_cb[df_cb["band_pair"] == "theta-alpha"].sort_values("bandwidth")
    if len(at) > 0:
        ax.plot(at["bandwidth"], at["r"], marker="s", markersize=3,
                lw=1.0, color=P["blue_main"], label="Alpha-Theta $r$")

    # Beta specificity (mean |r| with other bands)
    beta_spec = df_cb[df_cb["band_pair"] == "beta-mean"].sort_values("bandwidth")
    if len(beta_spec) > 0:
        ax.plot(beta_spec["bandwidth"], beta_spec["r"], marker="D", markersize=3,
                lw=1.0, color="#D28E2C", label="Beta mean $|r|$ (specificity)")
    else:
        # Fallback: compute from cross-band data
        for bw in bandwidths:
            beta_rows = df_cb[(df_cb["bandwidth"] == bw) &
                              (df_cb["band_pair"].str.contains("beta"))]
            if len(beta_rows) > 0:
                ax.scatter([bw], [beta_rows["r"].abs().mean()], marker="D",
                           s=12, color="#D28E2C")
    ax.set_xlabel("Bandwidth $h$ (years)")
    ax.set_ylabel("Correlation")
    ax.set_title("Cross-Band Structure", fontsize=7, pad=3)
    ax.set_xticks(BW_TICKS)
    ax.legend(fontsize=5, loc="center right")
    _panel_label(ax, "c")

    fig.subplots_adjust(wspace=0.38)
    _save(fig, "figureS1_bandwidth_sensitivity.png")


if __name__ == "__main__":
    import traceback
    try:
        figureS1()
    except Exception:
        logger.error("figureS1 failed:\n%s", traceback.format_exc())
        # Re-raise so the shell pipeline notices the failure
        raise
