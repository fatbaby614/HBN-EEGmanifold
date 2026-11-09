# -*- coding: utf-8 -*-
"""
Step 4c — MGD 与 CBCL 的偏相关（控制年龄 + 性别）

背景
----
`mgd_correlations.csv` 中的 `r_pearson` 列实际存放的是
`step04b_mgd_analysis.analyse_mgd_correlations()` 调用
`partial_corr(mgd, cbcl, covariates=[age, sex])` 的结果（列名有误导），
但该文件没有：
  1. 95% 置信区间；
  2. 明确的"偏相关"语义（列名 / 表注无法体现口径）；
  3. 与年龄分层结果（mgd_age_stratified.csv）的统一呈现。

本脚本从 `mgd_results.csv`（已落盘的 MGD z 分数 + CBCL 因子 + 年龄 + 性别）
重新计算偏相关，输出 `mgd_partial_corr.csv`：
  - r_partial / p_partial（控制年龄与性别）
  - r_age_only / p_age_only（仅控制年龄，用于核对 Figure 6c 图内标注口径）
  - r_raw / p_raw（零偏相关）
  - 95% CI（Fisher z）、FDR 校正

统计口径
--------
偏相关 = 残差法：以 [1, age, sex] 为设计矩阵分别回归 MGD 与 CBCL，
残差的相关即偏相关。p 值用 t 检验，df = n - k - 2（k = 协变量个数 = 2）。
95% CI 用 Fisher z：SE = 1 / sqrt(n - k - 3)（k = 2 时即 n - 5）。

用法
----
    python step04c_mgd_partial_corr.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

from config import BAND_ORDER, RESULT_DIR
from step04_statistics import partial_corr

BANDS = BAND_ORDER + ["combined"]
CBCL_VARS = ["p_factor", "attention", "internalizing", "externalizing"]


# ── 工具函数 ──────────────────────────────────────────────────────────────

def fisher_ci(r, n, n_covars=2, alpha=0.05):
    """偏相关的 Fisher z 置信区间。SE = 1 / sqrt(n - k - 3)。"""
    if n - n_covars - 3 <= 0 or not np.isfinite(r):
        return np.nan, np.nan
    r_c = max(min(r, 0.999999), -0.999999)
    z = 0.5 * np.log((1 + r_c) / (1 - r_c))
    se = 1.0 / np.sqrt(n - n_covars - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    lo, hi = z - z_crit * se, z + z_crit * se
    return (np.exp(2 * lo) - 1) / (np.exp(2 * lo) + 1), \
           (np.exp(2 * hi) - 1) / (np.exp(2 * hi) + 1)


def t_test_partial(r, n, n_covars=2):
    """偏相关 p 值：t = r * sqrt(df / (1 - r^2))，df = n - k - 2。"""
    df = n - n_covars - 2
    if df <= 0 or not np.isfinite(r) or abs(r) >= 1.0:
        return np.nan
    t_val = r * np.sqrt(df / (1 - r ** 2))
    return 2 * (1 - t_dist.cdf(abs(t_val), df=df))


def fdr_bh(pvals):
    """Benjamini–Hochberg FDR 校正（NaN 视为不参与校正）。"""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    res = np.empty(n)
    res[order] = adj
    out[ok] = res
    return out


def sig_label(p):
    if not np.isfinite(p):
        return "ns"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    src = RESULT_DIR / "mgd_results.csv"
    if not src.exists():
        raise SystemExit("missing %s — run step04b_mgd_analysis.py first" % src)

    df = pd.read_csv(src)
    ages = df["age"].to_numpy(dtype=float)
    # 性别编码与 step04b 一致：M = 1, F = 0
    sex_num = (df["sex"].astype(str).str.upper().str.startswith("M")).astype(float).to_numpy()

    rows = []
    for band in BANDS:
        col = "mgd_z_" + band
        if col not in df.columns:
            continue
        mgd = df[col].to_numpy(dtype=float)
        for var in CBCL_VARS:
            y = df[var].to_numpy(dtype=float)
            mask = np.isfinite(mgd) & np.isfinite(y) & np.isfinite(ages) & np.isfinite(sex_num)
            n = int(mask.sum())
            if n < 20:
                continue

            x_m = mgd[mask]
            y_m = y[mask]
            age_m = ages[mask]
            sex_m = sex_num[mask]

            # 偏相关：控制年龄 + 性别
            r_par, p_par = partial_corr(x_m, y_m, np.column_stack([age_m, sex_m]))
            # 仅控制年龄（对应 Figure 6c 图内标注口径）
            r_age, p_age = partial_corr(x_m, y_m, age_m.reshape(-1, 1))
            # 零偏相关（原始 Pearson）
            r_raw, p_raw = partial_corr(x_m, y_m, np.empty((n, 0)))

            ci_lo, ci_hi = fisher_ci(r_par, n, n_covars=2)

            rows.append({
                "band": band, "variable": var, "n": n,
                "r_partial": r_par, "p_partial": p_par,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "r_age_only": r_age, "p_age_only": p_age,
                "r_raw": r_raw, "p_raw": p_raw,
            })

    result = pd.DataFrame(rows)
    if result.empty:
        raise SystemExit("no rows computed")

    result["p_fdr"] = fdr_bh(result["p_partial"].values)
    result["sig"] = [sig_label(p) for p in result["p_fdr"]]
    result = result[[
        "band", "variable", "n",
        "r_partial", "p_partial", "ci_lo", "ci_hi", "p_fdr", "sig",
        "r_age_only", "p_age_only", "r_raw", "p_raw",
    ]]

    out = RESULT_DIR / "mgd_partial_corr.csv"
    result.to_csv(out, index=False)

    print("wrote %s  (%d rows)" % (out, len(result)))
    print()
    print("=== MGD ~ CBCL, partial correlation (controlling age + sex) ===")
    for _, r in result.iterrows():
        print("  %-9s ~ %-13s r=%+.3f  p=%.4f  CI=[%+.3f, %+.3f]  FDR=%.3f %s  (n=%d)"
              % (r["band"], r["variable"], r["r_partial"], r["p_partial"],
                 r["ci_lo"], r["ci_hi"], r["p_fdr"], r["sig"], r["n"]))

    # 与既有 mgd_correlations.csv 交叉核对
    old_path = RESULT_DIR / "mgd_correlations.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        merged = result.merge(old, on=["band", "variable"], how="inner", suffixes=("", "_old"))
        if len(merged):
            d_par = (merged["r_partial"] - merged["r_pearson"]).abs().max()
            d_p = (merged["p_partial"] - merged["p_pearson"]).abs().max()
            print()
            print("=== cross-check vs mgd_correlations.csv ('r_pearson' column) ===")
            print("  max |d r| = %.3e   max |d p| = %.3e  over %d rows"
                  % (d_par, d_p, len(merged)))
            print("  -> existing 'r_pearson' column IS the age+sex partial correlation"
                  if d_par < 1e-9 else "  -> MISMATCH: existing column differs")


if __name__ == "__main__":
    main()
