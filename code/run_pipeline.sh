#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  run_pipeline.sh — HBN 全 Release 一键流式处理脚本
#
#  用法:  bash run_pipeline.sh              # 断点续跑（跳过已有特征文件的 Release）
#         bash run_pipeline.sh --full       # 全部重新处理（删除旧特征文件和预处理文件）
#
#  功能:
#    1. 检查 AWS CLI
#    2. 依次处理 Release 1-10 (全部已下载的 Release)
#       每个 Release: 下载 → 预处理 → 提特征 → 保留原始数据
#    3. 合并所有特征 → 建模 → 统计 → 出图
#    4. 输出最终报告
# ════════════════════════════════════════════════════════════════════════════

set -uo pipefail  # 不用 -e，单个 Release 失败不中断整体

# ── 解析参数 ──────────────────────────────────────────────────────────────
FULL_MODE=false
if [ "${1:-}" = "--full" ]; then
    FULL_MODE=true
fi

# ── 路径配置 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/../results"
FEATURES_DIR="$RESULTS_DIR/features"
PREPROC_DIR="$RESULTS_DIR/preprocessed"
META_DIR="$RESULTS_DIR/metadata"
LOG_DIR="$RESULTS_DIR/logs"

# 时间戳
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
RUN_LOG="$LOG_DIR/pipeline_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$LOG_DIR" "$FEATURES_DIR" "$PREPROC_DIR" "$META_DIR"

# 日志函数：同时输出到屏幕和文件
log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"
}

log "══════════════════════════════════════════════════════"
log "  HBN 全 Release 批处理流水线"
log "  开始时间: $START_TIME"
log "  日志文件: $RUN_LOG"
log "══════════════════════════════════════════════════════"

# ── Step 0: 检查环境 ──────────────────────────────────────────────────────
log ""
log "── Step 0: 环境检查 ──"

# AWS CLI
if command -v aws &>/dev/null; then
    AWS_VERSION=$(aws --version 2>&1)
    log "  AWS CLI: $AWS_VERSION"
else
    log "  ERROR: AWS CLI 未安装。请运行: pip install awscli"
    exit 1
fi

# Python
if command -v python &>/dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    log "  Python: $PYTHON_VERSION"
else
    log "  ERROR: Python 未找到"
    exit 1
fi

# 磁盘空间检查
log "  磁盘空间:"
df -h "$SCRIPT_DIR/.." | tee -a "$RUN_LOG"

# 关键 Python 依赖
log "  检查 Python 依赖..."
python -c "import mne, sklearn, pyriemann, scipy, numpy, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    log "  ERROR: 缺少 Python 依赖，请检查环境"
    exit 1
fi
log "  Python 依赖: OK"

# ── Step 1: 清理旧的单 Release 特征文件（如果存在） ─────────────────────
log ""
log "── Step 1: 检查已有特征文件 ──"

# 列出当前特征文件
log "  当前特征文件:"
ls -lh "$FEATURES_DIR"/*.npz 2>/dev/null | tee -a "$RUN_LOG"

# ── Step 1b: --full 模式清理旧结果 ─────────────────────────────────────
if [ "$FULL_MODE" = "true" ]; then
    log ""
    log "── Step 1b: --full 模式，清理旧特征和预处理文件 ──"
    rm -f "$FEATURES_DIR"/RestingState_features_R*.npz
    rm -f "$FEATURES_DIR"/RestingState_features.npz
    rm -f "$FEATURES_DIR"/seqLearning*_features_R*.npz
    rm -f "$PREPROC_DIR"/*.fif 2>/dev/null
    rm -f "$META_DIR"/qc_preprocessing.csv 2>/dev/null
    rm -f "$META_DIR"/included_resting.csv 2>/dev/null
    log "  已清理旧特征文件和预处理文件"
fi

# ── Step 2: 逐个处理 Release ──────────────────────────────────────────────
# 处理 R1-R10 (全部 10 个已下载的 Release)
# Pull the release list from config.py so the two files stay in sync.
# Fall back to the hard-coded R1-R10 list if the Python call fails (e.g.
# config.py is missing from cwd), so the pipeline does not silently skip
# every release.
RELEASES_TO_PROCESS=$(python -c "from config import RELEASES_TO_PROCESS; print(' '.join(map(str, RELEASES_TO_PROCESS)))" 2>/dev/null || echo "1 2 3 4 5 6 7 8 9 10")
log "  RELEASES_TO_PROCESS: $RELEASES_TO_PROCESS"

# Defensive: if Python failed silently (output empty), fall back again.
if [ -z "$RELEASES_TO_PROCESS" ]; then
    log "  WARNING: RELEASES_TO_PROCESS is empty; using fallback R1-R10"
    RELEASES_TO_PROCESS="1 2 3 4 5 6 7 8 9 10"
fi

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for R in $RELEASES_TO_PROCESS; do
    R_START=$(date '+%Y-%m-%d %H:%M:%S')
    log ""
    log "══════════════════════════════════════════════════════"
    log "  Release $R — 开始: $R_START"
    log "══════════════════════════════════════════════════════"

    # 检查该 Release 的特征文件是否已存在（断点续跑）
    # 同时校验特征文件内容是否为空，避免 step02 部分失败留下空 npz 被静默跳过
    R_FEATURE="$FEATURES_DIR/RestingState_features_R${R}.npz"
    if [ -f "$R_FEATURE" ]; then
        N_FEAT=$(python -c "import numpy as np; d=np.load('${R_FEATURE}', allow_pickle=True); print(int(d['ages'].size))" 2>/dev/null || echo "0")
        if [ "${N_FEAT:-0}" -gt 0 ]; then
            log "  特征文件已存在: $R_FEATURE (${N_FEAT} subjects)，跳过此 Release"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        else
            log "  WARNING: $R_FEATURE 存在但内容为空，删除并重新处理"
            rm -f "$R_FEATURE"
            # Also drop any seq-learning / partial files for this release
            rm -f "$FEATURES_DIR"/seqLearning*_features_R${R}.npz
        fi
    fi

    # ── 2a: 下载 ──
    log "  ── 下载 Release $R ──"
    python step00_download.py --release $R 2>&1 | tee -a "$RUN_LOG"
    DOWNLOAD_OK=${PIPESTATUS[0]}

    if [ $DOWNLOAD_OK -ne 0 ]; then
        log "  ERROR: Release $R 下载失败，跳过"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        # 默认保留原始目录 (用户可手动清理或重下)
        continue
    fi

    # ── 2b: 预处理 ──
    log "  ── 预处理 Release $R ──"
    python step01_preprocess.py --release $R 2>&1 | tee -a "$RUN_LOG"
    PREPROC_OK=${PIPESTATUS[0]}

    if [ $PREPROC_OK -ne 0 ]; then
        log "  ERROR: Release $R 预处理失败，跳过"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        # 默认保留原始目录 (用户可手动清理或重下)
        continue
    fi

    # ── 2c: 提取特征 ──
    log "  ── 提取特征 Release $R ──"
    python step02_extract_features.py --release $R 2>&1 | tee -a "$RUN_LOG"
    FEATURE_OK=${PIPESTATUS[0]}

    if [ $FEATURE_OK -ne 0 ]; then
        log "  ERROR: Release $R 特征提取失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        log "  Release $R 特征提取成功"
    fi

    # ── 2d: 保留预处理 .fif 文件以便后续复用 ──
    # 默认不清理 .fif; 如果需要释放空间，请手动运行:
    #   find "$PREPROC_DIR" -name "*.fif" -delete
    log "  ── 保留 Release $R 预处理 .fif 文件 (默认策略) ──"

    # ── 2e: 保留原始数据 (默认策略) ──
    # 默认保留 release 原始目录以便:
    #   - 重跑预处理/特征提取 (无需重新下载)
    #   - 调试和交叉验证
    #   - 其他特征流水线复用
    # 如果需要释放空间，请手动运行:
    #   python step00_download.py --release $R --delete
    log "  ── 保留 Release $R 原始数据 (默认策略) ──"

    # 显示当前磁盘状态
    R_END=$(date '+%Y-%m-%d %H:%M:%S')
    log "  Release $R 完成: $R_END"
    log "  磁盘空间:"
    df -h "$SCRIPT_DIR/.." | tee -a "$RUN_LOG"
    log "  特征文件累积:"
    du -sh "$FEATURES_DIR" 2>/dev/null | tee -a "$RUN_LOG"
    ls -lh "$FEATURES_DIR"/RestingState_features_R*.npz 2>/dev/null | tee -a "$RUN_LOG"
done

# ── Step 3: 合并所有特征并运行最终分析 ─────────────────────────────────────
log ""
log "══════════════════════════════════════════════════════"
log "  所有 Release 处理完成，开始最终分析"
log "  成功: $SUCCESS_COUNT  失败: $FAIL_COUNT  跳过: $SKIP_COUNT"
log "══════════════════════════════════════════════════════"
log ""
log "  特征文件汇总:"
ls -lh "$FEATURES_DIR"/RestingState_features_R*.npz 2>/dev/null | tee -a "$RUN_LOG"
log ""

# Step 3: 建模（合并所有 Release 的特征）
log "── Step 3: 脑龄预测模型 (合并所有 Release) ──"
python step03_brain_age_model.py --merge-releases 2>&1 | tee -a "$RUN_LOG"

# Step 4: 统计分析
log ""
log "── Step 4: 统计分析 ──"
python step04_statistics.py 2>&1 | tee -a "$RUN_LOG"

# Step 4b: 流形几何偏离度分析 (MGD)
log ""
log "── Step 4b: 流形几何偏离度分析 (MGD) ──"
python step04b_mgd_analysis.py 2>&1 | tee -a "$RUN_LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    log "  WARNING: step04b 返回非零退出码, 后续 step05 系列结果可能受影响"
fi

# Step 5: 出图
log ""
log "── Step 5: 可视化 (标准图表) ──"
python step05_figures.py 2>&1 | tee -a "$RUN_LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    log "  WARNING: step05_figures 返回非零退出码"
fi

# Step 5a: Nature 风格主图 (F1-F4, F7-F10)
log ""
log "── Step 5a: Nature 风格主图 (F1-F4, F7-F10) ──"
python step05_figures_nature.py 2>&1 | tee -a "$RUN_LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    log "  WARNING: step05_figures_nature 返回非零退出码"
fi

# Step 5b: MGD 图表 (F5, F6)
log ""
log "── Step 5b: MGD 图表 (F5, F6) ──"
python step05b_mgd_figures_nature.py 2>&1 | tee -a "$RUN_LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    log "  WARNING: step05b_mgd_figures_nature 返回非零退出码"
fi

# Step 5c: 补充材料图表 (S1-S3)
log ""
log "── Step 5c: 补充材料图表 ──"
python step05c_supplementary_figures.py 2>&1 | tee -a "$RUN_LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    log "  WARNING: step05c_supplementary_figures 返回非零退出码"
fi

# ── 最终报告 ──────────────────────────────────────────────────────────────
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$(python -c "
from datetime import datetime
s = datetime.strptime('$START_TIME', '%Y-%m-%d %H:%M:%S')
e = datetime.strptime('$END_TIME', '%Y-%m-%d %H:%M:%S')
h = (e - s).total_seconds() / 3600
print(f'{h:.1f} 小时 ({h/24:.1f} 天)')
")

log ""
log "══════════════════════════════════════════════════════"
log "  全部完成!"
log "  开始: $START_TIME"
log "  结束: $END_TIME"
log "  总耗时: $ELAPSED"
log "  成功: $SUCCESS_COUNT / 10 个 Release"
log ""
log "  结果文件:"
log "    分析 CSV: $RESULTS_DIR/analysis/"
log "    图表:     $RESULTS_DIR/figures/"
log "    特征:     $FEATURES_DIR/"
log "══════════════════════════════════════════════════════"

# 显示最终模型性能
log ""
log "  最终模型性能:"
if [ -f "$RESULTS_DIR/analysis/cv5_metrics.csv" ]; then
    python -c "
import pandas as pd
m = pd.read_csv('$RESULTS_DIR/analysis/cv5_metrics.csv')
m = m.sort_values('cor', ascending=False)
for _, r in m.head(5).iterrows():
    print(f'    {r[\"model\"]:20s}: COR={r[\"cor\"]:.3f}  RMSE={r[\"rmse\"]:.3f}')
" 2>/dev/null | tee -a "$RUN_LOG"
else
    log "  (跳过) cv5_metrics.csv 不存在, 可能是 step03 未成功"
fi
