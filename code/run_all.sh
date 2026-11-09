#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  run_all.sh — HBN 全流程一键启动脚本
#
#  整合以下步骤为单一命令：
#    1. 检查各 Release 大小 (--check)
#    2. 下载全部 10 个 Release (R1-R10)
#    3. 逐个处理: 预处理 → 提特征 → 保留原始数据
#    4. 合并特征 → 建模 → 统计 → MGD → 出图
#
#  用法:
#    bash run_all.sh                 # 完整流程: 检查 → 下载 → 处理 → 分析
#    bash run_all.sh --skip-check    # 跳过大小检查, 直接下载+处理
#    bash run_all.sh --download-only # 仅下载, 不处理
#    bash run_all.sh --process-only  # 仅处理 (假设数据已下载)
#    bash run_all.sh --full          # 全部重新处理 (删除旧特征文件)
#
#  日志: results/logs/run_all_YYYYMMDD_HHMMSS.log
# ════════════════════════════════════════════════════════════════════════════

set -uo pipefail  # 单步失败不中断整体

# ── 路径配置 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/../results"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
RUN_LOG="$LOG_DIR/run_all_$(date '+%Y%m%d_%H%M%S').log"

# 日志函数
log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"
}

# ── 解析参数 ──────────────────────────────────────────────────────────────
MODE="full"
SKIP_CHECK=false
FULL_REPROCESS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-check)     SKIP_CHECK=true; shift ;;
        --download-only)  MODE="download"; shift ;;
        --process-only)   MODE="process"; shift ;;
        --full)           FULL_REPROCESS=true; shift ;;
        *)                echo "未知参数: $1"; echo "用法: bash run_all.sh [--skip-check|--download-only|--process-only|--full]"; exit 1 ;;
    esac
done

log "══════════════════════════════════════════════════════"
log "  HBN 全流程一键启动"
log "  开始时间: $START_TIME"
log "  日志文件: $RUN_LOG"
log "  模式: $MODE"
log "══════════════════════════════════════════════════════"

# ── Step 0: 环境检查 ──────────────────────────────────────────────────────
log ""
log "── Step 0: 环境检查 ──"

# AWS CLI
if ! command -v aws &>/dev/null; then
    log "  ERROR: AWS CLI 未安装。请运行: pip install awscli"
    exit 1
fi
log "  AWS CLI: $(aws --version 2>&1)"

# Python
if ! command -v python &>/dev/null; then
    log "  ERROR: Python 未找到"
    exit 1
fi
log "  Python: $(python --version 2>&1)"

# Python 依赖
log "  检查 Python 依赖..."
python -c "import mne, sklearn, pyriemann, scipy, numpy, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    log "  ERROR: 缺少 Python 依赖"
    log "  请运行: pip install mne scikit-learn pyriemann scipy numpy pandas pingouin"
    exit 1
fi
log "  Python 依赖: OK"

# 磁盘空间
log "  磁盘空间:"
df -h /mnt/data1 2>/dev/null | tee -a "$RUN_LOG" || df -h . | tee -a "$RUN_LOG"

# ════════════════════════════════════════════════════════════════════════════
#  阶段 1: 下载
# ════════════════════════════════════════════════════════════════════════════

if [ "$MODE" != "process" ]; then
    # ── Step 1a: 检查各 Release 大小 ──
    if [ "$SKIP_CHECK" = "false" ]; then
        log ""
        log "── Step 1a: 检查各 Release 大小 ──"
        bash download_all_releases.sh --check 2>&1 | tee -a "$RUN_LOG"
    fi

    # ── Step 1b: 下载全部 Release ──
    log ""
    log "── Step 1b: 下载全部 10 个 Release ──"
    set +o pipefail  # Temporarily disable pipefail so we can read PIPESTATUS
    bash download_all_releases.sh 2>&1 | tee -a "$RUN_LOG"
    DOWNLOAD_STATUS=${PIPESTATUS[0]}
    set -uo pipefail  # Re-enable pipeline failure detection

    if [ "$DOWNLOAD_STATUS" -ne 0 ]; then
        log ""
        log "  WARNING: 部分 Release 下载失败 (exit=$DOWNLOAD_STATUS)"
        log "  可以先处理已下载的 Release, 稍后用以下命令重试:"
        log "    bash download_all_releases.sh --release <N>"
        log "    bash run_all.sh --process-only"
    fi
fi

# 如果只是下载模式, 到此结束
if [ "$MODE" = "download" ]; then
    log ""
    log "══════════════════════════════════════════════════════"
    log "  下载完成。运行以下命令开始处理:"
    log "    bash run_all.sh --process-only"
    log "══════════════════════════════════════════════════════"
    exit 0
fi

# ════════════════════════════════════════════════════════════════════════════
#  阶段 2: 处理流水线
# ════════════════════════════════════════════════════════════════════════════

log ""
log "══════════════════════════════════════════════════════"
log "  开始处理流水线"
log "══════════════════════════════════════════════════════"

# 传递 --full 参数给 run_pipeline.sh
PIPELINE_ARGS=""
if [ "$FULL_REPROCESS" = "true" ]; then
    PIPELINE_ARGS="--full"
fi

bash run_pipeline.sh $PIPELINE_ARGS 2>&1 | tee -a "$RUN_LOG"

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
log "  全流程完成!"
log "  开始: $START_TIME"
log "  结束: $END_TIME"
log "  总耗时: $ELAPSED"
log ""
log "  结果文件:"
log "    分析 CSV: $RESULTS_DIR/analysis/"
log "    图表:     $RESULTS_DIR/figures/"
log "    特征:     $RESULTS_DIR/features/"
log "    日志:     $LOG_DIR/"
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
