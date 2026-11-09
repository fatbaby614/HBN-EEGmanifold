#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  download_all_releases.sh
#  一键下载 HBN EEG 全部 10 个已下载的 Release (R1-R10) 到 Linux 工作站
#
#  用法:
#    bash download_all_releases.sh              # 下载全部 R1-R10
#    bash download_all_releases.sh --check      # 仅查看各 Release 大小
#    bash download_all_releases.sh --release 4  # 只下载 R4
#
#  目标路径: /mnt/data1/home/tanhuang/datasets/HBN/
#  数据来源: AWS S3 (fcp-indi 公开桶, 无需认证)
#  要求:     aws CLI 已安装
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── 配置 ────────────────────────────────────────────────────────────────────
TARGET_DIR="/mnt/data1/home/tanhuang/datasets/HBN"
LOG_FILE="${TARGET_DIR}/download_log_$(date +%Y%m%d_%H%M%S).txt"

# R1-R10: 全部 10 个已下载的 Release (OpenNeuro ds005505-ds005515)
# R11 (ds005516) 尚未下载, 如需添加请取消注释最后一行
# Release NC (458 subjects) 仅在 S3 上, 非商业许可, 不下载
RELEASES=(
    "1|ds005505|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1|136"
    "2|ds005506|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R2|150"
    "3|ds005507|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R3|184"
    "4|ds005508|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R4|324"
    "5|ds005509|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R5|330"
    "6|ds005510|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R6|135"
    "7|ds005511|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R7|381"
    "8|ds005512|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R8|257"
    "9|ds005514|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R9|295"
    "10|ds005515|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R10|533"
    # "11|ds005516|s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R11|430"
)

# ── 工具函数 ─────────────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

check_aws() {
    if ! command -v aws &>/dev/null; then
        echo "ERROR: aws CLI 未安装"
        echo "  安装方法: pip install awscli  或  sudo apt install awscli"
        exit 1
    fi
    log "AWS CLI: $(aws --version 2>&1)"
}

check_disk() {
    local avail_gb
    avail_gb=$(df -BG /mnt/data1 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ -z "$avail_gb" ]; then
        # fallback: use target dir's mount point
        avail_gb=$(df -BG "$TARGET_DIR" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
    fi
    log "磁盘可用空间: ${avail_gb} GB"
    if [ "${avail_gb:-0}" -lt 500 ]; then
        log "WARNING: 可用空间 < 500GB, HBN 全部 Release 约需 400-600 GB"
        read -p "继续下载? (y/N) " -r
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
    fi
}

# ── 解析参数 ─────────────────────────────────────────────────────────────────
MODE="download"
SPECIFIC_RELEASE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)    MODE="check"; shift ;;
        --release)  SPECIFIC_RELEASE="$2"; shift 2 ;;
        *)          echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 主逻辑 ───────────────────────────────────────────────────────────────────
mkdir -p "$TARGET_DIR"

log "═══════════════════════════════════════════════════════════════"
log "  HBN EEG Release 下载脚本"
log "  模式: $MODE"
log "  目标: $TARGET_DIR"
log "═══════════════════════════════════════════════════════════════"

check_aws
if [ "$MODE" = "download" ]; then
    check_disk
fi

SUCCESS=0
FAILED=0
SKIPPED=0

for entry in "${RELEASES[@]}"; do
    IFS='|' read -r num dataset s3_uri n_subj <<< "$entry"

    # 如果指定了 --release N, 只处理该 Release
    if [ -n "$SPECIFIC_RELEASE" ] && [ "$num" != "$SPECIFIC_RELEASE" ]; then
        continue
    fi

    local_path="${TARGET_DIR}/HBN-Release${num}-${dataset}"

    log ""
    log "── Release ${num} (${dataset}) ──"
    log "  S3:       ${s3_uri}"
    log "  Local:    ${local_path}"
    log "  期望受试者: ${n_subj}"

    if [ "$MODE" = "check" ]; then
        # 仅查看大小
        log "  正在查询 S3..."
        result=$(aws s3 ls --no-sign-request --summarize --human-readable \
                 "${s3_uri}/" 2>&1)
        size_line=$(echo "$result" | grep "Total Size" || echo "  (无法获取大小)")
        count_line=$(echo "$result" | grep "Total Objects" || echo "  (无法获取数量)")
        log "  ${count_line}"
        log "  ${size_line}"
        continue
    fi

    # 检查是否已下载
    if [ -d "$local_path" ] && [ -f "${local_path}/participants.tsv" ]; then
        actual_subj=$(find "$local_path" -maxdepth 1 -type d -name "sub-*" | wc -l)
        if [ "$actual_subj" -ge $((n_subj * 9 / 10)) ]; then
            log "  已下载 (${actual_subj} 受试者), 跳过"
            SKIPPED=$((SKIPPED + 1))
            continue
        else
            log "  受试者数偏少 (${actual_subj}/${n_subj}), 备份旧目录后重新下载..."
            mv "$local_path" "${local_path}.incomplete"
            log "  旧目录已备份至: ${local_path}.incomplete"
        fi
    fi

    # 下载
    mkdir -p "$local_path"
    log "  开始下载..."

    if aws s3 sync --no-sign-request "${s3_uri}/" "${local_path}/"; then
        # 验证
        if [ -f "${local_path}/participants.tsv" ]; then
            actual_subj=$(find "$local_path" -maxdepth 1 -type d -name "sub-*" | wc -l)
            log "  下载完成: ${actual_subj}/${n_subj} 受试者"
            SUCCESS=$((SUCCESS + 1))
        else
            # 尝试单独下载 participants.tsv
            log "  WARNING: participants.tsv 缺失, 尝试单独下载..."
            aws s3 cp --no-sign-request \
                "${s3_uri}/participants.tsv" \
                "${local_path}/participants.tsv" 2>/dev/null || true

            if [ -f "${local_path}/participants.tsv" ]; then
                actual_subj=$(find "$local_path" -maxdepth 1 -type d -name "sub-*" | wc -l)
                log "  修复完成: ${actual_subj} 受试者"
                SUCCESS=$((SUCCESS + 1))
            else
                log "  ERROR: participants.tsv 下载失败"
                FAILED=$((FAILED + 1))
            fi
        fi
    else
        log "  ERROR: 下载失败"
        FAILED=$((FAILED + 1))
    fi
done

# ── 汇总 ─────────────────────────────────────────────────────────────────────
log ""
log "═══════════════════════════════════════════════════════════════"
log "  下载汇总"
log "  成功: ${SUCCESS}  跳过: ${SKIPPED}  失败: ${FAILED}"
log "═══════════════════════════════════════════════════════════════"

if [ "$MODE" = "download" ] && [ "$FAILED" -gt 0 ]; then
    log ""
    log "失败的 Release 可以单独重试:"
    log "  bash download_all_releases.sh --release <N>"
    exit 1
fi

# 列出最终目录结构
log ""
log "── 目录结构 ──"
ls -lh "$TARGET_DIR"/HBN-Release*/ 2>/dev/null | head -30
du -sh "$TARGET_DIR"/HBN-Release* 2>/dev/null | tee -a "$LOG_FILE"
