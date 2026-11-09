#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  diagnose.sh — 工作站环境诊断脚本
#  用法: bash diagnose.sh
# ════════════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════"
echo "  HBN 工作站环境诊断"
echo "  时间: $(date)"
echo "═══════════════════════════════════════════════════"

echo ""
echo "── 1. 当前目录 ──"
pwd
echo "  脚本所在目录: $(cd "$(dirname "$0")" && pwd)"

echo ""
echo "── 2. Python 环境 ──"
which python
python --version 2>&1
python -c "import mne; print(f'  mne: {mne.__version__}')" 2>&1
python -c "import sklearn; print(f'  sklearn: {sklearn.__version__}')" 2>&1
python -c "import pyriemann; print(f'  pyriemann: {pyriemann.__version__}')" 2>&1

echo ""
echo "── 3. 目录结构 ──"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$CODE_DIR")"
echo "  CODE_DIR:      $CODE_DIR"
echo "  PROJECT_ROOT:  $PROJECT_ROOT"
echo "  results/:      $PROJECT_ROOT/results"
echo "  features/:     $PROJECT_ROOT/results/features"
echo "  preprocessed/: $PROJECT_ROOT/results/preprocessed"
echo "  metadata/:     $PROJECT_ROOT/results/metadata"

echo ""
echo "  实际存在:"
for d in "$PROJECT_ROOT/results" \
         "$PROJECT_ROOT/results/features" \
         "$PROJECT_ROOT/results/preprocessed" \
         "$PROJECT_ROOT/results/metadata" \
         "$PROJECT_ROOT/results/analysis" \
         "$PROJECT_ROOT/results/figures"; do
    if [ -d "$d" ]; then
        n_files=$(ls -1 "$d" 2>/dev/null | wc -l)
        echo "    [OK] $d ($n_files files)"
    else
        echo "    [MISSING] $d"
    fi
done

echo ""
echo "── 4. 特征文件 (.npz) ──"
FEATURES_DIR="$PROJECT_ROOT/results/features"
if [ -d "$FEATURES_DIR" ]; then
    npz_count=$(ls -1 "$FEATURES_DIR"/*.npz 2>/dev/null | wc -l)
    echo "  .npz 文件数: $npz_count"
    if [ "$npz_count" -gt 0 ]; then
        ls -lh "$FEATURES_DIR"/*.npz
    else
        echo "  [WARNING] 没有找到 .npz 文件!"
    fi
else
    echo "  [ERROR] features 目录不存在!"
fi

echo ""
echo "── 5. 预处理文件 (.fif) ──"
PREPROC_DIR="$PROJECT_ROOT/results/preprocessed"
if [ -d "$PREPROC_DIR" ]; then
    fif_count=$(ls -1 "$PREPROC_DIR"/*.fif 2>/dev/null | wc -l)
    echo "  .fif 文件数: $fif_count"
else
    echo "  [WARNING] preprocessed 目录不存在或为空"
fi

echo ""
echo "── 6. QC / metadata ──"
META_DIR="$PROJECT_ROOT/results/metadata"
if [ -d "$META_DIR" ]; then
    ls -lh "$META_DIR"/*.csv 2>/dev/null || echo "  没有 CSV 文件"
else
    echo "  [WARNING] metadata 目录不存在"
fi

echo ""
echo "── 7. config.py 路径解析 ──"
cd "$CODE_DIR"
python -c "
import config
print(f'  DATA_ROOT:    {config.DATA_ROOT}')
print(f'  RAW_BASE:     {config.RAW_BASE}')
print(f'  PROJECT_ROOT: {config.PROJECT_ROOT}')
print(f'  FEATURE_DIR:  {config.FEATURE_DIR}')
print(f'  PREPROC_DIR:  {config.PREPROC_DIR}')
print(f'  FEATURE_DIR exists: {config.FEATURE_DIR.exists()}')
print(f'  PREPROC_DIR exists: {config.PREPROC_DIR.exists()}')
" 2>&1

echo ""
echo "── 8. step00 下载的原始数据 ──"
RAW_BASE="$PROJECT_ROOT/data/raw"
if [ -d "$RAW_BASE" ]; then
    echo "  $RAW_BASE:"
    ls -d "$RAW_BASE"/*/ 2>/dev/null || echo "  没有子目录"
else
    echo "  $RAW_BASE 不存在"
    # 检查 config.py 中定义的路径
    python -c "
import config
import os
rb = str(config.RAW_BASE)
if os.path.exists(rb):
    print(f'  {rb}:')
    for d in os.listdir(rb):
        full = os.path.join(rb, d)
        if os.path.isdir(full):
            has_tsv = os.path.exists(os.path.join(full, 'participants.tsv'))
            n_subs = len([x for x in os.listdir(full) if x.startswith('sub-') and os.path.isdir(os.path.join(full, x))])
            print(f'    {d}: {n_subs} subjects, participants.tsv={has_tsv}')
else:
    print(f'  {rb} 不存在')
" 2>&1
fi

echo ""
echo "── 9. step02 --release 参数测试 ──"
python -c "
import sys
sys.argv = ['step02', '--help']
try:
    import step02_extract_features
    step02_extract_features.main()
except SystemExit:
    pass
except Exception as e:
    print(f'  ERROR: {e}')
" 2>&1

echo ""
echo "═══════════════════════════════════════════════════"
echo "  诊断完成"
echo "═══════════════════════════════════════════════════"
