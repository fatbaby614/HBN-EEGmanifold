#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# run_release.sh — Streaming pipeline for processing HBN releases
#
# Strategy: download → preprocess → extract features
# Features (.npz), preprocessed (.fif), and raw data are ALL preserved.
# Use --cleanup flag to opt in to deletion of preprocessed .fif files
# after feature extraction. Raw release directories are NEVER auto-deleted.
#
# Usage:
#   ./run_release.sh 1                # process release 1 (keep all data)
#   ./run_release.sh 1 2 3            # process releases 1, 2, 3
#   ./run_release.sh all              # process all releases (R1-R10)
#   ./run_release.sh --cleanup 1      # process R1, also delete .fif after extraction
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# R1-R10 are downloaded and processed.
# R11 (ds005516) is publicly available on OpenNeuro but NOT yet downloaded;
# step00_download.py will refuse and fail loudly.  Run `./run_release.sh 11`
# only after downloading R11 separately.
ALL_RELEASES="1 2 3 4 5 6 7 8 9 10"

CLEANUP_FIF=false
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cleanup)
            CLEANUP_FIF=true
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "${#ARGS[@]}" -eq 0 ]; then
    echo "ERROR: No releases specified"
    echo "Usage: ./run_release.sh [--cleanup] <release... | all>"
    exit 1
fi

if [ "${ARGS[0]}" == "all" ]; then
    RELEASES=$ALL_RELEASES
else
    RELEASES="${ARGS[@]}"
fi

echo "========================================================"
echo "  HBN Streaming Pipeline"
echo "  Releases: $RELEASES"
echo "  Cleanup preprocessed .fif: $CLEANUP_FIF"
echo "  Raw data deletion: NEVER (manual: python step00_download.py --release <N> --delete)"
echo "  $(date)"
echo "========================================================"

for R in $RELEASES; do
    echo ""
    echo "========================================================"
    echo "  Release $R"
    echo "  $(date)"
    echo "========================================================"

    # Step 0: Download
    echo "--- Downloading Release $R ---"
    python step00_download.py --release $R
    if [ $? -ne 0 ]; then
        echo "ERROR: Download failed for Release $R, skipping"
        # Raw data kept on disk for inspection / retry
        continue
    fi

    # Step 1: Preprocess
    echo "--- Preprocessing Release $R ---"
    python step01_preprocess.py --release $R
    if [ $? -ne 0 ]; then
        echo "ERROR: Preprocessing failed for Release $R"
        # Raw + partial .fif kept on disk for debugging
        continue
    fi

    # Step 2: Extract features
    if [ "$CLEANUP_FIF" = "true" ]; then
        echo "--- Extracting features Release $R (with --cleanup of preprocessed .fif) ---"
        python step02_extract_features.py --release $R --cleanup
    else
        echo "--- Extracting features Release $R (preprocessed .fif preserved) ---"
        python step02_extract_features.py --release $R
    fi
    if [ $? -ne 0 ]; then
        echo "ERROR: Feature extraction failed for Release $R"
    fi

    # Raw data: PRESERVED (manual deletion only)
    echo "--- Raw data Release $R preserved on disk ---"

    echo "Release $R complete: $(date)"

    # Show disk usage
    echo "--- Disk usage ---"
    df -h /mnt/data1 2>/dev/null || df -h . || true
    du -sh ../results/features/ 2>/dev/null || true
done

echo ""
echo "========================================================"
echo "  All releases processed. Checking for feature files..."
echo "  $(date)"
echo "========================================================"

# Step 3-5c: Merge all features and run full analysis pipeline
# Only run if at least one .npz feature file was produced.
FEATURE_COUNT=$(find ../results/features/ -name "*.npz" 2>/dev/null | wc -l)
if [ "$FEATURE_COUNT" -eq 0 ]; then
    echo "ERROR: No .npz feature files found in ../results/features/"
    echo "       All releases failed. Skipping analysis pipeline."
    exit 1
fi
echo "Found $FEATURE_COUNT feature files. Running analysis pipeline..."

# Keep this in sync with run_pipeline.sh so /code/run_release.sh delivers the
# same artefacts as /code/run_all.sh --process-only.
python step03_brain_age_model.py --merge-releases
python step04_statistics.py
python step04b_mgd_analysis.py
python step05_figures.py
python step05_figures_nature.py
python step05b_mgd_figures_nature.py
python step05c_supplementary_figures.py

echo ""
echo "========================================================"
echo "  Pipeline complete: $(date)"
echo "========================================================"

# Show feature file summary
echo "--- Feature files ---"
ls -lh ../results/features/*.npz 2>/dev/null

echo "--- Analysis files ---"
ls -lh ../results/analysis/*.csv 2>/dev/null
