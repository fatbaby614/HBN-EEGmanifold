# -*- coding: utf-8 -*-
"""
Configuration for HBN Brain Age Prediction Pipeline.

All paths, EEG parameters, frequency bands, and model hyperparameters
are defined here.  Modify DATA_ROOT if the dataset moves.
"""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
# Dataset root — auto-detect platform or override via HBN_DATA_ROOT env var
# In multi-release mode, DATA_ROOT is overridden per-release by step01/step02.
# The default here points to RAW_BASE for single-release fallback.
_env_root = os.environ.get("HBN_DATA_ROOT")
if _env_root:
    DATA_ROOT = Path(_env_root)
elif os.name == "nt":  # Windows
    DATA_ROOT = Path(r"E:\datasets\HBN")
else:                   # Linux / WSL
    DATA_ROOT = Path("/mnt/data1/home/tanhuang/datasets/HBN")

# ── Multi-Release Configuration ─────────────────────────────────────────
# HBN EEG has 12 releases on S3 (R1-R11 + Release NC).
# R1-R11 are publicly available on OpenNeuro (ds005505-ds005516, skipping ds005513).
# Release NC (458 subjects) is S3-only with non-commercial license, not included.
# Streaming pipeline: download → preprocess → extract features → cleanup
RELEASES = {
    1:  {"dataset": "ds005505", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1",  "n_subjects": 136},
    2:  {"dataset": "ds005506", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R2",  "n_subjects": 150},
    3:  {"dataset": "ds005507", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R3",  "n_subjects": 184},
    4:  {"dataset": "ds005508", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R4",  "n_subjects": 324},
    5:  {"dataset": "ds005509", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R5",  "n_subjects": 330},
    6:  {"dataset": "ds005510", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R6",  "n_subjects": 135},
    7:  {"dataset": "ds005511", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R7",  "n_subjects": 381},
    8:  {"dataset": "ds005512", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R8",  "n_subjects": 257},
    9:  {"dataset": "ds005514", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R9",  "n_subjects": 295},
    10: {"dataset": "ds005515", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R10", "n_subjects": 533},
    11: {"dataset": "ds005516", "s3": "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R11", "n_subjects": 430},
}

# Releases to process — R1-R10 are downloaded and available.
# R11 (ds005516) is publicly available on OpenNeuro but NOT yet downloaded.
# Release NC (458 subjects) is S3-only with non-commercial license, not included.
RELEASES_TO_PROCESS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Raw data root for a specific release (platform-dependent)
if os.name == "nt":
    RAW_BASE = Path(r"E:\datasets\HBN")
else:
    RAW_BASE = Path("/mnt/data1/home/tanhuang/datasets/HBN")

# Project root = parent of the code/ directory
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

# Output directories — all under results/ at project root
RESULTS_DIR = PROJECT_ROOT / "results"
PREPROC_DIR = RESULTS_DIR / "preprocessed"
FEATURE_DIR = RESULTS_DIR / "features"
RESULT_DIR  = RESULTS_DIR / "analysis"
META_DIR    = RESULTS_DIR / "metadata"
FIGURE_DIR  = RESULTS_DIR / "figures"

for d in (RESULTS_DIR, PREPROC_DIR, FEATURE_DIR, RESULT_DIR, META_DIR, FIGURE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── EEG parameters ───────────────────────────────────────────────────────
SFREQ = 500                 # Hz
LINE_FREQ = 60              # Hz
N_CHANNELS = 128
REFERENCE = "Cz"            # original recording reference
REREference = "average"     # re-reference target
MONTAGE = "GSN-HydroCel-128"

# EGI GSN 200 v2.1 channel names
CHANNEL_NAMES = [f"E{i}" for i in range(1, 129)]

# EOG proxy channels for ICA artifact detection.
# Matches paper specification (Langer 2017, p.12):
#   "EOG electrodes were placed on the participant's forehead, outer and
#    inner canthi (#'s 8, 14, 17, 21, 25, 125, 126, 127, and 128)"
# E17 = inner canthus (blink detection); E125-E128 = lower peri-ocular
EOG_PROXY_CHANNELS = ["E8", "E14", "E17", "E21", "E25",
                      "E125", "E126", "E127", "E128"]

# ── Preprocessing ────────────────────────────────────────────────────────
BANDPASS_LOW = 0.5          # Hz
BANDPASS_HIGH = 45.0        # Hz
NOTCH_FREQ = 60.0           # Hz

ICA_N_COMPONENTS = 30
ICA_METHOD = "fastica"
ICA_RANDOM_STATE = 42
ICA_MAX_VARIANCE_LOSS = 0.40   # exclude subject if > 40 % variance lost

BAD_CH_STD_FACTOR = 5.0       # channel std > 5 × median → bad
BAD_CH_CORR_THRESHOLD = 0.05  # corr with global mean < 0.05 → bad
MAX_BAD_CH_RATIO = 0.30       # > 30 % bad channels → exclude subject

MIN_REST_SAMPLES = 80_000      # ≈ 160 s at 500 Hz (4 × 40-s eyes-closed blocks)
REST_EYES_CLOSED_DURATION = 40.0   # seconds per block

# ── Frequency bands ──────────────────────────────────────────────────────
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
PRIORITY_BANDS = ["theta", "alpha"]

# ── Covariance / Tangent space ───────────────────────────────────────────
COV_ESTIMATOR = "lwf"          # Ledoit-Wolf shrinkage (more stable than sch)
TANGENT_REF = "global"         # use global mean_riemann as reference

# Epoch-based covariance estimation
EPOCH_LENGTH = 2.0             # seconds per epoch for cov estimation
EPOCH_OVERLAP = 0.5            # overlap fraction (0 = no overlap)
MAX_EPOCH_REJECT_FRAC = 0.2    # reject epochs with variance > 5× median

# ── Feature selection ───────────────────────────────────────────────────
FS_K = 500                     # SelectKBest top-K features
FS_K_SEARCH = [50, 100, 200, 500, 1000]

# PCA dimensionality reduction (applied after SelectKBest)
# Using float = fraction of variance to preserve (auto-adapts to sample size)
PCA_N_COMPONENTS = 0.95         # default: keep 95% variance
PCA_N_SEARCH = [0.90, 0.95, 0.99]  # grid search: variance fractions

# ── Model hyperparameters ───────────────────────────────────────────────
SVR_PARAMS = {
    "kernel": ["rbf"],
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01],
    "epsilon": [0.01, 0.1, 0.5],
}
RIDGE_PARAMS = {"alpha": [0.1, 1, 10, 100]}  # removed 0.01 (causes divergence with PCA)

# Ridge with PCA (simpler, more stable for small samples)
RIDGE_PCA_GRID = {
    "fs__k": FS_K_SEARCH,
    "pca__n_components": PCA_N_SEARCH,
    "ridge__alpha": RIDGE_PARAMS["alpha"],
}

CV_FOLDS = 5
INNER_CV_FOLDS = 3
RANDOM_STATE = 42

# ── Statistical thresholds ──────────────────────────────────────────────
ALPHA = 0.05
N_PERMUTATIONS = 5_000      # log-Euclidean permutation is fast; 5000 gives p ≥ 0.0002
RISK_PERCENTILE_HIGH = 75
RISK_PERCENTILE_LOW = 25

# ── Manifold Geometry Deviation (MGD) ───────────────────────────────────
MGD_BANDWIDTH = 1.5          # Gaussian kernel bandwidth (years) for normative trajectory
MGD_N_AGE_BINS = 12          # Number of age bins for z-score normalization
MGD_N_PERMUTATIONS = 5_000   # Permutations for ROC significance test
MGD_N_BOOTSTRAP = 2_000      # Bootstrap iterations for AUC confidence intervals

# ── Tasks ───────────────────────────────────────────────────────────────
REST_TASK = "RestingState"
SEQ6_TASK = "seqLearning6target"
SEQ8_TASK = "seqLearning8target"

# ── Age groups ──────────────────────────────────────────────────────────
# Age groups defined as HALF-OPEN intervals [lo, hi): the lower bound
# is inclusive, the upper bound is exclusive.  The label uses the
# inclusive lower bound and the *last* year in the group.  So the
# "17-21" group covers ages 17..21.999.  Last group's upper bound is
# 22 (i.e. exclusive) so 22-year-olds fall through to "other".
AGE_GROUPS = {
    "5-8":   (5, 9),
    "9-12":  (9, 13),
    "13-16": (13, 17),
    "17-21": (17, 22),
}
