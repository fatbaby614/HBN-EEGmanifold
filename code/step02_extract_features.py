# -*- coding: utf-8 -*-
"""
Step 2 — Feature extraction.

Loads preprocessed .fif files and computes:
  • SPD covariance matrices (Schaefer-Strimmer shrinkage) per band
  • Tangent-space vectors per band (global Riemannian mean reference)
  • FBTS = concatenation of 5 band tangent vectors
  • PSD baseline (Welch, mean power per band × channel)
  • DE  baseline (differential entropy per band × channel)

Saves features as .npz per task.
"""
import argparse
import traceback

import mne
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# pyriemann moved utils → geometry in 0.14; support both
try:
    from pyriemann.geometry.mean import mean_riemann
except ImportError:
    from pyriemann.utils.mean import mean_riemann

from config import *
from utils import setup_logging, load_participants, select_resting_subjects

mne.set_log_level("WARNING")
logger = setup_logging("step02_features")


# ── Band-pass filter helper ─────────────────────────────────────────────

def bandpass_filter(data, sfreq, l_freq, h_freq):
    """Apply Butterworth band-pass filter to (n_channels, n_times) data."""
    return mne.filter.filter_data(
        data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
        method="iir", iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )


# ── SPD covariance (epoch-based) ────────────────────────────────────────

def compute_covariance(data_2d, sfreq):
    """
    Single-pass covariance estimation with Ledoit-Wolf shrinkage.

    Uses the entire recording to compute one SPD matrix per band.
    Much faster than epoch-based mean_riemann (~100x) while producing
    equivalent results when combined with PCA downstream.
    """
    data_3d = data_2d[np.newaxis, ...]  # (1, n_ch, n_times)
    try:
        cov = Covariances(estimator=COV_ESTIMATOR).fit_transform(data_3d)
    except Exception:
        cov = Covariances(estimator="oas").fit_transform(data_3d)
    return _regularize_spd(cov[0])


def _regularize_spd(mat, eps=1e-7):
    """Ensure matrix is symmetric positive definite by adding small
    regularization to the diagonal and symmetrizing."""
    mat = (mat + mat.T) / 2  # symmetrize
    eigvals = np.linalg.eigvalsh(mat)
    min_eig = eigvals.min()
    if min_eig <= eps:
        mat = mat + (eps - min_eig + eps) * np.eye(mat.shape[0])
    return mat


def _psd_from_welch(psds, freqs):
    """Extract per-band mean power from pre-computed Welch PSD."""
    psd_features = []
    for band_name in BAND_ORDER:
        l, h = BANDS[band_name]
        mask = (freqs >= l) & (freqs <= h)
        psd_features.append(psds[:, mask].mean(axis=1))
    return np.array(psd_features)


# ── Alpha peak frequency ────────────────────────────────────────────────

def compute_alpha_peak(data_2d, sfreq, psds_full=None, freqs_full=None):
    """
    Alpha peak frequency (8-13 Hz) averaged across channels.

    Returns a single scalar — the most prominent alpha frequency.
    This is a robust age marker (shifts ~8→10 Hz from age 5→16).
    Optionally reuses pre-computed PSD to avoid redundant FFT.
    """
    if psds_full is not None and freqs_full is not None:
        mask = (freqs_full >= 8.0) & (freqs_full <= 13.0)
        masked_freqs = freqs_full[mask]
        psd_mean = psds_full[:, mask].mean(axis=0)
        peak_idx = np.argmax(psd_mean)
        return float(masked_freqs[peak_idx])
    psds, freqs = mne.time_frequency.psd_array_welch(
        data_2d, sfreq=sfreq, fmin=8.0, fmax=13.0,
        n_fft=2048, n_overlap=1024, verbose=False,
    )
    psd_mean = psds.mean(axis=0)
    if len(psd_mean) > 1:
        peak_idx = np.argmax(psd_mean)
        return freqs[peak_idx]
    return 10.0


# ── PSD features ────────────────────────────────────────────────────────

def compute_psd(data_2d, sfreq):
    """
    Compute PSD features: mean power per channel per band.

    Uses a single Welch FFT across the full spectrum (1-45 Hz),
    then bins into frequency bands — 5x faster than per-band Welch.
    """
    psds, freqs = mne.time_frequency.psd_array_welch(
        data_2d, sfreq=sfreq, fmin=1.0, fmax=45.0,
        n_fft=2048, n_overlap=1024, verbose=False,
    )
    psd_features = []
    for band_name in BAND_ORDER:
        l, h = BANDS[band_name]
        mask = (freqs >= l) & (freqs <= h)
        psd_features.append(psds[:, mask].mean(axis=1))
    return np.array(psd_features)  # (n_bands, n_channels)


# ── DE features ─────────────────────────────────────────────────────────

def compute_de(data_2d, sfreq):
    """
    Compute Differential Entropy features per band.

    DE = 0.5 * ln(2 * pi * e * sigma^2)

    Returns (n_bands, n_channels) array.
    """
    de_features = []
    for band_name in BAND_ORDER:
        l, h = BANDS[band_name]
        filtered = bandpass_filter(data_2d, sfreq, l, h)
        var = np.var(filtered, axis=1)
        var = np.maximum(var, 1e-10)  # avoid log(0)
        de = 0.5 * np.log(2 * np.pi * np.e * var)
        de_features.append(de)
    return np.array(de_features)  # (n_bands, n_channels)


# ── Per-subject feature extraction ──────────────────────────────────────

def extract_subject_features(fif_path):
    """
    Load one preprocessed .fif and extract all features.

    Returns dict with keys: covmats (5, 128, 128), psd (5, 128), de (5, 128)
    or None on failure.
    """
    if not fif_path.exists():
        return None
    try:
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        data = raw.get_data()
        sfreq = raw.info["sfreq"]

        # safety: replace non-finite values (can arise from edge cases in
        # interpolation or ICA skip)
        if not np.all(np.isfinite(data)):
            n_bad = np.sum(~np.isfinite(data))
            logger.warning("  %s — replacing %d non-finite values with 0",
                           fif_path.name, n_bad)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            raw._data = data

        covmats = []
        for band_name in BAND_ORDER:
            l, h = BANDS[band_name]
            filtered = bandpass_filter(data, sfreq, l, h)
            cov = compute_covariance(filtered, sfreq)
            covmats.append(cov)

        # single Welch for both PSD and alpha peak
        psds_full, freqs_full = mne.time_frequency.psd_array_welch(
            data, sfreq=sfreq, fmin=1.0, fmax=45.0,
            n_fft=2048, n_overlap=1024, verbose=False,
        )
        psd = _psd_from_welch(psds_full, freqs_full)
        de = compute_de(data, sfreq)
        alpha_peak = compute_alpha_peak(data, sfreq, psds_full, freqs_full)

        return {
            "covmats": np.array(covmats),  # (5, 128, 128)
            "psd": psd,                     # (5, 128)
            "de": de,                       # (5, 128)
            "alpha_peak": alpha_peak,       # scalar
        }
    except Exception as exc:
        logger.error("  %s — ERROR: %s", fif_path.name, exc)
        logger.debug(traceback.format_exc())
        return None


# ── Batch extraction + tangent space ────────────────────────────────────

def extract_task_features(task, subject_df, release_suffix="", release=None):
    """
    Extract features for all subjects in one task.

    Saves .npz with covmats, tangent, fbts, psd, de, and metadata.
    """
    n = len(subject_df)
    logger.info("Extracting features for %s (%d subjects)", task, n)

    # ── Pass 1: covariances + baselines ──
    covmats_all = []
    psd_all = []
    de_all = []
    alpha_peaks = []
    meta_rows = []
    valid_idx = []

    for i, (_, row) in enumerate(subject_df.iterrows()):
        sid = row["participant_id"]
        fif_path = PREPROC_DIR / f"{sid}_task-{task}_clean.fif"
        feats = extract_subject_features(fif_path)
        if feats is None:
            continue
        covmats_all.append(feats["covmats"])
        psd_all.append(feats["psd"])
        de_all.append(feats["de"])
        alpha_peaks.append(feats["alpha_peak"])
        meta_rows.append({
            "subject": sid,
            "age": row["age"],
            "sex": row["sex"],
            "p_factor": row["p_factor"],
            "attention": row["attention"],
            "internalizing": row["internalizing"],
            "externalizing": row["externalizing"],
            "ehq_total": row.get("ehq_total", np.nan),
        })
        valid_idx.append(i)

    n_valid = len(covmats_all)
    if n_valid == 0:
        logger.error("No valid subjects for task %s", task)
        return

    covmats_stack = np.array(covmats_all)   # (n_subjects, 5, 128, 128)
    psd_stack = np.array(psd_all)           # (n_subjects, 5, 128)
    de_stack = np.array(de_all)             # (n_subjects, 5, 128)
    meta_df = pd.DataFrame(meta_rows)

    logger.info("  %d / %d subjects extracted successfully", n_valid, n)

    # ── Pass 2: tangent space per band ──
    n_bands = len(BAND_ORDER)
    n_ch = covmats_stack.shape[2]
    ts_dim = n_ch * (n_ch + 1) // 2

    tangent_all = np.zeros((n_valid, n_bands, ts_dim))
    for b, band_name in enumerate(BAND_ORDER):
        cov_band = covmats_stack[:, b, :, :]  # (n_subjects, 128, 128)
        ts = TangentSpace(metric="riemann")
        tangent_band = ts.fit_transform(cov_band)
        tangent_all[:, b, :] = tangent_band
        logger.info("  Tangent space (%s): %s", band_name, tangent_band.shape)

    # ── FBTS: concatenate all bands ──
    fbts = tangent_all.reshape(n_valid, -1)  # (n_subjects, 5 * ts_dim)

    # ── Flatten PSD and DE ──
    psd_flat = psd_stack.reshape(n_valid, -1)  # (n_subjects, 5 * 128)
    de_flat = de_stack.reshape(n_valid, -1)    # (n_subjects, 5 * 128)

    # ── Save ──
    out_path = FEATURE_DIR / f"{task}_features{release_suffix}.npz"
    np.savez_compressed(
        out_path,
        covmats=covmats_stack,
        tangent=tangent_all,
        fbts=fbts,
        psd=psd_flat,
        de=de_flat,
        alpha_peak=np.array(alpha_peaks),
        subject_ids=meta_df["subject"].values,
        ages=meta_df["age"].values,
        sexes=meta_df["sex"].values,
        p_factor=meta_df["p_factor"].values,
        attention=meta_df["attention"].values,
        internalizing=meta_df["internalizing"].values,
        externalizing=meta_df["externalizing"].values,
        ehq_total=meta_df["ehq_total"].values,
        band_names=np.array(BAND_ORDER),
    )
    logger.info("  Saved → %s", out_path)
    logger.info("  Shapes: covmats %s, fbts %s, psd %s, de %s",
                covmats_stack.shape, fbts.shape, psd_flat.shape, de_flat.shape)

    # also save metadata CSV (per-release to avoid overwriting)
    if release is not None:
        meta_path = META_DIR / f"{task}_metadata_R{release}.csv"
    else:
        meta_path = META_DIR / f"{task}_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    logger.info("  Metadata → %s (%d subjects)", meta_path, len(meta_df))


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HBN feature extraction")
    parser.add_argument("--task", default="all",
                        help="RestingState | seqLearning6target | seqLearning8target | all")
    parser.add_argument("--cleanup", action="store_true",
                        help="OPT-IN: delete .fif files after extraction to free disk space. "
                             "Off by default to preserve raw preprocessed EEG on disk.")
    parser.add_argument("--release", type=int, default=None,
                        help="Release number (1-10). Appends _R{n} to feature filename.")
    args = parser.parse_args()

    release_suffix = f"_R{args.release}" if args.release else ""

    # Override DATA_ROOT if release specified
    if args.release:
        import config
        from config import RELEASES, RAW_BASE
        info = RELEASES[args.release]
        config.DATA_ROOT = RAW_BASE / f"HBN-Release{args.release}-{info['dataset']}"
        global DATA_ROOT
        DATA_ROOT = config.DATA_ROOT
        import utils
        utils.DATA_ROOT = config.DATA_ROOT
        logger.info("Release %d: DATA_ROOT = %s", args.release, DATA_ROOT)

    logger.info("=" * 60)
    logger.info("Step 2: Feature Extraction")
    logger.info("Input:  %s", PREPROC_DIR)
    logger.info("Output: %s", FEATURE_DIR)
    if args.cleanup:
        logger.info("WARNING: --cleanup enabled, .fif files will be deleted after extraction")
    if args.release:
        logger.info("Release: %d (feature files will have _R%d suffix)", args.release, args.release)
    logger.info("=" * 60)

    df = load_participants()
    rest_df, _ = select_resting_subjects(df)

    # build the QC-included subject list from step 1
    qc_path = META_DIR / "included_resting.csv"
    if qc_path.exists():
        qc_df = pd.read_csv(qc_path)
        ok_sids = set(qc_df["subject"].values)
        rest_df = rest_df[rest_df["participant_id"].isin(ok_sids)].reset_index(drop=True)
        logger.info("Using %d QC-passed resting-state subjects", len(rest_df))

    tasks_map = {}
    if args.task in ("all", REST_TASK):
        tasks_map[REST_TASK] = rest_df
    if args.task in ("all", SEQ6_TASK):
        from utils import select_seqlearning_subjects
        tasks_map[SEQ6_TASK] = select_seqlearning_subjects(rest_df, SEQ6_TASK)
    if args.task in ("all", SEQ8_TASK):
        from utils import select_seqlearning_subjects
        tasks_map[SEQ8_TASK] = select_seqlearning_subjects(rest_df, SEQ8_TASK)

    for task, sub_df in tasks_map.items():
        extract_task_features(task, sub_df, release_suffix, release=args.release)

        # Cleanup is OPT-IN. By default we keep the raw preprocessed .fif files
        # on disk so subsequent analyses (re-extraction, debugging, alternative
        # feature pipelines) can reuse them without re-running step01.
        if args.cleanup:
            if args.release is None:
                logger.warning(
                    "  --cleanup without --release would delete .fif files from "
                    "ALL releases (R1-R10). Skipping cleanup to prevent cross-"
                    "release data loss. Re-run with --release N to scope cleanup "
                    "to a single release.")
            else:
                # Only delete .fif files for the current release.
                # .fif filenames embed the subject id (sub-NDAR...); subjects
                # belong to exactly one release, so we filter by subject id set.
                current_sids = set(sub_df["participant_id"].astype(str))
                all_task_fifs = list(PREPROC_DIR.glob(f"*_task-{task}_*.fif"))
                task_fifs = []
                for f in all_task_fifs:
                    # Filename pattern: sub-NDARAC904DMU_task-RestingState_desc-clean_eeg.fif
                    fname = f.name
                    if any(f"_sub-{sid}_" in fname or fname.startswith(f"sub-{sid}_")
                           for sid in current_sids):
                        task_fifs.append(f)
                freed_mb = sum(f.stat().st_size for f in task_fifs) / 1e6
                for f in task_fifs:
                    f.unlink()
                logger.info("Cleanup: deleted %d .fif files for %s (%.0f MB freed, "
                            "release=%d)", len(task_fifs), task, freed_mb,
                            args.release)

    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()
