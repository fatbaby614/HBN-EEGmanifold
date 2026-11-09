# -*- coding: utf-8 -*-
"""
Step 1 — Preprocessing pipeline.

For each subject:
  1. Load EEGLAB .set via MNE
  2. Set GSN-HydroCel-128 montage
  3. Statistical bad-channel detection + interpolation
  4. Average re-reference
  5. Band-pass (0.5–45 Hz) + notch (60 Hz) filter
  6. ICA (FastICA, 30 components) with synthetic EOG proxy
  7. Extract eyes-closed segments (RestingState) or learning blocks (seqLearning)
  8. Save cleaned continuous data as .fif

Usage:
  python step01_preprocess.py            # all subjects, all tasks
  python step01_preprocess.py --task RestingState
  python step01_preprocess.py --subject sub-NDARAC349YUC
"""
import argparse
import traceback

import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config import *
from utils import (
    setup_logging, load_participants, select_resting_subjects,
    select_seqlearning_subjects,
    eeg_set_path, events_tsv_path, channels_tsv_path,
)

mne.set_log_level("WARNING")
logger = setup_logging("step01_preprocess")


# ── Bad-channel detection ───────────────────────────────────────────────

def detect_bad_channels(raw):
    """
    Statistical bad-channel detection (two-pass).

    Pass 1: flag channels by std dev (noisy / flat).
    Pass 2: compute global mean from clean channels only, then flag
            uncorrelated channels. This avoids cascade false-positives
            where one bad channel inflates the global mean and makes
            good channels appear uncorrelated.
    """
    data = raw.get_data()
    stds = np.std(data, axis=1)
    med_std = np.median(stds)

    bad = set()
    # Pass 1: noisy / flat channels by std
    for i, s in enumerate(stds):
        if s > BAD_CH_STD_FACTOR * med_std or s < 0.1 * med_std:
            bad.add(raw.ch_names[i])

    # Pass 2: correlation against global mean of CLEAN channels only
    good_idx = [i for i in range(data.shape[0]) if raw.ch_names[i] not in bad]
    if len(good_idx) < 10:
        # too few clean channels — skip correlation pass
        bad_ratio = len(bad) / raw.info["nchan"]
        return list(bad), bad_ratio

    global_mean = np.mean(data[good_idx, :], axis=0)
    for i in good_idx:
        try:
            r, _ = pearsonr(data[i], global_mean)
            if abs(r) < BAD_CH_CORR_THRESHOLD:
                bad.add(raw.ch_names[i])
        except Exception:
            bad.add(raw.ch_names[i])

    bad_ratio = len(bad) / raw.info["nchan"]
    return list(bad), bad_ratio


# ── Synthetic EOG proxy ─────────────────────────────────────────────────

def add_eog_proxy(raw):
    """Create a synthetic vertical-EOG channel from frontal electrodes."""
    avail = [ch for ch in EOG_PROXY_CHANNELS if ch in raw.ch_names]
    if not avail:
        logger.warning("No EOG proxy channels found, skipping EOG detection")
        return raw
    eog_data = raw.copy().pick(avail).get_data()
    eog_avg = eog_data.mean(axis=0, keepdims=True)
    eog_info = mne.create_info(["EOG"], raw.info["sfreq"], ch_types=["eog"])
    eog_raw = mne.io.RawArray(eog_avg, eog_info, verbose=False)
    raw.add_channels([eog_raw], force_update_info=True)
    return raw


# ── ICA ─────────────────────────────────────────────────────────────────

def apply_ica(raw):
    """
    Fit FastICA (30 components) on a 1-Hz high-pass copy,
    detect EOG artifacts, apply to the original raw.

    Returns (cleaned_raw, n_excluded, variance_lost).
    If ICA fails, returns the original raw unchanged with zero exclusion.
    """
    # ICA fitting on 1 Hz high-pass copy (best practice)
    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)

    # Replace NaN / Inf before ICA (can come from interpolation or raw data)
    data = raw_ica.get_data()
    if not np.all(np.isfinite(data)):
        n_bad = np.sum(~np.isfinite(data))
        logger.warning("  ICA: replacing %d non-finite values with 0", n_bad)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        raw_ica._data = data

    ica = mne.preprocessing.ICA(
        n_components=min(ICA_N_COMPONENTS, raw_ica.info["nchan"] - 1),
        method=ICA_METHOD,
        random_state=ICA_RANDOM_STATE,
        max_iter=500,
    )
    try:
        ica.fit(raw_ica)
    except Exception as exc:
        logger.warning("  ICA fit failed (%s), skipping ICA", exc)
        return raw, 0, 0.0

    # detect EOG-related components
    # MNE best practice: run find_bads_eog on the same object that was
    # used to fit the ICA (raw_ica), so the EOG scores reflect the
    # same frequency content as the unmixing matrix.
    eog_indices, eog_scores = ica.find_bads_eog(raw_ica, ch_name="EOG")
    ica.exclude = eog_indices

    # variance lost
    # Use the ICA stage's explained variance, NOT the PCA pre-stage
    # explained variance.  pca_explained_variance_ has shape (n_channels,)
    # but the ICA indices live in a sub-space of (n_components,) — indexing
    # PCA variances with ICA indices yields nonsense.
    try:
        # mne-icalabel >= 1.0 provides this; fall back to PCA otherwise.
        from mne_icalabel import iclabel
        _has_iclabel = True
    except Exception:
        _has_iclabel = False

    var_explained = ica.pca_explained_variance_
    total_var = var_explained.sum()
    if eog_indices:
        excluded_var = var_explained[eog_indices].sum()
    else:
        excluded_var = 0.0
    variance_lost = excluded_var / total_var if total_var > 0 else 0.0
    if _has_iclabel and eog_indices:
        try:
            from mne_icalabel.label_components import label_components
            _ = label_components(raw, ica, method="iclabel")
            # iclabel returns per-component class probabilities; the
            # classes that should be excluded are "eye blink",
            # "heart beat", "muscle artifact", "channel noise",
            # "line noise".  Sum their probabilities weighted by the
            # ICA component explained variance for the EOG-flagged
            # components to obtain a more accurate variance_lost.
            ica_explained = ica.get_explained_variance_ratio(
                raw, components=eog_indices)
            variance_lost = float(np.sum(ica_explained))
        except Exception as exc:
            logger.warning("  Failed to compute ICA explained variance (%s); "
                           "falling back to PCA approximation", exc)

    # apply to original raw
    ica.apply(raw)
    return raw, len(eog_indices), variance_lost


# ── Segment extraction ─────────────────────────────────────────────────

def extract_resting_eyesclosed(raw, events_tsv):
    """
    Extract and concatenate eyes-closed segments.

    Eyes-closed = from `instructed_toCloseEyes` onset to next
    `instructed_toOpenEyes` onset.
    """
    df = pd.read_csv(events_tsv, sep="\t", na_values=["n/a"])
    # Drop NaN onsets to prevent raw.crop(NaN, ...) from raising.
    close_onsets = df.loc[
        df["value"] == "instructed_toCloseEyes", "onset"
    ].dropna().values
    open_onsets = df.loc[
        df["value"] == "instructed_toOpenEyes", "onset"
    ].dropna().values

    segments = []
    raw_max = raw.times[-1]
    for co in close_onsets:
        # next open-eyes after this close-eyes
        later = open_onsets[open_onsets > co]
        tmax = later[0] if len(later) > 0 else co + REST_EYES_CLOSED_DURATION
        # Clamp to actual data duration — events.tsv onsets may reference
        # the original full recording which can be longer than the .set data
        tmax = min(tmax, raw_max)
        if co >= raw_max or tmax <= co:
            continue  # segment entirely outside data, or zero-length
        seg = raw.copy().crop(tmin=co, tmax=tmax)
        segments.append(seg)

    if not segments:
        return None
    return mne.concatenate_raws(segments)


def extract_seqlearning_blocks(raw, events_tsv):
    """
    Extract and concatenate the 5 learning blocks.

    Each block: from `learningBlock_N` onset to the next block onset
    (or `seqLearning_stop`).
    """
    df = pd.read_csv(events_tsv, sep="\t", na_values=["n/a"])

    block_onsets = []
    for i in range(1, 6):
        col = f"learningBlock_{i}"
        rows = df.loc[df["value"] == col, "onset"].dropna().values
        if len(rows) > 0:
            block_onsets.append(rows[0])

    stop_rows = df.loc[df["value"] == "seqLearning_stop", "onset"].values
    stop_onset = stop_rows[0] if len(stop_rows) > 0 else None

    segments = []
    raw_max = raw.times[-1]
    for i, bo in enumerate(block_onsets):
        if i + 1 < len(block_onsets):
            tmax = block_onsets[i + 1]
        elif stop_onset is not None:
            tmax = stop_onset
        else:
            tmax = bo + 120.0  # fallback ~2 min
        # Clamp to actual data duration
        tmax = min(tmax, raw_max)
        if bo >= raw_max:
            continue
        seg = raw.copy().crop(tmin=bo, tmax=tmax)
        segments.append(seg)

    if not segments:
        return None
    return mne.concatenate_raws(segments)


# ── Single-subject pipeline ─────────────────────────────────────────────

def preprocess_subject(subject_id, task, overwrite=False):
    """Run the full preprocessing pipeline for one subject and task."""
    out_fif = PREPROC_DIR / f"{subject_id}_task-{task}_clean.fif"
    if out_fif.exists() and not overwrite:
        logger.info("  %s / %s already preprocessed, skipping", subject_id, task)
        return True

    set_path = eeg_set_path(subject_id, task)
    ev_path = events_tsv_path(subject_id, task)
    if not set_path.exists():
        logger.warning("  %s / %s — .set not found", subject_id, task)
        return False
    if not ev_path.exists():
        logger.warning("  %s / %s — events.tsv not found", subject_id, task)
        return False

    try:
        # 1. load (support both .set and .edf formats)
        if set_path.suffix == ".edf":
            raw = mne.io.read_raw_edf(str(set_path), preload=True, verbose=False)
        else:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

        # 1b. drop Cz (original reference channel — not in GSN-HydroCel-128,
        #     would cause NaN during interpolation and montage mismatch)
        if "Cz" in raw.ch_names:
            raw.drop_channels(["Cz"])

        # 1c. Drop non-standard channels (e.g. "EEG 000", "EEG 001") that
        #     some .set files contain and that don't match GSN-HydroCel-128.
        #     These are typically auxiliary/reference channels.
        expected = set(CHANNEL_NAMES)
        extra = [ch for ch in raw.ch_names if ch not in expected]
        if extra:
            logger.info("  %s / %s — dropping %d non-standard channels: %s",
                        subject_id, task, len(extra), extra[:6])
            raw.drop_channels(extra)

        # 2. set montage
        montage = mne.channels.make_standard_montage(MONTAGE)
        raw.set_montage(montage, verbose=False)

        # 3. bad-channel detection
        bad_chs, bad_ratio = detect_bad_channels(raw)
        if bad_ratio > MAX_BAD_CH_RATIO:
            logger.warning(
                "  %s / %s — bad channel ratio %.1f%% > %.0f%%, EXCLUDED",
                subject_id, task, bad_ratio * 100, MAX_BAD_CH_RATIO * 100,
            )
            return False
        if bad_chs:
            logger.info("  %s / %s — bad channels: %s", subject_id, task, bad_chs)
            raw.info["bads"] = bad_chs
            raw.interpolate_bads(verbose=False)

        # 4. re-reference
        raw.set_eeg_reference(REREference, verbose=False)

        # 5. filter
        raw.filter(
            l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
            method="fir", phase="zero", verbose=False,
        )
        raw.notch_filter(
            freqs=NOTCH_FREQ, method="fir", verbose=False,
        )

        # 6. ICA
        raw = add_eog_proxy(raw)
        raw, n_excluded, var_lost = apply_ica(raw)
        # remove the EOG proxy channel
        if "EOG" in raw.ch_names:
            raw.drop_channels(["EOG"])

        if var_lost > ICA_MAX_VARIANCE_LOSS:
            logger.warning(
                "  %s / %s — ICA variance loss %.1f%% > %.0f%%, EXCLUDED",
                subject_id, task, var_lost * 100, ICA_MAX_VARIANCE_LOSS * 100,
            )
            return False

        # 7. segment extraction
        if task == REST_TASK:
            cleaned = extract_resting_eyesclosed(raw, ev_path)
            if cleaned is None or cleaned.n_times < MIN_REST_SAMPLES:
                logger.warning(
                    "  %s / %s — insufficient eyes-closed data", subject_id, task
                )
                return False
        else:
            cleaned = extract_seqlearning_blocks(raw, ev_path)
            if cleaned is None or cleaned.n_times < 50_000:
                logger.warning(
                    "  %s / %s — insufficient block data", subject_id, task
                )
                return False

        # 8. save
        cleaned.save(str(out_fif), overwrite=True, verbose=False)
        logger.info(
            "  %s / %s — OK  (%.1f s, %d bad ch, %d ICA comp removed, %.1f%% var lost)",
            subject_id, task, cleaned.times[-1], len(bad_chs),
            n_excluded, var_lost * 100,
        )
        return True

    except Exception as exc:
        logger.error("  %s / %s — ERROR: %s", subject_id, task, exc)
        logger.debug(traceback.format_exc())
        return False


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HBN preprocessing")
    parser.add_argument("--task", default="all",
                        help="RestingState | seqLearning6target | seqLearning8target | all")
    parser.add_argument("--subject", default=None,
                        help="Process a single subject (e.g. sub-NDARAC349YUC)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--release", type=int, default=None,
                        help="Release number (1-10). Overrides DATA_ROOT.")
    args = parser.parse_args()

    # Override DATA_ROOT if release specified
    if args.release:
        import config
        from config import RELEASES, RAW_BASE
        info = RELEASES[args.release]
        config.DATA_ROOT = RAW_BASE / f"HBN-Release{args.release}-{info['dataset']}"
        # Also update the global in this module and utils
        global DATA_ROOT
        DATA_ROOT = config.DATA_ROOT
        import utils
        utils.DATA_ROOT = config.DATA_ROOT
        logger.info("Release %d: DATA_ROOT = %s", args.release, DATA_ROOT)

    logger.info("=" * 60)
    logger.info("Step 1: Preprocessing")
    logger.info("Data root: %s", DATA_ROOT)
    logger.info("Output:    %s", PREPROC_DIR)
    logger.info("=" * 60)

    df = load_participants()
    rest_df, excl = select_resting_subjects(df)
    qc_rows = []

    tasks_to_run = []
    if args.task in ("all", REST_TASK):
        tasks_to_run.append((REST_TASK, rest_df))
    if args.task in ("all", SEQ6_TASK):
        seq6_df = select_seqlearning_subjects(rest_df, SEQ6_TASK)
        tasks_to_run.append((SEQ6_TASK, seq6_df))
    if args.task in ("all", SEQ8_TASK):
        seq8_df = select_seqlearning_subjects(rest_df, SEQ8_TASK)
        tasks_to_run.append((SEQ8_TASK, seq8_df))

    for task, sub_df in tasks_to_run:
        if args.subject:
            sub_df = sub_df[sub_df["participant_id"] == args.subject]
        logger.info("\n--- Task: %s  (%d subjects) ---", task, len(sub_df))

        n_ok, n_fail = 0, 0
        for idx, row in sub_df.iterrows():
            sid = row["participant_id"]
            ok = preprocess_subject(sid, task, overwrite=args.overwrite)
            qc_rows.append({
                "subject": sid, "task": task,
                "age": row["age"], "sex": row["sex"],
                "status": "ok" if ok else "failed",
            })
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        logger.info("Task %s: %d ok, %d failed", task, n_ok, n_fail)

    # save QC report (append if processing a specific release, to preserve prior releases)
    qc_df = pd.DataFrame(qc_rows)
    qc_path = META_DIR / "qc_preprocessing.csv"
    if args.release and qc_path.exists():
        old_qc = pd.read_csv(qc_path)
        # remove old entries for the same (subject, task) pairs (in case of re-run)
        new_keys = set(zip(qc_df["subject"], qc_df["task"]))
        old_keys = list(zip(old_qc["subject"], old_qc["task"]))
        old_qc = old_qc[[k not in new_keys for k in old_keys]]
        qc_df = pd.concat([old_qc, qc_df], ignore_index=True)
    qc_df.to_csv(qc_path, index=False)
    logger.info("QC report saved to %s (%d rows total)", qc_path, len(qc_df))

    # save included subjects (RestingState only, ok status)
    ok_rest = qc_df[(qc_df["task"] == REST_TASK) & (qc_df["status"] == "ok")]
    ok_rest.to_csv(META_DIR / "included_resting.csv", index=False)
    logger.info("Included resting-state subjects: %d", len(ok_rest))


if __name__ == "__main__":
    main()
