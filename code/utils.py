# -*- coding: utf-8 -*-
"""
Shared utilities: logging, participant loading, subject selection,
behavioural metric extraction from events.tsv.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_ROOT, REST_TASK, SEQ6_TASK, SEQ8_TASK,
    MIN_REST_SAMPLES, AGE_GROUPS,
)

# ── Logging ─────────────────────────────────────────────────────────────


def setup_logging(name="hbn_pipeline", log_file=None):
    """Configure root logger with console + optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Participant table ───────────────────────────────────────────────────

PHENO_COLS = [
    "participant_id", "sex", "age", "ehq_total",
    "p_factor", "attention", "internalizing", "externalizing",
    REST_TASK, SEQ6_TASK, SEQ8_TASK, "symbolSearch",
]


def load_participants():
    """Load participants.tsv and return a DataFrame."""
    tsv = DATA_ROOT / "participants.tsv"
    df = pd.read_csv(tsv, sep="\t")
    # keep only columns we use
    cols = [c for c in PHENO_COLS if c in df.columns]
    df = df[cols].copy()
    # numeric coercion
    for c in ["age", "ehq_total", "p_factor",
              "attention", "internalizing", "externalizing"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def select_resting_subjects(df):
    """
    Apply inclusion / exclusion criteria for resting-state analysis.

    Returns the filtered DataFrame and a dict of exclusion counts.
    """
    n0 = len(df)
    excl = {}

    if REST_TASK not in df.columns:
        # R11 participants.tsv lacks task availability columns.
        # Infer RestingState availability by checking for .set files on disk.
        logging.warning("'%s' column not found in participants.tsv — checking files on disk", REST_TASK)
        available_subjects = []
        for _, row in df.iterrows():
            subj = row["participant_id"]
            eeg_dir = DATA_ROOT / subj / "eeg"
            if eeg_dir.is_dir():
                has_set = any(f.suffix == ".set" and "restingstate" in f.name.lower()
                              for f in eeg_dir.iterdir())
                if has_set:
                    available_subjects.append(subj)
        df[REST_TASK] = df["participant_id"].apply(
            lambda s: "available" if s in available_subjects else "unavailable"
        )
        logging.info("Inferred RestingState availability from files: %d/%d available",
                     len(available_subjects), n0)

    # Case-insensitive match: paper uses "Available" but data may use lowercase
    mask = df[REST_TASK].astype(str).str.lower() == "available"
    excl["resting_unavailable"] = int((~mask).sum())
    df1 = df[mask]

    pheno_cols = ["p_factor", "attention", "internalizing", "externalizing"]
    mask2 = df1[pheno_cols].notna().all(axis=1)
    excl["missing_pheno"] = int((~mask2).sum())
    df2 = df1[mask2]

    mask3 = (df2["age"] >= 5) & (df2["age"] <= 21)
    excl["age_out_of_range"] = int((~mask3).sum())
    df3 = df2[mask3]

    excl["final_n"] = len(df3)
    logging.info(
        "Subject selection: %d → %d (excluded: %s)", n0, len(df3), excl
    )
    return df3.reset_index(drop=True), excl


def select_seqlearning_subjects(df, task):
    """
    Filter for seqLearning analysis.

    Parameters
    ----------
    df : already-filtered resting-state DataFrame
    task : 'seqLearning6target' or 'seqLearning8target'
    """
    if task not in df.columns:
        logging.warning("seqLearning task '%s' not found in participants.tsv — 0 subjects", task)
        return df.iloc[0:0].copy().reset_index(drop=True)
    mask = df[task].astype(str).str.lower() == "available"
    sub = df[mask].copy()
    logging.info("seqLearning selection (%s): %d subjects", task, len(sub))
    return sub.reset_index(drop=True)


# ── File path helpers ───────────────────────────────────────────────────

def eeg_set_path(subject, task):
    """Return path to the .set EEG file for a subject/task.
    
    Tries BIDS naming first, then falls back to fuzzy matching.
    """
    # Standard BIDS path
    bids_path = (DATA_ROOT / subject / "eeg"
                 / f"{subject}_task-{task}_eeg.set")
    if bids_path.exists():
        return bids_path

    # Fallback: search for any .set file with the task name in the eeg/ dir
    eeg_dir = DATA_ROOT / subject / "eeg"
    if eeg_dir.is_dir():
        for f in eeg_dir.iterdir():
            if f.suffix == ".set" and task.lower() in f.name.lower():
                return f
        # If only one .set file exists, use it
        set_files = list(eeg_dir.glob("*.set"))
        if len(set_files) == 1:
            return set_files[0]

    # Also try .edf format (some releases use EDF instead of EEGLAB)
    edf_path = (DATA_ROOT / subject / "eeg"
                / f"{subject}_task-{task}_eeg.edf")
    if edf_path.exists():
        return edf_path

    # Return the BIDS path even if it doesn't exist (for error reporting)
    return bids_path


def events_tsv_path(subject, task):
    """Return path to the events.tsv for a subject/task."""
    # Standard BIDS path
    bids_path = (DATA_ROOT / subject / "eeg"
                 / f"{subject}_task-{task}_events.tsv")
    if bids_path.exists():
        return bids_path

    # Fallback: search in eeg/ dir
    eeg_dir = DATA_ROOT / subject / "eeg"
    if eeg_dir.is_dir():
        for f in eeg_dir.iterdir():
            if f.suffix == ".tsv" and "event" in f.name.lower() and task.lower() in f.name.lower():
                return f

    return bids_path


def channels_tsv_path(subject, task):
    """Return path to the channels.tsv for a subject/task."""
    # Standard BIDS path
    bids_path = (DATA_ROOT / subject / "eeg"
                 / f"{subject}_task-{task}_channels.tsv")
    if bids_path.exists():
        return bids_path

    # Fallback: search in eeg/ dir
    eeg_dir = DATA_ROOT / subject / "eeg"
    if eeg_dir.is_dir():
        for f in eeg_dir.iterdir():
            if f.suffix == ".tsv" and "channel" in f.name.lower():
                return f

    return bids_path


# ── Behavioural metrics from events.tsv ─────────────────────────────────

def parse_seqlearning_events(subject, task):
    """
    Parse seqLearning events.tsv and return behavioural accuracy.

    Returns
    -------
    dict with keys:
        accuracy : float (0-1), mean across 5 blocks
        n_blocks  : int
        user_answers : list of str
        correct_answers : list of str
    """
    tsv = events_tsv_path(subject, task)
    if not tsv.exists():
        return None

    df = pd.read_csv(tsv, sep="\t", na_values=["n/a"])
    # rows where user_answer is not NaN
    ans = df.dropna(subset=["user_answer"])
    if len(ans) == 0:
        return None

    user = ans["user_answer"].tolist()
    correct = ans["correct_answer"].tolist()
    accs = []
    for u, c in zip(user, correct):
        u_list = [int(x) for x in str(u).split("-") if x.strip()]
        c_list = [int(x) for x in str(c).split("-") if x.strip()]
        if len(c_list) == 0:
            continue
        matches = sum(1 for a, b in zip(u_list, c_list) if a == b)
        accs.append(matches / len(c_list))

    return {
        "accuracy": float(np.mean(accs)) if accs else 0.0,
        "n_blocks": len(accs),
        "user_answers": user,
        "correct_answers": correct,
    }


# ── Age group helpers ───────────────────────────────────────────────────

def age_group(age):
    """Map a numeric age to a group label string.

    Boundary handling: each group's age range is closed on the *lower* end and
    open on the *upper* end (half-open interval [lo, hi)), except for the
    last group which is closed on both ends.  This avoids the ambiguous
    boundary cases (8.0, 12.0, 16.0, 21.0) that would otherwise be lost
    to "other" or double-counted.

    Examples (with default AGE_GROUPS):
        age_group(7.0)  == "5-8"
        age_group(8.0)  == "5-8"   # boundary belongs to the lower group
        age_group(8.5)  == "9-12"
        age_group(12.0) == "9-12"
        age_group(21.0) == "17-21"  # last group, closed upper bound
    """
    try:
        age_f = float(age)
        if age_f != age_f:  # NaN check (NaN != NaN)
            return "other"
    except (TypeError, ValueError):
        return "other"
    sorted_groups = sorted(AGE_GROUPS.items(), key=lambda kv: kv[1][0])
    for _label, (lo, hi) in sorted_groups:
        # All groups use half-open intervals [lo, hi).  The upper bound hi
        # of the last group is intentionally exclusive so that ages just
        # above the cohort range fall through to "other".
        if lo <= age_f < hi:
            return _label
    return "other"


def age_group_series(ages):
    """Vectorised version of age_group for a pandas Series."""
    return pd.Series([age_group(a) for a in ages])
