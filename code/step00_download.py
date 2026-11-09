#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0 — Download HBN EEG release from AWS S3 (public, no credentials needed).

Usage:
  python step00_download.py --release 1
  python step00_download.py --release 1 --check  # only show size, don't download

Requires: AWS CLI (aws command) installed.
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

# Add code dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RELEASES, RAW_BASE


def get_release_path(release_num):
    """Return local path for a release's raw data."""
    info = RELEASES[release_num]
    return RAW_BASE / f"HBN-Release{release_num}-{info['dataset']}"


def download_release(release_num, dry_run=False):
    """Download a release from S3 (public bucket, no credentials)."""
    info = RELEASES[release_num]
    s3_uri = info["s3"]
    local_path = get_release_path(release_num)

    print(f"Release {release_num}: {info['dataset']}")
    print(f"  S3:     {s3_uri}")
    print(f"  Local:  {local_path}")
    print(f"  Expected subjects: {info['n_subjects']}")

    if dry_run:
        # Just estimate size
        cmd = ["aws", "s3", "ls", "--no-sign-request", "--summarize",
               "--human-readable", s3_uri + "/"]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Print last 5 lines (size summary)
        lines = result.stderr.strip().split('\n') if result.stderr else result.stdout.strip().split('\n')
        for line in lines[-5:]:
            print(f"  {line}")
        return

    # Check if already downloaded and valid
    if local_path.exists() and (local_path / "participants.tsv").exists():
        print(f"  Already downloaded, skipping.")
        return

    # If directory exists but participants.tsv is missing, it's a partial/failed download.
    # Rename to .partial suffix instead of deleting, to prevent accidental data loss.
    if local_path.exists() and not (local_path / "participants.tsv").exists():
        import shutil
        # Use parent / name + ".partial" — with_suffix() requires a non-empty
        # existing suffix, which fails for directory names like
        # "HBN-Release1-ds005505" (no dot in stem).
        backup_path = local_path.parent / (local_path.name + ".partial")
        # Rename FIRST (preserve incomplete data), only then clean up any
        # previous .partial backup. This avoids the situation where a failed
        # rename leaves both directories destroyed.
        print(f"  Partial download detected (no participants.tsv), backing up...")
        local_path.rename(backup_path)
        print(f"  Backed up to: {backup_path}")
        print(f"  (You can safely delete this manually if not needed)")

    local_path.mkdir(parents=True, exist_ok=True)

    # Download with aws s3 sync (resumable)
    # Do NOT exclude .json — BIDS sidecar JSONs (*_eeg.json, *_channels.json)
    # are required by MNE-Python to read the EEG data.
    cmd = ["aws", "s3", "sync", "--no-sign-request",
           s3_uri + "/", str(local_path) + "/"]

    print(f"  Downloading...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: Download failed with code {result.returncode}")
        sys.exit(1)

    # Post-download verification
    if not (local_path / "participants.tsv").exists():
        print(f"  ERROR: Download completed but participants.tsv not found!")
        print(f"  This may indicate an S3 bucket structure issue.")
        print(f"  Trying aws s3 cp for participants.tsv specifically...")
        cp_cmd = ["aws", "s3", "cp", "--no-sign-request",
                  s3_uri + "/participants.tsv", str(local_path / "participants.tsv")]
        cp_result = subprocess.run(cp_cmd, capture_output=True, text=True)
        if cp_result.returncode != 0:
            print(f"  ERROR: Could not download participants.tsv: {cp_result.stderr}")
            sys.exit(1)

    # Count subjects
    n_subjects = len([d for d in local_path.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    print(f"  Done. {n_subjects} subjects at: {local_path}")


def delete_release(release_num):
    """Delete raw data for a release to free disk space."""
    local_path = get_release_path(release_num)
    if local_path.exists():
        import shutil
        size_gb = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file()) / 1e9
        shutil.rmtree(local_path)
        print(f"  Deleted {local_path} ({size_gb:.1f} GB freed)")
    else:
        print(f"  Path not found: {local_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HBN EEG release from S3")
    parser.add_argument("--release", type=int, required=True,
                        help="Release number (1-10)")
    parser.add_argument("--check", action="store_true",
                        help="Only check size, don't download")
    parser.add_argument("--delete", action="store_true",
                        help="Delete raw data for this release")
    args = parser.parse_args()

    if args.release not in RELEASES:
        print(f"Invalid release {args.release}. Available: {list(RELEASES.keys())}")
        sys.exit(1)

    if args.delete:
        delete_release(args.release)
    else:
        download_release(args.release, dry_run=args.check)
