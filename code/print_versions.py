# -*- coding: utf-8 -*-
"""
Print the exact software versions used by the analysis pipeline.

Run this on the machine that produced the results (the same environment that
runs step01-step04), then paste the output into the Methods section of the
manuscript.  Nothing is estimated here: every number is read from the
installed distribution at run time.

Usage:
    python print_versions.py

Optional: save to file
    python print_versions.py > versions.txt
"""
import platform
import sys

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 62)
print("Analysis environment")
print("=" * 62)
print("Python            :", sys.version.split()[0], "|", platform.python_implementation())
print("Platform          :", platform.platform())
print("Machine           :", platform.machine())
print("Processor         :", platform.processor() or "n/a")
print()

# ── third-party packages ────────────────────────────────────────────────
PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("mne", "MNE-Python"),
    ("pyriemann", "pyRiemann"),
    ("pingouin", "pingouin"),
    ("matplotlib", "matplotlib"),
    ("statsmodels", "statsmodels"),
    ("seaborn", "seaborn"),
    ("h5py", "h5py"),
    ("joblib", "joblib"),
]

print("=" * 62)
print("Package versions")
print("=" * 62)
missing = []
for mod, label in PACKAGES:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "unknown")
        print("%-18s: %s" % (label, ver))
    except Exception as exc:            # noqa: BLE001 - report, do not crash
        missing.append(label)
        print("%-18s: NOT INSTALLED (%s)" % (label, type(exc).__name__))
print()

# ── pipeline configuration actually used ────────────────────────────────
print("=" * 62)
print("Pipeline configuration (from config.py)")
print("=" * 62)
try:
    import config

    rows = [
        ("SFREQ (Hz)", config.SFREQ),
        ("N_CHANNELS", config.N_CHANNELS),
        ("REREference", config.REREference),
        ("BANDPASS (Hz)", "%s-%s" % (config.BANDPASS_LOW, config.BANDPASS_HIGH)),
        ("NOTCH (Hz)", config.NOTCH_FREQ),
        ("BANDS", ", ".join("%s %s-%s Hz" % (b, config.BANDS[b][0], config.BANDS[b][1])
                            for b in config.BAND_ORDER)),
        ("COV_ESTIMATOR", config.COV_ESTIMATOR),
        ("TANGENT_REF", config.TANGENT_REF),
        ("MIN_REST_SAMPLES", "%d (= %.0f s at %d Hz)"
                             % (config.MIN_REST_SAMPLES,
                                config.MIN_REST_SAMPLES / config.SFREQ,
                                config.SFREQ)),
        ("ICA components", config.ICA_N_COMPONENTS),
        ("ICA max variance loss", config.ICA_MAX_VARIANCE_LOSS),
        ("MAX_BAD_CH_RATIO", config.MAX_BAD_CH_RATIO),
        ("CV_FOLDS", config.CV_FOLDS),
        ("INNER_CV_FOLDS", config.INNER_CV_FOLDS),
        ("RANDOM_STATE", config.RANDOM_STATE),
        ("PCA_N_COMPONENTS", config.PCA_N_COMPONENTS),
        ("FS_K", config.FS_K),
        ("AGE_GROUPS", ", ".join("%s = [%g, %g)" % (k, v[0], v[1])
                                for k, v in config.AGE_GROUPS.items())),
    ]
    for label, value in rows:
        print("%-24s: %s" % (label, value))
except Exception as exc:                # noqa: BLE001
    print("config.py could not be imported:", exc)
print()

if missing:
    print("WARNING: not installed ->", ", ".join(missing))
print("Paste the block above into the manuscript Methods section.")
