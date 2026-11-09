# eeg-covariance-maturation


This pipeline takes raw resting-state EEG from the Healthy Brain Network (HBN) dataset through preprocessing, Riemannian covariance feature extraction, nested-cross-validated brain-age prediction, and Manifold Geometry Deviation (MGD) normative modeling, and regenerates every quantitative figure used in the paper.

---

## Dataset

- **Healthy Brain Network (HBN) EEG**, releases R1–R10 (public releases on OpenNeuro: `ds005505`–`ds005516`, skipping `ds005513`), mirrored on AWS S3 at `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/` (no credentials required).
- Resting-state recordings, 128-channel EGI GSN HydroCel, 500 Hz.
- Final analysis sample after quality control: **1,644 participants, ages 5–21** (see Supplementary Figure S4 for the full enrollment flow).
- Raw data are **not** redistributed with this repository; `step00_download.py` fetches them directly from S3/OpenNeuro.

## Requirements

- Python ≥ 3.9
- AWS CLI (`pip install awscli`) — only needed for downloading raw data
- Python dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `mne`, `pyriemann`, `scikit-learn`, `numpy`, `scipy`, `pandas`, `matplotlib`, `pingouin`, `joblib`.

## Repository layout

```
.
├── README.md
├── requirements.txt
└── code/
    ├── config.py                     # all paths, EEG parameters, bands, hyperparameters
    ├── utils.py                      # logging, age-group helpers
    ├── step00_download.py            # download one HBN release from S3
    ├── step01_preprocess.py          # filtering, ICA, bad-channel handling, segment extraction
    ├── step02_extract_features.py    # band-wise covariance → tangent space (FBTS), PSD, DE
    ├── step03_brain_age_model.py     # nested-CV brain-age models (SVR/Ridge/GBR, ablation, LOSO)
    ├── step04_statistics.py          # BAG–phenotype statistics, risk-group ROC, rest–task shift
    ├── step04b_mgd_analysis.py       # MGD normative modeling (leave-one-out Riemannian mean)
    ├── step04c_mgd_partial_corr.py   # MGD–CBCL partial correlations (age + sex), CIs, FDR
    ├── step05_figures.py             # main figures 2–8 and supplementary figures S2–S3
    ├── step05b_mgd_figures_nature.py # MGD trajectory / biomarker figure variants
    ├── step05c_supplementary_figures.py  # Figure S1 (bandwidth sensitivity)
    ├── step05_figures_nature.py      # alternate (Nature-style) rendering of the same figures
    ├── step05_all_figures.py         # dispatcher: runs all step05* scripts in order
    ├── figureS4_sample_flow.py       # Figure S4 participant flow chart
    ├── step06_revision_analyses.py   # revision analyses (full-pipeline PSD/DE baselines, feature
    │                                 #   stability, exclusion sensitivity, broadband control)
    ├── print_versions.py             # print environment versions (for reporting)
    └── orchestration scripts:
        run_all.sh / run_pipeline.sh / run_release.sh
        download_all_releases.sh / run_all.ps1 / diagnose.sh
```

All outputs are written to `results/` at the project root (`results/preprocessed/`, `results/features/`, `results/analysis/`, `results/figures/`, `results/logs/`).

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 0 | `step00_download.py --release N` | Download one HBN EEG release from S3 |
| 1 | `step01_preprocess.py` | MNE preprocessing: montage, bad channels, average reference, 0.5–45 Hz + 60 Hz notch, FastICA (30 comps) with EOG proxies, eyes-closed segment extraction |
| 2 | `step02_extract_features.py` | Band-wise SPD covariance (shrinkage estimator), tangent-space projection against a global Riemannian mean → FBTS features; PSD and DE baselines |
| 3 | `step03_brain_age_model.py` | Brain-age prediction: SVR + FBTS (main), Ridge/GBR baselines, SVR-PSD / SVR-DE baselines, single-band ablation; nested 5-fold (outer) × 3-fold (inner) CV, plus LOSO |
| 4 | `step04_statistics.py` | BAG vs. psychopathology partial correlations, risk-group comparisons, ROC, sex-difference permutation tests, rest-to-task centroid shift |
| 4b | `step04b_mgd_analysis.py` | MGD: leave-one-out kernel-weighted Riemannian normative trajectory, geodesic deviation, age-bin z-scoring |
| 4c | `step04c_mgd_partial_corr.py` | MGD–CBCL partial correlations controlling age + sex, with 95% CIs and FDR correction |
| 5 | `step05*.py` | All paper figures (see table below) |
| 6 | `step06_revision_analyses.py` | Revision analyses requested during peer review (`--parts A,B,C,D,E`) |

### Running

Quick start (Linux/macOS/WSL, full pipeline):

```bash
bash code/run_all.sh            # check → download → preprocess → features → models → stats → figures
```

Windows (PowerShell):

```powershell
.\code\run_all.ps1              # run all steps; or: .\code\run_all.ps1 2 3  (specific steps)
```

Individual steps (from the `code/` directory):

```bash
export HBN_DATA_ROOT=/path/to/HBN          # override the default data location (config.py)
python step00_download.py --release 1
python step01_preprocess.py
python step02_extract_features.py
python step03_brain_age_model.py
python step04_statistics.py
python step04b_mgd_analysis.py
python step04c_mgd_partial_corr.py
python step05_all_figures.py
```

The data root defaults to `E:\datasets\HBN` on Windows and `/mnt/data1/home/tanhuang/datasets/HBN` on Linux; set the `HBN_DATA_ROOT` environment variable to point it elsewhere.

## Figures

| Paper figure | Output file | Generated by |
|--------------|-------------|--------------|
| Figure 1 | — | conceptual schematic (illustration; not produced by this pipeline) |
| Figure 2 (cohort) | `figure01_demographics.png` | `step05_figures.py :: figure01` |
| Figure 3 (t-SNE) | `figure02_tsne.png` | `step05_figures.py :: figure02` |
| Figure 4 (model performance, mean ± fold SD) | `figure03_model_performance.png` | `step05_figures.py :: figure03` |
| Figure 5 (BAG vs. psychopathology, age- and sex-adjusted) | `figure04_bag_clinical.png` | `step05_figures.py :: figure04` |
| Figure 6 (MGD biomarker) | `figure05_mgd_biomarker.png` | `step05_figures.py :: figure05` |
| Figure 7 (cortical mapping) | `figure07_cortical_mapping.png` | `step05_figures.py :: figure07` |
| Figure 8 (MGD trajectory) | `figure06_mgd_trajectory.png` | `step05_figures.py :: figure06` |
| Figure S1 (bandwidth sensitivity) | `figureS1_bandwidth_sensitivity.png` | `step05c_supplementary_figures.py` |
| Figure S2 (rest-to-task shift) | `figure08_rest_task_shift.png` | `step05_figures.py :: figure08` |
| Figure S3 (sex differences) | `figure10_sex_differences.png` | `step05_figures.py :: figure10` |
| Figure S4 (sample flow) | `figureS4_sample_flow.png` | `figureS4_sample_flow.py` |

## Reproducibility notes

- All random seeds are fixed in `config.py` (`RANDOM_STATE = 42`); ICA, cross-validation splits, t-SNE, and permutation tests are seeded.
- Feature selection (SelectKBest) and PCA are fitted **inside** each training fold only; all reported brain-age metrics are pooled out-of-fold predictions, with fold-level variability reported alongside.
- Figure 4 error bars are the standard deviation of per-fold correlations; Figure 5 panels residualize both age and sex before correlating BAG with CBCL scores (matching the tabulated statistics).
- `config.py` documents every analysis parameter (frequency bands, CV folds, hyperparameter grids, MGD bandwidth and age bins). Check it before re-running.

## Data availability & citation

Raw HBN data are publicly available via OpenNeuro and the HBN S3 bucket. When using this code, please cite the paper (DOI will be added here after acceptance).

## License

No license has been applied yet — add one (e.g., MIT) before making the repository public.
