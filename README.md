# Boundary-Anchored Mass-Partitioned Distillation

This repository is a lightweight method release for **Boundary-Anchored
Mass-Partitioned Distillation (BA-MPD)**, proposed for low-resource acoustic
recognition with imperfect teachers.

The release focuses on the BA-MPD implementation and the artifacts needed to
run the released setting. It is not intended to be a full baseline zoo. Baseline
implementations should be taken from their original papers or official code.

## Contents

```text
ba_mpd/
  losses/distillation.py      # boundary anchoring, MPD, BA-MPD
  datasets/beans_cbi.py       # lazy BEANS-CBI parquet loader
  models/cnn14.py             # PANNs-style CNN14 student
  train_bampd_beans.py        # BEANS-CBI BA-MPD training entry
configs/main/                 # main-result configuration records
splits/beans_cbi/             # released fixed low-resource BEANS-CBI trial
splits/tau/                   # official DCASE 2024 Task 1 split metadata
scripts/run_beans_cbi.sh      # reproduce BEANS-CBI BA-MPD settings
scripts/run_tau.sh            # reproduce TAU BA-MPD settings
artifacts/                    # teacher logits, teacher checkpoints, reports
```

## Installation

```bash
pip install -e .
```

The code expects Python 3.9 or newer. Main dependencies are PyTorch,
torchaudio, pandas, pyarrow, soundfile, and scikit-learn.

## GitHub Artifact Upload

This reproduction folder includes teacher checkpoints and teacher logits. The
checkpoint files are larger than GitHub's ordinary 100 MB file limit, so use Git
LFS before adding the repository:

```bash
git lfs install
git add .gitattributes
git add .
git commit -m "Release BA-MPD reproduction package"
```

The `.gitattributes` file tracks `*.pt`, `*.pth`, `*.ckpt`, `*.npy`, and
`*.npz` through Git LFS. As an alternative, the `artifacts/` directory can be
uploaded as a GitHub Release asset while keeping the code repository small.

After downloading or cloning the artifacts, users can verify file integrity with:

```bash
shasum -a 256 -c artifacts/checksums.sha256
```

## Released Splits

We provide one fixed low-resource BEANS-CBI trial for lightweight
reproduction:

```text
splits/beans_cbi/
  train_10.csv
  train_25.csv
  train_50.csv
  valid.csv
  test.csv
  label_mapping.csv
```

The CSV files store parquet shard names, row-group indices, sample IDs, and
labels. They do not contain machine-specific absolute paths. Pass the local
BEANS-CBI parquet directory through `BEANS_CBI_DATA_DIR` or `--data-dir`.

The paper reports averages over multiple low-resource trials. The files here
define the fixed release trial used by the provided artifacts.

We also include the official DCASE 2024 Task 1 split metadata used for TAU
Urban Acoustic Scenes:

```text
splits/tau/
  train_5.csv
  train_10.csv
  train_25.csv
  train_full.csv
  test.csv
```

These TAU files use tab-separated `filename` and `scene_label` columns while
retaining the original `.csv` extension.

## Teacher Logits and Checkpoints

BA-MPD student training uses fixed teacher logits. We recommend releasing the
teacher logits as the primary reproduction artifact, because they exactly fix
the teacher predictive distribution used during distillation and make student
training independent of teacher checkpoint loading, preprocessing drift, or
forward-pass differences. Teacher checkpoints are included for reference and
for users who want to re-export logits.

Released layout:

```text
artifacts/
  teacher_logits/
    beans_cbi/
      train_10_logits.npy
      train_10_logits.index.json
      train_25_logits.npy
      train_25_logits.index.json
      train_50_logits.npy
      train_50_logits.index.json
    tau/
      train_5_logits.npy
      train_5_logits.index.json
      train_10_logits.npy
      train_10_logits.index.json
      train_25_logits.npy
      train_25_logits.index.json
  teacher_checkpoints/
    beans_cbi/
    tau/
  reports/
```

Each `*.index.json` file must contain `sample_ids` in the same order as the
rows of the corresponding logits array. The file
`artifacts/reports/teacher_accuracy_report.json` records the teacher metrics
used to verify the released artifacts against the paper reference rows.

For each logits file, the release fixes the benchmark, budget, teacher
architecture, split file, class order, sample ID order, and whether values are
raw logits. The class order for BEANS-CBI is defined by
`splits/beans_cbi/label_mapping.csv`.

Raw audio is not included. Users should obtain the original datasets from their
official sources and place them according to the paths used at runtime.

## Reproduce Main BA-MPD Results

To run the BA-MPD experiments corresponding to the paper's main-result budgets,
use the dataset-specific scripts:

```bash
export BEANS_CBI_DATA_DIR=/path/to/beans_cbi/parquet
export DCASE_TASK1_REPO=/path/to/dcase2024_task1_baseline_official
export DCASE24_DATASET_DIR=/path/to/TAU-urban-acoustic-scenes-2022-mobile-development

bash scripts/run_tau.sh
bash scripts/run_beans_cbi.sh
```

The default scripts run TAU 5/10/25% and BEANS-CBI 10/25/50%. The release uses
one fixed trial, while the paper reports averages over multiple trials. Exact
equality is therefore not expected, especially for the smallest low-resource
budgets.

For a BEANS-CBI-only run:

```bash
export BEANS_CBI_DATA_DIR=/path/to/beans_cbi/parquet
export TEACHER_LOGITS_DIR=artifacts/teacher_logits/beans_cbi
BUDGETS="25" bash scripts/run_beans_cbi.sh
```

The main BA-MPD hyperparameters are:

- `TOP_K=5` for BEANS-CBI.
- `ALPHA=2.0` for the complement-region relation loss.
- `temperature=2.0`.
- `kd_weight=0.3`.

To run MPD without boundary anchoring:

```bash
OBJECTIVE=mpd BUDGETS="25" bash scripts/run_beans_cbi.sh
```

## TAU Urban Acoustic Scenes

For TAU Urban Acoustic Scenes, the paper follows the official DCASE 2024 Task 1
CP-Mobile training recipe. This release provides the official split metadata,
the portable BA-MPD loss implementation, and the main-result configuration
records under `configs/main/`. Integrate `ba_mpd.losses.ba_mpd_loss` into the
official DCASE training loop and use the fixed teacher logits under
`artifacts/teacher_logits/tau/`.

## Citation

```bibtex
@article{your_bampd_2026,
  title={Learning from Imperfect Teachers for Low-Resource Acoustic Generalization},
  author={TBD},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2026}
}
```
