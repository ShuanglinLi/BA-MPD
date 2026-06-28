# Boundary-Anchored Mass-Partitioned Distillation

This repository is a release for **low-resource acoustic
recognition with imperfect teachers**.

## Installation

The code expects Python 3.9 or newer. Main dependencies are PyTorch,
torchaudio, pandas, pyarrow, soundfile, and scikit-learn.

## Released Splits

We provide fixed low-resource BEANS-CBI splits for reproduction:

```text
splits/beans_cbi/
  train_10.csv
  train_25.csv
  train_50.csv
  valid.csv
  test.csv
  label_mapping.csv
```

We also include the official DCASE 2024 Task 1 split metadata (https://github.com/CPJKU/dcase2024_task1_baseline/releases/tag/files) used for TAU Urban Acoustic Scenes:

```text
splits/tau/
  train_5.csv
  train_10.csv
  train_25.csv
  train_full.csv
  test.csv
```

## Teacher Logits and Checkpoints

BA-MPD student training uses fixed teacher logits. Teacher checkpoints are included for reference and
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
The class order for BEANS-CBI is defined by `splits/beans_cbi/label_mapping.csv`.

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

For a BEANS-CBI-only run:

```bash
export BEANS_CBI_DATA_DIR=/path/to/beans_cbi/parquet
export TEACHER_LOGITS_DIR=artifacts/teacher_logits/beans_cbi
BUDGETS="25" bash scripts/run_beans_cbi.sh
```
