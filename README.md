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

## Student Checkpoints

Student checkpoints for TAU and BEANS-CBI are available in the GitHub release:

[Download student checkpoints (TAU and BEANS-CBI)](https://github.com/ShuanglinLi/BA-MPD/releases/tag/v1.0-student-checkpoints)

## Quick Start

After setting up the datasets, run the following scripts to reproduce the paper’s main results across different label budgets:

```bash
bash scripts/run_tau.sh
bash scripts/run_beans_cbi.sh
```

To run a specific BEANS-CBI label budget:

```bash
BUDGETS="25" bash scripts/run_beans_cbi.sh
```
