# TAU Urban Acoustic Scenes Splits

This directory contains the official DCASE 2024 Task 1 low-resource split files
used by the TAU Urban Acoustic Scenes experiments:

```text
train_5.csv
train_10.csv
train_25.csv
train_full.csv
test.csv
```

The files keep the original DCASE-style `.csv` naming, but the columns are
tab-separated. Each row contains `filename` and `scene_label`. Raw audio is not
included; users should obtain the official TAU/DCASE data separately.

The paper uses `train_5.csv`, `train_10.csv`, and `train_25.csv` for the
low-resource budgets. `train_full.csv` is included as the official full training
metadata for completeness.
