# Artifacts

This directory stores the teacher artifacts used by the released BA-MPD
setting. Because the checkpoints are large, keep them out of ordinary git
history and publish this directory through Git LFS or an external release asset.

Current subdirectories:

```text
teacher_logits/
teacher_checkpoints/
reports/
```

Teacher logits should be paired with an index JSON containing `sample_ids` in
the same order as the logits array.

The `reports/teacher_accuracy_report.json` file records the teacher metrics
used to check the artifact set. TAU teacher metrics are evaluated on the
official test split. BEANS-CBI teacher metrics are reported for the released
fixed split.
