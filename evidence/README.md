# Small evidence bundle

Only compact, non-sensitive metrics are included here. Raw train/test data, OOF arrays, submissions, PDFs, model weights, and local dependency trees are deliberately excluded.

- `metrics/style_aware_local_metrics.json`: verified four-view CPU grouped OOF summary.
- `metrics/cpu_numeric_style_local_metrics.json`: verified four-view plus numeric-style CPU grouped OOF summary.
- `metrics/baseline_results.json`: original CPU baseline grouped/random diagnostic.
- `metrics/cpu_blend_probe_results.json`: conservative CPU blend probe and warnings.
- `metrics/verified_data_audit.json`: schema, counts, and data hash from the private local snapshot.

The data hash is a provenance check, not a substitute for attaching the official competition data. New OOF arrays, predictions, submissions, and confusion tables belong outside this repository.
