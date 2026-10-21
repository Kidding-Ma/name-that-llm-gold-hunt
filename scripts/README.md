# Scripts

## Kaggle-ready submissions

- `kaggle_submit.py` — verified CPU style-aware four-view SVC.
- `kaggle_submit_blend.py` — CPU style-aware + original SVC blend; separate experiment.
- `kaggle_best_two_stage_lora.py` — GPU two-stage LoRA + public pool recipe; requires external Kaggle inputs.

## GPU setup

- `lora_kaggle_preflight.py` — checks Kaggle GPU, competition data, public datasets, model inputs, and checkpoint markers.
- `qwen_blend.py` — Qwen3-1.7B embedding + TF-IDF experiment; requires a Qwen model input.

## Local diagnostics

- `style_aware_runner.py` — prompt-grouped local OOF and visible-test smoke submission.
- `cpu_blend_inference.py` — local full-train CPU blend candidate.
- `cpu_blend_probe.py` — read-only diagnostic for existing OOF score banks.
- `final_inference.py` — local full-train style-aware inference.
- `audit_dataset.py` — local data/schema audit; run only when data is supplied privately.

Do not run a Kaggle-ready script against an incomplete local three-row test and interpret it as a leaderboard result.
