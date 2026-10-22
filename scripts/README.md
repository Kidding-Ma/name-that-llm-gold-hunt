# Scripts

## Kaggle-ready submissions

- `kaggle_submit.py` — verified CPU style-aware four-view SVC.
- `kaggle_submit_blend.py` — CPU style-aware + original SVC blend; separate experiment.
- `kaggle_best_two_stage_lora.py` — GPU two-stage LoRA + public pool control; requires external Kaggle inputs. Supports 8B/14B-only ablations and optional private member score export (`--scores-output`); weights/default predictions are unchanged.

## GPU setup

- `lora_kaggle_preflight.py` — inventory check for the compact Nishant two-stage recipe and the broader experimental stack.
- `prvsiyan_route_preflight.py` — fail-closed CUDA/CSV/adapter/config screening for current v10 metadata; named mounts are heuristics, historical v4 branch-B completeness is not checked, and it does not reproduce or guarantee 0.604.
- `qwen_blend.py` — optional Qwen3-1.7B embedding + TF-IDF experiment; requires a Qwen model input and uses transductive clustering only when `ADAPT=True`.

## Local/private-data tools

All local scripts require explicit private input/output paths. They do not depend on the original Windows workspace or its vendored packages.

- `style_aware_runner.py` — prompt-grouped local OOF; writes private score banks and diagnostics.
- `cpu_blend_inference.py` — local full-train CPU blend candidate.
- `cpu_blend_probe.py` — read-only diagnostic for private OOF score banks.
- `final_inference.py` — local full-train style-aware + numeric-style inference. Its compact metadata intentionally redacts local input/output paths.
- `audit_dataset.py` — local data/schema audit; writes compact metadata only.
- `release_audit.py` — dependency-free public-tree policy audit; run from the repository root before every push.

Example:

```bash
python scripts/audit_dataset.py --data-dir /private/name-that-llm --output-dir /private/artifacts/audit
python scripts/style_aware_runner.py --data-dir /private/name-that-llm --output-dir /private/artifacts/style
python scripts/final_inference.py --data-dir /private/name-that-llm --output /private/artifacts/submission.parquet
```

Do not run a Kaggle-ready script against an incomplete local three-row test and interpret it as a leaderboard result. Do not commit the private output directory.
