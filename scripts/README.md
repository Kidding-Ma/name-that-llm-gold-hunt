# Scripts

## Kaggle-ready submissions

- `kaggle_submit.py` — verified CPU style-aware four-view SVC.
- `kaggle_submit_blend.py` — CPU style-aware + original SVC blend; separate experiment.
- `kaggle_best_two_stage_lora.py` — GPU two-stage LoRA + public pool candidate; requires external Kaggle inputs. The independent control requires `--no-twin --no-balance`; `--allow-transductive` explicitly enables test-batch clustering/uniform-prior balancing as a separate hypothesis. Supports 8B/14B-only ablations and optional private member score export (`--scores-output`).

## GPU setup and remote server

- `setup_gpu_env.sh` — creates an external venv, installs server packages without replacing CUDA torch, downloads/verifies compact inputs when requested, and runs strict preflight.
- `check_gpu_env.sh` — records non-secret GPU/import/disk/input checks; `--strict` is a fail gate.
- `run_gpu_control.sh` — creates a run directory, runs strict checks, executes the compact control, and saves hashes/logs/scores outside GitHub.
- `gpu_input_manifest.py` — stdlib-only Kagglehub/CLI downloader, safe archive extraction and exact input-tree verification.
- `server_preflight.py` — nonzero fail-closed private-server check for CUDA, packages, competition schema, pool, adapters, Qwen configs and disk.
- `lora_kaggle_preflight.py` — legacy broad Kaggle notebook inventory for experimental routes; **not** the compact server gate.
- `prvsiyan_route_preflight.py` — fail-closed CUDA/CSV/adapter/config screening for current v10 metadata; historical v4 branch-B completeness is not checked, and it does not reproduce or guarantee 0.604.
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
