# Remote GPU server runbook

This is the no-question path for a new AI with only this GitHub repository and
SSH access to a Linux GPU server. It targets the compact Nishant control first.
It cannot create Kaggle authorization or competition acceptance on its own.
If the owner has not supplied a Kaggle credential and accepted the competition,
stop at the download step with a concrete blocker; do not guess or ask broad
architecture questions.

## 0. Check the only external prerequisites

Before running commands, confirm the SSH server has Linux, Python 3.10+, an
NVIDIA driver/GPU, enough disk for two model inputs, and either Kaggle API
credentials or all four inputs already staged. The AI must not invent missing
credentials. If Kaggle access is unavailable, it can still run repository tests
and inspect code, but cannot download the competition/model inputs or produce a
valid leaderboard score.

## 1. Clone and enter the repository

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIR>
git status --short
git log -1 --oneline
```

If HTTPS clone is blocked but the server has the owner's GitHub SSH key, use the SSH URL for the actual remote repository instead:

```bash
git clone <REPOSITORY_SSH_URL>
```

Do not store a private key, token or Kaggle credential in the repository.

## 2. Set external paths and credentials

Keep all data, weights and outputs outside the clone:

```bash
export DATA_ROOT="$HOME/name-that-llm-data"
install -d -m 700 "$DATA_ROOT"
```

Follow [`KAGGLE_DOWNLOADS.md`](KAGGLE_DOWNLOADS.md) to install a Kaggle token
file or a secret-manager environment token. GitHub SSH is not Kaggle auth.

## 3. Bootstrap Python and CUDA

Use the setup script. It does not silently replace an existing CUDA-enabled
PyTorch installation:

```bash
# If the base Python has no CUDA torch, install a matching wheel first using
# the PyTorch selector linked below; setup deliberately does not guess a wheel.
bash scripts/setup_gpu_env.sh --data-root "$DATA_ROOT" --download
source "$DATA_ROOT/.venv/bin/activate"
```

If it reports that CUDA torch is missing, inspect `nvidia-smi`, then install a
wheel matching the host driver from [PyTorch's selector](https://pytorch.org/get-started/locally/). Do not blindly use a CPU wheel. Re-run setup after installation.

For an already staged input tree:

```bash
bash scripts/setup_gpu_env.sh --data-root "$DATA_ROOT" --no-install
```

## 4. Strict preflight

```bash
bash scripts/check_gpu_env.sh --data-root "$DATA_ROOT" --strict
```

This writes a non-secret report under `$DATA_ROOT/logs`. It checks Python
imports, CUDA devices, free disk, exact competition CSV headers/labels, pool
schema, both nested adapter bundles and Qwen architecture selectors. It does
not prove exact Kaggle versions, licensing or historical v4 parity.

If this fails:

- missing inputs + no credential: configure Kaggle auth, then rerun setup with `--download`;
- no CUDA: do not run the LoRA route; use a GPU server or CPU control;
- 14B OOM risk: run `--only-8b` as an isolated diagnostic, not the full reported blend;
- only a 3-row test exists: check the Kaggle competition access; do not call it a leaderboard run.

## 5. Run with a durable log

The wrapper creates a unique run directory, records the command, preflight,
submission hash and output metadata, and tees output to a log:

```bash
bash scripts/run_gpu_control.sh --data-root "$DATA_ROOT"
```

The wrapper defaults to the independent `--no-twin --no-balance` control. Use
`--allow-transductive` only after explicit rule review and pseudo-test stress
checks; this changes the hypothesis and the recorded run variant.

It runs:

```text
python scripts/kaggle_best_two_stage_lora.py \
  --input-dir "$DATA_ROOT/input" \
  --no-twin --no-balance \
  --output "$DATA_ROOT/runs/<run-id>/submission.parquet" \
  --metadata "$DATA_ROOT/runs/<run-id>/run.json" \
  --scores-output "$DATA_ROOT/runs/<run-id>/member_scores.npz"
```

Use separate runs for diagnostics (the wrapper's default remains the
independent control; these are named alternatives):

```bash
bash scripts/run_gpu_control.sh --data-root "$DATA_ROOT" --only-8b
bash scripts/run_gpu_control.sh --data-root "$DATA_ROOT" --only-14b
bash scripts/run_gpu_control.sh --data-root "$DATA_ROOT" --allow-transductive
```

The wrapper's default is an independent control (`--no-twin --no-balance`),
which disables test-batch clustering and uniform-prior balancing. To explicitly
run the upstream transductive recipe, use `--allow-transductive` as a separate
named run after rule review; do not combine it with `--no-twin` or `--no-balance`.
The current inference script has no mid-model checkpoint/resume. If it is
interrupted, fix the cause and rerun the command; earlier completed branches are
not automatically reusable. Use `tmux` to protect a long run from SSH loss:

```bash
tmux new -s ntllm
bash scripts/run_gpu_control.sh --data-root "$DATA_ROOT" 2>&1 | tee "$DATA_ROOT/logs/manual-$(date -u +%Y%m%dT%H%M%SZ).log"
# Detach with Ctrl-b then d; reconnect with: tmux attach -t ntllm
```

Never run multiple variants in the same run directory.

## 6. Inspect the artifact

```bash
find "$DATA_ROOT/runs" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
sha256sum "$DATA_ROOT"/runs/*/submission.parquet
python - "$DATA_ROOT" <<'PY'
import pathlib, sys, pandas as pd
root = pathlib.Path(sys.argv[1]).expanduser().resolve()
for path in sorted(root.glob("runs/*/submission.parquet")):
    frame = pd.read_parquet(path)
    print(path, frame.shape, list(frame.columns), frame["model"].value_counts().to_dict())
PY
```

Confirm full hidden-test row count, exact columns, unique IDs, legal labels and
test order before submission. If `test.csv` has only three rows, stop: it is not
a valid leaderboard artifact.

## 7. Submit only after review

The competition is code-only and may require a Kaggle Code Notebook rather than
an arbitrary server artifact. First try the Kaggle web submission flow. If the
CLI accepts direct submissions for this competition, the command is:

```bash
RUN_DIR="$(ls -td "$DATA_ROOT"/runs/* | head -1)"
kaggle competitions submit -c name-that-llm \
  -f "$RUN_DIR/submission.parquet" \
  -m "compact control $(basename "$RUN_DIR")"
kaggle competitions submissions -c name-that-llm
```

If direct upload is rejected because the competition is code-only, do not keep
retrying. Copy the same code/input setup to Kaggle Code, run there, Save Version,
and submit from the notebook. Record the notebook version and actual public and
private scores in [`EXPERIMENT_LOG_TEMPLATE.md`](EXPERIMENT_LOG_TEMPLATE.md).

## 8. Only then research higher routes

Do not start the prvsiyan route before recording the compact control score. The
historical v4 `0.604` is an upstream source claim. Current v10 is unscored and
branch-B-disabled; the compact setup intentionally does not download it. Read
[`PUBLIC_GPU_ROUTES.md`](PUBLIC_GPU_ROUTES.md) and verify the historical
Lightning dependency before calling anything a v4 reproduction.
