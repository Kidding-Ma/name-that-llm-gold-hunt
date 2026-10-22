# Kaggle inputs: authentication, download and verification

This document is for a private Linux GPU server. GitHub SSH access only clones
this repository; it does **not** grant Kaggle access. Before an AI can download
inputs or submit a score, the owner must join the competition, accept its rules,
and provide a Kaggle API credential to the server.

## Exact compact-route inputs

The canonical machine-readable list is [`config/kaggle_inputs.json`](../config/kaggle_inputs.json).
The default route downloads:

| Kind | Exact identifier | Required role |
|---|---|---|
| Competition | `name-that-llm` | `train.csv` and the full competition `test.csv` when the API exposes it |
| Dataset | `nishantdahal/ntl-two-stage-lora` | pool plus `lora8b_v1`/`lora14b_v1` adapters and heads |
| Model | `qwen-lm/qwen-3/Transformers/8b-base/1` | Qwen3 8B base |
| Model | `qwen-lm/qwen-3/Transformers/14b-base/1` | Qwen3 14B base |

The compact setup does **not** auto-download prvsiyan's current v10 inputs.
That notebook is explicitly unscored and disables historical v4's unavailable
branch-B Lightning checkpoint. See [`PUBLIC_GPU_ROUTES.md`](PUBLIC_GPU_ROUTES.md)
before doing optional research.

## Credential setup (never commit or print it)

### Option A: Kaggle API token file

Create a token in Kaggle account settings, transfer it over the already secured
SSH connection, and lock its permissions. Run these commands on the server:

```bash
install -d -m 700 "$HOME/.kaggle"
# Transfer kaggle.json from your local machine with scp; do not paste it into chat.
chmod 600 "$HOME/.kaggle/kaggle.json"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
```

### Option B: environment token

Use your secret manager or the SSH session environment; do not put the token in
a shell history, repository file or log:

```bash
export KAGGLE_API_TOKEN='(set this in the server secret manager, not GitHub)'
```

The repository scripts detect only the source label `env_token` or `kaggle_json`.
They never write the token to a manifest or print its value. If neither source
exists, setup can verify already-staged inputs but cannot download missing ones.

## Check account access before downloading

After installing the Kaggle CLI, run commands that do not expose the secret:

```bash
kaggle --version
kaggle competitions files -c name-that-llm
kaggle datasets status -d nishantdahal/ntl-two-stage-lora
```

A 401/403 usually means the credential is missing/invalid or the account has
not accepted the competition rules. Open the competition page, join it, accept
all rules, then retry. Do not retry indefinitely or use someone else's token.

## Automated download

From the repository root:

```bash
export DATA_ROOT="$HOME/name-that-llm-data"
bash scripts/setup_gpu_env.sh --data-root "$DATA_ROOT" --download
```

The setup calls [`gpu_input_manifest.py`](../scripts/gpu_input_manifest.py).
It tries `kagglehub` first, then falls back to the official Kaggle CLI model-
variation command if the returned files do not verify. Some `kagglehub`
releases ignore their requested `output_dir`; the wrapper re-materializes
returned cache paths and verifies the target after every attempt.
It uses these CLI families when the installed CLI supports them:

```bash
kaggle competitions download -c name-that-llm -p "$DATA_ROOT/input/competition/name-that-llm"
kaggle datasets download -d nishantdahal/ntl-two-stage-lora -p "$DATA_ROOT/input/nishantdahal/ntl-two-stage-lora"
kaggle models variations versions download \
  -p "$DATA_ROOT/input/models/qwen-3/8b-base" --untar \
  qwen-lm/qwen-3/Transformers/8b-base/1
kaggle models variations versions download \
  -p "$DATA_ROOT/input/models/qwen-3/14b-base" --untar \
  qwen-lm/qwen-3/Transformers/14b-base/1
```

Kaggle CLI model syntax varies by release. The current official reference uses
`models variations versions download <owner>/<model>/<framework>/<variation>/<version>`
with `--untar`; the wrapper uses that numbered variation-version form. See the
[official model-variation version reference](https://github.com/Kaggle/kaggle-cli/blob/f0afa32699d28c97f82691728ada3ed8c16c5abf/skills/references/model_variations_versions.md)
if the installed CLI changes its command syntax. The wrapper records attempted
commands without secrets. It does not trust an archive blindly: downloaded
ZIP/TAR archives are extracted with path-traversal and symlink checks, then
verified.

Do not assume a download contains the hidden leaderboard test. The visible local
file may contain only three rows, and a competition API may limit files by access
or competition policy. A successful server inference on three rows is a format
smoke test, not a leaderboard submission.

## Verify and record versions

```bash
python scripts/gpu_input_manifest.py verify --data-root "$DATA_ROOT" --hash metadata
python scripts/gpu_input_manifest.py write --data-root "$DATA_ROOT" --hash all
python scripts/server_preflight.py --data-root "$DATA_ROOT" --strict
```

`manifest.json` is generated under the external data root and is ignored by the
public repository. It records handles, resolved paths, file sizes, optional
SHA-256 values, git commit, verification status and credential **source label**;
it never stores a token. Hashing all model weights can take a long time.

Inspect the current input license and exact attached version privately. Public
visibility does not grant permission to redistribute third-party data, adapters,
model weights or raw notebook blobs.
