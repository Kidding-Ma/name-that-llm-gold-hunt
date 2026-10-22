#!/usr/bin/env bash
# Bootstrap a private Linux GPU server without overwriting an existing CUDA torch.
set -Eeuo pipefail
IFS=$'\n\t'

DATA_ROOT="${DATA_ROOT:-$HOME/name-that-llm-data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV=""
DO_DOWNLOAD=0
NO_INSTALL=0
# Inherit a preinstalled CUDA-enabled torch from the base interpreter by default.
# Use --no-system-site-packages only when the selected base has no usable torch
# and the venv will be provisioned separately.
SYSTEM_SITE=1
HASH_MODE="metadata"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_gpu_env.sh [options]
  --data-root PATH       External data/log/output directory (default: $DATA_ROOT)
  --python PATH          Python 3.10+ executable (default: python3)
  --venv PATH            Virtualenv path (default: $DATA_ROOT/.venv)
  --download             Download missing compact-route Kaggle inputs
  --hash-all             SHA-256 hash all model weights (slow)
  --no-install              Do not pip-install requirements-server.txt (useful for a prebuilt venv)
  --system-site-packages    Let the venv see packages from the base Python (default)
  --no-system-site-packages Do not expose base Python packages; preinstall CUDA torch in the venv
  -h, --help                Show this help

Credentials are never printed. For downloads, preconfigure KAGGLE_API_TOKEN or
KAGGLE_CONFIG_DIR/kaggle.json and accept the competition rules first.
This script deliberately does not install torch: install a CUDA-matched torch
manually using the PyTorch selector before rerunning this setup.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --download) DO_DOWNLOAD=1; shift ;;
    --hash-all) HASH_MODE=all; shift ;;
    --no-install) NO_INSTALL=1; shift ;;
    --system-site-packages) SYSTEM_SITE=1; shift ;;
    --no-system-site-packages) SYSTEM_SITE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2; exit 2
fi
DATA_ROOT="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$DATA_ROOT")"
VENV="${VENV:-$DATA_ROOT/.venv}"
VENV="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$VENV")"
mkdir -p "$DATA_ROOT"/{input,logs,outputs,runs}

PY_MAJOR="$($PYTHON_BIN -c 'import sys; print(sys.version_info.major)' 2>/dev/null || true)"
PY_MINOR="$($PYTHON_BIN -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || true)"
if [[ "$PY_MAJOR" != 3 || -z "$PY_MINOR" || "$PY_MINOR" -lt 10 ]]; then
  echo "Python 3.10+ is required; found $($PYTHON_BIN --version 2>&1)" >&2; exit 2
fi

if [[ ! -f "$(dirname "$0")/../requirements-server.txt" ]]; then
  echo "Missing requirements-server.txt in repository root" >&2; exit 2
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  VENV_ARGS=()
  if (( SYSTEM_SITE )); then
    VENV_ARGS+=(--system-site-packages)
  fi
  "$PYTHON_BIN" -m venv "${VENV_ARGS[@]}" "$VENV"
elif (( SYSTEM_SITE )) && ! grep -q 'include-system-site-packages = true' "$VENV/pyvenv.cfg" 2>/dev/null; then
  echo "Existing venv does not expose system site packages; recreate it or choose a new --venv when using --system-site-packages." >&2
  exit 2
fi
VENV_PY="$VENV/bin/python"
if (( ! NO_INSTALL )); then
  "$VENV_PY" -m pip install --upgrade pip wheel
fi
if [[ ! -f "$DATA_ROOT/requirements-server-no-torch.txt" || ! -s "$DATA_ROOT/requirements-server-no-torch.txt" ]]; then
  grep -vE '^(torch|#.*torch)' "$(dirname "$0")/../requirements-server.txt" > "$DATA_ROOT/requirements-server-no-torch.txt"
fi
if (( ! NO_INSTALL )); then
  # Install server packages without torch. A CUDA-enabled torch is expected from
  # the base interpreter (system-site-packages) or must be installed separately.
  "$VENV_PY" -m pip install -r "$DATA_ROOT/requirements-server-no-torch.txt"
fi

if ! "$VENV_PY" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Torch with working CUDA is not available in the selected virtualenv.
Install a PyTorch wheel matching the host NVIDIA driver/CUDA from:
  https://pytorch.org/get-started/locally/
Do not guess a CUDA wheel and do not replace a working system torch.
Example only (verify compatibility first):
  "$VENV/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
Then rerun setup and server_preflight.py.
EOF
  exit 2
fi

CMD=("$VENV_PY" "$(dirname "$0")/gpu_input_manifest.py")
if (( DO_DOWNLOAD )); then
  "${CMD[@]}" download --data-root "$DATA_ROOT" --hash "$HASH_MODE"
else
  "${CMD[@]}" verify --data-root "$DATA_ROOT" --hash "$HASH_MODE" || true
fi
"$VENV_PY" "$(dirname "$0")/server_preflight.py" --data-root "$DATA_ROOT" --strict
cat <<EOF

SETUP COMPLETE
Activate: source "$VENV/bin/activate"
Run control: bash scripts/run_gpu_control.sh --data-root "$DATA_ROOT"
Output directory: "$DATA_ROOT/outputs"
Run logs: "$DATA_ROOT/logs" and "$DATA_ROOT/runs"
EOF
