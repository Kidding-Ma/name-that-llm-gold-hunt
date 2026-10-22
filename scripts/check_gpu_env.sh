#!/usr/bin/env bash
# Check private Linux GPU server and compact-route inputs without exposing secrets.
set -Eeuo pipefail
IFS=$'\n\t'
DATA_ROOT="${DATA_ROOT:-$HOME/name-that-llm-data}"
STRICT=0
while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    -h|--help) echo "Usage: bash scripts/check_gpu_env.sh [--data-root PATH] [--strict]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON" >&2; exit 2
fi
DATA_ROOT="$($PYTHON -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$DATA_ROOT")"
mkdir -p "$DATA_ROOT/logs"
LOG="$DATA_ROOT/logs/check-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "data_root=$DATA_ROOT"
echo "python=$($PYTHON --version 2>&1)"
echo "host=$(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader,nounits
else
  echo "MISSING nvidia-smi"
fi

set +e
"$PYTHON" "$SCRIPT_DIR/server_preflight.py" --data-root "$DATA_ROOT" --strict --json-out "$DATA_ROOT/logs/preflight.json"
STATUS=$?
set -e
echo "preflight_exit=$STATUS"
echo "check_log=$LOG"
if (( STRICT )) && (( STATUS != 0 )); then
  echo "STRICT CHECK FAILED; follow the printed remediation before running the model." >&2
  exit "$STATUS"
fi
exit 0
