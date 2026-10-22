#!/usr/bin/env bash
# One-command compact GPU control runner for a private Linux server.
set -Eeuo pipefail
IFS=$'\n\t'
DATA_ROOT="${DATA_ROOT:-$HOME/name-that-llm-data}"
# Independent control defaults: no test-batch clustering and no assumed uniform prior.
VARIANTS=(--no-twin --no-balance)
while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --only-8b|--only-14b) VARIANTS+=("$1"); shift ;;
    --no-twin|--no-balance) VARIANTS+=("$1"); shift ;;
    --allow-transductive) VARIANTS=(--allow-transductive); shift ;;
    --help|-h) echo "Usage: bash scripts/run_gpu_control.sh --data-root PATH [--only-8b|--only-14b] [--allow-transductive]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
BRANCH_FLAGS=0
for flag in "${VARIANTS[@]}"; do
  case "$flag" in
    --only-8b|--only-14b) BRANCH_FLAGS=$((BRANCH_FLAGS + 1)) ;;
  esac
done
if (( BRANCH_FLAGS > 1 )); then
  echo "--only-8b and --only-14b are mutually exclusive" >&2
  exit 2
fi
if [[ " ${VARIANTS[*]} " == *" --allow-transductive "* ]]; then
  for flag in --no-twin --no-balance; do
    if [[ " ${VARIANTS[*]} " == *" $flag "* ]]; then
      echo "--allow-transductive cannot be combined with $flag" >&2
      exit 2
    fi
  done
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$DATA_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi
if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2; exit 2
fi
DATA_ROOT="$($PYTHON -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$DATA_ROOT")"
case "$DATA_ROOT/" in
  "$REPO_ROOT/"*) echo "DATA_ROOT must be outside the repository: $DATA_ROOT" >&2; exit 2 ;;
esac
INPUT_DIR="$DATA_ROOT/input"
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
RUN_ID="$(date -u +%Y%m%dT%H%M%S.%NZ)-${GIT_COMMIT}-$$-$RANDOM"
RUN_DIR="$DATA_ROOT/runs/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists; refusing to overwrite: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR" "$DATA_ROOT/outputs" "$DATA_ROOT/logs"
LOG="$RUN_DIR/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "run_id=$RUN_ID"
echo "git_commit=$GIT_COMMIT"
echo "data_root=<external>"
echo "input_dir=<external>"
echo "run_dir=<external>"
echo "variant=${VARIANTS[*]:-full}"

"$PYTHON" "$SCRIPT_DIR/gpu_input_manifest.py" verify --data-root "$DATA_ROOT" --hash metadata
"$PYTHON" "$SCRIPT_DIR/server_preflight.py" --data-root "$DATA_ROOT" --strict --json-out "$RUN_DIR/preflight.json"

OUTPUT="$RUN_DIR/submission.parquet"
METADATA="$RUN_DIR/run.json"
SCORES="$RUN_DIR/member_scores.npz"
CMD=("$PYTHON" "$SCRIPT_DIR/kaggle_best_two_stage_lora.py" --input-dir "$INPUT_DIR" --output "$OUTPUT" --metadata "$METADATA" --scores-output "$SCORES")
CMD+=("${VARIANTS[@]}")
printf 'command=<redacted>; args='; printf '%q ' "${CMD[0]}" "${CMD[1]}"; echo
printf '%s\n' "${CMD[0]}" "<private-input-and-output-arguments-redacted>" > "$RUN_DIR/command.txt"
"${CMD[@]}"

"$PYTHON" - "$OUTPUT" "$METADATA" "$RUN_DIR" <<'PY'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1]); meta = pathlib.Path(sys.argv[2]); run = pathlib.Path(sys.argv[3])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
record = {
    'output_recorded': True, 'output_sha256': sha(out),
    'metadata_recorded': True, 'metadata_sha256': sha(meta),
    'run_dir_recorded': True, 'status': 'generated'
}
(run/'output_record.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))
PY

echo "DONE: inspect $OUTPUT and submit it manually or with the documented Kaggle command."
echo "This runner has no mid-model resume; rerun the same command after fixing a failure."
