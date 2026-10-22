# Experiment log template

Copy this file outside GitHub for each server or Kaggle run. Never put tokens,
private prompts, raw data or weights in the record.

```yaml
status: PASS  # or BLOCKED
run_id:
started_utc:
finished_utc:
repo_url:
commit:
route: nishant_compact_control
variant: independent_control  # default: --no-twin --no-balance; alternatives: --only-8b, --only-14b, --allow-transductive
command:
server:
python:
venv:
nvidia_driver:
cuda_runtime:
torch:
transformers:
accelerate:
gpu_names: []
gpu_count:
ram_gb:
free_disk_gb:

inputs:
  competition: name-that-llm
  dataset: nishantdahal/ntl-two-stage-lora
  models:
    - qwen-lm/qwen-3/Transformers/8b-base/1
    - qwen-lm/qwen-3/Transformers/14b-base/1
  attached_versions:
  manifest_path:
  manifest_sha256:
  input_tree_sha256:
  licenses_checked: false

artifacts:
  run_dir:
  log_path:
  preflight_path:
  submission_path:
  submission_sha256:
  submission_rows:
  metadata_path:
  scores_path:

scores:
  local_grouped_cv:
  public_lb:
  private_lb:

change_from_previous:
blocker:
next_action:
notes:
```

The compact upstream numbers (`0.592` public LB / `0.593` grouped CV) are
reference claims, not values to fill into `public_lb` unless this run actually
receives that score. The current repository has no D-tier GPU result yet.
