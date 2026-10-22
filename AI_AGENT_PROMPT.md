# GPU handoff prompt — Name That LLM

You are taking over the public repository for **Name That LLM**. Your job is to obtain a real, legal Kaggle score and improve it with evidence. Do not optimize for a plausible local CV number or claim an upstream score as your own.

## Read these first

Read these files in this order:

1. [`README.md`](README.md)
2. [`docs/REMOTE_GPU_RUNBOOK.md`](docs/REMOTE_GPU_RUNBOOK.md)
3. [`docs/KAGGLE_DOWNLOADS.md`](docs/KAGGLE_DOWNLOADS.md)
4. [`docs/INPUT_LAYOUT.md`](docs/INPUT_LAYOUT.md)
5. [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
6. [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)
7. [`docs/PUBLIC_GPU_ROUTES.md`](docs/PUBLIC_GPU_ROUTES.md)
8. [`scripts/run_gpu_control.sh`](scripts/run_gpu_control.sh)
9. [`scripts/kaggle_best_two_stage_lora.py`](scripts/kaggle_best_two_stage_lora.py)
10. [`docs/COMPETITION_CONTEXT.md`](docs/COMPETITION_CONTEXT.md)
11. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
12. [`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md)

Read `evidence/metrics/` only when you need the CPU controls. Do not assume competition data, checkpoints or weights are in this repository.

## Facts that control your decisions

- The official metadata endpoint reports the task description as “Identify the model behind the text” and the maximized evaluation algorithm as Balanced Accuracy Score. Re-fetch [`docs/COMPETITION_CONTEXT.md`](docs/COMPETITION_CONTEXT.md)'s official endpoint before final decisions.
- The local snapshot has 18 model labels and `train.csv`/`test.csv` schemas described in the context document; the official endpoint alone does not verify that the current hidden test has the same class count or exact schema.
- The local visible test has only 3 rows. It is a schema check, not the leaderboard test.
- Local validation should split by `prompt_id` when that field is present and prompt/topic leakage is plausible; confirm the current data before treating this as a competition rule.
- The official metadata says `submission.parquet`, row ID column `id`, and kernel-only submissions. The local control script additionally emits `model`; verify the current sample submission and rules before assuming the full column contract.
- Best public **source claim found**: historical prvsiyan v4, `0.604`, linked to COMPLETE submission `57428928` and [`scriptVersionId=360252286`](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286). This is not independently reproduced here.
- Current prvsiyan v10 is explicitly an **unscored** candidate: it disables historical branch B because its Lightning checkpoint is unavailable. Do not call v10, a branch-A port, or an ATLAS/Gaussian remix a `0.604` reproduction.
- Nishant's compact two-stage LoRA route reports `0.592` public LB / `0.593` grouped CV upstream. It is the first executable control in this repository.
- The user-reported `0.667` leader has no verified public code, version or submission mapping.

## Required execution order

### 0. Revalidate before any costly or external action

Do not download large inputs, launch a GPU run, create a Kaggle notebook version, or submit a score merely because this repository contains a command. First:

1. read `CLAUDE.md` and [`docs/COMPETITION_CONTEXT.md`](docs/COMPETITION_CONTEXT.md);
2. fetch the current official competition metadata, rules, data page, and sample submission, then record changes;
3. confirm competition membership, identity requirements, and rule acceptance;
4. record exact input handles/versions, licenses, and whether external pools/checkpoints are allowed;
5. run a CPU/schema smoke test and create a dated experiment record;
6. ask the owner for explicit approval before a long GPU run or a real submission.

The compact GPU route is **transductive**: centered twins use test-batch text clusters and Sinkhorn uses a uniform test-batch class prior. It is not a neutral independent baseline. Keep it disabled while establishing controls, or run it only as a separately named, rule-checked hypothesis after pseudo-test stress tests.

### 1. Establish a GPU and input baseline

On a private Linux GPU server, use this sequence only after the read-only revalidation and explicit owner approval described above:

```bash
export DATA_ROOT="$HOME/name-that-llm-data"
bash scripts/setup_gpu_env.sh --data-root "$DATA_ROOT" --download
source "$DATA_ROOT/.venv/bin/activate"
bash scripts/check_gpu_env.sh --data-root "$DATA_ROOT" --strict
bash scripts/run_gpu_control.sh --data-root "$DATA_ROOT"
```

If Kaggle credentials are absent, do not ask for architecture choices: report the exact missing prerequisite (`KAGGLE_API_TOKEN` or `KAGGLE_CONFIG_DIR/kaggle.json`) and wait. If inputs are already staged, run setup without `--download`. Do not put the repository URL, credentials, or private input paths in logs or chat.

Prefer Kaggle GPU before renting hardware; use 2×T4 for the compact route when available. On Kaggle, attach the competition, `nishantdahal/ntl-two-stage-lora`, and Qwen3 8B/14B inputs, then run the same Python script from the Kaggle runbook.

Run [`scripts/prvsiyan_route_preflight.py`](scripts/prvsiyan_route_preflight.py) only when auditing the larger public route. It checks current mounts, CUDA, CSV structure, adapter markers and Qwen dimensions. It does **not** prove historical v4 completeness, licenses, model contents or score parity.

### 2. Run the compact control unchanged

Run [`scripts/kaggle_best_two_stage_lora.py`](scripts/kaggle_best_two_stage_lora.py) with the independent-control switches `--no-twin --no-balance` first (the remote wrapper defaults to them). Do not change weights or several switches at once. The upstream transductive recipe is only a separately named `--allow-transductive` run after rule review.

Before submitting, verify:

- the test row count is the full Kaggle hidden-test input, not the local 3-row file;
- columns are exactly `id,model`;
- IDs are unique and remain in test order;
- every prediction is one of the 18 labels;
- prediction counts are not accidentally concentrated in one class;
- no hidden labels, prompt IDs, Hungarian matching or assumed class quotas were used.

Save a Kaggle notebook version, submit the file, and record the actual public/private score. For diagnostics, use separate notebook versions:

```text
--only-8b
--only-14b
--no-twin
--no-balance
```

The optional `--scores-output path.npz` is for private aligned member/logit analysis only. It is not OOF, not a submission, and must not be committed.

### 3. Only after the control, inspect the 0.604 route

Open the native [prvsiyan v4 source](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286) and inspect its actual cells and inputs before using Copy & Edit.

Proceed as a v4 reproduction only if all of the following are true:

- historical branch B's exact Lightning checkpoint is available and legally usable;
- every original dataset/model/kernel-output version is identified;
- licenses and competition-rule compatibility are recorded;
- the native source runs on the full hidden test;
- a new Kaggle submission receives a score.

If branch B or another dependency is unavailable, stop calling the run v4. Run Joseph's reported `0.596` route or Kirill's reported `0.600` route as separately named candidates, or label a new branch-A/Gaussian/ATLAS experiment as unscored.

### 4. Improve only after a real baseline score

Add one change at a time and save private member logits. Use prompt-grouped outer validation and fit blend weights/calibration only on the training side. Reasonable candidates are:

- an independently trained embedding or LoRA member;
- public-pool filtering and prompt-overlap checks;
- family-specific gates with prompt-balanced training data;
- centered twins or Sinkhorn only after pseudo-test stress tests;
- calibrated logit blending selected with nested grouped validation.

Do not repeatedly tune weights on one OOF matrix. A higher local CV is not a leaderboard improvement until a new full-test submission confirms it.

## Hard restrictions

- Never add competition data, hidden labels, weights, adapters, pools, tokens or raw notebook blobs to this repository.
- Never manually label hidden test rows.
- Never assume every hidden prompt contains one row per model.
- Never use Hungarian assignment or hidden-test quotas by default.
- Never present `0.604`, `0.600`, `0.592` or `0.667` as your own result without a matching run record.
- Do not delete or overwrite the caller's private source workspace.

## Required run report

For every Kaggle run, report:

```text
status: PASS / BLOCKED
code commit or notebook version:
command/variant:
GPU and runtime:
input identifiers and versions:
license/attribution check:
train/test/pool hashes where possible:
submission hash and row count:
public score:
private score:
change from previous run:
exact blocker and next action (if blocked):
```

Update [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) only with checkable facts. A task is complete only when a legal full-test submission, its score, reproducible inputs/code and the evidence record all exist.
