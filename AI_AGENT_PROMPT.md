# GPU handoff prompt — Name That LLM

You are taking over the public repository for **Name That LLM**. Your job is to obtain a real, legal Kaggle score and improve it with evidence. Do not optimize for a plausible local CV number or claim an upstream score as your own.

## Read these first

Read these files in this order:

1. [`README.md`](README.md)
2. [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)
3. [`docs/GPU_RUNBOOK_ZH.md`](docs/GPU_RUNBOOK_ZH.md)
4. [`docs/PUBLIC_GPU_ROUTES.md`](docs/PUBLIC_GPU_ROUTES.md)
5. [`scripts/kaggle_best_two_stage_lora.py`](scripts/kaggle_best_two_stage_lora.py)
6. [`scripts/prvsiyan_route_preflight.py`](scripts/prvsiyan_route_preflight.py)

Read `evidence/metrics/` only when you need the CPU controls. Do not assume competition data, checkpoints or weights are in this repository.

## Facts that control your decisions

- Task: classify 18 model labels; metric: balanced accuracy.
- Submission: `submission.parquet` with exactly `id,model`, in test-row order.
- The visible local test has only 3 rows. It is a schema check, not the leaderboard test.
- Validation must split by `prompt_id`; random-row CV is not acceptable as the main estimate.
- Best public **source claim found**: historical prvsiyan v4, `0.604`, linked to COMPLETE submission `57428928` and [`scriptVersionId=360252286`](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286). This is not independently reproduced here.
- Current prvsiyan v10 is explicitly an **unscored** candidate: it disables historical branch B because its Lightning checkpoint is unavailable. Do not call v10, a branch-A port, or an ATLAS/Gaussian remix a `0.604` reproduction.
- Nishant's compact two-stage LoRA route reports `0.592` public LB / `0.593` grouped CV upstream. It is the first executable control in this repository.
- The user-reported `0.667` leader has no verified public code, version or submission mapping.

## Required execution order

### 1. Establish a GPU and input baseline

1. Prefer Kaggle GPU before renting hardware; use 2×T4 for the compact route when available.
2. Attach the `name-that-llm` competition data, `nishantdahal/ntl-two-stage-lora`, and Qwen3 8B/14B Transformers inputs.
3. Run [`scripts/prvsiyan_route_preflight.py`](scripts/prvsiyan_route_preflight.py) only when auditing the larger public route. It checks current mounts, CUDA, CSV structure, adapter markers and Qwen dimensions. It does **not** prove historical v4 completeness, licenses, model contents or score parity.
4. If there is no CUDA, run only the CPU control; do not claim GPU progress.

### 2. Run the compact control unchanged

Run [`scripts/kaggle_best_two_stage_lora.py`](scripts/kaggle_best_two_stage_lora.py) with the default command first. Do not change weights or several switches at once.

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
