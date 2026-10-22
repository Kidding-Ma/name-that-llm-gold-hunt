# Claude Code handoff: Name That LLM

This repository is a **research handoff**, not a claim that any particular model or public score is correct. The next agent should preserve independent judgment, test alternatives, and make every conclusion auditable.

## Mission

Research the Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) and pursue the best legal score that can be supported by evidence. A gold medal is an objective, not a result to assume. A higher local cross-validation score is not a leaderboard improvement until a full competition submission confirms it.

## How to reason without being biased by this handoff

1. Treat each statement as one of **fact**, **inference**, or **hypothesis**.
2. A fact must have an official URL, a local file/hash, a reproducible command, or a versioned upstream source.
3. Keep unknowns unknown. Do not fill missing competition rules, hidden-test construction, class quotas, or leaderboard methods from intuition.
4. Public notebooks and reported scores are candidate evidence, not ground truth and not our own results.
5. The packaged LoRA and TF-IDF routes are controls/candidates. You may reject them after a fair comparison; do not treat the recommended order as an architectural conclusion.
6. When a result conflicts with the current context, record the conflict and obtain a newer source or run rather than silently choosing the more favorable number.

## Read first

Read these files before changing code or choosing a model:

1. [`README.md`](README.md)
2. [`docs/COMPETITION_CONTEXT.md`](docs/COMPETITION_CONTEXT.md)
3. [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)
4. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
5. [`docs/PUBLIC_GPU_ROUTES.md`](docs/PUBLIC_GPU_ROUTES.md)
6. [`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md)
7. [`AI_AGENT_PROMPT.md`](AI_AGENT_PROMPT.md)
8. [`docs/REMOTE_GPU_RUNBOOK.md`](docs/REMOTE_GPU_RUNBOOK.md) when using a private GPU server

Start with:

```bash
git status --short
python scripts/release_audit.py
python -m unittest discover -s tests -v
```

## Verified competition context at handoff

The official competition metadata endpoint was fetched on 2026-10-22 (Asia/Shanghai):

- competition: `name-that-llm`, title: **Name That LLM**;
- task description: identify the model behind a text;
- metric: **Balanced Accuracy Score**, maximized, public evaluation algorithm;
- submission filename: `submission.parquet`;
- submissions are **kernel-only** (`onlyAllowKernelSubmissions=true`);
- maximum daily submissions: 5; scored submissions: 2;
- CPU and GPU runtime limits: 540 minutes each;
- competition deadline: `2027-01-14T23:59:00.817Z`;
- rules are required; maximum team size is 5; new entrants/team merger deadlines are recorded in the context document;
- competition metadata license: CC BY-SA 4.0.

These are time-sensitive metadata facts. Re-fetch the official endpoint before a final submission. The official endpoint does **not** by itself establish the full train/test schema or hidden-test distribution.

## Current local evidence, not universal truth

The private snapshot used to prepare this repository reports 2,699 training rows, 150 prompt groups, 18 model labels, columns `id,prompt_id,text,model`, and a visible three-row local test file. The local test is a smoke fixture, not a leaderboard test. Its hash and other provenance are documented in [`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md).

Reported public routes include a two-stage Qwen/LoRA route around 0.592 public balanced accuracy and a historical source claim around 0.604. They are explicitly labeled as upstream evidence in this repository. The user-observed approximately 0.667 leader is not mapped to verified public code here.

## Neutral research protocol

Before expensive GPU work:

1. Re-acquire the competition inputs from Kaggle and record the attached versions and hashes privately.
2. Audit schema, nulls, duplicates, exact/near-duplicate text, label counts, text lengths, and prompt grouping without modifying raw data.
3. Freeze a prompt-aware validation protocol supported by the data. Do not use random-row splits as the primary estimate when prompt/topic leakage is plausible.
4. Establish sanity and simple text baselines before complex pretrained models.
5. Compare one intervention at a time under the same folds, seeds, data boundary, and compute budget.
6. Fit vectorizers, scalers, heads, calibration, and blend weights on the training side only. Treat test-time clustering, centering, balancing, pseudo-labeling, and quotas as separate hypotheses requiring rule review and pseudo-test stress tests.
7. Save OOF predictions and per-fold metrics for serious candidates. Log failed and negative experiments.
8. Generate submissions only through a checked pipeline. Verify exact filename/schema, row count, ID order, legal labels, and prediction distribution.
9. Save the submission hash, notebook version, input versions, public score, private score, runtime, and change from the previous run. Never call an upstream number our result.

## Run records

Every meaningful run should record:

```text
run_id, status, hypothesis, command, git commit, config/code revision,
data and fold hashes, input identifiers/versions, license/rule check,
Python/CUDA/PyTorch/Transformers versions, GPU and VRAM, precision,
batch/accumulation, wall time, peak memory, per-fold metrics,
submission path/hash, public score, private score, failure reason, next action
```

Keep run artifacts, raw data, checkpoints, adapters, public pools, hidden-test material, and credentials outside this public repository unless their license and the release policy explicitly permit inclusion.

## GPU entry points and side-effect gate

On Kaggle, attach the competition input and any permitted public inputs, then review the source before running `scripts/kaggle_best_two_stage_lora.py`. On a private Linux GPU server, follow [`docs/REMOTE_GPU_RUNBOOK.md`](docs/REMOTE_GPU_RUNBOOK.md). Do not assume GitHub authentication grants Kaggle authentication. Do not place tokens in chat, shell history, logs, or Git.

Before any download, model load, multi-minute GPU job, Kaggle notebook save, or submission, present a short plan containing: expected runtime/GPU cost, input handles and versions, license/rule status, data unknowns, command, output location, and stop conditions. Wait for explicit owner approval in the current interaction. Read-only local checks and dependency-free tests may run without that approval.

The current packaged GPU route is a **transductive candidate**, not a neutral baseline: centered twins use test-batch clustering and Sinkhorn imposes a uniform class prior. Keep those operations disabled for the initial independent control, or run them as a separately named hypothesis only after rule review and repeated pseudo-test stress tests. The agent is free to test another route if evidence, rules, or the resource budget support it. Any deviation must be named and recorded rather than presented as a reproduction of an upstream notebook.

## Stop conditions

Stop and report a concrete blocker when:

- Kaggle rules or identity/competition access is missing;
- required input versions, licenses, or model provenance cannot be established;
- the full hidden-test input is unavailable;
- the run would require hidden labels, manual labeling, credential exposure, or an unverified test quota;
- a claimed reproduction lacks a required checkpoint or source version;
- the output fails schema/order/label checks.

A blocked run is useful evidence. Do not work around a blocker by silently changing the task.
