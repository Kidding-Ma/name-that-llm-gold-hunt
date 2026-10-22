# Prompt for the next AI / GPU algorithm engineer

You are a senior Kaggle algorithm engineer taking over a public repository for the competition **Name That LLM**. The user's objective is to win a gold medal, not merely to produce a plausible local CV number.

## Start here

Read these files in order:

1. `README.md`
2. `docs/RESEARCH_STATUS.md`
3. `docs/GPU_RUNBOOK_ZH.md`
4. `docs/PUBLIC_SOURCES.md`
5. `docs/PUBLIC_GPU_ROUTES.md`
6. `evidence/metrics/` JSON files
7. the selected scripts in `scripts/`

Do not assume raw data is in this repository. Attach it from Kaggle or a private/local path.

## Competition facts

- Competition: https://www.kaggle.com/competitions/name-that-llm
- Metric: 18-class balanced accuracy
- Required output: `submission.parquet` with exactly `id,model`
- Training snapshot: 2,699 labeled rows, 150 prompts, 18 labels
- The same prompt appears with multiple model responses. Validation must be grouped by `prompt_id`.
- Local test examples are only three rows and are not the hidden test.
- User observed a current leaderboard leader near `0.667`; no public artifact has been verified as the source of that exact score.

## Current evidence

### Verified local CPU result (Evidence A)

The pure-style CPU control is four-view TF-IDF + `LinearSVC(C=0.5)`:

- raw char n-grams 3–5, max 250k;
- word/punctuation n-grams 1–3, max 180k, weight 0.7;
- shape char n-grams 2–6, max 60k, weight 0.5;
- function-word/punctuation n-grams 1–3, max 50k, weight 0.5;
- `min_df=2`, sublinear TF, case/Unicode/whitespace/punctuation preserved;
- 5-fold prompt-grouped OOF balanced accuracy `0.384484` (the shipped numeric-style control reaches `0.388936`; see the evidence ledger).

A conservative CPU blend probe is around `0.393`, but it was not a fully nested experiment and must be tested as a separate Kaggle submission.

### Highest credible public score target (Evidence B/C)

The strongest public source found in the audit is [prvsiyan's historical `The Prompt Is Not the Author` v4](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286). Current v10 source text binds the historical claim `0.604` balanced accuracy to COMPLETE submission `57428928`. This is author/source evidence, not an independent reproduction. The current v10 source also says it is an unscored candidate with the historical branch-B Lightning checkpoint unavailable; it is not v4 parity. Read `docs/PUBLIC_GPU_ROUTES.md` before attempting it.

Do not confuse the public notebook's current v10 page with a guarantee that current code produces the historical v4 score. First reproduce the compact Nishant control, then inspect the native v4 source and verify that branch B and every exact input/version can be mounted legally. If not, use a clearly named new candidate or the Joseph/Kirill routes.

### Verified public GPU starting point (Evidence B/C)

`scripts/kaggle_best_two_stage_lora.py` packages the public two-stage LoRA-on-public-pool recipe:

- official train plus a public pool of roughly 34,940 passages;
- style-aware SVC;
- Qwen3 8B and 14B LoRA adapters;
- centered twins on test content clusters of size 2–18;
- optional Sinkhorn class balancing;
- no prompt IDs, hidden labels, or Hungarian assignment.

The upstream notebook reports grouped CV `0.593` and public LB `0.592`. Reproduce it before modifying it. Prefer 2×T4. Use the preflight script if inputs are unclear.

## Immediate mission

### Phase 1 — reproduce and submit

1. Create/copy a Kaggle Code Notebook.
2. Attach `name-that-llm` competition data.
3. Attach `nishantdahal/ntl-two-stage-lora`.
4. Attach Qwen3 8B Base and Qwen3 14B Base Transformers models.
5. Use preferably 2×T4, internet off if all inputs are attached.
6. Run `scripts/kaggle_best_two_stage_lora.py`.
7. Verify test row count, ID order, legal labels, and prediction counts.
8. Save a version and submit.
9. Record: notebook version, runtime, GPU type, public score, private score if visible, and output hash.

If the full recipe fails, run these diagnostic variants in separate versions:

- `--only-8b`
- `--only-14b`
- `--no-twin`
- `--no-balance`

Do not silently change several components at once.

### Phase 2 — reproduce the 0.604 public target

If the compact 0.592 control is working and the score remains far below `0.667`:

1. Read `docs/PUBLIC_GPU_ROUTES.md` and run `scripts/prvsiyan_route_preflight.py`.
2. Open the native prvsiyan v4 URL and inspect actual cells. Verify historical branch B's Lightning checkpoint and every original dependency before using Copy & Edit; the current four datasets plus Qwen3 8B/14B list does not establish v4 completeness.
3. Record exact dataset/model/kernel versions, license/attribution, GPU, runtime and output hash before submitting.
4. Do not claim `0.604` reproduction unless a new full hidden-test submission is generated and scored.
5. If the route is too large or inputs are unavailable, try Joseph's 0.596 or Kirill's 0.600 public lineage only as separately logged candidates.
6. Compare prediction distributions and per-class recalls, especially Claude, GLM, GPT variants, Kimi, and Qwen.

The current live/user-reported `0.667` leader still has no public notebook/version/submission binding in this audit.

### Phase 3 — controlled improvements

Only add one family at a time and save logits:

1. better public pool filtering and deduplication;
2. a frozen Qwen/DeBERTa/ELECTRA embedding head;
3. a second independent LoRA adapter;
4. family-specific gates trained only on data with topic-balanced prompts;
5. centered twins with pseudo-test stress tests;
6. calibrated or constrained postprocessing only if it is validated on many pseudo-test draws.

For every candidate, use prompt-grouped outer folds and inner selection of hyperparameters/weights. Do not optimize on one OOF matrix repeatedly.

## Red lines

- Never upload or commit competition data, hidden data, model weights, adapters, or tokens to this repository.
- Never use hidden test labels or manually label test rows.
- Never use random-row CV as the main score.
- Never force one of each model per hidden prompt unless the competition contract proves that assumption and pseudo-test simulations support it.
- Never treat a binary AI detector metric as 18-class identity accuracy.
- Never report an upstream claim as a local reproduction.
- Do not delete or overwrite source data in the caller's workspace.

## Deliverables expected from your work

For each run, save a small metadata JSON outside the public repository or in a private artifact store containing:

- code commit/version;
- input dataset/model/checkpoint names and versions;
- SHA-256 hashes where possible;
- GPU type and runtime;
- fold protocol and score;
- submission file hash and row count;
- public/private leaderboard score;
- exact changes from the previous run.

Update `docs/RESEARCH_STATUS.md` only with facts that can be checked. If you find a stronger public solution, add its URL and input list to `docs/PUBLIC_SOURCES.md` and add a short reproduction note.

## Definition of success

The job is not complete when code runs. It is complete when:

1. a legal full-test submission is generated;
2. the score is recorded;
3. the method is reproducible by another engineer;
4. improvements are supported by held-out evidence or repeated leaderboard tests;
5. all public-repository changes remain free of data, credentials, and large binaries.
