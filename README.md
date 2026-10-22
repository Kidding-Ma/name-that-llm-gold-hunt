# Name That LLM — Gold Hunt Release

> GPU-ready handoff repository for the Kaggle **Name That LLM** competition.
>
> 目标：把可复现的 CPU 基线、已核对来源和输入的公开 GPU 起点、实验结论和交接提示词集中到一个小型公开仓库，交给有 GPU 的同学或 AI 继续冲击金牌。

## 重要事实

- Competition: [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)
- Metric: 18-class **Balanced Accuracy Score**
- Submission: `submission.parquet` with columns `id,model`
- Local train snapshot: 2,699 rows, 150 prompts, 18 model labels
- Local `test.csv`: only 3 visible examples; it is **not** the hidden test set
- User-reported current leaderboard leader: approximately `0.667`
- The exact code behind that score has not been publicly attributed or independently verified

## What is included

- `scripts/`: Kaggle-ready CPU and GPU inference scripts, preflight checks, and local diagnostics
- `docs/`: execution runbooks, evidence levels, research status, and public references
- `evidence/`: small, reproducible metrics and diagnostics only; no raw competition data
- `config/`: release manifest and public-safety metadata
- `AI_AGENT_PROMPT.md`: copy this prompt to another AI together with the repository
- `requirements-*.txt`: environment guidance; Kaggle already contains most CPU packages

## What is intentionally excluded

The public repository does **not** contain:

- `train.csv`, `test.csv`, `sample_submission.parquet`, or the downloaded ZIP
- hidden-test data or labels
- model weights, LoRA adapters, or large public pools
- PDFs, extracted papers, bundled binary dependencies, OOF arrays, or local submissions
- credentials, Kaggle tokens, GitHub tokens, or machine-specific absolute paths

The person running this repository must attach the competition data and public model/checkpoint inputs in Kaggle or another GPU environment.

## Recommended attack order

### Track A — fastest GPU path

Start with the directly packaged public recipe:

1. Open a Kaggle Code Notebook.
2. Attach the competition data.
3. Attach dataset `nishantdahal/ntl-two-stage-lora`.
4. Attach Qwen3 8B Base and Qwen3 14B Base Transformers model inputs.
5. Select preferably **2×T4**.
6. Paste/run [`scripts/kaggle_best_two_stage_lora.py`](scripts/kaggle_best_two_stage_lora.py).
7. Save the version and submit the generated `/kaggle/working/submission.parquet`.

The public author reported grouped CV `0.593` and public LB `0.592`; these are upstream claims, not an independent result from this repository. The script deliberately avoids hidden labels, prompt IDs, and Hungarian assignment.

**Gold-medal workflow:** reproduce this recipe first, submit it, record the real score, then search for newer public versions or add one independently validated model family at a time. A higher local CV without a better Kaggle submission is not progress.

Full steps: [`docs/GPU_RUNBOOK_ZH.md`](docs/GPU_RUNBOOK_ZH.md)

### Track B — CPU fallback and control

Use [`scripts/kaggle_submit.py`](scripts/kaggle_submit.py) first. It runs the verified style-aware four-view TF-IDF + LinearSVC recipe without a GPU.

A second CPU candidate is [`scripts/kaggle_submit_blend.py`](scripts/kaggle_submit_blend.py). It blends the style-aware model with the original char/word/style SVC. Its local fold-held-out diagnostic was about `0.393`, but this is not a leaderboard guarantee; test it as a separate Kaggle submission.

Full steps: [`docs/CPU_RUNBOOK_ZH.md`](docs/CPU_RUNBOOK_ZH.md)

## Evidence levels

- **A — local reproduction:** saved metrics from the supplied train snapshot and prompt-grouped CV.
- **B — public implementation:** code or checkpoint is publicly referenced and packaged here, but not independently reproduced locally.
- **C — author report:** upstream notebook comments or leaderboard claims; useful for prioritization, not a guarantee.

Do not describe an upstream `0.592`, `0.55`, or the user-observed `0.667` as a locally reproduced score.

## Validation rules

Every new experiment must:

1. split by `prompt_id`, never by random rows;
2. fit vectorizers, scalers, classifiers, and blend weights on the training side only;
3. preserve the submission contract `id,model` and test ID order;
4. avoid hidden-test labels and manual test predictions;
5. record whether a result is local CV, public LB, private LB, or an upstream claim;
6. treat prompt clustering, centered twins, Sinkhorn, and any test-batch prior as transductive methods requiring pseudo-test stress tests.

## Repository handoff

The next AI should read [`AI_AGENT_PROMPT.md`](AI_AGENT_PROMPT.md) first, then [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md), and execute the GPU runbook before inventing a new architecture.

## License and provenance

See [`NOTICE.md`](NOTICE.md). Third-party code, datasets, checkpoints, and upstream notebooks retain their original licenses and attribution requirements.
