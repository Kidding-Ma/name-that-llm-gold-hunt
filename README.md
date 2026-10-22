# Name That LLM — Gold Hunt Release

> GPU-ready handoff repository for the Kaggle **Name That LLM** competition.
>
> 目标：把可复现的 CPU 基线、已核对来源和输入的公开 GPU 起点、实验结论和交接提示词集中到一个小型公开仓库，交给有 GPU 的同学或 AI 继续冲击金牌。

## 重要事实

- Competition: [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)
- Official metadata endpoint: [CompetitionService/GetCompetition](https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionName=name-that-llm), fetched 2026-10-22 Asia/Shanghai. It reports the task description, maximized **Balanced Accuracy Score**, required `submission.parquet`, kernel-only submissions, a 2027-01-14 UTC deadline, 5 maximum daily submissions, and 540-minute CPU/GPU limits. Re-fetch before a final submission.
- Local snapshot schema: `train.csv` has `id,prompt_id,text,model`; `test.csv` has `id,text`.
- Local train snapshot: 2,699 rows, 150 prompts, 18 model labels.
- Local `test.csv`: only 3 visible examples; it is **not** the hidden test set.
- The full context, evidence labels, official links, and unresolved unknowns are in [`docs/COMPETITION_CONTEXT.md`](docs/COMPETITION_CONTEXT.md).
- Highest source-bound public score found in the audit: historical v4 `0.604` balanced accuracy, quoted by current prvsiyan source and bound to COMPLETE submission `57428928` / [scriptVersionId `360252286`](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286); historical cells/dependencies were not independently retrieved.
- User-reported current leaderboard leader: approximately `0.667`; exact code/version is not publicly attributed or independently verified here.

## New AI quickstart (SSH Linux GPU server)

Give the next AI this repository URL and SSH access to a Linux GPU server. Claude Code should read [`CLAUDE.md`](CLAUDE.md) first; [`AI_AGENT_PROMPT.md`](AI_AGENT_PROMPT.md) is the detailed handoff, not a substitute for the project instructions.

**Do not run an expensive download, GPU job, or Kaggle submission automatically.** First revalidate the current rules/data and explicitly record the input versions, licenses, and experiment plan. The commands below are an engineering smoke path only; the default public recipe includes transductive/test-batch operations and must be treated as a named hypothesis, not a neutral baseline.

```bash
git status --short
python scripts/release_audit.py
python -m unittest discover -s tests -v
```

For an authorized private Linux GPU environment, after reviewing the runbook and confirming Kaggle access:

```bash
export DATA_ROOT="$HOME/name-that-llm-data"
bash scripts/setup_gpu_env.sh --data-root "$DATA_ROOT" --download
source "$DATA_ROOT/.venv/bin/activate"
bash scripts/check_gpu_env.sh --data-root "$DATA_ROOT" --strict
# Run only after confirming the exact input versions and experiment record.
bash scripts/run_gpu_control.sh --data-root "$DATA_ROOT"
```

The setup downloads and verifies compact inputs, writes data/output/logs outside GitHub, and records a private manifest; it does not itself run inference. Run `run_gpu_control.sh` only after the explicit side-effect/owner approval gate in [`CLAUDE.md`](CLAUDE.md). Full details are in [`docs/REMOTE_GPU_RUNBOOK.md`](docs/REMOTE_GPU_RUNBOOK.md), [`docs/KAGGLE_DOWNLOADS.md`](docs/KAGGLE_DOWNLOADS.md), and [`docs/INPUT_LAYOUT.md`](docs/INPUT_LAYOUT.md).

**One unavoidable prerequisite:** GitHub SSH does not provide Kaggle access. The server must have a Kaggle API token (`KAGGLE_API_TOKEN` or `KAGGLE_CONFIG_DIR/kaggle.json`) and the account must have accepted the competition rules before missing inputs can be downloaded. If inputs are already staged, the token is not needed for verification. The repository never contains or prints credentials.

## What is included

- `scripts/`: Kaggle-ready CPU/GPU inference, input download/verification, server preflight, bootstrap, and one-command runner
- `docs/`: remote-server runbook, Kaggle download/auth instructions, exact input tree, troubleshooting, experiment log, evidence and public references
- `evidence/`: small, reproducible metrics and diagnostics only; no raw competition data
- `config/`: release manifest and compact-route input manifest
- `AI_AGENT_PROMPT.md`: concise instructions for the next AI
- `requirements-*.txt`: CPU, Kaggle GPU, and private-server dependency guidance

## What is intentionally excluded

The public repository does **not** contain:

- `train.csv`, `test.csv`, `sample_submission.parquet`, or the downloaded ZIP
- hidden-test data or labels
- model weights, LoRA adapters, or large public pools
- PDFs, extracted papers, bundled binary dependencies, OOF arrays, or local submissions
- credentials, Kaggle tokens, GitHub tokens, or machine-specific absolute paths

The person running this repository must attach the competition data and public model/checkpoint inputs in Kaggle or another GPU environment.

## Research tracks (neutral order)

### Track A — baseline ladder

1. Revalidate the current official rules, data page, sample submission, and input permissions.
2. Run schema/duplicate/provenance audits and freeze the validation protocol.
3. Establish a constant/sanity control and a simple CPU text baseline.
4. Compare one GPU representation at a time under the same folds and budget.
5. Only then evaluate external pools, pretrained checkpoints, transductive operations, or upstream notebooks as separately named hypotheses.

The repository packages public GPU routes as optional candidates, not conclusions. The compact script's default independent control is `--no-twin --no-balance`; `--allow-transductive` explicitly enables test-batch clustering and uniform-prior balancing and requires rule review plus pseudo-test stress tests. Do not submit any variant until the owner approves the run plan and the full-test/schema gates pass.

Upstream score claims and their source versions are recorded in [`docs/PUBLIC_GPU_ROUTES.md`](docs/PUBLIC_GPU_ROUTES.md). They are evidence for experiment prioritization only, never local results or guarantees.

Full steps: [`docs/GPU_RUNBOOK_ZH.md`](docs/GPU_RUNBOOK_ZH.md)

### Track B — CPU fallback and control

Use [`scripts/kaggle_submit.py`](scripts/kaggle_submit.py) first. It runs the verified style-aware four-view TF-IDF + LinearSVC recipe without a GPU.

A second CPU candidate is [`scripts/kaggle_submit_blend.py`](scripts/kaggle_submit_blend.py). It blends the style-aware model with the original char/word/style SVC. Its local fold-held-out diagnostic was about `0.393`, but this is not a leaderboard guarantee; test it as a separate Kaggle submission.

Full steps: [`docs/CPU_RUNBOOK_ZH.md`](docs/CPU_RUNBOOK_ZH.md)

## Evidence levels

- **A — local reproduction:** saved metrics from the supplied train snapshot and prompt-grouped CV.
- **B — public implementation:** public source/code inspected, not necessarily packaged or independently reproduced locally.
- **C — author report:** upstream notebook comments or leaderboard claims; useful for prioritization, not a guarantee.
- **D — our Kaggle result:** a recorded full hidden-test submission and its public/private score. No GPU result here has reached D yet.

Do not describe the public `0.604`, `0.600`, `0.592`, `0.55`, or the user-observed `0.667` as a locally reproduced score.

## Validation rules

Every new experiment must:

1. split by `prompt_id`, never by random rows;
2. fit vectorizers, scalers, classifiers, and blend weights on the training side only;
3. preserve the submission contract `id,model` and test ID order;
4. avoid hidden-test labels and manual test predictions;
5. record whether a result is local CV, public LB, private LB, or an upstream claim;
6. treat prompt clustering, centered twins, Sinkhorn, and any test-batch prior as transductive methods requiring pseudo-test stress tests.

## Repository handoff

Claude Code should read [`CLAUDE.md`](CLAUDE.md) first, then [`AI_AGENT_PROMPT.md`](AI_AGENT_PROMPT.md) and [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md). It must revalidate current facts and obtain owner approval before downloading, launching costly GPU work, saving a Kaggle version, or submitting.

## License and provenance

See [`NOTICE.md`](NOTICE.md). Third-party code, datasets, checkpoints, and upstream notebooks retain their original licenses and attribution requirements.
