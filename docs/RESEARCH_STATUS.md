# Research status and evidence ledger

更新时间：2026-10-21（工作区快照）

## Competition

- Competition: [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)
- Task: identify which of 18 LLM labels generated a text response.
- Metric: 18-class balanced accuracy.
- Submission: `submission.parquet`, exactly `id,model`.
- Local train snapshot: 2,699 rows, 150 prompts; 149 prompts have 18 rows and one has 17.
- Local visible test: 3 rows only. It is a schema smoke test, not the hidden leaderboard test.
- Train SHA-256 from the private snapshot: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.

## Score ledger

| Method | Score | Evidence | Interpretation |
|---|---:|---|---|
| Original char+word+style LinearSVC | 0.370508 grouped OOF | A: local run | Simple baseline; close to the user's reported ~0.365 submission |
| Official word+char LogisticRegression | 0.314479 grouped OOF | A: local run | Starter control |
| Style-aware four-view LinearSVC(C=0.5) | 0.384484 grouped OOF | A: local run | Best verified pure style-aware CPU recipe in the private snapshot |
| Style-aware + 25 numeric style statistics | 0.388936 grouped OOF | A: local run/script evidence | Stronger CPU submission script; still validate on the real hidden test |
| CPU style/original row-softmax blend | ~0.39337 fold-held-out probe | A/B: diagnostic only | Candidate; not nested retraining and not a leaderboard guarantee |
| Qwen3-1.7B embedding blend | ~0.431 in upstream notes | C: upstream notebook claim | GPU starting point; not independently reproduced here |
| Two-stage Qwen3 8B/14B LoRA + public pool | 0.593 grouped CV / 0.592 public LB in upstream notebook | B/C: public code and author report | Highest-priority reproducible public GPU recipe; still must be rerun |
| User-observed current leader | ~0.667 | C: user report | No public notebook/version attribution has been verified |

Do not present upstream or user-reported scores as local reproduction.

## Error pattern

The CPU style-aware model is strong on some providers but weak on fine-grained labels:

- strong: `gemini-3.1-pro-preview` 0.82 recall, `claude-opus-5` 0.66, `gpt-6-astra` 0.62;
- weak: `claude-fable-5` 0.107, `glm-5.3` 0.16, `glm-5.3-flash` 0.22, `gpt-5.6-sol` 0.18, `kimi-k3` 0.28;
- provider recall: Google 0.967, OpenAI 0.763, Anthropic 0.694, Qwen 0.467, ZAI 0.387, Moonshot AI 0.280.

The next model should add transferable representation and within-family discrimination rather than only more ordinary word n-grams.

## Recommended execution order

1. Reproduce the two-stage public LoRA recipe in a Kaggle GPU Notebook.
2. Submit it and record the actual public/private score.
3. Run `--only-8b`, `--no-twin`, and `--no-balance` as isolated ablations only if the full run works.
4. If the public recipe is substantially below the user's leader, search for newer public solutions and compare their input lists and legal data sources.
5. Add one new model family at a time, keeping saved logits and a score ledger.
6. Use the CPU scripts only as controls and fallback submissions.

## Validation and competition-safety rules

- Use prompt-grouped CV. Random-row splits leak topic and are diagnostic only.
- Fit vectorizers, scalers, heads, and blend weights on training folds only.
- Do not use hidden test labels or manual test labels.
- Do not assume each hidden prompt has exactly one row from every model.
- Treat test clustering, centered twins, Sinkhorn, and class quotas as transductive assumptions; validate them on many pseudo-test draws before trusting them.
- Keep third-party data/checkpoint licenses and attribution in the experiment record.

## Known environment note

The original private workspace contains raw data, PDFs, extracted research, OOF arrays, and a bundled dependency tree. Those are intentionally excluded from this public release. Run scripts in Kaggle or a clean environment with the requirements files here.
