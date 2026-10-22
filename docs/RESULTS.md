# Results and provenance

This file separates locally reproduced values from upstream claims and the user's leaderboard observation.

| Result | Value | Evidence tier | Status |
|---|---:|---|---|
| Original char/word/style SVC, prompt-grouped OOF | 0.370508 accuracy / 0.370403 BA | A | Local private snapshot |
| Style-aware four-view SVC, prompt-grouped OOF | 0.384484 BA | A | Local private snapshot |
| Style-aware plus numeric-style CPU script | 0.388936 BA | A | Private 5-fold run; compact summary in `evidence/metrics/cpu_numeric_style_local_metrics.json` |
| CPU row-softmax blend probe | ~0.39337 | A/B | Fold-held-out diagnostic, not nested retraining |
| Qwen3-1.7B blend | ~0.431 | C | Upstream notebook note; not reproduced here |
| Two-stage Qwen3 8B/14B LoRA + public pool | 0.593 grouped CV / 0.592 public LB | B/C | Upstream notebook report; reproduce with the packaged script |
| User-observed current leaderboard leader | ~0.667 | C | User report; exact notebook/version not identified |

## What a result means

- **Local OOF:** a development estimate on the supplied training snapshot. It can be selection-biased.
- **Public LB:** Kaggle's public leaderboard score for a submitted full-test artifact.
- **Private LB:** Kaggle's hidden/private score when available.
- **Upstream report:** a number written in somebody else's notebook or discussion; it is not a reproduction.

A result is not considered an improvement until the exact code/input version and a new Kaggle submission are recorded. The repository intentionally contains no submission file because the visible local test has only three rows.

## Gold-hunt log template

For each Kaggle run, record privately:

```text
commit:
notebook URL/version:
GPU:
input datasets/models:
run time:
public score:
private score:
submission hash:
changes from previous run:
```
