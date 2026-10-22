# Public GPU routes and score provenance

更新：2026-10-22（Asia/Shanghai）

This ledger records the strongest public Kaggle routes **found** during the handoff audit, not a complete leaderboard ranking. All public GPU scores below are author/source reports, not our own hidden-test submissions.

## Score ranking and actual availability

| Rank | Public notebook / version | Reported score | Evidence and availability | Decision |
|---|---|---:|---|---|
| 1 | [The Prompt Is Not the Author — historical v4](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286) | **0.604 public BA** | Current v10 source links v4 to COMPLETE submission **57428928**. Historical v4 cells were not independently fetched; v10 says branch B's Lightning checkpoint is unavailable. | Highest source-bound score found, **NOT a ready-to-run reproduction package** |
| 2 | [0.600 Teach the Model, Remove the Prompt](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt) | **0.600** | Public title/source claim; many external dependencies, not reproduced | Audit dependency availability before choosing it |
| 3 | [Two-stage LoRA + ATLAS and early layers — v3](https://www.kaggle.com/code/josephadamski91/two-stage-lora-atlas-and-early-layers?scriptVersionId=360008790) | **0.596** | Author/source report; not independently reproduced | Native candidate after the compact control; confirm all mounts |
| 4 | [Two-stage LoRA on the public pool — v2](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool?scriptVersionId=359887828) | **0.592 public LB / 0.593 grouped CV** | Public source inspected; adapter/pool bundle inspected privately, base models require Kaggle inputs | First executable control; packaged [inference script](../scripts/kaggle_best_two_stage_lora.py) |
| 5 | [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering) | **0.550 historical v4; 0.552 also mentioned** | Internally inconsistent historical score notes; many external inputs | Secondary reference only |

The user-observed leader around **0.667** is not mapped to public code/version/submission in this audit. Do not infer a method from that number.

## What was actually read for 0.604

The [public pull endpoint](https://www.kaggle.com/api/v1/kernels/pull/prvsiyan/the-prompt-is-not-the-author) returned current version **10**. Its nested notebook JSON was parsed read-only; remote code was not executed. The early embedded JPEG makes text-view fetches truncate, but a direct JSON GET exposed the relevant markdown.

Current v10 cell 3 says:

> The verified historical result for **v4 is public balanced accuracy 0.604** (COMPLETE submission `57428928`, [native version 4](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286), checked 20 October 2026). Version 3 scored 0.599; v5 was still pending at this reader check.

This is an **author statement in source**, not an independently retrieved Kaggle score receipt. The historical v4 source itself was not obtained through a version-specific API. The exact native URL remains the right place to inspect it in Kaggle.

### Critical reproduction blocker: historical branch B

Current v10 cell 1 explicitly says:

> This is an **unscored candidate**, deliberately different from the historical two-branch V4 result of **0.604**. It makes no V4 parity or improvement claim. Branch B is disabled before its extra-passage loading, classifiers or model inference. Its unavailable Lightning LoRA checkpoint is therefore not a required input; no other checkpoint is substituted under that identity.

Therefore:

- **Current v10 is not the 0.604 implementation.** Its later candidate cannot inherit v4's score.
- The four datasets/two models in current metadata are not proof that all historical v4 dependencies are available.
- The historical second branch needs its exact checkpoint identity/version and permission restored before calling a run a v4 reproduction.
- A replacement checkpoint, branch-A-only run, Gaussian addition, or ATLAS blend is a **new unscored experiment**, not 0.604 parity.
- The [route preflight](../scripts/prvsiyan_route_preflight.py) only screens current metadata, CUDA, CSV structure and named markers. Even exit 0 **does not clear this historical blocker**.

## Historical changes: do not misattribute later components to v4

The current source's version table reports:

| Historical version | Reported change | Reported public score |
|---|---|---:|
| v1 | Reimplemented 0.563 stack + four shared-prompt Gaussian group heads | 0.569 |
| v2 | Additional changes recorded upstream | 0.593 |
| v3 | Two own retrained LoRA seeds join branch A | 0.599, submission `57418859` |
| v4 | Gaussian heads extended to every two-stage LoRA, not only 8B; branch A returns to its own TF-IDF grouping | 0.604, submission `57428928` |
| v5 | Adds Joseph's frozen Qwen3-4B early-layer classifier and its Gaussian head | Pending at that editorial check |
| current v10 | Branch B removed; active branch A + four Gaussian heads + 10% ATLAS; tokenizer/provenance fixes | **Explicitly unscored** |

ATLAS and early-layer components described in later markdown must **not** be advertised as the reason v4 scored 0.604. V4 changed both grouping and Gaussian additions; the score difference is not an isolated causal estimate for any one head.

## Inputs reported by current v10 metadata

The following identifiers are metadata for **v10**, not a verified v4 inventory. Record exact mounted versions and private SHA-256 hashes; a slug alone is not a content pin.

- competition: `name-that-llm`
- dataset: `josephadamski91/ntl-atlas-qwen3-8b-v1`
- dataset: `prvsiyan/ntl-public-in-style-pool-v1`
- dataset: `prvsiyan/ntl-two-stage-lora-adapters-v1`
- dataset: `nishantdahal/ntl-two-stage-lora`
- model: `qwen-lm/qwen-3/Transformers/8b-base/1`
- model: `qwen-lm/qwen-3/Transformers/14b-base/1`
- GPU metadata: Tesla T4, GPU enabled, Internet off

The source's newest header requires reviewed ATLAS V2 and byte checks for selected assets. Reuse its native exact checks if researching that candidate, not just this repository's broad path-name preflight. Its older historical prose is preserved below newer contradictory candidate headers; follow explicit version-specific statements, not the most flattering paragraph.

## Recommended execution order

1. Run the compact [Nishant control](../scripts/kaggle_best_two_stage_lora.py) unchanged on Kaggle 2×T4, after checking its original inputs/tokenizer.
2. Submit the full hidden test and record your own score, notebook version, runtime and input versions.
3. Run `--only-8b`, `--only-14b`, `--no-twin`, `--no-balance` separately. Optional `--scores-output` saves private aligned SVC/plain-LoRA probabilities, weighted twin contributions, blend/final scores and inferred groups; these are **not OOF**.
4. Prefer native Joseph 0.596 as a second candidate if its exact dependencies are available; audit Kirill 0.600 separately.
5. Inspect historical prvsiyan v4. Proceed only if branch B and **all** original dependencies can be mounted legally. Otherwise stop claiming reproduction; choose a clearly named new branch-A/Gaussian/ATLAS experiment.
6. Accept an improvement only with a saved legal full-test score, or honestly label strict outer-fold evidence as local only.

## Hardware and cost

- **CPU only:** controls and submission-format tests; no GPU score claim.
- **1×T4 / 16GB:** 8B-only diagnostic or Qwen3-1.7B control, not full 14B.
- **2×T4:** recommended public inference environment. Use free Kaggle GPU before renting.
- **Larger single 24–48GB:** measure actual free memory; model loading now uses capacity, not a hard-coded two-card requirement. Available memory is not an OOM guarantee.
- **Complex multi-input research:** roughly 64GB+ host RAM and 200GB+ NVMe are prudent planning estimates, not benchmarked minima.

Do not rent multi-GPU servers merely because a historical notebook reports 0.604. Restore dependencies and obtain one real score first.

## Other routes and cautions

[Kirill's Gaussian Readout variant](https://www.kaggle.com/code/kirilltym/two-stage-lora-gaussian-readout) explicitly says **“No score is known for this modified version.”** Do not rank its indirect 0.595 association as verified.

[Lucas adaptive centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering) has historical notes near 0.541–0.550 and a header mention near 0.552, but inconsistent history. It is not a simpler guaranteed replacement.

## Legality and experiment record

For every route, privately record notebook URL and scriptVersionId; dataset/model/kernel-output versions; hashes; licenses and attribution; competition-rule compatibility; hardware/software/dtype; test row count and prediction distribution; runtime; submission hash; public/private score.

Do not copy competition data, weights, adapters, pools or raw notebook blobs into this repository. Public visibility is not redistribution permission. Competition metadata reports CC BY-SA 4.0 for the competition; third-party inputs/notebooks/models retain their own terms.

Fit all new vectorizers, adapters, heads, calibration and blend selection on training-side prompt groups. A fit-all public adapter is not honest grouped OOF just because its readout is refitted. Test centering, inferred clustering and Sinkhorn on varied partial-prompt and imbalanced-prior pseudo-tests; do not assume hidden quotas or use Hungarian assignment by default.

## Evidence labels

- **A:** local prompt-grouped reproduction.
- **B:** public source/implementation inspected.
- **C:** author/source score claim or user report.
- **D:** our own scored full hidden-test Kaggle submission.

The historical 0.604 is **B/C**; its dependency completeness is unconfirmed. No GPU score in this release is D.
