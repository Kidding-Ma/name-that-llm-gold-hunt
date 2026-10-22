# Competition context and verified facts

Last checked: **2026-10-22** (Asia/Shanghai). Re-fetch the official metadata before a final submission because competition settings, deadlines, and leaderboard state can change.

## Official metadata facts

Source: the official Kaggle CompetitionService endpoint for [`name-that-llm`](https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionName=name-that-llm), fetched on the date above.

| Field | Observed value | Meaning / caution |
|---|---|---|
| Competition name/title | `name-that-llm` / `Name That LLM` | Official competition identity |
| Brief description | `Identify the model behind the text` | Confirms the broad task; it does not define every data or leakage rule |
| Evaluation algorithm | `Balanced Accuracy Score` | Maximized; the metadata marks it public. Confirm the official rules and metric implementation before relying on edge cases |
| Submission filename | `submission.parquet` | Required filename in the metadata |
| Submission mode | `onlyAllowKernelSubmissions=true` | Treat this as a Code/Kernel submission workflow; a random server upload may not be accepted |
| Row ID column | `id` | Official row identifier field |
| Deadline | `2027-01-14T23:59:00.817Z` | UTC; convert carefully for local planning |
| New entrants deadline | `2027-01-06T23:59:00Z` | UTC; verify current page before joining |
| Team merger deadline | `2027-01-06T23:59:00Z` | UTC; maximum team size is 5 |
| Maximum daily submissions | `5` | Do not spend submissions on unlogged variants |
| Scored submissions in metadata | `2` | This field is time-sensitive and should be rechecked with the current competition page |
| Leaderboard percentage | `40` | Official metadata field; do not infer the full public/private split from this alone |
| Score display truncation | `3` decimals | Display metadata, not a license to over-interpret close scores |
| Submission size limit | `20,480 MB` | Metadata limit; generated artifacts should still be kept small and reproducible |
| Identity verification | required | Official metadata field; the account must satisfy Kaggle's current requirements |
| Runtime limits | CPU `540` minutes; GPU `540` minutes | Metadata limits, not a promise that every route fits comfortably |
| Rules | `rulesRequired=true` | Join and accept the current rules before downloading or submitting |
| Competition metadata license | `CC BY-SA 4.0` | This does not automatically license third-party inputs, models, or notebook code |
| Reward | USD 50,000 | Metadata fact; not a reason to relax research or competition-integrity controls |

The same endpoint reports a leaderboard, medals allowed, synchronous reruns, and that the competition is launched. It reports the aggregate counts shown on the endpoint at the time of retrieval; do not treat those counts as a stable ranking or as evidence for a particular method.

## Data facts from the available local snapshot

These are **local snapshot facts**, not guarantees about the current hidden test:

- `train.csv` columns: `id,prompt_id,text,model`;
- `test.csv` columns: `id,text`;
- 2,699 training rows, 150 prompt IDs, and 18 model labels in the private snapshot;
- the visible local `test.csv` contains only 3 rows, so it is a schema/format smoke fixture, not the leaderboard test;
- the private train snapshot SHA-256 is documented in [`DATA_AND_LICENSES.md`](DATA_AND_LICENSES.md).

The full Kaggle input must be attached in a Code Notebook or obtained through an authorized Kaggle route. The local three-row test must never be described as the hidden test.

## What is not established by this document

The official metadata endpoint does **not** establish all of the following. Confirm them from the current competition data page, rules, sample submission, or a legal experiment before using them as assumptions:

- the exact hidden-test row count and construction;
- whether every hidden prompt has the same number of rows or one row per class;
- whether prompt IDs are present in the hidden test and whether they may be used;
- the complete sample-submission schema beyond the required filename and row ID metadata;
- permission for external datasets, proxy generations, pretrained checkpoints, pseudo-labels, test-time adaptation, clustering, balancing, or internet access;
- the precise public/private leaderboard split and score timing;
- whether any public notebook score is reproducible from currently available inputs.

Keep these as unknowns until evidence resolves them. If a method depends on one of them, label the method as a hypothesis and test it with rule review and pseudo-test stress tests.

## Competition-safe defaults

1. Accept and follow the current Kaggle rules and identity requirements.
2. Keep raw competition data, hidden-test material, weights, adapters, public pools, and credentials outside this public repository.
3. Split validation according to the verified grouping structure; when prompt/topic leakage is plausible, do not use random-row CV as the primary estimate.
4. Fit preprocessing, calibration, class priors, and blend weights on training-side data only.
5. Generate submissions through a checked pipeline and preserve test-row order and IDs.
6. Record Code Notebook version, input versions, hashes, runtime, output hash, and public/private scores for every submission.

## Authoritative links

- [Competition page](https://www.kaggle.com/competitions/name-that-llm)
- [Competition data page](https://www.kaggle.com/competitions/name-that-llm/data)
- [Official competition metadata endpoint](https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionName=name-that-llm)
- [Kaggle competition discussion](https://www.kaggle.com/competitions/name-that-llm/discussion/749593)
- [Public-source and score ledger in this repository](PUBLIC_GPU_ROUTES.md)
