# Public sources and provenance

## Competition

- [Name That LLM competition](https://www.kaggle.com/competitions/name-that-llm)
- [Competition data page](https://www.kaggle.com/competitions/name-that-llm/data)
- [Competition API metadata](https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionName=name-that-llm)
- [Discussion 749432](https://www.kaggle.com/competitions/name-that-llm/discussion/749432)
- [Discussion 749593](https://www.kaggle.com/competitions/name-that-llm/discussion/749593)
- [CLIST standings mirror](https://clist.by/standings/name-that-llm-classification-balanced-accuracy-score-71276398/)

## Highest credible public score target

- [The Prompt Is Not the Author — prvsiyan v4](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286)
- [Current public notebook page](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author)

The current v10 public source records historical v4 public balanced accuracy `0.604`, COMPLETE submission `57428928`, and the native version URL above. Historical v4 cells were not independently fetched. Crucially, the newest v10 header calls itself an **unscored candidate**, disables branch B and says its Lightning checkpoint is unavailable. It does not claim v4 parity. See the exact excerpts and dependency caveat in [`PUBLIC_GPU_ROUTES.md`](PUBLIC_GPU_ROUTES.md). The [`preflight`](../scripts/prvsiyan_route_preflight.py) screens current mounts, not historical v4 completeness.

Current v10 metadata identifiers (not a verified v4 inventory) include:

- `josephadamski91/ntl-atlas-qwen3-8b-v1`
- `prvsiyan/ntl-public-in-style-pool-v1`
- `prvsiyan/ntl-two-stage-lora-adapters-v1`
- `nishantdahal/ntl-two-stage-lora`
- Qwen3 8B and 14B Transformers model inputs

## Compact public GPU control

- [Two-stage LoRA on the public pool](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool)
- [Dataset: nishantdahal/ntl-two-stage-lora](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora)

The upstream page reports grouped CV `0.593` and public LB `0.592`. This release packages an inference script, but does not claim independent reproduction.

## Other public references

- [0.600 Teach the Model, Remove the Prompt](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt)
- [Two-stage LoRA + ATLAS and early layers](https://www.kaggle.com/code/josephadamski91/two-stage-lora-atlas-and-early-layers?scriptVersionId=360008790)

- [Qwen blend](https://www.kaggle.com/code/lucasboesen/qwen-blend)
- [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)
- [Two-stage LoRA | Gaussian Readout](https://www.kaggle.com/code/kirilltym/two-stage-lora-gaussian-readout)
- [Style-Aware Char N-Grams](https://www.kaggle.com/code/yangkuangou/name-that-llm-style-aware-char-n-grams-cv-0-384)
- [TF-IDF SVM baseline](https://www.kaggle.com/code/dariushafshar/name-that-llm-cv-0-37-tf-idf-svm-baseline)
- [0.355 Honest CV by prompt + Style](https://www.kaggle.com/code/alekseyosipenko/0-355-name-that-llm-honest-cv-by-prompt-styl?scriptVersionId=358708313)
- [Prompt recovery warning](https://www.kaggle.com/code/lucasboesen/name-that-llm-prompt-recovery-trap)

## Public data/checkpoint identifiers mentioned by upstream work

These are references only. Do not mirror them into this GitHub repository:

- `seyitkaangunes/ntl-extra-api-passages`
- `abdullah1afify/ntl-generations`
- `dophuchung/ntl-proxy-generations`
- `lucasboesen/ntl-free-proxy-generations`
- `pjmathematician/ultrabert-380m-ntl`
- `lightningv08/ntl-lora24-fivefold-training`
- `bbucxi/qwen35-9b-safetensors`

## Methods references

- [LUAR](https://aclanthology.org/2021.emnlp-main.70/)
- [Style Embeddings](https://aclanthology.org/2022.repl4nlp-1.26/)
- [DeBERTa](https://arxiv.org/abs/2111.09543)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [DetectGPT](https://arxiv.org/abs/2301.11305)
- [Fast-DetectGPT](https://arxiv.org/abs/2310.05130)
- [DetectLLM](https://arxiv.org/abs/2303.07295)
- [Binoculars](https://arxiv.org/abs/2401.12070)

## Attribution caveat

The user's observed leaderboard leader near `0.667` has not been publicly mapped to a verified notebook or script version. The `0.355` notebook title likely reports its own CV/result and is not evidence of the current leader's method. Do not state otherwise without a direct versioned source.
