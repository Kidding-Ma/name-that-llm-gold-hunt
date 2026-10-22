# Data, checkpoints, and licenses

## Competition data

This repository does not redistribute the competition files. Obtain them from the official [Name That LLM competition](https://www.kaggle.com/competitions/name-that-llm) after joining the competition and accepting its rules.

Expected files when running locally or in Kaggle:

```text
train.csv: id,prompt_id,text,model
test.csv: id,text
```

The private workspace snapshot contained 2,699 training rows, 150 prompts, and 18 labels. Its train SHA-256 was:

```text
40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707
```

The local snapshot's `test.csv` had only three visible examples. That file is not the hidden leaderboard test and is not included here.

## GPU recipe inputs

The two-stage public GPU script expects these inputs to be attached separately in Kaggle:

1. Competition: `name-that-llm`
2. Dataset: `nishantdahal/ntl-two-stage-lora`
3. Qwen3 8B Base, Transformers format
4. Qwen3 14B Base, Transformers format

Expected public dataset contents include the public pool and the `lora8b_v1` / `lora14b_v1` adapter directories. The script validates model labels and fails closed when an input is missing.

Do not copy these assets into GitHub. Large files belong in Kaggle datasets/model inputs or a private artifact store.

## Higher-score public route references

The audit found a source-bound public score claim for [prvsiyan v4](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author?scriptVersionId=360252286): reported public balanced accuracy `0.604`, COMPLETE submission `57428928`. Its reported extra inputs are listed in [`PUBLIC_GPU_ROUTES.md`](PUBLIC_GPU_ROUTES.md), but they are not bundled here and have not been independently reproduced. Use the native Kaggle notebook/source and inspect every current input version before running.

## Other references

The repository mentions Qwen, public pools, UltraBERT, DeBERTa, ELECTRA, GPT proxy generations, and Claude API passages only as research references. Each external asset must be checked for:

- source URL and version;
- license and attribution;
- whether competition rules permit its use;
- hash and expected file names;
- whether prompts overlap with the competition data.

The public pool's license and attribution must be checked on its current Kaggle dataset page before use or redistribution. The repository does not grant permission to mirror it.

## External input manifest

The following identifiers are pinned by name/URL for reproducibility. Kaggle may expose a version number when the input is attached; record that version and the local SHA-256 in the private run log. No binary hash is asserted here because the artifacts are intentionally not redistributed.

| Input | Public identifier | Expected role | License/terms action |
|---|---|---|---|
| Competition | [`name-that-llm`](https://www.kaggle.com/competitions/name-that-llm) | official train/hidden test | accept and follow competition rules |
| Public pool + adapters | [`nishantdahal/ntl-two-stage-lora`](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora) | `pool_v1.parquet`, `lora8b_v1`, `lora14b_v1` | inspect current Kaggle dataset license/attribution |
| Qwen3 8B base | [`qwen-lm/qwen-3`](https://www.kaggle.com/models/qwen-lm/qwen-3) | Transformers base matching `hidden_size=4096`, `num_hidden_layers=36` | follow the Qwen license shown on the attached model version and Kaggle terms |
| Qwen3 14B base | [`qwen-lm/qwen-3`](https://www.kaggle.com/models/qwen-lm/qwen-3) | Transformers base matching `hidden_size=5120`, `num_hidden_layers=40` | follow the Qwen license shown on the attached model version and Kaggle terms |

The two-stage script records input SHA-256 values in its private metadata only. Do not paste private input paths or credentials into public issues.

## Output contract

The only competition output is a Parquet file with exactly two columns:

```text
id,model
```

The `id` values must be unique and preserve the input test order. Every `model` must be one of the labels present in the official training data.
