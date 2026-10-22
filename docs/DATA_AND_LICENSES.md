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

## Other references

The repository mentions Qwen, public pools, UltraBERT, DeBERTa, ELECTRA, GPT proxy generations, and Claude API passages only as research references. Each external asset must be checked for:

- source URL and version;
- license and attribution;
- whether competition rules permit its use;
- hash and expected file names;
- whether prompts overlap with the competition data.

The public pool's license and attribution must be checked on its current Kaggle dataset page before redistribution. The repository does not grant permission to mirror it.

## Output contract

The only competition output is a Parquet file with exactly two columns:

```text
id,model
```

The `id` values must be unique and preserve the input test order. Every `model` must be one of the labels present in the official training data.
