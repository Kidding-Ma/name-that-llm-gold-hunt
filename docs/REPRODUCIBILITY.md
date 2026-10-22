# Reproducibility contract

## Validation

Use 5-fold prompt-grouped validation:

- group column: `prompt_id`;
- primary metric: pooled 18-class balanced accuracy;
- no prompt ID overlap between train and validation;
- fit TF-IDF vectorizers, scalers, classifiers, and blend weights inside the training side of each fold;
- random-row splits are diagnostics only because they leak topic.

## CPU install

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements-cpu.txt
```

Kaggle's standard CPU image already includes the common scientific stack; do not add the old private `research/.deps` directory to a public clone.

## GPU install

Kaggle model inputs normally provide PyTorch and Transformers. For a private machine, install a CUDA-compatible PyTorch build first, then:

```bash
python -m pip install -r requirements-gpu.txt
```

The exact CUDA/PyTorch pairing is machine-specific; record it with every run.

## Reproduce a Kaggle CPU submission

1. Add the `Name That LLM` competition data to a Kaggle Code Notebook.
2. Paste `scripts/kaggle_submit.py` into a cell.
3. Run it against Kaggle's complete test input.
4. Validate `/kaggle/working/submission.parquet`.
5. Save a version and submit.

## Reproduce the public GPU starting point

1. Add the competition data.
2. Add `nishantdahal/ntl-two-stage-lora`.
3. Add Qwen3 8B Base and Qwen3 14B Base Transformers model inputs.
4. Select preferably 2×T4.
5. Run `scripts/kaggle_best_two_stage_lora.py`.
6. Save the metadata JSON privately; it includes input hashes and run time but intentionally redacts local input/output paths.

## No hidden-test shortcut

Do not use hidden labels, manually inspect hidden text, force a test quota, or use a prompt assignment rule unless the competition contract and repeated pseudo-test validation support that assumption.
