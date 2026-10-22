# Compact GPU input layout

The default server route is the Nishant two-stage LoRA control. Keep all
inputs under one external directory, normally `$DATA_ROOT/input`; do not put
any of these files in GitHub.

## Expected tree

```text
$DATA_ROOT/
├── input/
│   ├── competition/name-that-llm/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── nishantdahal/ntl-two-stage-lora/
│   │   ├── pool_v1.parquet                 # columns text, model
│   │   ├── lora8b_v1/
│   │   │   ├── meta.json
│   │   │   ├── adapter/
│   │   │   │   ├── adapter_config.json
│   │   │   │   └── adapter_model.safetensors
│   │   │   └── heads.safetensors
│   │   └── lora14b_v1/
│   │       ├── meta.json
│   │       ├── adapter/
│   │       │   ├── adapter_config.json
│   │       │   └── adapter_model.safetensors
│   │       └── heads.safetensors
│   └── models/
│       └── qwen-3/
│           ├── 8b-base/
│           │   ├── config.json
│           │   ├── tokenizer.json
│           │   └── *.safetensors
│           └── 14b-base/
│               ├── config.json
│               ├── tokenizer.json
│               └── *.safetensors
├── manifest.json
├── logs/
├── outputs/
└── runs/
```

The inference script searches recursively below `--input-dir`, so harmless
extra provider directories are okay. The verification scripts use the exact
configured targets first and reject symlink/path escapes.

## Required schemas

- `train.csv`: unique headers including `id,prompt_id,text,model`, at least one nonempty text row, and the exact 18 labels used by the script.
- `test.csv`: unique headers including `id,text`, at least one nonempty text row. The local repository's 3-row test is only a smoke file and cannot produce a leaderboard score.
- `pool_v1.parquet`: readable by PyArrow/Pandas with `text` and `model` columns.
- Both adapter directories: every file shown above must be nonempty; JSON files must parse as objects.
- Qwen 8B config: `model_type=qwen3`, `hidden_size=4096`, `num_hidden_layers=36`.
- Qwen 14B config: `model_type=qwen3`, `hidden_size=5120`, `num_hidden_layers=40`.
- Both model directories need a tokenizer and at least one nonempty `.safetensors` file.

Run verification before the model:

```bash
python scripts/gpu_input_manifest.py verify --data-root "$DATA_ROOT" --hash metadata
python scripts/server_preflight.py --data-root "$DATA_ROOT" --strict
```

These checks prove file visibility and basic structure only. They do not prove
license permission, exact Kaggle version identity, model numerical correctness,
historical prvsiyan v4 completeness, or leaderboard performance.
