"""Reproducible, dependency-light audit for Name That LLM data."""
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "research" / "evidence"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text.strip()).casefold()
    return text


def provider(label: str) -> str:
    if label.startswith("gpt-"):
        return "OpenAI"
    if label.startswith("gemini-"):
        return "Google"
    if label.startswith("claude-"):
        return "Anthropic"
    if label.startswith("qwen"):
        return "Qwen"
    if label.startswith("glm-"):
        return "ZAI"
    if label.startswith("kimi-"):
        return "Moonshot AI"
    return "other"


train_path = DATA / "train.csv"
test_path = DATA / "test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

required_train = ["id", "prompt_id", "text", "model"]
required_test = ["id", "text"]
assert list(train.columns) == required_train, train.columns.tolist()
assert list(test.columns) == required_test, test.columns.tolist()
assert train.id.is_unique and test.id.is_unique
assert train[required_train].notna().all().all()
assert test[required_test].notna().all().all()

train_norm = train.text.map(norm_text)
test_norm = test.text.map(norm_text)
prompt_model = train.groupby(["prompt_id", "model"], dropna=False).size()
prompt_sizes = train.groupby("prompt_id", dropna=False).size()
model_counts = train.model.value_counts().sort_index()
labels = sorted(train.model.unique())

train_texts = set(train.text.astype(str))
test_texts = set(test.text.astype(str))
train_norms = set(train_norm)
test_norms = set(test_norm)

# Connected components of exact-normalized text duplicates. This is a conservative
# diagnostic grouping: every duplicate text gets one component, singletons remain.
parent = list(range(len(train)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra
first = {}
for i, value in enumerate(train_norm):
    if value in first:
        union(i, first[value])
    else:
        first[value] = i
components = pd.Series([find(i) for i in range(len(train))])
component_sizes = components.value_counts()

report = {
    "python": sys.version,
    "train": {
        "rows": int(len(train)),
        "columns": train.columns.tolist(),
        "sha256": sha256(train_path),
        "nulls": {k: int(v) for k, v in train.isna().sum().items()},
        "duplicate_ids": int(train.id.duplicated().sum()),
        "duplicate_exact_texts": int(train.text.duplicated().sum()),
        "duplicate_normalized_texts": int(train_norm.duplicated().sum()),
        "normalized_text_unique": int(train_norm.nunique()),
    },
    "test_visible_example": {
        "rows": int(len(test)),
        "columns": test.columns.tolist(),
        "sha256": sha256(test_path),
        "nulls": {k: int(v) for k, v in test.isna().sum().items()},
        "duplicate_ids": int(test.id.duplicated().sum()),
        "text_overlap_exact_with_train": int(len(train_texts & test_texts)),
        "text_overlap_normalized_with_train": int(len(train_norms & test_norms)),
    },
    "labels": labels,
    "label_count": len(labels),
    "class_counts": {str(k): int(v) for k, v in model_counts.items()},
    "provider_counts": {str(k): int(v) for k, v in train.model.map(provider).value_counts().sort_index().items()},
    "prompts": {
        "count": int(train.prompt_id.nunique()),
        "size_min": int(prompt_sizes.min()),
        "size_max": int(prompt_sizes.max()),
        "size_counts": {str(k): int(v) for k, v in prompt_sizes.value_counts().sort_index().items()},
        "prompt_model_duplicate_pairs": int((prompt_model > 1).sum()),
        "max_rows_per_prompt_model": int(prompt_model.max()),
        "prompt_with_all_labels": int((train.groupby("prompt_id").model.nunique() == len(labels)).sum()),
    },
    "normalized_duplicate_components": {
        "component_count": int(component_sizes.size),
        "components_size_gt_1": int((component_sizes > 1).sum()),
        "rows_in_components_size_gt_1": int(component_sizes[component_sizes > 1].sum()),
        "max_component_size": int(component_sizes.max()),
        "cross_prompt_components": int(sum(train.loc[idx, "prompt_id"].nunique() > 1 for idx in components.groupby(components).groups.values())),
        "cross_model_components": int(sum(train.loc[idx, "model"].nunique() > 1 for idx in components.groupby(components).groups.values())),
    },
    "contract": {
        "hidden_test_prompt_id_available": False,
        "submission_columns": ["id", "model"],
        "visible_test_is_hidden_test": False,
        "recommended_validation": "5-fold prompt-grouped CV; fit all vectorizers/scalers inside fold",
        "primary_metric": "pooled 18-class balanced accuracy",
    },
}

(OUT / "verified_data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
prompt_model.unstack(fill_value=0).to_csv(OUT / "prompt_model_counts.csv")
train.assign(normalized_text=train_norm, duplicate_component=components).to_csv(OUT / "train_duplicate_diagnostics.csv", index=False)
print(json.dumps(report, ensure_ascii=False, indent=2))
