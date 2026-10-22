"""Kaggle-ready inference for the compact public two-stage reproduction control.

Method: Nishant Dahal's "Two-stage LoRA on the public pool" recipe.

Reported by the public notebook (not independently reproduced in this workspace):
    - grouped prompt-held-out CV: 0.593
    - public leaderboard: 0.592

This file is inference-only. The LoRA adapters were trained offline and are
expected in the public Kaggle dataset `nishantdahal/ntl-two-stage-lora`.

Required Kaggle inputs
----------------------
1. Competition data: Name That LLM.
2. Dataset: nishantdahal/ntl-two-stage-lora
   It should contain pool_v1.parquet, lora8b_v1/, and lora14b_v1/.
3. Model inputs:
   - qwen-lm/qwen-3, Transformers, 8b-base
   - qwen-lm/qwen-3, Transformers, 14b-base
4. GPU: preferably 2x T4, internet off.

The implementation deliberately does not use prompt IDs, Hungarian assignment,
or hidden labels. Its independent control uses only the public pool, pretrained
model inputs, test text, and the two public LoRA checkpoints, with centered
 twins and Sinkhorn disabled. `--allow-transductive` separately enables
content-TF-IDF test-batch clusters of size 2--18 and uniform-prior Sinkhorn;
that mode is a hypothesis requiring rule review and pseudo-test stress tests.
The full-test guard rejects the local three-row smoke fixture.

Usage as a Kaggle notebook cell
--------------------------------
Paste the whole file into a Kaggle notebook cell and run it. The notebook
runtime has ipykernel in sys.modules, so command-line parsing is disabled.

Usage as a Python file
----------------------
    python kaggle_best_two_stage_lora.py --no-twin --no-balance
    python kaggle_best_two_stage_lora.py --allow-transductive
    python kaggle_best_two_stage_lora.py --only-8b --no-twin --no-balance
    python kaggle_best_two_stage_lora.py --only-14b
    python kaggle_best_two_stage_lora.py --no-twin
    python kaggle_best_two_stage_lora.py --no-balance
    python kaggle_best_two_stage_lora.py --scores-output /private/scores.npz

Score archives are optional, private artifacts: aligned SVC/plain-LoRA
probabilities, twin weighted-log contributions, pre-balance blend logits,
final probabilities, IDs/labels and inferred groups. They are not competition
submissions or OOF estimates; never commit them. Default prediction weights
and the public recipe are unchanged by score export.

The default independent control disables test-batch clustering and Sinkhorn;
use `--allow-transductive` only for the separately named public-reference
hypothesis after rule review. `--only-8b` is a smoke/ablation mode and is not
the reported 0.592 blend.
"""
from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

# Keep BLAS from oversubscribing the Kaggle workers before importing NumPy.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np
import pandas as pd
import torch
from torch import nn

# Exact checkpoint class order from the public adapter metadata/notebook.
LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "claude-fable-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
K = len(LABELS)

W_SVC = 1.021
LORA_CONFIG = {
    "lora8b_v1": {
        "w": 0.226,
        "twin_w": 0.079,
        "bs": 16,
        "base": {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36},
    },
    "lora14b_v1": {
        "w": 0.429,
        "twin_w": 0.150,
        "bs": 8,
        "base": {"model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40},
    },
}
TWIN_THRESH = 0.85
TWIN_MIN_SIZE = 2
TWIN_MAX_SIZE = 18
BALANCE_MIN_ROWS = 50

FUNCTION_WORDS = set(
    "a an the and or but if while although because as so than that which who whom whose this these those it "
    "its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to "
    "from with without for of by over under between through into about is are was were be been being have has "
    "had do does did can could will would shall should may might must not no nor all any some each every both "
    "either neither other another such only also just very too more most less least here there when where why "
    "how what whether then now still yet even much many few little".split()
)
VIEWS = [
    ("char", 3, 5, 250000, 1.0),
    ("word", 1, 3, 180000, 0.7),
    ("shape", 2, 6, 60000, 0.5),
    ("function", 1, 3, 50000, 0.5),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0:
        return np.empty_like(logits)
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(np.clip(z, -80.0, 80.0))
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def shape_text(text: str) -> str:
    # Preserve case, whitespace, digits, punctuation and Unicode punctuation;
    # collapse alphabetic content into a case-shape view.
    return re.sub(r"[^\W\d_]+", lambda match: "X" if match[0].isupper() else "x", str(text))


def function_text(text: str) -> str:
    # Keep function words and punctuation in position; replace content words.
    return " ".join(
        token.lower() if token.lower() in FUNCTION_WORDS or not token.isalnum() else "CONTENT"
        for token in re.findall(r"\w+|[^\w\s]", str(text))
    )


def view_text(text: str, view: str) -> str:
    if view == "shape":
        return shape_text(text)
    if view == "function":
        return function_text(text)
    return str(text)


def find_competition_data(input_dir: Path) -> tuple[Path, Path]:
    """Find one unambiguous competition train/test pair, not auxiliary CSVs."""
    train_candidates: list[Path] = []
    for candidate in sorted(input_dir.rglob("train.csv")):
        try:
            header = pd.read_csv(candidate, nrows=2)
        except Exception:
            continue
        if {"id", "prompt_id", "text", "model"}.issubset(header.columns):
            train_candidates.append(candidate)
    if not train_candidates:
        raise FileNotFoundError(
            "Could not find competition train.csv with id,prompt_id,text,model. "
            "Attach the Name That LLM competition data."
        )
    if len(train_candidates) > 1:
        raise RuntimeError(f"Ambiguous competition train.csv candidates: {train_candidates}")
    train_path = train_candidates[0]

    same_dir = train_path.parent / "test.csv"
    candidates = [same_dir] if same_dir.exists() else []
    candidates += [p for p in sorted(input_dir.rglob("test.csv")) if p != same_dir]
    test_candidates: list[Path] = []
    for candidate in candidates:
        try:
            header = pd.read_csv(candidate, nrows=2)
        except Exception:
            continue
        if {"id", "text"}.issubset(header.columns) and "model" not in header.columns:
            test_candidates.append(candidate)
    if len(test_candidates) != 1:
        raise RuntimeError(f"Expected one competition test.csv paired with {train_path}; found {test_candidates}")
    return train_path, test_candidates[0]


def find_pool(input_dir: Path) -> Path:
    candidates = []
    for candidate in sorted(input_dir.rglob("pool_v1.parquet")):
        try:
            header = pd.read_parquet(candidate, columns=["text", "model"]).head(1)
        except Exception:
            continue
        if {"text", "model"}.issubset(header.columns) and len(header) > 0:
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            "Could not find a readable nonempty pool_v1.parquet with text,model. "
            "Attach the public dataset nishantdahal/ntl-two-stage-lora."
        )
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous pool_v1.parquet candidates: {candidates}")
    return candidates[0]


def find_checkpoint(input_dir: Path, name: str) -> Path:
    candidates = sorted(
        p.parent for p in input_dir.rglob("meta.json") if p.parent.name == name
    )
    valid = []
    required = {
        "meta.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "heads.safetensors",
    }
    for candidate in candidates:
        if all((candidate / relative).is_file() and (candidate / relative).stat().st_size > 0 for relative in required):
            valid.append(candidate)
    if not valid:
        raise FileNotFoundError(
            f"Could not find a complete {name} adapter bundle. Attach nishantdahal/ntl-two-stage-lora."
        )
    if len(valid) > 1:
        raise RuntimeError(f"Ambiguous {name} adapter bundles: {valid}")
    return valid[0]


def _config_matches(config: dict, spec: dict) -> bool:
    nested = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    return all(config.get(key, nested.get(key)) == value for key, value in spec.items())


def find_base_model(input_dir: Path, spec: dict) -> Path:
    candidates = []
    for config_path in sorted(input_dir.rglob("config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _config_matches(config, spec):
            candidates.append(config_path.parent)
    valid = []
    for candidate in candidates:
        if (candidate / "tokenizer.json").is_file() and any(candidate.glob("*.safetensors")):
            valid.append(candidate)
    if not valid:
        raise FileNotFoundError(
            f"Could not find a complete attached base model matching {spec}. Add the corresponding "
            "Qwen3 Transformers model input."
        )
    if len(valid) > 1:
        raise RuntimeError(f"Ambiguous base model directories matching {spec}: {valid}")
    return valid[0]


def align_decision(decision: np.ndarray, classes: Iterable[int]) -> np.ndarray:
    out = np.zeros((len(decision), K), dtype=np.float64)
    for column, class_id in enumerate(classes):
        out[:, int(class_id)] = decision[:, column]
    return out


def fit_ngram_member(train: pd.DataFrame, test: pd.DataFrame, pool: pd.DataFrame) -> np.ndarray:
    """Style-aware n-gram SVC on official train + the public pool."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    pool = pool.copy()
    if "model" not in pool.columns and "label" in pool.columns:
        pool = pool.rename(columns={"label": "model"})
    if not {"text", "model"}.issubset(pool.columns):
        raise ValueError("pool_v1.parquet must contain text and model columns")
    pool["text"] = pool["text"].fillna("").astype(str)
    pool["model"] = pool["model"].astype(str)
    pool = pool[pool["model"].isin(LABEL_TO_ID)].copy()
    if (pool["text"].str.strip() == "").any():
        raise ValueError("The attached pool contains empty text rows")
    conflicts = pool.groupby("text")["model"].nunique()
    if (conflicts > 1).any():
        raise ValueError("The attached pool contains exact text rows with conflicting labels")
    pool = pool.drop_duplicates(subset=["text", "model"]).reset_index(drop=True)
    train_texts_exact = set(train["text"].astype(str))
    test_texts_exact = set(test["text"].astype(str))
    overlap_train = pool["text"].isin(train_texts_exact)
    overlap_test = pool["text"].isin(test_texts_exact)
    if bool(overlap_train.any()) or bool(overlap_test.any()):
        raise ValueError("The external pool overlaps official train/test text; inspect provenance before use")
    pool_labels = set(pool["model"].unique())
    if pool_labels != set(LABELS):
        raise ValueError(f"The attached pool labels do not match the 18-class checkpoint set: {sorted(pool_labels)}")
    if len(pool) < 1000:
        raise ValueError(f"The attached pool has only {len(pool)} valid rows; expected roughly 34,940.")

    train_texts = train["text"].fillna("").astype(str).tolist()
    test_texts = test["text"].fillna("").astype(str).tolist()
    fit_texts = train_texts + pool["text"].tolist()
    y_train = train["model"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    y_pool = pool["model"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    fit_y = np.r_[y_train, y_pool]

    train_parts, test_parts = [], []
    for view, lo, hi, max_features, weight in VIEWS:
        actual = "char" if view == "shape" else "word" if view == "function" else view
        options = dict(
            analyzer=actual,
            ngram_range=(lo, hi),
            lowercase=False,
            sublinear_tf=True,
            min_df=2,
            max_features=max_features,
            dtype=np.float32,
        )
        if actual == "word":
            options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vectorizer = TfidfVectorizer(**options)
        fit_view = [view_text(text, view) for text in fit_texts]
        test_view = [view_text(text, view) for text in test_texts]
        train_parts.append(vectorizer.fit_transform(fit_view) * weight)
        test_parts.append(vectorizer.transform(test_view) * weight)
        print(f"n-gram view {view}: {train_parts[-1].shape[1]} features", flush=True)

    X_train = hstack(train_parts, format="csr")
    X_test = hstack(test_parts, format="csr")
    clf = LinearSVC(
        C=0.5,
        random_state=42,
        max_iter=5000,
        class_weight="balanced",
    ).fit(X_train, fit_y)
    decision = align_decision(clf.decision_function(X_test), clf.classes_)
    print(f"n-gram fit rows: {len(fit_y)} (official={len(train)}, pool={len(pool)})", flush=True)
    del train_parts, test_parts, X_train, X_test, clf, pool
    gc.collect()
    return softmax(decision)


def gpu_memory_plan(estimated_gib: float, free_gib: list[float], reserve_gib: float = 2.5) -> dict[int, float]:
    """Choose real CUDA capacity, not GPU count; never silently spill to CPU/disk.

    This is a rough weight-only estimate with an activation/merge reserve, not
    an OOM guarantee. A single 40/48GB card must not be rejected just because
    it is not a two-T4 Kaggle runtime.
    """
    budgets = {i: max(0.0, float(free) - reserve_gib) for i, free in enumerate(free_gib)}
    budgets = {i: budget for i, budget in budgets.items() if budget > 0}
    if not budgets:
        raise RuntimeError("No CUDA memory remains after the activation reserve.")
    largest = max(budgets, key=budgets.get)
    if estimated_gib <= budgets[largest]:
        return {largest: budgets[largest]}
    if estimated_gib > sum(budgets.values()):
        raise RuntimeError(
            f"Trunk estimate {estimated_gib:.1f} GiB exceeds usable CUDA memory "
            f"{sum(budgets.values()):.1f} GiB after reserves. Use a larger GPU, "
            "more GPU memory, or --only-8b; CPU/disk offload is not enabled."
        )
    return budgets


def load_model(base_dir: Path, keep_layers: int, dtype: torch.dtype) -> nn.Module:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(base_dir)
    config.num_hidden_layers = keep_layers
    config.use_cache = False
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = config.layer_types[:keep_layers]

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    estimated_gib = 2 * (
        config.vocab_size * config.hidden_size
        + keep_layers * (
            2 * config.hidden_size * config.num_attention_heads * head_dim
            + 2 * config.hidden_size * config.num_key_value_heads * head_dim
            + 3 * config.hidden_size * config.intermediate_size
        )
    ) / (2**30)

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("This two-stage LoRA method needs a CUDA GPU; use kaggle_submit.py for CPU.")
    free_gib = []
    for index in range(gpu_count):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        free_gib.append(float(free_bytes) / (2**30))
        print(
            f"cuda:{index} {torch.cuda.get_device_name(index)}: "
            f"free={free_bytes / (2**30):.1f} GiB total={total_bytes / (2**30):.1f} GiB",
            flush=True,
        )
    usable = gpu_memory_plan(estimated_gib, free_gib)
    if len(usable) > 1:
        max_memory = {index: f"{max(1, int(budget))}GiB" for index, budget in usable.items()}
        kwargs = {"config": config, "device_map": "auto", "max_memory": max_memory, "dtype": dtype}
    elif gpu_count > 1:
        only_index = next(iter(usable))
        kwargs = {"config": config, "device_map": {"": f"cuda:{only_index}"}, "dtype": dtype}
    else:
        kwargs = {"config": config, "dtype": dtype}

    try:
        model = AutoModel.from_pretrained(base_dir, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModel.from_pretrained(base_dir, **kwargs)
    if not hasattr(model, "hf_device_map"):
        target = next(iter(usable))
        model = model.to(f"cuda:{target}")
    else:
        placements = model.hf_device_map.values()
        if any(str(device).lower() in {"cpu", "disk", "meta"} for device in placements):
            del model
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Automatic dispatch selected CPU/disk offload; stopping rather than silently "
                "changing the GPU recipe. Free more CUDA memory or use --only-8b."
            )
    model.eval()
    if hasattr(model, "norm"):
        model.norm = nn.Identity()
    print(f"loaded trunk {base_dir.name}: keep={keep_layers}, estimate={estimated_gib:.1f} GiB", flush=True)
    return model


def merge_lora(model: nn.Module, adapter_dir: Path) -> None:
    from safetensors.torch import load_file

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    scale = float(adapter_config["lora_alpha"]) / float(adapter_config["r"])
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    modules = dict(model.named_modules())
    merged = 0
    with torch.no_grad():
        for key, matrix_a in state.items():
            if ".lora_A." not in key:
                continue
            raw_name = key.split(".lora_A.", 1)[0]
            module = modules.get(raw_name)
            if module is None:
                suffix = raw_name[raw_name.find("layers."):] if "layers." in raw_name else raw_name
                matches = [value for name, value in modules.items() if name.endswith(suffix)]
                if len(matches) != 1:
                    raise KeyError(f"Cannot resolve LoRA target {raw_name}; matches={len(matches)}")
                module = matches[0]
            matrix_b = state[key.replace(".lora_A.", ".lora_B.")]
            delta = matrix_b.float().to(module.weight.device) @ matrix_a.float().to(module.weight.device)
            module.weight.add_((delta * scale).to(module.weight.dtype))
            merged += 1
    if merged == 0:
        raise ValueError(f"No LoRA matrices found in {adapter_dir}")
    print(f"merged {merged} LoRA matrices from {adapter_dir.name}", flush=True)


@torch.inference_mode()
def extract_lora_features(
    checkpoint_dir: Path,
    base_dir: Path,
    texts: list[str],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    meta = json.loads((checkpoint_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("labels") != LABELS:
        raise ValueError("Checkpoint label order differs from this script; refusing to mislabel predictions.")
    cfg = meta["cfg"]
    layers = list(cfg["layers"])
    capability_device = torch.cuda.current_device()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(capability_device)[0] >= 8 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = load_model(base_dir, int(cfg["keep"]), dtype)
    merge_lora(model, checkpoint_dir / "adapter")

    heads = load_file(str(checkpoint_dir / "heads.safetensors"))
    hidden = int(model.config.hidden_size)
    # With device_map sharding the embedding is normally on cuda:0, while the
    # final decoder block can be on another GPU. Inputs must start at the
    # embedding device; pooled hidden states are explicitly moved there before
    # applying the head weights.
    input_device = model.get_input_embeddings().weight.device
    head_device = input_device
    norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in layers]).to(head_device).float()
    norms.load_state_dict({key[len("norms."):]: value for key, value in heads.items() if key.startswith("norms.")})

    token_ids = tokenizer(
        list(texts),
        add_special_tokens=False,
        truncation=True,
        max_length=int(cfg["max_len"]),
    )["input_ids"]
    if tokenizer.eos_token_id is not None:
        max_len = int(cfg["max_len"])
        token_ids = [(ids + [tokenizer.eos_token_id])[:max_len] for ids in token_ids]
    order = np.argsort([len(ids) for ids in token_ids], kind="stable")
    features = np.zeros((len(token_ids), hidden * len(layers)), dtype=np.float32)

    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch = tokenizer.pad(
            {"input_ids": [token_ids[i] for i in indices]}, return_tensors="pt"
        ).to(head_device)
        hidden_states = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        finite = all(torch.isfinite(hidden_states[layer]).all() for layer in layers)
        if not finite:
            raise FloatingPointError(
                "Non-finite hidden states detected; refusing unsafe dtype mutation on a possibly sharded model. "
                "Retry on a GPU with a bf16-capable CUDA device or adjust the explicit model dtype."
            )

        mask = batch["attention_mask"].to(head_device).unsqueeze(-1).float()
        # The first token is the attention sink used by the public recipe.
        mask[:, 0] = 0
        denominator = mask.sum(1).clamp(min=1)
        pooled = []
        for layer, norm in zip(layers, norms):
            hidden = hidden_states[layer].to(head_device).float()
            pooled.append(norm((hidden * mask).sum(1) / denominator))
        features[indices] = torch.cat(pooled, dim=-1).cpu().numpy()
        del batch, hidden_states, pooled

    head_arrays = {
        key: value.float().cpu().numpy()
        for key, value in heads.items()
        if not key.startswith("norms.")
    }
    del model, tokenizer, norms, heads
    gc.collect()
    torch.cuda.empty_cache()
    return np.nan_to_num(features), head_arrays


def topic_clusters(test_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(test_texts)
    if n < 2:
        groups = np.ones(n, dtype=int)
        return groups, np.ones(n, dtype=bool)
    try:
        tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(test_texts)
        distance = np.clip(1.0 - (tfidf @ tfidf.T).toarray(), 0.0, None)
        np.fill_diagonal(distance, 0.0)
        groups = fcluster(
            linkage(squareform(distance, checks=False), method="average"),
            TWIN_THRESH,
            criterion="distance",
        )
    except ValueError:
        groups = np.arange(n, dtype=int) + 1
    sizes = np.bincount(groups)[groups]
    twin_rows = (sizes >= TWIN_MIN_SIZE) & (sizes <= TWIN_MAX_SIZE)
    print(
        f"topic clusters: singletons={int((sizes == 1).sum())}, "
        f"centered rows={int(twin_rows.sum())}/{n}",
        flush=True,
    )
    return groups, twin_rows


def apply_centered_twin(
    logit: np.ndarray,
    features: np.ndarray,
    heads: dict[str, np.ndarray],
    groups: np.ndarray,
    twin_rows: np.ndarray,
    weight: float,
) -> None:
    if not twin_rows.any():
        return
    centered = features.copy()
    for group in np.unique(groups[twin_rows]):
        rows = np.flatnonzero(groups == group)
        centered[rows] -= centered[rows].mean(axis=0)
    centered_logits = centered[twin_rows] @ heads["centered.weight"].T + heads["centered.bias"]
    logit[twin_rows] += weight * np.log(np.clip(softmax(centered_logits), 1e-9, 1.0))


def sinkhorn_balance(logit: np.ndarray, minimum_rows: int) -> np.ndarray:
    probabilities = softmax(logit)
    if len(probabilities) < minimum_rows:
        print(f"skip Sinkhorn balancing for only {len(probabilities)} test rows", flush=True)
        return probabilities
    target = np.full(K, len(probabilities) / K, dtype=np.float64)
    for _ in range(100):
        probabilities *= target / np.clip(probabilities.sum(axis=0), 1e-12, None)
        probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    return np.nan_to_num(probabilities)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/submission.parquet"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--only-8b", action="store_true", help="Use only the 8B branch")
    parser.add_argument("--only-14b", action="store_true", help="Use only the 14B branch")
    parser.add_argument("--no-twin", action="store_true", help="Disable centered twin evidence (recommended for independent control)")
    parser.add_argument("--no-balance", action="store_true", help="Disable Sinkhorn class balancing (recommended for independent control)")
    parser.add_argument("--allow-transductive", action="store_true", help="Allow test-batch clustering and uniform-prior balancing; requires separate rule review")
    parser.add_argument("--balance-min-rows", type=int, default=BALANCE_MIN_ROWS)
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=None,
        help="Optional private .npz archive of aligned member probabilities; never commit it",
    )
    # An explicitly pasted Kaggle notebook call may pass [] and should use the
    # independent control; normal CLI invocation with argv=None remains fail-closed.
    if argv == []:
        argv = ["--no-twin", "--no-balance"]
    args = parser.parse_args(argv)
    if args.only_8b and args.only_14b:
        parser.error("--only-8b and --only-14b are mutually exclusive")
    if args.balance_min_rows < 1:
        parser.error("--balance-min-rows must be positive")
    if not args.allow_transductive and not args.no_twin:
        parser.error("--no-twin is required unless --allow-transductive is set")
    if not args.allow_transductive and not args.no_balance:
        parser.error("--no-balance is required unless --allow-transductive is set")
    if args.scores_output is not None and args.scores_output.suffix.lower() != ".npz":
        parser.error("--scores-output must end in .npz")
    output = args.output.resolve()
    metadata = args.metadata.resolve() if args.metadata else output.with_suffix(".json")
    if output == metadata:
        parser.error("--metadata must not overwrite the submission")
    if args.scores_output is not None and args.scores_output.resolve() in {output, metadata}:
        parser.error("--scores-output must not overwrite the submission or metadata")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.time()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. For a CPU-only submission use scripts/kaggle_submit.py; "
            "the public 0.592 recipe needs attached Qwen3 LoRA inputs."
        )

    train_path, test_path = find_competition_data(input_dir)
    pool_path = find_pool(input_dir)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["text"] = train["text"].fillna("").astype(str)
    test["text"] = test["text"].fillna("").astype(str)
    if set(train["model"].astype(str).unique()) != set(LABELS):
        raise ValueError("Competition labels do not match the public checkpoint label set.")
    if not train["id"].is_unique or not test["id"].is_unique:
        raise ValueError("Duplicate IDs in competition input.")
    if len(test) < 50:
        raise ValueError("The visible test input is too small for a leaderboard submission; use a full Kaggle test set.")

    pool = pd.read_parquet(pool_path)
    print(f"train={train.shape}, test={test.shape}, pool={pool.shape}", flush=True)
    logit = np.zeros((len(test), K), dtype=np.float64)
    score_bank: dict[str, np.ndarray] = {}
    p_svc = fit_ngram_member(train, test, pool)
    if args.scores_output is not None:
        score_bank["svc"] = p_svc
    logit += W_SVC * np.log(np.clip(p_svc, 1e-9, 1.0))

    if args.allow_transductive:
        groups, twin_rows = topic_clusters(test["text"].tolist())
    else:
        groups = np.arange(len(test), dtype=int) + 1
        twin_rows = np.zeros(len(test), dtype=bool)
        print("transductive centered-twin clustering disabled", flush=True)
    if args.only_8b:
        selected = ["lora8b_v1"]
    elif args.only_14b:
        selected = ["lora14b_v1"]
    else:
        selected = list(LORA_CONFIG)
    for checkpoint_name in selected:
        config = LORA_CONFIG[checkpoint_name]
        checkpoint = find_checkpoint(input_dir, checkpoint_name)
        base = find_base_model(input_dir, config["base"])
        features, heads = extract_lora_features(
            checkpoint, base, test["text"].tolist(), int(config["bs"])
        )
        plain_logits = features @ heads["plain.weight"].T + heads["plain.bias"]
        p_member = softmax(plain_logits)
        if args.scores_output is not None:
            score_bank[checkpoint_name] = p_member
        logit += float(config["w"]) * np.log(np.clip(p_member, 1e-9, 1.0))
        if not args.no_twin:
            before_twin = logit.copy() if args.scores_output is not None else None
            apply_centered_twin(
                logit, features, heads, groups, twin_rows, float(config["twin_w"])
            )
            if before_twin is not None:
                score_bank[checkpoint_name + "_twin_log_contribution"] = logit - before_twin
        print(f"{checkpoint_name} done ({time.time() - started:.0f}s)", flush=True)
        del features, heads
        gc.collect()
        torch.cuda.empty_cache()

    probabilities = (
        softmax(logit)
        if args.no_balance
        else sinkhorn_balance(logit, max(1, int(args.balance_min_rows)))
    )
    if args.scores_output is not None:
        score_bank["blend_logit_before_balance"] = logit
        score_bank["final_probabilities"] = probabilities
        score_bank["topic_groups"] = groups
        score_bank["twin_rows"] = twin_rows
    prediction = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
    submission = pd.DataFrame({"id": test["id"].astype(str), "model": prediction.astype(str)})
    submission = submission[["id", "model"]]
    assert len(submission) == len(test)
    assert submission["id"].is_unique
    assert submission["id"].tolist() == test["id"].astype(str).tolist()
    assert submission["model"].isin(LABELS).all()
    submission.to_parquet(output_path, index=False)
    pd.testing.assert_frame_equal(submission, pd.read_parquet(output_path))

    metadata_path = args.metadata.resolve() if args.metadata else output_path.with_suffix(".json")
    if args.scores_output is not None:
        scores_path = args.scores_output.resolve()
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scores_path,
            ids=np.asarray(test["id"].astype(str).tolist(), dtype=str),
            labels=np.asarray(LABELS),
            **score_bank,
        )
        print(f"private score bank: {scores_path}", flush=True)
    metadata = {
        "method": "two_stage_lora_public_pool_8b_14b_plus_style_svc",
        "public_reference": "https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool",
        "reported_public_lb": 0.592,
        "reported_grouped_cv": 0.593,
        "reported_scores_are_upstream": True,
        "score_evidence_accessed_utc": "2026-10-22T00:00:00Z",
        "external_pool_overlap_checked": True,
        "near_duplicate_overlap_checked": False,
        "license_and_rules_confirmed_by_runner": False,
        "labels": LABELS,
        "selected_checkpoints": selected,
        "is_full_reference_recipe": args.allow_transductive and not args.only_8b and not args.only_14b and not args.no_twin and not args.no_balance and args.balance_min_rows == BALANCE_MIN_ROWS,
        "own_hidden_test_score": None,
        "weights": {"svc": W_SVC, **{k: LORA_CONFIG[k]["w"] for k in selected}},
        "twin_weights": {k: LORA_CONFIG[k]["twin_w"] if not args.no_twin else 0.0 for k in selected},
        "centered_twin": {
            "enabled": not args.no_twin,
            "threshold": TWIN_THRESH,
            "min_size": TWIN_MIN_SIZE,
            "max_size": TWIN_MAX_SIZE,
            "rows": int(twin_rows.sum()),
        },
        "sinkhorn": {
            "enabled": not args.no_balance,
            "minimum_rows": int(args.balance_min_rows),
            "applied": len(test) >= int(args.balance_min_rows) and not args.no_balance,
        },
        "only_8b": bool(args.only_8b),
        "only_14b": bool(args.only_14b),
        "allow_transductive": bool(args.allow_transductive),
        "score_bank_written": args.scores_output is not None,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "train_rows": len(train),
        "test_rows": len(test),
        "pool_rows_loaded": len(pool),
        "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path),
        "pool_sha256": sha256(pool_path),
        "input_paths_recorded": True,
        "output_path_recorded": True,
        "input_paths": {
            "train": str(train_path),
            "test": str(test_path),
            "pool": str(pool_path),
        },
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "seconds": time.time() - started,
        "uses_prompt_id_for_inference": False,
        "uses_hidden_test_labels": False,
        "uses_hungarian_assignment": False,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("prediction counts:")
    print(submission["model"].value_counts().to_string())
    print(f"submission: {output_path}", flush=True)
    print(f"metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    # When pasted into a Kaggle notebook, ignore the kernel's -f argument.
    main([] if "ipykernel" in sys.modules else None)
