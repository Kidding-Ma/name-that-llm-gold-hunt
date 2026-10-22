"""Kaggle-ready inference for the strongest directly verified public recipe.

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
or hidden labels. It uses only the public pool, pretrained model inputs, test
text, and the two public LoRA checkpoints. Centered twins are applied only to
content-TF-IDF clusters of size 2--18. Sinkhorn balancing is skipped for tiny
interactive test files such as the local 3-row example.

Usage as a Kaggle notebook cell
--------------------------------
Paste the whole file into a Kaggle notebook cell and run it. The notebook
runtime has ipykernel in sys.modules, so command-line parsing is disabled.

Usage as a Python file
----------------------
    python kaggle_best_two_stage_lora.py
    python kaggle_best_two_stage_lora.py --only-8b
    python kaggle_best_two_stage_lora.py --no-twin --no-balance

The normal/full recipe is the first command. `--only-8b` is a smoke/ablation
mode and is not the reported 0.592 blend.
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
    """Find competition train/test rather than an auxiliary dataset's CSVs."""
    train_path = None
    for candidate in sorted(input_dir.rglob("train.csv")):
        try:
            header = pd.read_csv(candidate, nrows=2)
        except Exception:
            continue
        if {"id", "prompt_id", "text", "model"}.issubset(header.columns):
            train_path = candidate
            break
    if train_path is None:
        raise FileNotFoundError(
            "Could not find competition train.csv with id,prompt_id,text,model. "
            "Attach the Name That LLM competition data."
        )

    same_dir = train_path.parent / "test.csv"
    candidates = [same_dir] if same_dir.exists() else []
    candidates += [p for p in sorted(input_dir.rglob("test.csv")) if p != same_dir]
    for candidate in candidates:
        try:
            header = pd.read_csv(candidate, nrows=2)
        except Exception:
            continue
        if {"id", "text"}.issubset(header.columns) and "model" not in header.columns:
            return train_path, candidate
    raise FileNotFoundError("Could not find competition test.csv with id,text columns.")


def find_pool(input_dir: Path) -> Path:
    candidates = []
    for candidate in sorted(input_dir.rglob("pool_v1.parquet")):
        try:
            header = pd.read_parquet(candidate, columns=None).head(1)
        except Exception:
            continue
        if {"text", "model"}.issubset(header.columns):
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            "Could not find pool_v1.parquet. Attach the public dataset "
            "nishantdahal/ntl-two-stage-lora."
        )
    return candidates[0]


def find_checkpoint(input_dir: Path, name: str) -> Path:
    candidates = sorted(
        p.parent for p in input_dir.rglob("meta.json") if p.parent.name == name
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {name}/meta.json. Attach nishantdahal/ntl-two-stage-lora."
        )
    return candidates[0]


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
    if not candidates:
        raise FileNotFoundError(
            f"Could not find attached base model matching {spec}. Add the corresponding "
            "Qwen3 Transformers model input."
        )
    return candidates[0]


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
    pool = pool[pool["model"].isin(LABEL_TO_ID)].drop_duplicates(subset=["text"]).reset_index(drop=True)
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
    if estimated_gib > 13 and gpu_count > 1:
        max_memory = {i: "13GiB" for i in range(gpu_count)}
        kwargs = {"config": config, "device_map": "auto", "max_memory": max_memory, "dtype": dtype}
    elif estimated_gib > 15 and gpu_count == 1:
        raise RuntimeError(
            f"Attached {config.hidden_size}-wide trunk needs about {estimated_gib:.1f} GiB; "
            "use a 2x-T4 Kaggle session."
        )
    else:
        kwargs = {"config": config, "dtype": dtype}

    try:
        model = AutoModel.from_pretrained(base_dir, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModel.from_pretrained(base_dir, **kwargs)
    if not hasattr(model, "hf_device_map"):
        model = model.to("cuda:0")
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
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

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
        token_ids = [ids + [tokenizer.eos_token_id] for ids in token_ids]
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
            print("non-finite fp16 batch; retrying in bf16", flush=True)
            model.to(torch.bfloat16)
            hidden_states = model(**batch, output_hidden_states=True, use_cache=False).hidden_states

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
    parser.add_argument("--only-8b", action="store_true", help="Ablation/smoke mode; not the reported full blend")
    parser.add_argument("--no-twin", action="store_true", help="Disable centered twin evidence")
    parser.add_argument("--no-balance", action="store_true", help="Disable Sinkhorn class balancing")
    parser.add_argument("--balance-min-rows", type=int, default=BALANCE_MIN_ROWS)
    return parser.parse_args(argv)


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

    pool = pd.read_parquet(pool_path)
    print(f"train={train.shape}, test={test.shape}, pool={pool.shape}", flush=True)
    logit = np.zeros((len(test), K), dtype=np.float64)
    p_svc = fit_ngram_member(train, test, pool)
    logit += W_SVC * np.log(np.clip(p_svc, 1e-9, 1.0))

    groups, twin_rows = topic_clusters(test["text"].tolist())
    selected = ["lora8b_v1"] if args.only_8b else list(LORA_CONFIG)
    for checkpoint_name in selected:
        config = LORA_CONFIG[checkpoint_name]
        checkpoint = find_checkpoint(input_dir, checkpoint_name)
        base = find_base_model(input_dir, config["base"])
        features, heads = extract_lora_features(
            checkpoint, base, test["text"].tolist(), int(config["bs"])
        )
        plain_logits = features @ heads["plain.weight"].T + heads["plain.bias"]
        logit += float(config["w"]) * np.log(np.clip(softmax(plain_logits), 1e-9, 1.0))
        if not args.no_twin:
            apply_centered_twin(
                logit, features, heads, groups, twin_rows, float(config["twin_w"])
            )
        print(f"{checkpoint_name} done ({time.time() - started:.0f}s)", flush=True)
        del features, heads
        gc.collect()
        torch.cuda.empty_cache()

    probabilities = (
        softmax(logit)
        if args.no_balance
        else sinkhorn_balance(logit, max(1, int(args.balance_min_rows)))
    )
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
    metadata = {
        "method": "two_stage_lora_public_pool_8b_14b_plus_style_svc",
        "public_reference": "https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool",
        "reported_public_lb": 0.592,
        "reported_grouped_cv": 0.593,
        "labels": LABELS,
        "weights": {"svc": W_SVC, **{k: v["w"] for k, v in LORA_CONFIG.items()}},
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
        "train_rows": len(train),
        "test_rows": len(test),
        "pool_rows_loaded": len(pool),
        "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path),
        "pool_sha256": sha256(pool_path),
        "input_paths_recorded": False,
        "output_path_recorded": False,
        "output_sha256": sha256(output_path),
        "seconds": time.time() - started,
        "uses_prompt_id_for_inference": False,
        "uses_hidden_test_labels": False,
        "uses_hungarian_assignment": False,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("prediction counts:")
    print(submission["model"].value_counts().to_string())
    print(f"submission: {output_path}", flush=True)
    print(f"metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    # When pasted into a Kaggle notebook, ignore the kernel's -f argument.
    main([] if "ipykernel" in sys.modules else None)
