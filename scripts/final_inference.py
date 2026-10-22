"""Portable local full-fit inference for Name That LLM.

The script expects a private directory containing train.csv and test.csv and
writes the Parquet submission to a private output path. It never uses prompt_id,
test-batch quotas, clustering, assignment, or hidden labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

FUNCTION_WORDS = set(
    "a an the and or but if while although because as so than that which who whom whose this these those it "
    "its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to "
    "from with without for of by over under between through into about is are was were be been being have has "
    "had do does did can could will would shall should may might must not no nor all any some each every both "
    "either neither other another such only also just very too more most less least here there when where why "
    "how what whether then now still yet even much many few little".split()
)
FEATURES = [
    ("char", 3, 5, False, True, 250000, 1.0),
    ("word", 1, 3, False, True, 180000, 0.7),
    ("shape", 2, 6, False, True, 60000, 0.5),
    ("function", 1, 3, False, True, 50000, 0.5),
]


def shape_text(text: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", str(text))


def function_text(text: str) -> str:
    return " ".join(
        token.lower() if token.lower() in FUNCTION_WORDS or not token.isalnum() else "CONTENT"
        for token in re.findall(r"\w+|[^\w\s]", str(text))
    )


def view_text(text: str, analyzer: str) -> str:
    if analyzer == "shape":
        return shape_text(text)
    if analyzer == "function":
        return function_text(text)
    return str(text)


def numeric_style(text: str) -> list[float]:
    text = str(text)
    length = max(len(text), 1)
    words = text.split()
    return [
        np.log1p(len(text)), np.log1p(len(words)), text.count("\n") / length * 100.0,
        text.count("\n\n"),
        *[text.count(ch) for ch in ["—", "–", ",", ".", ";", ":", '"', "'", "’", "“", "*", "#", "(", "-"]],
        sum(char.isupper() for char in text) / length,
        np.mean([len(word) for word in words]) if words else 0.0,
        len(re.findall(r"[.!?]", text)) / max(len(words), 1),
        *[text.count(ch) for ch in ["!", "?", "…", "..."]],
    ]


def build_features(train_texts: list[str], test_texts: list[str]):
    train_parts, test_parts = [], []
    for analyzer, lo, hi, lowercase, sublinear, max_features, weight in FEATURES:
        actual = "char" if analyzer == "shape" else "word" if analyzer == "function" else analyzer
        options = dict(
            analyzer=actual, ngram_range=(lo, hi), lowercase=lowercase,
            sublinear_tf=sublinear, min_df=2, max_features=max_features, dtype=np.float64,
        )
        if actual == "word":
            options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vectorizer = TfidfVectorizer(**options)
        train_parts.append(vectorizer.fit_transform([view_text(x, analyzer) for x in train_texts]) * weight)
        test_parts.append(vectorizer.transform([view_text(x, analyzer) for x in test_texts]) * weight)

    style_train = np.asarray([numeric_style(x) for x in train_texts], dtype=np.float64)
    style_test = np.asarray([numeric_style(x) for x in test_texts], dtype=np.float64)
    scaler = StandardScaler().fit(style_train)
    train_parts.append(csr_matrix(scaler.transform(style_train) * 0.3))
    test_parts.append(csr_matrix(scaler.transform(style_test) * 0.3))
    return hstack(train_parts, format="csr"), hstack(test_parts, format="csr")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Private directory containing train.csv and test.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="Private output submission.parquet path")
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    assert {"id", "prompt_id", "text", "model"}.issubset(train.columns)
    assert {"id", "text"}.issubset(test.columns)
    assert train.id.is_unique and test.id.is_unique
    assert train[["id", "text", "model"]].notna().all().all()
    assert test[["id", "text"]].notna().all().all()
    classes = np.array(sorted(train.model.astype(str).unique()), dtype=object)
    assert len(classes) == 18

    start = time.perf_counter()
    X, T = build_features(train.text.astype(str).tolist(), test.text.astype(str).tolist())
    clf = LinearSVC(C=0.5, random_state=42, max_iter=5000).fit(X, train.model.astype(str))
    pred = clf.predict(T).astype(str)
    submission = pd.DataFrame({"id": test.id.astype(str), "model": pred})[["id", "model"]]
    assert len(submission) == len(test)
    assert submission.id.is_unique
    assert submission.id.tolist() == test.id.astype(str).tolist()
    assert submission.model.notna().all() and submission.model.isin(classes).all()
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output, index=False)
    pd.testing.assert_frame_equal(submission, pd.read_parquet(output))

    metadata_path = args.metadata.resolve() if args.metadata else output.with_suffix(".json")
    metadata_path.write_text(json.dumps({
        "recipe": "char35_styletext_plus_numeric_style__svc_0p5",
        "data_dir": str(data_dir), "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path), "output": str(output),
        "output_sha256": sha256(output), "train_rows": len(train),
        "test_rows": len(test), "columns": submission.columns.tolist(),
        "uses_prompt_id_for_inference": False,
        "uses_test_grouping_or_assignment": False,
        "seconds": time.perf_counter() - start,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"submission": str(output), "rows": len(submission), "metadata": str(metadata_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
