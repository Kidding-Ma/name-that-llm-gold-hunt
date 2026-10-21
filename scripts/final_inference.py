"""Offline final inference for Name That LLM.

Fits the reviewed style-aware four-view TF-IDF + LinearSVC recipe on all
training rows and predicts every row in test.csv independently. It does not
use prompt_id, test-batch quotas, clustering, assignment, or hidden labels.

Examples
--------
python final_inference.py
python final_inference.py --data-dir D:/Kaggle/NameThatLLM/data --output submission.parquet
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]
DEPS = PROJECT / "research" / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
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
        token.lower() if token.lower() in FUNCTION_WORDS or token.isalnum() is False else "CONTENT"
        for token in re.findall(r"\w+|[^\w\s]", str(text))
    )


def view_text(text: str, analyzer: str) -> str:
    if analyzer == "shape":
        return shape_text(text)
    if analyzer == "function":
        return function_text(text)
    return str(text)


def build_features(train_texts, test_texts):
    train_parts, test_parts = [], []
    for analyzer, lo, hi, lowercase, sublinear, max_features, weight in FEATURES:
        actual = "char" if analyzer == "shape" else "word" if analyzer == "function" else analyzer
        options = {
            "analyzer": actual,
            "ngram_range": (lo, hi),
            "lowercase": lowercase,
            "sublinear_tf": sublinear,
            "min_df": 2,
            "max_features": max_features,
            "dtype": np.float64,
        }
        if actual == "word":
            options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vectorizer = TfidfVectorizer(**options)
        train_view = [view_text(x, analyzer) for x in train_texts]
        test_view = [view_text(x, analyzer) for x in test_texts]
        train_parts.append(vectorizer.fit_transform(train_view) * weight)
        test_parts.append(vectorizer.transform(test_view) * weight)
    return hstack(train_parts, format="csr"), hstack(test_parts, format="csr")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT / "research" / "evidence" / "submission.parquet")
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
    clf = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(X, train.model.astype(str))
    pred = clf.predict(T).astype(str)
    submission = pd.DataFrame({"id": test.id.astype(str), "model": pred})[["id", "model"]]
    assert len(submission) == len(test)
    assert submission.id.is_unique
    assert submission.id.tolist() == test.id.astype(str).tolist()
    assert submission.model.notna().all()
    assert submission.model.isin(classes).all()
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output, index=False)
    reread = pd.read_parquet(output)
    pd.testing.assert_frame_equal(submission, reread)
    meta = {
        "recipe": "char35_styletext__svc_0p5",
        "data_dir": str(data_dir),
        "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path),
        "submission": str(output),
        "submission_sha256": sha256(output),
        "train_rows": len(train),
        "test_rows": len(test),
        "classes": classes.tolist(),
        "columns": submission.columns.tolist(),
        "uses_prompt_id_for_inference": False,
        "uses_test_grouping_or_assignment": False,
        "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__, "sklearn": __import__("sklearn").__version__},
        "seconds": time.perf_counter() - start,
    }
    metadata = args.metadata.resolve() if args.metadata else output.with_suffix(".json")
    metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
