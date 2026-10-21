"""Kaggle-ready CPU candidate: style-aware + original SVC probability blend.

Paste this file into a Kaggle Notebook attached to Name That LLM. It reads the
complete hidden test supplied by Kaggle and writes /kaggle/working/submission.parquet.
The blend is independent per row: it does not use prompt IDs, test quotas,
Hungarian assignment, Sinkhorn balancing, or hidden labels.

This candidate is based on a local diagnostic probe. The style branch receives
weight 0.70 and the original char+word+scalar-style branch 0.30 after applying
row-wise softmax to each LinearSVC decision matrix. It must be treated as a
separate Kaggle experiment, not as a guaranteed improvement.
"""
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "2")

CANDIDATES = [
    Path("/kaggle/input/competitions/name-that-llm"),
    Path("/kaggle/input/name-that-llm"),
]
DATA_DIR = next((p for p in CANDIDATES if (p / "train.csv").exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError("找不到比赛数据。请添加 Name That LLM 竞赛数据输入。")
OUT = Path("/kaggle/working")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
assert {"id", "prompt_id", "text", "model"}.issubset(train.columns)
assert {"id", "text"}.issubset(test.columns)
assert train.id.is_unique and test.id.is_unique
assert train[["id", "text", "model"]].notna().all().all()
assert test[["id", "text"]].notna().all().all()

classes = np.array(sorted(train.model.astype(str).unique()), dtype=object)
assert len(classes) == 18
y = np.searchsorted(classes, train.model.astype(str).to_numpy())
train_texts = train.text.fillna("").astype(str).tolist()
test_texts = test.text.fillna("").astype(str).tolist()

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())


def shape_text(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", str(text))


def function_text(text):
    return " ".join(
        token.lower() if token.lower() in FUNCTION_WORDS or not token.isalnum() else "CONTENT"
        for token in re.findall(r"\w+|[^\w\s]", str(text))
    )


def view_text(text, analyzer):
    if analyzer == "shape":
        return shape_text(text)
    if analyzer == "function":
        return function_text(text)
    return str(text)


def row_softmax(decision):
    z = np.asarray(decision, dtype=np.float64)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def align(decision, clf):
    out = np.zeros((len(decision), len(classes)), dtype=np.float64)
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = decision[:, j]
    return out

# ---------------- style-aware four-view SVC ----------------
STYLE_VIEWS = [
    ("char", 3, 5, 250000, 1.0),
    ("word", 1, 3, 180000, 0.7),
    ("shape", 2, 6, 60000, 0.5),
    ("function", 1, 3, 50000, 0.5),
]
style_train, style_test = [], []
for analyzer, lo, hi, max_features, weight in STYLE_VIEWS:
    actual = "char" if analyzer == "shape" else "word" if analyzer == "function" else analyzer
    options = dict(
        analyzer=actual, ngram_range=(lo, hi), lowercase=False,
        sublinear_tf=True, min_df=2, max_features=max_features, dtype=np.float64,
    )
    if actual == "word":
        options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
    vectorizer = TfidfVectorizer(**options)
    style_train.append(vectorizer.fit_transform([view_text(x, analyzer) for x in train_texts]) * weight)
    style_test.append(vectorizer.transform([view_text(x, analyzer) for x in test_texts]) * weight)
    print("完成 style 分支:", analyzer, "维度:", style_train[-1].shape[1])
style_clf = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(style_train, format="csr"), y)
style_prob = row_softmax(align(style_clf.decision_function(hstack(style_test, format="csr")), style_clf))

# ---------------- original char+word+scalar-style SVC ----------------
def scalar_style(text):
    text = str(text)
    length = max(len(text), 1)
    words = text.split()
    return [
        np.log1p(len(text)), np.log1p(len(words)), text.count("\n") / length * 100,
        text.count("\n\n"),
        *[text.count(ch) for ch in ["—", "–", ",", ".", ";", ":", '"', "'", "’", "“", "*", "#", "(", "-"]],
        sum(c.isupper() for c in text) / length,
        np.mean([len(w) for w in words]) if words else 0.0,
        len(re.findall(r"[.!?]", text)) / max(len(words), 1),
        *[text.count(ch) for ch in ["!", "?", "…", "..."]],
    ]

char = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True,
                       lowercase=False, min_df=2, max_features=300000)
word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                       lowercase=False, min_df=2, token_pattern=r"\S+")
S_train = np.asarray([scalar_style(x) for x in train_texts], dtype=np.float64)
S_test = np.asarray([scalar_style(x) for x in test_texts], dtype=np.float64)
scaler = StandardScaler().fit(S_train)
original_train = hstack([
    char.fit_transform(train_texts), word.fit_transform(train_texts), scaler.transform(S_train) * 0.3,
], format="csr")
original_test = hstack([
    char.transform(test_texts), word.transform(test_texts), scaler.transform(S_test) * 0.3,
], format="csr")
original_clf = LinearSVC(C=1.0, random_state=42, max_iter=5000).fit(original_train, y)
original_prob = row_softmax(align(original_clf.decision_function(original_test), original_clf))

STYLE_WEIGHT = 0.70
blend_prob = STYLE_WEIGHT * style_prob + (1.0 - STYLE_WEIGHT) * original_prob
pred = classes[blend_prob.argmax(axis=1)]
submission = pd.DataFrame({"id": test.id.astype(str), "model": pred.astype(str)})[["id", "model"]]
assert len(submission) == len(test)
assert submission.id.is_unique
assert submission.id.tolist() == test.id.astype(str).tolist()
assert submission.model.notna().all() and submission.model.isin(classes).all()

output_path = OUT / "submission.parquet"
submission.to_parquet(output_path, index=False)
pd.testing.assert_frame_equal(submission, pd.read_parquet(output_path))
print("style top-1:", classes[style_prob.argmax(axis=1)].tolist())
print("original top-1:", classes[original_prob.argmax(axis=1)].tolist())
print("blend top-1:", pred.tolist())
print("style weight:", STYLE_WEIGHT)
print("训练行数:", len(train), "测试行数:", len(test))
print("提交文件:", output_path)
print(submission.head(10).to_string(index=False))
print("提交文件已通过回读检查，可以点击 Submit to Competition。")
