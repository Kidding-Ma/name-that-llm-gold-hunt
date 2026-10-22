# Name That LLM - v3: plain TF-IDF + plain Qwen3-1.7B-Base embedding heads, log-prob blend.
# Optional (ADAPT): prompt-centered models for passages whose test topic-cluster has >= ADAPT_MIN members.
# Single file: paste into one Kaggle notebook cell.
# Notebook settings: GPU on, internet off, add input model "qwen-lm/qwen-3" (Transformers, 1.7b-base).
ADAPT = False                       # use centered models inside test clusters of size >= ADAPT_MIN
ADAPT_MIN = 4
THRESH = 0.9                        # average-linkage cosine distance cut for topic clusters
TFIDF_C, TFIDF_MAXF = 10.0, 30000   # plain TF-IDF (identical to the LB-0.347 control)
TFIDF_C_CTR, TFIDF_MAXF_CTR = 3.0, 30000
LLM_LAYERS = [17, 20]               # mean-pooled hidden states of Qwen3-1.7B-Base (28 layers)
LLM_C = 0.005
LLM_C_CTR = 0.002
W_TFIDF, W_LAYER = 1.082, {17: 0.352, 20: 0.356}   # log-prob blend weights, fit out-of-fold (CV 0.431)

import glob
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

T0 = time.time()
LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
DATA_DIR = next(p for p in [pathlib.Path('/kaggle/input/competitions/name-that-llm'),
                            pathlib.Path('/kaggle/input/name-that-llm'),
                            pathlib.Path('data')] if (p / 'train.csv').exists())
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
test['text'] = test['text'].fillna('').astype(str)
y = train['model'].map({l: i for i, l in enumerate(LABELS)}).values
print('train', train.shape, 'test', test.shape, flush=True)


def log_proba(clf, X):
    p = np.full((X.shape[0], K), 1e-9)
    p[:, clf.classes_] = clf.predict_proba(X)
    return np.log(np.clip(p, 1e-9, 1))


def topic_clusters(texts, thresh=THRESH):
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts)
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), method='average'), thresh, criterion='distance')


def group_center(X, groups):
    X = np.asarray(X.toarray() if hasattr(X, 'toarray') else X, dtype=np.float32).copy()
    for g in np.unique(groups):
        i = groups == g
        X[i] -= X[i].mean(0)
    return X


g_tr = train['prompt_id'].values
if ADAPT:
    g_te = topic_clusters(test['text'].values)
    big = np.bincount(g_te)[g_te] >= ADAPT_MIN
    print('test clusters', len(np.unique(g_te)), 'passages in clusters >=', ADAPT_MIN, ':', int(big.sum()), flush=True)

# ---- 1) TF-IDF (plain; centered variant only if ADAPT) ----
feats = FeatureUnion([
    ('word', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                             token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=2, max_features=TFIDF_MAXF)),
    ('char', TfidfVectorizer(analyzer='char', ngram_range=(1, 5), sublinear_tf=True, lowercase=False, min_df=2,
                             max_features=TFIDF_MAXF)),
])
Xtr = feats.fit_transform(train['text'])
Xte = feats.transform(test['text'])
lp_tfidf = log_proba(LogisticRegression(C=TFIDF_C, max_iter=3000).fit(Xtr, y), Xte)
if ADAPT and big.any():
    fc = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                                 token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=2, max_features=TFIDF_MAXF_CTR)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(1, 5), sublinear_tf=True, lowercase=False,
                                 min_df=2, max_features=TFIDF_MAXF_CTR)),
    ])
    Ctr = group_center(fc.fit_transform(train['text']), g_tr)
    clf = LogisticRegression(C=TFIDF_C_CTR, max_iter=3000).fit(Ctr, y)
    Cte = group_center(fc.transform(test['text'][big]), g_te[big])
    lp_tfidf[big] = log_proba(clf, Cte)
    del Ctr, Cte
del Xtr, Xte
print(f'tfidf done ({time.time() - T0:.0f}s)', flush=True)


# ---- 2) Qwen3 embeddings ----
def find_model_dir():
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            with open(cfg, encoding='utf-8') as handle:
                c = json.load(handle)
        except Exception:
            continue
        if c.get('model_type') == 'qwen3' and c.get('hidden_size') == 2048 and c.get('num_hidden_layers') == 28:
            return os.path.dirname(cfg)
    raise FileNotFoundError(
        'No attached local Qwen3-1.7B Transformers input found under /kaggle/input. '
        'Attach the model input; this release does not download weights at runtime.'
    )


@torch.no_grad()
def embed(texts, model, tok, layers, bs=8, max_len=512):
    out = np.zeros((len(texts), len(layers), model.config.hidden_size), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    for b0 in range(0, len(texts), bs):
        idx = order[b0:b0 + bs]
        batch = [(tok.bos_token + texts[i]) if tok.bos_token else texts[i] for i in idx]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc['attention_mask'].unsqueeze(-1).float()
        for li, L in enumerate(layers):
            out[idx, li] = ((hs[L].float() * m).sum(1) / m.sum(1)).cpu().numpy()
    return np.nan_to_num(out)


mdir = find_model_dir()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if dev == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
print('LLM', mdir, dev, dtype, torch.cuda.get_device_name(0) if dev == 'cuda' else '', flush=True)
tok = AutoTokenizer.from_pretrained(mdir)
tok.padding_side = 'right'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
try:
    llm = AutoModel.from_pretrained(mdir, dtype=dtype)
except TypeError:  # transformers < 4.56 uses torch_dtype
    llm = AutoModel.from_pretrained(mdir, torch_dtype=dtype)
llm = llm.to(dev).eval()
E_tr = embed(list(train['text']), llm, tok, LLM_LAYERS)
E_te = embed(list(test['text']), llm, tok, LLM_LAYERS)
del llm
if dev == 'cuda':
    torch.cuda.empty_cache()
print(f'embeddings done ({time.time() - T0:.0f}s)', flush=True)

logit = W_TFIDF * lp_tfidf
for li, L in enumerate(LLM_LAYERS):
    sc = StandardScaler().fit(E_tr[:, li])
    clf = LogisticRegression(C=LLM_C, max_iter=5000).fit(sc.transform(E_tr[:, li]), y)
    lp = log_proba(clf, sc.transform(E_te[:, li]))
    if ADAPT and big.any():
        Ctr = group_center(E_tr[:, li], g_tr)
        sc2 = StandardScaler().fit(Ctr)
        clf2 = LogisticRegression(C=LLM_C_CTR, max_iter=5000).fit(sc2.transform(Ctr), y)
        lp[big] = log_proba(clf2, sc2.transform(group_center(E_te[big, li], g_te[big])))
    logit += W_LAYER[L] * lp
print(f'llm heads done ({time.time() - T0:.0f}s)', flush=True)

pred = logit.argmax(1)
sub = pd.DataFrame({'id': test['id'], 'model': [LABELS[i] for i in pred]})
assert sub['id'].is_unique and sub['model'].isin(LABELS).all() and len(sub) == len(test)
output_path = pathlib.Path('/kaggle/working/submission.parquet')
sub.to_parquet(output_path, index=False)
print(sub['model'].value_counts().to_string())
print('submission:', output_path)
print(f'total {time.time() - T0:.0f}s')
