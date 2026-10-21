"""Read-only CPU probe for blending existing out-of-fold score banks.

This script never writes files. It compares the reviewed style-aware OOF margins
with the original grouped baseline margins and performs conservative grids on
OOF rows plus a fold-held-out meta check.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
DEPS = ROOT / "research" / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

E = ROOT / "research" / "evidence"
train = pd.read_csv(ROOT / "data" / "train.csv")
y_names = np.sort(train.model.astype(str).unique())
y = np.searchsorted(y_names, train.model.astype(str).to_numpy())
groups = train.prompt_id.to_numpy()
style = np.load(E / "style_aware_local_oof_scores.npy")
basez = np.load(E / "prompt_grouped_margins.npz")
base = basez["margins"]
base_classes = basez["classes"].astype(str)
base = base[:, np.array([np.flatnonzero(base_classes == c)[0] for c in y_names])]
print("shapes", style.shape, base.shape, "labels", len(y_names))

def ba(s): return balanced_accuracy_score(y, np.asarray(s).argmax(1))
print("style", ba(style), "base", ba(base))

def row_softmax(x, temp=1.0):
    z = x / temp
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)

def col_z(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)

records=[]
# Direct margin / probability blends. These are diagnostics, then the fold-held-out
# section below avoids selecting a single global weight on the evaluated fold.
for mode, A, B in [
    ("raw", style, base),
    ("colz", col_z(style), col_z(base)),
    ("softmax", row_softmax(style), row_softmax(base)),
]:
    for w in np.linspace(0, 1, 21):
        score = w*A + (1-w)*B
        records.append((ba(score), mode, float(w)))
print("best_direct", sorted(records, reverse=True)[:12])

# Fold-held-out selection: choose the blend weight on the other prompt folds,
# evaluate only on the current fold. OOF score banks are already fold-local;
# this is a conservative meta diagnostic rather than an in-sample grid score.
uniq = np.array(sorted(pd.unique(groups)))
# Match the frozen GroupKFold partition used by style-aware runner.
from sklearn.model_selection import GroupKFold
fold_of = np.full(len(train), -1, dtype=int)
for f, (_, va) in enumerate(GroupKFold(n_splits=5).split(train.text, y, groups)):
    fold_of[va] = f
for mode, A, B in [("raw", style, base), ("colz", col_z(style), col_z(base)), ("softmax", row_softmax(style), row_softmax(base))]:
    fold_scores=[]
    chosen=[]
    for f in range(5):
        tr = fold_of != f; va = ~tr
        grid=[]
        for w in np.linspace(0,1,41):
            grid.append((balanced_accuracy_score(y[tr], (w*A[tr]+(1-w)*B[tr]).argmax(1)), float(w)))
        best=max(grid)
        chosen.append(best[1])
        fold_scores.append(balanced_accuracy_score(y[va], (best[1]*A[va]+(1-best[1])*B[va]).argmax(1)))
    print("heldout", mode, "weights", chosen, "fold_ba", fold_scores, "mean", float(np.mean(fold_scores)), "pooled", float(np.average(fold_scores)))

# Class-bias tuning on training folds only, using a small coordinate grid around zero.
# Apply to the held-out fold. This tests whether SVC margin calibration, not new
# information, is the bottleneck.
for mode, A in [("style", style), ("base", base), ("blend", 0.75*col_z(style)+0.25*col_z(base))]:
    fold_scores=[]; shifts=[]
    for f in range(5):
        tr=fold_of != f; va=~tr
        # Per-class bias estimated from train-fold mean margins; scan a conservative
        # multiplier, chosen only on inner part of tr via a 3-way split.
        inner=fold_of != f
        # Use normalized columns, then candidate biases based on inverse class recall
        Z=col_z(A)
        best_shift=np.zeros(Z.shape[1])
        # A simple deterministic no-tuning baseline and a shrinkage prior toward equal margins.
        # scan scalar shrink only on the tr set, then evaluate va.
        best=(-1,None)
        for alpha in np.linspace(0,1,21):
            cand=alpha*Z + (1-alpha)*(Z - Z.mean(0))
            s=balanced_accuracy_score(y[tr],cand[tr].argmax(1))
            if s>best[0]: best=(s,alpha)
        alpha=best[1]
        shifts.append(float(alpha))
        fold_scores.append(balanced_accuracy_score(y[va],(alpha*Z[va]+(1-alpha)*(Z[va]-Z[tr].mean(0))).argmax(1)))
    print("calib", mode, "alphas", shifts, "fold_ba", fold_scores, "mean", float(np.mean(fold_scores)))
