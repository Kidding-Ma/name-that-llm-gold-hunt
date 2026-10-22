"""Diagnostic-only blend probe for private OOF score banks.

This script requires private train/evidence artifacts and writes nothing. Use it
to inspect a blend hypothesis; its best weight is not an unbiased estimate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold


def row_softmax(x):
    z = np.asarray(x, dtype=np.float64)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def col_z(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Private directory containing train.csv")
    parser.add_argument("--evidence-dir", type=Path, required=True,
                        help="Private directory containing the two OOF score banks")
    args = parser.parse_args()
    data_dir, evidence_dir = args.data_dir.resolve(), args.evidence_dir.resolve()
    train = pd.read_csv(data_dir / "train.csv")
    labels = np.sort(train.model.astype(str).unique())
    y = np.searchsorted(labels, train.model.astype(str).to_numpy())
    groups = train.prompt_id.to_numpy()
    style = np.load(evidence_dir / "style_aware_local_oof_scores.npy")
    basez = np.load(evidence_dir / "prompt_grouped_margins.npz")
    base = basez["margins"]
    base_classes = basez["classes"].astype(str)
    base = base[:, np.array([np.flatnonzero(base_classes == c)[0] for c in labels])]
    if style.shape != base.shape or style.shape[0] != len(train):
        raise ValueError(f"score-bank shape mismatch: style={style.shape}, base={base.shape}, rows={len(train)}")
    print("shapes", style.shape, base.shape, "labels", len(labels))

    def ba(scores):
        return balanced_accuracy_score(y, np.asarray(scores).argmax(1))

    print("style", ba(style), "base", ba(base))
    records = []
    for mode, a, b in (("raw", style, base), ("colz", col_z(style), col_z(base)),
                       ("softmax", row_softmax(style), row_softmax(base))):
        for weight in np.linspace(0, 1, 21):
            records.append((ba(weight * a + (1 - weight) * b), mode, float(weight)))
    print("best_direct", sorted(records, reverse=True)[:12])

    fold_of = np.full(len(train), -1, dtype=int)
    for fold, (_, valid) in enumerate(GroupKFold(n_splits=5).split(train.text, y, groups)):
        fold_of[valid] = fold
    for mode, a, b in (("raw", style, base), ("colz", col_z(style), col_z(base)),
                       ("softmax", row_softmax(style), row_softmax(base))):
        fold_scores, chosen = [], []
        for fold in range(5):
            fit = fold_of != fold
            valid = ~fit
            candidates = [
                (balanced_accuracy_score(y[fit], (weight * a[fit] + (1 - weight) * b[fit]).argmax(1)), float(weight))
                for weight in np.linspace(0, 1, 41)
            ]
            best = max(candidates)
            chosen.append(best[1])
            fold_scores.append(balanced_accuracy_score(y[valid], (best[1] * a[valid] + (1 - best[1]) * b[valid]).argmax(1)))
        print("heldout", mode, "weights", chosen, "fold_ba", fold_scores, "mean", float(np.mean(fold_scores)))


if __name__ == "__main__":
    main()
