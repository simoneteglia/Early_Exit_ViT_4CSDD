from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def binary_metrics(labels, preds, scores=None) -> dict:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "cm": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "n": int(len(labels)),
    }
    if scores is not None and len(np.unique(labels)) == 2:
        out["auc"] = float(roc_auc_score(labels, scores))
    return out


def expected_calibration_error(labels, scores, n_bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - scores[mask].mean())
    return float(ece)


def per_exit_metrics(labels, exit_scores, threshold: float = 0.5) -> dict[int, dict]:
    """exit_scores: array (n_exits, N) of P(fake)."""
    return {e: binary_metrics(labels, (s > threshold).astype(int), s)
            for e, s in enumerate(np.asarray(exit_scores))}


class ScoreCollector:
    """Accumulates labels, content-sensitivity scores and per-exit P(fake)."""

    def __init__(self):
        self.labels, self.css, self.scores = [], [], []

    def add(self, labels, css, outputs):
        if torch.is_tensor(outputs):
            outputs = [outputs]
        self.labels.append(np.asarray(labels.detach().cpu()))
        self.css.append(np.asarray(css.detach().cpu()))
        self.scores.append(np.stack([torch.sigmoid(o.detach().float()).reshape(-1).cpu().numpy()
                                     for o in outputs]))

    def arrays(self):
        return (np.concatenate(self.labels), np.concatenate(self.css),
                np.concatenate(self.scores, axis=1))
