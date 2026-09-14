"""Score calibration and base-threshold search (importable module so fitted
calibrators pickle/unpickle across scripts)."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score


class Calibrator:
    """Uniform wrapper: ``fit(scores_1d, labels)``, ``predict(scores_1d) -> 1d``.
    Beta calibration (as in ETA-DyNN) when ``betacal`` is installed, else isotonic."""

    def __init__(self, method: str = "beta"):
        self.method = method
        self.model = None
        if method == "beta":
            try:
                from betacal import BetaCalibration
                self.model = BetaCalibration(parameters="abm")
            except ImportError:
                self.method = "isotonic"
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, scores, labels):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        if self.method == "beta":
            self.model.fit(scores.reshape(-1, 1), labels)
        elif self.method == "isotonic":
            self.model.fit(scores, labels)
        return self

    def predict(self, scores):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if self.method == "identity":
            return scores
        return np.clip(np.asarray(self.model.predict(scores)).reshape(-1), 0.0, 1.0)


def confidence_scorer(labels, scores, lower, upper, gamma=0.8, beta=0.2):
    """ETA-DyNN's threshold objective: quality of the returned answers blended
    with the fraction of samples that receive an answer."""
    answers = np.full(len(scores), -1)
    answers[scores < lower] = 0
    answers[scores > upper] = 1
    mask = answers != -1
    if mask.sum() == 0:
        return 0.0
    acc = accuracy_score(labels[mask], answers[mask])
    f1 = f1_score(labels[mask], answers[mask], zero_division=0)
    return gamma * (beta * f1 + (1 - beta) * acc) + (1 - gamma) * mask.mean()


def search_thresholds(labels, scores, n_grid=50):
    """Grid search of (lower, upper) maximising ``confidence_scorer``."""
    lowers = np.linspace(0.0, 0.5, n_grid, endpoint=False)
    uppers = np.linspace(0.5, 1.0, n_grid + 1)[1:]
    best, best_pair = -1.0, (0.5, 0.5)
    for lo in lowers:
        for up in uppers:
            s = confidence_scorer(labels, scores, lo, up)
            if s > best:
                best, best_pair = s, (float(lo), float(up))
    return best_pair, best
