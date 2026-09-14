"""Human-centered threshold policy: the analogue of ETA-DyNN's
environment-aware agent. Content sensitivity, user sensitivity and policy
shape the confidence band an early exit must clear; prediction confidence is
what gets compared against it.

    m = clip(m_0 + alpha*s_c + beta*s_u + gamma*p, 0, m_max)
    lower' = lower * (1 - m)
    upper' = upper + (1 - upper) * m

m = 0 reproduces confidence-only early exit; m -> 1 forces full depth.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

CSS_MIN, CSS_MAX = 1.0, 4.0
# Content-sensitivity tiers: one per level of the 1-4 CSS scale (fractional
# scores, e.g. averaged across annotators, are rounded to the nearest level).
CSS_TIERS = {1: "1-NONE", 2: "2-LOW", 3: "3-MODERATE", 4: "4-HIGH"}
MIST_MAX = 20  # MIST-20 total score (number of correctly judged headlines)


def normalize_css(score) -> np.ndarray:
    """Map a 1-4 content-sensitivity score to [0, 1]."""
    score = np.asarray(score, dtype=np.float32)
    return np.clip((score - CSS_MIN) / (CSS_MAX - CSS_MIN), 0.0, 1.0)


def css_tier(score) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    level = np.clip(np.floor(score + 0.5), CSS_MIN, CSS_MAX).astype(int)
    return np.array([CSS_TIERS[int(l)] for l in level.reshape(-1)], dtype=object).reshape(score.shape)


# ---------------------------------------------------------------- MIST -----

def load_mist(path, score_column: str | None = None) -> np.ndarray:
    """Load MIST total scores from the OSF data file (CSV/TSV/XLSX/SAV).

    MIST (Maertens et al., 2023): higher score = better veracity discernment
    = *less* susceptible to misinformation. The column is auto-detected when
    not given (first column whose name contains 'mist' and 'total'/'score',
    else any column containing 'mist').
    """
    path = str(path)
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    elif path.endswith(".sav"):
        df = pd.read_spss(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    if score_column is None:
        cols = [c for c in df.columns if "mist" in str(c).lower()]
        pref = [c for c in cols if any(k in str(c).lower() for k in ("total", "score", "sum"))]
        cands = pref or cols
        if not cands:
            raise ValueError(f"No MIST column found in {path}; columns: {list(df.columns)}")
        score_column = cands[0]
    scores = pd.to_numeric(df[score_column], errors="coerce").dropna().to_numpy(dtype=np.float32)
    return scores


def user_sensitivity(mist_score, reference: np.ndarray | None = None) -> np.ndarray:
    """Susceptibility s_u in [0, 1] from a MIST score: 1 - percentile rank of
    the score within the reference population (falls back to a linear map on
    the 0-20 scale with a warning when no reference data is available)."""
    mist_score = np.asarray(mist_score, dtype=np.float32)
    if reference is None or len(reference) == 0:
        warnings.warn("No MIST reference population given; using linear 0-20 scale.")
        return np.clip(1.0 - mist_score / MIST_MAX, 0.0, 1.0)
    ref = np.sort(np.asarray(reference, dtype=np.float32))
    pct = np.searchsorted(ref, mist_score, side="right") / len(ref)
    return np.clip(1.0 - pct, 0.0, 1.0)


# -------------------------------------------------------------- policy -----

@dataclass
class ThresholdPolicy:
    m0: float = 0.0
    alpha: float = 0.5   # weight of content sensitivity
    beta: float = 0.3    # weight of user susceptibility
    gamma: float = 0.2   # weight of policy strictness
    m_max: float = 1.0

    def margin(self, s_c, s_u=0.0, p=0.0) -> np.ndarray:
        s_c = np.asarray(s_c, dtype=np.float32)
        m = self.m0 + self.alpha * s_c + self.beta * np.float32(s_u) + self.gamma * np.float32(p)
        return np.clip(m, 0.0, self.m_max)

    @staticmethod
    def widen(lower, upper, m):
        """Effective (lower, upper) after widening the abstention band by m.
        Broadcasts: lower/upper of shape (E,) with m of shape (N,) -> (E, N)."""
        lower = np.asarray(lower, dtype=np.float32)
        upper = np.asarray(upper, dtype=np.float32)
        m = np.asarray(m, dtype=np.float32)
        if lower.ndim == 1 and m.ndim == 1:
            lower, upper, m = lower[:, None], upper[:, None], m[None, :]
        return lower * (1.0 - m), upper + (1.0 - upper) * m

    def thresholds(self, base_lower, base_upper, s_c, s_u=0.0, p=0.0):
        """Per-exit, per-sample effective thresholds, shape (E, N)."""
        return self.widen(base_lower, base_upper, self.margin(s_c, s_u, p))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
