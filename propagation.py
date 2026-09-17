"""Mis/disinformation propagation on a social network with the early-exit
deepfake detector in the loop (the analogue of ETA-DyNN's activity emulator).

Model
-----
* One content item (a test image with its recorded per-exit scores) = one
  Independent-Cascades run (NDlib): a user who shares the item gets one chance
  to pass it to each neighbour; the neighbour shares with probability
      P(v shares) = p_share * g(s_u(v), label) * (1 - w * flagged(item, v))
  where g raises the sharing of *fake* items for susceptible users and
  ``flagged`` is the detector's verdict for user v.
* The detector runs once per (item, exposed user) with that user's
  susceptibility s_u, the item's content sensitivity s_c and the policy p, so
  the answering exit — and therefore the inference cost — differs per user.
  Verdicts use the calibrated scores stored in ``experiments.db``.
* Scenarios: ``none`` (no detector), ``full`` (final head for everyone),
  ``ee_conf`` (early exit, confidence only, m = 0), ``ee_human`` (early exit
  with the human-centered margin). All scenarios share seeds and random
  streams per item, so differences are due to the detector alone.
* Assumptions: items independent, static synthetic network, a user shares an
  item at most once, susceptibility independent of network position, the
  intervention (warning w < 1, block w = 1) applies equally to false positives.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import networkx as nx
import numpy as np

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

from experiments_db import EXPERIMENTS_DB, PERF_COLUMNS
from human_factors import ThresholdPolicy, normalize_css, user_sensitivity

SCENARIOS = ("none", "full", "ee_conf", "ee_human")
SCENARIO_LABELS = {"none": "No detector", "full": "Full ViT", "ee_conf": "Early exit (confidence)",
                   "ee_human": "Early exit (human-centered)"}
STATUS_S, STATUS_I, STATUS_R = 0, 1, 2

# MIST-20 stand-in population when the OSF data file is not provided
# (assumption; replace with human_factors.load_mist on the real file).
MIST_MEAN, MIST_STD = 13.0, 3.5


@dataclass
class Item:
    sample_id: str
    label: int
    css: float
    tier: str
    source: str
    exit_scores: np.ndarray   # (n_side,) calibrated P(fake) of the side exits
    final_score: float
    exit_costs: np.ndarray    # (n_side, len(PERF_COLUMNS)) cumulative inference cost per sample
    final_cost: np.ndarray    # (len(PERF_COLUMNS),)


@dataclass
class SimParams:
    n_nodes: int = 1000
    ba_m: int = 3
    p_share: float = 0.15          # base sharing probability per exposure
    kappa: float = 0.5             # susceptibility effect on sharing fakes: g = 1 + kappa*(2*s_u - 1)
    intervention: float = 0.5      # w: share-probability reduction when flagged (1 = block)
    seeds_per_item: int = 1
    seed_strategy: str = "random"  # or "hub" (highest-degree nodes)
    max_steps: int = 50
    seed: int = 0
    mist_path: str | None = None
    n_items: int = 60
    source: str = "ALL"            # restrict items to one source dataset (sample-id prefix)

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------- data ---

def load_items(db_path, experiment, dataset) -> list[Item]:
    db = EXPERIMENTS_DB(db_path)
    exits = db.get_exit_indexes(experiment, dataset)
    samples = db.get_dataset_samples(dataset)  # (label, file_path, css, tier, scenario)
    side = np.array([[np.mean(s) for s in db.get_scores(experiment, dataset, "ee_vit", e)] for e in exits])
    final = np.array([np.mean(s) for s in db.get_scores(experiment, dataset, "full_vit")])
    side_cost = np.array([db.get_performance(experiment, dataset, "ee_vit", e)["inference"] for e in exits])
    final_cost = np.array(db.get_performance(experiment, dataset, "full_vit")["inference"])
    db.close()
    items = []
    for i, (label, path, css, tier, _) in enumerate(samples):
        sid = str(path)
        items.append(Item(sid, int(label), 1.0 if css is None else float(css), tier or "ALL",
                          sid.split("_", 1)[0] if "_" in sid else "ALL",
                          side[:, i], float(final[i]), side_cost[:, i, :], final_cost[i]))
    return items


def select_items(items, n, rng, stratify=True, source="ALL"):
    """Balanced subsample: equal numbers of real and fake, spread over tiers,
    optionally restricted to one source dataset."""
    if source and source != "ALL":
        items = [it for it in items if it.source == source]
    if n <= 0 or n >= len(items):
        return list(items)
    if not stratify:
        return [items[i] for i in rng.choice(len(items), n, replace=False)]
    groups = {}
    for it in items:
        groups.setdefault((it.label, it.tier), []).append(it)
    chosen = []
    labels = sorted({k[0] for k in groups})
    per_label = {lb: n // len(labels) for lb in labels}
    for label, quota in per_label.items():
        keys = [k for k in groups if k[0] == label]
        pools = {k: list(rng.permutation(groups[k])) for k in keys}
        while quota > 0 and any(pools.values()):
            for k in keys:
                if quota > 0 and pools[k]:
                    chosen.append(pools[k].pop())
                    quota -= 1
    return chosen


# -------------------------------------------------------------- network ---

def build_network(params: SimParams):
    g = nx.barabasi_albert_graph(params.n_nodes, params.ba_m, seed=params.seed)
    return g, g.to_directed()


def sample_user_sensitivity(n, rng, mist_path=None):
    """s_u per node from the MIST population (OSF file) or a stand-in normal."""
    if mist_path:
        from human_factors import load_mist
        ref = load_mist(mist_path)
        mist = rng.choice(ref, n, replace=True)
    else:
        ref = None
        mist = np.clip(rng.normal(MIST_MEAN, MIST_STD, n), 0, 20)
    return user_sensitivity(mist, reference=ref if ref is not None else mist)


def choose_seeds(g, k, strategy, rng):
    if strategy == "hub":
        return [n for n, _ in sorted(g.degree, key=lambda d: -d[1])[:k]]
    return [int(v) for v in rng.choice(g.number_of_nodes(), k, replace=False)]


# ------------------------------------------------------------- detector ---

def detector_decision(item: Item, s_u, scenario, policy: ThresholdPolicy, base_lower, base_upper, p):
    """Per-user verdict for one item. Returns (answers (N,), exit_idx (N,));
    exit_idx = -1 for scenario 'none', n_side for the final head."""
    n = len(s_u)
    n_side = len(item.exit_scores)
    if scenario == "none":
        return np.full(n, -1), np.full(n, -1)
    if scenario == "full" or n_side == 0:
        return np.full(n, int(item.final_score > 0.5)), np.full(n, n_side)
    pol = ThresholdPolicy(0, 0, 0, 0, 0) if scenario == "ee_conf" else policy
    lower, upper = pol.thresholds(base_lower, base_upper, np.full(n, normalize_css(item.css)), s_u, p)
    answers = np.full(n, -1)
    exit_idx = np.full(n, n_side)
    for e in range(n_side):
        s = item.exit_scores[e]
        undecided = answers == -1
        real = undecided & (s < lower[e])
        fake = undecided & (s > upper[e])
        answers[real], answers[fake] = 0, 1
        exit_idx[real | fake] = e
    answers[answers == -1] = int(item.final_score > 0.5)
    return answers, exit_idx


# -------------------------------------------------------------- cascade ---

def share_probabilities(item: Item, flagged, s_u, params: SimParams):
    """Probability that each node shares the item when exposed."""
    g = 1.0 + params.kappa * (2.0 * s_u - 1.0) if item.label == 1 else np.ones_like(s_u)
    prob = np.clip(params.p_share * g, 0.0, 1.0)
    return prob * np.where(flagged, 1.0 - params.intervention, 1.0)


def run_cascade(g_dir, seeds, node_prob, max_steps, np_seed):
    """One NDlib Independent-Cascades run; the edge threshold into v is v's share probability."""
    model = ep.IndependentCascadesModel(g_dir)
    cfg = mc.Configuration()
    cfg.add_model_initial_configuration("Infected", list(seeds))
    for u, v in g_dir.edges():
        cfg.add_edge_configuration("threshold", (u, v), float(node_prob[v]))
    model.set_initial_status(cfg)
    np.random.seed(np_seed)  # NDlib draws from numpy's global RNG
    counts, history = [], []
    for _ in range(max_steps + 1):
        it = model.iteration()
        counts.append(it["node_count"])
        history.append({int(k): int(v) for k, v in it["status"].items()})  # per-step status changes
        if it["node_count"][STATUS_I] == 0 and it["iteration"] > 0:
            break
    status = np.array([model.status[v] for v in range(g_dir.number_of_nodes())])
    return status, counts, history


def exposed_nodes(g_dir, status):
    """Users who received the item: seeds/sharers plus every neighbour of a sharer."""
    sharers = np.flatnonzero(status != STATUS_S)
    exposed = set(sharers.tolist())
    for u in sharers:
        exposed.update(g_dir.successors(int(u)))
    return np.fromiter(exposed, dtype=int)


# ------------------------------------------------------------ simulate ---

def simulate(items, g_dir, s_u, policy, base_lower, base_upper, p, params: SimParams,
             scenarios=SCENARIOS, progress=None):
    """Run every item under every scenario (paired seeds). Returns a list of
    per-(item, scenario) result dicts."""
    rng = np.random.default_rng(params.seed)
    n = g_dir.number_of_nodes()
    n_side = len(items[0].exit_scores) if items else 0
    rows = []
    for i, item in enumerate(items):
        seeds = choose_seeds(g_dir, params.seeds_per_item, params.seed_strategy, rng)
        np_seed = int(rng.integers(0, 2**31 - 1))
        for sc in scenarios:
            answers, exit_idx = detector_decision(item, s_u, sc, policy, base_lower, base_upper, p)
            flagged = answers == 1
            prob = share_probabilities(item, flagged, s_u, params)
            status, counts, history = run_cascade(g_dir, seeds, prob, params.max_steps, np_seed)
            exposed = exposed_nodes(g_dir, status)
            reached = int((status != STATUS_S).sum())
            # inference cost: one detector run per exposed user, charged at the exit that answered
            cost = np.zeros(len(PERF_COLUMNS))
            exits_hist = np.zeros(n_side + 1, dtype=int)
            if sc != "none":
                ex = exit_idx[exposed]
                for e in range(n_side):
                    k = int((ex == e).sum())
                    exits_hist[e] = k
                    cost += k * item.exit_costs[e]
                k = int((ex == n_side).sum())
                exits_hist[n_side] = k
                cost += k * item.final_cost
            rows.append({
                "item": item.sample_id, "label": item.label, "tier": item.tier, "source": item.source,
                "scenario": sc, "seeds": seeds,
                "reach": reached / n, "n_shared": reached,
                "n_exposed": int(len(exposed)), "invocations": int(exits_hist.sum()),
                "exits_hist": exits_hist.tolist(),
                "flagged_exposed": int(flagged[exposed].sum()) if sc != "none" else 0,
                "steps": len(counts) - 1,
                "trend": [(c[STATUS_I] + c[STATUS_R]) / n for c in counts],
                "history": history, "flagged": np.flatnonzero(flagged).tolist() if sc != "none" else [],
                "exit_idx": exit_idx.tolist(),
                **{f"cost_{m}": float(cost[j]) for j, m in enumerate(PERF_COLUMNS)},
            })
        if progress is not None:
            progress(i + 1, len(items))
    return rows


def summarize(rows, scenarios=SCENARIOS):
    """Per-scenario aggregates: reach of fake / real items, reduction vs. no
    detector, detector cost and savings vs. the full-ViT scenario."""
    by = {sc: [r for r in rows if r["scenario"] == sc] for sc in scenarios}
    out = {}
    none_fake = np.mean([r["reach"] for r in by.get("none", []) if r["label"] == 1] or [np.nan])
    none_real = np.mean([r["reach"] for r in by.get("none", []) if r["label"] == 0] or [np.nan])
    full_cost = {m: sum(r[f"cost_{m}"] for r in by.get("full", [])) for m in PERF_COLUMNS}
    for sc, rs in by.items():
        fake = [r["reach"] for r in rs if r["label"] == 1]
        real = [r["reach"] for r in rs if r["label"] == 0]
        cost = {m: sum(r[f"cost_{m}"] for r in rs) for m in PERF_COLUMNS}
        hist = np.sum([r["exits_hist"] for r in rs], axis=0) if rs else np.zeros(1)
        out[sc] = {
            "label": SCENARIO_LABELS.get(sc, sc),
            "fake_reach": float(np.mean(fake)) if fake else np.nan,
            "real_reach": float(np.mean(real)) if real else np.nan,
            "fake_reach_reduction": float(1 - np.mean(fake) / none_fake) if fake and none_fake > 0 else np.nan,
            "real_reach_loss": float(1 - np.mean(real) / none_real) if real and none_real > 0 else np.nan,
            "invocations": int(sum(r["invocations"] for r in rs)),
            "exit_distribution": (hist / max(hist.sum(), 1)).tolist(),
            "cost": cost,
            "savings": {m: (1 - cost[m] / full_cost[m]) * 100 if full_cost[m] > 0 else 0.0 for m in PERF_COLUMNS},
            "mean_trend": mean_trend(rs),
        }
    return out


def mean_trend(rows, label=None):
    """Average cumulative-reach curve over items (padded with the final value)."""
    rs = [r for r in rows if label is None or r["label"] == label]
    if not rs:
        return []
    length = max(len(r["trend"]) for r in rs)
    mat = np.array([r["trend"] + [r["trend"][-1]] * (length - len(r["trend"])) for r in rs])
    return mat.mean(axis=0).tolist()
