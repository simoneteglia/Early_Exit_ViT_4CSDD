"""Headless propagation experiment: run the four detector scenarios over a
stratified subset of test items and report reach / cost per scenario.

    python simulate.py --db experiments.db --experiment exp_5k --dataset css_5k \
        --thresholds runs/ee_vit_5k/thresholds.json --items 60 --out results/sim_5k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments_db import PERF_COLUMNS
from human_factors import ThresholdPolicy
from propagation import (SCENARIOS, SimParams, build_network, load_items, sample_user_sensitivity,
                         select_items, simulate, summarize)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="experiments.db")
    p.add_argument("--experiment", required=True)
    p.add_argument("--dataset", default="css_test")
    p.add_argument("--thresholds", required=True, help="thresholds.json from calibrate.py")
    p.add_argument("--policy", default=None, help="JSON with m0/alpha/beta/gamma/m_max")
    p.add_argument("--policy-strictness", type=float, default=0.0, help="p in [0, 1]")
    p.add_argument("--items", type=int, default=60, help="Number of test items (0 = all)")
    p.add_argument("--source", default="ALL", help="Restrict items to one source dataset (sample-id prefix), e.g. rrdataset")
    p.add_argument("--nodes", type=int, default=1000)
    p.add_argument("--ba-m", type=int, default=3)
    p.add_argument("--p-share", type=float, default=0.15)
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--intervention", type=float, default=0.5, help="w: 0 = no effect, 1 = block")
    p.add_argument("--seeds-per-item", type=int, default=1)
    p.add_argument("--seed-strategy", choices=["random", "hub"], default="random")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--mist", default=None, help="MIST data file (OSF) for the user population")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Output prefix for results JSON / trends PNG")
    return p.parse_args()


def main():
    args = parse_args()
    params = SimParams(args.nodes, args.ba_m, args.p_share, args.kappa, args.intervention,
                       args.seeds_per_item, args.seed_strategy, args.max_steps, args.seed, args.mist,
                       n_items=args.items, source=args.source)
    policy = ThresholdPolicy.from_dict(json.load(open(args.policy))) if args.policy else ThresholdPolicy()
    thr = json.load(open(args.thresholds))
    base_lower = [t["lower"] for t in thr["side_exits"]]
    base_upper = [t["upper"] for t in thr["side_exits"]]

    rng = np.random.default_rng(args.seed)
    items = select_items(load_items(args.db, args.experiment, args.dataset), args.items, rng, source=args.source)
    if not items:
        raise SystemExit(f"no items for source '{args.source}'")
    g, g_dir = build_network(params)
    s_u = sample_user_sensitivity(g.number_of_nodes(), rng, args.mist)
    print(f"{len(items)} items ({sum(i.label for i in items)} fake, source={args.source}), network n={g.number_of_nodes()} "
          f"edges={g.number_of_edges()}, s_u mean={s_u.mean():.2f}, p={args.policy_strictness}, w={args.intervention}")

    rows = simulate(items, g_dir, s_u, policy, base_lower, base_upper, args.policy_strictness, params,
                    progress=lambda i, n: print(f"\r  item {i}/{n}", end="", flush=True))
    print()
    summary = summarize(rows)

    print(f"{'scenario':<28}{'fake reach':>11}{'(-%)':>7}{'real reach':>11}{'(-%)':>7}{'runs':>8}"
          f"{'time s':>9}{'saving':>8}{'energy Wh':>11}{'saving':>8}  exit distribution")
    for sc in SCENARIOS:
        s = summary[sc]
        print(f"{s['label']:<28}{s['fake_reach']:>11.3f}{s['fake_reach_reduction']*100:>7.1f}"
              f"{s['real_reach']:>11.3f}{s['real_reach_loss']*100:>7.1f}{s['invocations']:>8d}"
              f"{s['cost']['duration']:>9.3f}{s['savings']['duration']:>7.1f}%"
              f"{s['cost']['tot_energy']*1000:>11.4f}{s['savings']['tot_energy']:>7.1f}%  "
              + " ".join(f"{d*100:.0f}%" for d in s["exit_distribution"]))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{out}_results.json", "w") as f:
            json.dump({"params": params.to_dict(), "policy": policy.to_dict(), "p": args.policy_strictness,
                       "thresholds": {"lower": base_lower, "upper": base_upper},
                       "summary": summary, "rows": rows}, f, indent=1, default=float)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, label, title in ((axes[0], 1, "Fake items"), (axes[1], 0, "Real items")):
            for sc in SCENARIOS:
                from propagation import mean_trend
                tr = mean_trend([r for r in rows if r["scenario"] == sc], label)
                ax.plot(range(len(tr)), [t * 100 for t in tr], label=summary[sc]["label"])
            ax.set_title(f"{title}: mean cumulative reach")
            ax.set_xlabel("step")
            ax.set_ylabel("% of users reached")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out}_trends.png", dpi=100)
        print(f"wrote {out}_results.json and {out}_trends.png")


if __name__ == "__main__":
    main()
