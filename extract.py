"""Populate ``experiments.db`` with per-sample calibrated scores for every exit
and the cumulative cost of reaching each exit (port of ETA-DyNN's
``--experiment`` extraction).

Rows written per test sample:
    ee_vit / exit_idx=e   calibrated score of side exit e, cost of computing up to exit e
    full_vit / NULL       calibrated score of the final head, cost of the full forward

Costs are measured once per exit on --cost-batches batches (duration always;
energy via codecarbon when available) and stored per sample as the mean
per-sample value, so the emulator's cost table sums them exactly like the
reference implementation.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data import DATASET_NAME, build_datasets, build_loaders, prepare
from ee_vit import load_model
from experiments_db import EXPERIMENTS_DB
from metrics import ScoreCollector

logging.basicConfig(level=logging.INFO)


class EnergyMeter:
    """codecarbon wrapper returning kWh (tot, cpu, gpu, ram); zeros when unavailable."""

    def __init__(self, enabled=True):
        self.tracker = None
        if enabled:
            try:
                from codecarbon import EmissionsTracker
                self.tracker = EmissionsTracker(save_to_file=False, log_level="error",
                                                allow_multiple_runs=True)
            except Exception as e:  # noqa: BLE001
                logging.warning(f"codecarbon unavailable ({e}); energy set to 0")

    def __enter__(self):
        if self.tracker is not None:
            self.tracker.start()
        return self

    def __exit__(self, *exc):
        if self.tracker is not None:
            self.tracker.stop()

    def energy(self):
        if self.tracker is None:
            return (0.0, 0.0, 0.0, 0.0)
        t = self.tracker
        return (float(t._total_energy.kWh), float(t._total_cpu_energy.kWh),
                float(t._total_gpu_energy.kWh), float(t._total_ram_energy.kWh))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_costs(model, dataset, loader, device, n_batches, energy=True):
    """Per-sample (duration s, tot, cpu, gpu, ram kWh) for preprocessing and for
    reaching each exit (cumulative)."""
    n_side = len(model.exit_points)
    stages = list(range(n_side)) + [None]  # None = full forward
    batches = []
    for i, (x, _, _) in enumerate(loader):
        if i >= n_batches:
            break
        batches.append(x.to(device))
    n_samples = sum(len(b) for b in batches)

    # preprocessing = image decode + transform
    idx = np.random.default_rng(0).choice(len(dataset), min(len(dataset), 64), replace=False)
    with EnergyMeter(energy) as em:
        t0 = time.perf_counter()
        for i in idx:
            dataset[int(i)]
        pre_dur = (time.perf_counter() - t0) / len(idx)
    pre = (pre_dur, *(v / len(idx) for v in em.energy()))

    model(batches[0][:2])  # warm-up
    costs = {}
    for stage in stages:
        with EnergyMeter(energy) as em:
            sync(device)
            t0 = time.perf_counter()
            for x in batches:
                model(x, force_exit=stage)
            sync(device)
            dur = (time.perf_counter() - t0) / n_samples
        costs[stage] = (dur, *(v / n_samples for v in em.energy()))
    return pre, costs


@torch.no_grad()
def collect(model, loader, device, calibrators):
    collector = ScoreCollector()
    for x, y, css in tqdm(loader, desc="scores", leave=False):
        collector.add(y, css, model(x.to(device)))
    labels, css, raw = collector.arrays()
    cal = np.stack([calibrators[e].predict(raw[e]) for e in range(raw.shape[0])])
    return labels, css, raw, cal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calibration-dir", default=None, help="Dir with calibrators.pkl (default: checkpoint dir)")
    p.add_argument("--db", default="experiments.db")
    p.add_argument("--experiment", required=True, help="Experiment codename")
    p.add_argument("--dataset-codename", default="css_test")
    p.add_argument("--split", default="test")
    p.add_argument("--dataset", default=DATASET_NAME)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--manifest", default="manifests/splits.json")
    p.add_argument("--synthetic", type=int, default=0)
    p.add_argument("--max-per-class", type=int, default=0,
                   help="Keep at most N samples per label (balanced subsample; 0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--cost-batches", type=int, default=10)
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--reset-db", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    cal_dir = Path(args.calibration_dir or Path(args.checkpoint).parent)
    with open(cal_dir / "calibrators.pkl", "rb") as f:
        calibrators = pickle.load(f)

    model = load_model(args.checkpoint, device)
    hf_ds, df, splits = prepare(args.dataset, args.manifest, args.synthetic, args.seed,
                                cache_dir=args.cache_dir, max_per_class=args.max_per_class)
    datasets = build_datasets(hf_ds, df, splits, args.image_size)
    dataset = datasets[args.split]
    loader = build_loaders({args.split: dataset}, args.batch_size, args.num_workers,
                           pin_memory=device.type == "cuda")[args.split]

    print("measuring per-exit costs...")
    pre_cost, stage_costs = measure_costs(model, dataset, loader, device, args.cost_batches,
                                          energy=not args.no_energy)
    for stage, c in stage_costs.items():
        print(f"  {'final' if stage is None else f'exit {stage}'}: {c[0]*1000:.2f} ms/sample, {c[1]*1000:.4f} Wh/sample")

    labels, css, raw, cal = collect(model, loader, device, calibrators)
    assert np.array_equal(labels, dataset.labels)

    db = EXPERIMENTS_DB(args.db)
    if args.reset_db:
        db.reset_database()
    new_dataset = db.add_dataset(args.dataset_codename)
    if new_dataset:
        db.add_samples(args.dataset_codename, zip(dataset.sample_ids, dataset.labels, dataset.css_scores,
                                                  dataset.tiers, dataset.scenarios))
    sample_ids = db.get_sample_ids(args.dataset_codename)
    assert len(sample_ids) == len(dataset), "DB dataset size differs from the split; use --reset-db or a new codename"
    db.add_experiment(args.experiment)
    db.add_dataset_to_experiment(args.experiment, args.dataset_codename)

    n_side = len(model.exit_points)
    for i, sid in enumerate(tqdm(sample_ids, desc="db rows")):
        for e in range(n_side):
            db.add_modelrun_with_performance(sid, args.experiment, "ee_vit", e, [cal[e, i]],
                                             pre_cost, stage_costs[e], commit=False)
        db.add_modelrun_with_performance(sid, args.experiment, "full_vit", None, [cal[-1, i]],
                                         pre_cost, stage_costs[None], commit=False)
    db.commit()
    db.close()

    with open(cal_dir / f"costs_{args.experiment}.json", "w") as f:
        json.dump({"preprocessing": pre_cost,
                   **{("final" if k is None else f"exit_{k}"): v for k, v in stage_costs.items()}}, f, indent=2)
    print(f"wrote {len(sample_ids)} samples x {n_side + 1} model runs to {args.db} "
          f"(experiment={args.experiment}, dataset={args.dataset_codename})")


if __name__ == "__main__":
    main()
