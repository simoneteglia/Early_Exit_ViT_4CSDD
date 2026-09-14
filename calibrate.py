"""Per-exit calibration, base threshold search and ``scores.pkl`` export
(port of ETA-DyNN's ``fit_calibrator`` / ``fit_thresholds`` / score extraction).

Outputs (in --out-dir, default next to the checkpoint):
    raw_scores.npz     uncalibrated P(fake) per split/exit, labels, css, tiers, ids
    calibrators.pkl    list of Calibrator (side exits + final), fitted on --calibration-split
    thresholds.json    base (lower, upper) per side exit found on --calibration-split
    scores.pkl         {split: {"scores": [{"exit_idx", "scores": (N,2)}...],
                                "labels", "css_scores", "tiers", "sample_ids"}}
                       calibrated; the format the emulator's ThresholdsWindow reads
    calibration_report.json  per-exit metrics / ECE before and after calibration
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from calibration import Calibrator, search_thresholds
from data import DATASET_NAME, build_datasets, build_loaders, prepare
from ee_vit import load_model
from metrics import ScoreCollector, binary_metrics, expected_calibration_error


@torch.no_grad()
def collect_scores(model, loader, device, desc=""):
    collector = ScoreCollector()
    for x, y, css in tqdm(loader, desc=desc, leave=False):
        collector.add(y, css, model(x.to(device)))
    return collector.arrays()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dataset", default=DATASET_NAME)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--manifest", default="manifests/splits.json")
    p.add_argument("--synthetic", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--calibration-split", default="val")
    p.add_argument("--max-train-samples", type=int, default=0,
                   help="Subsample the train split used as reference distribution (0 = all)")
    p.add_argument("--method", choices=["beta", "isotonic", "identity"], default="beta")
    p.add_argument("--grid", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    out_dir = Path(args.out_dir or Path(args.checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    hf_ds, df, splits = prepare(args.dataset, args.manifest, args.synthetic, args.seed, cache_dir=args.cache_dir)
    if args.max_train_samples and len(splits["train"]) > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        splits["train"] = np.sort(rng.choice(splits["train"], args.max_train_samples, replace=False))
    datasets = build_datasets(hf_ds, df, splits, args.image_size)
    datasets["train"].transform = datasets["val"].transform  # no augmentation for reference scores
    loaders = build_loaders(datasets, args.batch_size, args.num_workers, pin_memory=device.type == "cuda")

    raw = {}
    for split, loader in loaders.items():
        labels, css, scores = collect_scores(model, loader, device, desc=f"scores {split}")
        raw[split] = {"labels": labels, "css_scores": css, "scores": scores,
                      "tiers": datasets[split].tiers, "sample_ids": datasets[split].sample_ids}
    np.savez(out_dir / "raw_scores.npz",
             **{f"{s}_{k}": v for s, d in raw.items() for k, v in d.items()})

    cal = raw[args.calibration_split]
    calibrators = [Calibrator(args.method).fit(cal["scores"][e], cal["labels"]) for e in range(model.n_exits)]
    with open(out_dir / "calibrators.pkl", "wb") as f:
        pickle.dump(calibrators, f)

    calibrated = {split: np.stack([calibrators[e].predict(d["scores"][e]) for e in range(model.n_exits)])
                  for split, d in raw.items()}

    thresholds = []
    for e in range(model.n_exits - 1):
        (lo, up), score = search_thresholds(cal["labels"], calibrated[args.calibration_split][e], args.grid)
        thresholds.append({"exit_idx": e, "lower": lo, "upper": up, "objective": score})
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump({"exit_points": model.config["exit_points"], "side_exits": thresholds,
                   "final": {"lower": 0.5, "upper": 0.5}, "calibration_split": args.calibration_split,
                   "method": calibrators[0].method}, f, indent=2)

    scores_pkl = {}
    for split, d in raw.items():
        scores_pkl[split] = {
            "scores": [{"exit_idx": e, "scores": np.stack([1 - calibrated[split][e], calibrated[split][e]], axis=1)}
                       for e in range(model.n_exits)],
            "labels": d["labels"], "css_scores": d["css_scores"], "tiers": d["tiers"],
            "sample_ids": d["sample_ids"],
        }
    with open(out_dir / "scores.pkl", "wb") as f:
        pickle.dump(scores_pkl, f)

    report = {}
    for split, d in raw.items():
        report[split] = {}
        for e in range(model.n_exits):
            r, c = d["scores"][e], calibrated[split][e]
            report[split][f"exit_{e}"] = {
                "raw": {**binary_metrics(d["labels"], (r > 0.5).astype(int), r),
                        "ece": expected_calibration_error(d["labels"], r)},
                "calibrated": {**binary_metrics(d["labels"], (c > 0.5).astype(int), c),
                               "ece": expected_calibration_error(d["labels"], c)},
            }
    with open(out_dir / "calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"method={calibrators[0].method}  thresholds={[(t['lower'], t['upper']) for t in thresholds]}")
    for split in raw:
        print(split, "  ".join(f"exit{e}: acc={report[split][f'exit_{e}']['calibrated']['accuracy']:.3f} "
                               f"ece {report[split][f'exit_{e}']['raw']['ece']:.3f}->"
                               f"{report[split][f'exit_{e}']['calibrated']['ece']:.3f}"
                               for e in range(model.n_exits)))
    print(f"wrote {out_dir}/{{raw_scores.npz, calibrators.pkl, thresholds.json, scores.pkl, calibration_report.json}}")


if __name__ == "__main__":
    main()
