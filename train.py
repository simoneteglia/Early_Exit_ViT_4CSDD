"""Joint fine-tuning of the early-exit ViT (port of ETA-DyNN's ``joint_train``).

Example:
    python train.py --manifest manifests/splits.json --out-dir runs/deit_s_ee --epochs 10
    python train.py --synthetic 128 --epochs 1 --out-dir /tmp/smoke   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from data import DATASET_NAME, build_datasets, build_loaders, prepare
from ee_vit import DEFAULT_BACKBONE, EarlyExitViT, multi_exit_loss, save_checkpoint
from metrics import ScoreCollector, per_exit_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Joint training of the early-exit ViT")
    p.add_argument("--dataset", default=DATASET_NAME)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--manifest", default="manifests/splits.json", help="Split manifest (created if missing)")
    p.add_argument("--synthetic", type=int, default=0, help="Use N random images instead of the dataset")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="Keep at most N samples per label (balanced subsample; 0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="timm ViT, e.g. vit_small_patch16_dinov3.lvd1689m, vit_base_patch16_dinov3.lvd1689m, deit_small_patch16_224")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--exit-points", default=None, help="Comma-separated block indices, e.g. 3,7")
    p.add_argument("--disable-ee", action="store_true", help="Train the plain ViT baseline (no side exits)")
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--random-crop", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-heads", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    p.add_argument("--loss-weights", default=None, help="Comma-separated per-exit loss weights")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="runs/ee_vit")
    p.add_argument("--tensorboard", action="store_true")
    return p.parse_args()


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(model, loader, device, weights, optimizer=None, scaler=None, amp=False, desc=""):
    training = optimizer is not None
    model.train(training)
    collector = ScoreCollector()
    loss_sum, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y, css in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                out = model(x)
                outs = [out] if torch.is_tensor(out) else out
                loss = multi_exit_loss(outs, y, weights)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            collector.add(y, css, outs)
            loss_sum += float(loss) * len(y)
            n += len(y)
    labels, _, scores = collector.arrays()
    return loss_sum / max(n, 1), per_exit_metrics(labels, scores)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_ds, df, splits = prepare(args.dataset, args.manifest, args.synthetic, args.seed,
                                cache_dir=args.cache_dir, max_per_class=args.max_per_class)
    datasets = build_datasets(hf_ds, df, splits, args.image_size, args.random_crop)
    loaders = build_loaders(datasets, args.batch_size, args.num_workers, pin_memory=device.type == "cuda")
    print({k: len(v) for k, v in datasets.items()})

    exit_points = None if args.exit_points is None else [int(v) for v in args.exit_points.split(",")]
    model = EarlyExitViT(args.backbone, pretrained=not args.no_pretrained, exit_points=exit_points,
                         disable_ee=args.disable_ee, drop_path_rate=args.drop_path).to(device)
    print(f"backbone={args.backbone} depth={model.depth} exit_points={model.exit_points} "
          f"disable_ee={args.disable_ee} device={device}")

    weights = None if args.loss_weights is None else [float(w) for w in args.loss_weights.split(",")]
    if args.freeze_backbone:
        for prm in model.backbone.parameters():
            prm.requires_grad_(False)
    head_params = list(model.exits.parameters()) + list(model.head.parameters())
    groups = [{"params": head_params, "lr": args.lr_heads}]
    if not args.freeze_backbone:
        groups.append({"params": model.backbone.parameters(), "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = {"cosine": lambda: CosineAnnealingLR(optimizer, T_max=args.epochs),
                 "step": lambda: StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1),
                 "none": lambda: None}[args.scheduler]()
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(args), "model": model.config}, f, indent=2)

    history, best_loss = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = run_epoch(model, loaders["train"], device, weights, optimizer,
                                             scaler, args.amp, desc=f"train {epoch}")
        val_loss, val_metrics = run_epoch(model, loaders["val"], device, weights, desc=f"val {epoch}")
        if scheduler is not None:
            scheduler.step()

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                  "train": train_metrics, "val": val_metrics, "seconds": time.time() - t0}
        history.append(record)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}  " + "  ".join(
            f"exit{e}: acc={m['accuracy']:.3f} auc={m.get('auc', float('nan')):.3f}"
            for e, m in val_metrics.items()))
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            for e, m in val_metrics.items():
                for k in ("accuracy", "precision", "recall", "f1"):
                    writer.add_scalar(f"{k}/exit_{e}/val", m[k], epoch)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(model, out_dir / "last.pt", epoch=epoch, val_loss=val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, out_dir / "best.pt", epoch=epoch, val_loss=val_loss)

    if writer is not None:
        writer.close()
    print(f"done. best val loss {best_loss:.4f}; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
