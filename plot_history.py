#!/usr/bin/env python3
"""Plot training history from history.json as a multi-panel dashboard.

Usage:
    python plot_history.py [--history runs/ee_vit_test1k/history.json] [--out dashboard.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(path):
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="runs/ee_vit_test1k/history.json")
    p.add_argument("--out", default=None, help="Output PNG path (default: next to history.json)")
    args = p.parse_args()

    history = load_history(args.history)
    out_path = args.out or str(Path(args.history).parent / "dashboard.png")

    epochs = [h["epoch"] for h in history]
    n_exits = len(history[0]["val"])
    exit_labels = {int(k): f"Exit {k}" if int(k) < n_exits - 1 else "Final head"
                   for k in history[0]["val"]}

    # Colours for each exit
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, n_exits - 1)) for i in range(n_exits)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Early-Exit ViT — Training Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # ── 1. Loss curves ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, [h["train_loss"] for h in history], "o-", color="#2196F3", label="Train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], "s--", color="#F44336", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / Val Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 2-4. Per-exit metrics (accuracy, AUC, F1) ────────────────
    metric_panels = [
        ("accuracy", "Accuracy", axes[0, 1]),
        ("auc",      "AUC",      axes[0, 2]),
        ("f1",       "F1 Score", axes[1, 0]),
    ]
    for metric_key, title, ax in metric_panels:
        for eidx, (exit_key, label) in enumerate(exit_labels.items()):
            train_vals = [h["train"].get(str(exit_key), {}).get(metric_key, float("nan")) for h in history]
            val_vals   = [h["val"].get(str(exit_key), {}).get(metric_key, float("nan")) for h in history]
            ax.plot(epochs, val_vals, "s-", color=colors[eidx], label=f"{label} (val)")
            ax.plot(epochs, train_vals, "o--", color=colors[eidx], alpha=0.4, label=f"{label} (train)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Per-Exit {title}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        if metric_key in ("accuracy", "f1"):
            ax.set_ylim(max(0, ax.get_ylim()[0] - 0.02), 1.02)
        elif metric_key == "auc":
            ax.set_ylim(max(0.5, ax.get_ylim()[0] - 0.02), 1.01)

    # ── 5. Precision / Recall for final epoch ────────────────────
    ax = axes[1, 1]
    last = history[-1]
    exits = sorted(last["val"].keys(), key=int)
    x_pos = np.arange(len(exits))
    width = 0.35
    prec = [last["val"][e]["precision"] for e in exits]
    rec  = [last["val"][e]["recall"] for e in exits]
    ax.bar(x_pos - width/2, prec, width, label="Precision", color="#4CAF50", alpha=0.85)
    ax.bar(x_pos + width/2, rec,  width, label="Recall",    color="#FF9800", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([exit_labels[int(e)] for e in exits])
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall (Epoch {last['epoch']})")
    ax.legend()
    ax.set_ylim(0.7, 1.02)
    ax.grid(axis="y", alpha=0.3)

    # ── 6. Summary table ─────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    last_val = last["val"]
    headers = ["Exit", "Acc", "AUC", "Prec", "Rec", "F1"]
    rows = []
    for e in exits:
        m = last_val[e]
        rows.append([exit_labels[int(e)],
                      f"{m['accuracy']:.3f}",
                      f"{m.get('auc', float('nan')):.3f}",
                      f"{m['precision']:.3f}",
                      f"{m['recall']:.3f}",
                      f"{m['f1']:.3f}"])
    rows.append(["", "", "", "", "", ""])
    rows.append(["Best val loss", f"{min(h['val_loss'] for h in history):.4f}", "", "", "", ""])
    rows.append(["Final val loss", f"{last['val_loss']:.4f}", "", "", "", ""])

    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    # Style header row
    for j, key in enumerate(headers):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Final Metrics (Val, Epoch {last['epoch']})", pad=20)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved to {out_path}")


if __name__ == "__main__":
    main()
