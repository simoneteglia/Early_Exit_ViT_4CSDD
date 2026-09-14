"""Global results panel: confusion matrix and metrics of the whole early-exit
system, exit distribution, and the cost table (system vs. always-full-depth
baseline). Port of ETA-DyNN's ``summary_plot.py`` generalised to N side
exits; the final ViT head replaces the remote ViT4V stage.
"""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from experiments_db import EXPERIMENTS_DB, PERF_COLUMNS

P_IDX = {name: i for i, name in enumerate(PERF_COLUMNS)}
PHASES = ("preprocessing", "inference")


class SummaryPlot:
    def __init__(self, db_path, experiment_codename, dataset_codename,
                 confidence_plots, final_scores, labels):
        self.cps = list(confidence_plots)
        self.final_scores = np.asarray(final_scores, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.test_mask = np.ones(len(self.labels), dtype=bool)
        self.n_side = len(self.cps)
        for cp in self.cps:
            cp.set_final_plot(self)

        db = EXPERIMENTS_DB(db_path)
        self.energy_performance = {
            "exits": [{ph: np.array(v, dtype=np.float64) for ph, v in
                       db.get_performance(experiment_codename, dataset_codename, "ee_vit", e).items()}
                      for e in range(self.n_side)],
            "final": {ph: np.array(v, dtype=np.float64) for ph, v in
                      db.get_performance(experiment_codename, dataset_codename, "full_vit").items()},
            "transfer": np.array(db.get_transferring_performance(experiment_codename, dataset_codename),
                                 dtype=np.float64),
        }
        db.close()
        if len(self.energy_performance["final"]["inference"]) != len(self.labels):
            raise ValueError("Performance rows do not match the number of samples in the DB")

        self.fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1.6, 1, 1])
        self.ax_costs = self.fig.add_subplot(gs[:, 0])
        self.ax_total_cm = self.fig.add_subplot(gs[0, 1])
        self.ax_total_metrics = self.fig.add_subplot(gs[1, 1])
        self.ax_exits = self.fig.add_subplot(gs[:, 2])
        self.update_plot()
        plt.tight_layout(pad=3)

    def set_mask(self, mask):
        self.test_mask = np.asarray(mask, dtype=bool)

    # ---------------------------------------------------------- answers --

    def retrieve_answers(self):
        self.returned_by = [cp.curr_returned & self.test_mask for cp in self.cps]
        any_side = np.any(self.returned_by, axis=0) if self.n_side else np.zeros_like(self.test_mask)
        self.returned_by_final = self.test_mask & ~any_side
        self.final_answers = (self.final_scores > 0.5).astype(np.int64)

        self.global_answers = np.full(len(self.labels), -1, dtype=np.int64)
        for cp, mask in zip(self.cps, self.returned_by):
            self.global_answers[mask] = cp.complete_answers[mask]
        self.global_answers[self.returned_by_final] = self.final_answers[self.returned_by_final]

    # ------------------------------------------------------------ costs --

    def compute_costs(self, p_metric, phase):
        col = P_IDX[p_metric]
        baseline = self.energy_performance["final"][phase][:, col].copy()
        baseline[~self.test_mask] = 0
        system = np.zeros_like(baseline)
        for e, mask in enumerate(self.returned_by):
            system[mask] = self.energy_performance["exits"][e][phase][mask, col]
        system[self.returned_by_final] = baseline[self.returned_by_final]
        return float(system.sum()), float(baseline.sum())

    def compute_transferring_costs(self, p_metric):
        tr = self.energy_performance["transfer"]
        if tr.size == 0:
            return 0.0
        cost = tr[:, P_IDX[p_metric]].copy()
        cost[~self.returned_by_final] = 0
        return float(cost.sum())

    def retrieve_costs(self):
        self.costs = {}
        for metric in PERF_COLUMNS:
            pre, pre_base = self.compute_costs(metric, "preprocessing")
            inf, inf_base = self.compute_costs(metric, "inference")
            trans = self.compute_transferring_costs(metric)
            self.costs[metric] = {
                "baseline": {"preprocessing": pre_base, "inference": inf_base, "total": pre_base + inf_base},
                "system": {"preprocessing": pre, "inference": inf, "transferring": trans,
                           "total": pre + inf + trans},
            }

    def compute_metrics(self):
        self.retrieve_answers()
        self.retrieve_costs()
        labels = self.labels[self.test_mask]
        answers = self.global_answers[self.test_mask]
        self.metrics = {
            "cm": confusion_matrix(labels, answers, labels=[0, 1]),
            "acc": accuracy_score(labels, answers) if len(labels) else 0.0,
            "prec": precision_score(labels, answers, zero_division=0) if len(labels) else 0.0,
            "rec": recall_score(labels, answers, zero_division=0) if len(labels) else 0.0,
            "f1": f1_score(labels, answers, zero_division=0) if len(labels) else 0.0,
        }

    # ------------------------------------------------------------ plots --

    def savings(self, metric):
        base = self.costs[metric]["baseline"]["total"]
        if base == 0:
            return 0.0
        return (base - self.costs[metric]["system"]["total"]) / base * 100

    def update_plot(self, draw=True):
        self.compute_metrics()
        if not draw:
            return

        cm = self.metrics["cm"]
        self.ax_total_cm.clear()
        self.ax_total_cm.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                self.ax_total_cm.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color=color)
        self.ax_total_cm.set_title("Confusion Matrix", fontsize=11)
        self.ax_total_cm.set_xlabel("Predicted")
        self.ax_total_cm.set_ylabel("Ground Truth")
        self.ax_total_cm.set_xticks([0, 1], labels=["real", "fake"])
        self.ax_total_cm.set_yticks([0, 1], labels=["real", "fake"])

        n = max(self.test_mask.sum(), 1)
        text = (f'Accuracy:  {self.metrics["acc"]:.4f}\n'
                f'Precision: {self.metrics["prec"]:.4f}\n'
                f'Recall:    {self.metrics["rec"]:.4f}\n'
                f'F1 Score:  {self.metrics["f1"]:.4f}\n\n'
                f'Samples:   {int(self.test_mask.sum())}\n'
                f'Percentage returned: {(np.any(self.returned_by, axis=0) | self.returned_by_final).sum() / n * 100:.2f}%')
        self.ax_total_metrics.clear()
        self.ax_total_metrics.axis("off")
        self.ax_total_metrics.set_title("Metrics", fontsize=11)
        self.ax_total_metrics.text(0.05, 0.95, text, va="top", fontsize=9, family="monospace")

        self.ax_exits.clear()
        names = [f"exit {e}" for e in range(self.n_side)] + ["final"]
        fracs = [m.sum() / n * 100 for m in self.returned_by] + [self.returned_by_final.sum() / n * 100]
        bars = self.ax_exits.bar(names, fracs, color=["tab:blue"] * self.n_side + ["tab:orange"])
        for b, f in zip(bars, fracs):
            self.ax_exits.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{f:.1f}%",
                               ha="center", fontsize=8)
        self.ax_exits.set_ylim(0, 110)
        self.ax_exits.set_ylabel("% of samples answered")
        self.ax_exits.set_title("Exit distribution", fontsize=11)

        self.ax_costs.clear()
        self.ax_costs.axis("off")
        self.ax_costs.set_title("Costs", fontsize=11)
        rows = []
        for metric, label, unit, scale in (("duration", "Duration", "s", 1),
                                           ("tot_energy", "Tot Energy", "Wh", 1000),
                                           ("cpu_energy", "CPU Energy", "Wh", 1000),
                                           ("gpu_energy", "GPU Energy", "Wh", 1000),
                                           ("ram_energy", "RAM Energy", "Wh", 1000)):
            c = self.costs[metric]
            rows.append([f'{c["system"]["preprocessing"] * scale:.4f} {unit}',
                         f'{c["baseline"]["preprocessing"] * scale:.4f} {unit}',
                         f'{c["system"]["inference"] * scale:.4f} {unit}',
                         f'{c["baseline"]["inference"] * scale:.4f} {unit}',
                         f'{c["system"]["transferring"] * scale:.4f} {unit}',
                         f'{c["system"]["total"] * scale:.4f} {unit}',
                         f'{c["baseline"]["total"] * scale:.4f} {unit}',
                         f'{self.savings(metric):.2f}%'])
        table = self.ax_costs.table(
            cellText=rows,
            colLabels=["Sys (Pre)", "Base (Pre)", "Sys (Inf)", "Base (Inf)",
                       "Sys (Trans)", "Sys (Tot)", "Base (Tot)", "Savings"],
            rowLabels=["Duration", "Tot Energy", "CPU Energy", "GPU Energy", "RAM Energy"],
            bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        self.fig.canvas.draw_idle()

    def close(self):
        plt.close(self.fig)
