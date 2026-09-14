"""Per-exit panel: confidence histogram of the reference split with two
draggable base-threshold lines, and train/test confusion matrices + metrics
for the samples this exit answers. Port of ETA-DyNN's ``confidence_plot.py``
generalised to per-sample effective thresholds (base thresholds widened by
the human-factor margin) and to an optional sample mask (sensitivity tier).
"""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from human_factors import ThresholdPolicy


class ConfidencePlot:
    def __init__(self,
                 exit_idx: int,
                 ref_scores: np.ndarray, ref_labels: np.ndarray,
                 test_scores: np.ndarray, test_labels: np.ndarray,
                 prev_pred: ConfidencePlot | None = None,
                 next_pred: ConfidencePlot | None = None,
                 lower: float = 0.0, upper: float = 1.0):
        self.exit_idx = exit_idx
        self.ref_scores = np.asarray(ref_scores, dtype=np.float32)
        self.ref_labels = np.asarray(ref_labels, dtype=np.int64)
        self.test_scores = np.asarray(test_scores, dtype=np.float32)
        self.test_labels = np.asarray(test_labels, dtype=np.int64)

        # Human-factor context: per-sample margin (0 = confidence-only) and mask.
        self.ref_margin = np.zeros(len(self.ref_scores), dtype=np.float32)
        self.test_margin = np.zeros(len(self.test_scores), dtype=np.float32)
        self.ref_mask = np.ones(len(self.ref_scores), dtype=bool)
        self.test_mask = np.ones(len(self.test_scores), dtype=bool)

        self.returned = np.zeros_like(self.test_labels, dtype=bool)
        self.curr_returned = np.zeros_like(self.test_labels, dtype=bool)
        self.complete_answers = None
        self.prev_pred = prev_pred
        self.next_pred = next_pred
        self.final_plot = None
        self.training_metrics = None
        self.testing_metrics = None

        self.fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(2, 3)
        self.ax_hist = self.fig.add_subplot(gs[:, 0])
        self.ax_train_cm = self.fig.add_subplot(gs[0, 1])
        self.ax_train_metrics = self.fig.add_subplot(gs[1, 1])
        self.ax_test_cm = self.fig.add_subplot(gs[0, 2])
        self.ax_test_metrics = self.fig.add_subplot(gs[1, 2])

        self._bars = None
        self._band = None
        self.max_hist_y = 0.0
        self.redraw_hist()
        self.ax_hist.set_xlabel("Confidence (P(fake))")
        self.ax_hist.set_ylabel("Density")
        self.ax_hist.set_xlim(0, 1)

        self.line1 = self.ax_hist.axvline(lower, color="blue", linewidth=2, picker=5)
        self.line2 = self.ax_hist.axvline(upper, color="red", linewidth=2, picker=5)
        self.lower_threshold = float(lower)
        self.upper_threshold = float(upper)
        self.selected_line = None

        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        self.update_metrics()
        plt.tight_layout(pad=3)

    # ----------------------------------------------------------- events --

    def on_pick(self, event):
        if event.artist in (self.line1, self.line2):
            self.selected_line = event.artist

    def on_mouse_move(self, event):
        if self.selected_line is None or event.inaxes != self.ax_hist:
            return
        self.selected_line.set_xdata([event.xdata, event.xdata])
        self.lower_threshold = float(self.line1.get_xdata()[0])
        self.upper_threshold = float(self.line2.get_xdata()[0])
        self.update_metrics()

    def on_release(self, event):
        self.selected_line = None

    # ------------------------------------------------------------ state --

    def set_thresholds(self, lower, upper, draw=True):
        self.line1.set_xdata([lower, lower])
        self.line2.set_xdata([upper, upper])
        self.lower_threshold = float(lower)
        self.upper_threshold = float(upper)
        self.update_metrics(draw=draw)

    def get_thresholds(self):
        return self.lower_threshold, self.upper_threshold

    def set_context(self, ref_margin=None, test_margin=None, ref_mask=None, test_mask=None):
        """Set human-factor margins / tier masks without recomputing (call
        ``update_metrics`` on the first exit of the chain afterwards)."""
        if ref_margin is not None:
            self.ref_margin = np.asarray(ref_margin, dtype=np.float32)
        if test_margin is not None:
            self.test_margin = np.asarray(test_margin, dtype=np.float32)
        if ref_mask is not None:
            ref_mask = np.asarray(ref_mask, dtype=bool)
            if not np.array_equal(ref_mask, self.ref_mask):
                self.ref_mask = ref_mask
                self.redraw_hist()
        if test_mask is not None:
            self.test_mask = np.asarray(test_mask, dtype=bool)

    def set_prev_pred(self, prev_pred):
        self.prev_pred = prev_pred

    def set_next_pred(self, next_pred):
        self.next_pred = next_pred

    def set_final_plot(self, final_plot):
        self.final_plot = final_plot

    def set_hist_ylim(self, lower_lim, upper_lim):
        self.ax_hist.set_ylim(lower_lim, upper_lim)

    # ---------------------------------------------------------- answers --

    def effective_thresholds(self, margin):
        lo, up = ThresholdPolicy.widen(self.lower_threshold, self.upper_threshold, margin)
        return np.asarray(lo, dtype=np.float32), np.asarray(up, dtype=np.float32)

    def _triage(self, scores, margin, mask):
        lo, up = self.effective_thresholds(margin)
        answers = np.full(len(scores), -1, dtype=np.int64)
        answers[scores < lo] = 0
        answers[scores > up] = 1
        answers[~mask] = -1
        return answers

    def compute_answers(self):
        answers = self._triage(self.test_scores, self.test_margin, self.test_mask)
        self.complete_answers = answers.copy()
        valid = answers != -1
        if self.prev_pred is not None:
            self.returned = self.prev_pred.returned | valid
            self.curr_returned = ~self.prev_pred.returned & valid
        else:
            self.returned = valid
            self.curr_returned = valid
        return self.test_labels[self.curr_returned], answers[self.curr_returned]

    def compute_training_answers(self):
        answers = self._triage(self.ref_scores, self.ref_margin, self.ref_mask)
        valid = answers != -1
        self.valid_training_fraction = valid.sum() / max(self.ref_mask.sum(), 1)
        return self.ref_labels[valid], answers[valid]

    @staticmethod
    def _metrics(labels, answers):
        if len(answers) == 0:
            return None
        cm = confusion_matrix(labels, answers, labels=[0, 1])
        return {"cm": cm,
                "acc": accuracy_score(labels, answers),
                "prec": precision_score(labels, answers, zero_division=0),
                "rec": recall_score(labels, answers, zero_division=0),
                "f1": f1_score(labels, answers, zero_division=0)}

    def compute_metrics(self):
        self.training_metrics = self._metrics(*self.compute_training_answers())
        self.testing_metrics = self._metrics(*self.compute_answers())

    def update_metrics(self, draw=True):
        """Recompute answers/metrics for this exit and cascade down the chain."""
        self.compute_metrics()
        if draw:
            self.update_training_testing_plots()
        if self.next_pred is not None:
            self.next_pred.update_metrics(draw=draw)
        elif self.final_plot is not None:
            self.final_plot.update_plot(draw=draw)

    # ------------------------------------------------------------ plots --

    def redraw_hist(self):
        if self._bars is not None:
            self._bars.remove()
        data = self.ref_scores[self.ref_mask]
        counts, _, self._bars = self.ax_hist.hist(data if len(data) else [0.0], bins=50, range=(0, 1),
                                                  color="tab:blue", edgecolor="black", density=True)
        self.max_hist_y = float(np.max(counts)) if len(data) else 0.0

    def _draw_band(self):
        if self._band is not None:
            self._band.remove()
            self._band = None
        m = self.test_margin[self.test_mask]
        if len(m) == 0:
            return
        lo, up = self.effective_thresholds(np.mean(m))
        self._band = self.ax_hist.axvspan(float(lo), float(up), color="grey", alpha=0.15)

    def update_training_testing_plots(self):
        self.update_plots(self.training_metrics, self.ax_train_cm, self.ax_train_metrics, "training")
        self.update_plots(self.testing_metrics, self.ax_test_cm, self.ax_test_metrics, "testing")
        self._draw_band()
        self.fig.canvas.draw_idle()

    def update_plots(self, metrics, ax_cm, ax_metrics, stage):
        ax_cm.clear()
        if metrics is not None:
            ax_cm.imshow(metrics["cm"], cmap="Blues")
            for i in range(2):
                for j in range(2):
                    color = "white" if metrics["cm"][i, j] > metrics["cm"].max() / 2 else "black"
                    ax_cm.text(j, i, metrics["cm"][i, j], ha="center", va="center", fontsize=9, color=color)
            text = (f'Accuracy:  {metrics["acc"]:.4f}\n'
                    f'Precision: {metrics["prec"]:.4f}\n'
                    f'Recall:    {metrics["rec"]:.4f}\n'
                    f'F1 Score:  {metrics["f1"]:.4f}\n\n')
        else:
            text = "Accuracy:  N/A\nPrecision: N/A\nRecall:    N/A\nF1 Score:  N/A\n\n"
        if stage == "testing":
            n = max(self.test_mask.sum(), 1)
            text += (f"Answered here:       {self.curr_returned.sum() / n * 100:.2f}%\n"
                     f"Percentage returned: {self.returned.sum() / n * 100:.2f}%")
        else:
            text += f"Valid answers:       {getattr(self, 'valid_training_fraction', 0) * 100:.2f}%"
        m = self.test_margin[self.test_mask]
        mean_m = float(np.mean(m)) if len(m) else 0.0
        ax_cm.set_title(f"Confusion Matrix\n(lower={self.lower_threshold:.3f}, upper={self.upper_threshold:.3f}, "
                        f"mean margin={mean_m:.2f})", fontsize=10)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Ground Truth")
        ax_cm.set_xticks([0, 1], labels=["real", "fake"])
        ax_cm.set_yticks([0, 1], labels=["real", "fake"])

        ax_metrics.clear()
        ax_metrics.axis("off")
        ax_metrics.set_title("Metrics", fontsize=11)
        ax_metrics.text(0.05, 0.95, text, va="top", fontsize=9, family="monospace")

    def close(self):
        plt.close(self.fig)
