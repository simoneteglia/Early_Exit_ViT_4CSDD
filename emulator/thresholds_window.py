"""Screen 1: per-exit confidence panels + global summary, generalised to N
side exits, with human-factor controls (user sensitivity, policy, margin
coefficients) and a content-sensitivity tier filter.
Port of ETA-DyNN's ``thresholds_window.py``.
"""
from __future__ import annotations

import itertools
import json
import pickle

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)
from tqdm import tqdm

from confidence_plot import ConfidencePlot
from experiments_db import EXPERIMENTS_DB
from human_factors import CSS_TIERS, ThresholdPolicy, normalize_css
from summary_plot import SummaryPlot


def _title_label(text):
    label = QLabel(text)
    label.setAlignment(Qt.AlignCenter)
    font = QFont("Arial", 14)
    font.setBold(True)
    font.setItalic(True)
    label.setFont(font)
    label.setFixedWidth(140)
    return label


class ConfidencePlotWidget(QWidget):
    def __init__(self, cp: ConfidencePlot, model_codename="ee_vit"):
        super().__init__()
        self.cp = cp
        layout = QHBoxLayout(self)
        layout.addWidget(_title_label(f"Model: {model_codename}\nExit: {cp.exit_idx}"))
        self.canvas = FigureCanvas(cp.fig)
        self.canvas.setMinimumHeight(360)
        layout.addWidget(self.canvas)


class SummaryPlotWidget(QWidget):
    def __init__(self, sp: SummaryPlot):
        super().__init__()
        self.sp = sp
        layout = QHBoxLayout(self)
        layout.addWidget(_title_label("Global Results"))
        self.canvas = FigureCanvas(sp.fig)
        self.canvas.setMinimumHeight(360)
        layout.addWidget(self.canvas)


class ThresholdsWindow(QWidget):
    def __init__(self, stacked_widget, scores_path, db_path, experiment_codename, dataset_codename,
                 reference_split="train", thresholds_path=None, policy: ThresholdPolicy | None = None):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.simulation_widget = None
        self.policy = policy or ThresholdPolicy()
        self.setWindowTitle("Early-exit ViT — thresholds")

        # Reference distribution (from calibrate.py) and test data (from extract.py).
        with open(scores_path, "rb") as f:
            ref = pickle.load(f)[reference_split]
        self.ref_labels = np.asarray(ref["labels"], dtype=np.int64)
        self.ref_css = np.asarray(ref["css_scores"], dtype=np.float32)
        self.ref_tiers = np.asarray(ref.get("tiers", ["ALL"] * len(self.ref_labels)), dtype=object)
        ref_scores = {s["exit_idx"]: np.asarray(s["scores"])[:, 1] for s in ref["scores"]}

        db = EXPERIMENTS_DB(db_path)
        self.exit_indexes = db.get_exit_indexes(experiment_codename, dataset_codename)
        self.test_labels = np.asarray(db.get_labels(dataset_codename), dtype=np.int64)
        css = db.get_css_scores(dataset_codename)
        self.test_css = np.asarray([1.0 if c is None else c for c in css], dtype=np.float32)
        tiers = db.get_sensitivity_tiers(dataset_codename)
        self.test_tiers = np.asarray(["ALL" if t is None else t for t in tiers], dtype=object)
        test_scores = {e: np.array([np.mean(s) for s in db.get_scores(experiment_codename, dataset_codename, "ee_vit", e)],
                                   dtype=np.float32) for e in self.exit_indexes}
        final_scores = np.array([np.mean(s) for s in db.get_scores(experiment_codename, dataset_codename, "full_vit")],
                                dtype=np.float32)
        db.close()
        if len(final_scores) != len(self.test_labels):
            raise ValueError("full_vit scores missing for some samples; re-run extract.py")

        base = self._load_base_thresholds(thresholds_path)

        # Per-exit panels chained in a waterfall.
        self.confidence_panels: list[ConfidencePlot] = []
        for e in self.exit_indexes:
            lo, up = base.get(e, (0.0, 1.0))
            cp = ConfidencePlot(e, ref_scores[e], self.ref_labels, test_scores[e], self.test_labels,
                                prev_pred=self.confidence_panels[-1] if self.confidence_panels else None,
                                lower=lo, upper=up)
            if self.confidence_panels:
                self.confidence_panels[-1].set_next_pred(cp)
            self.confidence_panels.append(cp)
        self.summary = SummaryPlot(db_path, experiment_codename, dataset_codename,
                                   self.confidence_panels, final_scores, self.test_labels)

        # ---------------------------------------------------------- layout --
        outer = QVBoxLayout(self)
        outer.addWidget(self._build_controls())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        self.plot_widgets = [ConfidencePlotWidget(cp) for cp in self.confidence_panels]
        for w in self.plot_widgets:
            layout.addWidget(w)
        self.summary_widget = SummaryPlotWidget(self.summary)
        layout.addWidget(self.summary_widget)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        buttons = QHBoxLayout()
        self.reset_btn = QPushButton("Reset base thresholds" + (" (from thresholds.json)" if base else ""))
        self.reset_btn.clicked.connect(lambda: self.set_thresholds(base))
        self.compare_btn = QPushButton("Compare thresholds performance")
        self.compare_btn.clicked.connect(self.compare_thresholds)
        self.proceed_btn = QPushButton("Proceed to simulation")
        self.proceed_btn.clicked.connect(self.switch_to_simulation)
        self.proceed_btn.setEnabled(False)
        for b in (self.reset_btn, self.compare_btn, self.proceed_btn):
            buttons.addWidget(b)
        outer.addLayout(buttons)

        self.resize_histograms()
        self.apply_context()

    # -------------------------------------------------------- controls --

    def _build_controls(self):
        box = QGroupBox("Human-centered factors")
        grid = QGridLayout(box)

        self.tier_combo = QComboBox()
        levels = list(CSS_TIERS.values())
        extra = sorted(set(self.test_tiers.tolist()) - set(levels) - {"ALL"})
        self.tier_combo.addItems(["ALL"] + levels + extra)
        self.tier_combo.currentTextChanged.connect(lambda _: self.apply_context())
        grid.addWidget(QLabel("Content sensitivity tier:"), 0, 0)
        grid.addWidget(self.tier_combo, 0, 1)

        self.user_slider, self.user_value = self._slider("User susceptibility s_u", grid, 1)
        self.policy_slider, self.policy_value = self._slider("Policy strictness p", grid, 2)

        self.coef_boxes = {}
        for col, (name, tip) in enumerate((("m0", "base margin"), ("alpha", "content sensitivity weight"),
                                           ("beta", "user susceptibility weight"), ("gamma", "policy weight"),
                                           ("m_max", "margin cap"))):
            grid.addWidget(QLabel(f"{name} ({tip})"), 3, 2 * col)
            sb = QDoubleSpinBox()
            sb.setRange(0.0, 1.0)
            sb.setSingleStep(0.05)
            sb.setDecimals(2)
            sb.setValue(getattr(self.policy, name))
            sb.valueChanged.connect(lambda v, n=name: self._set_coef(n, v))
            grid.addWidget(sb, 3, 2 * col + 1)
            self.coef_boxes[name] = sb

        self.margin_label = QLabel("")
        grid.addWidget(self.margin_label, 4, 0, 1, 6)
        return box

    def _slider(self, name, grid, row):
        grid.addWidget(QLabel(name + ":"), row, 0)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(0)
        value = QLabel("0.00")
        slider.valueChanged.connect(lambda v: (value.setText(f"{v / 100:.2f}"), self.apply_context()))
        grid.addWidget(slider, row, 1, 1, 4)
        grid.addWidget(value, row, 5)
        return slider, value

    def _set_coef(self, name, value):
        setattr(self.policy, name, float(value))
        self.apply_context()

    @property
    def user_sensitivity(self):
        return self.user_slider.value() / 100

    @property
    def policy_strictness(self):
        return self.policy_slider.value() / 100

    # ----------------------------------------------------------- state --

    @staticmethod
    def _load_base_thresholds(path):
        if not path:
            return {}
        with open(path) as f:
            data = json.load(f)
        return {t["exit_idx"]: (t["lower"], t["upper"]) for t in data["side_exits"]}

    def apply_context(self, draw=True):
        """Recompute margins and tier masks from the controls and refresh everything."""
        tier = self.tier_combo.currentText()
        ref_mask = np.ones(len(self.ref_labels), bool) if tier == "ALL" else self.ref_tiers == tier
        test_mask = np.ones(len(self.test_labels), bool) if tier == "ALL" else self.test_tiers == tier
        s_u, p = self.user_sensitivity, self.policy_strictness
        ref_margin = self.policy.margin(normalize_css(self.ref_css), s_u, p)
        test_margin = self.policy.margin(normalize_css(self.test_css), s_u, p)
        for cp in self.confidence_panels:
            cp.set_context(ref_margin, test_margin, ref_mask, test_mask)
        self.summary.set_mask(test_mask)
        m = test_margin[test_mask]
        self.margin_label.setText(
            f"margin m on the selected test samples: mean={np.mean(m) if len(m) else 0:.2f}, "
            f"min={np.min(m) if len(m) else 0:.2f}, max={np.max(m) if len(m) else 0:.2f}   "
            f"({int(test_mask.sum())} samples)")
        self.refresh(draw)

    def refresh(self, draw=True):
        if self.confidence_panels:
            self.confidence_panels[0].update_metrics(draw=draw)
        else:
            self.summary.update_plot(draw=draw)
        if draw:
            self.resize_histograms()

    def resize_histograms(self):
        if not self.confidence_panels:
            return
        max_y = max(cp.max_hist_y for cp in self.confidence_panels)
        for cp in self.confidence_panels:
            cp.set_hist_ylim(0, max_y + 0.05 * max_y)
            cp.fig.canvas.draw_idle()

    def set_thresholds(self, thresholds: dict | None = None):
        thresholds = thresholds or {}
        for cp in self.confidence_panels:
            lo, up = thresholds.get(cp.exit_idx, (0.0, 1.0))
            cp.line1.set_xdata([lo, lo])
            cp.line2.set_xdata([up, up])
            cp.lower_threshold, cp.upper_threshold = float(lo), float(up)
        self.refresh()

    def get_classification_data(self):
        sp = self.summary
        return {
            "returned_by": [m.copy() for m in sp.returned_by],
            "returned_by_final": sp.returned_by_final.copy(),
            "global_answers": sp.global_answers.copy(),
            "final_answers": sp.final_answers.copy(),
            "labels": sp.labels.copy(),
            "css_scores": self.test_css.copy(),
            "energy_performance": sp.energy_performance,
            "thresholds": [cp.get_thresholds() for cp in self.confidence_panels],
            "policy": self.policy.to_dict(),
            "user_sensitivity": self.user_sensitivity,
            "policy_strictness": self.policy_strictness,
        }

    def set_simulation_widget(self, widget):
        self.simulation_widget = widget
        self.proceed_btn.setEnabled(widget is not None)

    def switch_to_simulation(self):
        if self.simulation_widget is None:
            return
        self.simulation_widget.update_simulation_data(self.get_classification_data())
        self.stacked_widget.setCurrentIndex(1)

    def compare_thresholds(self, n=4, out_path="thresholds_search_data.pkl"):
        """Grid search over per-exit base thresholds under the current human-factor
        context; records (thresholds..., energy saving %, accuracy) like the reference."""
        current = [cp.get_thresholds() for cp in self.confidence_panels]
        grids = [(np.linspace(0.01, 0.30, n), np.linspace(0.70, 0.99, n)) for _ in self.confidence_panels]
        rows = []
        for combo in tqdm(list(itertools.product(*[itertools.product(lo, up) for lo, up in grids]))):
            for cp, (lo, up) in zip(self.confidence_panels, combo):
                cp.line1.set_xdata([lo, lo])
                cp.line2.set_xdata([up, up])
                cp.lower_threshold, cp.upper_threshold = float(lo), float(up)
            self.refresh(draw=False)
            rows.append(np.array([*np.ravel(combo), self.summary.savings("tot_energy"), self.summary.metrics["acc"]]))
        with open(out_path, "wb") as f:
            pickle.dump(np.array(rows), f)
        for cp, (lo, up) in zip(self.confidence_panels, current):
            cp.line1.set_xdata([lo, lo])
            cp.line2.set_xdata([up, up])
            cp.lower_threshold, cp.upper_threshold = lo, up
        self.refresh()
        print(f"wrote {len(rows)} threshold combinations to {out_path}")

    def close_panels(self):
        for cp in self.confidence_panels:
            cp.close()
        self.summary.close()
