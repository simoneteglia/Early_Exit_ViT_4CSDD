"""Screen 2: mis/disinformation propagation on a social network with the
early-exit detector in the loop. Runs the four scenarios of ``propagation``
over a subset of test items and shows the network animation, mean reach
curves and the reach / cost table with savings vs. the full ViT.
"""
from __future__ import annotations

import json
import traceback

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.spatial import cKDTree
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QProgressBar, QPushButton, QSlider, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from experiments_db import PERF_COLUMNS
from human_factors import ThresholdPolicy
from propagation import (SCENARIOS, STATUS_I, STATUS_R, STATUS_S, SimParams, build_network, load_items,
                         mean_trend, sample_user_sensitivity, select_items, simulate, summarize)

STATUS_COLORS = {STATUS_S: "#d9d9d9", STATUS_I: "#ff7f0e", STATUS_R: "#b22222"}


class NetworkToolbar(NavigationToolbar):
    """Matplotlib toolbar that reports the view after its pan / zoom-box tools."""

    def __init__(self, canvas, parent, on_view_change):
        super().__init__(canvas, parent)
        self._on_view_change = on_view_change

    def push_current(self):
        super().push_current()
        self._on_view_change()  # remember, then re-equalize the scale around the chosen rectangle
SCENARIO_COLORS = dict(zip(SCENARIOS, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))


class SimulationWorker(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, items, params, policy, base_lower, base_upper, p):
        super().__init__()
        self.args = (items, params, policy, base_lower, base_upper, p)

    def run(self):
        items, params, policy, base_lower, base_upper, p = self.args
        try:
            rng = np.random.default_rng(params.seed)
            chosen = select_items(items, params.n_items, rng, source=params.source)
            if not chosen:
                raise ValueError(f"no test items for source '{params.source}'")
            g, g_dir = build_network(params)
            s_u = sample_user_sensitivity(g.number_of_nodes(), rng, params.mist_path)
            pos = nx.spring_layout(g, seed=params.seed, k=1.2 / np.sqrt(g.number_of_nodes()))
            rows = simulate(chosen, g_dir, s_u, policy, base_lower, base_upper, p, params,
                            progress=lambda i, n: self.progress.emit(i, n))
            self.done.emit({"rows": rows, "summary": summarize(rows), "graph": g, "pos": pos,
                            "s_u": s_u, "items": chosen, "params": params})
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())


class SimulationWindow(QWidget):
    def __init__(self, stacked_widget, db_path, experiment_codename, dataset_codename):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.db_args = (db_path, experiment_codename, dataset_codename)
        self.items = None
        self.result = None
        self.worker = None
        # Human-centered settings; overwritten by screen 1 on "Proceed to simulation".
        self.base_lower, self.base_upper = [], []
        self.policy = ThresholdPolicy()
        self.policy_strictness = 0.0

        outer = QVBoxLayout(self)
        header = QHBoxLayout()
        self.settings_label = QLabel("Thresholds/policy: not set (use 'Proceed to simulation' on screen 1)")
        self.settings_label.setWordWrap(True)
        self.back_btn = QPushButton("Back to thresholds")
        self.back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        header.addWidget(self.settings_label, 1)
        header.addWidget(self.back_btn)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.addWidget(self._build_controls())
        right = QVBoxLayout()
        right.addWidget(self._build_network_panel(), 3)
        bottom = QHBoxLayout()
        self.trend_fig, self.trend_axes = plt.subplots(1, 2, figsize=(9, 3.2))
        self.trend_canvas = FigureCanvas(self.trend_fig)
        bottom.addWidget(self.trend_canvas, 3)
        self.table = QTableWidget()
        bottom.addWidget(self.table, 4)
        right.addLayout(bottom, 2)
        body.addLayout(right, 1)
        outer.addLayout(body)

    # -------------------------------------------------------- controls --

    def _build_controls(self):
        box = QGroupBox("Propagation settings")
        form = QFormLayout(box)
        self.nodes = QSpinBox(); self.nodes.setRange(50, 20000); self.nodes.setValue(1000)
        self.ba_m = QSpinBox(); self.ba_m.setRange(1, 20); self.ba_m.setValue(3)
        self.p_share = QDoubleSpinBox(); self.p_share.setRange(0.0, 1.0); self.p_share.setSingleStep(0.01); self.p_share.setValue(0.15)
        self.kappa = QDoubleSpinBox(); self.kappa.setRange(0.0, 1.0); self.kappa.setSingleStep(0.05); self.kappa.setValue(0.5)
        self.intervention = QDoubleSpinBox(); self.intervention.setRange(0.0, 1.0); self.intervention.setSingleStep(0.05); self.intervention.setValue(0.5)
        self.seeds_per_item = QSpinBox(); self.seeds_per_item.setRange(1, 100); self.seeds_per_item.setValue(1)
        self.seed_strategy = QComboBox(); self.seed_strategy.addItems(["random", "hub"])
        self.n_items = QSpinBox(); self.n_items.setRange(2, 5000); self.n_items.setValue(60)
        self.source_combo = QComboBox(); self.source_combo.addItem("ALL")
        self.source_combo.setToolTip("Restrict the simulated items to one source dataset "
                                     "(only the mixed real/fake source is free of the source shortcut)")
        self.max_steps = QSpinBox(); self.max_steps.setRange(1, 500); self.max_steps.setValue(50)
        self.rng_seed = QSpinBox(); self.rng_seed.setRange(0, 10**6); self.rng_seed.setValue(0)
        self.mist_path = QLineEdit(); self.mist_path.setPlaceholderText("optional: MIST data file (OSF)")
        for label, w in (("Users (nodes)", self.nodes), ("BA edges per node", self.ba_m),
                         ("Base share prob. p_share", self.p_share),
                         ("Susceptibility effect κ", self.kappa),
                         ("Intervention w (1 = block)", self.intervention),
                         ("Seeds per item", self.seeds_per_item), ("Seed strategy", self.seed_strategy),
                         ("Test items", self.n_items), ("Source dataset", self.source_combo),
                         ("Max steps", self.max_steps),
                         ("Random seed", self.rng_seed), ("MIST file", self.mist_path)):
            form.addRow(label, w)
        self.run_btn = QPushButton("Run simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        self.progress = QProgressBar()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        form.addRow(self.run_btn)
        form.addRow(self.progress)
        form.addRow(self.status)
        box.setMaximumWidth(340)
        return box

    def _build_network_panel(self):
        box = QGroupBox("Network")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        self.item_combo = QComboBox()
        self.item_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.item_combo.setMinimumContentsLength(12)
        self.scenario_combo = QComboBox()
        self.scenario_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.scenario_combo.setMinimumContentsLength(8)
        for sc in SCENARIOS:
            self.scenario_combo.addItem(sc, sc)
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_label = QLabel("step 0")
        for w, stretch in ((QLabel("Item:"), 0), (self.item_combo, 2), (QLabel("Scenario:"), 0),
                           (self.scenario_combo, 1), (QLabel("Step:"), 0), (self.step_slider, 2),
                           (self.step_label, 0)):
            row.addWidget(w, stretch)
        layout.addLayout(row)
        self.item_summary = QLabel("")
        self.item_summary.setWordWrap(True)
        self.item_summary.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.item_summary)
        self.net_fig, self.net_ax = plt.subplots(figsize=(9, 5))
        self.net_fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9)  # canvas fully used
        self.net_canvas = FigureCanvas(self.net_fig)
        self.net_toolbar = NetworkToolbar(self.net_canvas, self, self._on_toolbar_view)  # pan / zoom-box / home / save
        for action in self.net_toolbar.actions():
            if action.text() == "Home":
                action.triggered.connect(self.reset_view)
        layout.addWidget(self.net_toolbar)
        layout.addWidget(self.net_canvas)
        # wheel = zoom around the cursor, left-drag = pan (when no toolbar tool is active), hover = tooltip
        self._view = None          # (xlim, ylim) set only by user zoom/pan; restored when the scene is rebuilt
        self._scene_key = None     # (item index, scenario) the current artists belong to
        self._nodes = self._flag_ring = None
        self._drag_origin = None
        self._tree = None
        self._tooltip = None
        self.net_canvas.mpl_connect("scroll_event", self._on_scroll)
        self.net_canvas.mpl_connect("button_press_event", self._on_press)
        self.net_canvas.mpl_connect("button_release_event", self._on_release)
        self.net_canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.net_canvas.mpl_connect("resize_event", self._on_resize)
        self.item_combo.currentIndexChanged.connect(lambda _: self._on_selection_changed())
        self.scenario_combo.currentIndexChanged.connect(lambda _: self._on_selection_changed())
        self.step_slider.valueChanged.connect(lambda _: self.draw_network())
        return box

    # --------------------------------------------------------- screen 1 --

    def update_simulation_data(self, data):
        """Receive thresholds / policy chosen on the thresholds screen."""
        if self.items is None:
            self.items = load_items(*self.db_args)
            self._populate_sources()
        self.base_lower = [lo for lo, _ in data["thresholds"]]
        self.base_upper = [up for _, up in data["thresholds"]]
        self.policy = ThresholdPolicy.from_dict(data["policy"])
        self.policy_strictness = float(data["policy_strictness"])
        thr = ", ".join(f"exit {e}: ({lo:.2f}, {up:.2f})" for e, (lo, up) in enumerate(data["thresholds"]))
        pol = self.policy.to_dict()
        self.settings_label.setText(
            f"Base thresholds — {thr} | margin: m0={pol['m0']:.2f} α={pol['alpha']:.2f} β={pol['beta']:.2f} "
            f"γ={pol['gamma']:.2f} m_max={pol['m_max']:.2f} | policy p={self.policy_strictness:.2f} "
            f"(user susceptibility is sampled per node from the MIST population)")

    # -------------------------------------------------------------- run --

    def _params(self):
        return SimParams(self.nodes.value(), self.ba_m.value(), self.p_share.value(), self.kappa.value(),
                         self.intervention.value(), self.seeds_per_item.value(),
                         self.seed_strategy.currentText(), self.max_steps.value(), self.rng_seed.value(),
                         self.mist_path.text().strip() or None)

    def _populate_sources(self):
        """Fill the source selector with the sources present in the DB, with a
        note on which contain both real and fake items."""
        counts = {}
        for it in self.items:
            counts.setdefault(it.source, [0, 0])[it.label] += 1
        current = self.source_combo.currentText()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("ALL")
        for src in sorted(counts):
            n_real, n_fake = counts[src]
            self.source_combo.addItem(src)
            self.source_combo.setItemData(self.source_combo.count() - 1,
                                          f"{n_real} real / {n_fake} fake" + ("" if n_real and n_fake else " (single class)"),
                                          Qt.ToolTipRole)
        idx = self.source_combo.findText(current)
        self.source_combo.setCurrentIndex(max(idx, 0))
        self.source_combo.blockSignals(False)

    def run_simulation(self):
        if self.worker is not None and self.worker.isRunning():
            return
        if self.items is None:
            self.items = load_items(*self.db_args)
            self._populate_sources()
        params = self._params()
        params.n_items = self.n_items.value()
        params.source = self.source_combo.currentText()
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("running...")
        self.worker = SimulationWorker(self.items, params, self.policy, self.base_lower, self.base_upper,
                                       self.policy_strictness)
        self.worker.progress.connect(lambda i, n: self.progress.setValue(int(i / n * 100)))
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_failed(self, msg):
        self.run_btn.setEnabled(True)
        self.status.setText(msg[-600:])

    def _on_done(self, result):
        self.result = result
        self.run_btn.setEnabled(True)
        self.progress.setValue(100)
        s = result["summary"]["ee_human"]
        red = s["fake_reach_reduction"]
        self.status.setText(f"done: {len(result['items'])} items ({result['params'].source}), "
                            f"{result['graph'].number_of_nodes()} users. Human-centered early exit: "
                            f"fake reach {'n/a' if np.isnan(red) else f'−{red*100:.0f}%'}, "
                            f"inference energy saving {s['savings']['tot_energy']:.0f}% vs full ViT.")
        self.fill_table()
        self.draw_trends()
        self._view = None
        self._scene_key = None
        xy = np.array([result["pos"][v] for v in range(result["graph"].number_of_nodes())])
        self._tree = cKDTree(xy)
        degrees = np.array([d for _, d in result["graph"].degree()])
        self._sizes = 8 + 40 * degrees / max(degrees.max(), 1)
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        pad = 0.03 * (hi - lo)
        self._extent = ((lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1]))
        self.item_combo.blockSignals(True)
        self.item_combo.clear()
        for it in result["items"]:
            self.item_combo.addItem(f"{it.sample_id}  [{'fake' if it.label else 'real'}, {it.tier}]")
        self.item_combo.blockSignals(False)
        self.scenario_combo.setCurrentIndex(SCENARIOS.index("ee_human"))
        self._on_selection_changed()

    # ------------------------------------------------------------ views --

    def _current_row(self):
        if self.result is None or self.item_combo.currentIndex() < 0:
            return None
        item = self.result["items"][self.item_combo.currentIndex()]
        sc = self.scenario_combo.currentData()
        for r in self.result["rows"]:
            if r["item"] == item.sample_id and r["scenario"] == sc:
                return r
        return None

    def _update_item_summary(self):
        """One line comparing the selected item across scenarios."""
        if self.result is None or self.item_combo.currentIndex() < 0:
            self.item_summary.setText("")
            return
        item = self.result["items"][self.item_combo.currentIndex()]
        n = self.result["graph"].number_of_nodes()
        rows = {r["scenario"]: r for r in self.result["rows"] if r["item"] == item.sample_id}
        parts = []
        for sc in SCENARIOS:
            r = rows.get(sc)
            if r is None:
                continue
            flag = "" if sc == "none" else (" flagged" if r["flagged"] else " not flagged")
            parts.append(f"{sc}: {r['n_shared']} users ({r['reach'] * 100:.1f}%), {r['steps']} steps{flag}")
        same = len({rows[sc]["n_shared"] for sc in rows}) == 1
        note = "   -> identical in all scenarios (detector never flagged this item)" if same else ""
        self.item_summary.setText(f"reach of this {'fake' if item.label else 'real'} item, {n} users  |  "
                                  + "  ·  ".join(parts) + note)

    def _on_selection_changed(self):
        row = self._current_row()
        if row is None:
            return
        self._update_item_summary()
        self.step_slider.blockSignals(True)
        self.step_slider.setRange(0, row["steps"])
        self.step_slider.setValue(row["steps"])
        self.step_slider.blockSignals(False)
        self.draw_network()

    def _status_at(self, row, step):
        status = np.full(self.result["graph"].number_of_nodes(), STATUS_S)
        for h in row["history"][:step + 1]:
            for node, st in h.items():
                status[node] = st
        return status

    def _build_scene(self, row):
        """Draw the static artists for one (item, scenario); the view is left
        untouched by later step updates, so zoom/pan persists."""
        g, pos = self.result["graph"], self.result["pos"]
        ax = self.net_ax
        ax.clear()
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.08, width=0.5)
        self._nodes = nx.draw_networkx_nodes(g, pos, ax=ax, node_size=self._sizes,
                                             node_color=STATUS_COLORS[STATUS_S], linewidths=0)
        self._flagged = list(row["flagged"])
        self._flag_ring = None
        if self._flagged:
            self._flag_ring = nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=self._flagged,
                                                     node_size=self._sizes[self._flagged], node_color="none",
                                                     edgecolors="black", linewidths=0.8)
        nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=row["seeds"], node_size=self._sizes[row["seeds"]] + 40,
                               node_color="none", edgecolors="green", linewidths=1.5)
        ax.set_aspect("auto")
        ax.set_autoscale_on(False)
        ax.set_axis_off()
        self._apply_view()
        self._scene_key = (self.item_combo.currentIndex(), self.scenario_combo.currentData())

    def draw_network(self):
        row = self._current_row()
        if row is None:
            return
        key = (self.item_combo.currentIndex(), self.scenario_combo.currentData())
        if self._scene_key != key or self._nodes is None:
            self._build_scene(row)
        step = self.step_slider.value()
        status = self._status_at(row, step)
        self._nodes.set_facecolor([STATUS_COLORS[st] for st in status])
        if self._flag_ring is not None:
            self._flag_ring.set_edgecolor(["black" if status[v] != STATUS_S else "none" for v in self._flagged])
        if self._tooltip is not None:
            self._tooltip.remove()
            self._tooltip = None
        n = self.result["graph"].number_of_nodes()
        reached = int((status != STATUS_S).sum())
        self.net_ax.set_title(f"{row['item']} ({'fake' if row['label'] else 'real'}, {row['tier']}) — "
                              f"{self.scenario_combo.currentData()} — step {step}/{row['steps']}: "
                              f"{reached} users reached ({reached / n * 100:.1f}%)\n"
                              f"grey: unaware · orange: sharing · red: shared · black ring: flagged · green ring: seed",
                              fontsize=8)
        self.step_label.setText(f"step {step}")
        self.net_canvas.draw_idle()

    # ----------------------------------------------------- interaction --

    def _remember_view(self):
        self._view = (self.net_ax.get_xlim(), self.net_ax.get_ylim())

    def _equalized_limits(self, xlim, ylim):
        """Limits with the same data-units-per-pixel on both axes that contain
        the given ranges and fill the axes box: no distortion, no clipping."""
        ax = self.net_ax
        pos = ax.get_position()
        w = self.net_fig.bbox.width * pos.width
        h = self.net_fig.bbox.height * pos.height
        if w <= 0 or h <= 0:
            return xlim, ylim
        scale = max((xlim[1] - xlim[0]) / w, (ylim[1] - ylim[0]) / h)
        xc, yc = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
        return (xc - scale * w / 2, xc + scale * w / 2), (yc - scale * h / 2, yc + scale * h / 2)

    def _on_toolbar_view(self):
        self._remember_view()
        self._apply_view()
        self._remember_view()
        self.net_canvas.draw_idle()

    def _apply_view(self):
        """Fit the whole network (or the user's view) to the current canvas size."""
        if self._nodes is None:
            return
        xlim, ylim = self._equalized_limits(*(self._view if self._view is not None else self._extent))
        self.net_ax.set_xlim(xlim)
        self.net_ax.set_ylim(ylim)

    def reset_view(self):
        self._view = None
        self._scene_key = None
        self.draw_network()

    def _on_resize(self, event):
        self._apply_view()

    def _on_scroll(self, event):
        if event.inaxes != self.net_ax or event.xdata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.net_ax.get_xlim()
        y0, y1 = self.net_ax.get_ylim()
        x, y = event.xdata, event.ydata
        self.net_ax.set_xlim(x - (x - x0) * factor, x + (x1 - x) * factor)
        self.net_ax.set_ylim(y - (y - y0) * factor, y + (y1 - y) * factor)
        self._remember_view()
        self.net_canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes == self.net_ax and event.button == 1 and self.net_toolbar.mode == "":
            self._drag_origin = (event.xdata, event.ydata)

    def _on_release(self, event):
        self._drag_origin = None

    def _on_motion(self, event):
        if event.inaxes != self.net_ax or event.xdata is None:
            return
        if self._drag_origin is not None:
            dx = event.xdata - self._drag_origin[0]
            dy = event.ydata - self._drag_origin[1]
            x0, x1 = self.net_ax.get_xlim()
            y0, y1 = self.net_ax.get_ylim()
            self.net_ax.set_xlim(x0 - dx, x1 - dx)
            self.net_ax.set_ylim(y0 - dy, y1 - dy)
            self._remember_view()
            self.net_canvas.draw_idle()
            return
        self._show_tooltip(event)

    def _show_tooltip(self, event):
        row = self._current_row()
        if row is None or self._tree is None:
            return
        x0, x1 = self.net_ax.get_xlim()
        dist, node = self._tree.query([event.xdata, event.ydata])
        if self._tooltip is not None:
            self._tooltip.remove()
            self._tooltip = None
        if dist < 0.015 * (x1 - x0):  # within ~1.5% of the visible width
            g = self.result["graph"]
            status = self._status_at(row, self.step_slider.value())
            names = {STATUS_S: "unaware", STATUS_I: "sharing", STATUS_R: "shared"}
            e = row["exit_idx"][node]
            n_side = len(row["exits_hist"]) - 1
            verdict = ("no detector" if e < 0 else
                       f"{'final head' if e == n_side else f'exit {e}'} -> {'fake' if node in set(row['flagged']) else 'real'}")
            text = (f"user {node}  degree {g.degree(node)}  s_u {self.result['s_u'][node]:.2f}\n"
                    f"{names[int(status[node])]}  |  {verdict}")
            self._tooltip = self.net_ax.annotate(
                text, xy=self.result["pos"][node], xytext=(12, 12), textcoords="offset points", fontsize=8,
                bbox=dict(boxstyle="round", fc="lightyellow", ec="grey", alpha=0.95), zorder=10)
        self.net_canvas.draw_idle()

    def draw_trends(self):
        rows = self.result["rows"]
        for ax, label, title in ((self.trend_axes[0], 1, "Fake items"), (self.trend_axes[1], 0, "Real items")):
            ax.clear()
            for sc in SCENARIOS:
                tr = mean_trend([r for r in rows if r["scenario"] == sc], label)
                ax.plot(range(len(tr)), [t * 100 for t in tr], color=SCENARIO_COLORS[sc], label=sc)
            ax.set_title(f"{title}: mean cumulative reach", fontsize=9)
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("% users reached", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
        self.trend_fig.tight_layout()
        self.trend_canvas.draw_idle()

    def fill_table(self):
        summary = self.result["summary"]
        n_side = len(summary["ee_human"]["exit_distribution"]) - 1
        cols = ["Scenario", "Fake reach", "Fake reduction", "Real reach", "Real loss", "Detector runs",
                "Inference time", "Time saving", "Inference energy", "Energy saving",
                "Exits " + " / ".join([f"e{e}" for e in range(n_side)] + ["final"])]
        self.table.clear()
        self.table.setRowCount(len(SCENARIOS))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        for i, sc in enumerate(SCENARIOS):
            s = summary[sc]
            pct = lambda v, d=2: "n/a" if v is None or np.isnan(v) else f"{v*100:.{d}f}%"
            values = [s["label"], pct(s["fake_reach"]), pct(s["fake_reach_reduction"], 1),
                      pct(s["real_reach"]), pct(s["real_reach_loss"], 1), f"{s['invocations']}",
                      f"{s['cost']['duration']:.3f} s", f"{s['savings']['duration']:.1f}%",
                      f"{s['cost']['tot_energy']*1000:.4f} Wh", f"{s['savings']['tot_energy']:.1f}%",
                      " / ".join(f"{d*100:.0f}%" for d in s["exit_distribution"])]
            for j, v in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(v))
        self.table.resizeColumnsToContents()

    def export_results(self, path):
        with open(path, "w") as f:
            json.dump({"params": self.result["params"].to_dict(), "summary": self.result["summary"]},
                      f, indent=1, default=float)
