"""Interactive threshold explorer for the early-exit ViT (screen 1 of the
ETA-DyNN activity emulator, adapted).

    python emulator/main.py --scores runs/ee_vit/scores.pkl --db experiments.db \
        --experiment exp_1 --dataset css_test --thresholds runs/ee_vit/thresholds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from PyQt5.QtWidgets import QApplication, QStackedWidget  # noqa: E402

from human_factors import ThresholdPolicy  # noqa: E402
from simulation_window import SimulationWindow  # noqa: E402
from thresholds_window import ThresholdsWindow  # noqa: E402


class MainWindow(QStackedWidget):
    def __init__(self, args):
        super().__init__()
        policy = ThresholdPolicy.from_dict(json.load(open(args.policy))) if args.policy else ThresholdPolicy()
        self.thresholds_window = ThresholdsWindow(self, args.scores, args.db, args.experiment, args.dataset,
                                                  reference_split=args.reference_split,
                                                  thresholds_path=args.thresholds, policy=policy)
        self.simulation_window = SimulationWindow(self, args.db, args.experiment, args.dataset)
        self.thresholds_window.set_simulation_widget(self.simulation_window)
        self.addWidget(self.thresholds_window)  # index 0
        self.addWidget(self.simulation_window)  # index 1
        self.setWindowTitle("Deepfake early-exit explorer")
        self.showMaximized()

    def close_panels(self):
        self.thresholds_window.close_panels()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="scores.pkl written by calibrate.py")
    p.add_argument("--db", default="experiments.db", help="experiments.db written by extract.py")
    p.add_argument("--experiment", required=True, help="Experiment codename")
    p.add_argument("--dataset", default="css_test", help="Dataset codename")
    p.add_argument("--thresholds", default=None, help="thresholds.json written by calibrate.py")
    p.add_argument("--reference-split", default="train", help="Split of scores.pkl shown as reference distribution")
    p.add_argument("--policy", default=None, help="JSON with m0/alpha/beta/gamma/m_max")
    return p.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    w = MainWindow(args)
    w.show()
    app.aboutToQuit.connect(lambda: (w.close_panels(), print("Gracefully stopping...")))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
