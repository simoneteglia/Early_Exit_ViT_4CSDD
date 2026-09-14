"""SQLite store of per-sample scores and per-exit costs, adapted from
ETA-DyNN's ``ee_cnn/experiments_db.py``.

Differences: model codenames are ``ee_vit`` (side exits, ``exit_idx`` set),
``full_vit`` (final head only, ``exit_idx`` NULL) and ``transfer``; samples
carry their content-sensitivity score, tier and scenario.
"""
from __future__ import annotations

import json
import logging
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)

MODEL_CODENAMES = ("ee_vit", "full_vit", "transfer")
PHASES = ("preprocessing", "inference", "transferring")
PERF_COLUMNS = ("duration", "tot_energy", "cpu_energy", "gpu_energy", "ram_energy")


class EXPERIMENTS_DB:
    def __init__(self, path="experiments.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.cur = self.conn.cursor()
        self._create_schema()

    def _create_schema(self):
        self.cur.executescript(f"""
        CREATE TABLE IF NOT EXISTS Experiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            codename TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS Dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            codename TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS ExperimentDataset (
            experiment_id INTEGER NOT NULL,
            dataset_id INTEGER NOT NULL,
            PRIMARY KEY (experiment_id, dataset_id),
            FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE,
            FOREIGN KEY (dataset_id) REFERENCES Dataset(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS Sample (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            label INTEGER NOT NULL CHECK (label in (0, 1)),
            css_score REAL,
            sensitivity_tier TEXT,
            scenario TEXT,
            UNIQUE (dataset_id, file_path),
            FOREIGN KEY (dataset_id) REFERENCES Dataset(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS ModelRun (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL,
            experiment_id INTEGER NOT NULL,
            model_codename TEXT NOT NULL CHECK (model_codename IN {MODEL_CODENAMES}),
            exit_idx INTEGER,
            scores BLOB,
            FOREIGN KEY (sample_id) REFERENCES Sample(id) ON DELETE CASCADE,
            FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS Performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modelrun_id INTEGER NOT NULL,
            phase TEXT NOT NULL CHECK (phase IN {PHASES}),
            duration REAL NOT NULL,
            tot_energy REAL NOT NULL,
            cpu_energy REAL NOT NULL,
            gpu_energy REAL NOT NULL,
            ram_energy REAL NOT NULL,
            FOREIGN KEY (modelrun_id) REFERENCES ModelRun(id) ON DELETE CASCADE,
            UNIQUE (modelrun_id, phase)
        );
        """)
        self.conn.commit()

    def reset_database(self):
        self.cur.execute("PRAGMA foreign_keys = OFF;")
        for t in ["Performance", "ModelRun", "Sample", "ExperimentDataset", "Experiment", "Dataset"]:
            self.cur.execute(f"DROP TABLE IF EXISTS {t};")
        self.conn.commit()
        self.cur.execute("PRAGMA foreign_keys = ON;")
        self._create_schema()
        logger.info("Database fully reset and recreated.")

    # ------------------------------------------------------------ inserts --

    def add_experiment(self, codename):
        try:
            self.cur.execute("INSERT INTO Experiment (codename) VALUES (?)", (codename,))
        except sqlite3.IntegrityError:
            logger.info(f"Experiment {codename} already exists. Skipping.")
            return 0
        self.conn.commit()
        return 1

    def add_dataset(self, codename):
        try:
            self.cur.execute("INSERT INTO Dataset (codename) VALUES (?)", (codename,))
        except sqlite3.IntegrityError:
            logger.info(f"Dataset {codename} already exists. Skipping.")
            return 0
        self.conn.commit()
        return 1

    def add_dataset_to_experiment(self, experiment_codename, dataset_codename):
        self.cur.execute(
            "INSERT OR IGNORE INTO ExperimentDataset (experiment_id, dataset_id) VALUES (?, ?)",
            (self.get_experiment_id(experiment_codename), self.get_dataset_id(dataset_codename)))
        self.conn.commit()

    def add_sample(self, dataset_codename, file_path, label, css_score=None,
                   sensitivity_tier=None, scenario=None):
        self.cur.execute(
            "INSERT INTO Sample (dataset_id, file_path, label, css_score, sensitivity_tier, scenario) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self.get_dataset_id(dataset_codename), str(file_path), int(label),
             None if css_score is None else float(css_score), sensitivity_tier, scenario))
        return self.cur.lastrowid

    def add_samples(self, dataset_codename, rows):
        """rows: iterable of (file_path, label, css_score, sensitivity_tier, scenario)."""
        dataset_id = self.get_dataset_id(dataset_codename)
        self.cur.executemany(
            "INSERT INTO Sample (dataset_id, file_path, label, css_score, sensitivity_tier, scenario) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(dataset_id, str(fp), int(lb), None if css is None else float(css), tier, sc)
             for fp, lb, css, tier, sc in rows])
        self.conn.commit()

    def add_modelrun(self, sample_id, experiment_codename, model_codename, exit_idx, scores,
                     commit=True):
        self.cur.execute(
            "INSERT INTO ModelRun (sample_id, experiment_id, model_codename, exit_idx, scores) "
            "VALUES (?, ?, ?, ?, ?)",
            (sample_id, self.get_experiment_id(experiment_codename), model_codename, exit_idx,
             json.dumps([float(s) for s in np.atleast_1d(scores)])))
        if commit:
            self.conn.commit()
        return self.cur.lastrowid

    def add_performance(self, modelrun_id, phase, duration, tot_energy=0.0, cpu_energy=0.0,
                        gpu_energy=0.0, ram_energy=0.0, commit=True):
        self.cur.execute(
            "INSERT INTO Performance (modelrun_id, phase, duration, tot_energy, cpu_energy, gpu_energy, ram_energy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (modelrun_id, phase, float(duration), float(tot_energy), float(cpu_energy),
             float(gpu_energy), float(ram_energy)))
        if commit:
            self.conn.commit()
        return self.cur.lastrowid

    def add_modelrun_with_performance(self, sample_id, experiment_codename, model_codename,
                                      exit_idx, scores, preprocessing_perf, inference_perf,
                                      commit=True):
        """*_perf: (duration, tot_energy, cpu_energy, gpu_energy, ram_energy)."""
        mr = self.add_modelrun(sample_id, experiment_codename, model_codename, exit_idx, scores,
                               commit=False)
        self.add_performance(mr, "preprocessing", *preprocessing_perf, commit=False)
        self.add_performance(mr, "inference", *inference_perf, commit=False)
        if commit:
            self.conn.commit()
        return mr

    def commit(self):
        self.conn.commit()

    # ------------------------------------------------------------ lookups --

    def get_dataset_id(self, codename):
        self.cur.execute("SELECT id FROM Dataset WHERE codename=?", (codename,))
        row = self.cur.fetchone()
        if not row:
            raise ValueError(f"Dataset '{codename}' not found.")
        return row[0]

    def get_experiment_id(self, codename):
        self.cur.execute("SELECT id FROM Experiment WHERE codename=?", (codename,))
        row = self.cur.fetchone()
        if not row:
            raise ValueError(f"Experiment '{codename}' not found.")
        return row[0]

    def get_sample_ids(self, dataset_codename):
        self.cur.execute(
            "SELECT s.id FROM Sample s JOIN Dataset d ON s.dataset_id = d.id WHERE d.codename = ? ORDER BY s.id",
            (dataset_codename,))
        return [r[0] for r in self.cur.fetchall()]

    def get_experiments_datasets(self):
        self.cur.execute("""
        SELECT d.codename, e.codename
        FROM ExperimentDataset ed
        JOIN Experiment e ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        """)
        return self.cur.fetchall()

    def get_dataset_samples(self, dataset_codename):
        """Rows of (label, file_path, css_score, sensitivity_tier, scenario) ordered by sample id."""
        self.cur.execute("""
        SELECT s.label, s.file_path, s.css_score, s.sensitivity_tier, s.scenario
        FROM Dataset d JOIN Sample s ON s.dataset_id = d.id
        WHERE d.codename = ? ORDER BY s.id
        """, (dataset_codename,))
        return self.cur.fetchall()

    def get_labels(self, dataset_codename):
        return [r[0] for r in self.get_dataset_samples(dataset_codename)]

    def get_css_scores(self, dataset_codename):
        return [r[2] for r in self.get_dataset_samples(dataset_codename)]

    def get_sensitivity_tiers(self, dataset_codename):
        return [r[3] for r in self.get_dataset_samples(dataset_codename)]

    def get_exit_indexes(self, experiment_codename, dataset_codename, model_codename="ee_vit"):
        self.cur.execute("""
        SELECT DISTINCT m.exit_idx
        FROM ExperimentDataset ed
        JOIN Experiment e ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        JOIN Sample s ON s.dataset_id = d.id
        JOIN ModelRun m ON m.sample_id = s.id AND m.experiment_id = e.id
        WHERE e.codename = ? AND d.codename = ? AND m.model_codename = ? AND m.exit_idx IS NOT NULL
        ORDER BY m.exit_idx
        """, (experiment_codename, dataset_codename, model_codename))
        return [r[0] for r in self.cur.fetchall()]

    def _exit_clause(self, exit_idx):
        return ("m.exit_idx IS NULL", ()) if exit_idx is None else ("m.exit_idx = ?", (exit_idx,))

    def get_scores(self, experiment_codename, dataset_codename, model_codename, exit_idx=None):
        """List (one entry per sample, ordered by sample id) of score lists."""
        clause, params = self._exit_clause(exit_idx)
        self.cur.execute(f"""
        SELECT m.scores
        FROM ExperimentDataset ed
        JOIN Experiment e ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        JOIN Sample s ON s.dataset_id = d.id
        JOIN ModelRun m ON m.sample_id = s.id AND m.experiment_id = e.id
        WHERE e.codename = ? AND d.codename = ? AND m.model_codename = ? AND {clause}
        ORDER BY s.id
        """, (experiment_codename, dataset_codename, model_codename, *params))
        return [json.loads(r[0]) for r in self.cur.fetchall()]

    def get_performance(self, experiment_codename, dataset_codename, model_codename, exit_idx=None):
        """{'preprocessing': [[duration, tot, cpu, gpu, ram], ...], 'inference': [...]} per sample."""
        clause, params = self._exit_clause(exit_idx)
        cols = ",\n".join(
            f"SUM(CASE WHEN p.phase = '{ph}' THEN p.{c} END)" for ph in ("preprocessing", "inference")
            for c in PERF_COLUMNS)
        self.cur.execute(f"""
        SELECT {cols}
        FROM ExperimentDataset ed
        JOIN Experiment e ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        JOIN Sample s ON s.dataset_id = d.id
        JOIN ModelRun m ON m.sample_id = s.id AND m.experiment_id = e.id
        JOIN Performance p ON p.modelrun_id = m.id
        WHERE e.codename = ? AND d.codename = ? AND m.model_codename = ? AND {clause}
        GROUP BY m.id ORDER BY s.id
        """, (experiment_codename, dataset_codename, model_codename, *params))
        rows = self.cur.fetchall()
        n = len(PERF_COLUMNS)
        return {"preprocessing": [list(r)[:n] for r in rows],
                "inference": [list(r)[n:] for r in rows]}

    def get_transferring_performance(self, experiment_codename, dataset_codename,
                                     model_codename="transfer", exit_idx=None):
        clause, params = self._exit_clause(exit_idx)
        self.cur.execute(f"""
        SELECT p.duration, p.tot_energy, p.cpu_energy, p.gpu_energy, p.ram_energy
        FROM ExperimentDataset ed
        JOIN Experiment e ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        JOIN Sample s ON s.dataset_id = d.id
        JOIN ModelRun m ON m.sample_id = s.id AND m.experiment_id = e.id
        JOIN Performance p ON p.modelrun_id = m.id
        WHERE e.codename = ? AND d.codename = ? AND m.model_codename = ? AND {clause}
        AND p.phase = 'transferring'
        ORDER BY s.id
        """, (experiment_codename, dataset_codename, model_codename, *params))
        return self.cur.fetchall()

    def close(self):
        self.conn.close()
