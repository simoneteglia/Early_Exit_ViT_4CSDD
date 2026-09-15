"""CSS deepfake dataset: loading, stratified splits with persisted manifests,
transforms and PyTorch datasets/loaders.

Dataset: https://huggingface.co/datasets/simoneteglia/css-deepfake-dataset
(58.4k images, single ``train`` split; access requires a Hugging Face token —
``huggingface-cli login`` or ``HF_TOKEN``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from human_factors import css_tier

DATASET_NAME = "simoneteglia/css-deepfake-dataset"

COL_IMAGE = "image"
COL_ID = "image_id"
COL_FILENAME = "filename"
COL_LABEL_IDX = "label_idx"
COL_LABEL = "label"
COL_CSS = "final_css_score"
COL_TIER = "sensitivity_tier"
COL_SCENARIO = "scenario"
COL_GENERATOR = "generator_model"
COL_SOURCE = "source_dataset"

LABEL_MAP = {"real": 0, "fake": 1}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BICUBIC = transforms.InterpolationMode.BICUBIC
SPLITS = ("train", "val", "test")


# ------------------------------------------------------------- loading -----

def load_css_dataset(name=DATASET_NAME, split="train", cache_dir=None, token=None):
    """Load the HF dataset and drop tampered samples (label_idx == 2).

    The model uses binary BCE loss (real=0 / fake=1); tampered images
    have label_idx=2 and must be excluded."""
    from datasets import load_dataset
    token = token or os.environ.get("HF_TOKEN")
    ds = load_dataset(name, split=split, cache_dir=cache_dir, token=token)
    n_before = len(ds)
    if COL_LABEL_IDX in ds.column_names:
        ds = ds.filter(lambda row: row[COL_LABEL_IDX] in (0, 1))
    elif COL_LABEL in ds.column_names:
        ds = ds.filter(lambda row: row[COL_LABEL].lower() in LABEL_MAP)
    n_dropped = n_before - len(ds)
    if n_dropped:
        print(f"Filtered out {n_dropped} tampered samples ({n_before} → {len(ds)}).")
    return ds


def synthetic_dataset(n: int = 256, image_size: int = 224, seed: int = 0):
    """In-memory stand-in with the same columns (random images) for smoke tests."""
    from datasets import Dataset as HFDataset, Features, Image as HFImage, Value
    rng = np.random.default_rng(seed)
    imgs = [Image.fromarray(rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8))
            for _ in range(n)]
    labels = rng.integers(0, 2, n)
    css = rng.choice([1.0, 2.0, 3.0, 4.0], n)
    tiers = css_tier(css).tolist()
    return HFDataset.from_dict({
        COL_IMAGE: imgs,
        COL_ID: [f"synthetic_{i:05d}" for i in range(n)],
        COL_FILENAME: [f"synthetic_{i:05d}.png" for i in range(n)],
        COL_LABEL: ["fake" if l else "real" for l in labels],
        COL_LABEL_IDX: labels.tolist(),
        COL_CSS: css.tolist(),
        COL_TIER: tiers,
        COL_SCENARIO: rng.choice(["Everyday", "Politics", "War & Conflict"], n).tolist(),
        COL_GENERATOR: rng.choice(["gen_a", "gen_b", "none"], n).tolist(),
    }, features=Features({
        COL_IMAGE: HFImage(), COL_ID: Value("string"), COL_FILENAME: Value("string"),
        COL_LABEL: Value("string"), COL_LABEL_IDX: Value("int64"), COL_CSS: Value("float64"),
        COL_TIER: Value("string"), COL_SCENARIO: Value("string"), COL_GENERATOR: Value("string"),
    }))


def metadata_frame(hf_ds) -> pd.DataFrame:
    """Non-image columns as a DataFrame with normalized fields
    ``sample_id``, ``label`` (0 real / 1 fake), ``css_score``, ``tier``."""
    df = hf_ds.remove_columns([COL_IMAGE]).to_pandas()
    df["row"] = np.arange(len(df))

    if COL_ID in df:
        df["sample_id"] = df[COL_ID].astype(str)
    elif COL_FILENAME in df:
        df["sample_id"] = df[COL_FILENAME].astype(str)
    else:
        df["sample_id"] = df["row"].astype(str)

    if COL_LABEL_IDX in df:
        df["label"] = df[COL_LABEL_IDX].astype(int)
    else:
        df["label"] = df[COL_LABEL].astype(str).str.lower().map(LABEL_MAP).astype(int)

    df["css_score"] = pd.to_numeric(df[COL_CSS], errors="coerce") if COL_CSS in df else np.nan
    # Tiers mirror the 1-4 CSS levels; the dataset's own tier column is only a
    # fallback for rows without a score.
    if df["css_score"].notna().any():
        df["tier"] = css_tier(df["css_score"].fillna(1.0).to_numpy())
    elif COL_TIER in df:
        df["tier"] = df[COL_TIER].astype(str).str.upper()
    else:
        df["tier"] = "ALL"
    if COL_SCENARIO not in df:
        df[COL_SCENARIO] = None
    return df


# -------------------------------------------------------------- splits -----

def make_splits(df: pd.DataFrame, fractions=(0.7, 0.15, 0.15), seed: int = 42,
                stratify_cols=("label", "tier")) -> dict[str, np.ndarray]:
    """Stratified train/val/test row indices. Strata too small to split
    (< 2 members) fall back to label-only stratification."""
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1"
    key = df[list(stratify_cols)].astype(str).agg("|".join, axis=1)
    counts = key.value_counts()
    small = key.isin(counts[counts < 3].index)
    key = key.where(~small, df["label"].astype(str))

    rows = df["row"].to_numpy()
    train_rows, rest_rows, _, rest_key = train_test_split(
        rows, key.to_numpy(), test_size=1 - fractions[0], random_state=seed, stratify=key)
    test_share = fractions[2] / (fractions[1] + fractions[2])
    val_rows, test_rows = train_test_split(
        rest_rows, test_size=test_share, random_state=seed, stratify=rest_key)
    return {"train": np.sort(train_rows), "val": np.sort(val_rows), "test": np.sort(test_rows)}


def save_manifest(path, splits: dict, df: pd.DataFrame, **meta):
    payload = {**meta, "splits": {
        name: {"rows": [int(r) for r in rows],
               "sample_ids": df["sample_id"].to_numpy()[rows].tolist()}
        for name, rows in splits.items()}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def load_manifest(path, df: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    with open(path) as f:
        payload = json.load(f)
    splits = {}
    for name, s in payload["splits"].items():
        rows = np.asarray(s["rows"], dtype=np.int64)
        if df is not None:
            ids = df["sample_id"].to_numpy()[rows]
            if list(ids) != list(s["sample_ids"]):
                raise ValueError(f"Manifest split '{name}' does not match the loaded dataset "
                                 "(sample ids differ); regenerate the manifest.")
        splits[name] = rows
    return splits


# ---------------------------------------------------------- transforms -----

def train_transform(image_size=224, random_crop=False):
    """Resize the short side and center-crop (timm's DINOv3 eval config), plus a
    horizontal flip. Augmentation is mild on purpose: heavy crops/color jitter
    can erase the generator artifacts a deepfake detector relies on."""
    crop = (transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=BICUBIC) if random_crop
            else transforms.Compose([transforms.Resize(image_size, interpolation=BICUBIC),
                                     transforms.CenterCrop(image_size)]))
    return transforms.Compose([
        crop,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size=224):
    """Resize short side to image_size (aspect ratio preserved), center crop."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ------------------------------------------------------------ datasets -----

class CSSDeepfakeDataset(Dataset):
    """Yields ``(image_tensor, label, css_score)`` for the rows of one split."""

    def __init__(self, hf_ds, df: pd.DataFrame, rows, transform):
        self.hf_ds = hf_ds
        self.rows = np.asarray(rows, dtype=np.int64)
        sub = df.iloc[self.rows]
        self.labels = sub["label"].to_numpy(dtype=np.int64)
        self.css_scores = sub["css_score"].fillna(1.0).to_numpy(dtype=np.float32)
        self.tiers = sub["tier"].to_numpy()
        self.sample_ids = sub["sample_id"].to_numpy()
        self.scenarios = sub[COL_SCENARIO].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = self.hf_ds[int(self.rows[i])][COL_IMAGE]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        x = self.transform(img.convert("RGB"))
        return x, torch.tensor(self.labels[i]), torch.tensor(self.css_scores[i])


def build_datasets(hf_ds, df, splits, image_size=224, random_crop=False):
    return {
        "train": CSSDeepfakeDataset(hf_ds, df, splits["train"], train_transform(image_size, random_crop)),
        "val": CSSDeepfakeDataset(hf_ds, df, splits["val"], eval_transform(image_size)),
        "test": CSSDeepfakeDataset(hf_ds, df, splits["test"], eval_transform(image_size)),
    }


def build_loaders(datasets, batch_size=64, num_workers=4, pin_memory=True):
    return {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"),
                         num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        for name, ds in datasets.items()
    }


def prepare(dataset_name=DATASET_NAME, manifest=None, synthetic: int = 0, seed=42,
            fractions=(0.7, 0.15, 0.15), cache_dir=None, max_per_class: int = 0):
    """Load the HF dataset (or a synthetic stand-in), the metadata frame and the
    split manifest (creating it if missing).

    ``max_per_class``: if > 0, keep at most this many samples per label
    (balanced subsample *before* splitting), useful for quick test runs.
    """
    hf_ds = synthetic_dataset(synthetic, seed=seed) if synthetic else load_css_dataset(dataset_name, cache_dir=cache_dir)

    # --- optional balanced subsample --------------------------------
    if max_per_class > 0 and not synthetic:
        rng = np.random.default_rng(seed)
        df_tmp = metadata_frame(hf_ds)
        keep = []
        for lbl in sorted(df_tmp["label"].unique()):
            rows_lbl = df_tmp.loc[df_tmp["label"] == lbl, "row"].to_numpy()
            if len(rows_lbl) > max_per_class:
                rows_lbl = rng.choice(rows_lbl, max_per_class, replace=False)
            keep.extend(rows_lbl.tolist())
        keep = sorted(keep)
        hf_ds = hf_ds.select(keep)
        print(f"Subsampled to {len(hf_ds)} images ({max_per_class} per class).")

    df = metadata_frame(hf_ds)
    if manifest is not None and Path(manifest).exists():
        splits = load_manifest(manifest, df)
    else:
        splits = make_splits(df, fractions=fractions, seed=seed)
        if manifest is not None:
            save_manifest(manifest, splits, df, dataset="synthetic" if synthetic else dataset_name,
                          seed=seed, fractions=list(fractions), stratify=["label", "tier"])
    return hf_ds, df, splits
