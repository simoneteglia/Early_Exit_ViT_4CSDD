"""On-demand image fetch for a single ``image_id`` from the gated HF dataset
``simoneteglia/css-deepfake-dataset``, without downloading the full ~10 GB of
parquet shards.

The dataset has no per-file image endpoint (images are embedded in 7 parquet
shards), so a lookup works in two steps, both done with HTTP range reads via
``HfFileSystem``:

1. locate the (shard, row-group) holding this id — the *image_id* column of a
   shard is read once and the mapping is cached to ``_shard_index.json``, so
   later lookups anywhere in that shard skip this step entirely;
2. read just that one row group's *image* column (a few hundred rows) and
   decode the target row.

Fetched images are cached to disk under ``image_cache/`` keyed by id, so a
given image is downloaded from the Hub at most once.
"""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path

from PIL import Image

DATASET_REPO = "simoneteglia/css-deepfake-dataset"
N_SHARDS = 7
CACHE_DIR = Path(__file__).resolve().parent.parent / "image_cache"
INDEX_PATH = CACHE_DIR / "_shard_index.json"

_LOCK = threading.Lock()  # serialize Hub access / index read-modify-write


def _shard_repo_path(shard: int) -> str:
    return f"datasets/{DATASET_REPO}/data/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"


def _load_index() -> dict:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    return {"ids": {}, "scanned_shards": []}


def _save_index(index: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index))


def _scan_shard(fs, shard: int, index: dict) -> None:
    """Record which row group every id in this shard lives in."""
    import pyarrow.parquet as pq

    with fs.open(_shard_repo_path(shard), "rb") as f:
        pf = pq.ParquetFile(f)
        boundaries = []
        start = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            boundaries.append((start, start + n))
            start += n
        ids = pf.read(columns=["image_id"]).column("image_id").to_pylist()
    b = 0
    for row, sid in enumerate(ids):
        while row >= boundaries[b][1]:
            b += 1
        index["ids"][sid] = [shard, b]
    index["scanned_shards"].append(shard)


def fetch_image(sample_id: str, token: str | None = None) -> Path | None:
    """Return a local PNG for this sample, downloading/caching it on first
    use. Returns None if the id isn't found in the dataset."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{sample_id}.png"
    if cached.exists():
        return cached

    with _LOCK:
        if cached.exists():  # re-check: another thread may have just written it
            return cached
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq

        fs = HfFileSystem(token=token)
        index = _load_index()
        if sample_id not in index["ids"]:
            for shard in range(N_SHARDS):
                if shard in index["scanned_shards"]:
                    continue
                _scan_shard(fs, shard, index)
                _save_index(index)
                if sample_id in index["ids"]:
                    break
        if sample_id not in index["ids"]:
            return None

        shard, rg = index["ids"][sample_id]
        with fs.open(_shard_repo_path(shard), "rb") as f:
            pf = pq.ParquetFile(f)
            tbl = pf.read_row_group(rg, columns=["image", "image_id"])
        ids = tbl.column("image_id").to_pylist()
        row = ids.index(sample_id)
        img_struct = tbl.column("image")[row].as_py()
        img_bytes = img_struct["bytes"] if isinstance(img_struct, dict) else img_struct
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        im.save(cached)
        return cached
