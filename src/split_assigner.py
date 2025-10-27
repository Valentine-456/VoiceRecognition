#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Tuple


def _stable_bucket(label: str, source_id: str, seed: int) -> int:
    m = hashlib.md5()
    m.update(f"{seed}|{label}|{source_id}".encode("utf-8"))
    # 0..9999 for finer bucketing, then map to 0..99
    val = int.from_bytes(m.digest()[:2], "big") % 100
    return val


def assign_split(label: str, source_id: str, seed: int, splits: Tuple[int, int, int]) -> str:
    """Deterministically assign a split by hashing label + source_id + seed.

    Returns one of: "train", "val", "test".
    """
    train_pct, val_pct, test_pct = splits
    total = train_pct + val_pct + test_pct
    if total != 100:
        raise ValueError(f"splits must sum to 100, got {total}")
    b = _stable_bucket(label, source_id, seed)
    if b < train_pct:
        return "train"
    if b < train_pct + val_pct:
        return "val"
    return "test"


def _ensure_dirs(splits_root: Path, dataset_name: str, kind: str, split: str) -> Path:
    d = splits_root / dataset_name / kind / split
    d.mkdir(parents=True, exist_ok=True)
    return d


def next_destination(splits_root: Path, dataset_name: str, kind: str, split: str, label: str, ext: str) -> Path:
    """Compute destination path like <root>/<name>/<kind>/<split>/<label>_<split>_<idx><ext>.

    Chooses the next available idx by counting existing matches.
    """
    split_dir = _ensure_dirs(splits_root, dataset_name, kind, split)
    suffix = ext.lower().lstrip(".")
    pattern = f"{label}_{split}_*.{suffix}"
    existing = list(split_dir.glob(pattern))
    idx = len(existing)
    filename = f"{label}_{split}_{idx:03d}.{suffix}"
    return split_dir / filename


def _csv_path(splits_root: Path, dataset_name: str, kind: str, split: str) -> Path:
    # type-specific CSVs at the dataset root: e.g., spectrograms_train.csv
    return splits_root / dataset_name / f"{kind}_{split}.csv"


def append_csv(splits_root: Path, dataset_name: str, kind: str, split: str, dst_path: Path, label: str, source_path: Path) -> None:
    """Append one row to the CSV for this split. Creates header if needed."""
    csv_path = _csv_path(splits_root, dataset_name, kind, split)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["filepath", "label", "source"])
        w.writerow([dst_path.as_posix(), label, source_path.as_posix()])

