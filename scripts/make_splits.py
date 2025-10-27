#!/usr/bin/env python3
"""
Create stratified train/val/test splits from a folder of files (clips or spectrograms).

Features
- Stratified by label
- Copies files into train/val/test directories with clear names like label_split_000.ext
- Writes CSVs listing file path and label

Label parsing
- prefix: label is the part of the filename stem before the first underscore
  e.g., ObamaSpeech_0010.pt -> label=ObamaSpeech
- parent: label is the parent directory name

Outputs
- data/custom_dataset/splits/<output_name>/
  - train/, val/, test/
  - train.csv, val.csv, test.csv (columns: filepath,label,source)
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create stratified train/val/test splits for files")
    p.add_argument("--input-dir", required=True, help="Folder containing files to split")
    p.add_argument("--ext", required=True, help="File extension to include (e.g., .pt, .png, .wav)")
    p.add_argument(
        "--label-from",
        choices=["prefix", "parent"],
        default="prefix",
        help="How to derive labels: from filename prefix or parent folder name",
    )
    p.add_argument(
        "--splits",
        nargs=3,
        type=int,
        default=[80, 10, 10],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split percentages that sum to 100 (default: 80 10 10)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    p.add_argument("--output-name", required=True, help="Name under data/custom_dataset/splits/")
    p.add_argument("--recursive", action="store_true", help="Search input dir recursively")
    p.add_argument(
        "--link",
        action="store_true",
        help="Create symlinks instead of copying files into split directories",
    )
    return p.parse_args()


def find_files(root: Path, ext: str, recursive: bool) -> List[Path]:
    ext = ext.lower()
    files: List[Path] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() == ext:
                files.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() == ext:
                files.append(p)
    files.sort()
    return files


def get_label(p: Path, label_from: str) -> str:
    if label_from == "parent":
        return p.parent.name
    # prefix from stem before first underscore
    stem = p.stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def stratified_split(items: List[Tuple[Path, str]], splits: Tuple[int, int, int], seed: int) -> Dict[str, List[Tuple[Path, str]]]:
    train_pct, val_pct, test_pct = splits
    total_pct = train_pct + val_pct + test_pct
    if total_pct != 100:
        raise ValueError(f"splits must sum to 100, got {total_pct}")

    # group by label
    by_label: Dict[str, List[Path]] = {}
    for p, lbl in items:
        by_label.setdefault(lbl, []).append(p)

    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}

    for lbl, paths in by_label.items():
        rng.shuffle(paths)
        n = len(paths)
        n_train = round(n * train_pct / 100)
        n_val = round(n * val_pct / 100)
        # ensure total counts match n (account for rounding)
        n_test = n - n_train - n_val

        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

        out["train"].extend((p, lbl) for p in train_paths)
        out["val"].extend((p, lbl) for p in val_paths)
        out["test"].extend((p, lbl) for p in test_paths)

    return out


def write_csv(csv_path: Path, rows: List[Tuple[str, str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label", "source"])  # headers
        w.writerows(rows)


def main() -> int:
    args = parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERROR] Input directory not found: {in_dir}")
        return 2

    files = find_files(in_dir, args.ext, args.recursive)
    if not files:
        print(f"[WARN] No files with extension {args.ext} found in {in_dir}")
        return 0

    items: List[Tuple[Path, str]] = [(p, get_label(p, args.label_from)) for p in files]

    splits = tuple(args.splits)  # type: ignore[assignment]
    split_map = stratified_split(items, splits, args.seed)

    out_root = Path("data/custom_dataset/splits") / args.output_name
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "test").mkdir(parents=True, exist_ok=True)

    csv_rows: Dict[str, List[Tuple[str, str, str]]] = {"train": [], "val": [], "test": []}

    # Maintain per-label counters for naming
    per_label_counts: Dict[Tuple[str, str], int] = {}

    for split_name in ("train", "val", "test"):
        for src_path, label in split_map[split_name]:
            idx = per_label_counts.get((label, split_name), 0)
            per_label_counts[(label, split_name)] = idx + 1

            new_name = f"{label}_{split_name}_{idx:03d}{src_path.suffix.lower()}"
            dst_path = out_root / split_name / new_name

            if args.link:
                try:
                    # Remove existing broken link if any, then symlink
                    if dst_path.exists() or dst_path.is_symlink():
                        dst_path.unlink()
                    dst_path.symlink_to(src_path.resolve())
                except Exception as e:
                    print(f"[WARN] Failed to symlink {src_path} -> {dst_path}: {e}. Falling back to copy.")
                    shutil.copy2(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

            rel_dst = dst_path.as_posix()
            csv_rows[split_name].append((rel_dst, label, src_path.as_posix()))

    # Write CSVs
    write_csv(out_root / "train.csv", csv_rows["train"]) 
    write_csv(out_root / "val.csv", csv_rows["val"]) 
    write_csv(out_root / "test.csv", csv_rows["test"]) 

    total = sum(len(v) for v in csv_rows.values())
    print(f"[OK] Wrote {total} items under {out_root} (train/val/test) with CSVs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

