# scripts/make_splits.py

## What the script does
Builds stratified train/val/test directories plus CSV manifests from a folder of existing files (clips or spectrograms).

## Where it is used
Run it after generating clips or spectrograms to prepare datasets for training; `scripts/preprocess_dataset.py` can invoke it automatically when `--do-split` is set.

## How it works
The script scans for files with a given extension, infers a label from either the filename prefix or the parent directory, shuffles per label with a fixed seed, picks counts based on your split percentages, copies files into `train/`, `val/`, and `test/`, and writes CSVs recording `filepath,label,source`.

## Example command
```bash
python scripts/make_splits.py \
  --input-dir data/custom_dataset/spectrograms/specs_v1 \
  --ext .pt \
  --label-from prefix \
  --splits 80 10 10 \
  --seed 42 \
  --output-name specs_v1_split
```

## Parameters and flags
- `--input-dir DIR` (required): Folder containing the files to split.
- `--ext .EXT` (required): File extension to include (e.g., `.pt`, `.png`, `.wav`).
- `--label-from {prefix,parent}` (default `prefix`): Label inference method (`prefix` takes text before the first underscore, `parent` uses the parent directory name).
- `--splits TRAIN VAL TEST` (default `80 10 10`): Percentages for train/val/test; must total 100.
- `--seed INT` (default `42`): RNG seed for the per-label shuffle.
- `--output-name TEXT` (required): Subdirectory name under `data/custom_dataset/splits/`.
- `--recursive` (flag): Recurse into subdirectories when scanning input files.
