# Make Splits (scripts/make_splits.py)

Create stratified train/val/test splits from a folder of files (clips or spectrograms).

What it does
- Finds files by extension (e.g., `.pt`, `.png`, `.wav`).
- Derives a class label per file (`prefix` or `parent`).
- Stratifies by label into train/val/test with your percentages.
- Copies (or symlinks) files into split folders and writes CSVs.
- Filenames in splits: `<label>_<split>_<idx><ext>` (e.g., `ObamaSpeech_train_000.pt`).

Examples
- Split spectrogram tensors (.pt):
  - `python scripts/make_splits.py --input-dir data/custom_dataset/spectrograms/specs_regular --ext .pt --label-from prefix --splits 80 10 10 --seed 42 --output-name specs_regular_split`
- Split PNG spectrograms:
  - `python scripts/make_splits.py --input-dir data/custom_dataset/spectrograms/specs_regular --ext .png --label-from prefix --splits 80 10 10 --seed 42 --output-name specs_regular_png_split`
- Split clips (.wav):
  - `python scripts/make_splits.py --input-dir data/custom_dataset/audio/clips_regular --ext .wav --label-from prefix --splits 80 10 10 --seed 42 --output-name clips_regular_split`

Arguments
- `--input-dir` (path): Folder to scan.
- `--ext` (e.g., .pt, .png, .wav): File extension to include.
- `--label-from` (prefix | parent):
  - `prefix` (default): label is filename stem up to first underscore (e.g., `ObamaSpeech_0001.pt` → `ObamaSpeech`).
  - `parent`: label is the parent directory name.
- `--splits` (three ints; default `80 10 10`): Percentages for train, val, test; must sum to 100.
- `--seed` (int; default 42): RNG seed for shuffling per label.
- `--output-name` (str): Name under `data/custom_dataset/splits/`.
- `--recursive` (flag): Search subfolders.
- `--link` (flag): Use symlinks instead of copying files.

Outputs
- `data/custom_dataset/splits/<output_name>/`:
  - Folders: `train/`, `val/`, `test/` (each contains renamed files).
  - CSVs: `train.csv`, `val.csv`, `test.csv` with columns `filepath,label,source`.

Notes
- Use this to retrofit older flat datasets. For new data, consider using the split‑at‑save options in `clip_audio.py` and `generate_spectrograms.py` to skip the copy step.
