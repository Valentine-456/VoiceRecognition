# Voice Recognition 
---
### Other Script Docs
- Audio clipping: `scripts/clip_audio_README.md`
- Spectrogram generation: `scripts/generate_spectrograms_README.md`
- Make stratified splits: `scripts/make_splits_README.md`
The goal of the project is to prepare a machine learning module that can be hypothetically used in an automated, voice-based intercom device. Imagine that you are working in a team of several programmers and the access to your floor is restricted with doors. There is an intercom that can be used to open the door. You are implementing a machine learning module that will recognize if a given person has the permission to open the door or not.

---

### 📂 Folder Descriptions

| Folder          | Purpose                                                     |
|-----------------|-------------------------------------------------------------|
| `src/`          | Reusable code: functions, classes, model definitions        |
| `scripts/`      | Entry points / runnable scripts                             |
| `data/`         | Our dataset (audio + metadata CSV)                         |
| `notebooks/`    | Experiments, data exploration, prototyping                  |


---

## ⚡ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Valentine-456/VoiceRecognition.git
cd VoiceRecognition/
```

### 2. Set up virtual environment:

```bash
# If you're in another venv or conda env, deactivate it
deactivate  
# or `conda deactivate`
python -m venv .venv

source .venv/Scripts/activate
# OR for PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Run a script:

```bash
python scripts/preprocess_dataset.py
```

---

## 🧪 Full Pipeline: Clip → Spectrograms

Use `scripts/preprocess_dataset.py` to run both steps in sequence.

Example:
```bash
python scripts/preprocess_dataset.py \
  --input data/full_audio_files \
  --seconds 3 \
  --clips-name clips \
  -r \
  --silence-top-db 40 \
  --spec-output-name clips_specs \
  --spec-format png \
  --spec-type mel \
  --spec-sr 16000 \
  --spec-time-pool 2 --spec-freq-pool 2 --spec-pool-mode avg \
  --spec-grayscale --spec-image-scale 0.5
```

Outputs:
- Clips: `data/custom_dataset/audio/clips/`
- Spectrograms: `data/custom_dataset/spectrograms/clips_specs/`

### Preset Pipeline Examples

- Regular (baseline)
```bash
python scripts/preprocess_dataset.py \
  --input data/full_audio_files \
  --seconds 10 \
  --clips-name clips_regular \
  -r \
  --spec-output-name specs_regular \
  --spec-format png \
  --spec-type mel \
  --spec-sr 16000
```

- Low Res (moderate simplification)
```bash
python scripts/preprocess_dataset.py \
  --input data/full_audio_files \
  --seconds 10 \
  --clips-name clips_lowres \
  -r \
  --spec-output-name specs_lowres \
  --spec-format png \
  --spec-type mel \
  --spec-sr 16000 \
  --spec-time-pool 2 --spec-freq-pool 2 --spec-pool-mode avg
```

- Lower Res (stronger simplification)
```bash
python scripts/preprocess_dataset.py \
  --input data/full_audio_files \
  --seconds 10 \
  --clips-name clips_lowerres \
  -r \
  --spec-output-name specs_lowerres \
  --spec-format png \
  --spec-type mel \
  --spec-sr 8000 \
  --spec-time-pool 3 --spec-freq-pool 3 --spec-pool-mode avg
```
### Pipeline Flags Reference

- Input and traversal
  - `--input` (path): File or directory to process.
  - `-r`, `--recursive` (flag): When input is a directory, include all subfolders.

- Clipping step (scripts/clip_audio.py)
  - `--seconds` (float, > 0): Length of each clip in seconds. Example: 3, 5, 10.
  - `--clips-name` (str): Output folder name under `data/custom_dataset/audio/`.
  - `--clip-prefix` (str, optional): Prefix for saved clip filenames.
  - `--keep-remainder` (flag): Also save the final short clip if leftover audio exists.
  - `--clip-target-sr` (int): Resample audio before clipping. Common values: 16000 (default choice), 8000.
  - `--silence-top-db` (float): Silence removal aggressiveness; higher removes more. Typical 20–60; default 30.

- Spectrogram step (scripts/generate_spectrograms.py)
  - `--spec-output-name` (str): Folder name under `data/custom_dataset/spectrograms/`.
  - `--spec-format` (png | pt | both): How to save spectrograms. Use `pt` for CNN training; `png` for viewing.
  - `--spec-type` (mel | linear): Mel is default for speech; linear is raw magnitude.
  - `--spec-sr` (int): Sample rate used when loading clips for spectrograms. Common: 16000 (default), 8000 (more compact).
  - `--spec-time-pool` (int ≥ 1): Downsample along time; 1=no downsample, 2=halve, 3=third.
  - `--spec-freq-pool` (int ≥ 1): Downsample along frequency; 1=no downsample, 2=halve, 3=third.
  - `--spec-pool-mode` (avg | max): How pooled blocks are merged; avg = smoother, max = sharper.
  - `--spec-grayscale` (flag): Save PNGs as grayscale. Has no effect if `--spec-format pt`.
  - `--spec-image-scale` (float > 0): Scale saved PNG size (e.g., 0.5 halves width/height). Has no effect for `.pt`.
  - `--spec-cmap` (str): Matplotlib colormap for PNGs when not grayscale (e.g., magma, viridis). No effect for `.pt`.

---
### Dataset Splits
- Recommended: Save-time split (writes directly to train/val/test while creating files).
```bash
python scripts/preprocess_dataset.py \
  --input data/full_audio_files \
  --seconds 10 \
  --clips-name clips_run \
  -r \
  --spec-output-name specs_run \
  --spec-format pt \
  --save-split specs \
  --save-split-name dataset_v1 \
  --save-split-label-from prefix \
  --save-split-sizes 80 10 10 \
  --save-split-seed 42
```
- Outputs: `data/custom_dataset/splits/dataset_v1/spectrograms/{train,val,test}/` and CSVs `spectrograms_{split}.csv`.
- Alternative: If you already have a flat folder, use the standalone splitter. See `scripts/make_splits_README.md`.
