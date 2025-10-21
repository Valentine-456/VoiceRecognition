# Voice Recognition 
---
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

## 🎙️ Clip Audio Into Fixed-Length Segments

Use `scripts/clip_audio.py` to split any audio file into N‑second clips and save them under `data/custom_dataset/audio/<name>/`.

### Example (10‑second clips)

```bash
# Split an MP3 into 10-second clips named under testRunObama/
python scripts/clip_audio.py \
  --input "data/full_audio_files/ObamaSpeech.mp3" \
  --seconds 10 \
  --name testRunObama \
  --keep-remainder \
  --target-sr 16000
```

This writes clips to: `data/custom_dataset/audio/testRunObama/`

### Arguments

- `--input`: Path to the source audio (MP3/WAV supported by torchaudio).
- `--seconds`: Clip length in seconds (e.g., `3`, `10`).
- `--name`: Subfolder name under `data/custom_dataset/audio/` to save clips.
- `--keep-remainder`: Also saves the final shorter clip if the audio length is not a multiple of `--seconds`.
- `--target-sr`: Optional resample target (e.g., `16000`) to standardize sample rate (recommended for speech).

Notes:
- Keep quotes around paths with spaces.
- Output files are saved as WAV for reliability and lossless quality.

---

## 📈 Generate Spectrograms From Audio Folder

Use `scripts/generate_spectrograms.py` to turn all audio in a folder into spectrograms.

### Example (mel spectrograms)

```bash
# Generate mel spectrograms (as PNG images) from all files in data/full_audio_files
python scripts/generate_spectrograms.py \
  --input-dir data/full_audio_files \
  --output-name obamaSpectros \
  --sr 16000 \
  --type mel \
  --format png

# Save as PyTorch tensors (.pt) instead of PNGs
python scripts/generate_spectrograms.py \
  --input-dir data/full_audio_files \
  --output-name obamaSpectros_pt \
  --sr 16000 \
  --type mel \
  --format pt

# Save both PNG and .pt for each file
python scripts/generate_spectrograms.py \
  --input-dir data/full_audio_files \
  --output-name obamaSpectros_both \
  --format both
```

Outputs are saved to: `data/custom_dataset/spectrograms/obamaSpectros/` with the same base names
as the inputs (e.g., `ObamaSpeech.mp3` -> `ObamaSpeech.png`).

### Arguments
- `--input-dir`: Folder containing audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, ...).
- `--output-name`: Subfolder created under `data/custom_dataset/spectrograms/`.
- `--sr`: Target sample rate for loading (default `16000`).
- `--type`: `mel` (default) or `linear` magnitude spectrograms.
- `--format`: `png` (default), `pt` (PyTorch tensor), or `both`.
- `--recursive`: Search subfolders recursively.
- `--mono`/`--stereo`: Convert to mono (default) or keep stereo.

Notes:
- PNGs are easy to view; `.pt` files are efficient to load in PyTorch.
