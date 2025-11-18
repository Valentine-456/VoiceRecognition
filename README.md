# Voice Recognition
### Folder Descriptions

| Folder          | Purpose                                                     |
|-----------------|-------------------------------------------------------------|
| `src/`          | Reusable code: functions, classes, model definitions        |
| `scripts/`      | Entry points / runnable scripts                             |
| `data/`         | Our dataset (audio + metadata CSV)                         |
| `notebooks/`    | Experiments, data exploration, prototyping                  |


---

## Getting Started

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

---

## Full Pipeline: Audio files → Spectrogram's

Use `scripts/preprocess_dataset.py` conver your voice data into spectrogram sata set.

### Preset Pipeline Examples

- Do it all single command for converting audio files to split spectrogram. To make it work.
- Create dir `/full_audio_files/accept & reject` add accepted audios to `accept` sub dir and rejected audios to `reject` sun dir 
```bash
python scripts/preprocess_dataset.py \
 --input data/full_audio_files \
 --seconds 3 \
 --clips-name clips_lowerres -r \
 --clip-prefix-from-parent \
 --full-noise-audio \
 --full-noise-snr-db 10 \
 --spec-output-name spectograms \
 --spec-format png --spec-type mel --spec-sr 8000 --spec-time-pool 3 \
 --spec-freq-pool 3 --spec-pool-mode avg \
 --do-split --split-target specs --split-ext .png \
 --split-output-name v1 --split-label-from prefix --split-sizes 80 10 10 --split-seed 30
```
### Pipeline Flags (short)

- Input
  - `--input`: file or folder to process
  - `-r` / `--recursive`: include subfolders

- Full-audio augment (pre-clip)
  - `--full-reverse-audio`: write `...Reversed.*` next to originals
  - `--full-noise-audio`: write `...Noised.*` next to originals
  - `--full-noise-snr-db`: noise SNR in dB (default 10)

- Clip (scripts/clip_audio.py)
  - `--seconds`: clip length
  - `--clips-name`: output folder under `data/custom_dataset/audio/`
- `--clip-prefix`: filename prefix (optional)
- `--clip-prefix-from-parent`: use parent folder (accept/reject) as prefix
- `--clip-target-sr`: resample before clipping (e.g., 16000)
- `--silence-top-db`: silence removal threshold (dB, default 30)

- Spectrograms (scripts/generate_spectrograms.py)
  - `--spec-output-name`: folder under `data/custom_dataset/spectrograms/`
  - `--spec-format`: `png` | `pt` | `both`
  - `--spec-type`: `mel` | `linear`
  - `--spec-sr`: sample rate to load audio
  - `--spec-time-pool` / `--spec-freq-pool`: downsample factors
  - `--spec-pool-mode`: `avg` | `max`
  - `--spec-grayscale`: PNGs in grayscale
  - `--spec-image-scale`: PNG size scale
  - `--spec-cmap`: PNG colormap

- Split (post-hoc)
  - `--do-split`: run splitter after spectrograms
  - `--split-target`: `specs` | `clips`
  - `--split-ext`: extension to include (e.g., `.png`, `.pt`, `.wav`)
  - `--split-output-name`: dataset name under `data/custom_dataset/splits/`
  - `--split-label-from`: `prefix` | `parent`
  - `--split-sizes`: `TRAIN VAL TEST` (sum to 100)
  - `--split-seed`: RNG seed
  - `--split-recursive`: search subfolders for splitting

See script READMEs under `scripts/` for running each step independently.
