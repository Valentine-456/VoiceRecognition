# scripts/augment_full_audio.py

## What the script does
Creates reversed and/or noise-augmented copies of every audio file in a directory while keeping the originals untouched.

## Where it is used
Run it before `scripts/clip_audio.py` (manually or via the `--full-*` switches in `scripts/preprocess_dataset.py`) when you want more data variety at the full-audio stage.

## How it works
It enumerates audio files (optionally recursively), skips anything that already carries the chosen suffix, builds a plan of outputs to avoid reprocessing newly created files, applies the selected augmentations with torchaudio/torch, and writes sibling files that share the source extension (falling back to WAV if needed).

## Example command
```bash
python scripts/augment_full_audio.py \
  --input-dir data/full_audio_files \
  --recursive \
  --reverse \
  --noise \
  --noise-snr-db 8 \
  --suffix Boosted
```

## Parameters and flags
- `--input-dir DIR` (required): Directory that holds the source audio files.
- `--recursive` (flag): Include files from subdirectories.
- `--reverse` (flag): Emit time-reversed versions.
- `--noise` (flag): Emit white-noise versions.
- `--noise-snr-db VALUE` (float, default `10.0`): Target SNR for `--noise` copies.
- Reverse outputs always end with `Reversed`; noise outputs end with `Noised`.
  Existing destination files are left untouched (no overwrites).
