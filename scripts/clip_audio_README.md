# scripts/clip_audio.py

## What the script does
Cuts a single audio file or an entire directory of files into fixed-length WAV clips while optionally trimming silence beforehand.

## Where it is used
Use it whenever you need raw clips under `data/custom_dataset/audio/<name>`; it is also invoked automatically inside `scripts/preprocess_dataset.py` after any optional augmentation.

## How it works
The script loads each audio file with torchaudio, resamples if requested, removes silence via `librosa.effects.split`, then slices the waveform into contiguous N-second windows and saves numbered WAV files with deterministic prefixes.

## Example command
```bash
python scripts/clip_audio.py \
  --input data/full_audio_files \
  --seconds 3 \
  --name clips_v1 \
  --silence-top-db 40 \
  --target-sr 16000 \
  --prefix-from-parent \
  -r
```

## Parameters and flags
- `--input PATH` (required): Audio file or directory to process.
- `--seconds FLOAT` (required): Clip duration in seconds; must be positive.
- `--name TEXT` (required): Output folder inside `data/custom_dataset/audio/`.
- `--prefix TEXT`: Filename prefix; defaults to the `--name` (single-file mode) or source stem (directory mode).
- `--prefix-from-parent` (flag): Prepend the source file’s parent folder name to each saved file for label retention.
- `-r`, `--recursive` (flag): When the input is a directory, search through subdirectories.
- `--target-sr INT`: Resample audio before splitting.
- `--silence-top-db FLOAT` (default `30.0`): Threshold for removing silence with librosa.
