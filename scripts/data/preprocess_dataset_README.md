# scripts/preprocess_dataset.py

## What the script does
Runs the full preprocessing pipeline: optional full-audio augmentation, clipping with silence removal, spectrogram generation, and (optionally) stratified dataset splits.

## Where it is used
Use it when you want a single command to go from raw recordings to model-ready spectrograms and splits; it chains together `augment_full_audio.py`, `clip_audio.py`, `generate_spectrograms.py`, and `make_splits.py`.

## How it works
The script builds subprocess commands for each stage, executes them in order, and stops immediately if any stage fails. Augmentations happen in-place on the original directory, clips land under `data/custom_dataset/audio/<clips-name>`, spectrograms go to `data/custom_dataset/spectrograms/<spec-output-name>`, and splits (if requested) are created from either the clips or spectrograms.

## Example command
```bash
python scripts/preprocess_dataset.py \
  --input data/full_audio_files \
  --seconds 3 \
  --clips-name clips_v2 \
  --clip-prefix-from-parent \
  --silence-top-db 40 \
  --spec-output-name specs_v2 \
  --spec-format both \
  --spec-type mel \
  --spec-sr 16000 \
  --spec-time-pool 2 \
  --spec-freq-pool 2 \
  --do-split \
  --split-target specs \
  --split-ext .pt \
  --split-output-name specs_v2_split
```

## Parameters and flags
**Shared/clip stage**
- `--input PATH` (required): Raw audio file or directory processed by the clip stage.
- `--seconds FLOAT` (required): Clip duration for `clip_audio.py`.
- `--clips-name TEXT` (required): Destination folder under `data/custom_dataset/audio/`.
- `--clip-prefix TEXT`: Prefix for saved clip filenames.
- `--clip-prefix-from-parent` (flag): Include the source file’s parent directory in each clip prefix.
- `-r`, `--recursive` (flag): Recurse through subdirectories for both augmentation and clipping steps.
- `--clip-target-sr INT`: Resample audio before clipping.
- `--silence-top-db FLOAT` (default `30.0`): Silence removal threshold handed to `clip_audio.py`.

**Full-audio augmentation options**
- `--full-reverse-audio` (flag): Run `augment_full_audio.py` with `--reverse` before clipping.
- `--full-noise-audio` (flag): Run `augment_full_audio.py` with `--noise`.
- `--full-noise-snr-db FLOAT` (default `10.0`): Target SNR for the noise augmentation.

**Spectrogram stage**
- `--spec-output-name TEXT`: Folder under `data/custom_dataset/spectrograms/`; defaults to `--clips-name`.
- `--spec-sr INT` (default `16000`): Sample rate for spectrogram loading.
- `--spec-type {mel,linear}` (default `mel`): Spectrogram type.
- `--spec-format {png,pt,both}` (default `png`): Saved output formats.
- `--spec-time-pool INT` (default `1`): Downsample factor along time.
- `--spec-freq-pool INT` (default `1`): Downsample factor along frequency.
- `--spec-pool-mode {avg,max}` (default `avg`): Pooling operation.
- `--spec-grayscale` (flag): Render PNGs in grayscale.
- `--spec-image-scale FLOAT` (default `1.0`): Scale factor for PNG resolution.
- `--spec-cmap NAME` (default `magma`): Colormap for PNGs when not grayscale.

**Split stage**
- `--do-split` (flag): Run `make_splits.py` after spectrogram creation.
- `--split-target {specs,clips}` (default `specs`): Choose which artifacts to split.
- `--split-ext EXT`: File extension to include when splitting; defaults to `.pt` or `.png` for spectrograms and `.wav` for clips.
- `--split-output-name TEXT`: Folder under `data/custom_dataset/splits/`; defaults to the spectrogram/clip name.
- `--split-label-from {prefix,parent}` (default `prefix`): Label source for the splitter.
- `--split-sizes TRAIN VAL TEST` (default `80 10 10`): Percentages for train/val/test.
- `--split-seed INT` (default `42`): RNG seed for the splitter.
- `--split-recursive` (flag): Allow the splitter to search subdirectories.
