# scripts/generate_spectrograms.py

## What the script does
Converts every audio file in a directory into spectrograms saved as PNGs, PyTorch tensors, or both under `data/custom_dataset/spectrograms/<output_name>/`.

## Where it is used
Run it after `scripts/clip_audio.py` (directly or through `scripts/preprocess_dataset.py`) to create model-ready spectrogram inputs; it feeds the CNN defined in `src/voice_cnn.py`.

## How it works
The script loads audio with librosa at a fixed sample rate, computes either mel or linear spectrograms, optionally downsamples along time/frequency, then saves PNGs (matplotlib) and/or `.pt` tensors (torch) for each source file.

## Example command
```bash
python scripts/generate_spectrograms.py \
  --input-dir data/custom_dataset/audio/clips_v1 \
  --output-name specs_v1 \
  --sr 16000 \
  --type mel \
  --format both \
  --time-pool 2 \
  --freq-pool 2 \
  --pool-mode avg \
  --grayscale
```

## Parameters and flags
- `--input-dir DIR` (required): Directory containing audio clips.
- `--output-name TEXT` (required): Subfolder name under `data/custom_dataset/spectrograms/`.
- `--sr INT` (default `16000`): Sample rate to use when loading audio.
- `--type {mel,linear}` (default `mel`): Spectrogram variant.
- `--n-fft INT` (default `1024`): FFT window size.
- `--hop-length INT` (default `256`): Hop length between FFT windows.
- `--n-mels INT` (default `80`): Number of mel bins (mel mode only).
- `--time-pool INT` (default `1`): Downsample factor along the time axis after computation.
- `--freq-pool INT` (default `1`): Downsample factor along the frequency axis.
- `--pool-mode {avg,max}` (default `avg`): Pooling operation for downsampling.
- `--grayscale` (flag): Render PNGs in grayscale.
- `--cmap NAME` (default `magma`): Colormap for PNGs when not using grayscale.
- `--image-scale FLOAT` (default `1.0`): Scale factor for PNG resolution.
- `--mono` (flag, default): Collapse to mono during loading.
- `--stereo` (flag): Keep stereo channels.
- `--format {png,pt,both}` (default `png`): Output format selection.
- `--recursive` (flag): Recurse into subdirectories when gathering audio.
