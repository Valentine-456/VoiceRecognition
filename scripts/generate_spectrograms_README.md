# Generate Spectrograms (scripts/generate_spectrograms.py)

Convert a folder of audio files into spectrograms saved as PNG images, PyTorch tensors (.pt), or both. Outputs are written to `data/custom_dataset/spectrograms/<output_name>/`.

Examples
- Basic mel spectrograms (PNG)
  - `python scripts/generate_spectrograms.py --input-dir data/custom_dataset/audio/clips --output-name specs --sr 16000 --type mel --format png`
- Save PyTorch tensors instead of PNGs
  - `python scripts/generate_spectrograms.py --input-dir data/custom_dataset/audio/clips --output-name specs_pt --sr 16000 --type mel --format pt`
- Lower‑resolution PNGs for CNNs
  - `python scripts/generate_spectrograms.py --input-dir data/custom_dataset/audio/clips --output-name specs_low --sr 16000 --type mel --format png --time-pool 2 --freq-pool 2 --pool-mode avg --grayscale --image-scale 0.5`

Arguments
- Core
  - `--input-dir` (path): Folder with audio files.
  - `--output-name` (str): Folder name under `data/custom_dataset/spectrograms/`.
  - `--sr` (int, default 16000): Target sample rate when loading audio.
  - `--type` (mel | linear, default mel): Spectrogram type.
  - `--format` (png | pt | both, default png): Save as images, tensors, or both.
  - `--recursive` (flag): Search subfolders.
  - `--mono` / `--stereo` (flags; default mono): Convert to mono or keep stereo when loading.
- Resolution/quality controls
  - `--n-fft` (int, default 1024): FFT window size.
  - `--hop-length` (int, default 256): Time step between frames (larger = fewer frames).
  - `--n-mels` (int, default 80): Number of mel bands (fewer = coarser vertical detail; mel only).
  - `--time-pool` (int ≥ 1, default 1): Downsample along time after computation (1=no downsample; 2=half; 3=third).
  - `--freq-pool` (int ≥ 1, default 1): Downsample along frequency after computation.
  - `--pool-mode` (avg | max, default avg): How pooling merges blocks.
  - `--grayscale` (flag): Save PNGs in grayscale.
  - `--cmap` (str, default magma): Colormap for PNGs when not grayscale.
  - `--image-scale` (float > 0, default 1.0): Scales saved PNG size (0.5 halves width/height).

Behavior
- PNGs are rendered with axes removed and tight bounds for ML ingestion.
- `.pt` saves tensors directly and is recommended for CNN training to avoid image decoding overhead.

Tips
- To reduce size/complexity: lower `--n-mels`, increase `--hop-length`, apply pooling (`--time-pool`, `--freq-pool`), and/or use `--sr 8000`.

Next step: create dataset splits
- After generating spectrograms, create train/val/test splits using the standalone splitter:
  - `python scripts/make_splits.py --input-dir data/custom_dataset/spectrograms/<output_name> --ext .pt --label-from prefix --splits 80 10 10 --seed 42 --output-name dataset_v1`
