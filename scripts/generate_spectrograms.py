#!/usr/bin/env python3
"""
Generate spectrograms for a directory of audio files.

Input: a directory containing audio files (wav, mp3, flac, ogg, m4a).
Output: spectrograms under `data/custom_dataset/spectrograms/<output_name>/` as PNG, PT, or both.

Each output file is named after the input audio's stem, e.g.,
  input:  songA.mp3  -> output: songA.png

Examples:
  python scripts/generate_spectrograms.py \
    --input-dir data/full_audio_files \
    --output-name obamaSpectros \
    --sr 16000 --type mel

Note: Splitting into train/val/test is done afterward using
`scripts/make_splits.py` (or via the pipeline's `--do-split`).

Dependencies: librosa, matplotlib, numpy, tqdm
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List

import numpy as np
import librosa
import librosa.display  # noqa: F401  # needed for spec plotting side-effects
import matplotlib

# Use a non-interactive backend for safe file saving
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# (No split-at-save; splitting is done post‑hoc by scripts/make_splits.py.)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback if tqdm missing
    def tqdm(x: Iterable, **_: object) -> Iterable:
        return x


AUDIO_EXTS: List[str] = [
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert directory of audio files to spectrograms (PNG/PT)")
    p.add_argument("--input-dir", required=True, help="Directory containing audio files")
    p.add_argument(
        "--output-name",
        required=True,
        help="Subfolder name under data/custom_dataset/spectrograms/ for outputs",
    )
    p.add_argument("--sr", type=int, default=16000, help="Target sample rate for loading (default: 16000)")
    p.add_argument(
        "--type",
        choices=["mel", "linear"],
        default="mel",
        help="Spectrogram type: mel or linear magnitude (default: mel)",
    )
    p.add_argument("--n-fft", type=int, default=1024, help="FFT window size (default: 1024)")
    p.add_argument("--hop-length", type=int, default=256, help="Hop length (default: 256)")
    p.add_argument("--n-mels", type=int, default=80, help="Number of mel bands (mel only; default: 80)")
    # Resolution/quality controls
    p.add_argument("--time-pool", type=int, default=1, help="Downsample factor along time axis via pooling (default: 1)")
    p.add_argument("--freq-pool", type=int, default=1, help="Downsample factor along frequency axis via pooling (default: 1)")
    p.add_argument("--pool-mode", choices=["avg", "max"], default="avg", help="Pooling type for downsampling (default: avg)")
    p.add_argument("--grayscale", action="store_true", help="Save PNGs in grayscale instead of colormap")
    p.add_argument("--cmap", default="magma", help="Matplotlib colormap for PNGs when not grayscale (default: magma)")
    p.add_argument("--image-scale", type=float, default=1.0, help="Scale factor for saved PNG resolution (e.g., 0.5 halves width/height)")
    mono_group = p.add_mutually_exclusive_group()
    mono_group.add_argument("--mono", dest="mono", action="store_true", help="Convert to mono (default)")
    mono_group.add_argument("--stereo", dest="mono", action="store_false", help="Keep stereo if present")
    p.set_defaults(mono=True)
    p.add_argument(
        "--format",
        choices=["png", "pt", "both"],
        default="png",
        help="Output format for spectrograms: 'png' image, 'pt' PyTorch tensor, or 'both' (default: png)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="If set, search input directory recursively for audio files",
    )
    return p.parse_args()


def find_audio_files(root: Path, recursive: bool) -> List[Path]:
    files: List[Path] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(p)
    files.sort()
    return files


def save_spectrogram_png(spec_db: np.ndarray, out_path: Path, *, grayscale: bool, cmap: str, image_scale: float) -> None:
    # Tight figure with no axes, suitable for ML ingestion
    h, w = spec_db.shape
    dpi = 100
    scale = max(1e-3, float(image_scale))
    fig_w = max(1.0, (w / dpi) * scale)
    fig_h = max(1.0, (h / dpi) * scale)

    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    plt.axis("off")
    chosen_cmap = "gray" if grayscale else cmap
    plt.imshow(spec_db, origin="lower", aspect="auto", cmap=chosen_cmap)
    plt.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_spectrogram_pt(spec_db: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer zero‑copy path, but fall back if PyTorch's NumPy bridge is unavailable
    arr32 = np.asarray(spec_db, dtype=np.float32)
    try:
        tensor = torch.from_numpy(arr32)
    except Exception:
        # Fallback that avoids NumPy bridge (slower, but robust)
        tensor = torch.tensor(arr32.tolist(), dtype=torch.float32)
    torch.save(tensor, out_path)


def compute_spectrogram(y: np.ndarray, sr: int, kind: str, n_fft: int, hop_length: int, n_mels: int) -> np.ndarray:
    if kind == "mel":
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    else:
        S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        return S_db

def _pool_axis(arr: np.ndarray, axis: int, k: int, mode: str) -> np.ndarray:
    if k <= 1:
        return arr
    if arr.shape[axis] < k:
        return arr
    trim = arr.shape[axis] - (arr.shape[axis] % k)
    if trim != arr.shape[axis]:
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(0, trim)
        arr = arr[tuple(slicer)]
    new_shape = list(arr.shape)
    new_shape[axis] = new_shape[axis] // k
    new_shape.insert(axis + 1, k)
    arr = arr.reshape(new_shape)
    if mode == "max":
        pooled = arr.max(axis=axis + 1)
    else:
        pooled = arr.mean(axis=axis + 1)
    return pooled

def downsample_spec(spec_db: np.ndarray, time_pool: int, freq_pool: int, mode: str) -> np.ndarray:
    # spec_db shape: (freq, time)
    out = _pool_axis(spec_db, axis=1, k=int(time_pool), mode=mode)
    out = _pool_axis(out, axis=0, k=int(freq_pool), mode=mode)
    return out


def main() -> int:
    args = parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERROR] Input directory not found: {in_dir}", file=sys.stderr)
        return 2

    out_base = Path("data/custom_dataset/spectrograms") / args.output_name
    out_base.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(in_dir, args.recursive)
    if not audio_files:
        print(f"[WARN] No audio files found in {in_dir}")
        return 0

    print(f"[INFO] Found {len(audio_files)} audio files in {in_dir}")
    print(f"[INFO] Writing spectrograms to {out_base}")
    # Splitting is not handled here; run scripts/make_splits.py after generation.

    converted = 0
    for ap in tqdm(audio_files, desc="Converting"):
        try:
            y, sr = librosa.load(ap, sr=args.sr, mono=args.mono)
        except Exception as e:
            print(f"[WARN] Failed to load {ap}: {e}", file=sys.stderr)
            continue

        try:
            spec_db = compute_spectrogram(y=y, sr=sr, kind=args.type, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels)
            # Optional downsampling/pooling to reduce resolution
            if args.time_pool > 1 or args.freq_pool > 1:
                spec_db = downsample_spec(spec_db, time_pool=args.time_pool, freq_pool=args.freq_pool, mode=args.pool_mode)
        except Exception as e:
            print(f"[WARN] Failed to compute spectrogram for {ap}: {e}", file=sys.stderr)
            continue

        ok_for_this = True
        if args.format in ("png", "both"):
            flat_png = out_base / (ap.stem + ".png")
            try:
                save_spectrogram_png(spec_db, flat_png, grayscale=args.grayscale, cmap=args.cmap, image_scale=args.image_scale)
            except Exception as e:
                ok_for_this = False
                print(f"[WARN] Failed to save PNG {flat_png}: {e}", file=sys.stderr)

        if args.format in ("pt", "both"):
            flat_pt = out_base / (ap.stem + ".pt")
            try:
                save_spectrogram_pt(spec_db, flat_pt)
            except Exception as e:
                ok_for_this = False
                print(f"[WARN] Failed to save PT {flat_pt}: {e}", file=sys.stderr)

        if ok_for_this:
            converted += 1

    print(f"[OK] Converted {converted}/{len(audio_files)} files to spectrograms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
