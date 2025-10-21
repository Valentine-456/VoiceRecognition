#!/usr/bin/env python3
"""
Generate spectrogram images for all audio files in a directory.

Input: a directory containing audio files (wav, mp3, flac, ogg, m4a).
Output: spectrogram PNGs saved under data/custom_dataset/spectrograms/<output_name>/

Each output PNG is named after the corresponding audio file's stem, e.g.,
  input:  songA.mp3  -> output: songA.png

Examples:
  python scripts/generate_spectrograms.py \
    --input-dir data/full_audio_files \
    --output-name obamaSpectros \
    --sr 16000 --type mel

Dependencies: librosa, matplotlib, numpy, tqdm (already listed in requirements.txt)
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

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback if tqdm missing
    def tqdm(x: Iterable, **_: object) -> Iterable:
        return x


AUDIO_EXTS: List[str] = [
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert directory of audio files to spectrogram PNGs")
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


def save_spectrogram_png(spec_db: np.ndarray, out_path: Path) -> None:
    # Tight figure with no axes, suitable for ML ingestion
    h, w = spec_db.shape
    dpi = 100
    fig_w = max(1.0, w / dpi)
    fig_h = max(1.0, h / dpi)

    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    plt.axis("off")
    plt.imshow(spec_db, origin="lower", aspect="auto", cmap="magma")
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

    converted = 0
    for ap in tqdm(audio_files, desc="Converting"):
        try:
            y, sr = librosa.load(ap, sr=args.sr, mono=args.mono)
        except Exception as e:
            print(f"[WARN] Failed to load {ap}: {e}", file=sys.stderr)
            continue

        try:
            spec_db = compute_spectrogram(y=y, sr=sr, kind=args.type, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels)
        except Exception as e:
            print(f"[WARN] Failed to compute spectrogram for {ap}: {e}", file=sys.stderr)
            continue

        ok_for_this = True
        if args.format in ("png", "both"):
            png_path = out_base / (ap.stem + ".png")
            try:
                save_spectrogram_png(spec_db, png_path)
            except Exception as e:
                ok_for_this = False
                print(f"[WARN] Failed to save PNG {png_path}: {e}", file=sys.stderr)

        if args.format in ("pt", "both"):
            pt_path = out_base / (ap.stem + ".pt")
            try:
                save_spectrogram_pt(spec_db, pt_path)
            except Exception as e:
                ok_for_this = False
                print(f"[WARN] Failed to save PT {pt_path}: {e}", file=sys.stderr)

        if ok_for_this:
            converted += 1

    print(f"[OK] Converted {converted}/{len(audio_files)} files to spectrograms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
