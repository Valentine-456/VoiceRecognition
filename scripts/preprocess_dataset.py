#!/usr/bin/env python3
"""
Pipeline: clip audio (with optional silence removal) then generate spectrograms.

This script orchestrates:
  1) scripts/clip_audio.py
  2) scripts/generate_spectrograms.py

Usage example:
  python scripts/preprocess_dataset.py \
    --input data/full_audio_files \
    --seconds 3 \
    --clips-name clips \
    -r \
    --silence-top-db 40 \
    --spec-output-name clips_specs \
    --spec-format png \
    --spec-type mel \
    --spec-sr 16000

Notes:
  - All clips are written to data/custom_dataset/audio/<clips-name>/
  - Spectrograms are written to data/custom_dataset/spectrograms/<spec-output-name>/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clip audio then generate spectrograms")

    # Clipping inputs
    p.add_argument("--input", required=True, help="Path to input audio file or directory")
    p.add_argument("--seconds", type=float, required=True, help="Clip length in seconds")
    p.add_argument("--clips-name", required=True, help="Output folder name under data/custom_dataset/audio/")
    p.add_argument("--clip-prefix", default=None, help="Optional filename prefix for clips")
    p.add_argument("-r", "--recursive", action="store_true", help="Recurse when input is a directory")
    p.add_argument("--keep-remainder", action="store_true", help="Keep final short clip")
    p.add_argument("--clip-target-sr", type=int, default=None, help="Resample target SR for clipping step")
    p.add_argument("--silence-top-db", type=float, default=30.0, help="Silence removal threshold (dB)")

    # Spectrogram options
    p.add_argument(
        "--spec-output-name",
        default=None,
        help="Output folder name under data/custom_dataset/spectrograms/. Defaults to <clips-name>.",
    )
    p.add_argument("--spec-sr", type=int, default=16000, help="SR for spectrogram loading (default: 16000)")
    p.add_argument("--spec-type", choices=["mel", "linear"], default="mel", help="Spectrogram type")
    p.add_argument("--spec-format", choices=["png", "pt", "both"], default="png", help="Output format")
    # Pass-through quality controls for spectrograms
    p.add_argument("--spec-time-pool", type=int, default=1, help="Time-axis pooling factor")
    p.add_argument("--spec-freq-pool", type=int, default=1, help="Frequency-axis pooling factor")
    p.add_argument("--spec-pool-mode", choices=["avg", "max"], default="avg", help="Pooling mode")
    p.add_argument("--spec-grayscale", action="store_true", help="Save PNGs as grayscale")
    p.add_argument("--spec-image-scale", type=float, default=1.0, help="Scale factor for PNG size")
    p.add_argument("--spec-cmap", default="magma", help="Colormap for PNGs when not grayscale")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 1) Clip step
    clip_cmd = [
        sys.executable,
        str(Path("scripts/clip_audio.py")),
        "--input", args.input,
        "--seconds", str(args.seconds),
        "--name", args.clips_name,
        "--silence-top-db", str(args.silence_top_db),
    ]
    if args.clip_prefix:
        clip_cmd += ["--prefix", args.clip_prefix]
    if args.recursive:
        clip_cmd += ["-r"]
    if args.keep_remainder:
        clip_cmd += ["--keep-remainder"]
    if args.clip_target_sr:
        clip_cmd += ["--target-sr", str(args.clip_target_sr)]

    print(f"[PIPELINE] Running clip step: {' '.join(clip_cmd)}")
    rc = subprocess.call(clip_cmd)
    if rc != 0:
        print(f"[PIPELINE][ERROR] Clip step failed with exit code {rc}", file=sys.stderr)
        return rc

    # 2) Spectrogram step
    spec_name = args.spec_output_name or args.clips_name
    clips_dir = Path("data/custom_dataset/audio") / args.clips_name
    spec_cmd = [
        sys.executable,
        str(Path("scripts/generate_spectrograms.py")),
        "--input-dir", str(clips_dir),
        "--output-name", spec_name,
        "--sr", str(args.spec_sr),
        "--type", args.spec_type,
        "--format", args.spec_format,
    ]
    # Quality controls
    spec_cmd += [
        "--time-pool", str(args.spec_time_pool),
        "--freq-pool", str(args.spec_freq_pool),
        "--pool-mode", args.spec_pool_mode,
        "--image-scale", str(args.spec_image_scale),
        "--cmap", args.spec_cmap,
    ]
    if args.spec_grayscale:
        spec_cmd += ["--grayscale"]
    print(f"[PIPELINE] Running spectrogram step: {' '.join(spec_cmd)}")
    rc = subprocess.call(spec_cmd)
    if rc != 0:
        print(f"[PIPELINE][ERROR] Spectrogram step failed with exit code {rc}", file=sys.stderr)
        return rc

    print("[PIPELINE][OK] Finished clipping and spectrogram generation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
