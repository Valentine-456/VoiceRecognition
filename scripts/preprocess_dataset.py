#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clip audio then generate spectrograms")

    p.add_argument("--input", required=True, help="Path to input audio file or directory")
    p.add_argument("--seconds", type=float, required=True, help="Clip length in seconds")
    p.add_argument("--clips-name", required=True, help="Output folder name under data/custom_dataset/audio/")
    p.add_argument("--clip-prefix", default=None, help="Optional filename prefix for clips")
    p.add_argument("--clip-prefix-from-parent", action="store_true", help="Use source file's parent folder as per-file prefix (e.g., accept/reject)")
    p.add_argument("-r", "--recursive", action="store_true", help="Recurse when input is a directory")
    p.add_argument("--clip-target-sr", type=int, default=None, help="Resample target SR for clipping step")
    p.add_argument("--silence-top-db", type=float, default=30.0, help="Silence removal threshold (dB)")
    p.add_argument("--full-reverse-audio", action="store_true", help="Before clipping, create reversed copies of full audio in-place")
    p.add_argument("--full-noise-audio", action="store_true", help="Before clipping, create noise-augmented copies of full audio in-place")
    p.add_argument("--full-noise-snr-db", type=float, default=10.0, help="SNR in dB for noise augmentation (default 10.0)")

    p.add_argument(
        "--spec-output-name",
        default=None,
        help="Output folder name under data/custom_dataset/spectrograms/. Defaults to <clips-name>.",
    )
    p.add_argument("--spec-sr", type=int, default=16000, help="SR for spectrogram loading (default: 16000)")
    p.add_argument("--spec-type", choices=["mel", "linear"], default="mel", help="Spectrogram type")
    p.add_argument("--spec-format", choices=["png", "pt", "both"], default="png", help="Output format")
    p.add_argument("--spec-time-pool", type=int, default=1, help="Time-axis pooling factor")
    p.add_argument("--spec-freq-pool", type=int, default=1, help="Frequency-axis pooling factor")
    p.add_argument("--spec-pool-mode", choices=["avg", "max"], default="avg", help="Pooling mode")
    p.add_argument("--spec-grayscale", action="store_true", help="Save PNGs as grayscale")
    p.add_argument("--spec-image-scale", type=float, default=1.0, help="Scale factor for PNG size")
    p.add_argument("--spec-cmap", default="magma", help="Colormap for PNGs when not grayscale")

    p.add_argument("--do-split", action="store_true", help="Create stratified train/val/test splits after spectrograms")
    p.add_argument("--split-target", choices=["specs", "clips"], default="specs", help="Which outputs to split (default: specs)")
    p.add_argument("--split-ext", default=None, help="Extension to include for split (.pt, .png, .wav). Defaults based on target")
    p.add_argument("--split-output-name", default=None, help="Name under data/custom_dataset/splits/ (defaults to spec/clips name)")
    p.add_argument("--split-label-from", choices=["prefix", "parent"], default="prefix", help="How to derive labels for split")
    p.add_argument("--split-sizes", nargs=3, type=int, default=[80, 10, 10], metavar=("TRAIN", "VAL", "TEST"), help="Split percentages that sum to 100")
    p.add_argument("--split-seed", type=int, default=42, help="Random seed for split")
    p.add_argument("--split-recursive", action="store_true", help="Search input dir recursively for split")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.full_reverse_audio or args.full_noise_audio:
        aug_cmd = [
            sys.executable,
            str(Path("scripts/augment_full_audio.py")),
            "--input-dir", args.input,
        ]
        if args.recursive:
            aug_cmd += ["--recursive"]
        if args.full_reverse_audio:
            aug_cmd += ["--reverse"]
        if args.full_noise_audio:
            aug_cmd += ["--noise", "--noise-snr-db", str(args.full_noise_snr_db)]
        print(f"[PIPELINE] Running full-audio augmentation: {' '.join(aug_cmd)}")
        rc = subprocess.call(aug_cmd)
        if rc != 0:
            print(f"[PIPELINE][ERROR] Augmentation step failed with exit code {rc}", file=sys.stderr)
            return rc

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
    if args.clip_prefix_from_parent:
        clip_cmd += ["--prefix-from-parent"]
    if args.recursive:
        clip_cmd += ["-r"]
    if args.clip_target_sr:
        clip_cmd += ["--target-sr", str(args.clip_target_sr)]
    print(f"[PIPELINE] Running clip step: {' '.join(clip_cmd)}")
    rc = subprocess.call(clip_cmd)
    if rc != 0:
        print(f"[PIPELINE][ERROR] Clip step failed with exit code {rc}", file=sys.stderr)
        return rc

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
    if not args.do_split:
        return 0

    if args.split_target == "specs":
        split_in_dir = Path("data/custom_dataset/spectrograms") / (args.spec_output_name or args.clips_name)
        default_ext = ".pt" if args.spec_format in ("pt", "both") else ".png"
    else:
        split_in_dir = Path("data/custom_dataset/audio") / args.clips_name
        default_ext = ".wav"
    split_ext = args.split_ext or default_ext

    split_out_name = args.split_output_name or (args.spec_output_name or args.clips_name)

    split_cmd = [
        sys.executable,
        str(Path("scripts/make_splits.py")),
        "--input-dir", str(split_in_dir),
        "--ext", str(split_ext),
        "--label-from", args.split_label_from,
        "--splits", str(args.split_sizes[0]), str(args.split_sizes[1]), str(args.split_sizes[2]),
        "--seed", str(args.split_seed),
        "--output-name", split_out_name,
    ]
    if args.split_recursive:
        split_cmd += ["--recursive"]

    print(f"[PIPELINE] Running split step: {' '.join(split_cmd)}")
    rc = subprocess.call(split_cmd)
    if rc != 0:
        print(f"[PIPELINE][ERROR] Split step failed with exit code {rc}", file=sys.stderr)
        return rc

    print("[PIPELINE][OK] Finished splits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
