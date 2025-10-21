#!/usr/bin/env python3
"""
Split an input audio file into fixed-length clips and save them
into a new directory under `data/custom_dataset/audio/`.

Example:
  python scripts/clip_audio.py \
    --input path/to/file.wav \
    --seconds 3 \
    --name my_speaker

This creates `data/custom_dataset/audio/my_speaker/` and writes
`my_speaker_0000.wav`, `my_speaker_0001.wav`, ... of 3 seconds each.
"""

import argparse
from pathlib import Path
import sys

import torchaudio


def positive_float(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}")
    if v <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut an audio file into N-second clips and save under data/custom_dataset/audio/<name>"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input audio file (any format supported by torchaudio)",
    )
    parser.add_argument(
        "--seconds",
        type=positive_float,
        required=True,
        help="Clip length in seconds (e.g., 3)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output subfolder name under data/custom_dataset/audio. Defaults to input file stem.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix for saved clips. Defaults to <name>.",
    )
    parser.add_argument(
        "--keep-remainder",
        action="store_true",
        help="If set, also saves a final clip shorter than --seconds (if any).",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=None,
        help="Optional target sample rate to resample input before splitting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 2

    # Load audio (Tensor shape: [channels, frames])
    try:
        waveform, sample_rate = torchaudio.load(str(input_path))
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}", file=sys.stderr)
        return 3

    # Optional resample
    if args.target_sr and args.target_sr > 0 and args.target_sr != sample_rate:
        try:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=args.target_sr)
            waveform = resampler(waveform)
            sample_rate = args.target_sr
        except Exception as e:
            print(f"[ERROR] Failed to resample audio: {e}", file=sys.stderr)
            return 4

    # Compute segmenting parameters
    clip_len_samples = int(round(args.seconds * sample_rate))
    if clip_len_samples <= 0:
        print("[ERROR] Computed non-positive clip length in samples.", file=sys.stderr)
        return 5

    total_samples = waveform.shape[1]
    if total_samples == 0:
        print("[ERROR] Input audio appears empty (0 samples).", file=sys.stderr)
        return 6

    num_full = total_samples // clip_len_samples
    remainder = total_samples % clip_len_samples

    # Prepare output paths
    out_name = args.name or input_path.stem
    prefix = args.prefix or out_name
    base_dir = Path("data/custom_dataset/audio")
    out_dir = base_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total_clips = num_full + (1 if (args.keep_remainder and remainder > 0) else 0)
    zpad = max(4, len(str(max(1, total_clips - 1))))

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Duration: {total_samples / sample_rate:.2f}s @ {sample_rate} Hz, Channels: {waveform.shape[0]}")
    print(f"[INFO] Clip length: {args.seconds:.3f}s ({clip_len_samples} samples)")
    print(f"[INFO] Saving to: {out_dir}")

    saved = 0
    # Save full clips
    for i in range(num_full):
        start = i * clip_len_samples
        end = start + clip_len_samples
        clip = waveform[:, start:end]
        out_path = out_dir / f"{prefix}_{i:0{zpad}d}.wav"
        try:
            torchaudio.save(str(out_path), clip, sample_rate)
        except Exception as e:
            print(f"[WARN] Failed to save clip {i} ({out_path}): {e}", file=sys.stderr)
            continue
        saved += 1

    # Save remainder if requested
    if remainder > 0 and args.keep_remainder:
        start = num_full * clip_len_samples
        clip = waveform[:, start:]
        out_path = out_dir / f"{prefix}_{num_full:0{zpad}d}.wav"
        try:
            torchaudio.save(str(out_path), clip, sample_rate)
            saved += 1
        except Exception as e:
            print(f"[WARN] Failed to save remainder clip ({out_path}): {e}", file=sys.stderr)

    if saved == 0:
        print("[WARN] No clips saved. Consider --keep-remainder if audio is shorter than clip length.")
    else:
        print(f"[OK] Saved {saved} clips to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
