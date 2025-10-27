#!/usr/bin/env python3
"""
Split audio into fixed-length clips with optional silence removal.

Supports processing a single file or a directory of files. When a directory
is provided, it can search recursively with -r/--recursive.

Single file example:
  python scripts/clip_audio.py \
    --input path/to/file.wav \
    --seconds 3 \
    --name my_speaker

Directory example (recursive):
  python scripts/clip_audio.py \
    --input path/to/folder \
    --seconds 3 \
    --name run_all \
    -r

By default, silence is removed before splitting using a dB threshold. Use
--silence-top-db to control aggressiveness.

All clips are written under `data/custom_dataset/audio/<name>/`. In directory
mode, filenames are prefixed with the source file's stem to avoid collisions.
"""

import argparse
from pathlib import Path
import sys

import torchaudio
import librosa
import torch
from typing import List


AUDIO_EXTS: List[str] = [
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma",
]


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
        description=(
            "Cut audio into N-second clips with pre-splitting silence removal. "
            "Accepts a file or a directory."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input audio file or directory",
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
        "-r", "--recursive",
        action="store_true",
        help="When input is a directory, search files recursively",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=None,
        help="Optional target sample rate to resample input before splitting.",
    )
    parser.add_argument(
        "--silence-top-db",
        type=float,
        default=30.0,
        help=(
            "Threshold in dB for non-silent detection (librosa.effects.split). "
            "Higher removes more; default: 30."
        ),
    )
    return parser.parse_args()


def remove_all_silence(waveform: torch.Tensor, top_db: float) -> torch.Tensor:
    """Remove all silent regions (start/middle/end) from a waveform.

    Detection is performed on a mono reference using librosa.effects.split,
    then the detected intervals are concatenated for all channels.

    Args:
        waveform: Tensor of shape [channels, samples]
        top_db: dB threshold for non‑silent detection

    Returns:
        Tensor [channels, new_samples] with silence removed (may be length 0)
    """
    if waveform.numel() == 0 or waveform.shape[1] == 0:
        return waveform

    # Use a mono reference for robust detection
    y_mono = waveform.mean(dim=0).detach().cpu().numpy()
    intervals = librosa.effects.split(y=y_mono, top_db=top_db)

    if len(intervals) == 0:
        # Everything considered silent
        return waveform[:, :0]

    # Fast path: nothing to remove
    if len(intervals) == 1 and intervals[0][0] == 0 and intervals[0][1] == waveform.shape[1]:
        return waveform

    # Concatenate non‑silent pieces for all channels
    pieces = [waveform[:, s:e] for (s, e) in intervals]
    return torch.cat(pieces, dim=1) if pieces else waveform[:, :0]


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path not found: {input_path}", file=sys.stderr)
        return 2

    def find_audio_files(root: Path, recursive: bool) -> List[Path]:
        files: List[Path] = []
        if root.is_file():
            if root.suffix.lower() in AUDIO_EXTS:
                files.append(root)
        elif root.is_dir():
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

    # Prepare output base directory
    out_name = args.name or input_path.stem
    base_dir = Path("data/custom_dataset/audio")
    out_base = base_dir / out_name
    out_base.mkdir(parents=True, exist_ok=True)

    # Build file list depending on input type
    files_to_process: List[Path] = find_audio_files(input_path, args.recursive)
    if not files_to_process:
        print(f"[WARN] No audio files found in {input_path}")
        return 0

    print(f"[INFO] Found {len(files_to_process)} file(s) to process from {input_path}")

    total_saved = 0
    for ap in files_to_process:
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(str(ap))
        except Exception as e:
            print(f"[WARN] Failed to load {ap}: {e}")
            continue

        # Optional resample
        if args.target_sr and args.target_sr > 0 and args.target_sr != sample_rate:
            try:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=args.target_sr)
                waveform = resampler(waveform)
                sample_rate = args.target_sr
            except Exception as e:
                print(f"[WARN] Failed to resample {ap}: {e}")
                continue

        # Silence removal (graceful fallback if it fails)
        original_samples = waveform.shape[1]
        try:
            waveform = remove_all_silence(waveform, top_db=float(args.silence_top_db))
            trimmed_samples = waveform.shape[1]
        except Exception as e:
            print(f"[WARN] Silence removal failed for {ap} ({e}); proceeding without it.")
            trimmed_samples = original_samples

        # Compute segmenting parameters
        clip_len_samples = int(round(args.seconds * sample_rate))
        if clip_len_samples <= 0:
            print("[ERROR] Computed non-positive clip length in samples.", file=sys.stderr)
            return 5

        total_samples = waveform.shape[1]
        if total_samples == 0:
            print(f"[INFO] Skipping {ap.name}: considered silent at current threshold")
            continue

        num_full = total_samples // clip_len_samples
        remainder = total_samples % clip_len_samples

        # Output directory logic (flat output folder)
        file_out_dir = out_base
        safe_stem = ap.stem
        if input_path.is_file():
            # Backward-compatible single-file naming
            prefix = args.prefix or out_name
        else:
            # Directory mode: include file stem in prefix to avoid collisions
            prefix = (f"{args.prefix}_{safe_stem}" if args.prefix else safe_stem)
        file_out_dir.mkdir(parents=True, exist_ok=True)

        total_clips = num_full + (1 if (args.keep_remainder and remainder > 0) else 0)
        zpad = max(4, len(str(max(1, total_clips - 1))))

        print(
            f"[INFO] {ap.name}: {original_samples / sample_rate:.2f}s -> "
            f"{trimmed_samples / sample_rate:.2f}s @ {sample_rate} Hz; saving to {file_out_dir}"
        )

        saved = 0
        # Save full clips
        for i in range(num_full):
            start = i * clip_len_samples
            end = start + clip_len_samples
            clip = waveform[:, start:end]
            out_path = file_out_dir / f"{prefix}_{i:0{zpad}d}.wav"
            try:
                torchaudio.save(str(out_path), clip, sample_rate)
            except Exception as e:
                print(f"[WARN] Failed to save clip {i} for {ap.name} ({out_path}): {e}")
                continue
            saved += 1

        # Save remainder if requested
        if remainder > 0 and args.keep_remainder:
            start = num_full * clip_len_samples
            clip = waveform[:, start:]
            out_path = file_out_dir / f"{prefix}_{num_full:0{zpad}d}.wav"
            try:
                torchaudio.save(str(out_path), clip, sample_rate)
                saved += 1
            except Exception as e:
                print(f"[WARN] Failed to save remainder clip for {ap.name} ({out_path}): {e}")

        if saved == 0:
            print(f"[WARN] No clips saved for {ap.name}. Consider --keep-remainder or a shorter --seconds.")
        else:
            print(f"[OK] Saved {saved} clips for {ap.name} to {file_out_dir}")
            total_saved += saved

    print(f"[OK] Saved a total of {total_saved} clip(s) under {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
