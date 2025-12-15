#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torchaudio
import librosa
import torch

from file_utils import collect_files, AUDIO_EXTENSIONS


def positive_float(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}")
    if v <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut audio into fixed-length clips with optional silence removal.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input audio file or directory",
    )
    parser.add_argument(
        "--seconds",
        type=positive_float,
        required=True,
        help="Clip length in seconds",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output folder name under data/custom_dataset/audio/",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix for clips",
    )
    parser.add_argument(
        "--prefix-from-parent",
        action="store_true",
        help="Use parent folder name as prefix",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search directories recursively",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=None,
        help="Resample to this sample rate before splitting",
    )
    parser.add_argument(
        "--silence-top-db",
        type=float,
        default=30.0,
        help="Silence removal threshold in dB",
    )
    return parser.parse_args()


def remove_all_silence(waveform: torch.Tensor, top_db: float) -> torch.Tensor:
    """Remove silent regions before segmenting."""
    if waveform.numel() == 0 or waveform.shape[1] == 0:
        return waveform

    y_mono = waveform.mean(dim=0).detach().cpu().numpy()
    intervals = librosa.effects.split(y=y_mono, top_db=top_db)

    if len(intervals) == 0:
        return waveform[:, :0]

    if len(intervals) == 1 and intervals[0][0] == 0 and intervals[0][1] == waveform.shape[1]:
        return waveform

    pieces = [waveform[:, s:e] for (s, e) in intervals]
    return torch.cat(pieces, dim=1) if pieces else waveform[:, :0]


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path not found: {input_path}", file=sys.stderr)
        return 2

    out_name = args.name or input_path.stem
    base_dir = Path("data/custom_dataset/audio")
    out_base = base_dir / out_name
    out_base.mkdir(parents=True, exist_ok=True)

    files_to_process = collect_files(input_path, AUDIO_EXTENSIONS, args.recursive)
    if not files_to_process:
        print(f"[WARN] No audio files found in {input_path}")
        return 0

    print(f"[INFO] Found {len(files_to_process)} file(s) to process from {input_path}")

    total_saved = 0
    for ap in files_to_process:
        try:
            waveform, sample_rate = torchaudio.load(str(ap))
        except Exception as e:
            print(f"[WARN] Failed to load {ap}: {e}")
            continue

        if args.target_sr and args.target_sr > 0 and args.target_sr != sample_rate:
            try:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=args.target_sr)
                waveform = resampler(waveform)
                sample_rate = args.target_sr
            except Exception as e:
                print(f"[WARN] Failed to resample {ap}: {e}")
                continue

        original_samples = waveform.shape[1]
        try:
            waveform = remove_all_silence(waveform, top_db=float(args.silence_top_db))
            trimmed_samples = waveform.shape[1]
        except Exception as e:
            print(f"[WARN] Silence removal failed for {ap} ({e}); proceeding without it.")
            trimmed_samples = original_samples

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

        file_out_dir = out_base
        safe_stem = ap.stem
        if args.prefix_from_parent:
            prefix = f"{ap.parent.name}_{safe_stem}"
        elif input_path.is_file():
            prefix = args.prefix or out_name
        else:
            prefix = (f"{args.prefix}_{safe_stem}" if args.prefix else safe_stem)
        file_out_dir.mkdir(parents=True, exist_ok=True)

        total_clips = num_full
        zpad = max(4, len(str(max(1, total_clips - 1))))

        print(
            f"[INFO] {ap.name}: {original_samples / sample_rate:.2f}s -> "
            f"{trimmed_samples / sample_rate:.2f}s @ {sample_rate} Hz; saving to {file_out_dir}"
        )

        saved = 0
        for i in range(num_full):
            start = i * clip_len_samples
            end = start + clip_len_samples
            clip = waveform[:, start:end]
            flat_out = file_out_dir / f"{prefix}_{i:0{zpad}d}.wav"
            try:
                torchaudio.save(str(flat_out), clip, sample_rate)
            except Exception as e:
                print(f"[WARN] Failed to save clip {i} for {ap.name} ({flat_out}): {e}")
                continue
            saved += 1

        if saved == 0:
            print(f"[WARN] No clips saved for {ap.name}. Consider a shorter --seconds.")
        else:
            print(f"[OK] Saved {saved} clips for {ap.name} to {file_out_dir}")
            total_saved += saved

    print(f"[OK] Saved a total of {total_saved} clip(s) under {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
