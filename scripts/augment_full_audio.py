#!/usr/bin/env python3
"""
Augment full audio files by writing modified copies next to the originals.

Supported augmentations: reverse (time-flip) and additive white noise.

Behavior
- Scans the input directory (optionally recursively) for audio files.
- Builds the work list before doing any writes to avoid converting newly
  created files again.
- For each source file, writes a sibling file with CamelCase suffix before
  the extension, e.g., `egecan.mp3` -> `egecanReversed.mp3`.
- Skips files that already end with the suffix.
- Keeps originals intact.

Examples
- Reverse all files recursively under data/full_audio_files:
  python scripts/augment_full_audio.py \
    --input-dir data/full_audio_files \
    --recursive --reverse

- Add white noise at 10 dB SNR:
  python scripts/augment_full_audio.py \
    --input-dir data/full_audio_files --recursive --noise --noise-snr-db 10

- Use a custom CamelCase suffix for reverse outputs:
  python scripts/augment_full_audio.py \
    --input-dir data/full_audio_files --reverse --suffix Converted
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import sys

import torch
import torchaudio


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment full audio files in-place by writing modified copies")
    p.add_argument("--input-dir", required=True, help="Directory containing full audio files (class subfolders allowed)")
    p.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    # Augmentations
    p.add_argument("--reverse", action="store_true", help="Create time-reversed copies of each file")
    p.add_argument("--noise", action="store_true", help="Create noise-augmented copies at target SNR (white noise)")
    p.add_argument("--noise-snr-db", type=float, default=10.0, help="Target SNR in dB for noise augmentation (default: 10.0)")
    # Naming
    p.add_argument("--suffix", default="Reversed", help="CamelCase suffix to append (default: Reversed)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist")
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


def camel_case_suffix(s: str) -> str:
    # Ensure CamelCase-like formatting (Title case without spaces)
    s = "".join(part.capitalize() for part in s.replace("_", " ").replace("-", " ").split())
    return s or "Reversed"


def build_plan_ops(files: List[Path], op_suffixes: List[Tuple[str, str]], overwrite: bool) -> List[Tuple[Path, Path, str]]:
    """Return plan entries: (src, dst, op) for selected ops.

    op_suffixes: list of (op_name, suffix) where op_name in {"reverse","noise"}
    Skips sources that already end with any of the provided suffixes.
    """
    plan: List[Tuple[Path, Path, str]] = []
    suffix_set = {s for _, s in op_suffixes}
    for src in files:
        if any(src.stem.endswith(s) for s in suffix_set):
            continue  # already augmented with one of these ops
        for op_name, suffix in op_suffixes:
            dst = src.with_name(f"{src.stem}{suffix}{src.suffix}")
            if dst.exists() and not overwrite:
                continue
            plan.append((src, dst, op_name))
    return plan


def reverse_waveform(waveform: torch.Tensor) -> torch.Tensor:
    # waveform: [channels, samples]
    if waveform.ndim != 2:
        return waveform
    return torch.flip(waveform, dims=[1])


def signal_rms(waveform: torch.Tensor) -> torch.Tensor:
    # waveform: [channels, samples] -> per-channel RMS [channels]
    return torch.sqrt((waveform ** 2).mean(dim=1) + 1e-12)


def add_noise_snr(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add white Gaussian noise at target SNR (per channel)."""
    if waveform.ndim != 2:
        return waveform
    C, T = waveform.shape
    sig_rms = signal_rms(waveform)  # [C]
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise = torch.randn_like(waveform)
    noise_rms = torch.sqrt((noise ** 2).mean(dim=1) + 1e-12)  # [C]
    scale = (target_noise_rms / noise_rms).unsqueeze(1)  # [C,1]
    out = waveform + noise * scale
    return out.clamp_(-1.0, 1.0)


def save_like_source(dst: Path, waveform: torch.Tensor, sample_rate: int) -> bool:
    """Try saving with source extension; if it fails (e.g., codec not available),
    fall back to WAV with the same suffix name.
    Returns True if something was saved, False otherwise.
    """
    try:
        torchaudio.save(str(dst), waveform, sample_rate)
        return True
    except Exception as e:
        # Fallback to WAV
        try:
            alt = dst.with_suffix(".wav")
            torchaudio.save(str(alt), waveform, sample_rate)
            print(f"[AUG][INFO] Saved fallback WAV for {dst.name} as {alt.name} due to error: {e}")
            return True
        except Exception as e2:
            print(f"[AUG][WARN] Failed to save augmented file {dst} and fallback WAV: {e2}")
            return False


def main() -> int:
    args = parse_args()
    in_dir = Path(args.input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[AUG][ERROR] Input directory not found: {in_dir}", file=sys.stderr)
        return 2

    suffix_reverse = camel_case_suffix(args.suffix)
    suffix_noise = "Noised"
    if not (args.reverse or args.noise):
        print("[AUG][INFO] No augmentation selected; use --reverse and/or --noise.")
        return 0

    files = find_audio_files(in_dir, args.recursive)
    if not files:
        print(f"[AUG][WARN] No audio files found in {in_dir}")
        return 0

    op_suffixes: List[Tuple[str, str]] = []
    if args.reverse:
        op_suffixes.append(("reverse", suffix_reverse))
    if args.noise:
        op_suffixes.append(("noise", suffix_noise))
    plan = build_plan_ops(files, op_suffixes=op_suffixes, overwrite=args.overwrite)
    if not plan:
        print("[AUG][INFO] Nothing to do; outputs exist or sources already have the suffix.")
        return 0

    print(f"[AUG][INFO] Preparing to create {len(plan)} augmented file(s).")
    created = 0
    for src, dst, op in plan:
        try:
            waveform, sample_rate = torchaudio.load(str(src))
        except Exception as e:
            print(f"[AUG][WARN] Failed to load {src}: {e}")
            continue

        try:
            if op == "reverse":
                aug = reverse_waveform(waveform)
            elif op == "noise":
                aug = add_noise_snr(waveform, args.noise_snr_db)
            else:
                continue
        except Exception as e:
            print(f"[AUG][WARN] {op.capitalize()} failed for {src}: {e}")
            continue
        ok = save_like_source(dst, aug, sample_rate)
        if ok:
            created += 1

    print(f"[AUG][OK] Created {created} augmented file(s) under {in_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
