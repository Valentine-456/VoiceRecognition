#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import sys

import torch
import torchaudio

from file_utils import collect_files, AUDIO_EXTENSIONS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment full audio files in-place by writing modified copies")
    p.add_argument("--input-dir", required=True, help="Directory containing full audio files (class subfolders allowed)")
    p.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    p.add_argument("--reverse", action="store_true", help="Create time-reversed copies of each file")
    p.add_argument("--noise", action="store_true", help="Create white noise copies")
    p.add_argument("--noise-snr-db", type=float, default=10.0, help="Target SNR in dB for noise augmentation")
    return p.parse_args()


def build_file_names(files: List[Path], suffix_names: List[Tuple[str, str]]) -> List[Tuple[Path, Path, str]]:
    """Maps src to dst files with modified name."""
    plan: List[Tuple[Path, Path, str]] = []
    suffix_set = {s for _, s in suffix_names}
    for src in files:
        if any(src.stem.endswith(s) for s in suffix_set):
            continue
        for op_name, suffix in suffix_names:
            dst = src.with_name(f"{src.stem}{suffix}{src.suffix}")
            if dst.exists():
                continue
            plan.append((src, dst, op_name))
    return plan


def reverse_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 2:
        return waveform
    return torch.flip(waveform, dims=[1])


def signal_rms(waveform: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((waveform ** 2).mean(dim=1) + 1e-12)


def add_noise_snr(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add white noise at target SNR."""
    if waveform.ndim != 2:
        return waveform

    sig_rms = signal_rms(waveform)
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise = torch.randn_like(waveform)
    noise_rms = torch.sqrt((noise ** 2).mean(dim=1) + 1e-12)
    scale = (target_noise_rms / noise_rms).unsqueeze(1)
    out = waveform + noise * scale
    return out.clamp_(-1.0, 1.0)


def save_like_source(dst: Path, waveform: torch.Tensor, sample_rate: int) -> bool:
    """Save with the source extension or fall back to WAV."""
    try:
        torchaudio.save(str(dst), waveform, sample_rate)
        return True
    except Exception as e:
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

    suffix_reverse = "Reversed"
    suffix_noise = "Noised"
    if not (args.reverse or args.noise):
        print("[AUG][INFO] No augmentation selected; use --reverse and/or --noise.")
        return 0

    files = collect_files(in_dir, AUDIO_EXTENSIONS, args.recursive)
    if not files:
        print(f"[AUG][WARN] No audio files found in {in_dir}")
        return 0

    suffix_names: List[Tuple[str, str]] = []
    if args.reverse:
        suffix_names.append(("reverse", suffix_reverse))
    if args.noise:
        suffix_names.append(("noise", suffix_noise))
    plan = build_file_names(files, suffix_names)
    if not plan:
        print("[AUG][INFO] Nothing to do; outputs exist or sources already have the suffix.")
        return 0

    print(f"[AUG][INFO] Preparing to create {len(plan)} augmented file(s).")
    created = 0
    for src, dst, suffix in plan:
        try:
            waveform, sample_rate = torchaudio.load(str(src))
        except Exception as e:
            print(f"[AUG][WARN] Failed to load {src}: {e}")
            continue

        try:
            if suffix == "reverse":
                aug = reverse_waveform(waveform)
            elif suffix == "noise":
                aug = add_noise_snr(waveform, args.noise_snr_db)
            else:
                continue
        except Exception as e:
            print(f"[AUG][WARN] {suffix} failed for {src}: {e}")
            continue
        ok = save_like_source(dst, aug, sample_rate)
        if ok:
            created += 1

    print(f"[AUG][OK] Created {created} augmented file(s) under {in_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
