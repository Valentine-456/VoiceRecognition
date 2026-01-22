import os
from pathlib import Path
from typing import List, Callable, Optional

import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf



# ---------------------------------------------------------------
# Save a mel spectrogram as PNG (your original example cleaned up)
# ---------------------------------------------------------------
def save_mel_spectrogram_png(
    wav_path: Path,
    out_png_path: Path,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    cmap: str = "magma"
):
    y, sr_loaded = librosa.load(str(wav_path), sr=sr)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(1.28, 1.28), dpi=100)  # 128x128 px output
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(S_db[::-1, :], aspect="auto", origin="lower",
               interpolation="nearest", cmap=cmap)

    plt.savefig(out_png_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# -------------------------------------------------------------------
# Segment audio into fixed lengths + optional augmentation pipeline
# -------------------------------------------------------------------
def segment_audio(
    y: np.ndarray,
    sr: int,
    segment_length_sec: float = 3.0
) -> List[np.ndarray]:
    samples_per_segment = int(segment_length_sec * sr)
    segments = []

    for start in range(0, len(y), samples_per_segment):
        end = start + samples_per_segment
        seg = y[start:end]

        if len(seg) < samples_per_segment:
            break  # skip last short fragment

        segments.append(seg)

    return segments


# --------------------------------------------
# A simple augmentation registry (extensible)
# --------------------------------------------
def apply_augmentations(
    y: np.ndarray,
    augmentations: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None
) -> np.ndarray:
    if augmentations:
        for aug in augmentations:
            y = aug(y)
    return y


# ------------------------
# Example augmentations
# ------------------------
def add_noise(y: np.ndarray, noise_level: float = 0.005):
    return y + noise_level * np.random.randn(len(y))


def pitch_shift(y: np.ndarray, sr: int, n_steps: int = 2):
    return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)


def time_stretch(y: np.ndarray, rate: float = 1.1):
    return librosa.effects.time_stretch(y, rate)


# -------------------------------------------------------------------------
# MAIN PIPELINE: load wav → split → apply augmentations → save spectrograms
# -------------------------------------------------------------------------
def process_audio_file(
    audio_path: Path,
    out_dir: Path,
    segment_length_sec: float = 3.0,
    sr: int = 16000,
    augmentations: Optional[List[Callable]] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load full audio
    y, sr_loaded = librosa.load(str(audio_path), sr=sr)

    # Segment into X-second chunks
    segments = segment_audio(y, sr=sr, segment_length_sec=segment_length_sec)

    file_stem = audio_path.stem  # get base name without extension

    for i, seg in enumerate(segments):

    # Apply augmentation pipeline
     seg_aug = apply_augmentations(seg, augmentations)

    # Save temp wav (optional)
     tmp_wav = out_dir / f"{file_stem}_segment_{i}.wav"
     sf.write(str(tmp_wav), seg_aug, sr)

    # Save spectrogram as PNG
     out_png = out_dir / f"{file_stem}_{i}.png"
     save_mel_spectrogram_png(tmp_wav, out_png_path=out_png)

    # Remove temp wav
     os.remove(tmp_wav)


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    audio_path = Path("audios/ivan_2.wav")
    out_dir = Path("audios/segments_out")

    # Choose augmentations (optional)
    # augmentations = [
    #     lambda y: add_noise(y, noise_level=0.003),
    #     lambda y: pitch_shift(y, sr=16000, n_steps=1),
    # ]

    process_audio_file(
        audio_path=audio_path,
        out_dir=out_dir,
        segment_length_sec=3.0,
        sr=16000,
        #augmentations=augmentations,
    )
