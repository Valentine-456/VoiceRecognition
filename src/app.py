import os
import datetime
import numpy as np
from pathlib import Path
from scipy.io.wavfile import write
import sounddevice as sd
import ipywidgets as widgets
from IPython.display import display
import threading
import librosa
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

fs = 16000
recording = []
is_recording = False


save_folder = Path("recordings")
save_folder.mkdir(exist_ok=True)

spect_folder = Path("spectrograms")
spect_folder.mkdir(parents=True, exist_ok=True)

output = widgets.Output()


def record_thread():
    global recording, is_recording
    recording = []

    def callback(indata, frames, time, status):
        if is_recording:
            recording.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        while is_recording:
            sd.sleep(100)


def save_spectrogram_png(spec_db: np.ndarray, out_path: Path, *, grayscale: bool, cmap: str, image_scale: float) -> None:
    """Save a 2D spectrogram array as a PNG image (identical to reference implementation)."""
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

def compute_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute mel spectrogram (identical parameters to generate_spectrograms.py)."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)  
    return S_db

def save_spectrogram(audio_file: Path, spect_folder: Path) -> Path:
    """Generate mel-spectrogram from audio file and save as PNG identical to generator."""
    y, sr = librosa.load(audio_file, sr=fs, mono=True)
    spec_db = compute_spectrogram(y, sr)
    base_filename = audio_file.stem
    out_path = spect_folder / f"{base_filename}.png"
    save_spectrogram_png(spec_db, out_path, grayscale=False, cmap="magma", image_scale=1.0)
    return out_path

def toggle_recording(b):
    global is_recording, recording
    with output:
        output.clear_output()
        if not is_recording:
            is_recording = True
            button.description = "Stop Recording"
            button.button_style = "danger"
            print("Recording...")
            threading.Thread(target=record_thread, daemon=True).start()
        else:
            is_recording = False
            button.description = "Start Recording"
            button.button_style = "success"
            if not recording:
                print("No audio recorded.")
                return

            audio_data = np.concatenate(recording, axis=0)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_folder / f"recording_{timestamp}.wav"
            write(filename, fs, np.int16(audio_data * 32767))
            print(f"Recording finished! Audio saved as: {filename}")

            spect_filename = save_spectrogram(filename, spect_folder)
            print(f"Spectrogram saved as: {spect_filename}")

button = widgets.Button(description="Start Recording", button_style="success")
button.on_click(toggle_recording)

display(button, output)
