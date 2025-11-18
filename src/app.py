import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import threading
import datetime
from pathlib import Path
from IPython.display import display, Markdown
import ipywidgets as widgets

from PIL import Image
from pathlib import Path
import subprocess
import sys
import subprocess
from pathlib import Path
import time

from PIL import Image
import torch
scripts_path = Path("../scripts").resolve() 
sys.path.append(str(scripts_path))

scripts_path = Path("../scripts").resolve()
sys.path.append(str(scripts_path))



LABELS = ["accept", "reject"] 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleVoiceCNN()
model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
model.to(device)
model.eval()

fs_record = 16000
recording_buf = []
is_recording = False
save_folder = Path("recordings")
save_folder.mkdir(exist_ok=True)

def record_thread():
    global recording_buf, is_recording
    recording_buf = []

    def callback(indata, frames, time, status):
        if is_recording:
            recording_buf.append(indata.copy())

    with sd.InputStream(samplerate=fs_record, channels=1, callback=callback):
        while is_recording:
            sd.sleep(100)


def process_and_predict(wav_path):
    wav_path = Path(wav_path)

    spec_name = "temp_recordings"
    spec_folder = Path("data/custom_dataset/spectrograms") / spec_name
    spec_folder.mkdir(parents=True, exist_ok=True)
    
  

    spec_cmd = [
        sys.executable,
        r"C:/Users/ASUS/Desktop/MachineLearning/git/scripts/generate_spectrograms.py",
        "--input-dir", str(wav_path.parent), 
        "--output-name", spec_name,
        "--sr", "16000",
        "--type", "mel",
        "--format", "png",
    ]
    
    print(f"[PROCESS] Running spectrogram generator: {' '.join(spec_cmd)}")
    rc = subprocess.call(spec_cmd)
    if rc != 0:
        raise RuntimeError(f"Spectrogram generator failed with code {rc}")
    

    spec_file = spec_folder / f"{wav_path.stem}.png"
    wait_time = 0
    while not spec_file.exists() and wait_time < 3:
        time.sleep(0.3)
        wait_time += 0.3

    if not spec_file.exists():
        raise FileNotFoundError(f"Spectrogram was not found {wav_path}!")

    img = Image.open(spec_file).convert("RGB")

    img = img.resize((128,128)) 

    t = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(t)
        pred = out.argmax(dim=1).item()
        label = LABELS[pred]
        prob = torch.softmax(out, dim=1)[0, pred].item()

    return wav_path, spec_file, label, prob


def on_toggle(b):
    global is_recording, recording_buf
    if not is_recording:
        is_recording = True
        b.description = 'Stop Recording'
        b.button_style = 'danger'
        threading.Thread(target=record_thread, daemon=True).start()
       
    else:
        is_recording = False
        b.description = 'Start Recording'
        b.button_style = 'success'
        with output:
            output.clear_output()
            print('Processing...')

        if not recording_buf:
            with output:
                print('No audio recorded.')
            return

        audio = np.concatenate(recording_buf, axis=0).ravel()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = save_folder / f'recording_{timestamp}.wav'
        write(str(fname), fs_record, np.int16(audio * 32767))

        wav_path, spec_path, label, prob = process_and_predict(fname)

        with output:
            output.clear_output()
            display(Markdown(f'**Saved WAV:** {wav_path}'))
            display(Markdown(f'**Saved spectrogram PNG:** {spec_path}'))
            display(Markdown(f'**Prediction:** {label} (p={prob:.3f})'))

button = widgets.Button(description='Start Recording', button_style='success')
output = widgets.Output()
button.on_click(on_toggle)
display(button, output)