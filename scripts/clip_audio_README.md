# Clip Audio Script (scripts/clip_audio.py)

Split audio into fixed-length clips with optional silence removal. Works on a single file or a directory (with optional recursion). All clips are written to `data/custom_dataset/audio/<name>/`.

Examples
- Single file
  - `python scripts/clip_audio.py --input data/full_audio_files/ObamaSpeech.mp3 --seconds 10 --name obama_clips --keep-remainder --target-sr 16000`
- Directory (non-recursive)
  - `python scripts/clip_audio.py --input data/full_audio_files --seconds 3 --name all_clips --silence-top-db 40`
- Directory (recursive)
  - `python scripts/clip_audio.py --input data/full_audio_files --seconds 3 --name all_clips_recursive -r`

Arguments
- `--input` (path): Audio file or directory to process.
- `--seconds` (float > 0): Clip length in seconds (e.g., 3, 5, 10).
- `--name` (str): Output folder under `data/custom_dataset/audio/`.
- `--prefix` (str, optional): Filename prefix for clips. Defaults to `--name` in single‑file mode; in directory mode, the source file’s stem is used unless you provide a prefix (then `<prefix>_<stem>_####.wav`).
- `--keep-remainder` (flag): Also save the final short clip if leftover audio exists.
- `--target-sr` (int, optional): Resample target sample rate (e.g., 16000) before splitting.
- `--silence-top-db` (float, default 30): Remove all silent regions before splitting. Higher removes more (typical 20–60).
- `-r`, `--recursive` (flag): When input is a directory, include subfolders.

Behavior
- Silence removal uses a mono reference with `librosa.effects.split` and concatenates non‑silent parts across channels. If silence removal fails (e.g., NumPy mismatch), the script logs a warning and proceeds without it.
- Output naming is flat in `data/custom_dataset/audio/<name>/`.
  - Single file: `<prefix>_0000.wav`, `<prefix>_0001.wav`, ... (default prefix = `<name>`)
  - Directory input: `<stem>_0000.wav`, ... or `<prefix>_<stem>_0000.wav` if `--prefix` is provided.

Notes
- Supported formats depend on torchaudio (e.g., wav, mp3, flac, ogg, m4a).
