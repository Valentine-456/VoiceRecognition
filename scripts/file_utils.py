from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple


AUDIO_EXTENSIONS: Tuple[str, ...] = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
)


def _normalize_exts(extensions: Sequence[str]) -> Set[str]:
    norm: Set[str] = set()
    for ext in extensions:
        if not ext:
            continue
        ext = ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        norm.add(ext)
    return norm


def collect_files(root: Path, extensions: Sequence[str], recursive: bool) -> List[Path]:
    """Return sorted files under root that match the provided extensions."""
    ext_set = _normalize_exts(extensions)
    files: List[Path] = []

    if root.is_file():
        if root.suffix.lower() in ext_set:
            files.append(root)
        return files

    if not root.is_dir():
        return files

    iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.lower() in ext_set:
            files.append(candidate)

    files.sort()
    return files


__all__ = ["collect_files", "AUDIO_EXTENSIONS"]
