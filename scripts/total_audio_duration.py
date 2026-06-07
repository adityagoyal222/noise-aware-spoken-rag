from __future__ import annotations

import contextlib
import sys
import wave
from pathlib import Path

AUDIO_EXTS = {".wav"}


def get_wav_duration_seconds(path: Path) -> float:
    with contextlib.closing(wave.open(str(path), "rb")) as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return frames / float(rate) if rate else 0.0


def main() -> int:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path.cwd() / "data" / "raw_audio"

    if not root.exists():
        print(f"raw_audio folder not found: {root}")
        return 1

    total_seconds = 0.0
    skipped = 0
    scanned = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        scanned += 1
        if path.suffix.lower() in AUDIO_EXTS:
            try:
                total_seconds += get_wav_duration_seconds(path)
            except wave.Error:
                skipped += 1
        else:
            skipped += 1

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    print(f"Total duration: {hours}h {minutes}m {seconds:.1f}s")
    print(f"Scanned files: {scanned}")
    print(f"Skipped files (non-wav or unreadable): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
