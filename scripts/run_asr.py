"""
run_asr.py — Run Whisper ASR over all model sizes sequentially.

Each model writes to its own subdirectory under data/asr_outputs/<model>/.
Fully resumable: already-processed files are skipped via a per-model checkpoint
index. Re-run at any time to pick up from where it left off.

Usage:
    # Run all models (default)
    python scripts/run_asr.py

    # Run specific models only
    python scripts/run_asr.py --models medium large-v3

    # Force re-run of failed files
    python scripts/run_asr.py --retry-failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"
ASR_OUTPUT_ROOT = PROJECT_ROOT / "data" / "asr_outputs"

AUDIO_EXTS = {".wav"}
SESSION_PATTERN = re.compile(r"^(?P<session>[A-Z]{2}\d{4}[a-d])")

ALL_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

# faster-whisper compute settings per device type
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

LANGUAGE = None
TASK = "transcribe"
BEAM_SIZE = 5
TEMPERATURE = 0.0
VAD_FILTER = False
WORD_TIMESTAMPS = False


# ── helpers ───────────────────────────────────────────────────────────────────

def iter_audio_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def parse_session_id(name: str) -> Optional[str]:
    m = SESSION_PATTERN.match(name)
    return m.group("session") if m else None


def make_file_id(relative_path: str) -> str:
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]


def load_checkpoint(path: Path) -> pd.DataFrame:
    columns = [
        "file_id", "relative_path", "session_id", "status",
        "model_name", "device", "compute_type", "language", "task",
        "runtime_seconds", "audio_duration_seconds",
        "raw_json_path", "segments_json_path", "transcript_path",
        "error", "updated_at",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=columns)
    if "updated_at" in df.columns:
        df = df.sort_values("updated_at").drop_duplicates("file_id", keep="last")
    return df


def save_checkpoint(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def build_segments_payload(segments) -> list[dict]:
    return [
        {
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip(),
            "avg_logprob": float(s.avg_logprob),
            "compression_ratio": float(s.compression_ratio),
            "no_speech_prob": float(s.no_speech_prob),
        }
        for s in segments
    ]


# ── per-model run ─────────────────────────────────────────────────────────────

def run_model(model_name: str, audio_files: list[Path], retry_failed: bool) -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError("faster-whisper is not installed.") from exc

    model_dir = ASR_OUTPUT_ROOT / model_name
    raw_dir = model_dir / "raw"
    segments_dir = model_dir / "segments"
    transcripts_dir = model_dir / "transcripts"
    logs_dir = model_dir / "logs"
    checkpoints_dir = model_dir / "checkpoints"

    for d in [raw_dir, segments_dir, transcripts_dir, logs_dir, checkpoints_dir]:
        d.mkdir(parents=True, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_dir / f"asr_run_{run_id}.log"
    checkpoint_path = checkpoints_dir / "asr_index.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )

    logging.info("=== Model: %s ===", model_name)
    checkpoint_df = load_checkpoint(checkpoint_path).set_index("file_id", drop=False)

    logging.info("Initializing Whisper model: %s (device=%s, compute=%s)", model_name, DEVICE, COMPUTE_TYPE)
    model = WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)

    success, skipped, failed = 0, 0, 0

    for i, audio_path in enumerate(audio_files, start=1):
        relative_path = audio_path.relative_to(PROJECT_ROOT).as_posix()
        session_id = parse_session_id(audio_path.name)
        file_id = make_file_id(relative_path)

        if file_id in checkpoint_df.index:
            status = str(checkpoint_df.loc[file_id].get("status", "")).lower()
            if status != "failed" or not retry_failed:
                logging.info("[%d/%d] Skipping (checkpointed): %s", i, len(audio_files), relative_path)
                skipped += 1
                continue

        logging.info("[%d/%d] Transcribing: %s", i, len(audio_files), relative_path)
        t0 = time.perf_counter()
        raw_json_path = raw_dir / f"{file_id}.json"
        segments_json_path = segments_dir / f"{file_id}.json"
        transcript_path = transcripts_dir / f"{file_id}.txt"

        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=LANGUAGE,
                task=TASK,
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE,
                vad_filter=VAD_FILTER,
                word_timestamps=WORD_TIMESTAMPS,
            )
            segments_payload = build_segments_payload(segments)
            full_text = " ".join(s["text"] for s in segments_payload).strip()
            runtime = time.perf_counter() - t0

            write_json(raw_json_path, {
                "file_id": file_id,
                "session_id": session_id,
                "relative_path": relative_path,
                "audio_path": str(audio_path),
                "model_name": model_name,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
                "language": info.language if info else None,
                "task": TASK,
                "audio_duration_seconds": float(info.duration) if info else None,
                "runtime_seconds": float(runtime),
                "text": full_text,
                "segments": segments_payload,
                "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "status": "success",
            })
            write_json(segments_json_path, {
                "file_id": file_id,
                "session_id": session_id,
                "relative_path": relative_path,
                "segments": segments_payload,
            })
            write_text(transcript_path, full_text)

            checkpoint_df.loc[file_id] = {
                "file_id": file_id, "relative_path": relative_path,
                "session_id": session_id, "status": "success",
                "model_name": model_name, "device": DEVICE, "compute_type": COMPUTE_TYPE,
                "language": info.language if info else None, "task": TASK,
                "runtime_seconds": float(runtime),
                "audio_duration_seconds": float(info.duration) if info else None,
                "raw_json_path": str(raw_json_path),
                "segments_json_path": str(segments_json_path),
                "transcript_path": str(transcript_path),
                "error": None,
                "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            save_checkpoint(checkpoint_df.reset_index(drop=True), checkpoint_path)
            success += 1

        except Exception as exc:
            runtime = time.perf_counter() - t0
            logging.exception("Failed to transcribe %s", relative_path)
            checkpoint_df.loc[file_id] = {
                "file_id": file_id, "relative_path": relative_path,
                "session_id": session_id, "status": "failed",
                "model_name": model_name, "device": DEVICE, "compute_type": COMPUTE_TYPE,
                "language": None, "task": TASK,
                "runtime_seconds": float(runtime), "audio_duration_seconds": None,
                "raw_json_path": str(raw_json_path),
                "segments_json_path": str(segments_json_path),
                "transcript_path": str(transcript_path),
                "error": repr(exc),
                "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            save_checkpoint(checkpoint_df.reset_index(drop=True), checkpoint_path)
            failed += 1

    logging.info(
        "Model %s done: %d success, %d skipped, %d failed",
        model_name, success, skipped, failed,
    )
    return {"model": model_name, "success": success, "skipped": skipped, "failed": failed}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Whisper ASR for one or more model sizes.")
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS, metavar="MODEL",
        help=f"Model sizes to run (default: all). Choices: {ALL_MODELS}",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Re-process files previously marked as failed.",
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    audio_files = iter_audio_files(RAW_AUDIO_DIR)
    if not audio_files:
        print(f"No audio files found under {RAW_AUDIO_DIR}")
        return 1
    print(f"Found {len(audio_files)} audio file(s).")

    summary_rows = []
    for model_name in args.models:
        print(f"\n{'='*50}\nRunning model: {model_name}\n{'='*50}")
        result = run_model(model_name, audio_files, args.retry_failed)
        summary_rows.append(result)

    print("\n── ASR run complete ──")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
