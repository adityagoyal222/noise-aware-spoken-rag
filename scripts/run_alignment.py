"""
run_alignment.py — Align ASR outputs with diarization for a given model.

Reads ASR segments from data/asr_outputs/<model>/segments/ and diarization
from data/diarization_outputs/json/, then writes aligned chunk CSVs to
data/aligned_chunks/<model>/ and feature tables to data/metrics/<model>/.

Usage:
    # Align a single model
    python scripts/run_alignment.py --model medium

    # Align all models that have ASR output
    python scripts/run_alignment.py --all
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASR_OUTPUT_ROOT = PROJECT_ROOT / "data" / "asr_outputs"
DIAR_DIR = PROJECT_ROOT / "data" / "diarization_outputs" / "json"

ALL_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: str
    confidence: float
    seg_consistency: float
    overlap: float
    flip_rate: float


# ── loading ───────────────────────────────────────────────────────────────────

def load_asr_segments(path: Path) -> list[AsrSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        AsrSegment(
            start=float(row["start"]),
            end=float(row["end"]),
            text=str(row.get("text", "")),
            avg_logprob=float(row.get("avg_logprob", -10.0)),
            no_speech_prob=float(row.get("no_speech_prob", 0.0)),
        )
        for row in payload.get("segments", [])
    ]


def load_diar_segments(path: Path) -> list[DiarSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        DiarSegment(
            start=float(row["start"]),
            end=float(row["end"]),
            speaker=str(row.get("speaker", "UNKNOWN")),
            confidence=float(row.get("confidence", 0.0)),
            seg_consistency=float(row.get("seg_consistency", 0.0)),
            overlap=float(row.get("overlap", 0.0)),
            flip_rate=float(row.get("flip_rate", 0.0)),
        )
        for row in payload.get("segments", [])
    ]


# ── feature computation ───────────────────────────────────────────────────────

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _compute_overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    return max(0.0, min(e1, e2) - max(s1, s2))


def _normalize_logprobs(segments: list[AsrSegment]) -> dict[int, float]:
    if not segments:
        return {}
    values = [s.avg_logprob for s in segments]
    lo, hi = min(values), max(values)
    span = hi - lo
    return {
        i: (1.0 if span == 0 else (s.avg_logprob - lo) / span)
        for i, s in enumerate(segments)
    }


def _compute_redundancy(segments: list[AsrSegment], idx: int) -> float:
    current = _tokenize(segments[idx].text)
    neighbors = []
    if idx > 0:
        neighbors.append(_tokenize(segments[idx - 1].text))
    if idx + 1 < len(segments):
        neighbors.append(_tokenize(segments[idx + 1].text))
    if not neighbors:
        return 0.0
    return max(_jaccard(current, n) for n in neighbors)


def _compute_turn_completeness(seg: AsrSegment) -> float:
    duration = max(0.0, seg.end - seg.start)
    text = seg.text.strip()
    tokens = _tokenize(text)
    duration_score = min(1.0, duration / 5.0)
    if text.endswith((".", "?", "!")):
        punct_score = 1.0
    elif text.endswith(","):
        punct_score = 0.7
    else:
        punct_score = 0.85
    length_score = min(1.0, len(tokens) / 6.0)
    speech_score = max(0.0, 1.0 - seg.no_speech_prob)
    return duration_score * punct_score * length_score * speech_score


def _compute_diar_stability(overlaps: list[tuple[DiarSegment, float]]) -> float:
    if not overlaps:
        return 0.0
    total = sum(o for _, o in overlaps)
    if total <= 0.0:
        return 0.0
    speaker_overlap: dict[str, float] = {}
    for diar, o in overlaps:
        speaker_overlap[diar.speaker] = speaker_overlap.get(diar.speaker, 0.0) + o
    dominant = max(speaker_overlap.values()) / total
    wconf = wcons = wnoverlap = wnoflip = 0.0
    for diar, o in overlaps:
        w = o / total
        wconf += diar.confidence * w
        wcons += diar.seg_consistency * w
        wnoverlap += (1.0 - diar.overlap) * w
        wnoflip += (1.0 - diar.flip_rate) * w
    return max(0.0, min(1.0, dominant * wconf * wcons * wnoverlap * wnoflip))


# ── alignment ─────────────────────────────────────────────────────────────────

def align(asr_segments: list[AsrSegment], diar_segments: list[DiarSegment]) -> list[dict]:
    logprob_norm = _normalize_logprobs(asr_segments)
    rows = []

    for idx, asr in enumerate(asr_segments):
        overlaps = [
            (d, _compute_overlap(asr.start, asr.end, d.start, d.end))
            for d in diar_segments
            if _compute_overlap(asr.start, asr.end, d.start, d.end) > 0.0
        ]

        if overlaps:
            overlaps.sort(key=lambda x: (x[1], x[0].confidence, -x[0].start), reverse=True)
            best_diar, best_overlap = overlaps[0]
            speaker = best_diar.speaker
            diar_turn_start = best_diar.start
            diar_turn_end = best_diar.end
        else:
            best_overlap = 0.0
            speaker = "UNKNOWN"
            diar_turn_start = math.nan
            diar_turn_end = math.nan

        duration = max(0.0, asr.end - asr.start)
        purity = best_overlap / duration if duration > 0.0 else 0.0

        rows.append({
            "start": asr.start,
            "end": asr.end,
            "text": asr.text,
            "speaker": speaker,
            "ASRConf": logprob_norm.get(idx, 0.0),
            "DiarStab": _compute_diar_stability(overlaps),
            "TurnComp": _compute_turn_completeness(asr),
            "Redund": _compute_redundancy(asr_segments, idx),
            "Purity": purity,
            "MixPenalty": 1.0 - purity,
            "diar_turn_start": diar_turn_start,
            "diar_turn_end": diar_turn_end,
            "diar_overlap_sec": best_overlap,
        })

    return rows


# ── per-file processing ───────────────────────────────────────────────────────

def process_file(
    file_id: str,
    asr_dir: Path,
    aligned_dir: Path,
    metrics_dir: Path,
) -> bool:
    asr_path = asr_dir / f"{file_id}.json"
    diar_path = DIAR_DIR / f"{file_id}.json"

    if not asr_path.exists():
        print(f"  [skip] No ASR file: {asr_path.name}")
        return False
    if not diar_path.exists():
        print(f"  [skip] No diarization file: {diar_path.name}")
        return False

    asr_segments = load_asr_segments(asr_path)
    diar_segments = load_diar_segments(diar_path)
    aligned_rows = align(asr_segments, diar_segments)

    aligned_df = pd.DataFrame(aligned_rows)
    aligned_df.insert(0, "file_id", file_id)

    aligned_df.to_csv(aligned_dir / f"{file_id}_aligned.csv", index=False)

    metrics_cols = [
        "file_id", "start", "end", "speaker",
        "ASRConf", "DiarStab", "TurnComp", "Redund", "Purity", "MixPenalty",
    ]
    aligned_df[metrics_cols].to_csv(metrics_dir / f"{file_id}_features.csv", index=False)

    return True


# ── main ──────────────────────────────────────────────────────────────────────

def run_model(model_name: str) -> dict:
    asr_dir = ASR_OUTPUT_ROOT / model_name / "segments"
    aligned_dir = PROJECT_ROOT / "data" / "aligned_chunks" / model_name
    metrics_dir = PROJECT_ROOT / "data" / "metrics" / model_name

    if not asr_dir.exists():
        print(f"  [skip] ASR segments dir not found: {asr_dir}")
        return {"model": model_name, "processed": 0, "skipped": 0}

    aligned_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    asr_ids = {p.stem for p in asr_dir.glob("*.json")}
    diar_ids = {p.stem for p in DIAR_DIR.glob("*.json")} if DIAR_DIR.exists() else set()
    file_ids = sorted(asr_ids & diar_ids)

    if not file_ids:
        print(f"  [skip] No matching ASR+diarization file pairs for model '{model_name}'.")
        return {"model": model_name, "processed": 0, "skipped": len(asr_ids)}

    processed = sum(
        1 for fid in file_ids
        if process_file(fid, asr_dir, aligned_dir, metrics_dir)
    )
    return {"model": model_name, "processed": processed, "skipped": len(asr_ids) - processed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Align ASR outputs with diarization.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=ALL_MODELS, help="Single model to align.")
    group.add_argument("--all", action="store_true", help="Align all models that have ASR output.")
    args = parser.parse_args()

    models = ALL_MODELS if args.all else [args.model]
    summary = []
    for model_name in models:
        print(f"\n── Aligning: {model_name} ──")
        result = run_model(model_name)
        summary.append(result)
        print(f"   processed={result['processed']}  skipped={result['skipped']}")

    print("\n── Alignment complete ──")
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
