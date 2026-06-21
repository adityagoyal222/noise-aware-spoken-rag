"""
Compute per-meeting Diarization Error Rate (DER).

Reference annotations are derived from AMI gold word-level XML files
(data/gold_transcripts/<session>.*.words.xml), which contain per-speaker
word timestamps. Speaker segments are formed by grouping consecutive words
from the same speaker with gaps < GAP_TOLERANCE merged.

Hypothesis annotations are read from pyannote RTTM files
(data/diarization_outputs/rttm/<audio_id>.rttm).

DER is computed using pyannote.metrics with collar=0.25s (standard for AMI).

Output: data/analysis/der_per_meeting.csv

Usage:
    python scripts/compute_der.py
"""

import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT_ROOT / "data" / "gold_transcripts"
RTTM_DIR = PROJECT_ROOT / "data" / "diarization_outputs" / "rttm"
RAW_ASR_DIR = PROJECT_ROOT / "data" / "asr_outputs" / "medium" / "raw"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

COLLAR = 0.25       # seconds — standard AMI evaluation collar
GAP_TOLERANCE = 0.3  # merge same-speaker words within this gap into one segment


def _parse_gold_annotation(session_id: str) -> Annotation:
    """
    Build a pyannote Annotation from per-speaker gold word XML files.
    Each speaker's words are merged into contiguous segments (gaps < GAP_TOLERANCE).
    """
    xml_files = sorted(GOLD_DIR.glob(f"{session_id}.*.words.xml"))
    if not xml_files:
        return None

    ref = Annotation(uri=session_id)

    for xml_path in xml_files:
        # Speaker label from filename: ES2014a.A.words.xml → A
        speaker = xml_path.name.split(".")[1]

        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue

        root = tree.getroot()
        word_intervals = []

        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag != "w":
                continue
            is_punc = elem.get("punc", "false") == "true"
            if is_punc:
                continue
            try:
                start = float(elem.get("starttime", -1))
                end = float(elem.get("endtime", -1))
            except (TypeError, ValueError):
                continue
            if start < 0 or end < 0 or end <= start:
                continue
            word_intervals.append((start, end))

        if not word_intervals:
            continue

        word_intervals.sort()

        # Merge words into speaker turns
        seg_start, seg_end = word_intervals[0]
        for w_start, w_end in word_intervals[1:]:
            if w_start - seg_end <= GAP_TOLERANCE:
                seg_end = max(seg_end, w_end)
            else:
                if seg_end > seg_start:
                    ref[Segment(seg_start, seg_end)] = speaker
                seg_start, seg_end = w_start, w_end
        if seg_end > seg_start:
            ref[Segment(seg_start, seg_end)] = speaker

    return ref


def _parse_hyp_annotation(rttm_path: Path, audio_id: str) -> Annotation:
    """Parse pyannote RTTM file into a pyannote Annotation."""
    hyp = Annotation(uri=audio_id)
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            onset = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            end = onset + duration
            if end > onset:
                hyp[Segment(onset, end)] = speaker
    return hyp


def _audio_id_to_session(audio_id: str) -> str:
    raw_path = RAW_ASR_DIR / f"{audio_id}.json"
    if not raw_path.exists():
        return ""
    with open(raw_path) as f:
        meta = json.load(f)
    return meta.get("session_id", "")


def assign_der_tier(der: float) -> str:
    if der <= 0.15:
        return "low"
    elif der <= 0.30:
        return "medium"
    else:
        return "high"


def main():
    metric = DiarizationErrorRate(collar=COLLAR, skip_overlap=False)

    rttm_files = sorted(RTTM_DIR.glob("*.rttm"))
    if not rttm_files:
        print(f"No RTTM files found in {RTTM_DIR}")
        return

    rows = []
    for rttm_path in rttm_files:
        audio_id = rttm_path.stem
        session_id = _audio_id_to_session(audio_id)
        if not session_id:
            print(f"  [skip] {audio_id}: no session_id mapping")
            continue

        ref = _parse_gold_annotation(session_id)
        if ref is None or len(ref) == 0:
            print(f"  [skip] {session_id}: no gold annotations found")
            continue

        hyp = _parse_hyp_annotation(rttm_path, audio_id)
        if len(hyp) == 0:
            print(f"  [skip] {audio_id}: empty hypothesis RTTM")
            continue

        components = metric(ref, hyp, detailed=True)
        der = components["diarization error rate"]
        total_ref = components["total"]
        miss = components["missed detection"]
        fa = components["false alarm"]
        conf = components["confusion"]

        rows.append({
            "audio_id": audio_id,
            "session_id": session_id,
            "der": round(der, 4),
            "der_tier": assign_der_tier(der),
            "total_ref_seconds": round(total_ref, 2),
            "missed_seconds": round(miss, 2),
            "false_alarm_seconds": round(fa, 2),
            "confusion_seconds": round(conf, 2),
            "missed_rate": round(miss / total_ref if total_ref > 0 else 0, 4),
            "false_alarm_rate": round(fa / total_ref if total_ref > 0 else 0, 4),
            "confusion_rate": round(conf / total_ref if total_ref > 0 else 0, 4),
        })
        print(f"  {audio_id} ({session_id}): DER={der:.3f}")

    if not rows:
        print("No DER values computed.")
        return

    df = pd.DataFrame(rows)
    out_path = ANALYSIS_DIR / "der_per_meeting.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    tier_counts = df["der_tier"].value_counts().to_dict()
    print(f"Meetings: {len(df)} | "
          f"low={tier_counts.get('low', 0)}, "
          f"medium={tier_counts.get('medium', 0)}, "
          f"high={tier_counts.get('high', 0)}")
    print(f"DER: min={df['der'].min():.3f}, "
          f"mean={df['der'].mean():.3f}, "
          f"max={df['der'].max():.3f}")


if __name__ == "__main__":
    main()
