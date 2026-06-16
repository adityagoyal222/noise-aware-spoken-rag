"""
Compute per-meeting Word Error Rate (WER) for each Whisper model size.

For each ASR model:
  1. Load the plain-text Whisper transcript (data/asr_outputs/<model>/transcripts/<audio_id>.txt)
  2. Derive the AMI session_id from the raw ASR JSON (e.g. ES2015c)
  3. Parse all per-speaker word XML files in data/gold_transcripts/ for that session
  4. Normalise both hypothesis and reference (lowercase, strip punctuation)
  5. Compute WER with jiwer
  6. Write data/analysis/wer_per_meeting_<model>.csv

Usage:
    python scripts/compute_wer.py                   # all models
    python scripts/compute_wer.py --model medium    # single model
"""

import argparse
import glob
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import jiwer
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASR_DIR = PROJECT_ROOT / "data" / "asr_outputs"
GOLD_DIR = PROJECT_ROOT / "data" / "gold_transcripts"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def _parse_gold_words(session_id: str) -> str:
    """
    Parse all per-speaker word XML files for a session and return a single
    concatenated word string in temporal order.
    """
    xml_files = sorted(GOLD_DIR.glob(f"{session_id}.*.words.xml"))
    if not xml_files:
        return ""

    words = []
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()
        ns = {"nite": "http://nite.sourceforge.net/"}
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "w":
                punc = elem.get("{http://nite.sourceforge.net/}id", "")
                is_punc = elem.get("punc", "false") == "true"
                if not is_punc and elem.text and elem.text.strip():
                    start = float(elem.get("starttime", 0))
                    words.append((start, elem.text.strip()))

    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer_for_model(model: str) -> pd.DataFrame:
    raw_dir = ASR_DIR / model / "raw"
    transcript_dir = ASR_DIR / model / "transcripts"

    if not raw_dir.exists():
        print(f"  [skip] {model}: raw dir not found")
        return pd.DataFrame()

    rows = []
    raw_files = sorted(raw_dir.glob("*.json"))

    for raw_path in raw_files:
        audio_id = raw_path.stem

        with open(raw_path) as f:
            meta = json.load(f)

        session_id = meta.get("session_id", "")
        if not session_id:
            continue

        transcript_path = transcript_dir / f"{audio_id}.txt"
        if not transcript_path.exists():
            print(f"  [skip] {audio_id}: transcript not found")
            continue

        hypothesis = _normalize(transcript_path.read_text(encoding="utf-8"))
        reference = _normalize(_parse_gold_words(session_id))

        if not reference:
            print(f"  [skip] {session_id}: no gold words found in {GOLD_DIR}")
            continue
        if not hypothesis:
            print(f"  [skip] {audio_id}: empty hypothesis")
            continue

        out = jiwer.process_words(reference, hypothesis)
        rows.append({
            "audio_id": audio_id,
            "session_id": session_id,
            "model": model,
            "wer": round(out.wer, 4),
            "mer": round(out.mer, 4),
            "num_words_ref": out.substitutions + out.deletions + out.hits,
            "num_substitutions": out.substitutions,
            "num_deletions": out.deletions,
            "num_insertions": out.insertions,
            "num_hits": out.hits,
        })
        print(f"  {audio_id} ({session_id}): WER={out.wer:.3f}")

    return pd.DataFrame(rows)


def assign_wer_tier(wer: float) -> str:
    if wer <= 0.15:
        return "low"
    elif wer <= 0.30:
        return "medium"
    else:
        return "high"


def _cap_wer(wer: float) -> float:
    """Cap WER at 1.0 for display/stratification — WER > 1.0 means more insertions
    than reference words (Whisper hallucination), which still belongs to 'high' tier."""
    return min(wer, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Compute per-meeting WER")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to process (default: all)")
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS

    all_rows = []
    for model in models:
        print(f"\n=== {model} ===")
        df = compute_wer_for_model(model)
        if df.empty:
            continue

        df["wer_capped"] = df["wer"].apply(_cap_wer)
        df["wer_tier"] = df["wer_capped"].apply(assign_wer_tier)

        out_path = ANALYSIS_DIR / f"wer_per_meeting_{model}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

        tier_counts = df["wer_tier"].value_counts().to_dict()
        print(f"  Meetings: {len(df)} | "
              f"low={tier_counts.get('low',0)}, "
              f"medium={tier_counts.get('medium',0)}, "
              f"high={tier_counts.get('high',0)}")
        print(f"  WER: min={df['wer'].min():.3f}, "
              f"mean={df['wer'].mean():.3f}, "
              f"max={df['wer'].max():.3f}")

        all_rows.append(df)

    if len(all_rows) > 1:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_path = ANALYSIS_DIR / "wer_per_meeting_all_models.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined saved: {combined_path}")

        print("\n=== Summary across models ===")
        summary = combined.groupby("model")["wer"].agg(["mean", "min", "max"]).round(3)
        print(summary.to_string())


if __name__ == "__main__":
    main()
