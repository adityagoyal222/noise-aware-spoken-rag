from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DIR = PROJECT_ROOT / "data" / "aligned_chunks"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = EVAL_DIR / "meeting_manifest.csv"

# ── detection thresholds ──────────────────────────────────────────────────────

# Whisper hallucination: fraction of chunks whose text is the same repeated token
HALLUCINATION_REPEAT_THRESHOLD = 0.3

# Minimum number of unique non-trivial chunks a meeting must have
MIN_SUBSTANTIVE_CHUNKS = 20

# Minimum mean ASRConf across the meeting
MIN_MEAN_ASR_CONF = 0.4

# Words to skip at the start of a meeting before the transcript is usable
# (covers tool-training, PowerPoint loading, introductions)
PREAMBLE_SKIP_WORDS = 400

# Languages to exclude — detected by simple script heuristics
# Japanese katakana/kanji range, Dutch function words
JAPANESE_PATTERN = re.compile(r"[\u3000-\u9fff]")
DUTCH_WORDS = {"de", "het", "een", "van", "en", "in", "dat", "is", "op", "te",
               "zijn", "voor", "met", "aan", "er", "maar", "ook", "als", "bij",
               "niet", "heeft", "kan", "worden", "dit", "door", "zo", "dan",
               "wel", "nog", "wat", "hoe", "wij", "jullie"}
DUTCH_DETECTION_THRESHOLD = 0.15   # fraction of content tokens that are Dutch words


def detect_language(text: str) -> str:
    """Returns 'japanese', 'dutch', or 'english'."""
    if JAPANESE_PATTERN.search(text):
        return "japanese"
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return "unknown"
    dutch_fraction = sum(1 for t in tokens if t in DUTCH_WORDS) / len(tokens)
    if dutch_fraction >= DUTCH_DETECTION_THRESHOLD:
        return "dutch"
    return "english"


def is_hallucination_loop(texts: list[str]) -> bool:
    """
    Detect Whisper repetition loops: a large fraction of chunks contain
    the same repeated substring (e.g. 'I'm sorry I'm sorry ...' or
    Japanese characters repeated many times).
    """
    repeat_pattern = re.compile(r"(.{4,40}?)(\s*\1){4,}")
    loop_count = sum(1 for t in texts if repeat_pattern.search(t))
    return loop_count / max(1, len(texts)) > HALLUCINATION_REPEAT_THRESHOLD


def has_enough_substance(meeting_df: pd.DataFrame) -> bool:
    """
    Check that the meeting has enough real spoken content after skipping
    the preamble.
    """
    # Count chunks with >= 5 meaningful words
    substantive = meeting_df[
        meeting_df["text"].apply(lambda t: len(t.split()) >= 5)
    ]
    return len(substantive) >= MIN_SUBSTANTIVE_CHUNKS


def build_usable_transcript(meeting_df: pd.DataFrame) -> str:
    """
    Return transcript text after skipping the administrative preamble,
    using up to 4000 words of substantive content.
    """
    rows = meeting_df.sort_values("start")[["speaker_label", "text"]].values
    lines = []
    word_count = 0
    skipped_words = 0

    for speaker, text in rows:
        text = str(text).strip()
        if not text:
            continue
        if skipped_words < PREAMBLE_SKIP_WORDS:
            skipped_words += len(text.split())
            continue
        lines.append(f"{speaker}: {text}")
        word_count += len(text.split())
        if word_count >= 4000:
            break

    return "\n".join(lines)


def screen_meeting(audio_id: str, meeting_df: pd.DataFrame) -> dict:
    """
    Run all checks on one meeting. Returns a result dict with a
    'usable' flag and diagnostic fields.
    """
    texts = meeting_df["text"].astype(str).tolist()
    full_text = " ".join(texts)

    result = {
        "audio_id": audio_id,
        "chunk_count": len(meeting_df),
        "usable": False,
        "exclusion_reason": None,
        "language": None,
        "mean_asr_conf": meeting_df["ASRConf"].mean() if "ASRConf" in meeting_df.columns else None,
        "transcript_preview": " ".join(texts[:3])[:120],
    }

    # Check 1: hallucination loop
    if is_hallucination_loop(texts):
        result["exclusion_reason"] = "whisper_hallucination_loop"
        return result

    # Check 2: language
    lang = detect_language(full_text[:2000])
    result["language"] = lang
    if lang in ("japanese", "dutch", "unknown"):
        result["exclusion_reason"] = f"non_english_{lang}"
        return result

    # Check 3: enough substantive content
    if not has_enough_substance(meeting_df):
        result["exclusion_reason"] = "insufficient_content"
        return result

    # Check 4: ASR quality floor
    if result["mean_asr_conf"] is not None and result["mean_asr_conf"] < MIN_MEAN_ASR_CONF:
        result["exclusion_reason"] = "low_asr_confidence"
        return result

    result["usable"] = True
    return result


def main() -> None:
    paths = sorted(ALIGNED_DIR.glob("*_aligned.csv"))
    if not paths:
        raise FileNotFoundError(f"No aligned CSVs in {ALIGNED_DIR}")

    records = []
    usable_transcripts = {}  # audio_id -> transcript string for usable meetings

    for path in paths:
        df = pd.read_csv(path)

        # Normalise column names
        if "file_id" in df.columns and "audio_id" not in df.columns:
            df = df.rename(columns={"file_id": "audio_id"})
        if "speaker" in df.columns and "speaker_label" not in df.columns:
            df = df.rename(columns={"speaker": "speaker_label"})
        if "ASRConf" not in df.columns and "asr_conf" in df.columns:
            df = df.rename(columns={"asr_conf": "ASRConf"})

        for audio_id, meeting_df in df.groupby("audio_id"):
            result = screen_meeting(str(audio_id), meeting_df)
            records.append(result)

            if result["usable"]:
                usable_transcripts[str(audio_id)] = build_usable_transcript(meeting_df)

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(MANIFEST_CSV, index=False)

    usable = manifest_df[manifest_df["usable"]]
    excluded = manifest_df[~manifest_df["usable"]]

    print(f"\n── Meeting screening results ──")
    print(f"  Total meetings:  {len(manifest_df)}")
    print(f"  Usable:          {len(usable)}")
    print(f"  Excluded:        {len(excluded)}")
    print(f"\n  Exclusion breakdown:")
    for reason, count in excluded["exclusion_reason"].value_counts().items():
        print(f"    {reason}: {count}")

    print(f"\n  Usable meeting IDs:")
    for aid in usable["audio_id"].tolist():
        print(f"    {aid}")

    print(f"\n  Manifest saved to: {MANIFEST_CSV}")

    # Save usable transcripts for the query extraction step
    transcripts_path = EVAL_DIR / "usable_transcripts.json"
    import json
    with open(transcripts_path, "w") as f:
        json.dump(usable_transcripts, f, indent=2)
    print(f"  Transcripts saved to: {transcripts_path}")


if __name__ == "__main__":
    main()