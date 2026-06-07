"""
query_extraction.py — revised with section-based extraction for large meetings
and robust JSON parsing.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DIR = PROJECT_ROOT / "data" / "aligned_chunks"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

TRANSCRIPTS_JSON = EVAL_DIR / "usable_transcripts.json"
MANIFEST_CSV = EVAL_DIR / "meeting_manifest.csv"
LABELS_CSV = EVAL_DIR / "retrieval_eval_labels.csv"
QUERIES_CSV = EVAL_DIR / "retrieval_eval_queries.csv"
DIAG_CSV = EVAL_DIR / "retrieval_eval_diagnostics.csv"
CHECKPOINT_DIR = EVAL_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
VERBOSE = os.getenv("VERBOSE", "1") == "1"

# Words per section sent to the LLM — kept small enough for reliable JSON output
SECTION_WORD_LIMIT = int(os.getenv("SECTION_WORD_LIMIT", "1200"))

# Queries to request per section
QUERIES_PER_SECTION = int(os.getenv("QUERIES_PER_SECTION", "3"))

# Total query cap per meeting (across all sections)
MAX_QUERIES_PER_MEETING = int(os.getenv("MAX_QUERIES_PER_MEETING", "12"))

# Preamble words to skip at the start of each meeting
PREAMBLE_SKIP_WORDS = int(os.getenv("PREAMBLE_SKIP_WORDS", "300"))

COSINE_PREFILTER = float(os.getenv("COSINE_PREFILTER", "0.25"))
MAX_CANDIDATES_PER_QUERY = int(os.getenv("MAX_CANDIDATES_PER_QUERY", "35"))
NEGATIVES_PER_QUERY = int(os.getenv("NEGATIVES_PER_QUERY", "5"))

MAX_RETRIES = 3
RETRY_DELAY = 2.0

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "we", "i",
    "you", "he", "she", "they", "it", "this", "that", "to", "of",
    "in", "on", "for", "with", "as", "at", "by", "from", "be", "is",
    "are", "was", "were", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "just", "so",
    "yeah", "okay", "ok", "right", "like", "think", "know", "get",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _call_ollama(prompt: str, temperature: float = 0.2) -> str | None:
    import urllib.request, urllib.error
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "options": {"num_predict": 800},  # cap output tokens — prevents runaway generation
    }
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            return str(data.get("response", "")).strip()
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                if VERBOSE:
                    print(f"    [ollama error: {exc}]")
    return None


def parse_json_array(response: str) -> list[dict]:
    """
    Robustly extract a JSON array from an LLM response that may contain
    extra text, truncated output, or minor syntax errors.
    """
    if not response:
        return []

    # Strip markdown fences
    response = re.sub(r"```(?:json)?|```", "", response).strip()

    # Strategy 1: find a complete [...] block
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            if isinstance(items, list):
                return [i for i in items if isinstance(i, dict)]
        except json.JSONDecodeError:
            pass

    # Strategy 2: the array may be truncated — extract individual objects
    # Find all {...} blocks individually and parse each one
    objects = []
    for obj_match in re.finditer(r"\{[^{}]+\}", response, re.DOTALL):
        try:
            obj = json.loads(obj_match.group())
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    return objects


# ── transcript sectioning ─────────────────────────────────────────────────────

def build_sections(meeting_df: pd.DataFrame) -> list[str]:
    """
    Split the meeting transcript into sections of ~SECTION_WORD_LIMIT words,
    skipping the preamble. Returns a list of transcript strings, one per section.
    """
    # Tolerate both column name conventions
    df = meeting_df.copy()
    if "speaker_label" not in df.columns and "speaker" in df.columns:
        df = df.rename(columns={"speaker": "speaker_label"})

    rows = df.sort_values("start")[["speaker_label", "text"]].values

    # Skip preamble
    skipped = 0
    start_idx = 0
    for i, (_, text) in enumerate(rows):
        skipped += len(str(text).split())
        if skipped >= PREAMBLE_SKIP_WORDS:
            start_idx = i + 1
            break

    rows = rows[start_idx:]

    sections = []
    current_lines = []
    current_words = 0

    for speaker, text in rows:
        text = str(text).strip()
        if not text or len(text.split()) < 3:
            continue
        line = f"{speaker}: {text}"
        word_count = len(text.split())

        if current_words + word_count > SECTION_WORD_LIMIT and current_lines:
            sections.append("\n".join(current_lines))
            current_lines = [line]
            current_words = word_count
        else:
            current_lines.append(line)
            current_words += word_count

    if current_lines:
        sections.append("\n".join(current_lines))

    return sections


# ── Stage 1: query extraction per section ────────────────────────────────────

def extract_queries_from_section(
    audio_id: str,
    section_text: str,
    section_idx: int,
) -> list[dict]:
    """
    Extract up to QUERIES_PER_SECTION queries from one transcript section.
    Uses a short, focused prompt designed to produce clean JSON reliably.
    """
    prompt = f"""Read this excerpt from a meeting transcript and write {QUERIES_PER_SECTION} retrieval questions.

Rules:
- Each question must be answerable from THIS excerpt specifically.
- Questions must be about decisions, design choices, requirements, costs, or factual claims discussed.
- Do NOT write questions about greetings, PowerPoint problems, or filler talk.
- Do NOT write vague questions like "What did the team discuss?"
- Each question must be specific enough that only 1-3 passages would answer it.

Return ONLY a JSON array. Nothing else before or after it:
[
  {{"question": "...", "topic": "brief label"}},
  {{"question": "...", "topic": "brief label"}},
  {{"question": "...", "topic": "brief label"}}
]

EXCERPT:
{section_text}

JSON array:"""

    response = _call_ollama(prompt, temperature=0.2)
    items = parse_json_array(response or "")

    results = []
    for item in items:
        q = str(item.get("question", "")).strip()
        t = str(item.get("topic", "")).strip()
        if q and "?" in q and len(q.split()) >= 5:
            results.append({"query_text": q, "topic_summary": t})

    return results


def extract_queries(audio_id: str, meeting_df: pd.DataFrame) -> list[dict]:
    """
    Extract queries from all sections of a meeting, deduplicate, and
    cap at MAX_QUERIES_PER_MEETING.
    """
    sections = build_sections(meeting_df)
    if VERBOSE:
        print(f"  Sections: {len(sections)}")

    all_queries: list[dict] = []
    seen: set[str] = set()

    for i, section in enumerate(sections):
        if len(all_queries) >= MAX_QUERIES_PER_MEETING:
            break

        queries = extract_queries_from_section(audio_id, section, i)

        for q in queries:
            if len(all_queries) >= MAX_QUERIES_PER_MEETING:
                break
            # Deduplicate by first 6 content tokens
            key = " ".join(content_tokens(q["query_text"])[:6])
            if key not in seen:
                seen.add(key)
                all_queries.append(q)

    return all_queries


# ── Stage 2: relevance judging ────────────────────────────────────────────────

def judge_relevance(query: str, chunk_text: str) -> bool:
    prompt = f"""Does this passage help answer the question?

Question: {query}

Passage: {chunk_text}

Answer YES if the passage directly addresses the question or is part of the relevant discussion.
Answer NO if it is unrelated, filler, or only shares a word or two by coincidence.

One word only — YES or NO:"""

    response = _call_ollama(prompt, temperature=0.0)
    if not response:
        return False
    return response.strip().upper().startswith("YES")


def find_relevant_chunks(query_text: str, meeting_df: pd.DataFrame) -> list[str]:
    query_toks = content_tokens(query_text)

    scored = []
    for row in meeting_df.itertuples(index=False):
        text = str(row.text)
        if len(text.split()) < 4:
            continue
        sim = jaccard(query_toks, content_tokens(text))
        scored.append((sim, str(row.chunk_id), text))

    scored.sort(key=lambda x: x[0], reverse=True)

    above = [x for x in scored if x[0] >= COSINE_PREFILTER]
    candidates = above if len(above) >= 10 else scored[:10]
    candidates = candidates[:MAX_CANDIDATES_PER_QUERY]

    return [
        chunk_id for _, chunk_id, text in candidates
        if judge_relevance(query_text, text)
    ]


# ── negative sampling ─────────────────────────────────────────────────────────

def sample_hard_negatives(
    query_text: str,
    positive_ids: set[str],
    meeting_df: pd.DataFrame,
) -> list[str]:
    query_toks = content_tokens(query_text)
    scored = []
    for row in meeting_df.itertuples(index=False):
        cid = str(row.chunk_id)
        if cid in positive_ids or len(str(row.text).split()) < 4:
            continue
        sim = jaccard(query_toks, content_tokens(str(row.text)))
        scored.append((sim, cid))

    scored.sort(key=lambda x: x[0], reverse=True)
    skip = max(0, len(scored) // 10)
    pool = scored[skip: skip + NEGATIVES_PER_QUERY * 4]
    return [cid for _, cid in pool[:NEGATIVES_PER_QUERY]]


# ── checkpointing ─────────────────────────────────────────────────────────────

def checkpoint_path(audio_id: str) -> Path:
    return CHECKPOINT_DIR / f"{audio_id}.json"


def load_checkpoint(audio_id: str) -> dict | None:
    p = checkpoint_path(audio_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def save_checkpoint(audio_id: str, data: dict) -> None:
    checkpoint_path(audio_id).write_text(json.dumps(data, indent=2))


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class QueryRecord:
    query_id: str
    query_text: str
    topic_summary: str
    audio_id: str
    relevant_chunk_count: int


@dataclass
class LabelRow:
    query_id: str
    query_text: str
    audio_id: str
    chunk_id: str
    relevance: int


# ── per-meeting processing ────────────────────────────────────────────────────

def process_meeting(
    audio_id: str,
    meeting_df: pd.DataFrame,
    query_counter_start: int,
) -> tuple[list[QueryRecord], list[LabelRow], int]:

    cached = load_checkpoint(audio_id)
    if cached:
        if VERBOSE:
            print(f"  ✓ {audio_id}: from checkpoint ({cached['query_count']} queries)")
        return (
            [QueryRecord(**r) for r in cached["query_records"]],
            [LabelRow(**r) for r in cached["label_rows"]],
            cached["next_counter"],
        )

    if VERBOSE:
        print(f"\n── {audio_id} ({len(meeting_df)} chunks) ──")

    queries = extract_queries(audio_id, meeting_df)
    if VERBOSE:
        print(f"  Extracted {len(queries)} queries")

    query_records: list[QueryRecord] = []
    label_rows: list[LabelRow] = []
    counter = query_counter_start

    for q in queries:
        query_id = f"q{counter:04d}"
        counter += 1
        query_text = q["query_text"]

        if VERBOSE:
            print(f"  {query_id}: {query_text[:70]}...")

        relevant_ids = find_relevant_chunks(query_text, meeting_df)
        if VERBOSE:
            print(f"    → {len(relevant_ids)} relevant chunks")

        query_records.append(QueryRecord(
            query_id=query_id,
            query_text=query_text,
            topic_summary=q["topic_summary"],
            audio_id=audio_id,
            relevant_chunk_count=len(relevant_ids),
        ))

        pos_set = set(relevant_ids)
        for cid in relevant_ids:
            label_rows.append(LabelRow(
                query_id=query_id,
                query_text=query_text,
                audio_id=audio_id,
                chunk_id=cid,
                relevance=1,
            ))

        if pos_set:
            for cid in sample_hard_negatives(query_text, pos_set, meeting_df):
                label_rows.append(LabelRow(
                    query_id=query_id,
                    query_text=query_text,
                    audio_id=audio_id,
                    chunk_id=cid,
                    relevance=0,
                ))

    save_checkpoint(audio_id, {
        "query_count": len(query_records),
        "next_counter": counter,
        "query_records": [vars(r) for r in query_records],
        "label_rows": [vars(r) for r in label_rows],
    })

    return query_records, label_rows, counter


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError("Run meeting_filter.py first.")

    manifest_df = pd.read_csv(MANIFEST_CSV)
    usable_ids = set(manifest_df[manifest_df["usable"]]["audio_id"].astype(str))

    paths = sorted(ALIGNED_DIR.glob("*_aligned.csv"))
    chunks_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    if "file_id" in chunks_df.columns and "audio_id" not in chunks_df.columns:
        chunks_df = chunks_df.rename(columns={"file_id": "audio_id"})
    
        # Normalise column names once after loading
    rename_map = {
        "file_id": "audio_id",
        "speaker": "speaker_label",
        "purity": "Purity",
        "mixpenalty": "MixPenalty",
    }
    for src, dst in rename_map.items():
        if src in chunks_df.columns and dst not in chunks_df.columns:
            chunks_df = chunks_df.rename(columns={src: dst})
    
    if "chunk_id" not in chunks_df.columns:
        chunks_df["chunk_id"] = (
            chunks_df["audio_id"].astype(str) + "_"
            + chunks_df["start"].astype(float).round(3).astype(str) + "_"
            + chunks_df["end"].astype(float).round(3).astype(str)
        )
    chunks_df["text"] = chunks_df["text"].astype(str)

    all_query_records: list[QueryRecord] = []
    all_label_rows: list[LabelRow] = []
    query_counter = 1
    stats = {
        "meetings_processed": 0,
        "total_queries": 0,
        "queries_with_positives": 0,
        "total_positive_pairs": 0,
        "total_negative_pairs": 0,
    }

    for audio_id in sorted(usable_ids):
        meeting_df = chunks_df[chunks_df["audio_id"] == audio_id].copy()
        if meeting_df.empty:
            if VERBOSE:
                print(f"  ⚠ No chunks for {audio_id} — skipping")
            continue

        q_records, l_rows, query_counter = process_meeting(
            audio_id, meeting_df, query_counter
        )

        all_query_records.extend(q_records)
        all_label_rows.extend(l_rows)
        stats["meetings_processed"] += 1
        stats["total_queries"] += len(q_records)
        stats["queries_with_positives"] += sum(
            1 for r in q_records if r.relevant_chunk_count > 0
        )
        stats["total_positive_pairs"] += sum(1 for r in l_rows if r.relevance == 1)
        stats["total_negative_pairs"] += sum(1 for r in l_rows if r.relevance == 0)

    pd.DataFrame([vars(r) for r in all_label_rows]).to_csv(LABELS_CSV, index=False)
    pd.DataFrame([vars(r) for r in all_query_records]).to_csv(QUERIES_CSV, index=False)

    with DIAG_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in stats.items():
            writer.writerow([k, v])

    print(f"\n── Extraction complete ──")
    print(f"  Meetings processed:      {stats['meetings_processed']}")
    print(f"  Total queries:           {stats['total_queries']}")
    print(f"  Queries with positives:  {stats['queries_with_positives']}")
    print(f"  Positive pairs:          {stats['total_positive_pairs']}")
    print(f"  Negative pairs:          {stats['total_negative_pairs']}")
    avg = stats["total_positive_pairs"] / max(1, stats["queries_with_positives"])
    print(f"  Avg positives/query:     {avg:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())