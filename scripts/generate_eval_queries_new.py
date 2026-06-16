"""
generate_eval_queries_new.py — Synthetic evaluation dataset construction.

Generates retrieval queries from aligned ASR-diarization chunks, labels
relevant chunks, generates reference answers, and applies quality filters
to produce a clean benchmark dataset.

Quality pipeline per query:
  1. Chunk pre-filter (ASRConf, TurnComp, MixPenalty, length)
  2. LLM query generation (specific, non-trivial prompt)
  3. LLM relevance judging (per-chunk YES/NO)
  4. Post-hoc semantic scan for cross-window missed positives
  5. Cross-encoder re-scoring of all labeled pairs
  6. Reference answer generation (extraction-only prompt, speaker-attributed)
  7. NLI faithfulness check on reference answer
  8. Lexical overlap difficulty flag
  9. BM25 difficulty floor flag

Usage:
    python scripts/generate_eval_queries_new.py
    python scripts/generate_eval_queries_new.py --clear-checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load HF_TOKEN from .env if present
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

_HF_TOKEN = os.getenv("HF_TOKEN")
if _HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _HF_TOKEN
    os.environ["HF_TOKEN"] = _HF_TOKEN

_DEFAULT_MODEL = os.getenv("ASR_MODEL", "medium")
ALIGNED_DIR = PROJECT_ROOT / "data" / "aligned_chunks" / _DEFAULT_MODEL
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = EVAL_DIR / "meeting_manifest.csv"
LABELS_CSV = EVAL_DIR / "retrieval_eval_labels.csv"
QUERIES_CSV = EVAL_DIR / "retrieval_eval_queries.csv"
DIAG_CSV = EVAL_DIR / "retrieval_eval_diagnostics.csv"
CHECKPOINT_DIR = EVAL_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
VERBOSE = os.getenv("VERBOSE", "1") == "1"

# ── section / query generation settings ──────────────────────────────────────
SECTION_WORD_LIMIT = int(os.getenv("SECTION_WORD_LIMIT", "1200"))
QUERIES_PER_SECTION = int(os.getenv("QUERIES_PER_SECTION", "3"))
MAX_QUERIES_PER_MEETING = int(os.getenv("MAX_QUERIES_PER_MEETING", "12"))
PREAMBLE_SKIP_WORDS = int(os.getenv("PREAMBLE_SKIP_WORDS", "300"))
REFERENCE_ANSWER_MAX_CHUNKS = int(os.getenv("REFERENCE_ANSWER_MAX_CHUNKS", "3"))

# ── chunk quality pre-filter thresholds ──────────────────────────────────────
CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "8"))
CHUNK_ASR_CONF_MIN = float(os.getenv("CHUNK_ASR_CONF_MIN", "0.35"))
CHUNK_TURN_COMP_MIN = float(os.getenv("CHUNK_TURN_COMP_MIN", "0.3"))
CHUNK_MIX_PENALTY_MAX = float(os.getenv("CHUNK_MIX_PENALTY_MAX", "0.6"))

# ── relevance candidate selection ─────────────────────────────────────────────
JACCARD_PREFILTER = float(os.getenv("JACCARD_PREFILTER", "0.25"))
MAX_CANDIDATES_PER_QUERY = int(os.getenv("MAX_CANDIDATES_PER_QUERY", "35"))
NEGATIVES_PER_QUERY = int(os.getenv("NEGATIVES_PER_QUERY", "5"))

# ── post-hoc semantic scan ────────────────────────────────────────────────────
SEMANTIC_SCAN_THRESHOLD = float(os.getenv("SEMANTIC_SCAN_THRESHOLD", "0.60"))

# ── quality filter thresholds ─────────────────────────────────────────────────
TRIVIAL_OVERLAP_THRESHOLD = float(os.getenv("TRIVIAL_OVERLAP_THRESHOLD", "0.5"))
CE_SUSPECT_THRESHOLD = float(os.getenv("CE_SUSPECT_THRESHOLD", "0.2"))

MAX_RETRIES = 3
RETRY_DELAY = 2.0

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
NLI_ENTAILMENT_LABEL = "entailment"

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "we", "i",
    "you", "he", "she", "they", "it", "this", "that", "to", "of",
    "in", "on", "for", "with", "as", "at", "by", "from", "be", "is",
    "are", "was", "were", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "just", "so",
    "yeah", "okay", "ok", "right", "like", "think", "know", "get",
}


# ── text utilities ────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def rouge1_recall(hypothesis: list[str], reference: list[str]) -> float:
    """Fraction of reference tokens that appear in hypothesis."""
    if not reference:
        return 0.0
    ref_set = set(reference)
    hits = sum(1 for t in hypothesis if t in ref_set)
    return hits / len(ref_set)


# ── Ollama ────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, temperature: float = 0.2) -> str | None:
    import urllib.request, urllib.error
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "options": {"num_predict": 800},
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
    if not response:
        return []
    response = re.sub(r"```(?:json)?|```", "", response).strip()
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            if isinstance(items, list):
                return [i for i in items if isinstance(i, dict)]
        except json.JSONDecodeError:
            pass
    objects = []
    for obj_match in re.finditer(r"\{[^{}]+\}", response, re.DOTALL):
        try:
            obj = json.loads(obj_match.group())
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    return objects


# ── ML model loading ──────────────────────────────────────────────────────────

def load_models() -> tuple:
    """Load embedding model, cross-encoder, and NLI model once at startup."""
    import torch
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
    except ImportError as exc:
        raise ImportError("sentence-transformers is not installed.") from exc

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading ML models on {device} ...")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    ce_model = CrossEncoder(CE_MODEL_NAME, device=device)
    nli_model = CrossEncoder(NLI_MODEL_NAME, device=device)

    print("  Models loaded.")
    return embed_model, ce_model, nli_model


# ── chunk quality filter ──────────────────────────────────────────────────────

def is_quality_chunk(row) -> bool:
    """Return True if the chunk passes quality thresholds."""
    text = str(getattr(row, "text", ""))
    if len(text.split()) < CHUNK_MIN_WORDS:
        return False
    asr_conf = float(getattr(row, "ASRConf", 1.0))
    turn_comp = float(getattr(row, "TurnComp", 1.0))
    mix_penalty = float(getattr(row, "MixPenalty", 0.0))
    if asr_conf < CHUNK_ASR_CONF_MIN:
        return False
    if turn_comp < CHUNK_TURN_COMP_MIN:
        return False
    if mix_penalty > CHUNK_MIX_PENALTY_MAX:
        return False
    return True


# ── transcript sectioning ─────────────────────────────────────────────────────

def build_sections(meeting_df: pd.DataFrame) -> list[str]:
    df = meeting_df.copy()
    if "speaker_label" not in df.columns and "speaker" in df.columns:
        df = df.rename(columns={"speaker": "speaker_label"})

    rows = df.sort_values("start")[["speaker_label", "text", "ASRConf", "TurnComp", "MixPenalty"]].values

    # Skip preamble
    skipped = 0
    start_idx = 0
    for i, row in enumerate(rows):
        skipped += len(str(row[1]).split())
        if skipped >= PREAMBLE_SKIP_WORDS:
            start_idx = i + 1
            break
    rows = rows[start_idx:]

    sections = []
    current_lines = []
    current_words = 0

    for speaker, text, asr_conf, turn_comp, mix_penalty in rows:
        text = str(text).strip()
        if not text or len(text.split()) < 3:
            continue
        # Only include quality chunks in the section text shown to LLM
        if (float(asr_conf) < CHUNK_ASR_CONF_MIN or
                float(turn_comp) < CHUNK_TURN_COMP_MIN or
                float(mix_penalty) > CHUNK_MIX_PENALTY_MAX):
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


# ── Stage 1: query extraction ─────────────────────────────────────────────────

def extract_queries_from_section(section_text: str) -> list[dict]:
    prompt = f"""Read this excerpt from a meeting transcript and write {QUERIES_PER_SECTION} retrieval questions.

Rules:
- Each question must require reading this transcript to answer — it must NOT be answerable from general knowledge alone.
- Questions must target a specific decision, outcome, constraint, cost estimate, assignment, or factual claim made in the discussion.
- Do NOT write yes/no questions.
- Do NOT write questions that are paraphrases of a single sentence (e.g. "What did X say about Y?" where Y is mentioned only once).
- Do NOT write vague questions like "What did the team discuss?" or "What is the remote control?"
- Each question must be specific enough that only 1–3 passages in the entire meeting could answer it.
- The answer must not be guessable without reading the transcript.

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
        if q and "?" in q and len(q.split()) >= 6:
            results.append({"query_text": q, "topic_summary": t})
    return results


def extract_queries(meeting_df: pd.DataFrame) -> list[dict]:
    sections = build_sections(meeting_df)
    if VERBOSE:
        print(f"  Sections: {len(sections)}")

    all_queries: list[dict] = []
    seen: set[str] = set()

    for section in sections:
        if len(all_queries) >= MAX_QUERIES_PER_MEETING:
            break
        for q in extract_queries_from_section(section):
            if len(all_queries) >= MAX_QUERIES_PER_MEETING:
                break
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


def find_relevant_chunks(
    query_text: str,
    meeting_df: pd.DataFrame,
    embed_model,
) -> tuple[list[str], list[str]]:
    """
    Returns (llm_found_ids, semantic_found_ids).
    llm_found_ids: chunks labeled relevant by LLM from Jaccard-filtered candidates.
    semantic_found_ids: additional positives found by post-hoc semantic scan.
    """
    query_toks = content_tokens(query_text)

    # Only consider quality chunks
    quality_df = meeting_df[[is_quality_chunk(r) for r in meeting_df.itertuples(index=False)]]

    scored = []
    for row in quality_df.itertuples(index=False):
        text = str(row.text)
        sim = jaccard(query_toks, content_tokens(text))
        scored.append((sim, str(row.chunk_id), text))

    scored.sort(key=lambda x: x[0], reverse=True)
    above = [x for x in scored if x[0] >= JACCARD_PREFILTER]
    candidates = above if len(above) >= 10 else scored[:10]
    candidates = candidates[:MAX_CANDIDATES_PER_QUERY]

    llm_found = [
        chunk_id for _, chunk_id, text in candidates
        if judge_relevance(query_text, text)
    ]
    llm_set = set(llm_found)

    # Post-hoc semantic scan across ALL quality chunks in the meeting
    semantic_found = []
    if embed_model is not None:
        all_chunk_ids = quality_df["chunk_id"].astype(str).tolist()
        all_texts = quality_df["text"].astype(str).tolist()

        if all_texts:
            q_vec = embed_model.encode([query_text], normalize_embeddings=True)
            c_vecs = embed_model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
            sims = (c_vecs @ q_vec.T).flatten()

            for cid, sim in zip(all_chunk_ids, sims):
                if float(sim) >= SEMANTIC_SCAN_THRESHOLD and cid not in llm_set:
                    chunk_text = quality_df.loc[quality_df["chunk_id"] == cid, "text"].iloc[0]
                    if judge_relevance(query_text, str(chunk_text)):
                        semantic_found.append(cid)

    return llm_found, semantic_found


# ── Stage 3: reference answer generation ────────────────────────────────────

def generate_reference_answer(
    query: str,
    relevant_rows: list[dict],
) -> str:
    """
    Generate a reference answer using speaker-attributed chunk texts.
    relevant_rows: list of dicts with keys 'text' and 'speaker_label'.
    """
    if not relevant_rows:
        return ""

    context_parts = []
    for i, row in enumerate(relevant_rows[:REFERENCE_ANSWER_MAX_CHUNKS]):
        speaker = row.get("speaker_label", "SPEAKER")
        text = str(row.get("text", "")).strip()
        context_parts.append(f"[Passage {i+1}] [{speaker}]: {text}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are given a question and relevant passages from a meeting transcript.

Question: {query}

Passages:
{context}

Answer using only information explicitly stated in the provided passages.
- Use 1–2 sentences maximum.
- Quote key terms from the passage where helpful.
- Do NOT infer, generalize, or add information not present in the passages.
- Do NOT start with "The answer is" or similar preamble.
- If the passages do not contain sufficient information to answer, return exactly: UNANSWERABLE

Answer:"""

    response = _call_ollama(prompt, temperature=0.0)
    if not response:
        return ""
    response = response.strip()

    # Strip trailing LLM commentary lines (parenthetical notes, passage references, etc.)
    clean_lines = []
    for line in response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if (lower.startswith("note:") or lower.startswith("(note") or
                lower.startswith("(passage") or lower.startswith("[note") or
                lower.startswith("(this") or lower.startswith("(speaker") or
                lower.startswith("i don't see") or lower.startswith("i'm ready") or
                lower.startswith("please provide")):
            break  # stop at first commentary line
        clean_lines.append(stripped)
    response = " ".join(clean_lines).strip()

    # Reject if UNANSWERABLE appears anywhere in the response, or too short
    if "UNANSWERABLE" in response.upper() or len(response.split()) < 5:
        return ""
    return response


# ── Stage 4: NLI faithfulness check ──────────────────────────────────────────

def check_answer_faithful(
    answer: str,
    relevant_texts: list[str],
    nli_model,
) -> bool:
    """Return True if any relevant chunk entails the answer."""
    if not answer or not relevant_texts or nli_model is None:
        return True  # default to faithful if we can't check
    pairs = [(text, answer) for text in relevant_texts[:REFERENCE_ANSWER_MAX_CHUNKS]]
    try:
        scores = nli_model.predict(pairs)
        labels = nli_model.config.id2label
        for score_vec in scores:
            # scores may be a single float (binary) or array (multi-class)
            if hasattr(score_vec, "__len__"):
                label_idx = int(np.argmax(score_vec))
                label = labels.get(label_idx, "").lower()
            else:
                label = NLI_ENTAILMENT_LABEL if float(score_vec) > 0.5 else "not_entailment"
            if label == NLI_ENTAILMENT_LABEL:
                return True
        return False
    except Exception:
        return True


# ── Stage 5: cross-encoder re-scoring ────────────────────────────────────────

def score_labels_with_ce(
    query_text: str,
    label_chunk_ids: list[str],
    meeting_df: pd.DataFrame,
    ce_model,
) -> dict[str, float]:
    """Return {chunk_id: ce_score} for all labeled chunks."""
    if ce_model is None or not label_chunk_ids:
        return {cid: 0.0 for cid in label_chunk_ids}
    id_to_text = dict(zip(
        meeting_df["chunk_id"].astype(str),
        meeting_df["text"].astype(str),
    ))
    pairs = [(query_text, id_to_text.get(cid, "")) for cid in label_chunk_ids]
    try:
        raw_scores = ce_model.predict(pairs)
        return {cid: float(s) for cid, s in zip(label_chunk_ids, raw_scores)}
    except Exception:
        return {cid: 0.0 for cid in label_chunk_ids}


# ── Stage 6: difficulty flags ─────────────────────────────────────────────────

def compute_difficulty_flag(
    query_text: str,
    positive_ids: list[str],
    meeting_df: pd.DataFrame,
) -> str:
    if not positive_ids:
        return "no_positives"

    query_toks = content_tokens(query_text)
    id_to_text = dict(zip(
        meeting_df["chunk_id"].astype(str),
        meeting_df["text"].astype(str),
    ))

    # Lexical overlap check
    max_rouge = 0.0
    for cid in positive_ids:
        chunk_toks = content_tokens(id_to_text.get(cid, ""))
        r1 = rouge1_recall(query_toks, chunk_toks)
        max_rouge = max(max_rouge, r1)
    if max_rouge > TRIVIAL_OVERLAP_THRESHOLD:
        return "trivial_overlap"

    # BM25 difficulty floor
    try:
        from rank_bm25 import BM25Okapi
        all_ids = meeting_df["chunk_id"].astype(str).tolist()
        all_texts = meeting_df["text"].astype(str).tolist()
        corpus = [content_tokens(t) for t in all_texts]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_toks)
        top_idx = int(np.argmax(scores))
        top_id = all_ids[top_idx]
        if top_id in set(positive_ids):
            return "bm25_trivial"
    except ImportError:
        pass  # rank_bm25 not installed, skip BM25 check

    return "ok"


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


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class QueryRecord:
    query_id: str
    query_text: str
    topic_summary: str
    audio_id: str
    relevant_chunk_count: int
    reference_answer: str = ""
    difficulty_flag: str = "ok"       # "ok" | "trivial_overlap" | "bm25_trivial" | "no_positives"
    answer_faithful: bool = True


@dataclass
class LabelRow:
    query_id: str
    query_text: str
    audio_id: str
    chunk_id: str
    relevance: int
    ce_score: float = 0.0
    label_quality: str = "ok"         # "ok" | "suspect"
    found_by: str = "llm"             # "llm" | "semantic_scan"


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


def _load_query_record(r: dict) -> QueryRecord:
    return QueryRecord(
        query_id=r["query_id"],
        query_text=r["query_text"],
        topic_summary=r.get("topic_summary", ""),
        audio_id=r["audio_id"],
        relevant_chunk_count=r.get("relevant_chunk_count", 0),
        reference_answer=r.get("reference_answer", ""),
        difficulty_flag=r.get("difficulty_flag", "ok"),
        answer_faithful=r.get("answer_faithful", True),
    )


def _load_label_row(r: dict) -> LabelRow:
    return LabelRow(
        query_id=r["query_id"],
        query_text=r["query_text"],
        audio_id=r["audio_id"],
        chunk_id=r["chunk_id"],
        relevance=r.get("relevance", 1),
        ce_score=r.get("ce_score", 0.0),
        label_quality=r.get("label_quality", "ok"),
        found_by=r.get("found_by", "llm"),
    )


# ── per-meeting processing ────────────────────────────────────────────────────

def process_meeting(
    audio_id: str,
    meeting_df: pd.DataFrame,
    query_counter_start: int,
    embed_model,
    ce_model,
    nli_model,
) -> tuple[list[QueryRecord], list[LabelRow], int]:

    cached = load_checkpoint(audio_id)
    if cached:
        if VERBOSE:
            print(f"  ✓ {audio_id}: from checkpoint ({cached['query_count']} queries)")
        return (
            [_load_query_record(r) for r in cached["query_records"]],
            [_load_label_row(r) for r in cached["label_rows"]],
            cached["next_counter"],
        )

    if VERBOSE:
        print(f"\n── {audio_id} ({len(meeting_df)} chunks) ──")

    queries = extract_queries(meeting_df)
    if VERBOSE:
        print(f"  Extracted {len(queries)} queries")

    id_to_row = {
        str(r.chunk_id): {"text": str(r.text), "speaker_label": str(getattr(r, "speaker_label", "SPEAKER"))}
        for r in meeting_df.itertuples(index=False)
    }

    query_records: list[QueryRecord] = []
    label_rows: list[LabelRow] = []
    counter = query_counter_start

    for q in queries:
        query_id = f"q{counter:04d}"
        counter += 1
        query_text = q["query_text"]

        if VERBOSE:
            print(f"  {query_id}: {query_text[:70]}...")

        # Stage 2: find relevant chunks (LLM + semantic scan)
        llm_ids, semantic_ids = find_relevant_chunks(query_text, meeting_df, embed_model)
        all_positive_ids = llm_ids + semantic_ids
        pos_set = set(all_positive_ids)

        if VERBOSE:
            print(f"    → {len(llm_ids)} LLM + {len(semantic_ids)} semantic = {len(pos_set)} positives")

        # Stage 4: cross-encoder re-scoring
        ce_scores = score_labels_with_ce(query_text, all_positive_ids, meeting_df, ce_model)

        # Stage 3: reference answer generation (speaker-attributed)
        relevant_rows_data = [id_to_row[cid] for cid in all_positive_ids if cid in id_to_row]
        reference_answer = generate_reference_answer(query_text, relevant_rows_data)
        if VERBOSE:
            print(f"    → answer: {reference_answer[:80]}..." if reference_answer else "    → answer: (empty/unanswerable)")

        # Stage 5: NLI faithfulness check
        relevant_texts = [d["text"] for d in relevant_rows_data]
        answer_faithful = check_answer_faithful(reference_answer, relevant_texts, nli_model)

        # Stage 6: difficulty flag
        difficulty_flag = compute_difficulty_flag(query_text, all_positive_ids, meeting_df)

        if VERBOSE and difficulty_flag != "ok":
            print(f"    → difficulty_flag: {difficulty_flag}")

        query_records.append(QueryRecord(
            query_id=query_id,
            query_text=query_text,
            topic_summary=q["topic_summary"],
            audio_id=audio_id,
            relevant_chunk_count=len(pos_set),
            reference_answer=reference_answer,
            difficulty_flag=difficulty_flag,
            answer_faithful=answer_faithful,
        ))

        # Build label rows for positives
        for cid in llm_ids:
            ce_score = ce_scores.get(cid, 0.0)
            label_rows.append(LabelRow(
                query_id=query_id,
                query_text=query_text,
                audio_id=audio_id,
                chunk_id=cid,
                relevance=1,
                ce_score=ce_score,
                label_quality="suspect" if ce_score < CE_SUSPECT_THRESHOLD else "ok",
                found_by="llm",
            ))
        for cid in semantic_ids:
            ce_score = ce_scores.get(cid, 0.0)
            label_rows.append(LabelRow(
                query_id=query_id,
                query_text=query_text,
                audio_id=audio_id,
                chunk_id=cid,
                relevance=1,
                ce_score=ce_score,
                label_quality="suspect" if ce_score < CE_SUSPECT_THRESHOLD else "ok",
                found_by="semantic_scan",
            ))

        # Hard negatives
        if pos_set:
            for cid in sample_hard_negatives(query_text, pos_set, meeting_df):
                label_rows.append(LabelRow(
                    query_id=query_id,
                    query_text=query_text,
                    audio_id=audio_id,
                    chunk_id=cid,
                    relevance=0,
                    ce_score=0.0,
                    label_quality="ok",
                    found_by="llm",
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
    parser = argparse.ArgumentParser(
        description="Generate retrieval evaluation queries with quality filters."
    )
    parser.add_argument(
        "--clear-checkpoints", action="store_true",
        help="Delete all existing checkpoints before running (forces full regeneration).",
    )
    args = parser.parse_args()

    if args.clear_checkpoints:
        existing = list(CHECKPOINT_DIR.glob("*.json"))
        if existing:
            print(f"Deleting {len(existing)} checkpoint(s) from {CHECKPOINT_DIR} ...")
            for p in existing:
                p.unlink()
        else:
            print("No checkpoints to delete.")

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_CSV}. Run meeting_filter.py first.")

    manifest_df = pd.read_csv(MANIFEST_CSV)
    usable_ids = set(manifest_df[manifest_df["usable"]]["audio_id"].astype(str))

    paths = sorted(ALIGNED_DIR.glob("*_aligned.csv"))
    if not paths:
        raise FileNotFoundError(f"No aligned CSVs in {ALIGNED_DIR}")
    chunks_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

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

    for col, default in [("ASRConf", 1.0), ("TurnComp", 1.0), ("MixPenalty", 0.0)]:
        if col not in chunks_df.columns:
            chunks_df[col] = default
        chunks_df[col] = pd.to_numeric(chunks_df[col], errors="coerce").fillna(default)

    # Load all ML models once
    embed_model, ce_model, nli_model = load_models()

    all_query_records: list[QueryRecord] = []
    all_label_rows: list[LabelRow] = []
    query_counter = 1
    stats: dict[str, int] = {
        "meetings_processed": 0,
        "total_queries": 0,
        "queries_with_positives": 0,
        "trivial_overlap": 0,
        "bm25_trivial": 0,
        "unfaithful_answers": 0,
        "suspect_labels": 0,
        "semantic_scan_found": 0,
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
            audio_id, meeting_df, query_counter,
            embed_model, ce_model, nli_model,
        )

        all_query_records.extend(q_records)
        all_label_rows.extend(l_rows)
        stats["meetings_processed"] += 1
        stats["total_queries"] += len(q_records)
        stats["queries_with_positives"] += sum(1 for r in q_records if r.relevant_chunk_count > 0)
        stats["trivial_overlap"] += sum(1 for r in q_records if r.difficulty_flag == "trivial_overlap")
        stats["bm25_trivial"] += sum(1 for r in q_records if r.difficulty_flag == "bm25_trivial")
        stats["unfaithful_answers"] += sum(1 for r in q_records if not r.answer_faithful)
        stats["suspect_labels"] += sum(1 for r in l_rows if r.label_quality == "suspect")
        stats["semantic_scan_found"] += sum(1 for r in l_rows if r.found_by == "semantic_scan")
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
    print(f"  Trivial overlap:         {stats['trivial_overlap']}")
    print(f"  BM25 trivial:            {stats['bm25_trivial']}")
    print(f"  Unfaithful answers:      {stats['unfaithful_answers']}")
    print(f"  Suspect labels (CE):     {stats['suspect_labels']}")
    print(f"  Semantic scan found:     {stats['semantic_scan_found']}")
    print(f"  Positive pairs:          {stats['total_positive_pairs']}")
    print(f"  Negative pairs:          {stats['total_negative_pairs']}")
    avg = stats["total_positive_pairs"] / max(1, stats["queries_with_positives"])
    print(f"  Avg positives/query:     {avg:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
