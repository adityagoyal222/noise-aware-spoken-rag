from __future__ import annotations

import csv
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DIR = PROJECT_ROOT / "data" / "aligned_chunks"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

LABELS_CSV = EVAL_DIR / "retrieval_eval_labels.csv"
REVIEW_CSV = EVAL_DIR / "retrieval_eval_manual_review.csv"
DIAG_CSV = EVAL_DIR / "retrieval_eval_diagnostics.csv"
REVIEW_TABLE_CSV = EVAL_DIR / "retrieval_eval_review_table.csv"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

USE_OLLAMA = os.getenv("USE_OLLAMA", "1") == "1"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
VERBOSE = os.getenv("VERBOSE", "1") == "1"
LOG_EVERY = int(os.getenv("LOG_EVERY", "200"))
WINDOW_SEC = float(os.getenv("WINDOW_SEC", "30"))
MIN_LEXICAL_OVERLAP = float(os.getenv("MIN_LEXICAL_OVERLAP", "0.2"))
NEGATIVE_SAMPLES = int(os.getenv("NEGATIVE_SAMPLES", "5"))

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "we",
    "i",
    "you",
    "he",
    "she",
    "they",
    "it",
    "this",
    "that",
    "these",
    "those",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "be",
    "is",
    "are",
    "was",
    "were",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "should",
    "could",
    "can",
    "about",
    "into",
    "over",
    "also",
    "so",
    "just",
    "not",
    "no",
    "yes",
}

EVIDENCE_KEYWORDS = {
    "decide",
    "decided",
    "decision",
    "agree",
    "agreed",
    "action",
    "task",
    "todo",
    "next",
    "step",
    "schedule",
    "timeline",
    "deadline",
    "budget",
    "cost",
    "price",
    "design",
    "feature",
    "issue",
    "problem",
    "risk",
    "plan",
    "proposal",
    "option",
    "tradeoff",
    "choice",
    "assignment",
    "responsible",
    "owner",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")

CUE_PATTERNS: list[tuple[str, str]] = [
    (r"\bwe(?:'ve)?\s+decided(?:\s+that)?\s+(?P<clause>.+)", "decide"),
    (r"\bwe\s+agree(?:d)?\s+(?:to|that)\s+(?P<clause>.+)", "decide"),
    (r"\bthe\s+decision\s+is\s+(?P<clause>.+)", "decide"),
    (r"\bwe\s+plan(?:ned)?\s+(?:to|that)?\s+(?P<clause>.+)", "plan"),
    (r"\bwe\s+need\s+to\s+(?P<clause>.+)", "plan"),
    (r"\bwe\s+should\s+(?P<clause>.+)", "plan"),
    (r"\bwe\s+will\s+(?P<clause>.+)", "plan"),
    (r"\b(let's|lets)\s+(?P<clause>.+)", "plan"),
]

LEADING_FILLERS = {
    "yeah",
    "okay",
    "ok",
    "well",
    "so",
    "and",
    "but",
    "then",
    "right",
}

BAD_CLAUSE_STARTERS = {
    "because",
    "if",
    "that",
    "which",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
}

BAD_CLAUSE_TOKENS = {
    "yeah",
    "okay",
    "ok",
    "uh",
    "um",
    "like",
    "you",
    "know",
}


@dataclass
class EvidenceSpan:
    query_id: str
    query_text: str
    chunk_id: str
    audio_id: str
    start: float
    end: float
    text: str


@dataclass
class MatchResult:
    query_id: str
    query_text: str
    audio_id: str
    span_start: float
    span_end: float
    matched_chunk_ids: list[str]
    excerpt: str
    confidence: str


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def load_chunks(aligned_dir: Path) -> pd.DataFrame:
    paths = sorted(aligned_dir.glob("*_aligned.csv"))
    if not paths:
        raise FileNotFoundError(f"No aligned chunks found in {aligned_dir}")

    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)

    rename_map = {
        "speaker": "speaker_label",
        "file_id": "audio_id",
        "purity": "Purity",
        "mixpenalty": "MixPenalty",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    if "chunk_id" not in df.columns:
        if "audio_id" not in df.columns and "file_id" in df.columns:
            df = df.rename(columns={"file_id": "audio_id"})
        if "audio_id" in df.columns:
            df["chunk_id"] = (
                df["audio_id"].astype(str)
                + "_"
                + df["start"].astype(float).round(3).astype(str)
                + "_"
                + df["end"].astype(float).round(3).astype(str)
            )

    required_cols = [
        "chunk_id",
        "audio_id",
        "start",
        "end",
        "text",
        "speaker_label",
        "Purity",
        "ASRConf",
        "DiarStab",
        "TurnComp",
        "Redund",
        "MixPenalty",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["Purity", "ASRConf", "DiarStab", "TurnComp", "Redund", "MixPenalty"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["text"] = df["text"].astype(str)
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["audio_id"] = df["audio_id"].astype(str)
    return df


def load_transcripts(chunks_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["audio_id", "start", "end", "text", "chunk_id"]
    return chunks_df[cols].sort_values(["audio_id", "start", "end"]).reset_index(drop=True)


def extract_focus_phrase(text: str, max_words: int = 8) -> str | None:
    tokens = [tok for tok in tokenize(text) if tok not in STOPWORDS]
    if len(tokens) < 3:
        return None
    return " ".join(tokens[:max_words])


def _clean_clause(clause: str, max_words: int = 12) -> str | None:
    cleaned = clause.strip().strip("\"'“””).,;:!?")
    cleaned = re.sub(r"\s+", " ", cleaned)

    tokens = cleaned.split()
    while tokens and tokens[0].lower() in LEADING_FILLERS:
        tokens.pop(0)
    while tokens and tokens[0].lower() in STOPWORDS:
        tokens.pop(0)
    if tokens and tokens[0].lower() in BAD_CLAUSE_STARTERS:
        return None

    tokens = tokens[:max_words]
    if len(tokens) < 3:
        return None

    non_stop = [tok for tok in tokens if tok.lower() not in STOPWORDS]
    if len(non_stop) < 2:
        return None

    bad_token_hits = sum(1 for tok in tokens if tok.lower() in BAD_CLAUSE_TOKENS)
    if bad_token_hits >= 2:
        return None

    stop_ratio = sum(1 for tok in tokens if tok.lower() in STOPWORDS) / max(1, len(tokens))
    if stop_ratio > 0.45:
        return None

    return " ".join(tokens).strip("\"'“””).,;:!?")


def extract_query_text(text: str) -> str | None:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return None

    if not has_evidence_keyword(normalized):
        return None

    lowered = normalized.lower()
    for pattern, kind in CUE_PATTERNS:
        match = re.search(pattern, lowered)
        if not match:
            continue
        clause = normalized[match.start("clause") : match.end("clause")]
        phrase = _clean_clause(clause)
        if not phrase:
            continue
        if kind == "decide":
            return f"What did the team decide about {phrase}?"
        return f"What is the plan for {phrase}?"

    return None


def _rewrite_with_ollama(query_text: str, excerpt: str) -> str | None:
    prompt = (
        "Rewrite the question to be clear and grammatical, using only information in the excerpt. "
        "Do not add new facts or nouns. Return a single question ending with a question mark.\n\n"
        f"EXCERPT: {excerpt}\n"
        f"QUESTION: {query_text}\n"
        "REWRITE:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None

    text = str(data.get("response", "")).strip()
    if not text:
        return None
    if not text.endswith("?"):
        text = text.rstrip(".") + "?"
    return text


def _write_with_ollama(excerpt: str) -> str | None:
    prompt = (
        "Write a single, clear question that is answerable only from the excerpt. "
        "Do not add new facts or nouns. Return one question ending with a question mark.\n\n"
        f"EXCERPT: {excerpt}\n"
        "QUESTION:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None

    text = str(data.get("response", "")).strip()
    if not text:
        return None
    if not text.endswith("?"):
        text = text.rstrip(".") + "?"
    return text


def _is_rewrite_safe(rewrite: str, excerpt: str) -> bool:
    rewrite_tokens = [tok for tok in tokenize(rewrite) if tok not in STOPWORDS]
    excerpt_tokens = set(tok for tok in tokenize(excerpt) if tok not in STOPWORDS)
    if len(rewrite_tokens) < 4:
        return False

    if not rewrite.strip().endswith("?"):
        return False

    new_tokens = [tok for tok in rewrite_tokens if tok not in excerpt_tokens]
    if len(new_tokens) > 2:
        return False

    overlap = len([tok for tok in rewrite_tokens if tok in excerpt_tokens])
    if overlap / max(1, len(rewrite_tokens)) < 0.6:
        return False

    return True


def is_high_quality(row: pd.Series) -> bool:
    if row["Purity"] < 0.6:
        return False
    if row["ASRConf"] < 0.4:
        return False
    if row["DiarStab"] < 0.4:
        return False
    if row["TurnComp"] < 0.3:
        return False
    if row["MixPenalty"] > 0.4:
        return False
    token_count = len(tokenize(row["text"]))
    if token_count < 6 or token_count > 40:
        return False
    return True


def has_evidence_keyword(text: str) -> bool:
    tokens = set(tokenize(text))
    return bool(tokens & EVIDENCE_KEYWORDS)


def generate_candidate_queries(chunks_df: pd.DataFrame) -> tuple[list[EvidenceSpan], dict[str, int]]:
    spans: list[EvidenceSpan] = []
    counter = 1
    stats = {
        "candidates": 0,
        "llm_attempted": 0,
        "llm_accepted": 0,
        "llm_rejected": 0,
        "fallback_used": 0,
    }
    scanned = 0

    for row in chunks_df.itertuples(index=False):
        row_dict = row._asdict()
        scanned += 1
        if VERBOSE and LOG_EVERY > 0 and scanned % LOG_EVERY == 0:
            print(
                "Scanned", scanned,
                "| candidates", stats["candidates"],
                "| llm", stats["llm_accepted"],
            )
        if not is_high_quality(pd.Series(row_dict)):
            continue
        query_text = None
        if USE_OLLAMA:
            stats["llm_attempted"] += 1
            drafted = _write_with_ollama(row_dict["text"])
            if drafted and _is_rewrite_safe(drafted, row_dict["text"]):
                query_text = drafted
                stats["llm_accepted"] += 1
            else:
                stats["llm_rejected"] += 1

        if not query_text:
            fallback = extract_query_text(row_dict["text"])
            if not fallback:
                continue
            query_text = fallback
            stats["fallback_used"] += 1
        stats["candidates"] += 1
        query_id = f"q{counter:04d}"
        counter += 1
        spans.append(
            EvidenceSpan(
                query_id=query_id,
                query_text=query_text,
                chunk_id=str(row_dict["chunk_id"]),
                audio_id=str(row_dict["audio_id"]),
                start=float(row_dict["start"]),
                end=float(row_dict["end"]),
                text=str(row_dict["text"]),
            )
        )

    if VERBOSE:
        print(
            "Finished scanning", scanned,
            "| candidates", stats["candidates"],
            "| llm", stats["llm_accepted"],
            "| rejected", stats["llm_rejected"],
            "| fallback", stats["fallback_used"],
        )

    return spans, stats


def find_answer_spans(spans: list[EvidenceSpan]) -> list[EvidenceSpan]:
    return spans


def overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    duration = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if duration == 0.0:
        return 0.0
    return overlap / duration


def map_spans_to_chunks(spans: list[EvidenceSpan], chunks_df: pd.DataFrame) -> list[MatchResult]:
    results: list[MatchResult] = []

    for span in spans:
        window_start = span.start - WINDOW_SEC
        window_end = span.end + WINDOW_SEC
        candidates = chunks_df[
            (chunks_df["audio_id"] == span.audio_id)
            & (chunks_df["start"] <= window_end)
            & (chunks_df["end"] >= window_start)
        ]
        matched: list[str] = []
        for row in candidates.itertuples(index=False):
            text_sim = jaccard(tokenize(span.text), tokenize(str(row.text)))
            if text_sim >= MIN_LEXICAL_OVERLAP:
                matched.append(str(row.chunk_id))

        confidence = "auto" if matched else "skipped"

        results.append(
            MatchResult(
                query_id=span.query_id,
                query_text=span.query_text,
                audio_id=span.audio_id,
                span_start=span.start,
                span_end=span.end,
                matched_chunk_ids=matched,
                excerpt=span.text[:160],
                confidence=confidence,
            )
        )

    return results


def create_label_rows(matches: list[MatchResult], chunks_df: pd.DataFrame) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for match in matches:
        if match.confidence != "auto":
            continue
        pos_ids = set(match.matched_chunk_ids)
        for chunk_id in pos_ids:
            rows.append(
                {
                    "query_id": match.query_id,
                    "query_text": match.query_text,
                    "chunk_id": chunk_id,
                    "relevance": 1,
                }
            )
        window_start = match.span_start - WINDOW_SEC
        window_end = match.span_end + WINDOW_SEC
        negatives = chunks_df[
            (chunks_df["audio_id"] == match.audio_id)
            & ((chunks_df["end"] < window_start) | (chunks_df["start"] > window_end))
            & (~chunks_df["chunk_id"].isin(pos_ids))
        ].head(NEGATIVE_SAMPLES)
        for row in negatives.itertuples(index=False):
            rows.append(
                {
                    "query_id": match.query_id,
                    "query_text": match.query_text,
                    "chunk_id": str(row.chunk_id),
                    "relevance": 0,
                }
            )
    return rows


def create_review_table(matches: list[MatchResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for match in matches:
        if match.confidence == "auto":
            selected = ",".join(match.matched_chunk_ids)
        else:
            selected = ""
        rows.append(
            {
                "query_id": match.query_id,
                "query_text": match.query_text,
                "selected_positive_chunk_ids": selected,
                "excerpt": match.excerpt,
                "status": match.confidence,
            }
        )
    return rows


def save_outputs(
    label_rows: list[dict[str, str | int]],
    review_rows: list[dict[str, str]],
    matches: list[MatchResult],
    stats: dict[str, int],
) -> None:
    labels_df = pd.DataFrame(label_rows)
    review_df = pd.DataFrame(review_rows)

    manual_review = review_df[review_df["status"].isin(["manual_review", "skipped"])]

    labels_df.to_csv(LABELS_CSV, index=False)
    review_df.to_csv(REVIEW_TABLE_CSV, index=False)
    manual_review.to_csv(REVIEW_CSV, index=False)

    auto_count = sum(1 for match in matches if match.confidence == "auto")
    skipped_count = sum(1 for match in matches if match.confidence == "skipped")
    manual_count = sum(1 for match in matches if match.confidence == "manual_review")

    with DIAG_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["auto_labeled", auto_count])
        writer.writerow(["skipped", skipped_count])
        writer.writerow(["manual_review_needed", manual_count])
        writer.writerow(["candidate_queries", stats.get("candidates", 0)])
        writer.writerow(["llm_attempted", stats.get("llm_attempted", 0)])
        writer.writerow(["llm_accepted", stats.get("llm_accepted", 0)])
        writer.writerow(["llm_rejected", stats.get("llm_rejected", 0)])
        writer.writerow(["fallback_used", stats.get("fallback_used", 0)])

    print(f"Auto-labeled queries: {auto_count}")
    print(f"Skipped queries: {skipped_count}")
    print(f"Manual review needed: {manual_count}")
    print(f"Saved labels to: {LABELS_CSV}")
    print(f"Saved review table to: {REVIEW_TABLE_CSV}")
    print(f"Saved manual review list to: {REVIEW_CSV}")
    print(f"Saved diagnostics to: {DIAG_CSV}")


def main() -> int:
    chunks_df = load_chunks(ALIGNED_DIR)
    _ = load_transcripts(chunks_df)
    spans, stats = generate_candidate_queries(chunks_df)
    spans = find_answer_spans(spans)
    matches = map_spans_to_chunks(spans, chunks_df)
    label_rows = create_label_rows(matches, chunks_df)
    review_rows = create_review_table(matches)
    save_outputs(label_rows, review_rows, matches, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
