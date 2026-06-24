"""
Oracle retrieval system: dense cosine-similarity retrieval over AMI gold transcripts.

Provides the performance ceiling — what retrieval quality would be achievable
with perfect transcription and speaker attribution (no ASR or diarization errors).

Gold transcripts are parsed from data/gold_transcripts/<session>.*.words.xml,
aligned per-speaker and sorted by time, then chunked into fixed-duration windows
(default 10 seconds) to match the granularity of Whisper ASR segments.

Embeddings are computed with sentence-transformers/all-MiniLM-L6-v2 (same model
as all other retrieval systems). Query embeddings are loaded from the medium model
cache — queries are text and do not depend on the ASR model.

Usage:
    python scripts/retrieval_oracle.py [--topk 10] [--window 10]
"""

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT_ROOT / "data" / "gold_transcripts"
RAW_ASR_DIR = PROJECT_ROOT / "data" / "asr_outputs" / "medium" / "raw"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
RESULTS_DIR = PROJECT_ROOT / "data" / "retrieval_results" / "oracle"
SUMMARY_DIR = PROJECT_ROOT / "data" / "metrics" / "oracle"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GAP_TOLERANCE = 0.3  # seconds — merge same-speaker words within this gap


def _audio_id_to_session(audio_id: str) -> str:
    raw_path = RAW_ASR_DIR / f"{audio_id}.json"
    if not raw_path.exists():
        return ""
    with open(raw_path) as f:
        return json.load(f).get("session_id", "")


def _parse_gold_words(session_id: str) -> list[tuple[float, float, str, str]]:
    """
    Parse per-speaker word XMLs → list of (start, end, word, speaker).
    """
    xml_files = sorted(GOLD_DIR.glob(f"{session_id}.*.words.xml"))
    records = []
    for xml_path in xml_files:
        speaker = xml_path.name.split(".")[1]
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        for elem in tree.getroot().iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag != "w":
                continue
            if elem.get("punc", "false") == "true":
                continue
            if not elem.text or not elem.text.strip():
                continue
            try:
                start = float(elem.get("starttime", -1))
                end = float(elem.get("endtime", -1))
            except (TypeError, ValueError):
                continue
            if start < 0 or end < start:
                continue
            records.append((start, end, elem.text.strip(), speaker))
    records.sort(key=lambda x: x[0])
    return records


def _chunk_gold_words(words: list, window_sec: float) -> list[dict]:
    """
    Group words into time-window chunks of ~window_sec seconds.
    Returns list of chunk dicts with text, start, end, dominant_speaker.
    """
    if not words:
        return []
    chunks = []
    win_start = words[0][0]
    current_words = []
    speaker_time: dict[str, float] = {}

    for start, end, word, speaker in words:
        if start - win_start >= window_sec and current_words:
            dominant = max(speaker_time, key=speaker_time.get)
            text = " ".join(w for _, _, w, _ in current_words)
            chunks.append({
                "start": win_start,
                "end": current_words[-1][1],
                "text": text,
                "speaker": dominant,
            })
            win_start = start
            current_words = []
            speaker_time = {}
        current_words.append((start, end, word, speaker))
        speaker_time[speaker] = speaker_time.get(speaker, 0) + max(end - start, 0)

    if current_words:
        dominant = max(speaker_time, key=speaker_time.get)
        text = " ".join(w for _, _, w, _ in current_words)
        chunks.append({
            "start": win_start,
            "end": current_words[-1][1],
            "text": text,
            "speaker": dominant,
        })
    return chunks


def build_oracle_corpus(audio_ids: list[str], window_sec: float) -> pd.DataFrame:
    rows = []
    for audio_id in audio_ids:
        session_id = _audio_id_to_session(audio_id)
        if not session_id:
            continue
        words = _parse_gold_words(session_id)
        chunks = _chunk_gold_words(words, window_sec)
        for c in chunks:
            chunk_id = f"oracle_{audio_id}_{c['start']:.1f}_{c['end']:.1f}"
            rows.append({
                "audio_id": audio_id,
                "session_id": session_id,
                "start": c["start"],
                "end": c["end"],
                "text": c["text"],
                "speaker": c["speaker"],
                "chunk_id": chunk_id,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ndcg_at_k(ranked: list, k: int) -> float:
    k = min(k, len(ranked))
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(ranked[:k]))
    ideal = sorted(ranked, reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _mrr(ranked: list) -> float:
    for i, r in enumerate(ranked):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(ranked: list, k: int) -> float:
    total = sum(1 for r in ranked if r > 0)
    if total == 0:
        return 0.0
    return sum(1 for r in ranked[:k] if r > 0) / total


def _query_metrics(group: pd.DataFrame, k: int) -> dict:
    ranked = group.sort_values("semantic_score", ascending=False)["relevance"].tolist()
    return {
        "ndcg": _ndcg_at_k(ranked, k),
        "mrr": _mrr(ranked),
        "recall": _recall_at_k(ranked, k),
        "precision": sum(1 for r in ranked[:k] if r > 0) / k,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--window", type=float, default=10.0,
                        help="Gold transcript chunk window in seconds (default 10)")
    args = parser.parse_args()

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Oracle | topk={args.topk} window={args.window}s")

    # Load queries and labels
    queries_df = pd.read_csv(EVAL_DIR / "retrieval_eval_queries.csv")
    labels_df = pd.read_csv(EVAL_DIR / "retrieval_eval_labels.csv")
    # Only evaluate queries that have positives
    eval_queries = queries_df[queries_df["difficulty_flag"] != "no_positives"]
    audio_ids = eval_queries["audio_id"].unique().tolist()

    # Build oracle corpus from gold transcripts
    print(f"\nBuilding oracle corpus from {len(audio_ids)} meetings...")
    corpus = build_oracle_corpus(audio_ids, args.window)
    print(f"  Oracle corpus: {len(corpus)} chunks")

    if corpus.empty:
        print("No oracle chunks built — check gold transcript paths.")
        return

    # Embed corpus
    print("\nEmbedding oracle corpus...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    corpus_texts = corpus["text"].tolist()
    corpus_embs = model.encode(corpus_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True)
    corpus_embs = corpus_embs.astype(np.float32)

    # Load query embeddings from medium model cache (queries are text-only)
    cache_path = METRICS_DIR / "chunk_embeddings_minilm_medium.npz"
    if cache_path.exists():
        # Just re-encode queries directly (they're few)
        pass

    query_texts = eval_queries["query_text"].tolist()
    query_ids = eval_queries["query_id"].tolist()
    query_audio = eval_queries["audio_id"].tolist()
    print(f"\nEmbedding {len(query_texts)} queries...")
    query_embs = model.encode(query_texts, batch_size=64, show_progress_bar=False,
                              normalize_embeddings=True).astype(np.float32)

    # Build relevance map from labels
    # Oracle uses text-based chunk IDs so we need meeting-level relevance:
    # a query is relevant to a gold chunk if that chunk's meeting == query's meeting
    # and any positive label row overlaps temporally with the gold chunk.
    labels_pos = labels_df[labels_df["relevance"] == 1].copy()
    # Parse label chunk_id timestamps: audio_id_start_end
    def _parse_chunk_ts(chunk_id: str):
        parts = chunk_id.rsplit("_", 2)
        if len(parts) == 3:
            try:
                return float(parts[1]), float(parts[2])
            except ValueError:
                pass
        return None, None

    labels_pos["lbl_start"] = labels_pos["chunk_id"].apply(lambda x: _parse_chunk_ts(x)[0])
    labels_pos["lbl_end"] = labels_pos["chunk_id"].apply(lambda x: _parse_chunk_ts(x)[1])

    # Build FAISS index per meeting (avoid cross-meeting retrieval)
    print("\nRunning retrieval per meeting...")
    result_rows = []
    per_query_rows = []

    for q_idx, (qid, q_audio) in enumerate(zip(query_ids, query_audio)):
        # Filter corpus to this meeting
        meeting_corpus = corpus[corpus["audio_id"] == q_audio].reset_index(drop=True)
        if meeting_corpus.empty:
            continue

        m_embs = corpus_embs[corpus.index[corpus["audio_id"] == q_audio].tolist()]
        q_emb = query_embs[q_idx].reshape(1, -1)

        # Cosine similarity (embeddings already normalized)
        scores = (m_embs @ q_emb.T).flatten()
        top_n = min(args.topk, len(scores))
        top_idx = np.argsort(scores)[::-1][:top_n]

        # Determine relevance: temporal overlap with any positive label for this query
        pos_for_q = labels_pos[
            (labels_pos["query_id"] == qid) & (labels_pos["audio_id"] == q_audio)
        ]

        for rank, idx in enumerate(top_idx, start=1):
            chunk = meeting_corpus.iloc[idx]
            c_start, c_end = chunk["start"], chunk["end"]

            # relevant if any positive chunk overlaps with this gold chunk
            rel = 0
            for _, lrow in pos_for_q.iterrows():
                if lrow["lbl_start"] is not None:
                    overlap = min(c_end, lrow["lbl_end"]) - max(c_start, lrow["lbl_start"])
                    if overlap > 0:
                        rel = 1
                        break

            result_rows.append({
                "query_id": qid,
                "audio_id": q_audio,
                "chunk_id": chunk["chunk_id"],
                "start": c_start,
                "end": c_end,
                "text": chunk["text"],
                "speaker": chunk["speaker"],
                "semantic_score": float(scores[idx]),
                "rank": rank,
                "relevance": rel,
            })

        # Per-query metrics
        ranked_rel = [r["relevance"] for r in result_rows
                      if r["query_id"] == qid and r["audio_id"] == q_audio]
        ranked_rel = ranked_rel[-top_n:]  # only current query's rows
        per_query_rows.append({
            "query_id": qid,
            "audio_id": q_audio,
            "ndcg": _ndcg_at_k(ranked_rel, args.topk),
            "mrr": _mrr(ranked_rel),
            "recall": _recall_at_k(ranked_rel, args.topk),
            "precision": sum(1 for r in ranked_rel[:args.topk] if r > 0) / args.topk,
        })

    results_df = pd.DataFrame(result_rows)
    pq_df = pd.DataFrame(per_query_rows)

    summary = pd.DataFrame([{
        "pipeline": "oracle",
        "mrr": pq_df["mrr"].mean(),
        "recall": pq_df["recall"].mean(),
        "precision": pq_df["precision"].mean(),
        "ndcg": pq_df["ndcg"].mean(),
        "n_queries": len(pq_df),
        "window_sec": args.window,
    }])

    results_path = RESULTS_DIR / f"results_oracle_{tag}.csv"
    pq_path = SUMMARY_DIR / f"per_query_oracle_{tag}.csv"
    summary_path = SUMMARY_DIR / f"summary_oracle_{tag}.csv"

    results_df.to_csv(results_path, index=False)
    pq_df.to_csv(pq_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*50}")
    print(f"Oracle results — top-{args.topk}, window={args.window}s")
    print(f"{'='*50}")
    print(summary[["pipeline", "ndcg", "mrr", "recall", "precision"]].to_string(index=False))
    print(f"\nSaved:\n  {results_path}\n  {pq_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
