"""
retrieval_bm25.py — BM25 lexical retrieval baseline (no reranking).

Builds a BM25Okapi index over aligned chunk texts, retrieves top-K candidates
per query, evaluates against relevance labels, and writes results + metrics to
data/retrieval_results/bm25/.

Usage:
    python scripts/retrieval_bm25.py [--model medium] [--topk 10]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:
    raise ImportError("rank_bm25 is not installed. Run: pip install rank-bm25") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
QUERIES_CSV = EVAL_DIR / "retrieval_eval_queries.csv"
CHECKPOINTS_DIR = EVAL_DIR / "checkpoints"

TOP_K_DEFAULT = 10
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
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# ── data loading ──────────────────────────────────────────────────────────────

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {"speaker": "speaker_label", "file_id": "audio_id"}
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    if "speaker_label" not in df.columns:
        df["speaker_label"] = "UNKNOWN"
    if "audio_id" not in df.columns:
        df["audio_id"] = "UNKNOWN"
    if "chunk_id" not in df.columns:
        df["chunk_id"] = (
            df["audio_id"].astype(str) + "_"
            + df["start"].astype(float).round(3).astype(str) + "_"
            + df["end"].astype(float).round(3).astype(str)
        )
    numeric_defaults = {
        "ASRConf": 0.0, "DiarStab": 0.0, "TurnComp": 0.0,
        "Redund": 0.0, "MixPenalty": 0.0, "Purity": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    if "text" not in df.columns:
        df["text"] = ""
    df["text"] = df["text"].astype(str)
    return df


def load_aligned_chunks(aligned_dir: Path) -> pd.DataFrame:
    paths = sorted(aligned_dir.glob("*_aligned.csv"))
    if not paths:
        raise FileNotFoundError(f"No aligned CSVs found in {aligned_dir}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return _standardize_columns(df)


def load_query_labels(queries_csv: Path, checkpoints_dir: Path) -> pd.DataFrame:
    queries_df = pd.read_csv(queries_csv)
    required = {"query_id", "query_text", "audio_id"}
    missing = required - set(queries_df.columns)
    if missing:
        raise ValueError(f"retrieval_eval_queries.csv missing columns: {sorted(missing)}")

    label_rows = []
    for row in queries_df.itertuples(index=False):
        cp = checkpoints_dir / f"{row.audio_id}.json"
        if not cp.exists():
            continue
        payload = json.loads(cp.read_text())
        for label in payload.get("label_rows", []):
            if label.get("query_id") != row.query_id:
                continue
            label_rows.append({
                "query_id": str(label["query_id"]),
                "query_text": str(label["query_text"]),
                "audio_id": str(row.audio_id),
                "chunk_id": str(label["chunk_id"]),
                "relevance": int(label.get("relevance", 1)),
            })

    if not label_rows:
        raise ValueError("No label rows found in checkpoint files.")

    return pd.DataFrame(label_rows)


# ── BM25 retrieval ────────────────────────────────────────────────────────────

def build_bm25_index(chunks_df: pd.DataFrame) -> tuple[BM25Okapi, list[str]]:
    chunk_ids = chunks_df["chunk_id"].astype(str).tolist()
    corpus = [tokenize(t) for t in chunks_df["text"].astype(str).tolist()]
    bm25 = BM25Okapi(corpus)
    return bm25, chunk_ids


def retrieve_bm25(
    query: str,
    top_k: int,
    bm25: BM25Okapi,
    chunks_df: pd.DataFrame,
    chunk_ids: list[str],
) -> pd.DataFrame:
    query_toks = tokenize(query)
    scores = bm25.get_scores(query_toks)
    top_indices = np.argsort(scores)[::-1][:top_k]

    rows = []
    for rank, idx in enumerate(top_indices, start=1):
        if scores[idx] <= 0.0:
            continue
        chunk_id = chunk_ids[idx]
        match = chunks_df.loc[chunks_df["chunk_id"] == chunk_id]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        row["bm25_score"] = float(scores[idx])
        row["rank"] = rank
        rows.append(row)
    return pd.DataFrame(rows)


# ── evaluation ────────────────────────────────────────────────────────────────

def _query_metrics(relevance: list[int], total_relevant: int) -> dict[str, float]:
    if not relevance:
        return {"mrr": 0.0, "recall": 0.0, "precision": 0.0, "ndcg": 0.0}
    first_hit = next((i + 1 for i, r in enumerate(relevance) if r > 0), None)
    mrr = 1.0 / first_hit if first_hit else 0.0
    hits = sum(1 for r in relevance if r > 0)
    precision = hits / len(relevance)
    recall = hits / total_relevant if total_relevant > 0 else 0.0
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevance))
    ideal_hits = min(total_relevant, len(relevance))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"mrr": mrr, "recall": recall, "precision": precision, "ndcg": ndcg}


def evaluate(
    labels_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    bm25: BM25Okapi,
    chunk_ids: list[str],
    top_k: int,
    per_meeting: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Pre-build per-meeting BM25 indices if needed
    meeting_indices: dict = {}
    if per_meeting:
        for meeting_id, grp in chunks_df.groupby("audio_id"):
            m_ids = grp["chunk_id"].astype(str).tolist()
            corpus = [tokenize(t) for t in grp["text"].astype(str).tolist()]
            non_empty = [toks for toks in corpus if toks]
            if not non_empty:
                continue
            meeting_indices[meeting_id] = (BM25Okapi(corpus), m_ids)

    query_meta = labels_df[["query_id", "query_text", "audio_id"]].drop_duplicates("query_id")

    result_rows = []
    metric_rows = []

    for row in query_meta.itertuples(index=False):
        query_id = row.query_id
        query_text = row.query_text
        relevant_set = set(
            labels_df.loc[
                (labels_df["query_id"] == query_id) & (labels_df["relevance"] > 0),
                "chunk_id",
            ].astype(str).tolist()
        )

        if per_meeting and row.audio_id in meeting_indices:
            m_bm25, m_ids = meeting_indices[row.audio_id]
            m_chunks = chunks_df[chunks_df["audio_id"] == row.audio_id]
            candidates = retrieve_bm25(query_text, top_k, m_bm25, m_chunks, m_ids)
        else:
            candidates = retrieve_bm25(query_text, top_k, bm25, chunks_df, chunk_ids)

        if candidates.empty:
            metric_rows.append({"query_id": query_id, "mrr": 0.0, "recall": 0.0, "precision": 0.0, "ndcg": 0.0})
            continue

        candidates["query_id"] = query_id
        candidates["query_text"] = query_text
        candidates["relevance"] = candidates["chunk_id"].astype(str).isin(relevant_set).astype(int)
        result_rows.append(candidates)

        metrics = _query_metrics(candidates["relevance"].tolist(), len(relevant_set))
        metric_rows.append({"query_id": query_id, **metrics})

    results_df = pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    summary_df = metrics_df.mean(numeric_only=True).to_frame().T
    pipeline_name = "bm25_pm" if per_meeting else "bm25"
    summary_df.insert(0, "pipeline", pipeline_name)
    return results_df, metrics_df, summary_df


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="BM25 lexical retrieval baseline.")
    parser.add_argument("--model", default="medium", help="ASR model name (e.g. medium, large-v3)")
    parser.add_argument("--topk", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--per-meeting", action="store_true",
                        help="Restrict retrieval to chunks from the query's own meeting")
    args = parser.parse_args()

    aligned_dir = PROJECT_ROOT / "data" / "aligned_chunks" / args.model
    metrics_dir = PROJECT_ROOT / "data" / "metrics" / "bm25"
    retrieval_dir = PROJECT_ROOT / "data" / "retrieval_results" / "bm25"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading aligned chunks from {aligned_dir} ...")
    chunks_df = load_aligned_chunks(aligned_dir)
    print(f"  {len(chunks_df)} chunks loaded.")

    print("Building BM25 index ...")
    bm25, chunk_ids = build_bm25_index(chunks_df)
    print(f"  Index built: {len(chunk_ids)} chunks.")

    print("Loading query labels ...")
    labels_df = load_query_labels(QUERIES_CSV, CHECKPOINTS_DIR)
    print(f"  {labels_df['query_id'].nunique()} queries, {len(labels_df)} label rows.")

    pm_tag = "_pm" if args.per_meeting else ""
    print(f"Evaluating (top-{args.topk}, per_meeting={args.per_meeting}) ...")
    results_df, metrics_df, summary_df = evaluate(
        labels_df, chunks_df, bm25, chunk_ids, args.topk, args.per_meeting
    )

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(retrieval_dir / f"results_{args.model}{pm_tag}_{run_tag}.csv", index=False)
    metrics_df.to_csv(metrics_dir / f"per_query_{args.model}{pm_tag}_{run_tag}.csv", index=False)
    summary_df.to_csv(metrics_dir / f"summary_{args.model}{pm_tag}_{run_tag}.csv", index=False)

    print("\n── BM25 retrieval summary ──")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
