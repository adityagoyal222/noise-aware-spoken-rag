"""
retrieval_dense.py — Dense cosine-similarity retrieval baseline (no reranking).

Loads aligned chunks, builds/caches sentence-transformer embeddings, retrieves
top-K candidates per query by cosine similarity, evaluates against relevance
labels, and writes results + metrics to data/retrieval_results/dense/.

Usage:
    python scripts/retrieval_dense.py [--model medium] [--topk 10]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError as exc:
    raise ImportError("faiss is not installed. Run: pip install faiss-cpu") from exc

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as exc:
    raise ImportError("sentence-transformers is not installed.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
QUERIES_CSV = EVAL_DIR / "retrieval_eval_queries.csv"
CHECKPOINTS_DIR = EVAL_DIR / "checkpoints"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
TOP_K_DEFAULT = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── data loading ──────────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    chunk_id: str
    audio_id: str
    start: float
    end: float
    text: str
    speaker_label: str
    ASRConf: float
    DiarStab: float
    TurnComp: float
    Redund: float
    MixPenalty: float
    Purity: float


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "speaker": "speaker_label",
        "audio_id": "audio_id",
        "file_id": "audio_id",
    }
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
    """
    Returns a flat DataFrame with columns:
    query_id, query_text, chunk_id, relevance
    assembled from the queries CSV and per-meeting checkpoint JSON files.
    """
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
                "chunk_id": str(label["chunk_id"]),
                "relevance": int(label.get("relevance", 1)),
            })

    if not label_rows:
        raise ValueError("No label rows found in checkpoint files.")

    return pd.DataFrame(label_rows)


# ── embeddings ────────────────────────────────────────────────────────────────

def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_embeddings(texts: list[str], model: SentenceTransformer, batch_size: int) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return _normalize(vectors)


def load_embedding_cache(path: Path) -> tuple[list[str], np.ndarray] | None:
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=True)
    ids = payload.get("ids")
    vectors = payload.get("vectors")
    if ids is None or vectors is None:
        return None
    return list(ids.tolist()), vectors


def save_embedding_cache(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    np.savez(path, ids=np.array(ids), vectors=vectors)


def get_embeddings(df: pd.DataFrame, model: SentenceTransformer, cache_path: Path, batch_size: int) -> tuple[np.ndarray, list[str]]:
    ids = df["chunk_id"].astype(str).tolist()
    cached = load_embedding_cache(cache_path)
    if cached is not None:
        cached_ids, cached_vectors = cached
        if cached_ids == ids and cached_vectors.shape[0] == len(ids):
            print("  Using cached embeddings.")
            return cached_vectors, ids
    print("  Building embeddings...")
    vectors = build_embeddings(df["text"].tolist(), model, batch_size)
    save_embedding_cache(cache_path, ids, vectors)
    return vectors, ids


# ── retrieval ─────────────────────────────────────────────────────────────────

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    return index


def retrieve(
    query: str,
    top_k: int,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    df: pd.DataFrame,
    id_map: list[str],
) -> pd.DataFrame:
    q_vec = _normalize(model.encode([query], convert_to_numpy=True, normalize_embeddings=False))
    scores, indices = index.search(q_vec.astype(np.float32), top_k)
    scores, indices = scores.flatten(), indices.flatten()

    rows = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        if idx < 0 or idx >= len(id_map):
            continue
        chunk_id = id_map[idx]
        match = df.loc[df["chunk_id"] == chunk_id]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        row["semantic_score"] = float(score)
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
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    id_map: list[str],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query_texts = labels_df[["query_id", "query_text"]].drop_duplicates("query_id")

    result_rows = []
    metric_rows = []

    for row in query_texts.itertuples(index=False):
        query_id = row.query_id
        query_text = row.query_text
        relevant_set = set(
            labels_df.loc[
                (labels_df["query_id"] == query_id) & (labels_df["relevance"] > 0),
                "chunk_id",
            ].astype(str).tolist()
        )

        candidates = retrieve(query_text, top_k, model, index, chunks_df, id_map)
        candidates["query_id"] = query_id
        candidates["query_text"] = query_text
        candidates["relevance"] = candidates["chunk_id"].astype(str).isin(relevant_set).astype(int)
        result_rows.append(candidates)

        metrics = _query_metrics(candidates["relevance"].tolist(), len(relevant_set))
        metric_rows.append({"query_id": query_id, **metrics})

    results_df = pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    summary_df = metrics_df.mean(numeric_only=True).to_frame().T
    summary_df.insert(0, "pipeline", "dense")
    return results_df, metrics_df, summary_df


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Dense retrieval baseline.")
    parser.add_argument("--model", default="medium", help="ASR model name (e.g. medium, large-v3)")
    parser.add_argument("--topk", type=int, default=TOP_K_DEFAULT)
    args = parser.parse_args()

    aligned_dir = PROJECT_ROOT / "data" / "aligned_chunks" / args.model
    metrics_dir = PROJECT_ROOT / "data" / "metrics" / "dense"
    retrieval_dir = PROJECT_ROOT / "data" / "retrieval_results" / "dense"
    embed_cache = PROJECT_ROOT / "data" / "metrics" / f"chunk_embeddings_minilm_{args.model}.npz"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading aligned chunks from {aligned_dir} ...")
    chunks_df = load_aligned_chunks(aligned_dir)
    print(f"  {len(chunks_df)} chunks loaded.")

    print(f"Loading sentence-transformer on {DEVICE} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

    embeddings, id_map = get_embeddings(chunks_df, embed_model, embed_cache, BATCH_SIZE)
    index = build_index(embeddings)
    print(f"  Index built: {index.ntotal} vectors.")

    print("Loading query labels ...")
    labels_df = load_query_labels(QUERIES_CSV, CHECKPOINTS_DIR)
    print(f"  {labels_df['query_id'].nunique()} queries, {len(labels_df)} label rows.")

    print(f"Evaluating (top-{args.topk}) ...")
    results_df, metrics_df, summary_df = evaluate(
        labels_df, chunks_df, embed_model, index, id_map, args.topk
    )

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(retrieval_dir / f"results_{args.model}_{run_tag}.csv", index=False)
    metrics_df.to_csv(metrics_dir / f"per_query_{args.model}_{run_tag}.csv", index=False)
    summary_df.to_csv(metrics_dir / f"summary_{args.model}_{run_tag}.csv", index=False)

    print("\n── Dense retrieval summary ──")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
