"""
NAES-L: Learned Noise-Aware Evidence Selection.

Trains a logistic regression on (query, chunk, relevance) triples using
5-fold cross-validation stratified by meeting (queries from each meeting are
assigned to the same fold, preventing within-meeting train/test leakage).

Features: [semantic_score, ASRConf, DiarStab, TurnComp, Redund, MixPenalty]

Training data: data/eval/retrieval_eval_labels.csv (binary relevance labels)
  - Only queries with difficulty_flag != "no_positives" are used.
  - Both positive (relevance=1) and negative (relevance=0) label rows are used.

CV procedure (query-level, meeting-stratified):
  - Assign each unique meeting to one of 5 folds (round-robin by meeting)
  - All queries from a meeting go into the same fold
  - Train on 4 folds, evaluate NDCG@K on the held-out fold
  - Grid search over C in [0.01, 0.1, 1.0, 10.0]

This is better than leave-one-meeting-out (LOMO) because each training fold has
~80% of the data (~60+ queries), giving the logistic regression enough positive
examples to learn meaningful weights.

Final model: retrain on all data with the best C found via CV.
Final rankings written to data/retrieval_results/naes_l/.

Usage:
    python scripts/retrieval_naes_l.py [--model medium] [--topk 10] [--rerank-pool 50]
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
RESULTS_DIR = PROJECT_ROOT / "data" / "retrieval_results" / "naes_l"
SUMMARY_DIR = PROJECT_ROOT / "data" / "metrics" / "naes_l"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["semantic_score", "ASRConf", "DiarStab", "TurnComp", "Redund", "MixPenalty"]
FEATURE_DEFAULTS = {
    "ASRConf": 0.5, "DiarStab": 0.5, "TurnComp": 0.5,
    "Redund": 0.0, "MixPenalty": 0.0, "semantic_score": 0.0,
}
C_GRID = [0.01, 0.1, 1.0, 10.0]


# ---------------------------------------------------------------------------
# Dense retrieval pool (reused from retrieval_dense.py cache)
# ---------------------------------------------------------------------------

def _latest(directory: Path, prefix: str) -> Path:
    candidates = sorted(directory.glob(f"{prefix}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.csv in {directory}")
    return candidates[-1]


def load_dense_pool(model: str, pool_size: int) -> pd.DataFrame:
    """
    Load dense retrieval results and return the top-pool_size candidates per query.
    The dense results already contain all NAES feature columns.
    """
    results_path = _latest(
        PROJECT_ROOT / "data" / "retrieval_results" / "dense",
        f"results_{model}_",
    )
    df = pd.read_csv(results_path)
    # Keep only top-pool_size by dense rank
    df = df[df["rank"] <= pool_size].copy()
    for col, val in FEATURE_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


def load_labels(model: str) -> pd.DataFrame:
    """Load eval labels joined with difficulty flags."""
    labels = pd.read_csv(EVAL_DIR / "retrieval_eval_labels.csv")
    queries = pd.read_csv(EVAL_DIR / "retrieval_eval_queries.csv")[
        ["query_id", "audio_id", "difficulty_flag"]
    ]
    labels = labels.merge(queries, on=["query_id", "audio_id"], how="left")
    # Exclude no_positives queries from training
    labels = labels[labels["difficulty_flag"] != "no_positives"].copy()
    return labels


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _ndcg_at_k(ranked_relevance: list, k: int) -> float:
    k = min(k, len(ranked_relevance))
    dcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(ranked_relevance[:k])
    )
    ideal = sorted(ranked_relevance, reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _mrr(ranked_relevance: list) -> float:
    for i, rel in enumerate(ranked_relevance):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(ranked_relevance: list, k: int) -> float:
    total = sum(1 for r in ranked_relevance if r > 0)
    if total == 0:
        return 0.0
    hits = sum(1 for r in ranked_relevance[:k] if r > 0)
    return hits / total


def _query_metrics(group: pd.DataFrame, k: int) -> dict:
    ranked = group.sort_values("naes_l_score", ascending=False)["relevance"].tolist()
    return {
        "ndcg": _ndcg_at_k(ranked, k),
        "mrr": _mrr(ranked),
        "recall": _recall_at_k(ranked, k),
        "precision": sum(1 for r in ranked[:k] if r > 0) / k,
    }


# ---------------------------------------------------------------------------
# Meeting-stratified 5-fold cross-validation
# ---------------------------------------------------------------------------

N_FOLDS = 5


def _assign_folds(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each meeting to a fold (round-robin). All queries from the same
    meeting go into the same fold, preventing within-meeting leakage.
    """
    meetings = sorted(labels["audio_id"].unique())
    fold_map = {m: i % N_FOLDS for i, m in enumerate(meetings)}
    labels = labels.copy()
    labels["fold"] = labels["audio_id"].map(fold_map)
    return labels


def stratified_cv(pool_df: pd.DataFrame, labels: pd.DataFrame, k: int, c_value: float) -> dict:
    """
    Run 5-fold meeting-stratified CV with a fixed C value.
    Returns mean NDCG@k across held-out folds.
    """
    labels = _assign_folds(labels)
    per_query_rows = []

    for fold_id in range(N_FOLDS):
        train_labels = labels[labels["fold"] != fold_id]
        held_labels  = labels[labels["fold"] == fold_id]
        held_pool    = pool_df[pool_df["audio_id"].isin(held_labels["audio_id"].unique())].copy()

        if held_pool.empty:
            continue

        train_pool = pool_df[pool_df["audio_id"].isin(train_labels["audio_id"].unique())]
        train_data = train_labels.merge(
            train_pool[["query_id", "chunk_id"] + FEATURE_COLS],
            on=["query_id", "chunk_id"],
            how="inner",
        ).dropna(subset=FEATURE_COLS)

        if len(train_data) < 10 or train_data["relevance"].nunique() < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[FEATURE_COLS].values)
        y_train = train_data["relevance"].values

        clf = LogisticRegression(C=c_value, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        X_held = scaler.transform(held_pool[FEATURE_COLS].values)
        held_pool = held_pool.copy()
        held_pool["naes_l_score"] = clf.predict_proba(X_held)[:, 1]

        rel_map = held_labels.set_index("chunk_id")["relevance"].to_dict()
        held_pool["relevance"] = held_pool["chunk_id"].map(rel_map).fillna(0).astype(int)

        for qid, grp in held_pool.groupby("query_id"):
            m = _query_metrics(grp, k)
            m["query_id"] = qid
            m["audio_id"] = grp["audio_id"].iloc[0]
            per_query_rows.append(m)

    if not per_query_rows:
        return {"mean_ndcg": 0.0, "per_query": pd.DataFrame()}

    pq = pd.DataFrame(per_query_rows)
    return {"mean_ndcg": pq["ndcg"].mean(), "per_query": pq}


# ---------------------------------------------------------------------------
# Final model training + ranking
# ---------------------------------------------------------------------------

def train_final_model(pool_df: pd.DataFrame, labels: pd.DataFrame, best_c: float):
    """Train logistic regression on all available data."""
    merged = labels.merge(
        pool_df[["query_id", "chunk_id"] + FEATURE_COLS],
        on=["query_id", "chunk_id"],
        how="inner",
    ).dropna(subset=FEATURE_COLS)

    X = merged[FEATURE_COLS].values
    y = merged["relevance"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
    clf.fit(X_s, y)
    return clf, scaler


def rerank_pool(pool_df: pd.DataFrame, clf, scaler, labels: pd.DataFrame,
                top_k: int) -> pd.DataFrame:
    """Apply learned model to full pool, return top-k per query."""
    pool = pool_df.copy()
    X = pool[FEATURE_COLS].values
    pool["naes_l_score"] = clf.predict_proba(scaler.transform(X))[:, 1]

    # Attach ground-truth relevance
    rel_map = labels.set_index(["query_id", "chunk_id"])["relevance"].to_dict()
    pool["relevance"] = pool.apply(
        lambda r: rel_map.get((r["query_id"], r["chunk_id"]), 0), axis=1
    ).astype(int)

    rows = []
    for qid, grp in pool.groupby("query_id"):
        ranked = grp.sort_values("naes_l_score", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        rows.append(ranked.head(top_k))
    return pd.concat(rows, ignore_index=True)


def evaluate(results_df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_query = []
    for qid, grp in results_df.groupby("query_id"):
        m = _query_metrics(grp, top_k)
        m["query_id"] = qid
        m["audio_id"] = grp["audio_id"].iloc[0]
        per_query.append(m)
    pq = pd.DataFrame(per_query)
    summary = pd.DataFrame([{
        "pipeline": "naes_l",
        "mrr": pq["mrr"].mean(),
        "recall": pq["recall"].mean(),
        "precision": pq["precision"].mean(),
        "ndcg": pq["ndcg"].mean(),
        "n_queries": len(pq),
        "best_c": None,  # filled by caller
    }])
    return pq, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rerank-pool", type=int, default=50)
    args = parser.parse_args()

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"NAES-L | model={args.model} topk={args.topk} pool={args.rerank_pool}")

    print("Loading dense pool and labels...")
    pool_df = load_dense_pool(args.model, args.rerank_pool)
    labels = load_labels(args.model)
    print(f"  Pool: {len(pool_df)} rows across {pool_df['query_id'].nunique()} queries")
    print(f"  Labels: {len(labels)} rows, {labels['audio_id'].nunique()} meetings")

    # 5-fold meeting-stratified grid search over C
    print(f"\n5-fold meeting-stratified CV over C={C_GRID} ...")
    best_c, best_ndcg = C_GRID[0], -1.0
    for c in C_GRID:
        result = stratified_cv(pool_df, labels, args.topk, c)
        ndcg = result["mean_ndcg"]
        print(f"  C={c:6.2f}  NDCG@{args.topk}={ndcg:.4f}")
        if ndcg > best_ndcg:
            best_ndcg, best_c = ndcg, c
    print(f"  → Best C={best_c}, NDCG@{args.topk}={best_ndcg:.4f}")

    # Train final model on all data
    print("\nTraining final model on full dataset...")
    clf, scaler = train_final_model(pool_df, labels, best_c)
    coef = dict(zip(FEATURE_COLS, clf.coef_[0].round(4)))
    print("  Learned coefficients:")
    for feat, w in coef.items():
        print(f"    {feat:20s}: {w:+.4f}")

    # Generate final rankings
    print("\nGenerating final rankings...")
    results_df = rerank_pool(pool_df, clf, scaler, labels, args.topk)
    per_query, summary = evaluate(results_df, args.topk)
    summary["best_c"] = best_c
    summary["lomo_ndcg"] = round(best_ndcg, 4)
    for feat, w in coef.items():
        summary[f"coef_{feat}"] = w

    # Save outputs
    results_path = RESULTS_DIR / f"results_{args.model}_{tag}.csv"
    pq_path = SUMMARY_DIR / f"per_query_{args.model}_{tag}.csv"
    summary_path = SUMMARY_DIR / f"summary_{args.model}_{tag}.csv"

    results_df.to_csv(results_path, index=False)
    per_query.to_csv(pq_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*55}")
    print(f"NAES-L results — {args.model}, top-{args.topk}")
    print(f"{'='*55}")
    print(summary[["pipeline", "ndcg", "mrr", "recall", "precision"]].to_string(index=False))
    print(f"\nSaved:\n  {results_path}\n  {pq_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
