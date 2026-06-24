"""
Ablation study for NAES-L: drop one feature at a time and measure NDCG@10 delta.

Answers RQ3: what is the relative contribution of each signal?

For each feature in [ASRConf, DiarStab, TurnComp, Redund, MixPenalty, semantic_score]:
  - Replace that feature with 0.0 (zero-out ablation)
  - Re-run LOMO CV with best C from the full NAES-L run
  - Record NDCG@10 delta vs. full model

The best C is loaded from the most recent NAES-L summary CSV.
If no NAES-L summary exists, a default C=1.0 is used.

Usage:
    python scripts/ablation_naes_l.py [--model medium] [--topk 10] [--rerank-pool 50]
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
SUMMARY_DIR = PROJECT_ROOT / "data" / "metrics" / "naes_l"
ABLATION_DIR = PROJECT_ROOT / "data" / "metrics" / "ablation"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["semantic_score", "ASRConf", "DiarStab", "TurnComp", "Redund", "MixPenalty"]
FEATURE_DEFAULTS = {
    "ASRConf": 0.5, "DiarStab": 0.5, "TurnComp": 0.5,
    "Redund": 0.0, "MixPenalty": 0.0, "semantic_score": 0.0,
}
DEFAULT_C = 1.0


def _latest(directory: Path, prefix: str) -> Path | None:
    candidates = sorted(directory.glob(f"{prefix}*.csv"))
    return candidates[-1] if candidates else None


def load_dense_pool(model: str, pool_size: int) -> pd.DataFrame:
    results_path = _latest(
        PROJECT_ROOT / "data" / "retrieval_results" / "dense",
        f"results_{model}_",
    )
    if results_path is None:
        raise FileNotFoundError(f"No dense results for model={model}")
    df = pd.read_csv(results_path)
    df = df[df["rank"] <= pool_size].copy()
    for col, val in FEATURE_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


def load_labels(model: str) -> pd.DataFrame:
    labels = pd.read_csv(EVAL_DIR / "retrieval_eval_labels.csv")
    queries = pd.read_csv(EVAL_DIR / "retrieval_eval_queries.csv")[
        ["query_id", "audio_id", "difficulty_flag"]
    ]
    labels = labels.merge(queries, on=["query_id", "audio_id"], how="left")
    return labels[labels["difficulty_flag"] != "no_positives"].copy()


def _ndcg_at_k(ranked: list, k: int) -> float:
    k = min(k, len(ranked))
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(ranked[:k]))
    ideal = sorted(ranked, reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


N_FOLDS = 5


def _assign_folds(labels: pd.DataFrame) -> pd.DataFrame:
    meetings = sorted(labels["audio_id"].unique())
    fold_map = {m: i % N_FOLDS for i, m in enumerate(meetings)}
    labels = labels.copy()
    labels["fold"] = labels["audio_id"].map(fold_map)
    return labels


def stratified_ndcg(pool_df: pd.DataFrame, labels: pd.DataFrame,
                    feature_cols: list, k: int, c_value: float) -> float:
    """Run 5-fold meeting-stratified CV with a given feature set. Returns mean NDCG@k."""
    labels = _assign_folds(labels)
    ndcgs = []

    for fold_id in range(N_FOLDS):
        train_labels = labels[labels["fold"] != fold_id]
        held_labels  = labels[labels["fold"] == fold_id]
        held_pool = pool_df[pool_df["audio_id"].isin(held_labels["audio_id"].unique())].copy()
        if held_pool.empty:
            continue

        train_pool = pool_df[pool_df["audio_id"].isin(train_labels["audio_id"].unique())]
        train_data = train_labels.merge(
            train_pool[["query_id", "chunk_id"] + feature_cols],
            on=["query_id", "chunk_id"], how="inner",
        ).dropna(subset=feature_cols)

        if len(train_data) < 10 or train_data["relevance"].nunique() < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[feature_cols].values)
        y_train = train_data["relevance"].values

        clf = LogisticRegression(C=c_value, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        X_held = scaler.transform(held_pool[feature_cols].values)
        held_pool = held_pool.copy()
        held_pool["score"] = clf.predict_proba(X_held)[:, 1]

        rel_map = held_labels.set_index("chunk_id")["relevance"].to_dict()
        held_pool["relevance"] = held_pool["chunk_id"].map(rel_map).fillna(0).astype(int)

        for _, grp in held_pool.groupby("query_id"):
            ranked = grp.sort_values("score", ascending=False)["relevance"].tolist()
            ndcgs.append(_ndcg_at_k(ranked, k))

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def get_best_c(model: str) -> float:
    summary_path = _latest(SUMMARY_DIR, f"summary_{model}_")
    if summary_path is not None:
        df = pd.read_csv(summary_path)
        if "best_c" in df.columns and not df["best_c"].isna().all():
            return float(df["best_c"].iloc[0])
    print(f"  [warn] No NAES-L summary found — using default C={DEFAULT_C}")
    return DEFAULT_C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rerank-pool", type=int, default=100)
    args = parser.parse_args()

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"NAES-L Ablation | model={args.model} topk={args.topk} pool={args.rerank_pool}")

    pool_df = load_dense_pool(args.model, args.rerank_pool)
    labels = load_labels(args.model)
    best_c = get_best_c(args.model)
    print(f"  Using C={best_c}")

    # Baseline: full feature set
    print("\nBaseline (all features)...")
    baseline_ndcg = stratified_ndcg(pool_df, labels, FEATURE_COLS, args.topk, best_c)
    print(f"  Baseline NDCG@{args.topk} = {baseline_ndcg:.4f}")

    # Ablation: zero out one feature at a time
    rows = [{"feature_dropped": "none (baseline)", "ndcg": round(baseline_ndcg, 4),
             "delta_ndcg": 0.0}]

    for feat in FEATURE_COLS:
        print(f"\nAblating: {feat}...")
        ablated_pool = pool_df.copy()
        ablated_pool[feat] = 0.0
        ndcg = stratified_ndcg(ablated_pool, labels, FEATURE_COLS, args.topk, best_c)
        delta = ndcg - baseline_ndcg
        rows.append({
            "feature_dropped": feat,
            "ndcg": round(ndcg, 4),
            "delta_ndcg": round(delta, 4),
        })
        print(f"  NDCG@{args.topk} = {ndcg:.4f}  Δ = {delta:+.4f}")

    results = pd.DataFrame(rows)
    out_path = ABLATION_DIR / f"ablation_{args.model}_{tag}.csv"
    results.to_csv(out_path, index=False)

    print(f"\n{'='*55}")
    print(f"Ablation results — {args.model}, top-{args.topk}")
    print(f"{'='*55}")
    print(results.sort_values("delta_ndcg").to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
