"""
Answer generation and quality evaluation.

For each retrieval pipeline's top-K results:
  1. Concatenate the top-K retrieved chunks as context
  2. Prompt a local LLM (via Ollama) to answer the query
  3. Compare generated answer against reference answer from retrieval_eval_queries.csv
  4. Compute: Exact Match (EM), Token F1, BERTScore

Prerequisites:
  - Ollama running locally with a model available (default: llama3.1:8b)
  - bert-score installed: uv add bert-score
  - All retrieval pipelines run (results in data/retrieval_results/)

Usage:
    python scripts/generate_answers.py [--model medium] [--topk 5]
        [--pipelines bm25 dense crossencoder naes_h naes_l]
        [--llm llama3.1:8b] [--ollama-url http://localhost:11434]

Outputs:
    data/answer_results/<pipeline>/answers_<model>_<tag>.csv
    data/answer_results/<pipeline>/summary_<model>_<tag>.csv
"""

import argparse
import re
import string
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
RESULTS_DIR = PROJECT_ROOT / "data" / "retrieval_results"
ANSWER_DIR = PROJECT_ROOT / "data" / "answer_results"
ANSWER_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PIPELINES = ["bm25", "dense", "crossencoder", "naes_h", "naes_l"]
DEFAULT_LLM = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

ANSWER_PROMPT = """You are answering a question about a meeting based on the provided passages.

Answer using ONLY the information explicitly stated in the passages below.
Do not infer, generalize, or add information not present in the passages.
Keep your answer concise (1-3 sentences).
If the passages do not contain enough information to answer the question, respond with exactly: UNANSWERABLE

Question: {question}

Passages:
{context}

Answer:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest(directory: Path, prefix: str) -> Path | None:
    candidates = sorted(directory.glob(f"{prefix}*.csv"))
    return candidates[-1] if candidates else None


def _normalize_text(text: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_f1(pred: str, ref: str) -> float:
    pred_toks = _normalize_text(pred).split()
    ref_toks = _normalize_text(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    common = set(pred_toks) & set(ref_toks)
    if not common:
        return 0.0
    precision = len([t for t in pred_toks if t in common]) / len(pred_toks)
    recall = len([t for t in ref_toks if t in common]) / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _exact_match(pred: str, ref: str) -> int:
    return int(_normalize_text(pred) == _normalize_text(ref))


def _build_context(chunks: pd.DataFrame, rank_col: str = "rank") -> str:
    sorted_chunks = chunks.sort_values(rank_col)
    parts = []
    for i, (_, row) in enumerate(sorted_chunks.iterrows(), start=1):
        speaker = row.get("speaker_label", row.get("speaker", "SPEAKER"))
        parts.append(f"[Passage {i}] [{speaker}]: {row['text']}")
    return "\n\n".join(parts)


def call_ollama(prompt: str, model: str, url: str) -> str:
    """Call Ollama generate API and return the response text."""
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "OLLAMA_UNAVAILABLE"
    except Exception as e:
        return f"ERROR: {e}"


def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    """Compute BERTScore F1 for a list of (pred, ref) pairs."""
    try:
        from bert_score import score as bert_score_fn
        _, _, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
        return F1.tolist()
    except ImportError:
        print("  [warn] bert-score not installed — BERTScore will be 0.0")
        return [0.0] * len(predictions)
    except Exception as e:
        print(f"  [warn] BERTScore failed: {e}")
        return [0.0] * len(predictions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(pipeline: str, model: str, top_k: int, llm: str,
                 ollama_url: str, queries_df: pd.DataFrame, tag: str):
    results_path = _latest(RESULTS_DIR / pipeline, f"results_{model}_")
    if results_path is None:
        print(f"  [skip] No results found for pipeline={pipeline} model={model}")
        return

    print(f"\n  Loading: {results_path.name}")
    results_df = pd.read_csv(results_path)

    out_dir = ANSWER_DIR / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_map = dict(zip(queries_df["query_id"], queries_df["reference_answer"]))
    query_text_map = dict(zip(queries_df["query_id"], queries_df["query_text"]))

    answer_rows = []
    predictions, references = [], []

    query_ids = results_df["query_id"].unique()
    for i, qid in enumerate(query_ids):
        ref_answer = ref_map.get(qid, "")
        if not ref_answer or str(ref_answer).strip() == "":
            continue

        chunks = results_df[results_df["query_id"] == qid]
        chunks = chunks[chunks["rank"] <= top_k]
        if chunks.empty:
            continue

        context = _build_context(chunks)
        question = query_text_map.get(qid, "")
        prompt = ANSWER_PROMPT.format(question=question, context=context)

        print(f"    [{i+1}/{len(query_ids)}] {qid}...", end=" ", flush=True)
        generated = call_ollama(prompt, llm, ollama_url)

        if generated == "OLLAMA_UNAVAILABLE":
            print("OLLAMA unavailable — stopping.")
            return

        em = _exact_match(generated, ref_answer)
        f1 = _token_f1(generated, ref_answer)
        print(f"EM={em} F1={f1:.3f}")

        answer_rows.append({
            "query_id": qid,
            "query_text": question,
            "generated_answer": generated,
            "reference_answer": ref_answer,
            "exact_match": em,
            "token_f1": round(f1, 4),
            "bertscore_f1": 0.0,  # filled after batch
        })
        predictions.append(generated)
        references.append(ref_answer)

    if not answer_rows:
        print("  No answers generated.")
        return

    # Batch BERTScore
    print("  Computing BERTScore...")
    bs_scores = compute_bertscore(predictions, references)
    for row, score in zip(answer_rows, bs_scores):
        row["bertscore_f1"] = round(float(score), 4)

    answers_df = pd.DataFrame(answer_rows)
    summary = pd.DataFrame([{
        "pipeline": pipeline,
        "model": model,
        "llm": llm,
        "topk": top_k,
        "n_queries": len(answers_df),
        "exact_match": round(answers_df["exact_match"].mean(), 4),
        "token_f1": round(answers_df["token_f1"].mean(), 4),
        "bertscore_f1": round(answers_df["bertscore_f1"].mean(), 4),
    }])

    answers_path = out_dir / f"answers_{model}_{tag}.csv"
    summary_path = out_dir / f"summary_{model}_{tag}.csv"
    answers_df.to_csv(answers_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"  Results: EM={summary['exact_match'].iloc[0]:.4f} "
          f"F1={summary['token_f1'].iloc[0]:.4f} "
          f"BERTScore={summary['bertscore_f1'].iloc[0]:.4f}")
    print(f"  Saved: {answers_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of top chunks to include in context")
    parser.add_argument("--pipelines", nargs="+", default=DEFAULT_PIPELINES)
    parser.add_argument("--llm", default=DEFAULT_LLM,
                        help="Ollama model name (default: llama3.1:8b)")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    args = parser.parse_args()

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Answer generation | model={args.model} topk={args.topk} llm={args.llm}")

    # Test Ollama connectivity
    try:
        resp = requests.get(f"{args.ollama_url}/api/tags", timeout=5)
        available = [m["name"] for m in resp.json().get("models", [])]
        print(f"Ollama models available: {available}")
        if args.llm not in available:
            print(f"  [warn] {args.llm} not in available models — will attempt anyway")
    except Exception:
        print(f"  [warn] Cannot connect to Ollama at {args.ollama_url}. "
              f"Start with: ollama serve")

    queries_df = pd.read_csv(EVAL_DIR / "retrieval_eval_queries.csv")
    # Only use queries with non-empty reference answers and positives
    queries_df = queries_df[
        (queries_df["difficulty_flag"] != "no_positives") &
        (queries_df["reference_answer"].notna()) &
        (queries_df["reference_answer"].str.strip() != "")
    ]
    print(f"Evaluating on {len(queries_df)} queries with reference answers\n")

    for pipeline in args.pipelines:
        print(f"=== Pipeline: {pipeline} ===")
        run_pipeline(pipeline, args.model, args.topk, args.llm,
                     args.ollama_url, queries_df, tag)

    print("\nDone.")


if __name__ == "__main__":
    main()
