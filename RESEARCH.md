# Spoken RAG Thesis — Research Tracker

## Research Objective

Quantify how ASR and diarization errors degrade retrieval and answer quality in a spoken RAG system, then design and evaluate a noise-aware reranking method that exploits ASR confidence, diarization stability, and speaker-turn signals to recover from those degradations.

---

## Research Questions

| # | Question |
|---|----------|
| RQ1 | How do WER and DER, individually and together, affect retrieval and answer quality in a cascaded spoken RAG pipeline on meeting QA data? |
| RQ2 | Can a confidence-aware reranking method using ASR confidence, diarization stability, and speaker-turn signals improve over a noise-unaware baseline? |
| RQ3 | What is the relative contribution of each individual signal (ablation)? |
| RQ4 | How does the method perform across noise levels — is there a threshold where confidence-aware reranking yields diminishing returns? |

---

## Systems to Build and Evaluate

Listed in order from floor to ceiling:

| System | Description | Status |
|--------|-------------|--------|
| **BM25-only** | BM25 on ASR transcripts, no neural retrieval, no reranking | Done (`scripts/retrieval_bm25.py`) |
| **Dense baseline** | Cosine similarity dense retrieval, no reranking | Done (`scripts/retrieval_dense.py`) |
| **Cross-encoder** | Dense retrieval + `ms-marco-MiniLM-L-6-v2` reranker (most important comparison) | Done (`scripts/retrieval_crossencoder.py`) |
| **NAES-H** (Heuristic) | Dense retrieval + noise-aware heuristic reranker | Done (`scripts/retrieval_naes_h.py`) |
| **NAES-L** (Learned) | Dense retrieval + learned noise-aware reranker | Not started |
| **Oracle** | Manual AMI gold transcripts as upper-bound performance ceiling | Not started |

**Reranking formula (NAES):**
```
R(c) = α·s(q,c) + β·ASRConf(c) + γ·DiarStab(c) + δ·TurnComp(c) − ε·Redund(c) − μ·MixPenalty(c)
```

---

## ASR Models to Run

All pipelines must be run across these Whisper model sizes to enable WER stratification:

| Model | ASR | Alignment |
|-------|-----|-----------|
| `tiny` | Done | Done (`data/aligned_chunks/tiny/`) |
| `base` | Done | Done (`data/aligned_chunks/base/`) |
| `small` | Done | Done (`data/aligned_chunks/small/`) |
| `medium` | Done | Done (`data/aligned_chunks/medium/`) |
| `large-v3` | Done | Done (`data/aligned_chunks/large-v3/`) |

---

## Evaluation Metrics

**Retrieval quality:** NDCG@5, MRR, Recall@K

**Answer quality:** Exact Match, F1, BERTScore (answer generation via local LLM — Mistral-7B-Instruct or LLaMA-3-8B-Instruct via Ollama, inference-only)

**Stratification dimensions:**
- WER tier: 0–15%, 15–30%, >30% (WER computed per meeting — `data/analysis/wer_per_meeting_<model>.csv`)
- DER tier: 0–15%, 15–30%, >30% (DER computed per meeting — `data/analysis/der_per_meeting.csv`; mean DER=0.124, range 0.06–0.31)
- Meeting length (short / medium / long by word count)
- Number of active speakers (from diarization output)
- Difficulty tier: `difficulty_flag = "ok"` vs. full set

**Ablation studies (RQ3):**
- Drop each NAES feature one at a time, fix remaining weights, measure NDCG@10 delta
- Features to ablate: ASRConf, DiarStab, TurnComp, Redund, MixPenalty
- Provides per-signal contribution estimates for RQ3

**Deliverables beyond aggregate metrics:**
1. Failure case analysis — queries where NAES-L < dense baseline; inspect chunks, ASRConf, DiarStab values
2. Anticipated failure modes: (a) high-confidence but wrong transcription (proper nouns, domain terms); (b) within-turn diarization errors that don't trigger the stability flag
3. 2–3 qualitative examples: query + gold answer + cross-encoder top chunk + NAES-L top chunk with annotated ASR conf and diarization stability

---

## Current Pipeline State

```
data/raw_audio/                     ← audio subset used in experiments
data/asr_outputs/medium/            ← Whisper medium ASR (other models pending)
data/diarization_outputs/           ← pyannote output (done, not re-run)
data/aligned_chunks/medium/         ← ASR + diarization aligned by time overlap
data/eval/                          ← query sets, relevance labels, reference answers
  retrieval_eval_queries.csv        ← queries with reference_answer field (updated)
  retrieval_eval_labels.csv
  checkpoints/                      ← per-meeting labels + reference answers
data/metrics/medium/                ← per-chunk feature tables
data/metrics/chunk_embeddings_minilm_medium.npz
data/retrieval_results/dense/       ← dense baseline run outputs
```

## Scripts

All pipeline stages run as standalone scripts. Notebooks are kept as read-only reference only.

| Script | Status | Purpose |
|--------|--------|---------|
| `scripts/run_asr.py` | Done | Run Whisper ASR for all model sizes (`--models`, `--retry-failed`) |
| `scripts/run_alignment.py` | Done | Align ASR + diarization (`--model <name>` or `--all`) |
| `scripts/compute_wer.py` | Done | Compute per-meeting WER by aligning Whisper output against AMI gold transcripts |
| `scripts/compute_der.py` | Done | Compute per-meeting DER by comparing pyannote RTTM against AMI gold speaker annotations |
| `scripts/generate_eval_queries_new.py` | Done | Generate queries, relevance labels, and reference answers |
| `scripts/retrieval_dense.py` | Done | Dense cosine-similarity retrieval baseline (`--model`, `--topk`) |
| `scripts/retrieval_bm25.py` | Done | BM25-only retrieval |
| `scripts/retrieval_crossencoder.py` | Done | Dense + cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) |
| `scripts/retrieval_naes_h.py` | Done | Dense + heuristic noise-aware reranker |
| `scripts/retrieval_naes_l.py` | Not started | Dense + learned noise-aware reranker (logistic regression, leave-one-meeting-out CV) |
| `scripts/generate_answers.py` | Not started | Answer generation via local LLM over retrieved chunks; computes EM, F1, BERTScore |

---

## Baseline Results (medium model, top-10, 96 queries with positives)

| Pipeline      | NDCG@10 | MRR    | Recall@10 |
|---------------|---------|--------|-----------|
| BM25          | 0.2761  | 0.2929 | 0.3507    |
| Dense         | 0.3669  | 0.3858 | 0.4583    |
| Cross-Encoder | 0.4644  | 0.5000 | 0.5339    |
| NAES-H        | TBD     | TBD    | TBD       |
| NAES-L        | TBD     | TBD    | TBD       |

---

## Next Steps

**1. Run NAES-H on medium model** ← current
- Script ready: `python scripts/retrieval_naes_h.py --model medium --topk 10 --rerank-pool 50`
- Compare against cross-encoder to see if heuristic weights beat text-only reranking.

**2. Compute WER per meeting** ✓ Done
- `scripts/compute_wer.py` — aligns Whisper output against AMI gold transcripts using `jiwer`.
- Outputs: `data/analysis/wer_per_meeting_<model>.csv` and combined across all models.
- Medium model WER range: 0.20–1.03, mean 0.41. Some meetings show WER > 1.0 (Whisper hallucination on TS3010 series). These cap to "high" tier.

**3. Implement and run NAES-L**
- Learned version: logistic regression trained on (query, chunk, relevance) triples, leave-one-meeting-out CV.
- Features: `[semantic_score, ASRConf, DiarStab, TurnComp, Redund, MixPenalty]` — same as NAES-H formula.
- Use `sklearn.linear_model.LogisticRegression`; optimize NDCG@10 on held-out meeting folds.
- Script: `scripts/retrieval_naes_l.py`

**4. Cross-model sweep (all 5 ASR model sizes)**
- Run all systems (BM25, Dense, CE, NAES-H, NAES-L) across tiny/base/small/medium/large-v3.
- Use Option A: medium chunk IDs as canonical index, swap chunk text per model.
- Produces WER-stratification table for RQ1 and RQ4.

**5. Answer quality evaluation**
- Script needed: `scripts/generate_answers.py` — given retrieval results, prompt local LLM (Mistral-7B or LLaMA-3-8B via Ollama) to generate answers, then compute EM, F1, BERTScore against reference answers.

**6. Ablation studies (RQ3)**
- Drop one NAES feature at a time from NAES-L, retrain, measure NDCG@10 delta.
- Quantifies per-signal contribution for RQ3.

**7. Final analysis and write-up**
- Failure case analysis: queries where NAES-L < dense baseline; inspect chunks + feature values.
- 2–3 annotated qualitative examples.
- Update results tables with all final numbers.

---

## Decisions Made

- Alignment is done by time overlap (not exact timestamps); multi-speaker chunks are flagged with `DiarStab` and `MixPenalty` rather than collapsed.
- Binary relevance labels first; graded labels are a stretch goal.
- Query generation uses a local LLM (Ollama `llama3.1:8b`) on section-windowed transcript text.
- Cross-encoder comparison uses `ms-marco-MiniLM-L-6-v2` — this is the most important baseline: if NAES-L beats it, audio-specific signals add value over text-only reranking.
- Hardware constraints: Apple M2 Pro and Google Colab (inference-only, no training large models).

---

## Deferred / Out of Scope (for now)

- Manual review queue (`retrieval_eval_manual_review_*.csv`) is generated but not yet processed
- Graded relevance labels (binary first)
- MeetingQA data integration (not used unless explicitly available and permitted)
- Any training of neural retrieval encoders
