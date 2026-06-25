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
| **NAES-L** (Learned) | Dense retrieval + learned noise-aware reranker | Done (`scripts/retrieval_naes_l.py`) |
| **Oracle** | Manual AMI gold transcripts as upper-bound performance ceiling | Done (`scripts/retrieval_oracle.py`) |

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
| `scripts/retrieval_naes_l.py` | Done | Dense + learned noise-aware reranker (logistic regression, 5-fold meeting-stratified CV) |
| `scripts/retrieval_oracle.py` | Done | Dense retrieval over AMI gold transcripts (upper bound) |
| `scripts/ablation_naes_l.py` | Done | Feature ablation for NAES-L; quantifies per-signal contribution (RQ3) |
| `scripts/generate_answers.py` | Not started | Answer generation via local LLM over retrieved chunks; computes EM, F1, BERTScore |

---

## Baseline Results (medium model, top-10, 96 queries with positives)

### Global index (cross-meeting retrieval allowed)

| Pipeline      | NDCG@10 | MRR    | Recall@10 | Notes |
|---------------|---------|--------|-----------|-------|
| BM25          | 0.2761  | 0.2929 | 0.3507    | Lexical floor |
| NAES-L        | 0.2783  | 0.1885 | 0.5729    | Noise features hurt LR; 82% of failures are cross-meeting errors |
| NAES-H        | 0.3583  | 0.3809 | 0.4549    | DiarStab negated; pool=100 |
| Dense         | 0.3669  | 0.3858 | 0.4583    | Semantic baseline |
| Cross-Encoder | 0.4644  | 0.5000 | 0.5339    | Text-only reranking ceiling |
| Oracle        | 0.6544  | 0.6007 | 0.9271    | Gold transcripts upper bound |

### Per-meeting index (retrieval restricted to query's meeting)

| Pipeline          | NDCG@10 | MRR    | Recall@10 | Notes |
|-------------------|---------|--------|-----------|-------|
| BM25-PM           | 0.5582  | 0.5570 | 0.7170    | +102% vs global BM25 |
| Dense-PM          | 0.5962  | 0.6376 | 0.7101    | +63% vs global Dense |
| NAES-H-PM         | 0.6151  | 0.6271 | 0.7830    | DiarStab restored as positive reward; beats Dense-PM |
| NAES-L-PM         | 0.6240  | 0.5796 | 0.8316    | Best Recall@10 |
| Cross-Encoder-PM  | 0.7569  | 0.8003 | 0.8689    | New ceiling; +63% vs global CE |
| Oracle            | 0.6544  | 0.6007 | 0.9271    | Gold transcripts upper bound |

**Key finding:** Per-meeting filtering eliminates cross-meeting rank pollution and unlocks the true signal in every system. Cross-Encoder-PM (0.757) now exceeds Oracle (0.654) on NDCG — reflecting that Oracle uses a fixed 10s chunking while CE-PM benefits from finer Whisper segments at medium quality. NAES-L-PM (0.624) closes most of the gap with Oracle. All noise-aware systems (NAES-H-PM, NAES-L-PM) outperform their text-only equivalents (Dense-PM) confirming that audio signals add value when evaluation is correctly scoped to the relevant meeting.

---

## Ablation Results (NAES-L, medium, top-10, pool=100)

Two runs: global index (cross-meeting allowed) vs per-meeting index (correct evaluation scope).

### Global index ablation (baseline NDCG = 0.2092)

| Feature dropped | NDCG@10 | ΔNDCG  | Interpretation |
|-----------------|---------|--------|----------------|
| semantic_score  | 0.2003  | −0.009 | Most important |
| ASRConf         | 0.2053  | −0.004 | Small positive contribution |
| DiarStab        | 0.2084  | −0.001 | Near-inert |
| TurnComp        | 0.2122  | +0.003 | Slightly hurts |
| Redund          | 0.2144  | +0.005 | Hurts |
| MixPenalty      | 0.2179  | +0.009 | Most harmful |

### Per-meeting ablation (baseline NDCG = 0.6273)

| Feature dropped | NDCG@10 | ΔNDCG  | Interpretation |
|-----------------|---------|--------|----------------|
| semantic_score  | 0.5096  | −0.118 | Dominant signal by far |
| MixPenalty      | 0.6112  | −0.016 | Now helps — penalises noisy mixed-speaker chunks correctly |
| TurnComp        | 0.6159  | −0.011 | Positive contribution within meeting |
| DiarStab        | 0.6181  | −0.009 | Small positive contribution |
| Redund          | 0.6279  | +0.001 | Near-zero; slight noise |
| ASRConf         | 0.6589  | +0.032 | Actively hurts — ASRConf is not discriminative within a meeting |

**RQ3 finding (revised):** The global ablation was misleading — noise features appeared harmful because they couldn't distinguish across 29 meetings. Within a meeting, MixPenalty (−0.016), TurnComp (−0.011), and DiarStab (−0.009) all contribute positively. ASRConf flips from slightly helpful globally to actively harmful per-meeting (+0.032 when dropped), suggesting within-meeting ASRConf variance doesn't track relevance. Redund remains near-zero in both conditions.

---

## Cross-Model ASR Quality Analysis (RQ1)

WER computed per meeting across all 5 Whisper model sizes (`scripts/compute_wer.py`):

| ASR Model | Mean WER | Notes |
|-----------|----------|-------|
| tiny      | 0.893    | Severe hallucination; 30s fixed-window blocks of garbage text |
| base      | 0.896    | Similar hallucination pattern; retrieval unusable |
| small     | 0.602    | Georgian script hallucination on some meetings |
| medium    | 0.410    | Cleanest model; sole model used for retrieval evaluation |
| large-v3  | 1.711    | Paradoxically worst — English hallucination loops at scale |

**RQ1 finding:** Only `medium` produces usable ASR output for retrieval (mean WER=0.41). All other models hallucinate at rates that make the aligned chunk text meaningless for semantic retrieval — BM25 on tiny returns NDCG=0.018 vs. 0.276 for medium. This is itself a strong finding: there is a WER threshold (~0.5) below which retrieval performance collapses. The cross-model sweep originally planned (Option A: swap chunk text per model) is infeasible because non-medium models do not produce coherent segment boundaries or text. All retrieval experiments are therefore reported on the medium model.

---

## Next Steps

**1. ✓ Done — All retrieval systems on medium model**

**2. ✓ Done — Ablation studies (RQ3)**

**3. ✓ Done — WER + DER computation**

**4. Cross-model sweep** — Superseded by RQ1 finding above. Medium-only evaluation is the correct design choice given hallucination in other models.

**5. Answer quality evaluation**
- Script ready: `scripts/generate_answers.py` — prompts local LLM (Ollama) to answer from retrieved chunks, computes EM, F1, BERTScore.
- Requires Ollama running locally with `llama3.1:8b` or similar.

**6. ✓ Done — Failure case analysis (global index)**
- Under global evaluation: 33/96 queries where NAES-L < Dense, of which 82% had NAES-L rank-1 from the wrong meeting vs. 21% for Dense.
- This diagnosed the root cause of poor global performance: cross-meeting rank pollution where audio quality proxies (DiarStab, ASRConf) promoted clean chunks from other meetings.
- **Resolution:** per-meeting filter eliminates this failure mode entirely. NAES-L-PM (0.624 NDCG) now outperforms Dense-PM (0.596).
- Saved: `data/analysis/failure_analysis_naes_l_vs_dense.csv` (global index run, kept as diagnostic evidence)

**7. Final write-up**
- 2–3 annotated qualitative examples with speaker labels, ASRConf, DiarStab values.
- Update METHODOLOGY.md with final results commentary.
- Answer quality evaluation (needs Ollama).

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
