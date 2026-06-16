 # Methodology Notes

Detailed bullet points for the thesis methodology chapter. Organized by pipeline stage. Update this file after each significant implementation change.

---

## 1. Dataset

- **Corpus:** AMI Meeting Corpus — a collection of scenario-based meetings recorded in instrumented meeting rooms, totalling approximately 100 hours of multi-party audio.
- **Subset used:** A manually filtered subset of meetings from `data/raw_audio/`, selected for usability based on audio quality, speaker count, and meeting duration.
- **Usability criteria:** Meetings were filtered using a manifest (`data/eval/meeting_manifest.csv`) that marks each meeting as usable or not based on criteria defined in the filtering script. Unusable meetings include those with excessively short duration, missing diarization output, or corrupted audio.
- **Audio format:** WAV files; each meeting is a single mixed-channel recording of multiple speakers.
- **Speaker setup:** AMI meetings typically involve 3–5 participants. Each participant has an individual headset microphone in addition to the mixed-channel recording; the mixed-channel recording is used in this work to reflect realistic deployment conditions.
- **Meeting topics:** Scripted product design scenarios; meetings are structured in phases (e.g. project kick-off, design review, evaluation), ensuring topical diversity across the corpus.

---

## 2. Automatic Speech Recognition (ASR)

- **Model family:** OpenAI Whisper, accessed via `faster-whisper` (a CTranslate2-based reimplementation of Whisper with improved inference speed).
- **Model sizes evaluated:** `tiny`, `base`, `small`, `medium`, `large-v3` — chosen to span a wide range of word error rates (WER), enabling WER-stratified analysis of downstream retrieval degradation.
- **Inference settings:**
  - Device: CPU (Apple M2 Pro); compute type: `int8` for memory efficiency.
  - Task: transcription (not translation).
  - Beam size: 5; temperature: 0 (greedy decoding); VAD filter: disabled.
  - Language: auto-detected per file.
- **Output structure:** Per model size, outputs are stored in `data/asr_outputs/<model>/` with subdirectories for raw JSON (full Whisper output), segment-level JSON (start/end/text/logprob per segment), plain-text transcripts, run logs, and checkpoint indices.
- **Resumability:** Processing is fully checkpointed via a per-model CSV index (`checkpoints/asr_index.csv`). Re-running the script skips already-processed files. Failed files can be retried with `--retry-failed`.
- **Per-segment fields retained:** `start`, `end`, `text`, `avg_logprob`, `no_speech_prob`, `compression_ratio`. These are used downstream for feature computation.
- **`avg_logprob` as ASR confidence proxy:** Whisper does not output token-level probabilities in a standardized form; `avg_logprob` (average log-probability of tokens in the segment) is used as a segment-level ASR confidence signal. Values are min-max normalized per meeting to produce `ASRConf ∈ [0, 1]`.

---

## 3. Speaker Diarization

- **Model:** pyannote.audio speaker diarization pipeline (version 3.x), run once on the full audio subset and not re-run.
- **Output format:** Per-meeting JSON files in `data/diarization_outputs/json/`, each containing a list of diarization segments with fields: `start`, `end`, `speaker`, `confidence`, `seg_consistency`, `overlap`, `flip_rate`.
- **One-time processing rationale:** Diarization is computationally expensive and does not depend on ASR model choice. Running it once and reusing it across all ASR model experiments ensures consistent speaker assignments and isolates ASR errors as the independent variable.
- **Diarization quality signals retained per segment:**
  - `confidence`: pyannote's speaker assignment confidence score.
  - `seg_consistency`: consistency of the speaker label within the diarization segment.
  - `overlap`: fraction of the segment overlapping with another active speaker.
  - `flip_rate`: rate of speaker label changes within the segment window.
  These are used to compute the `DiarStab` feature downstream.

---

## 4. ASR–Diarization Alignment

- **Alignment unit:** Whisper ASR segments are used as the base retrieval chunks. Diarization is aligned to ASR segments by time overlap, not by exact boundary matching. This is because Whisper and pyannote operate independently and produce non-aligned boundaries.
- **Dominant-speaker assignment:** For each ASR segment, all diarization segments with non-zero temporal overlap are identified. The diarization segment with the largest overlap (seconds) is assigned as the dominant speaker. In case of tie, higher confidence wins.
- **Multi-speaker handling:** Rather than collapsing all multi-speaker ASR segments into a single speaker, the alignment preserves the overlap structure. A `Purity` score (dominant overlap / total segment duration) and `MixPenalty = 1 − Purity` are computed to numerically capture the degree of speaker mixing.
- **Features computed per aligned chunk:**

  | Feature | Description |
  |---------|-------------|
  | `ASRConf` | Min-max normalized `avg_logprob` of the ASR segment across the meeting |
  | `DiarStab` | Composite stability score: dominant-speaker fraction × weighted confidence × seg_consistency × (1 − overlap) × (1 − flip_rate) |
  | `TurnComp` | Turn completeness: combination of segment duration, punctuation at end of text, token count, and no-speech probability |
  | `Redund` | Maximum Jaccard similarity between the chunk's token set and its immediate neighbors (detects repeated or duplicated speech) |
  | `Purity` | Fraction of segment duration covered by the dominant speaker's diarization segment |
  | `MixPenalty` | 1 − Purity; penalizes segments with significant multi-speaker overlap |

- **Output:** Per-model aligned chunk CSVs in `data/aligned_chunks/<model>/` and per-chunk feature tables in `data/metrics/<model>/`.
- **Unknown speaker handling:** If no diarization segment overlaps with an ASR segment (e.g. a long silence or a segment at the very start/end of the recording), the speaker is assigned `UNKNOWN` and `DiarStab = 0`.

---

## 5. Evaluation Dataset Construction

The evaluation dataset consists of retrieval queries with labeled relevant chunks and reference answers, constructed synthetically from the aligned transcript corpus. The construction pipeline applies multiple quality filters to ensure the dataset is non-trivial and suitable as a retrieval benchmark.

### 5.1 Chunk Quality Pre-filtering

Before any chunk is shown to the LLM or used as a query source, it must pass the following thresholds:

- `ASRConf > 0.35` — excludes low-confidence transcriptions that are likely garbled
- `TurnComp > 0.3` — excludes incomplete utterances (too short or cut off mid-sentence)
- `MixPenalty < 0.6` — excludes heavily mixed-speaker segments where attribution is unreliable
- `len(text.split()) > 8` — excludes segments too short to carry meaningful information

This filter is applied both when building the transcript sections shown to the LLM for query generation and when selecting candidate relevant chunks for relevance judging. Its effect is to prevent the LLM from generating queries based on garbled or uninformative text, and to prevent low-quality chunks from being labeled as relevant.

### 5.2 Query Generation

- **Approach:** The meeting transcript is split into sections of approximately 1200 words (skipping the first 300-word preamble, which typically contains administrative setup). Each section is passed to a local LLM (Ollama `llama3.1:8b`) prompted to generate up to 3 retrieval questions.
- **Prompt design:** The prompt explicitly prohibits:
  - Yes/no questions
  - Questions answerable from general knowledge without reading the transcript
  - Paraphrase questions (direct reformulations of a single sentence)
  - Vague questions ("What did the team discuss?", "What is X?")
  It requires questions to target specific decisions, outcomes, constraints, cost estimates, assignments, or factual claims — specific enough that only 1–3 passages in the entire meeting could answer them.
- **Motivation for strict prompt:** Earlier experiments with permissive prompts produced a high proportion of trivial queries that any retrieval system would answer at rank 1, providing no discriminative signal. The strict prompt is designed to force the LLM to generate queries that require genuine retrieval.
- **Deduplication:** Queries are deduplicated by their first 6 content tokens (stopwords removed) to avoid near-identical questions from overlapping sections.
- **Cap:** At most 12 queries are retained per meeting.

### 5.3 Relevance Labeling

- **Candidate selection:** For each query, all quality-filtered chunks in the meeting are ranked by Jaccard similarity between query content tokens and chunk content tokens. The top candidates (up to 35) above a similarity threshold of 0.25 are passed to the LLM for relevance judgment.
- **LLM relevance judgment:** Each candidate chunk is judged independently with a binary YES/NO prompt asking whether the passage directly addresses the question. The LLM is instructed to answer YES only if the passage directly addresses the question, not merely shares some vocabulary.
- **Limitation of window-based labeling:** The generation window that produced a query may not contain all relevant passages in the meeting. Labeling only within-window candidates would produce false negatives (relevant chunks that are never seen).
- **Post-hoc semantic scan for missed positives:** After LLM labeling, all quality-filtered chunks in the meeting are encoded with `sentence-transformers/all-MiniLM-L6-v2`. Chunks with cosine similarity to the query embedding above 0.60 that were not already labeled relevant are submitted to the LLM relevance judge. Confirmed positives are added to the relevant set and tagged `found_by = "semantic_scan"`. This cross-window scan addresses the false-negative problem and improves recall of the positive label set.

### 5.4 Cross-Encoder Re-scoring

- After relevance labeling, each `(query, positive_chunk)` pair is scored independently with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- The score is stored as `ce_score` in the label rows.
- Label rows with `relevance = 1` and `ce_score < 0.2` are flagged as `label_quality = "suspect"`.
- Flagged labels are not automatically removed but are available for threshold-based filtering during evaluation.
- **Rationale:** The LLM relevance judge operates on surface text similarity and may over-label chunks that share vocabulary but do not actually answer the question. The cross-encoder provides an independent signal trained specifically for passage relevance to a query.

### 5.5 Reference Answer Generation

- **Context construction:** The top 3 relevant chunks are assembled into a context string. Each chunk is prefixed with its speaker label: `[Passage N] [SPEAKER_XX]: <text>`. Speaker attribution is included because in meeting QA, who said something is often part of a complete answer (e.g. "The project manager said the budget is £500").
- **Prompt design:** The LLM is instructed to answer using only information explicitly stated in the provided passages, without inferring or generalizing. If the passages do not contain sufficient information, the LLM is instructed to return the exact string `UNANSWERABLE`.
- **Post-generation filtering:** Answers flagged as `UNANSWERABLE`, or shorter than 5 words, are set to empty string and the query is available for manual review.

### 5.6 NLI Faithfulness Check

- After answer generation, an NLI model (`cross-encoder/nli-deberta-v3-small`) checks whether the reference answer is entailed by at least one of the relevant chunks.
- If no chunk entails the answer (all chunk–answer pairs yield `contradiction` or `neutral`), the query is flagged `answer_faithful = False`.
- **Rationale:** LLMs instructed to extract answers can still produce answers that draw on parametric knowledge rather than the provided context. The NLI check catches reference answers that are not grounded in the labeled chunks, which would make them unreliable as gold answers for evaluation.

### 5.7 Difficulty Filtering

Two difficulty flags are computed per query and stored in the `difficulty_flag` column of the queries CSV:

- **`trivial_overlap`:** ROUGE-1 recall between query content tokens and the tokens of any relevant chunk exceeds 0.5. These are paraphrase queries — the query text is a near-direct restatement of the relevant chunk text. Every retrieval system will find them at rank 1, so they provide no discriminative signal between systems.
- **`bm25_trivial`:** BM25 retrieval (over all meeting chunks using `rank_bm25`) returns the relevant chunk at rank 1. These queries are solvable by simple keyword matching; including them would inflate all systems' metrics equally. If `trivial_overlap` is already set, it takes precedence.
- **`no_positives`:** The query has no labeled relevant chunks. These queries cannot be used to compute retrieval metrics (no hits are possible).
- **Evaluation practice:** Aggregate metrics are reported both on the full query set and on the subset with `difficulty_flag = "ok"`. The non-trivial subset is the primary evaluation set. This follows standard IR evaluation practice (used in BEIR, TREC, MS MARCO) of evaluating only on judged, non-trivial queries.

### 5.8 Negative Sampling

- For each query with at least one positive chunk, up to 5 hard negatives are sampled.
- Negatives are drawn from the pool of non-positive chunks ranked by Jaccard similarity to the query.
- The top 10% of the pool (most similar non-positives) is skipped to avoid sampling chunks that might actually be relevant but were not labeled as such.
- Negatives are sampled from the middle of the ranked pool (moderately similar but confirmed non-relevant).
- Hard negatives are stored in the label rows with `relevance = 0` and are used to support learning-to-rank training for NAES-L.

### 5.9 Manual Human Review

- Following automated construction, a manual human review pass was conducted over the generated query set.
- Review focused on:
  - Queries with `difficulty_flag = "ok"` and `answer_faithful = False` — these have a relevant chunk but an NLI-unfaithful reference answer, requiring manual correction or removal.
  - Queries where the reference answer is semantically empty, circular, or does not directly address the question.
  - Queries that slipped through the prompt filter and are still vague or trivially paraphrase-like.
  - Queries with `no_positives` where manual inspection reveals a relevant chunk that the automated pipeline missed.
- **Rationale for including manual review:** Automated construction with LLM judging and NLI faithfulness checking substantially reduces the burden of manual verification, but cannot guarantee correctness. Manual review provides the final quality gate and is disclosed as part of the methodology to support reproducibility and scientific validity.
- The combination of automated quality filters (chunk pre-filter, CE re-scoring, NLI faithfulness, difficulty flags) and manual review is a standard and accepted practice in dataset construction for IR benchmarks (cf. MS MARCO, BEIR, QAMPARI).

### 5.10 Checkpointing and Reproducibility

- Query generation is fully checkpointed per meeting. Each completed meeting is saved to `data/eval/checkpoints/<audio_id>.json`.
- Re-running the script skips already-processed meetings, allowing incremental processing and recovery from interruptions.
- Old checkpoints must be explicitly cleared (via `--clear-checkpoints`) when the data schema changes, to prevent mixing old and new formats.
- All thresholds and constants are configurable via environment variables, documented in the script header.
- **Circular evaluation concern:** The evaluation dataset is constructed from `medium` Whisper chunks by design choice. This means `medium`-model queries may be marginally easier for the `medium` retrieval system. This is mitigated by: (a) the chunk quality pre-filter using only the least-noisy chunks regardless of model; (b) cross-model experiments where `large-v3` and `tiny` are evaluated against queries generated from `medium` — these cross-model comparisons are conservative and are the primary axis of analysis; (c) the Oracle upper bound uses gold transcripts entirely independent of Whisper output.

---

## 6. Retrieval Systems

All systems retrieve from the same aligned chunk corpus for a given ASR model size. Per-model experiments run each retrieval system over the aligned chunks for each Whisper model size, enabling WER-stratified comparison.

### 6.1 Dense Baseline

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense embeddings).
- **Index:** FAISS `IndexFlatIP` over L2-normalized chunk embeddings. Inner product of normalized vectors equals cosine similarity.
- **Embedding cache:** Chunk embeddings are cached per ASR model to `data/metrics/chunk_embeddings_minilm_<model>.npz` and reloaded on subsequent runs if chunk IDs have not changed.
- **No reranking.** Ranking is purely by cosine similarity between query and chunk embeddings.

### 6.2 BM25 (Lexical Baseline)

- **Library:** `rank_bm25` (BM25Okapi).
- **Tokenization:** Lowercased, regex-tokenized (`[a-z0-9]+`), with a fixed stopword list removed. The same stopword list is used in the query generation script for consistency.
- **Index:** A single `BM25Okapi` index built over all aligned chunk texts for the given ASR model. No stemming or lemmatization is applied — the corpus vocabulary reflects raw ASR transcription output, including recognition errors.
- **Scoring:** BM25 Okapi scores are computed per query token, summed across tokens. Chunks with a score of 0 (no token overlap) are excluded from results.
- **Output convention:** Identical to the dense baseline — per-query results, per-query metrics, and a summary CSV written to `data/retrieval_results/bm25/` and `data/metrics/bm25/`.
- **Role in evaluation:** BM25 serves as the lexical performance floor. If a neural system cannot beat BM25, it adds no value over keyword matching. The gap between BM25 and the dense baseline quantifies the benefit of semantic retrieval. The gap between BM25 and NAES-L quantifies the total benefit of neural + noise-aware reranking.

### 6.3 Cross-Encoder Reranker

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Pipeline:** Two-stage — (1) dense cosine-similarity retrieval with MiniLM to produce a candidate pool of size N (default 50); (2) each (query, passage) pair in the pool is scored jointly by the cross-encoder, which reads both together and produces a relevance score. The top-K of the reranked list is the final result.
- **Why two stages:** Cross-encoders are too slow to score all chunks in the corpus; the dense retrieval pool acts as a fast pre-filter. The pool size (50) is set large enough that the relevant chunks are almost certainly retrieved in stage 1, so the cross-encoder decides ordering rather than recall.
- **Significance:** This is the most important text-only comparison point. If NAES-L beats the cross-encoder, it demonstrates that audio-specific signals (ASR confidence, diarization stability) add value beyond what text-only reranking can provide. If NAES-L does not beat it, the contribution of audio signals is limited to cases where text-based reranking fails.
- **`--rerank-pool` argument:** Configurable (default 50). Increasing it improves recall of the pool at the cost of more cross-encoder inference time.

### 6.4 NAES-H (Heuristic Noise-Aware Reranking)

Reranking formula:

```
R(c) = α·s(q,c) + β·ASRConf(c) + γ·DiarStab(c) + δ·TurnComp(c) − ε·Redund(c) − μ·MixPenalty(c)
```

- `s(q,c)` is the dense cosine similarity score from the initial retrieval stage (first-stage dense retrieval pool of 50 candidates, same as cross-encoder).
- All feature values are in [0, 1] by construction — no additional normalization needed at scoring time.
- `Redund` is subtracted: high token overlap with neighboring chunks indicates a repeated or low-information segment.
- `MixPenalty` is subtracted: heavily mixed-speaker chunks are less reliable for speaker-attributed retrieval.
- `DiarStab` and `MixPenalty` are kept as separate terms to allow independent ablation of each signal.
- Default weights: α=1.0, β=0.3, γ=0.3, δ=0.2, ε=0.1, μ=0.4. All weights are configurable via CLI flags for manual tuning experiments.
- The weights stored in the summary CSV alongside metrics, so any run is fully reproducible from the output file alone.

### 6.5 NAES-L (Learned Noise-Aware Reranking)

- Same feature set as NAES-H: `[s(q,c), ASRConf, DiarStab, TurnComp, Redund, MixPenalty]`.
- **Training data:** (query, chunk, binary_relevance) triples from `data/eval/retrieval_eval_labels.csv`. Only queries with `difficulty_flag = "ok"` and at least one positive are used.
- **Model:** Logistic regression (`sklearn.linear_model.LogisticRegression`). The learned weights replace the manually set weights in the NAES-H formula. Logistic regression is chosen because (a) the feature set is small and scalar, (b) the model is interpretable — coefficients directly correspond to each signal's contribution — and (c) it adds negligible inference overhead.
- **Cross-validation:** Leave-one-meeting-out (LOMO) — for each meeting, train on triples from all other meetings, predict on the held-out meeting. This avoids leakage: the model never sees chunks or queries from the meeting being evaluated.
- **Optimization target:** NDCG@10 on the held-out fold, used to select the regularization strength `C` (grid search over `[0.01, 0.1, 1.0, 10.0]`).
- **Inference:** Given a dense retrieval pool of 50 candidates, score each with the learned logistic regression, sort descending by predicted relevance probability, return top-K.
- **Primary thesis contribution:** NAES-L is the principal proposed method. The comparison NAES-L vs. Cross-Encoder answers RQ2 (do audio signals add value over text-only reranking?). The comparison NAES-L vs. NAES-H answers whether learned weights outperform heuristic weights on this data.

### 6.6 Ablation Studies

- To answer RQ3 (per-signal contribution), NAES-L is retrained with one feature zeroed out at a time.
- Ablation schedule: remove ASRConf, DiarStab, TurnComp, Redund, MixPenalty individually; keep all other features at their learned weights.
- Metric: NDCG@10 delta vs. full NAES-L. A large negative delta indicates that feature is important; near-zero indicates it contributes little on this data.
- Results reported as a table: feature dropped → NDCG@10 → ΔNDCG.

### 6.7 Oracle

- Uses manual AMI gold transcripts (`data/gold_transcripts/`) as the retrieval corpus instead of Whisper ASR output.
- Provides the theoretical upper bound: performance achievable with perfect transcription and perfect speaker attribution.
- The gap between Oracle and Dense (medium) quantifies the total degradation attributable to ASR and diarization errors.

---

## 7a. WER Computation

- **Purpose:** Per-meeting WER is required for WER-stratified analysis (RQ1, RQ4). Each meeting is assigned to a WER tier (0–15%, 15–30%, >30%) based on its Whisper model output.
- **Method:** Whisper plain-text transcript (from `data/asr_outputs/<model>/transcripts/`) is aligned against the AMI gold transcript (from `data/gold_transcripts/`) using the `jiwer` library.
- **Normalization before comparison:** Both hypothesis and reference are lowercased, punctuation removed, and consecutive whitespace collapsed. This matches standard WER evaluation practice and ensures differences in punctuation/casing do not inflate WER.
- **Output:** `data/analysis/wer_per_meeting_<model>.csv` with columns: `audio_id`, `wer`, `num_words_ref`, `num_substitutions`, `num_deletions`, `num_insertions`.
- **Script:** `scripts/compute_wer.py` (not yet created).

---

## 7. Evaluation Protocol

### 7.1 Retrieval Metrics

- **NDCG@5** — normalized discounted cumulative gain at rank 5; primary metric. Rewards early placement of relevant chunks.
- **MRR** — mean reciprocal rank; measures how early the first relevant chunk appears in the ranked list.
- **Recall@K** — fraction of all relevant chunks for a query that appear in the top K results.
- Binary relevance labels are used (1 = relevant, 0 = not relevant).

### 7.2 Answer Quality Metrics

- **Exact Match (EM)** — binary; 1 if the generated answer exactly matches the reference answer after text normalization (lowercase, punctuation removal).
- **Token-level F1** — harmonic mean of precision and recall over shared tokens between generated and reference answer.
- **BERTScore** — semantic similarity between generated and reference answer using contextual embeddings; more robust to paraphrase than EM or F1.

### 7.3 Stratification Dimensions

Results are reported in aggregate and stratified by:

- **WER tier:** 0–15% (low noise), 15–30% (medium), >30% (high). WER is computed per meeting by aligning Whisper output against AMI gold transcripts using standard edit distance.
- **Meeting length:** Short / medium / long by total word count or duration quintile.
- **Number of active speakers:** As determined by the diarization output.
- **Difficulty tier:** Full set vs. `difficulty_flag = "ok"` subset.

### 7.4 Query Set Filtering

- Evaluation metrics are computed only over queries with `difficulty_flag != "no_positives"` (at least one labeled relevant chunk exists).
- Primary evaluation uses `difficulty_flag = "ok"` queries (non-trivial, non-BM25-solvable).
- Full-set results (including trivial queries) are reported for completeness.

### 7.5 Answer Quality Evaluation

- **Pipeline stage:** After retrieval/reranking, the top-K retrieved chunks are passed as context to a local LLM for answer generation.
- **LLM:** Mistral-7B-Instruct or LLaMA-3-8B-Instruct, quantized to 4-bit or 8-bit precision, running locally via Ollama. No fine-tuning.
- **Prompt format:** Query + top-K chunks (with speaker labels preserved) → system instruction to answer from the provided passages only, return `UNANSWERABLE` if the passages do not contain the answer.
- **Metrics computed against reference answers from `retrieval_eval_queries.csv`:**
  - **Exact Match (EM):** 1 if normalized generated answer equals normalized reference answer; else 0.
  - **Token-level F1:** Harmonic mean of token precision and recall between generated and reference.
  - **BERTScore:** Semantic similarity using contextual embeddings (`bert-score` library, `deberta-xlarge-mnli` or similar).
- **Stratification:** Same WER tiers and difficulty splits as retrieval metrics.
- **Script:** `scripts/generate_answers.py` (not yet created).
- **Scope note:** Answer quality evaluation is a secondary deliverable; retrieval metrics are the primary contribution of this thesis.

### 7.6 Failure Case Analysis

- For queries where NAES-L ranks below the dense baseline, the relevant chunks, `ASRConf`, and `DiarStab` values are inspected.
- Anticipated failure modes:
  - High `ASRConf` but incorrect transcription of proper nouns or domain-specific terminology.
  - Within-turn diarization errors that do not trigger `DiarStab` (errors contained inside a single diarization segment boundary).
- 2–3 qualitative case studies: query + reference answer + cross-encoder top result + NAES-L top result, annotated with feature values.
