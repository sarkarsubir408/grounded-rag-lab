---
doc_id: hybrid_rrf
title: Hybrid retrieval with reciprocal rank fusion
source_url: https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/rrf/
source_license: Apache-2.0 (OpenSearch documentation repository)
retrieved_date: 2026-11-13
provenance: Project-authored summary; no third-party prose copied. The original RRF paper is cited in source_manifest.json and is not reproduced.
---

# Hybrid retrieval with reciprocal rank fusion

<!-- passage_id: hybrid-01 -->
Hybrid retrieval combines complementary ranking signals, such as dense semantic similarity and BM25 lexical relevance. Dense retrieval can find paraphrases; lexical retrieval can preserve exact term matches. Running both against the same chunk collection gives the system two candidate rankings for one question.

<!-- passage_id: hybrid-02 -->
Reciprocal rank fusion (RRF) combines result lists by position. For a chunk d, its score is the sum of 1 divided by (k plus its rank) across the rankings where it appears. Ranks start at one; a missing chunk in a ranking contributes zero. The configurable constant k controls how quickly lower-ranked results lose influence.

<!-- passage_id: hybrid-03 -->
The common RRF constant k=60 is a starting configuration, not a proven optimum for every collection. A smaller constant emphasizes top ranks more strongly; a larger constant narrows the difference between adjacent ranks. Compare candidate values against held-out relevance judgments before claiming a tuned value.

<!-- passage_id: hybrid-04 -->
RRF uses ranks rather than the original score magnitudes, so cosine similarity and BM25 scores do not need score normalization before fusion. An RRF score is still not a confidence probability. Its value depends on the rank constant and result lists, so an absolute RRF cutoff is not a general answerability threshold.

<!-- passage_id: hybrid-05 -->
Fusion depth determines how many results from each retriever can contribute to the combined ranking. If a relevant chunk falls outside a short candidate list, RRF cannot reward it for appearing in both systems. Use the same candidate depth in a controlled experiment and record it with the RRF constant.
