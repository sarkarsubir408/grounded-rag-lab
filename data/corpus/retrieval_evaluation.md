---
doc_id: retrieval_evaluation
title: Retrieval and answer evaluation
source_url: https://github.com/beir-cellar/beir/wiki/Metrics-available
source_license: BEIR repository declares Apache-2.0; license for the GitHub wiki page is not established
retrieved_date: 2026-11-13
metric_definitions_source: ir-measures documentation (P, R, Success, RR/MRR, nDCG)
metric_definitions_url: https://ir-measur.es/en/latest/measures.html
hit_rate_reference_note: Project Hit Rate@K is the per-query binary Success@K indicator; it is distinct from Recall@K.
evaluation_guidance_source: OpenAI Cookbook prompt-engineering evaluation guidance
provenance: Project-authored summary; metric definitions checked against ir-measures documentation. The BEIR page lists available metrics, and the OpenAI Cookbook provides general evaluation guidance. No reuse permission is claimed for the referenced documentation.
---

# Retrieval and answer evaluation

<!-- passage_id: eval-01 -->
Retrieval evaluation needs explicit relevance judgments connecting each question to supporting passages. These judgments are often called qrels. Keep a rationale and reviewer status with every judgment so that provisional annotations are not mistaken for verified ground truth. A query with no supporting passage is an unanswerable case and should be analyzed separately from answerable-query ranking metrics.

<!-- passage_id: eval-02 -->
Precision@K is the fraction of the top K retrieved passages that are relevant. Recall@K is the fraction of all known relevant passages found in the top K. Hit Rate@K records whether at least one relevant passage appears in the top K. These measures answer different questions about retrieval quality.

<!-- passage_id: eval-03 -->
Mean Reciprocal Rank (MRR) gives each query the reciprocal of the first relevant result's rank, with zero when no relevant result appears. Normalized Discounted Cumulative Gain (nDCG) rewards relevant results near the top while accounting for graded relevance when judgments provide grades. With binary labels, relevant passages receive grade one and others grade zero.

<!-- passage_id: eval-04 -->
Macro averages first calculate each metric for each query and then take the mean across queries. This prevents queries with many labeled passages from dominating the aggregate. Always report per-query results alongside macro values; small question sets can hide large differences between individual cases.

<!-- passage_id: eval-05 -->
Use a validation split to choose configuration values such as an abstention threshold, then evaluate the selected values on a separate test split. Selecting and reporting a threshold on the same questions gives an optimistic estimate. If the validation set is small, report the threshold and classification results as provisional.

<!-- passage_id: eval-06 -->
Answer correctness, contextual relevance, faithfulness, and citation validity are related but distinct. A valid citation identifier only proves that the referenced chunk was retrieved. Human review is still needed to judge whether claims are supported by the cited text and whether the response answers the question correctly.
