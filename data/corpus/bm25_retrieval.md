---
doc_id: bm25_retrieval
title: BM25 lexical retrieval
source_url: https://github.com/dorianbrown/rank_bm25/blob/master/README.md
source_license: Apache-2.0 (rank_bm25 repository)
retrieved_date: 2026-11-13
provenance: Project-authored summary; no third-party prose copied. BM25 background references are listed in source_manifest.json.
---

# BM25 lexical retrieval

<!-- passage_id: bm25-01 -->
BM25 is a lexical ranking function used in information retrieval. It scores a document using query-term occurrence, inverse document frequency, and document-length normalization. Terms that are frequent in a particular document can contribute to its score, while terms common across the whole collection usually provide less discrimination.

<!-- passage_id: bm25-02 -->
BM25 uses term-frequency saturation: repeating a query term can raise a passage's score, but each additional occurrence contributes less than the previous one. Its length-normalization component reduces the tendency for long documents to score highly just because they contain more terms. The BM25 parameters control these effects.

<!-- passage_id: bm25-03 -->
Lexical retrieval is often useful for exact names, rare terms, product codes, and identifiers. It can miss relevant passages that express the same concept with different vocabulary. Tokenization choices, punctuation handling, and document length all affect the ranking and should be held constant when comparing systems.

<!-- passage_id: bm25-04 -->
BM25 scores are query- and corpus-dependent ranking signals. A larger raw score is not a universal measure of answerability and should not be compared directly with a cosine similarity score. In hybrid retrieval, combining ranks avoids assuming the score scales of lexical and dense methods are interchangeable.
