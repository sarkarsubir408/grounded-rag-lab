---
doc_id: bm25_retrieval
title: BM25 lexical retrieval
source_url: https://github.com/dorianbrown/rank_bm25/blob/master/README.md
source_license: Apache-2.0 (rank_bm25 repository)
retrieved_date: 2026-11-13
provenance: Project-authored summary. The rank_bm25 README is the software reference; the separate research-paper citation is recorded below and in data/source_manifest.json.
paper_title: Improvements to BM25 and Language Models Examined
paper_authors: Andrew Trotman; Antti Puurula; Blake Burgess
paper_publication: Proceedings of the 19th Australasian Document Computing Symposium (ADCS '14), 2014, pp. 58–65
paper_doi: 10.1145/2682862.2682863
paper_reference_url: https://doi.org/10.1145/2682862.2682863
paper_author_copy_url: https://cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
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
