# Portfolio-ready project wording

These descriptions refer to the current public-release version of Grounded RAG Lab. Keep the limitations explicit. Do not claim production readiness, live API validation, independently audited chunk-level qrels, or measured answer quality.

## Resume project entry

**Grounded RAG Lab | Python, Sentence Transformers, BM25, RRF, Streamlit**

- Built a modular RAG retrieval study and Streamlit prototype over six documents, with TXT, Markdown, and text-based PDF ingestion, in-memory indexing, dense retrieval, BM25, and Reciprocal Rank Fusion.
- Evaluated 25 project-owner-approved questions (8 validation, 17 held-out test) across a 4 × 3 retrieval matrix, K=3 and K=5, reporting 408 per-query rows and 24 macro summaries for Precision, Recall, Hit Rate, MRR, and nDCG.
- Mapped all 21 answerable questions to existing corpus passages and kept the four unanswerable cases evidence-free; unanswerable questions are excluded from answerable-query retrieval averages.
- Added corpus-fingerprint-bound abstention thresholds from the eight-question validation split. Thresholds are provisional and treated as retrieval signals, not probabilities.
- Implemented a gated OpenAI Responses API adapter and one-request authorization flow; automated tests use mocks, and live API behavior has not been validated.
- Performed six manual UI smoke checks on 2026-11-16 (C01 and C06 for each retrieval method). They checked retrieval-gate outcomes, signals, and thresholds; supporting evidence ranks were noted where evidence appeared. Comprehensive UI validation and generated-answer testing remain incomplete. See the [validation report](../reports/public_release/validation_results.md).

## LinkedIn project description

I built Grounded RAG Lab as a small retrieval experimentation project and document-question-answering prototype. The public-release benchmark uses six documents and compares dense, BM25, and RRF hybrid retrieval across four chunk-size/overlap configurations at K=3 and K=5. On this dataset, 150/20 had the highest macro nDCG@5 for all three methods, while 100/40 had the highest Precision@5. The app supports TXT, Markdown, and text-based PDF ingestion, displays retrieved source references, and includes a credential-gated OpenAI Responses API adapter. Abstention thresholds use only eight validation questions and remain provisional. Automated tests use mocked providers. Six manual smoke checks covered two questions per retrieval method and retrieval-gate behavior; supporting evidence ranks were noted where evidence appeared. They did not test generated answers. No live API calls were made, and comprehensive manual Streamlit UI validation remains incomplete. See the [validation report](../reports/public_release/validation_results.md).

## Interview explanation outline

1. **Why compare three retrievers?** Dense search may match paraphrases, BM25 may rank exact terms well, and RRF combines ranking positions without treating cosine and BM25 scores as directly comparable.
2. **How was retrieval measured?** Fifteen answerable held-out questions contribute to macro ranking metrics; two unanswerable test questions retain empty qrels and are not scored as retrieval misses. K=3 had one observed miss (T03, Dense 100/20); K=5 had none.
3. **How does abstention work?** Method-specific thresholds were selected from C01–C08 only, bound to the corpus fingerprint, embedding setup, chunk configuration, K, and RRF constant. They are provisional signals, not confidence probabilities.
4. **What remains to prove?** Complete comprehensive manual UI validation, independently audit chunk-level relevance mappings, evaluate answer correctness/faithfulness/citation support, and test live provider behavior only with explicit authorization. The current results do not establish production readiness.
