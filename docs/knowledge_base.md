# Technical knowledge base

The public-release corpus contains six documents. Five are project-authored technical summaries; chunking_guide.md is synthetic educational content created with ChatGPT assistance. The available source and provenance information is recorded in data/source_manifest.json.

## RAG workflow

A basic retrieval-augmented generation system separates document loading, normalization, chunking, indexing, retrieval, context assembly, and generation. This makes it easier to locate failures: retrieval can return weak evidence even when the prompt is clear, and a model can misuse relevant evidence. See [rag_workflow.md](../data/corpus/rag_workflow.md).

## Chunking and context

Chunking splits documents into smaller passages for embedding and retrieval. Fixed-size chunking is simple and consistent; overlap can retain context near boundaries but may also introduce redundant chunks. Semantic boundaries and document structure can help preserve meaning. The current public-release chunking guide discusses these trade-offs and evaluation: [chunking_guide.md](../data/corpus/chunking_guide.md).

## Dense retrieval and embeddings

An embedding maps text into a vector space. Dense search compares query and passage vectors, often with cosine similarity. Query and corpus vectors must come from compatible models. Similarity is a ranking signal, not a probability of truth or relevance. Dense retrieval can match paraphrases but may miss exact rare identifiers. See [dense_embeddings.md](../data/corpus/dense_embeddings.md).

## BM25 and hybrid search

BM25 is a lexical ranker using term frequency, inverse document frequency, and length normalization. It is often useful for exact names and rare terms; its score scale depends on the query and corpus. Reciprocal Rank Fusion combines rank positions without directly adding incomparable BM25 and cosine score magnitudes. With ranks starting at 1, an item receives 1 / (k + rank) from each ranking where it appears. RRF scores are not confidence probabilities. See [bm25_retrieval.md](../data/corpus/bm25_retrieval.md) and [hybrid_rrf.md](../data/corpus/hybrid_rrf.md).

## Retrieval evaluation

The reviewed evaluation set has 25 questions: 8 validation questions (6 answerable, 2 unanswerable) and 17 held-out test questions (15 answerable, 2 unanswerable). Precision@K, Recall@K, Hit Rate@K, MRR, and nDCG describe different ranking properties. Per-query results expose misses that macro averages can hide. See [retrieval_evaluation.md](../data/corpus/retrieval_evaluation.md) and [the experiment report](retrieval_experiments.md).

All 21 answerable questions reference existing corpus passages. Unanswerable questions have no gold evidence; their ranking metrics are undefined and excluded from answerable-query macro averages. Valid evidence mapping does not independently verify every derived chunk-level relevance judgment.

## Abstention and answer review

The app's provisional method-specific thresholds were calibrated on C01–C08 only, for 100/20 chunks at K=5 and RRF constant 60. They are retrieval signals, not confidence probabilities. Test questions were not used. See [calibration metadata](../reports/public_release/calibration_metadata.json). Citation ID validation confirms that a cited ID was supplied to the model; it does not establish semantic support, factual correctness, contextual relevance, or faithfulness. Those properties require separate answer review.

## Source and reuse notes

chunking_guide.md is synthetic educational content generated with ChatGPT assistance; no exact generation prompt or external source URL is recorded, and exact external provenance has not been independently established. The five other corpus documents are project-authored summaries; source titles, URLs, retrieval dates, and available license/provenance statements are recorded in [source_manifest.json](../data/source_manifest.json). Referenced source materials retain their own rights. The project's MIT software license does not apply to corpus content; see [CONTENT_NOTICE.md](../CONTENT_NOTICE.md).