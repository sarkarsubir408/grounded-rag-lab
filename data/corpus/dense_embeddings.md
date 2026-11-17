---
doc_id: dense_embeddings
title: Dense retrieval and text embeddings
source_url: https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/semantic-search/README.md
source_license: Apache-2.0 (Sentence Transformers repository)
retrieved_date: 2026-11-13
provenance: Project-authored summary; no third-party prose copied.
---

# Dense retrieval and text embeddings

<!-- passage_id: dense-01 -->
A text embedding is a numeric vector produced from text by an embedding model. Texts with related meanings can map to nearby vectors, allowing a retrieval system to find conceptually related passages even when a question and passage use different words. Embeddings represent model-dependent similarity; they are not factual truth scores.

<!-- passage_id: dense-02 -->
For dense semantic search, encode corpus passages and user questions in the same vector space with the same compatible model. Compare the question vector with passage vectors, commonly using cosine similarity, then rank passages from highest similarity to lowest. Mixing incompatible embedding models makes their vectors unsuitable for direct comparison.

<!-- passage_id: dense-03 -->
Cosine similarity compares the direction of two vectors and is commonly used to rank semantic matches. A high score indicates geometric similarity under the chosen model and preprocessing. It is not a calibrated probability that a passage is relevant, and a fixed cutoff can behave differently across queries and corpora.

<!-- passage_id: dense-04 -->
Dense retrieval can recover paraphrases and related concepts that exact term matching misses. It can be weaker for rare names, identifiers, numbers, or exact phrases. Keeping the original text and inspecting retrieved passages helps explain what the embedding ranking actually returned.

<!-- passage_id: dense-05 -->
Embedding models have a maximum input length measured in tokens. If a word-based chunk is longer than that limit, the encoder may truncate its tail. Chunk-size experiments should therefore report token-limit violations; a word count is only an approximate proxy for model tokens.
