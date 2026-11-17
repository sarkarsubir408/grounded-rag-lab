# Document Chunking for Retrieval-Augmented Generation

Document status: Approved synthetic educational content for indexing in the public-release project. Public publication remains subject to final project review.

## What document chunking means

A retrieval-augmented generation (RAG) system searches documents for evidence before generating an answer. Chunking divides each document into smaller passages that can be indexed and retrieved separately. A useful chunk is focused enough to match a question, yet complete enough to make its evidence clear.

Imagine a guide with one section on resetting an account and another on refunds. Putting both in one large chunk may return unrelated reset instructions for a refund question. Storing every sentence alone could separate a step from the condition that explains it.

## Fixed-size chunking

Fixed-size chunking splits text after a chosen amount, measured in words, characters, or tokens. It can repeat some text from one chunk in the next. Token limits align more closely with model input constraints; word or character limits can be easier to inspect. Record the unit and splitting rule so the process can be repeated.

The method is consistent, but a boundary may cut through a paragraph or example. It does not know which heading or qualification belongs with the passage.

## Choosing chunk size

Chunk size affects search specificity and the context available to a model. Smaller chunks can make a detail easier to find, but may omit a definition or condition needed to understand it. Larger chunks retain more explanation, but can combine topics and use more context.

There is no universally best size. A brief FAQ may suit short chunks, while a policy with exceptions may need more context. The choice also depends on the embedding model, questions, context limit, and number of retrieved results. Treat these as trade-offs to measure.

## Chunk overlap

Overlap repeats some text at the edge of neighboring chunks. It can help when an explanation crosses a boundary. For example, carrying a heading into the next passage may make a retrieved instruction clearer.

Overlap also repeats content in the index, increasing storage and sometimes producing near-duplicate results. Several overlapping passages may fill context without adding evidence. Choose a documented amount and check its effect rather than assuming more is better.

## Semantic chunking

Semantic chunking aims to split where a subject or document structure changes. Boundaries may follow headings, paragraphs, sentence meaning, or several signals. In an illustrative procedure, prerequisites could stay with dependent steps, while a separate troubleshooting topic starts a new passage.

Passages may read more naturally than fixed windows, but can vary in size and need more rules. Document those rules and keep them consistent in comparisons. Semantic splitting can still fail when headings are missing or topics shift abruptly.

## Document boundaries and context

Keep metadata such as source name, section, and position with each chunk. It helps readers trace evidence and prevents different files from being treated as one continuous text. Never let a chunk at one document's end overlap with another document's beginning.

Context also comes from headings, lists, tables, and captions. “Apply this limit after approval” may be unclear without the text defining “this limit.” Preserve structure when practical, and inspect retrieved passages with source information and surrounding text.

## Evaluating chunking strategies

Compare strategies on the same documents, questions, relevance labels, retrieval method, and cutoff K.

- **Precision@K:** Relevant results among the first K, divided by K.
- **Recall@K:** Relevant results found in the first K, divided by all relevant results for that question.
- **Hit Rate@K:** Whether at least one relevant result appears in the first K.
- **Mean reciprocal rank (MRR):** Average the reciprocal rank of the first relevant result across questions. Rank r contributes 1/r; no relevant result contributes 0.
- **Normalized discounted cumulative gain (nDCG):** Rewards relevant results near the top and can give more credit to higher relevance levels.
Review per-question results as well as averages. Inspect failures for split evidence, missing context, or irrelevant high-ranking passages. Keep calibration questions separate from held-out test questions, and do not tune on the test set. Record the corpus version, chunk settings, retrieval configuration, and metric definitions. Results apply to the tested setup; they do not prove one strategy is best for every collection.