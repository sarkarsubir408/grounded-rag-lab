---
doc_id: rag_workflow
title: RAG workflow and grounded answers
source_url: https://github.com/openai/openai-cookbook/blob/main/examples/data/oai_docs/optimizing-llm-accuracy.txt
source_license: MIT (OpenAI Cookbook repository)
retrieved_date: 2026-11-13
provenance: Project-authored summary; no third-party prose copied.
---

# RAG workflow and grounded answers

<!-- passage_id: rag-01 -->
Retrieval-augmented generation (RAG) supplies a language model with relevant information retrieved from an external corpus at question time. The corpus can hold domain material that is too private, recent, or specific to rely on the model's training data alone. Retrieval provides evidence in the request; it does not guarantee that the model will use the evidence correctly.

<!-- passage_id: rag-02 -->
A simple RAG workflow has distinct steps: load and normalize documents, split them into retrievable passages, index the passages, retrieve candidates for a question, assemble selected context, and generate an answer. Keeping retrieval and generation as separate stages makes failures easier to diagnose: the retriever may return poor evidence, or the model may use good evidence incorrectly.

<!-- passage_id: rag-03 -->
Too little retrieved context can omit a fact needed to answer. Too much irrelevant context can bury useful evidence and make an answer less reliable. Retrieval quality therefore involves both finding supporting passages and limiting noise. Inspecting results for representative questions is useful alongside aggregate retrieval metrics.

<!-- passage_id: rag-04 -->
An answer should be restricted to the supplied evidence. When retrieved passages do not support a response, the application should abstain and explain that the available documents are insufficient. A prompt instruction alone cannot prove that an answer is faithful; source references and answer claims need separate checks.

<!-- passage_id: rag-05 -->
Evaluate retrieval and generation separately. Retrieval evaluation compares ranked passages with relevance judgments. Answer evaluation compares generated claims with reference facts and evidence. A system can retrieve the correct passage and still produce an incorrect answer, so a single end-to-end score hides which stage needs improvement.

<!-- passage_id: rag-06 -->
When source documents change, the index must be refreshed so that retrieval reflects the current corpus. The application should retain document identity and passage metadata during ingestion. This allows a returned passage to be traced to its original file and, for paginated documents, its page number.
