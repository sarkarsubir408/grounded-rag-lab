# Architecture and runtime flow

## Components

`src/rag_portfolio/` keeps ingestion, chunking, embeddings, retrieval, evaluation, abstention, configuration, and generation in small modules. The application intentionally uses local in-memory indexes rather than a vector database or orchestration framework.

```text
TXT / Markdown / text PDF
        │
        ▼
parse + provenance metadata ──► word chunker (100 words / 20 overlap in app)
                                         │
                          ┌──────────────┴───────────────┐
                          ▼                              ▼
                   dense vectors                    BM25 index
                          └──────────────┬───────────────┘
                                         ▼
                              dense / BM25 / RRF
                                         ▼
                matching validation threshold? ── no ──► show retrieval; abstain
                                         │ yes
                                         ▼
              optional explicitly authorized OpenAI Responses call
                                         ▼
                 source-ID existence validation + source links
```

## Ingestion and provenance

TXT and Markdown are decoded as UTF-8; frontmatter may supply a stable `doc_id` and HTTPS `source_url`. Blank lines delimit evidence passages, project-authored `<!-- passage_id: ... -->` markers keep annotated passages stable, and headings are retained as context for the following passage. PDFs are parsed in memory with `pypdf`; page numbers are retained. Files are limited to 15 MiB each. Scanned PDFs, OCR, encrypted files, and non-text formats are out of scope. Upload bytes are not persisted by the app.

Chunk IDs are `<document_id>:chunk-<zero-based index>`. Chunk records retain source passage IDs and PDF page numbers. Embedding vectors are generated from one pinned Sentence Transformers model revision; query and corpus vectors use that same model. Dense retrieval ranks cosine similarity, BM25 uses `rank-bm25`, and hybrid retrieval combines full dense and BM25 rankings with one-based Reciprocal Rank Fusion (RRF), initially `k=60`.

## Calibration and abstention

Dense cosine, BM25, and RRF scores have different scales and are not probabilities. In particular, the hybrid gate uses the maximum dense cosine among the top five RRF results, not the RRF score. Thresholds maximize balanced accuracy on validation labels, with ties resolved toward fewer false alarms. Calibration requires every validation item to be marked human-reviewed. The active public-release artifact records `calibration_top_k: 5`; the app uses K=5 for answer generation. The UI's Top-K slider changes only the retrieved-passage preview, so it cannot silently change the K used by the abstention threshold. For compatibility, the loader also interprets an omitted K field in older artifacts as K=5.

The active [public-release calibration artifact](../reports/public_release/abstention_calibration.json) was calculated from C01–C08 only (six answerable and two unanswerable questions), with 100/20 chunking, K=5, and RRF constant 60. Its thresholds are provisional because the validation set is small. The artifact includes a corpus fingerprint, embedding name/revision, chunk configuration, K, and RRF constant; the app rejects a mismatch. See its [calibration metadata](../reports/public_release/calibration_metadata.json). Uploaded corpora require a separate reviewed validation set before answer generation.

Validation JSON schema for uploaded documents:

```json
{
  "dataset_version": "owner-reviewed-v1",
  "questions": [
    {"id": "V01", "split": "validation", "question": "...", "answerable": true, "review_status": "human-reviewed"},
    {"id": "V02", "split": "validation", "question": "...", "answerable": false, "review_status": "human-reviewed"}
  ]
}
```

Use at least two unique questions, include both answerability classes, and use validation questions only. The app requires `review_status: human-reviewed` for every uploaded calibration item. All 25 annotations in the current bundled `questions.json` record project-owner approval dated 2026-11-16. The eight-item validation set remains small, so thresholds are still provisional. Uploaded-corpus calibration is held in the current Streamlit session; it does not write or replace the bundled public-release artifact.

## Generation boundary and citations

`OpenAIResponsesProvider` is behind a provider protocol. It reads `OPENAI_API_KEY`, `OPENAI_MODEL`, and timeout from environment, calls the Responses API with `store=False`, and disables automatic retries. `RAG_ENABLE_LIVE_API` is false by default. When enabled, the query form submits authorization with its question; the callback binds consent to that question and clears the checkbox, and the app consumes the token before constructing a provider. Tests use fake providers. Retrieval and experiment commands make no LLM request.

The prompt marks document contents as untrusted data and asks for abstention when context is insufficient. A returned `[S#]` is checked against the IDs actually supplied in that request; the UI resolves it to a document, chunk ID, and PDF page where available. This verifies source-reference existence only. It does not prove that a claim follows from the cited text. Semantic support, correctness, relevance, and faithfulness need human review or a separately validated evaluator.

## Security and operational limits

- Keep `.env` local; `.gitignore` excludes it.
- Keys are never embedded in prompts, logs, source files, or review records.
- API calls require configuration plus a visible one-call authorization; no hidden retry is configured.
- The index is in memory and must be rebuilt after uploads or process restart.
- Uploaded content is sent to the configured embedding model locally through Sentence Transformers; if generation is authorized, retrieved context is sent to OpenAI. Review your data handling obligations before using confidential documents.
- This project has no authentication, multi-user isolation, persistent index, observability stack, rate limiting, OCR, or production deployment hardening.
