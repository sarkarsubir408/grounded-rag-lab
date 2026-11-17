# Public-release validation results

## Scope and evidence

This report records the current six-document public-release corpus, reviewed question set, fresh retrieval reports, separate abstention calibration, and automated test coverage. It distinguishes saved experiment outputs and automated checks from manual application testing.

## Retrieval experiment run

The fresh retrieval-only run is recorded in experiment_metadata.json (generated 2026-11-16). It reports:

- Six corpus documents and 25 approved questions.
- Validation: C01–C08, 6 answerable and 2 unanswerable.
- Held-out test: T01–T17, 15 answerable and 2 unanswerable.
- Dense, BM25, and Hybrid RRF across 100/20, 150/20, 100/0, and 100/40.
- K=3 and K=5; 12 completed method/configuration combinations.
- 408 per-query rows and 24 macro-summary rows.
- No failed question IDs or missing evidence mappings in the run metadata.
- No calibration, answer generation, or OpenAI calls during the retrieval run.

All 21 answerable questions have evidence references to existing document passages. Unanswerable questions have empty evidence; their retrieval metrics are undefined and excluded from answerable-query macro averages.

At K=5, all 12 combinations had no answerable misses. At K=3, Dense 100/20 missed T03; the other 11 combinations had no answerable misses. 150/20 produced the highest macro nDCG@5 for Dense, BM25, and Hybrid; 100/40 produced the highest Precision@5 for all three. Exact macro values are in retrieval_macro_summary.csv and are detailed in [docs/retrieval_experiments.md](../../docs/retrieval_experiments.md).

## Abstention calibration

The separate abstention_calibration.json and calibration_metadata.json were generated from C01–C08 only. Metadata records 6 answerable, 2 unanswerable, no test questions used, and no live OpenAI calls. The calibration is bound to corpus fingerprint:

    355cca24544ce738ec98687bc5480c67827a51fbd04b07185a227a1d24569c8c

Configuration: 100/20, K=5, RRF constant 60, with the pinned embedding model and revision. Thresholds:

| Method | Provisional threshold |
|---|---:|
| Dense | 0.5586917400360107 |
| BM25 | 9.107469618115733 |
| Hybrid | 0.5586917400360107 |

These thresholds are based on only eight validation questions and are not confidence probabilities. The app loads the public-release artifact, verifies the corpus and configuration, and does not silently use the historical artifact if calibration is missing or mismatched. In that case the app reports an uncalibrated state, leaves retrieval available, and blocks answer generation.

## Manual Streamlit UI smoke check (2026-11-16)

These are two-question manual smoke checks per retrieval method, not a comprehensive evaluation. The observations describe only the retrieval gate's accept or abstain decision and the displayed retrieval signal, threshold, and evidence rank. They do not establish generated-answer correctness.

| Method | Question | Observed result |
|---|---|---|
| Dense | C01 | Accepted; retrieval signal 0.6188, threshold 0.5587. Supporting evidence appeared at Rank 2. |
| Dense | C06 | Abstained; signal 0.4742, threshold 0.5587. |
| BM25 | C01 | False abstention; signal 7.1799, threshold 9.1075. Supporting evidence appeared at Rank 2. |
| BM25 | C06 | False accept; signal 11.5018, threshold 9.1075. Live generation was disabled. |
| Hybrid | C01 | Accepted; signal 0.6188, threshold 0.5587. Supporting evidence appeared at Rank 1. |
| Hybrid | C06 | Abstained; signal 0.4742, threshold 0.5587. |

No live OpenAI calls or generated answers were tested. In particular, the BM25 false accept records that the retrieval gate passed C06; it does not mean an answer was generated or judged correct. The BM25 false abstention records that the gate blocked C01 despite supporting evidence appearing in the retrieval preview.

## Automated tests

Command: `python -m pytest -q` using Python 3.11.4 from the public-release workspace.

**Result after the documentation update: 41 passed in 9.66s.**

The four focused calibration-loading regression tests are in tests/test_calibration_loading.py. They check the public-release artifact and all three method thresholds, fingerprint mismatch behavior, safe uncalibrated pipeline behavior, and absence of historical-file fallback; all four passed as part of the 41-test suite. The full suite also covers retrieval, abstention, ingestion, citation validation, and mocked generation. Tests use mocked providers; they do not call OpenAI.

## Manual and unverified behavior

- Comprehensive manual Streamlit UI validation has **not** been completed in this public-release workspace; only the smoke checks above were observed.
- Live OpenAI answer generation and real provider failure behavior have not been tested; no live API calls were made.
- The automated suite does not establish semantic citation support, answer correctness, contextual relevance, or faithfulness.
- PDF validation is limited to text-based PDFs; OCR and scanned documents are unsupported.
- The in-memory index does not persist across restarts.

## Remaining limitations

The corpus and evaluation set are small. Project-owner approval establishes the reviewed question content and labels, while derived chunk-level relevance judgments have not been independently audited. Eight calibration examples are insufficient to establish threshold generalization. Retrieval metrics describe this dataset only and do not establish production readiness.
