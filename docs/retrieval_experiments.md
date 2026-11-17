# Public-release retrieval experiments

## Design and data

The current benchmark uses six documents from data/corpus/ and the approved data/evaluation/questions.json dataset. It compares a controlled 4 × 3 matrix using the same held-out test questions:

| Chunk size / overlap (words) | Dense | BM25 | Hybrid RRF |
|---|---:|---:|---:|
| 100 / 20 | ✓ | ✓ | ✓ |
| 150 / 20 | ✓ | ✓ | ✓ |
| 100 / 0 | ✓ | ✓ | ✓ |
| 100 / 40 | ✓ | ✓ | ✓ |

The dataset contains 25 questions: validation C01–C08 has 6 answerable and 2 unanswerable questions; held-out test T01–T17 has 15 answerable and 2 unanswerable questions. All 25 annotations were approved by the project owner on 2026-11-16. Every answerable question has a valid evidence reference to an existing corpus document and passage. The unanswerable questions intentionally have empty evidence lists. Annotation approval and valid passage IDs do not constitute an independent audit of every derived chunk-level relevance judgment or generated answer.

The public-release run completed all 12 method/configuration combinations at K=3 and K=5. The [per-query report](../reports/public_release/retrieval_per_query.csv) contains 408 rows (17 questions × 4 chunk configurations × 3 retrieval methods × 2 K values); the [macro summary](../reports/public_release/retrieval_macro_summary.csv) contains 24 rows (4 × 3 × 2). The [experiment metadata](../reports/public_release/experiment_metadata.json) records no failed questions and no missing evidence mappings.

Dense retrieval uses the pinned sentence-transformers/all-MiniLM-L6-v2 revision and cosine similarity. BM25 uses the same chunks and a fixed lowercase word tokenizer. Hybrid retrieval uses Reciprocal Rank Fusion with one-based ranks and both complete rankings; the configured RRF constant is 60. There is no reranker or vector database.

## Relevance and metrics

The retrieval metric unit is a chunk. For a configuration, a chunk is relevant if it contains at least one evidence passage ID listed for that question. The runner verifies evidence references against generated chunks. Overlap can place a passage into multiple chunks, so the relevant-chunk set and Recall@K denominator can vary by configuration.

For answerable test questions, the runner reports per-query values and macro means for:

- **Precision@K:** relevant returned chunks divided by the number returned (up to K).
- **Recall@K:** unique relevant chunk IDs returned divided by all relevant chunk IDs for that configuration.
- **Hit Rate@K:** whether at least one relevant chunk appears in the first K.
- **MRR@K:** reciprocal rank of the first relevant result, or zero.
- **nDCG@K:** binary-gain discounted ranking score normalized against the ideal relevant ordering.

Unanswerable test questions have empty gold evidence. Their ranking metrics are undefined (blank in the CSV), remain in the per-query report, and are excluded from answerable-query macro averages. The per-query and summary files are the source of the values below.

## Current results at K=5

Values are reproduced from the [public-release macro summary](../reports/public_release/retrieval_macro_summary.csv) without rounding. Macro averages use the 15 answerable test questions; two unanswerable questions are excluded.

| Configuration | Method | Precision@5 | Recall@5 | Hit Rate@5 | MRR@5 | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| 100/20 | Dense | 0.28 | 0.9 | 1.0 | 0.8722222222222221 | 0.830256973760248 |
| 100/20 | BM25 | 0.29333333333333333 | 0.9333333333333333 | 1.0 | 0.9222222222222223 | 0.8811948485984207 |
| 100/20 | Hybrid | 0.29333333333333333 | 0.9333333333333333 | 1.0 | 1.0 | 0.9223867772340155 |
| 150/20 | Dense | 0.26666666666666666 | 0.9666666666666667 | 1.0 | 0.9666666666666667 | 0.9509378748381727 |
| 150/20 | BM25 | 0.2533333333333333 | 0.9333333333333333 | 1.0 | 1.0 | 0.940233980057931 |
| 150/20 | Hybrid | 0.26666666666666666 | 0.9666666666666667 | 1.0 | 1.0 | 0.9606722198167797 |
| 100/0 | Dense | 0.25333333333333335 | 0.9333333333333333 | 1.0 | 0.8555555555555556 | 0.840086739029286 |
| 100/0 | BM25 | 0.24000000000000007 | 0.9 | 1.0 | 0.9222222222222223 | 0.8660241672069006 |
| 100/0 | Hybrid | 0.25333333333333335 | 0.9333333333333333 | 1.0 | 0.9555555555555556 | 0.896196751944356 |
| 100/40 | Dense | 0.36000000000000004 | 0.8777777777777778 | 1.0 | 0.8222222222222223 | 0.7688763137483018 |
| 100/40 | BM25 | 0.3466666666666667 | 0.8444444444444444 | 1.0 | 0.9222222222222223 | 0.8335832255747176 |
| 100/40 | Hybrid | 0.38666666666666666 | 0.9333333333333332 | 1.0 | 0.9666666666666667 | 0.8956970715089654 |

150/20 has the highest macro nDCG@5 for Dense (0.9509378748381727), BM25 (0.940233980057931), and Hybrid (0.9606722198167797). 100/40 has the highest Precision@5 for Dense (0.36000000000000004), BM25 (0.3466666666666667), and Hybrid (0.38666666666666666).

At K=5, every one of the 12 combinations has Hit Rate@5 of 1.0; there were no answerable retrieval misses. At K=3, Dense with 100/20 missed T03. The other 11 combinations had no answerable misses. These are results for this small corpus and question set, not evidence that one chunk size is generally best.

## Abstention calibration

The app's current calibration is stored in the [public-release calibration artifact](../reports/public_release/abstention_calibration.json), with run details in [calibration metadata](../reports/public_release/calibration_metadata.json). Calibration used only validation questions C01–C08 (6 answerable, 2 unanswerable), 100/20 chunking, K=5, and RRF constant 60. The corpus fingerprint is 355cca24544ce738ec98687bc5480c67827a51fbd04b07185a227a1d24569c8c.

| Method | Threshold |
|---|---:|
| Dense | 0.5586917400360107 |
| BM25 | 9.107469618115733 |
| Hybrid | 0.5586917400360107 |

These thresholds are provisional because they use only eight validation questions. They are method-specific retrieval signals, not confidence probabilities. Hybrid abstention uses the maximum dense cosine among the top-five RRF results; it does not treat the RRF score as a confidence probability. T01–T17 were not used for calibration.

The app loads the public-release artifact, verifying corpus fingerprint, embedding model and revision, chunking configuration, K, and RRF constant. A missing or mismatched artifact leaves the app uncalibrated: retrieval remains available, answer generation is blocked, and there is no silent fallback to the historical calibration artifact. The full manual and automated validation record is in the [public-release validation report](../reports/public_release/validation_results.md).

## Historical records and limitations

Files under the top-level `reports/` directory outside `reports/public_release/` are local historical records from an earlier project dataset and corpus. They are not inputs to, or results of, the current public-release benchmark and must not be numerically combined with the current CSVs. Those legacy reports and the single-document comparison runner are excluded from the recommended public-release file list and are not active workflows. The original stabilized Jupyter notebook is intentionally excluded because its saved outputs contain excerpts from the previous corpus and local runtime URLs. It remains in the original project; no replacement notebook is included here.

This evaluation uses only six documents and 15 answerable test questions. Owner-approved passage references are valid, but chunk-level qrels are derived from those references and have not been independently audited. The embedding model has a token limit; word count is only a proxy for tokens. No significance test, live answer-quality measurement, or universal chunking recommendation is claimed.
