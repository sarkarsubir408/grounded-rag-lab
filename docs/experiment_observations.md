# Observed retrieval results

## Current public-release run (2026-11-16)

The current public-release benchmark uses six corpus documents and 25 owner-approved questions. The validation split contains 6 answerable and 2 unanswerable questions; the held-out test split contains 15 answerable and 2 unanswerable questions. The [per-query results](../reports/public_release/retrieval_per_query.csv), [macro summary](../reports/public_release/retrieval_macro_summary.csv), and [experiment metadata](../reports/public_release/experiment_metadata.json) are under `reports/public_release/`.

All 12 retrieval combinations completed for Dense, BM25, and Hybrid across 100/20, 150/20, 100/0, and 100/40 at K=3 and K=5. The run has 408 per-query rows and 24 macro rows. All 21 answerable questions have valid corpus passage evidence references. Unanswerable questions have no evidence and are excluded from answerable-query macro averages.

At K=5, 150/20 achieved the highest macro nDCG@5 for each method: Dense 0.9509378748381727, BM25 0.940233980057931, and Hybrid 0.9606722198167797. 100/40 achieved the highest Precision@5 for each: Dense 0.36000000000000004, BM25 0.3466666666666667, and Hybrid 0.38666666666666666. All 12 configurations had Hit Rate@5 1.0, so no answerable test question was missed at K=5. At K=3, the only answerable miss was T03 for Dense 100/20; the other combinations had no answerable misses. Exact metrics are in [retrieval_macro_summary.csv](../reports/public_release/retrieval_macro_summary.csv).

The public-release abstention thresholds were computed using C01–C08 only (6 answerable, 2 unanswerable), with 100/20 chunks, K=5, and RRF constant 60:

| Method | Threshold |
|---|---:|
| Dense | 0.5586917400360107 |
| BM25 | 9.107469618115733 |
| Hybrid | 0.5586917400360107 |

The corpus fingerprint is 355cca24544ce738ec98687bc5480c67827a51fbd04b07185a227a1d24569c8c. These eight-question thresholds are provisional retrieval signals, not confidence probabilities. The app's separate public-release calibration artifact is fingerprint-bound; missing or mismatched calibration leaves the app uncalibrated and blocks answer generation. No live OpenAI API calls were made.

## Historical pre-public-release observations (not current benchmark results)

The following table and notes are retained from an earlier project run dated 2026-11-13. They used a different corpus and question set, including 13 answerable and 4 unanswerable test questions. They are not results for the six-document public-release corpus or the current 15-answerable/2-unanswerable test split. The old root-level reports remain local historical artifacts only; they are not current workflows or part of the recommended public-release file list. The original stabilized Jupyter notebook is intentionally excluded because its saved outputs contain excerpts from the previous corpus and local runtime URLs. It remains in the original project; notebook-based historical comparisons cannot be rerun from this copy, and no replacement notebook is included.

### Historical expanded-corpus results, K=5

The earlier macro table used the 13 answerable questions from that run; four unanswerable questions were reported separately and excluded from ranking-metric averages.

| Chunk size / overlap | Method | Precision@5 | Recall@5 | Hit Rate@5 | MRR@5 | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| 100/20 | Dense | 0.292 | 0.782 | 0.923 | 0.865 | 0.769 |
| 100/20 | BM25 | 0.323 | 0.885 | 1.000 | 0.859 | 0.796 |
| 100/20 | Hybrid RRF | 0.338 | 0.923 | 1.000 | 0.859 | 0.832 |
| 150/20 | Dense | 0.292 | 0.923 | 0.923 | 0.862 | 0.825 |
| 150/20 | BM25 | 0.262 | 0.885 | 1.000 | 0.923 | 0.846 |
| 150/20 | Hybrid RRF | 0.292 | 0.962 | 1.000 | 0.891 | **0.851** |
| 100/0 | Dense | 0.231 | 0.769 | 0.923 | 0.865 | 0.748 |
| 100/0 | BM25 | 0.277 | 0.923 | 1.000 | **0.949** | **0.886** |
| 100/0 | Hybrid RRF | 0.277 | 0.923 | 1.000 | 0.910 | 0.852 |
| 100/40 | Dense | 0.338 | 0.718 | 0.846 | 0.769 | 0.696 |
| 100/40 | BM25 | **0.400** | 0.859 | 1.000 | **0.949** | 0.846 |
| 100/40 | Hybrid RRF | **0.400** | 0.872 | 1.000 | 0.891 | 0.816 |

In that earlier dataset, 150/20 Hybrid led in Recall@5, 100/0 BM25 led in MRR@5 and nDCG@5, and 100/40 BM25/Hybrid led in Precision@5. Those historical rankings should not be attributed to the current release.

### Historical single-document comparison

An earlier comparison preserved seven question labels and compared fresh dense retrieval with historical notebook rows using one document and chunk-index-based labels. Its per-query output and comparison summary remain local historical artifacts only; they are not part of the recommended public-release file list or a current runnable workflow. Because the original notebook and its original source document are not included in this release copy, that comparison is not reproducible here and is not part of the current public-release benchmark. Current retrieval results are in the [public-release macro summary](../reports/public_release/retrieval_macro_summary.csv).

### Historical calibration and abstention check

An earlier calibration used a different four-unanswerable-question test split. That historical held-out check recorded 10 of 12 unanswerable question/method cases abstaining: T06, T12, and T17 abstained for all methods, while Dense and Hybrid accepted T13 and BM25 abstained. Those old question IDs/results do not describe the current public-release evaluation and were not used for the new thresholds.

The current public-release thresholds use only C01–C08. A high validation score on eight items, or consistency when reproducing those thresholds, does not establish generalization reliability.
