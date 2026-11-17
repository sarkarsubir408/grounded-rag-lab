# Grounded RAG Lab

Grounded RAG Lab is a small, inspectable RAG retrieval study and Streamlit prototype. This public-release version uses a six-document corpus, a reviewed 25-question evaluation set, and a separate retrieval-only benchmark. It includes an OpenAI Responses API adapter, but live generation has not been validated and remains disabled unless configured and authorized for an individual request.

## Project status

- **Corpus:** six documents: one synthetic educational chunking guide and five technical notes. Their provenance is recorded in data/source_manifest.json.
- **Questions:** 25 project-owner-approved annotations. Validation contains 8 questions (6 answerable, 2 unanswerable); the held-out test split contains 17 (15 answerable, 2 unanswerable). Each of the 21 answerable questions has a valid corpus passage reference.
- **Retrieval:** Dense, BM25, and Hybrid using Reciprocal Rank Fusion (RRF), compared across 100/20, 150/20, 100/0, and 100/40 word chunk-size/overlap settings at K=3 and K=5.
- **Fresh experiment outputs:** 12 combinations, 408 per-query rows, and 24 macro-summary rows are stored under reports/public_release/. Unanswerable questions have empty evidence and are excluded from answerable-query macro averages.
- **Abstention:** the Streamlit app loads reports/public_release/abstention_calibration.json. Its thresholds were selected from C01–C08 only and are provisional; they are retrieval signals, not confidence probabilities.
- **Application:** TXT, Markdown, and text-based PDF ingestion, in-memory indexing, and dense/BM25/hybrid retrieval are implemented. The original stabilized Jupyter notebook is intentionally excluded: its saved outputs contain excerpts from the previous corpus and local runtime URLs. It remains in the original project; this public-release workspace contains no notebook or replacement notebook.

The results are specific to this small corpus and reviewed dataset. They do not establish general retrieval performance or production readiness. Six manual Streamlit smoke checks were recorded on 2026-11-16: C01 and C06 for each of Dense, BM25, and Hybrid. They checked retrieval-gate outcomes and displayed signals and thresholds; supporting evidence ranks were noted where evidence appeared. This was not comprehensive UI validation; no generated answers or live API calls were tested. See the [validation report](reports/public_release/validation_results.md).

## Setup (Python 3.11)

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    Copy-Item .env.example .env

The embedding model name and revision are pinned in .env.example. First use may download model weights. For offline execution after packages and model weights are cached, set HF_HUB_OFFLINE=1.

## Run the retrieval experiments

    python scripts/run_retrieval_only_experiments.py

This runs the 4 × 3 matrix using held-out questions T01–T17 and writes fresh CSVs and metadata under reports/public_release/. It does not calibrate thresholds, generate answers, or call an LLM. See the [retrieval experiments](docs/retrieval_experiments.md) for metric definitions and exact recorded results.

In the current saved run, 150/20 has the highest macro nDCG@5 for Dense (0.9509378748381727), BM25 (0.940233980057931), and Hybrid (0.9606722198167797). 100/40 has the highest Precision@5 for Dense (0.36000000000000004), BM25 (0.3466666666666667), and Hybrid (0.38666666666666666). These are descriptive results for this corpus and question set, not universal chunking recommendations. At K=5 there were no answerable retrieval misses; at K=3, Dense 100/20 missed T03 and the other 11 combinations had no answerable misses.

## Abstention calibration

The bundled calibration was freshly computed from validation questions C01–C08 only (6 answerable, 2 unanswerable), using the app's 100/20 chunking, K=5, and RRF constant 60. Its corpus fingerprint is:

    355cca24544ce738ec98687bc5480c67827a51fbd04b07185a227a1d24569c8c

| Method | Provisional threshold |
|---|---:|
| Dense | 0.5586917400360107 |
| BM25 | 9.107469618115733 |
| Hybrid | 0.5586917400360107 |

These values are method-specific retrieval signals, not probabilities. Eight validation questions are too few to establish reliable generalization. The app checks the artifact's corpus fingerprint, embedding configuration, chunking configuration, K, and RRF constant. If the configured artifact is missing or mismatched, the app reports an uncalibrated state, keeps retrieval available, and blocks answer generation; it does not silently fall back to the historical calibration file. See [calibration metadata](reports/public_release/calibration_metadata.json) and [validation results](reports/public_release/validation_results.md).

## Run the application

    streamlit run streamlit_app.py

Choose the bundled corpus or upload TXT, Markdown, and text-based PDF documents, then build the in-memory index. Retrieval works without OpenAI credentials. For uploaded documents, answer generation remains blocked until a human-reviewed validation set for that corpus is calibrated.

Live generation is disabled by default. To enable a billable request, configure an available model and OPENAI_API_KEY in your local .env, set RAG_ENABLE_LIVE_API=true, and authorize that individual submitted question in the UI. Authorization is tied to one question and consumed for one request. Tests use mocked providers only; no live OpenAI API call has been made. Do not commit .env or other environment files.

## Tests and validation

    python -m pytest -q

The public-release validation record is in [reports/public_release/validation_results.md](reports/public_release/validation_results.md). Root-level reports outside `reports/public_release/` are local historical records, not active public-release workflows or current benchmark results, and are excluded from the recommended first-public-commit file list. `scripts/run_legacy_comparison.py`, `data/evaluation/legacy_ground_truth.json`, and `data/evaluation/legacy_historical_metrics.json` remain local archival material only; they depend on the removed baseline corpus or notebook and are excluded from the public-release file list.

## Project layout

    data/corpus/                    Six current corpus documents
    data/evaluation/questions.json  25 approved questions (8 validation, 17 test)
    notebooks/                       No notebook included; see exclusion note above
    src/rag_portfolio/              Ingestion, chunking, embeddings, retrieval, evaluation, and app logic
    scripts/run_retrieval_only_experiments.py  Current retrieval benchmark runner
    scripts/run_one_answer_review.py           Optional one-call answer review tool
    streamlit_app.py                 Interactive UI
    tests/                           Offline automated tests
    reports/public_release/          Current retrieval and calibration artifacts
    reports/                         Local historical records; exclude root-level files from public release
    docs/                            Architecture, provenance, experiments, and portfolio notes

## Licensing and provenance

The MIT license applies only to the software files listed in [LICENSE](LICENSE). It does not license the corpus, documentation, reports, or other learning materials; see [CONTENT_NOTICE.md](CONTENT_NOTICE.md).

data/corpus/chunking_guide.md is synthetic educational content created with ChatGPT assistance. The exact generation prompt and an external source URL were not recorded, and its exact external provenance has not been independently established. No exclusive-ownership or copyright-uniqueness claim is made. The five other technical corpus documents are project-authored summaries with source information recorded in data/source_manifest.json; third-party source materials retain their own rights and licenses.
