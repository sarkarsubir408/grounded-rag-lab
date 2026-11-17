# Human review workflow

## Current public-release questions

The active evaluation dataset is [`data/evaluation/questions.json`](../data/evaluation/questions.json). It contains 25 project-owner-approved annotations: C01–C08 are the validation split, and T01–T17 are held-out test questions. Each question's `review_decision.date` is 2026-11-16; this date was checked against the dataset metadata. The decision log is in [`human_review_decisions.md`](human_review_decisions.md).

When reviewing an annotation, check the question and reference answer against the current corpus, confirm that each evidence `doc_id` and `passage_id` resolves to the smallest sufficient passage, and verify that unanswerable questions have no evidence. Keep the validation and test splits separate. Do not use test questions or results to select or tune abstention thresholds. Owner approval records the question content and labels; it does not independently validate every derived chunk-level relevance judgment or generated answer.

If an annotation is changed, record the new reviewer decision, date, and rationale in the question's review metadata and update this decision log as appropriate. Do not infer or alter a label solely from a retrieval result.

## Current bundled calibration

The application loads the active [public-release calibration artifact](../reports/public_release/abstention_calibration.json) and its [metadata](../reports/public_release/calibration_metadata.json). It was derived from C01–C08 only, using 100/20 chunks, K=5, and RRF constant 60. The thresholds are provisional retrieval signals, not confidence probabilities; eight validation questions do not establish generalization reliability. The app checks the corpus fingerprint and configuration before using the artifact. If it is missing or mismatched, the app remains uncalibrated, permits retrieval, and blocks answer generation.

Do not treat top-level historical reports as the active public-release calibration. For a different uploaded corpus, select **Upload documents**, provide a validation-only JSON set containing both answerability classes with every item marked `review_status: human-reviewed`, and use the app's calibration control. Uploaded-corpus calibration is held in the current Streamlit session; it does not write or replace the bundled artifact. No test questions may be used for calibration.

## Reviewing generated answers

The optional [`scripts/run_one_answer_review.py`](../scripts/run_one_answer_review.py) command produces one answer for a held-out test question and appends a record to `reports/answer_review_records.jsonl`. It requires the explicit `--authorize-billable-call` flag, live API configuration, and a valid key; it can incur an API charge. The record includes empty human-rating fields for correctness, contextual relevance, and faithfulness to be completed after reading the answer and its evidence. The generated record is a local review artifact, not a benchmark result.

No live OpenAI calls or generated answers have been tested in this public-release workspace. Automated generation tests use mocked providers. A citation ID resolving to a retrieved chunk confirms reference existence only; a reviewer must inspect the passage to judge whether it supports each answer claim.

See the [public-release validation report](../reports/public_release/validation_results.md) for the six manual retrieval-gate smoke checks and the remaining unverified behavior.
