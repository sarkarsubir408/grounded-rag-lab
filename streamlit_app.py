from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_portfolio.chunking import create_chunk_records
from rag_portfolio.config import Settings
from rag_portfolio.embeddings import load_embedding_model
from rag_portfolio.calibration import load_calibration_thresholds
from rag_portfolio.evaluation import CALIBRATION_TOP_K, calibrate_abstention_thresholds
from rag_portfolio.generation import OpenAIResponsesProvider
from rag_portfolio.authorization import (
    AUTHORIZATION_CHECKBOX_KEY,
    capture_question_authorization,
    create_authorized_provider,
)
from rag_portfolio.ingestion import (
    DocumentIngestionError,
    documents_fingerprint,
    load_documents_from_directory,
    load_uploaded_documents,
)
from rag_portfolio.pipeline import answer_question
from rag_portfolio.retrieval import build_index, search

st.set_page_config(page_title="Grounded RAG Lab", page_icon="🔎", layout="wide")
st.title("Grounded RAG Lab")
st.caption("Compare dense, BM25, and hybrid retrieval over your documents, then generate a cited answer when explicitly enabled.")

settings = Settings.from_environment()


@st.cache_resource(show_spinner="Loading the embedding model (first use may download weights)...")
def get_model(model_name: str, revision: str):
    return load_embedding_model(model_name, revision)


def _capture_live_authorization(generation_enabled: bool) -> None:
    capture_question_authorization(st.session_state, generation_enabled)


def load_thresholds(corpus_signature: str) -> tuple[dict[str, float], str]:
    path = PROJECT_ROOT / settings.calibration_file
    return load_calibration_thresholds(
        path,
        corpus_fingerprint=corpus_signature,
        embedding_model=settings.embedding_model_name,
        embedding_revision=settings.embedding_model_revision,
        rrf_constant=settings.rrf_constant,
    )


st.subheader("1. Choose documents")
corpus_mode = st.radio(
    "Index source",
    ("Bundled project corpus", "Upload documents"),
    horizontal=True,
)
uploaded_files = st.file_uploader(
    "Upload multiple TXT, Markdown, or text-based PDF documents",
    type=["txt", "md", "markdown", "pdf"],
    accept_multiple_files=True,
    max_upload_size=15,
    disabled=corpus_mode != "Upload documents",
    help="Each file can be up to 15 MiB. Larger files are rejected and cannot be indexed.",
)
uploaded_validation = st.file_uploader(
    "Optional validation JSON for this uploaded corpus",
    type=["json"],
    disabled=corpus_mode != "Upload documents",
    help="Use a human-reviewed validation-only set with both answerable and unanswerable questions. No test questions are accepted for calibration.",
)
st.caption("Uploads stay in this Streamlit session. Maximum document size is 15 MiB per file. Text extraction from scanned PDFs and OCR are not supported.")

try:
    if corpus_mode == "Bundled project corpus":
        active_documents = load_documents_from_directory(PROJECT_ROOT / "data" / "corpus")
    else:
        active_documents = load_uploaded_documents(
            [(uploaded.name, uploaded.getvalue()) for uploaded in uploaded_files or []]
        ) if uploaded_files else []
except (DocumentIngestionError, OSError) as error:
    active_documents = []
    st.error(str(error))

active_signature = documents_fingerprint(active_documents) if active_documents else ""
if active_documents:
    st.info(f"Ready to index {len(active_documents)} document(s). The index uses the baseline 100-word / 20-word-overlap chunker.")

if st.button("Build or refresh index", type="primary", disabled=not active_documents):
    try:
        encoder = get_model(settings.embedding_model_name, settings.embedding_model_revision)
        chunks = create_chunk_records(active_documents, chunk_size=100, overlap=20)
        index = build_index(chunks, encoder)
        st.session_state["rag_index"] = index
        st.session_state["rag_index_signature"] = active_signature
        st.session_state["rag_document_count"] = len(active_documents)
        st.success(f"Indexed {len(active_documents)} document(s) into {len(chunks)} chunks.")
    except Exception as error:
        st.error(f"Indexing failed: {error}")

index = st.session_state.get("rag_index")
index_is_current = index is not None and st.session_state.get("rag_index_signature") == active_signature
if index is not None and not index_is_current:
    st.warning("The selected documents differ from the current index. Rebuild the index before querying.")

st.subheader("2. Retrieve and answer")
method = st.selectbox("Retrieval method", ("dense", "bm25", "hybrid"), index=2)
preview_top_k = st.slider("Top-K passages in retrieval preview", min_value=1, max_value=10, value=5)
st.caption(
    f"Answer generation uses calibrated K={CALIBRATION_TOP_K}; "
    "the slider changes only the separate retrieval preview."
)

if uploaded_validation is not None and active_documents:
    try:
        validation_payload = json.loads(uploaded_validation.getvalue().decode("utf-8"))
        validation_questions = validation_payload["questions"]
        if not validation_questions or any(
            item.get("review_status") != "human-reviewed" for item in validation_questions
        ):
            raise ValueError("Every calibration item must have review_status set to 'human-reviewed'.")
        if st.button("Calibrate thresholds for these documents"):
            encoder = get_model(settings.embedding_model_name, settings.embedding_model_revision)
            calibration = calibrate_abstention_thresholds(
                active_documents, validation_questions, encoder, settings.rrf_constant
            )
            calibration["embedding_model"] = settings.embedding_model_name
            calibration["embedding_revision"] = settings.embedding_model_revision
            calibration["dataset_version"] = validation_payload.get("dataset_version", "user-provided")
            st.session_state["upload_calibration"] = {
                "corpus_fingerprint": active_signature,
                "payload": calibration,
            }
            st.success("Thresholds calibrated from this corpus's human-reviewed validation set.")
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        st.error(f"Validation file is not usable: {error}")

thresholds, threshold_status = load_thresholds(active_signature) if active_signature else ({}, "")
custom_calibration = st.session_state.get("upload_calibration", {})
if custom_calibration.get("corpus_fingerprint") == active_signature:
    custom_payload = custom_calibration["payload"]
    thresholds = {
        method_name: float(result["threshold"])
        for method_name, result in custom_payload.get("retrievers", {}).items()
    }
    threshold_status = "human-reviewed validation set for the current corpus"
if thresholds:
    st.caption(f"Abstention uses method-specific thresholds from the separate validation split ({threshold_status}).")
else:
    reason = threshold_status or "no corpus selected"
    st.warning(
        f"Application is uncalibrated for the selected corpus ({reason}). "
        "Retrieval remains available, but answer generation is blocked until a "
        "matching validation calibration is available."
    )

api_enabled = settings.live_api_enabled
generation_available = api_enabled and bool(thresholds)
if api_enabled and not thresholds:
    st.info("Live generation is enabled in configuration but unavailable without calibration matching this corpus.")
elif not api_enabled:
    st.info("Live generation is disabled. Set RAG_ENABLE_LIVE_API=true, configure OPENAI_API_KEY and OPENAI_MODEL, then authorize an individual call here.")

with st.form("query_form", clear_on_submit=False):
    question = st.text_input(
        "Question",
        placeholder="Ask something supported by the indexed documents",
        key="rag_question",
    )
    if generation_available:
        st.checkbox(
            "I authorize one OpenAI API call for this submitted question. Provider usage may be billed.",
            value=False,
            key=AUTHORIZATION_CHECKBOX_KEY,
        )
    run_query = st.form_submit_button(
        "Retrieve and answer",
        disabled=not index_is_current,
        on_click=_capture_live_authorization,
        args=(generation_available,),
    )

if run_query:
    provider = create_authorized_provider(
        st.session_state,
        question,
        lambda: OpenAIResponsesProvider.from_settings(
            settings, user_authorized_this_call=True
        ),
    )
    if not question.strip():
        st.warning("Enter a question before retrieving documents.")
    else:
        try:
            encoder = get_model(settings.embedding_model_name, settings.embedding_model_revision)
            result = answer_question(
                question=question,
                index=index,
                model=encoder,
                method=method,
                thresholds=thresholds,
                provider=provider,
                top_k=CALIBRATION_TOP_K,
                rrf_constant=settings.rrf_constant,
            )
            st.markdown("### Answer")
            st.markdown(result.answer)
            signal_display = f"{result.retrieval_signal:.4f}" if result.retrieval_signal is not None else "n/a"
            threshold_display = f"{result.threshold:.4f}" if result.threshold is not None else "n/a"
            st.caption(f"Status: {result.status} · Retrieval signal: {signal_display} · Threshold: {threshold_display}")

            if result.cited_sources:
                st.markdown("### Source references (IDs checked; semantic support unverified)")
                for source in result.cited_sources:
                    page_label = f", page(s) {', '.join(map(str, source.page_numbers))}" if source.page_numbers else ""
                    st.markdown(f"- `[{source.source_id}]` **{source.document_name}** · `{source.chunk_id}`{page_label}")
                    if source.source_url and source.source_url.startswith("https://"):
                        st.link_button(f"Open source for {source.source_id}", source.source_url)

            with st.expander("Retrieved passages"):
                hits = search(
                    question, index, encoder, method,
                    top_k=preview_top_k, rrf_constant=settings.rrf_constant,
                )
                for hit in hits:
                    page_label = f" · pages {', '.join(map(str, hit.chunk.page_numbers))}" if hit.chunk.page_numbers else ""
                    st.markdown(f"**Rank {hit.rank} · {hit.chunk.document_name} · `{hit.chunk.chunk_id}`{page_label} · score {hit.score:.4f}**")
                    st.write(hit.chunk.text)
        except Exception as error:
            st.error(f"The request could not be completed: {error}")

st.divider()
st.caption("This demo uses an in-memory index and provisional calibration labels. A valid chunk citation proves that the chunk was retrieved; human review is still needed to verify semantic support.")
