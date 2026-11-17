from __future__ import annotations

import numpy as np
import pytest

from rag_portfolio.config import (
    ExplicitAuthorizationRequired,
    LiveCallsDisabledError,
    Settings,
)
from rag_portfolio.generation import GenerationError, OpenAIResponsesProvider
from rag_portfolio.models import ChunkRecord
from rag_portfolio.pipeline import answer_question, validate_source_ids
from rag_portfolio.retrieval import build_index


class TokenVectorEncoder:
    def encode(self, texts, **kwargs):
        return np.asarray(
            [[1.0, 0.0] if "evidence" in text.lower() else [0.0, 0.0] for text in texts],
            dtype=np.float32,
        )


class FakeResponses:
    def __init__(self, output_text: str = "The answer is grounded [S1].", error: Exception | None = None):
        self.output_text = output_text
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return type("Response", (), {"output_text": self.output_text})()


class FakeClient:
    def __init__(self, responses: FakeResponses):
        self.responses = type("Namespace", (), {})()
        self.responses.create = responses.create


class FakeProvider:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        assert "RETRIEVED PASSAGES" in prompt
        return self.text


def index_fixture():
    records = [
        ChunkRecord("doc:chunk-0000", "doc", "evidence.txt", None, 0, "evidence about cats", ("p1",), ()),
        ChunkRecord("doc:chunk-0001", "doc", "evidence.txt", None, 1, "unrelated dogs", ("p2",), ()),
    ]
    return build_index(records, TokenVectorEncoder()), TokenVectorEncoder()


def test_live_api_configuration_requires_environment_enable_and_per_call_authorization(monkeypatch) -> None:
    monkeypatch.delenv("RAG_ENABLE_LIVE_API", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    disabled = Settings.from_environment(load_env_file=False)
    with pytest.raises(LiveCallsDisabledError):
        OpenAIResponsesProvider.from_settings(disabled, user_authorized_this_call=True)

    monkeypatch.setenv("RAG_ENABLE_LIVE_API", "true")
    enabled = Settings.from_environment(load_env_file=False)
    with pytest.raises(ExplicitAuthorizationRequired):
        OpenAIResponsesProvider.from_settings(enabled, user_authorized_this_call=False)
    provider = OpenAIResponsesProvider.from_settings(enabled, user_authorized_this_call=True)
    assert provider.model == "test-model"
    assert provider._client.max_retries == 0


def test_openai_responses_adapter_uses_configured_model_and_no_hidden_retries() -> None:
    response_capture = FakeResponses("Output from API")
    client = FakeClient(response_capture)
    provider = OpenAIResponsesProvider("configured-model", client)
    assert provider.generate("prompt") == "Output from API"
    assert response_capture.calls == [
        {"model": "configured-model", "input": "prompt", "store": False, "max_output_tokens": 600}
    ]


def test_openai_responses_adapter_wraps_provider_errors_without_retrying() -> None:
    response_capture = FakeResponses(error=RuntimeError("private provider detail"))
    provider = OpenAIResponsesProvider("configured-model", FakeClient(response_capture))
    with pytest.raises(GenerationError, match="request failed") as error:
        provider.generate("prompt")
    assert "private provider detail" not in str(error.value)
    assert len(response_capture.calls) == 1


def test_pipeline_abstains_without_calling_provider_below_threshold() -> None:
    index, model = index_fixture()
    provider = FakeProvider("Would answer [S1].")
    result = answer_question("unmatched question", index, model, "dense", {"dense": 0.9}, provider)
    assert result.abstained
    assert result.status == "abstained"
    assert provider.calls == 0


def test_pipeline_returns_only_validated_source_citations() -> None:
    index, model = index_fixture()
    provider = FakeProvider("The evidence supports this answer [S1].")
    result = answer_question("evidence about cats", index, model, "dense", {"dense": 0.0}, provider)
    assert result.status == "answered"
    assert result.source_references_valid is True
    assert result.cited_sources[0].chunk_id == "doc:chunk-0000"
    serialized = result.as_dict()
    assert "retrieved_sources" in serialized
    assert "cited_sources" in serialized
    assert "citations" not in serialized


def test_pipeline_returns_safe_generation_failure_status() -> None:
    class ErrorProvider:
        def generate(self, prompt: str) -> str:
            raise GenerationError("safe provider failure")

    index, model = index_fixture()
    result = answer_question("evidence about cats", index, model, "dense", {"dense": 0.0}, ErrorProvider())
    assert result.status == "generation_error"
    assert result.answer == "The answer request failed. No answer was returned."


def test_pipeline_withholds_answer_with_unknown_source_id() -> None:
    index, model = index_fixture()
    provider = FakeProvider("Unsupported answer [S99].")
    result = answer_question("evidence about cats", index, model, "dense", {"dense": 0.0}, provider)
    assert result.status == "invalid_source_references"
    assert result.source_references_valid is False
    assert result.answer.startswith("The generated answer was withheld")


def test_no_source_id_is_not_valid_citation() -> None:
    valid, used, unknown = validate_source_ids("An uncited answer.", {"S1"})
    assert (valid, used, unknown) == (False, (), ())
