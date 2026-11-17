from __future__ import annotations

import pytest

from rag_portfolio.authorization import (
    AUTHORIZATION_CHECKBOX_KEY,
    QUESTION_STATE_KEY,
    capture_question_authorization,
    create_authorized_provider,
)


def test_authorization_is_question_bound_and_only_consumed_once() -> None:
    state = {
        QUESTION_STATE_KEY: "  What is RRF?  ",
        AUTHORIZATION_CHECKBOX_KEY: True,
    }
    class MockProvider:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self) -> str:
            self.calls += 1
            return "mock answer"

    provider = MockProvider()
    factory_calls = 0

    def provider_factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        return provider

    capture_question_authorization(state, generation_enabled=True)
    assert state[AUTHORIZATION_CHECKBOX_KEY] is False
    assert create_authorized_provider(state, "What is RRF?", provider_factory) is provider
    assert provider.generate() == "mock answer"
    assert create_authorized_provider(state, "What is RRF?", provider_factory) is None
    assert factory_calls == 1
    assert provider.calls == 1


def test_authorization_for_one_question_cannot_transfer_to_another() -> None:
    state = {
        QUESTION_STATE_KEY: "Question A",
        AUTHORIZATION_CHECKBOX_KEY: True,
    }
    factory_calls = 0

    def provider_factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        return object()

    capture_question_authorization(state, generation_enabled=True)
    assert create_authorized_provider(state, "Question B", provider_factory) is None
    assert create_authorized_provider(state, "Question A", provider_factory) is None
    assert factory_calls == 0


@pytest.mark.parametrize(
    "generation_enabled,checkbox_checked",
    [(False, True), (True, False)],
)
def test_disabled_or_unchecked_consent_never_creates_provider(
    generation_enabled: bool, checkbox_checked: bool
) -> None:
    state = {
        QUESTION_STATE_KEY: "Question A",
        AUTHORIZATION_CHECKBOX_KEY: checkbox_checked,
    }
    capture_question_authorization(state, generation_enabled)
    assert create_authorized_provider(state, "Question A", object) is None


def test_authorization_is_consumed_if_mocked_generation_fails() -> None:
    state = {
        QUESTION_STATE_KEY: "Question A",
        AUTHORIZATION_CHECKBOX_KEY: True,
    }
    factory_calls = 0

    class FailingMockProvider:
        def generate(self) -> str:
            raise RuntimeError("mock generation failure")

    provider = FailingMockProvider()

    def provider_factory() -> FailingMockProvider:
        nonlocal factory_calls
        factory_calls += 1
        return provider

    capture_question_authorization(state, generation_enabled=True)
    authorized_provider = create_authorized_provider(state, "Question A", provider_factory)
    assert authorized_provider is provider
    with pytest.raises(RuntimeError, match="mock generation failure"):
        authorized_provider.generate()

    assert create_authorized_provider(state, "Question A", provider_factory) is None
    assert factory_calls == 1
