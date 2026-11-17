"""One-shot, question-bound authorization for live answer generation."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import TypeVar

QUESTION_STATE_KEY = "rag_question"
AUTHORIZATION_CHECKBOX_KEY = "authorize_live_api_call"
PENDING_AUTHORIZATION_KEY = "pending_live_api_question"

ProviderT = TypeVar("ProviderT")


def capture_question_authorization(
    state: MutableMapping[str, object], generation_enabled: bool
) -> None:
    """Capture consent for the submitted question, then clear the checkbox."""
    question = str(state.get(QUESTION_STATE_KEY, "")).strip()
    checkbox_checked = bool(state.get(AUTHORIZATION_CHECKBOX_KEY, False))
    state[PENDING_AUTHORIZATION_KEY] = (
        question if generation_enabled and checkbox_checked and question else None
    )
    state[AUTHORIZATION_CHECKBOX_KEY] = False


def create_authorized_provider(
    state: MutableMapping[str, object],
    submitted_question: str,
    provider_factory: Callable[[], ProviderT],
) -> ProviderT | None:
    """Consume matching one-shot consent before constructing a provider."""
    authorized_question = state.pop(PENDING_AUTHORIZATION_KEY, None)
    question = submitted_question.strip()
    if not question or authorized_question != question:
        return None
    return provider_factory()
