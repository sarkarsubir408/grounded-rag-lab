from __future__ import annotations

from typing import Protocol

from .config import Settings


class GenerationError(RuntimeError):
    """Safe application-level error for provider failures."""


class GenerationProvider(Protocol):
    def generate(self, prompt: str) -> str: ...


class OpenAIResponsesProvider:
    """OpenAI Responses API adapter. Client retries are disabled to avoid hidden billable repeats."""

    def __init__(self, model: str, client: object):
        self.model = model
        self._client = client

    @classmethod
    def from_settings(
        cls, settings: Settings, user_authorized_this_call: bool
    ) -> "OpenAIResponsesProvider":
        settings.validate_live_call(user_authorized_this_call)
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=0,
        )
        return cls(settings.openai_model, client)

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.responses.create(
                model=self.model,
                input=prompt,
                store=False,
                max_output_tokens=600,
            )
            text = getattr(response, "output_text", "")
        except Exception as error:
            # Do not expose request headers, credentials, or provider internals in the UI.
            raise GenerationError(
                "The OpenAI request failed. Check connectivity, model access, and API configuration; no automatic retry was made."
            ) from error
        if not isinstance(text, str) or not text.strip():
            raise GenerationError("The OpenAI response contained no text output.")
        return text.strip()
