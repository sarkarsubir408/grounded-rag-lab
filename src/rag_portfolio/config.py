from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from .embeddings import MODEL_NAME, MODEL_REVISION


class LiveCallsDisabledError(RuntimeError):
    pass


class ExplicitAuthorizationRequired(RuntimeError):
    pass


class ConfigurationError(RuntimeError):
    pass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    live_api_enabled: bool = False
    openai_model: str = ""
    openai_api_key: str = field(default="", repr=False)
    openai_timeout_seconds: float = 30.0
    embedding_model_name: str = MODEL_NAME
    embedding_model_revision: str = MODEL_REVISION
    rrf_constant: int = 60
    calibration_file: str = "reports/public_release/abstention_calibration.json"

    @classmethod
    def from_environment(cls, load_env_file: bool = True) -> "Settings":
        if load_env_file:
            load_dotenv(override=False)
        return cls(
            live_api_enabled=_as_bool(os.getenv("RAG_ENABLE_LIVE_API"), default=False),
            openai_model=os.getenv("OPENAI_MODEL", "").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", MODEL_NAME),
            embedding_model_revision=os.getenv("EMBEDDING_MODEL_REVISION", MODEL_REVISION),
            rrf_constant=int(os.getenv("RRF_CONSTANT", "60")),
            calibration_file=os.getenv(
                "ABSTENTION_CALIBRATION_FILE",
                "reports/public_release/abstention_calibration.json",
            ),
        )

    def validate_live_call(self, user_authorized_this_call: bool) -> None:
        if not self.live_api_enabled:
            raise LiveCallsDisabledError(
                "Live API calls are disabled. Set RAG_ENABLE_LIVE_API=true to enable them."
            )
        if not user_authorized_this_call:
            raise ExplicitAuthorizationRequired(
                "No request was sent. Authorize this individual billable API call first."
            )
        if not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required when live calls are enabled.")
        if not self.openai_model:
            raise ConfigurationError("Set OPENAI_MODEL to a model available to your OpenAI project.")
