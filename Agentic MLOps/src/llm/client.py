"""The single provider seam for every current and future LLM call."""

from __future__ import annotations

import os
from typing import Literal, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class LLMClientError(RuntimeError):
    """Normalize provider and response failures for application callers."""


class ChatMessage(BaseModel):
    """Use a provider-neutral chat boundary throughout the application."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    """Allow classifier and judge implementations to share one replaceable seam."""

    def generate_structured(
        self,
        *,
        messages: tuple[ChatMessage, ...],
        model_name: str,
        temperature: float,
        output_model: type[StructuredOutputT],
    ) -> StructuredOutputT:
        """Return provider output only after schema validation."""


class _OllamaOptions(BaseModel):
    temperature: float


class _OllamaChatRequest(BaseModel):
    model: str
    messages: tuple[ChatMessage, ...]
    format: dict[str, object]
    stream: bool = False
    options: _OllamaOptions


class _OllamaResponseMessage(BaseModel):
    role: str
    content: str


class _OllamaChatResponse(BaseModel):
    message: _OllamaResponseMessage
    prompt_eval_count: int = Field(default=0, ge=0)
    eval_count: int = Field(default=0, ge=0)


class OllamaClient:
    """Use Ollama structured outputs so local calls obey the same contract as tests."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client

    def generate_structured(
        self,
        *,
        messages: tuple[ChatMessage, ...],
        model_name: str,
        temperature: float,
        output_model: type[StructuredOutputT],
    ) -> StructuredOutputT:
        request = _OllamaChatRequest(
            model=model_name,
            messages=messages,
            format=output_model.model_json_schema(),
            options=_OllamaOptions(temperature=temperature),
        )
        try:
            response = self._post(request)
            response.raise_for_status()
            ollama_response = _OllamaChatResponse.model_validate_json(response.text)
            return output_model.model_validate_json(ollama_response.message.content)
        except (httpx.HTTPError, ValidationError) as exc:
            raise LLMClientError(f"Ollama structured generation failed: {exc}") from exc

    def _post(self, request: _OllamaChatRequest) -> httpx.Response:
        request_body = request.model_dump(mode="json")
        if self._http_client is not None:
            return self._http_client.post("/api/chat", json=request_body)

        with httpx.Client(base_url=self._base_url, timeout=self._timeout_seconds) as client:
            return client.post("/api/chat", json=request_body)


def get_llm_client() -> LLMClient:
    """Select the provider from environment config without leaking it into eval logic."""

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider != "ollama":
        raise LLMClientError(
            f"Unsupported LLM_PROVIDER {provider!r}; Phase 1 currently supports 'ollama'"
        )

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout_raw = os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as exc:
        raise LLMClientError("OLLAMA_TIMEOUT_SECONDS must be numeric") from exc
    if timeout_seconds <= 0:
        raise LLMClientError("OLLAMA_TIMEOUT_SECONDS must be greater than zero")

    return OllamaClient(base_url=base_url, timeout_seconds=timeout_seconds)
