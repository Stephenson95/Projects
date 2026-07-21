import json

import httpx
import pytest

from src.llm.client import ChatMessage, LLMClientError, OllamaClient, get_llm_client
from src.models import ClassificationOutput, EmailCategory


def test_ollama_client_uses_schema_and_validates_mocked_response():
    captured_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "category": "billing",
                            "summary": "The customer asks about an invoice charge.",
                        }
                    ),
                },
                "prompt_eval_count": 42,
                "eval_count": 12,
            },
        )

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://ollama.test"
    ) as http_client:
        client = OllamaClient(
            base_url="http://ignored.test",
            timeout_seconds=1.0,
            http_client=http_client,
        )
        result = client.generate_structured(
            messages=(ChatMessage(role="user", content="Why this charge?"),),
            model_name="llama3.1:8b",
            temperature=0.0,
            output_model=ClassificationOutput,
        )

    assert result == ClassificationOutput(
        category=EmailCategory.billing,
        summary="The customer asks about an invoice charge.",
    )
    assert captured_request["model"] == "llama3.1:8b"
    assert captured_request["stream"] is False
    assert captured_request["format"] == ClassificationOutput.model_json_schema()


def test_get_llm_client_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_PROVIDER", "paid-provider")

    with pytest.raises(LLMClientError, match="Unsupported LLM_PROVIDER"):
        get_llm_client()
