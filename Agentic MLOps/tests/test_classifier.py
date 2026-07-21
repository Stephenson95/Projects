from datetime import UTC, datetime

import pytest

from src import classifier
from src.models import ClassificationOutput, EmailCategory
from src.prompts.schema import FewShotExample, PromptConfig


class FakeLLMClient:
    def __init__(self, output: ClassificationOutput) -> None:
        self.output = output
        self.messages = ()
        self.model_name = ""
        self.temperature = -1.0
        self.output_model: type[ClassificationOutput] | None = None

    def generate_structured(
        self,
        *,
        messages: tuple,
        model_name: str,
        temperature: float,
        output_model: type[ClassificationOutput],
    ) -> ClassificationOutput:
        self.messages = messages
        self.model_name = model_name
        self.temperature = temperature
        self.output_model = output_model
        return self.output


def _config() -> PromptConfig:
    return PromptConfig(
        version_id="test_v1",
        timestamp=datetime(2026, 7, 21, tzinfo=UTC),
        system_prompt="Classify the email and return JSON.",
        model_name="llama3.1:8b",
        temperature=0.0,
        few_shot_examples=(
            FewShotExample(
                input_email="Refund the duplicate charge.",
                output=ClassificationOutput(
                    category=EmailCategory.billing,
                    summary="The customer requests a duplicate-charge refund.",
                ),
            ),
        ),
    )


def test_classify_email_returns_valid_output_and_builds_messages(monkeypatch: pytest.MonkeyPatch):
    expected = ClassificationOutput(
        category=EmailCategory.technical,
        summary="The customer reports that the application will not open.",
    )
    fake = FakeLLMClient(expected)
    monkeypatch.setattr(classifier.llm_client, "get_llm_client", lambda: fake)

    result = classifier.classify_email("  The app won't open.  ", _config())

    assert result == expected
    assert fake.model_name == "llama3.1:8b"
    assert fake.temperature == 0.0
    assert fake.output_model is ClassificationOutput
    assert [message.role for message in fake.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert fake.messages[-1].content == ("<customer_email>\nThe app won't open.\n</customer_email>")


def test_classify_email_rejects_blank_input(monkeypatch: pytest.MonkeyPatch):
    def unexpected_client_lookup() -> FakeLLMClient:
        raise AssertionError("LLM client should not be created for invalid input")

    monkeypatch.setattr(classifier.llm_client, "get_llm_client", unexpected_client_lookup)

    with pytest.raises(ValueError, match="non-whitespace"):
        classifier.classify_email("   ", _config())
