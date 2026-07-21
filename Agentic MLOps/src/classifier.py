"""Customer-support email classifier used as the evaluation target."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.llm import client as llm_client
from src.llm.client import ChatMessage
from src.models import ClassificationOutput
from src.prompts.loader import load_prompt
from src.prompts.schema import PromptConfig


def classify_email(email: str, config: PromptConfig) -> ClassificationOutput:
    """Keep the feature thin so evaluation measures prompt/model behavior, not glue code."""

    normalized_email = email.strip()
    if not normalized_email:
        raise ValueError("email must contain non-whitespace text")

    return llm_client.get_llm_client().generate_structured(
        messages=_build_messages(normalized_email, config),
        model_name=config.model_name,
        temperature=config.temperature,
        output_model=ClassificationOutput,
    )


def _build_messages(email: str, config: PromptConfig) -> tuple[ChatMessage, ...]:
    messages = [ChatMessage(role="system", content=config.system_prompt)]
    for example in config.few_shot_examples:
        messages.extend(
            (
                ChatMessage(role="user", content=_wrap_email(example.input_email)),
                ChatMessage(role="assistant", content=example.output.model_dump_json()),
            )
        )
    messages.append(ChatMessage(role="user", content=_wrap_email(email)))
    return tuple(messages)


def _wrap_email(email: str) -> str:
    return f"<customer_email>\n{email}\n</customer_email>"


def main(argv: Sequence[str] | None = None) -> int:
    """Provide a small manual smoke-test path without adding a separate CLI framework."""

    parser = argparse.ArgumentParser(description="Classify one support email with Ollama")
    parser.add_argument("email", help="Raw customer-support email text")
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("prompts/support_email_classifier_v1.yaml"),
        help="Path to a versioned prompt YAML file",
    )
    args = parser.parse_args(argv)
    result = classify_email(args.email, load_prompt(args.prompt))
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
