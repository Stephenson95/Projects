"""Load versioned YAML prompts into validated, immutable configuration."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from src.prompts.schema import PromptConfig


class PromptLoadError(ValueError):
    """Give callers one actionable error type for malformed prompt artifacts."""


def load_prompt(path: str | Path) -> PromptConfig:
    """Fail closed when a prompt cannot uniquely identify its on-disk version."""

    prompt_path = Path(path)
    try:
        raw = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PromptLoadError(f"Could not read prompt file: {prompt_path}") from exc
    except yaml.YAMLError as exc:
        raise PromptLoadError(f"Invalid YAML in prompt file: {prompt_path}") from exc

    if not isinstance(raw, dict):
        raise PromptLoadError(f"Prompt file must contain a YAML mapping: {prompt_path}")

    try:
        config = PromptConfig.model_validate(raw)
    except ValidationError as exc:
        raise PromptLoadError(f"Invalid prompt configuration in {prompt_path}: {exc}") from exc

    if config.version_id != prompt_path.stem:
        raise PromptLoadError(
            f"Prompt version_id {config.version_id!r} must match filename {prompt_path.stem!r}"
        )
    return config
