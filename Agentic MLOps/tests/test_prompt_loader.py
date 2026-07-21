from pathlib import Path

import pytest

from src.models import EmailCategory
from src.prompts.loader import PromptLoadError, load_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_real_prompt_yaml_loads_into_prompt_config():
    config = load_prompt(PROJECT_ROOT / "prompts" / "support_email_classifier_v1.yaml")

    assert config.version_id == "support_email_classifier_v1"
    assert config.model_name == "llama3.1:8b"
    assert config.temperature == 0.0
    assert len(config.few_shot_examples) == 4
    assert {example.output.category for example in config.few_shot_examples} == set(EmailCategory)
    assert config.timestamp.utcoffset() is not None


def test_prompt_version_must_match_filename(tmp_path: Path):
    prompt_path = tmp_path / "filename_v2.yaml"
    prompt_path.write_text(
        "\n".join(
            (
                "version_id: content_v1",
                'timestamp: "2026-07-21T00:00:00+10:00"',
                "system_prompt: classify",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(PromptLoadError, match="must match filename"):
        load_prompt(prompt_path)
