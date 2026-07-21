"""Typed prompt configuration used to reproduce classifier behavior."""

from datetime import datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from src.models import ClassificationOutput


class FewShotExample(BaseModel):
    """Keep examples structured so prompt files cannot silently drift in shape."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_email: str = Field(min_length=1)
    output: ClassificationOutput


class PromptConfig(BaseModel):
    """Capture every prompt-time input needed to reproduce a classifier call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    timestamp: AwareDatetime
    system_prompt: str = Field(min_length=1)
    few_shot_examples: tuple[FewShotExample, ...] = ()
    model_name: str = Field(default="llama3.1:8b", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @property
    def created_at(self) -> datetime:
        """Expose a plain datetime without weakening timezone validation at load time."""

        return self.timestamp
