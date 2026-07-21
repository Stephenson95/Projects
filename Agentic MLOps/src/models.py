"""Shared typed contracts for the feature under test."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EmailCategory(StrEnum):
    """The deliberately small routing taxonomy evaluated by this project."""

    billing = "billing"
    technical = "technical"
    account = "account"
    general = "general"


class ClassificationOutput(BaseModel):
    """Constrain model output before it can reach evaluation or routing code."""

    model_config = ConfigDict(extra="forbid")

    category: EmailCategory
    summary: str = Field(min_length=1, max_length=500)
