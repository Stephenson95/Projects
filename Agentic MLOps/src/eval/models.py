"""Typed contracts for golden-dataset records and provenance."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models import EmailCategory


class Difficulty(StrEnum):
    """Describe the behavior being stressed without implying label confidence."""

    easy = "easy"
    medium = "medium"
    hard = "hard"
    adversarial = "adversarial"


class TestCaseSource(StrEnum):
    """Make human verification an explicit data property, never an inference."""

    llm_placeholder = "llm_placeholder"
    human_verified = "human_verified"


class DatasetStatus(StrEnum):
    """Prevent a partially reviewed corpus from masquerading as ground truth."""

    placeholder_requires_human_review = "placeholder_requires_human_review"
    human_verified = "human_verified"


class TestCase(BaseModel):
    """Validate each expected behavior before the evaluation engine can consume it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    input_email: str = Field(min_length=1)
    expected_category: EmailCategory
    expected_summary: str = Field(min_length=1, max_length=500)
    difficulty: Difficulty
    notes: str = Field(min_length=1)
    source: TestCaseSource


class GoldenDataset(BaseModel):
    """Enforce the minimum coverage needed for a useful regression gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    status: DatasetStatus
    generation_method: str = Field(min_length=1)
    warning: str = Field(min_length=1)
    cases: tuple[TestCase, ...] = Field(min_length=50, max_length=100)

    @model_validator(mode="after")
    def validate_dataset_invariants(self) -> Self:
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("test case IDs must be unique")

        missing_categories = set(EmailCategory) - {case.expected_category for case in self.cases}
        if missing_categories:
            missing = ", ".join(sorted(category.value for category in missing_categories))
            raise ValueError(f"dataset is missing expected categories: {missing}")

        missing_difficulties = set(Difficulty) - {case.difficulty for case in self.cases}
        if missing_difficulties:
            missing = ", ".join(sorted(level.value for level in missing_difficulties))
            raise ValueError(f"dataset is missing difficulty tiers: {missing}")

        placeholder_count = sum(
            case.source is TestCaseSource.llm_placeholder for case in self.cases
        )
        if self.status is DatasetStatus.placeholder_requires_human_review:
            if placeholder_count == 0:
                raise ValueError("placeholder status requires at least one placeholder case")
            if "NOT GROUND TRUTH" not in self.warning:
                raise ValueError("placeholder dataset warning must say 'NOT GROUND TRUTH'")
        elif placeholder_count:
            raise ValueError("human-verified dataset cannot contain placeholder cases")

        return self
