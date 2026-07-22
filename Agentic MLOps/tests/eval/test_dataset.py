import json
from pathlib import Path
from typing import cast

import pytest

from src.eval.dataset import GoldenDatasetLoadError, load_golden_dataset
from src.eval.models import DatasetStatus, Difficulty
from src.eval.models import TestCaseSource as CaseSource
from src.models import EmailCategory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "golden" / "golden_v0_llm_placeholder.json"


def test_real_placeholder_dataset_validates_with_full_coverage():
    dataset = load_golden_dataset(DATASET_PATH)

    assert dataset.dataset_version == "0.1.0-llm-placeholder"
    assert dataset.status is DatasetStatus.placeholder_requires_human_review
    assert len(dataset.cases) == 60
    assert {case.expected_category for case in dataset.cases} == set(EmailCategory)
    assert {case.difficulty for case in dataset.cases} == set(Difficulty)
    assert all(case.source is CaseSource.llm_placeholder for case in dataset.cases)
    assert len({case.id for case in dataset.cases}) == len(dataset.cases)


def test_loader_rejects_duplicate_test_case_ids(tmp_path: Path):
    raw_dataset = _read_raw_dataset()
    cases = _get_raw_cases(raw_dataset)
    cases[1]["id"] = cases[0]["id"]
    invalid_path = _write_dataset(tmp_path, raw_dataset)

    with pytest.raises(GoldenDatasetLoadError, match="test case IDs must be unique"):
        load_golden_dataset(invalid_path)


def test_loader_rejects_record_without_provenance(tmp_path: Path):
    raw_dataset = _read_raw_dataset()
    del _get_raw_cases(raw_dataset)[0]["source"]
    invalid_path = _write_dataset(tmp_path, raw_dataset)

    with pytest.raises(GoldenDatasetLoadError, match="source"):
        load_golden_dataset(invalid_path)


def test_human_verified_status_rejects_placeholder_records(tmp_path: Path):
    raw_dataset = _read_raw_dataset()
    raw_dataset["status"] = "human_verified"
    invalid_path = _write_dataset(tmp_path, raw_dataset)

    with pytest.raises(
        GoldenDatasetLoadError,
        match="human-verified dataset cannot contain placeholder cases",
    ):
        load_golden_dataset(invalid_path)


def test_loader_rejects_missing_file(tmp_path: Path):
    with pytest.raises(GoldenDatasetLoadError, match="Could not read"):
        load_golden_dataset(tmp_path / "missing.json")


def _read_raw_dataset() -> dict[str, object]:
    return cast(dict[str, object], json.loads(DATASET_PATH.read_text(encoding="utf-8")))


def _get_raw_cases(dataset: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], dataset["cases"])


def _write_dataset(tmp_path: Path, dataset: dict[str, object]) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path
