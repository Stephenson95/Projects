"""Load a versioned JSON corpus through the Phase 2 validation boundary."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.eval.models import GoldenDataset, TestCaseSource


class GoldenDatasetLoadError(ValueError):
    """Present filesystem and schema failures through one caller-facing error."""


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    """Reject malformed or under-covered data before it can influence an eval run."""

    dataset_path = Path(path)
    try:
        raw_json = dataset_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GoldenDatasetLoadError(f"Could not read golden dataset: {dataset_path}") from exc

    try:
        return GoldenDataset.model_validate_json(raw_json)
    except ValidationError as exc:
        raise GoldenDatasetLoadError(f"Invalid golden dataset in {dataset_path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Make dataset validation available without introducing a CLI dependency."""

    parser = argparse.ArgumentParser(description="Validate a versioned golden dataset")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/golden/golden_v0_llm_placeholder.json"),
    )
    args = parser.parse_args(argv)
    dataset = load_golden_dataset(args.path)
    placeholder_count = sum(case.source is TestCaseSource.llm_placeholder for case in dataset.cases)
    print(
        f"Validated {len(dataset.cases)} cases from dataset {dataset.dataset_version}; "
        f"status={dataset.status.value}; placeholders={placeholder_count}"
    )
    if placeholder_count:
        print(f"WARNING: {dataset.warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
