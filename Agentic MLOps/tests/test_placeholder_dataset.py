import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_placeholder_dataset_is_full_and_explicitly_quarantined():
    dataset_path = PROJECT_ROOT / "data" / "golden" / "golden_v0_llm_placeholder.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"]

    assert 50 <= len(cases) <= 100
    assert dataset["status"] == "placeholder_requires_human_review"
    assert "NOT GROUND TRUTH" in dataset["warning"]
    assert {case["difficulty"] for case in cases} == {
        "easy",
        "medium",
        "hard",
        "adversarial",
    }
    assert {case["expected_category"] for case in cases} == {
        "billing",
        "technical",
        "account",
        "general",
    }
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["source"] == "llm_placeholder" for case in cases)
