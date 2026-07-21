# Project Spec: Model Regression Detection System

## Problem statement

AI teams ship prompt and model changes with no automated way to know if
quality regressed until users complain. This project builds a CI/CD-style
pipeline that evaluates any prompt/model change against a golden dataset,
flags regressions and slow drift, and blocks bad merges — the same rigor
software engineering already applies to code, applied to model behavior.

## The feature under test

A customer support email classifier: given raw email text, return a
category (`billing | technical | account | general`) and a one-sentence
summary. The classifier itself is intentionally simple — it exists to give
the eval system something real to evaluate.

## Data models (Pydantic v2)

```python
class PromptConfig(BaseModel):
    version_id: str
    timestamp: datetime
    system_prompt: str
    few_shot_examples: list[dict] = []
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0

class EmailCategory(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    general = "general"

class ClassificationOutput(BaseModel):
    category: EmailCategory
    summary: str

class TestCase(BaseModel):
    id: str
    input_email: str
    expected_category: EmailCategory
    expected_summary: str
    difficulty: Literal["easy", "medium", "hard", "adversarial"]
    notes: str

class EvalResult(BaseModel):
    test_case_id: str
    prompt_version: str
    actual_category: EmailCategory
    actual_summary: str
    category_match: bool
    summary_score: int  # 1-5, LLM-as-judge
    latency_ms: float
    tokens_used: int
    cost_usd: float

class RunSummary(BaseModel):
    run_id: str
    prompt_version: str
    timestamp: datetime
    pass_rate: float
    per_category_accuracy: dict[EmailCategory, float]
    regressions: list[str]   # test_case_ids that flipped pass->fail
    improvements: list[str]  # test_case_ids that flipped fail->pass
    status: Literal["pass", "warn", "critical"]
```

## Directory layout

See AGENTS.md → "Repo layout convention" for the canonical tree.

## Non-functional requirements

- Default to `gpt-4o-mini` for the classifier; reserve `gpt-4o` for
  LLM-as-judge scoring only, to keep eval-run cost low
- Async batching for the test runner — don't run 100 test cases serially
- Every eval run must be idempotent and reproducible from its
  `PromptConfig` version alone
- Configurable regression thresholds (default: warn at 3% delta, critical
  at 8% delta) — no magic numbers hardcoded in logic

## Optional MLflow swap

The base spec uses SQLite + JSON for run storage, which keeps the project
dependency-free and portable. If you'd rather have the artifact double as
a demonstration of your existing MLflow experience, swap `storage/db.py`
for MLflow Tracking: log each eval run as an MLflow run, log
`pass_rate`, `per_category_accuracy`, and `regression_count` as metrics,
and log the golden dataset version and prompt version as tags. This is a
drop-in replacement for Phase 3/4 storage — don't do it unless you
specifically want that framing, since it adds an external dependency the
base version doesn't need.

## Phase-by-phase plan

### Phase 1 — Define the LLM feature under test
- `classifier.py`: single function `classify_email(email: str, config: PromptConfig) -> ClassificationOutput`
- Prompts stored as versioned YAML in `/prompts` (version id, timestamp,
  system prompt, few-shot examples)
- **Done when:** calling the classifier with a mocked LLM client returns a
  valid `ClassificationOutput`; a real prompt YAML file exists and loads
  into a `PromptConfig`

### Phase 2 — Build the golden dataset
- 50-100 test cases in `data/golden/`, **hand-written, not LLM-generated**
- Deliberately include ambiguous, very short, typo-laden, mixed-language,
  and sarcastic emails, tagged with `difficulty`
- Dataset file itself is versioned (filename or field)
- **Done when:** the JSON file validates against `TestCase`, includes at
  least one case per difficulty tier, and a handful marked
  `# PLACEHOLDER` are clearly flagged for the author to replace with real
  hand-written cases before this is portfolio-ready

### Phase 3 — Evaluation engine
- `runner.py`: takes a `PromptConfig` + golden dataset, runs every case
  async, collects raw outputs
- `scoring.py`: category exact-match (binary) + summary relevance via
  LLM-as-judge (1-5) + latency + token usage, all stored per case
- `diff.py`: compares current run to previous run — pass-rate delta,
  per-category delta, specific regressions and improvements
- Configurable significance thresholds (3% warn / 8% critical)
- **Done when:** running the same `PromptConfig` twice produces identical
  `EvalResult`s (with mocked LLM), and running two different configs
  produces a correct `RunSummary` diff with the right regressions listed

### Phase 4 — Alerting and reporting
- HTML diff report: run metadata, scorecard vs. baseline, side-by-side
  regressed cases, trend chart over last N runs
- Slack webhook message: status, headline numbers, link to report
- Drift detection: 7-run rolling average; fire a "slow drift" warning if
  it crosses a threshold even when no single run triggered an alert
- **Done when:** a mocked regression run produces a valid HTML report and
  a correctly formatted (but not actually sent, in tests) Slack payload

### Phase 5 — CI/CD integration
- GitHub Action: triggers on PRs touching `/prompts`, runs eval, posts a
  PR comment with pass/fail, blocks merge on critical regressions
- Dockerfile: packages runner + golden dataset + reporting layer; accepts
  `OPENAI_API_KEY`, `SLACK_WEBHOOK_URL`, threshold configs as env vars
- README written as onboarding docs: setup, how to add test cases, how to
  adjust thresholds, architecture rationale
- **Done when:** `docker build` succeeds and the container runs the full
  pipeline against the sample dataset with mocked/test credentials

### Phase 6 — Portfolio polish
- Not code — but leave clear notes in README for: a 3-minute walkthrough
  script (change a prompt → trigger eval → show Slack alert → walk the
  diff report) and a short write-up of the golden-dataset design decision
  for interviews
