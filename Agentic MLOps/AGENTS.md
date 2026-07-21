# AGENTS.md

## Project

Model Regression Detection System — a CI/CD-style evaluation pipeline that
tests an LLM-powered customer-support email classifier against a
hand-labeled golden dataset on every prompt or model change, detects
quality regressions and slow drift, and alerts a team via Slack before bad
output reaches users.

This is a portfolio project. Every architectural decision should be one
the author can defend and explain in an interview — prefer explicit,
inspectable code over "magic" abstractions. Optimize for readability and
correctness over cleverness.

## Stack (pinned)

- Python 3.11+
- OpenAI API (`gpt-4o-mini` for the classifier under test, `gpt-4o` for
  LLM-as-judge scoring) — access goes through one abstraction so another
  provider can be swapped in later without touching eval logic
- Pydantic v2 for every typed boundary (prompt config, test cases, eval
  results, reports)
- SQLite + JSON files for storage (zero external infra required)
- RAGAS or DeepEval for LLM-as-judge summary scoring
- Slack Incoming Webhooks for alerting
- GitHub Actions for CI
- Streamlit or a static HTML report for the diff dashboard
- Docker for packaging

Optional swap: if you want this artifact to also demonstrate continuity
with your day-to-day stack, MLflow Tracking can replace the raw
SQLite run/metric log (see PROJECT_SPEC.md → "Optional MLflow swap").
Don't do this unless asked — it adds a dependency the base spec doesn't
need.

## Repo layout convention

```
src/
  classifier.py          # the LLM feature under test
  prompts/
    loader.py
    schema.py             # PromptConfig
  eval/
    models.py             # TestCase, EvalResult, RunSummary
    runner.py             # test runner, async batching
    scoring.py             # category match, LLM-as-judge, latency, tokens
    diff.py                  # comparison logic + significance thresholds
    drift.py                   # rolling average drift detection
  reporting/
    html_report.py
  alerting/
    slack.py
  storage/
    db.py                     # SQLite persistence
prompts/                       # versioned YAML prompt files
data/golden/                    # hand-labeled golden dataset (JSON)
tests/                              # mirrors src/
reports/                              # generated HTML diffs (sample only in git)
.github/workflows/
Dockerfile
.env.example
README.md
DECISIONS.md
```

## Coding standards

- Full type hints everywhere; no bare `dict`/`Any` at module boundaries —
  use Pydantic models instead
- Every LLM call goes through one wrapped client function so retries, cost
  logging, and test mocking have a single seam
- Docstrings explain *why*, not just *what*
- Format with `ruff format`; lint with `ruff check`; both clean before a
  phase is considered done
- No hardcoded API keys or webhook URLs — everything through environment
  variables, with `.env.example` kept current

## Testing rules

- Unit tests must NOT make real paid API calls. Mock the LLM client (fixed
  fixtures or a fake client returning canned responses) so the suite runs
  free and deterministically in CI.
- The golden dataset is test data, but the actual hand-labeled cases must
  never be LLM-generated — that undermines the entire premise of the
  project. If scaffolding requires placeholder cases, mark them clearly
  (`# PLACEHOLDER — replace with a hand-written, human-verified case`) and
  say so explicitly rather than silently shipping synthetic data as if it
  were ground truth.
- Every phase ends with `pytest` green and `ruff check` clean before moving
  to the next phase.

## Workflow

- Work one phase at a time, in the order defined in PROJECT_SPEC.md. Do not
  jump ahead.
- At the end of each phase: implement → test (mocked) → lint → update
  README → add a `DECISIONS.md` entry for any non-obvious choice → commit
  with a descriptive message → summarize what was built before starting
  the next phase.
- If a requirement is ambiguous, make the most defensible production-grade
  choice, note the assumption in `DECISIONS.md`, and continue. Only stop
  and ask if genuinely blocked (e.g., missing credentials).

## Definition of done (whole project)

- [ ] All 6 phases complete, tests passing
- [ ] `docker build` succeeds; container runs the eval pipeline end-to-end
      against the sample golden dataset
- [ ] GitHub Action triggers on PRs touching `/prompts`, posts a PR
      comment, blocks merge on critical regressions
- [ ] Slack alert fires correctly against a test webhook
- [ ] README reads like internal onboarding docs, not a tutorial
- [ ] `DECISIONS.md` has at least one entry per phase
