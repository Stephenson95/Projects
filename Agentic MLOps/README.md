# Model Regression Detection System

This project applies CI-style regression checks to an LLM customer-support email
classifier. Phase 1 defines the feature under test: given an email, the classifier
returns one routing category (`billing`, `technical`, `account`, or `general`) and a
one-sentence summary.

## Phase status

- **Phase 1 — complete:** typed classifier contract, versioned prompt loading,
  Ollama provider seam, mocked unit tests, and a manual smoke-test command.
- **Phases 2–6 — not started:** checkpoint-gated pending review.

The repository also contains a deliberately time-boxed, LLM-generated draft dataset
at `data/golden/golden_v0_llm_placeholder.json`. It is present now because the build
brief explicitly requests the full placeholder pass during Phase 1. Every record has
`"source": "llm_placeholder"`; none of these records is hand-verified ground truth.
Before this project is portfolio-ready, a person must review every case and either
promote it with documented verification or replace it. Phase 2 will implement the
actual golden-dataset validation workflow after the Phase 1 checkpoint is approved.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) for manual classifier calls
- Local model `llama3.1:8b`

No API key or paid service is used in Phase 1. Unit tests mock the provider boundary
and never contact Ollama.

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.1:8b
```

The code reads `LLM_PROVIDER`, `OLLAMA_BASE_URL`, and `OLLAMA_TIMEOUT_SECONDS` from
the process environment. `.env.example` documents them, but Phase 1 intentionally
does not auto-load `.env`; set variables in your shell or use the defaults shown.

## Run Phase 1

Run the deterministic test suite and lint checks:

```powershell
python -m pytest -v
python -m ruff check .
```

With Ollama running locally, classify one email:

```powershell
python -m src.classifier "I was charged twice this month; please refund one payment."
```

Expected output is validated JSON shaped like:

```json
{
  "category": "billing",
  "summary": "The customer requests a refund for a duplicate monthly charge."
}
```

## Phase 1 architecture

- `src/classifier.py` owns only feature orchestration and prompt message construction.
- `src/models.py` contains the category and structured-output contracts.
- `src/prompts/` validates immutable, version-addressed YAML prompt configuration.
- `src/llm/client.py` is the only provider boundary. It currently selects Ollama via
  environment config and requests schema-constrained output. A future OpenAI client
  can implement the same protocol without changing classifier or evaluation logic.
- `prompts/support_email_classifier_v1.yaml` captures the model, temperature, system
  prompt, and typed few-shot examples required to reproduce a call.
