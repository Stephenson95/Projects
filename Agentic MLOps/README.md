# Model Regression Detection System

This project applies CI-style regression checks to an LLM customer-support email
classifier. Given an email, the classifier returns one routing category (`billing`,
`technical`, `account`, or `general`) and a one-sentence summary. The current checkpoint
also validates the versioned corpus that will drive later evaluation runs.

## Phase status

- **Phase 1 — complete:** typed classifier contract, versioned prompt loading,
  Ollama provider seam, mocked unit tests, and a manual smoke-test command.
- **Phase 2 — complete:** typed `TestCase` and dataset contracts, fail-closed JSON
  loading, coverage/provenance validation, and a dataset validation command.
- **Phases 3–6 — not started:** checkpoint-gated pending review.

The repository also contains a deliberately time-boxed, LLM-generated draft dataset
at `data/golden/golden_v0_llm_placeholder.json`. It is present now because the build
brief explicitly requests the full placeholder pass during Phase 1. Every record has
`"source": "llm_placeholder"`; none of these records is hand-verified ground truth.
Before this project is portfolio-ready, a person must review every case and either
promote it with documented verification or replace it. Phase 2 deliberately validates
this provenance instead of upgrading the records: the dataset status cannot become
`human_verified` while even one `llm_placeholder` remains.

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

## Run Phase 2

Validate the checked-in dataset directly:

```powershell
python -m src.eval.dataset data/golden/golden_v0_llm_placeholder.json
```

The command reports the dataset version, case count, review status, and placeholder
count. It exits with an error if the file is unreadable, malformed, outside the 50–100
case range, has duplicate IDs, omits a category or difficulty tier, lacks per-record
provenance, or claims human verification while placeholder records remain.

Run only the Phase 2 tests:

```powershell
python -m pytest tests/eval/test_dataset.py -v
```

## Phase 2 dataset contract

- `src/eval/models.py` defines immutable `TestCase` and `GoldenDataset` boundaries.
- `src/eval/dataset.py` is the only JSON loading path and normalizes read/schema errors.
- The 60-case placeholder corpus covers every category and difficulty tier, including
  short, ambiguous, typo-laden, mixed-language, sarcastic, and adversarial inputs.
- `source` tracks provenance per record; `status` describes the corpus as a whole.
- No LLM is called during dataset loading or tests.
