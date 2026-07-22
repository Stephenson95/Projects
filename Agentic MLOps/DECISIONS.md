# Architectural Decisions

## 2026-07-21 — Phase 1: Typed provider seam with Ollama as the placeholder

**Decision.** All model access goes through the `LLMClient` protocol in
`src/llm/client.py`. The Phase 1 implementation selects local Ollama from environment
configuration and uses Ollama's JSON-schema structured-output mode. Prompt YAML is
validated into immutable Pydantic models, and the filename must equal `version_id`.

**Reasoning.** The classifier and the later judge need a common seam for deterministic
tests, retries, usage accounting, and an eventual provider swap. Passing provider-neutral
messages plus a requested Pydantic output type keeps those concerns outside evaluation
logic while avoiding a framework dependency that would obscure a small, interviewable
design. Keeping model name and temperature inside versioned prompt configuration makes
an evaluation reproducible from the prompt artifact. Matching the filename to the
embedded ID prevents two competing identities for the same configuration. Ollama
(`llama3.1:8b`) satisfies the brief's zero-cost placeholder requirement; adding OpenAI
later means implementing this protocol and changing environment configuration, not
rewriting the classifier.

**Trade-off.** Phase 1 supports only synchronous generation because its only consumer
is a single-email classifier. The provider contract will gain an async operation and
usage metadata in Phase 3, when batching and scoring require them; adding those now
would prematurely implement the evaluation engine.

## 2026-07-21 — Phase 1: Quarantine an LLM-generated dataset draft

**Decision.** Generate the requested full placeholder pass now as
`data/golden/golden_v0_llm_placeholder.json`, with 60 cases and
`"source": "llm_placeholder"` on every record. Do not treat it as the golden dataset
and do not implement Phase 2 dataset code at this checkpoint.

**Reasoning.** Synthetic examples are useful for exercising formats and edge-case
coverage, but calling model-generated labels ground truth would make the regression
signal circular and undermine the project. The project spec contains edge-case
descriptions rather than literal illustrative records, so the draft expands those
descriptions and the four-category feature definition. A conspicuous filename,
top-level warning, per-record source tag, and README warning make provenance auditable.
This is a deliberate, time-boxed scaffold only. Before the project is portfolio-ready,
a person must inspect every input, category, summary, difficulty, and note, then either
promote the case with human-verification provenance or replace it. Phase 2 remains the
checkpoint where validation and promotion policy will be implemented.

**Trade-off.** The draft provides breadth early, but its labels may encode model bias or
contain subtle mistakes. It must never be used to claim measured production quality.

## 2026-07-22 — Phase 2: Make provenance and coverage fail-closed invariants

**Decision.** Represent each case with an immutable Pydantic `TestCase` that requires an
explicit `source`, and represent the file with a `GoldenDataset` aggregate that validates
50–100 cases, unique IDs, complete category and difficulty coverage, and review status.
The aggregate rejects `status: human_verified` while any `llm_placeholder` remains.
All JSON access goes through `load_golden_dataset`, which normalizes file and validation
failures into one caller-facing error.

**Reasoning.** Provenance is part of label quality, not optional metadata. Validating it
at the same boundary as expected category and summary makes it impossible for later eval
code to accidentally load synthetic labels as trusted ground truth. Aggregate checks
belong on the dataset rather than in the future runner because duplicate IDs or missing
coverage make a corpus invalid regardless of how it is executed. Requiring the overall
status to remain provisional until the last placeholder is promoted supports incremental
human review without allowing a partially reviewed file to make a stronger claim than
its least-trusted record. The loader provides a single, testable seam for Phase 3.

This checkpoint intentionally leaves all 60 records as `llm_placeholder`. It is still a
time-boxed draft and is not portfolio-ready; a person must promote or replace every case
after reviewing the input, label, summary, difficulty, and notes.

**Trade-off.** The 50-case minimum and full-coverage rules make tiny experimental files
invalid through the production loader. Unit tests that need smaller fixtures should test
`TestCase` directly or deliberately construct complete aggregates; weakening production
validation for fixture convenience would make the regression gate less trustworthy.
