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
