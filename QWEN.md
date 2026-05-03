# marimo-investigate

## Project Overview

A Python-based document analysis tool for extracting structured information from **Russian government documents** related to national development goals, projects, and indicators. The project uses LLMs (via [pydantic-ai](https://github.com/pydantic/pydantic-ai)) for structured output extraction and [marimo](https://github.com/marimo-team/marimo) for interactive notebooks.

**Core purpose:** Parse Russian government documents (PDFs, DOCX files) and extract:
- National Development Goals (НЦР — Национальные цели развития)
- National Projects (НП — Национальные проекты)
- Federal Projects (ФП)
- State Programs (ГП — Государственные программы)
- Indicators with target values by year (2024–2036)
- ArchiMate-style relationships between entities

**Output:** Structured JSON matching a defined schema (`json-schema.json`), suitable for downstream analysis, graph databases (Neo4j), or ArchiMate modeling.

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.13+ |
| Package Manager | uv |
| LLM Framework | pydantic-ai (Agent-based, multi-provider) |
| LLM Providers | Azure OpenAI, OpenAI, OpenRouter, Anthropic, Mistral, Ollama, LM Studio |
| Notebooks | marimo (reactive, interactive) |
| Document Parsing | pypdf (PDFs), DOCX support via `common/document_reader.py` |
| Data Processing | pandas |
| Graph Database | neo4j (optional, for downstream storage) |
| Yandex AI | yandex-ai-studio-sdk, yandex-cloud-ml-sdk |
| Linting | ruff |
| Caching | diskcache |
| Logging | structlog |
| Retries | backoff |

## Project Structure

```
marimo-investigate/
├── __marimo__/              # Marimo notebook workspace (git-ignored)
├── adapters/
│   └── llm/                 # LLM abstraction layer
│       ├── manager.py       # LLMManager — main service class
│       ├── cache.py         # Response caching with diskcache
│       ├── models.py        # Response types (Live/Cached/Failed)
│       ├── rate_limiter.py  # RPM limiting, adaptive throttling, circuit breaker
│       ├── retry.py         # Exponential backoff with error classification
│       ├── model_registry.py # PydanticAI model factory
│       └── schemas/         # Pydantic models for structured output
│           ├── government_docs.py  # GovernmentDocumentExtraction schema
│           └── __init__.py
├── common/                  # Shared utilities
│   ├── document_reader.py   # PDF/DOCX text extraction
│   ├── chunking.py          # Text chunking utilities
│   ├── cache_utils.py       # General caching helpers
│   ├── json_utils.py        # JSON processing helpers
│   ├── logging.py           # Structured logging setup
│   ├── exceptions.py        # Custom exceptions (CircuitOpenError, etc.)
│   └── ...
├── prompts/                 # System and user prompts for LLM
│   ├── national-prj_system.md  # Russian: expert analyst system prompt
│   └── national-prj_user.md    # Russian: extraction instructions
├── input-docs/              # Sample input documents (PDF, DOCX)
├── output/                  # Generated JSON output files
├── test_docs/               # Test documents for Yandex AI Studio
├── main.py                  # CLI entry point for document extraction
├── json-schema.json         # JSON Schema for GovernmentDocumentExtraction
├── pyproject.toml           # uv project configuration
├── .env.example             # Environment variable template
├── assistant-yandex.py      # Marimo notebook: Yandex AI Studio assistant demo
└── elis_notebook.py         # Marimo notebook: LLM + Neo4j exploration
```

## Building and Running

### Prerequisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Environment Setup

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:
- `LLM_DEFAULT_PROVIDER`: Provider name (default: `openrouter`)
- `LLM_MODEL`: Model identifier (default: `qwen/qwen3-235b-a22b-thinking-2507`)
- `OPENROUTER_API_KEY`: Your API key
- `LLM_TEMPERATURE`: Sampling temperature (default: `0.7`)
- `LLM_RATE_LIMIT_RPM`: Rate limit (default: `60`)

### Running the CLI

```bash
# Run with example prompt (no input file)
uv run main.py

# Process a specific document
uv run main.py --input-doc input-docs/НП\ Семья.docx

# Process and export to JSON (auto-generates output path)
uv run main.py -i input-docs/НП\ Семья.docx --output-json

# Process with custom output file
uv run main.py -i input-docs/doc.pdf -o output/result.json

# Use custom prompts from files
uv run main.py -i doc.txt --system-prompt prompts/national-prj_system.md --user-prompt prompts/national-prj_user.md

# Verbose output with caching disabled
uv run main.py -i doc.txt -v --no-cache

# Pure JSON output (no _export_info metadata)
uv run main.py -i doc.txt --output-json --pure-json
```

### Running Marimo Notebooks

```bash
# Start marimo editor
uv run marimo edit elis_notebook.py

# Run notebook in run mode
uv run marimo run elis_notebook.py

# Tutorial
uv run marimo tutorial intro
```

## Key Components

### LLMManager (`adapters/llm/manager.py`)

The core service class that:
- Loads configuration from `.env` (supports multiple providers)
- Manages response caching (SHA256-based cache keys)
- Implements rate limiting with adaptive throttling and circuit breaker
- Uses pydantic-ai `Agent` for structured output with Pydantic models
- Returns typed responses: `LiveResponse`, `CachedResponse`, or `FailedResponse`

```python
from adapters.llm import LLMManager
from adapters.llm.schemas import GovernmentDocumentExtraction

llm = LLMManager(default_response_model=GovernmentDocumentExtraction)
result = llm.query("Extract goals from document...")
print(result.national_goals)  # Type-safe access
```

### GovernmentDocumentExtraction Schema (`adapters/llm/schemas/government_docs.py`)

Pydantic models matching `json-schema.json`:
- `DocumentInfo` — document metadata (title, type, date, number)
- `NationalGoal` — НЦР with indicators and factors
- `NationalProject` — НП with federal projects, curators, target groups
- `FederalProject` — ФП with indicators
- `StateProgram` — ГП
- `Indicator` — metrics with target values by year (2024–2036)
- `Relationship` — ArchiMate-style relationships (Aggregation, Composition, etc.)

### Document Reader (`common/document_reader.py`)

Extracts text from PDF and DOCX files for LLM processing.

### Prompts (`prompts/`)

- `national-prj_system.md` — Russian-language system prompt defining the LLM as an expert analyst of Russian government documents
- `national-prj_user.md` — User prompt with extraction instructions

## Development Conventions

### Code Style
- **Linter:** ruff (configured in `common/ruff.toml`)
- **Python version:** 3.13+ (specified in `.python-version`)
- **Type hints:** Used throughout, with `from __future__ import annotations`

### Architecture Patterns
- **Adapter pattern:** LLM providers abstracted through `LLMManager`
- **Factory pattern:** `get_pydantic_ai_model()` creates provider-specific models
- **Caching:** diskcache-based with SHA256 keys (prompt + model + schema)
- **Resilience:** Exponential backoff, rate limiting, circuit breaker

### Error Handling
- Custom exceptions in `common/exceptions.py` and `adapters/llm/models.py`
- `FailedResponse` type for graceful error propagation
- Circuit breaker pattern for provider outages

### Configuration
- Environment-driven via `.env` (python-dotenv)
- Two modes: Simple (`LLM_DEFAULT_PROVIDER` + `LLM_MODEL`) and Benchmark (multiple predefined models)
- No hardcoded secrets; all API keys from environment

## Key Files Reference

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point with argparse for document extraction |
| `json-schema.json` | JSON Schema definition for extraction output |
| `adapters/llm/schemas/government_docs.py` | Pydantic models matching the JSON schema |
| `adapters/llm/manager.py` | LLMManager — unified LLM interface |
| `prompts/national-prj_system.md` | System prompt (Russian) for government doc analysis |
| `.env.example` | Environment variable template with all options |
| `pyproject.toml` | uv project configuration and dependencies |

## Notes

- All extracted text and identifiers preserve original Russian terminology
- Output JSON includes optional `_export_info` metadata (timestamp, schema version, source)
- Cache can be disabled per-request or globally via `LLM_NOCACHE=true`
- The project is designed for batch processing of government documents to build structured knowledge graphs
