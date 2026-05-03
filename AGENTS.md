# AGENTS.md

## Commands

- **Run CLI**: `uv run main.py -i <file> --output-json`
- **Edit notebook**: `uv run marimo edit <notebook>.py`
- **Run notebook**: `uv run marimo run <notebook>.py`
- **Linting**: `ruff check .` (config at `common/ruff.toml`)

## Tech Stack

- Python 3.13+
- Package manager: uv (NOT pip or poetry)
- LLM: pydantic-ai (structured output)
- Notebooks: marimo (reactive, git-ignored workspace at `__marimo__/`)

## Project Structure

```
main.py           # CLI entry point
adapters/llm/     # LLM abstraction (manager, cache, rate limiting)
common/           # Document reading, chunking, logging, utils
prompts/           # Russian-language LLM prompts
input-docs/       # Source documents
output/           # JSON results
```

## Key Conventions

- All config via `.env` (copy from `.env.example`)
- Provider set by `LLM_DEFAULT_PROVIDER` + `LLM_MODEL`
- Default response model: `GovernmentDocumentExtraction` (Pydantic schema in `adapters/llm/schemas/`)
- Response caching enabled by default (SHA256 keys)
- Russian government documents → structured JSON matching `json-schema.json`

## What to Skip

- No test directory exists
- No pre-commit hooks
- No Docker or deployment config