# AGENTS.md

## Commands

- **Run CLI**: `uv run main.py -i <file> [--output-json [auto|path.json]]`
- **Edit notebook**: `uv run marimo edit <notebook>.py`
- **Run notebook**: `uv run marimo run <notebook>.py`
- **Linting**: `ruff check .` (config at `common/ruff.toml`)

## Tech Stack

- Python 3.13+, package manager: uv (NOT pip or poetry)
- LLM: pydantic-ai (structured output via `GovernmentDocumentExtraction` schema in `adapters/llm/schemas/`)
- Notebooks: marimo (reactive, git-ignored workspace at `__marimo__/`)

## Project Structure

```
main.py           # CLI entry point
adapters/llm/     # LLM abstraction (manager, cache, rate limiting, retry)
adapters/llm/schemas/  # Pydantic response schemas
common/           # Document reading, chunking, logging, utils (base layer, no internal imports)
prompts/          # Russian-language LLM prompts (system + user pairs)
input-docs/       # Source documents
output/           # JSON results
json-schema.json  # GovernmentDocumentExtraction schema
```

## Key Conventions

- Config via `.env` (copy from `.env.example`)
- Provider: `LLM_DEFAULT_PROVIDER` + `LLM_MODEL`; API key: provider-specific env var (e.g., `OPENROUTER_API_KEY`)
- Default response model: `GovernmentDocumentExtraction` (structured JSON)
- Caching enabled by default; SHA256 keys; disable with `--no-cache`
- `--output-json` without a path auto-generates `output/<stem>_<timestamp>.json`; `--pure-json` omits `_export_info` metadata

## What to Skip

- No test directory, no pre-commit hooks, no Docker config