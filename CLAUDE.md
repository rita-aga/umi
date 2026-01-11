# CLAUDE.md

Instructions for Claude Code when working in the Umi repository.

## What Umi Is

Umi is a memory library for AI agents. It provides:
- **Entity extraction** from text using LLMs
- **Dual retrieval** (fast search + LLM query expansion)
- **Evolution tracking** (detecting updates/contradictions in memories)

## Development Philosophy

### Simulation-First (Mandatory)

Every component MUST have a simulation implementation that:
1. Takes a `seed` parameter for deterministic behavior
2. Makes no external API calls
3. Produces identical results for identical inputs

```python
# Good - simulation mode
memory = Memory(seed=42)  # Deterministic, no LLM calls

# Production mode (requires API keys)
memory = Memory(provider="anthropic")
```

**Why?** Reliable tests. Same seed = same results = reproducible bugs.

### TigerStyle Assertions

Every public function should have:
- **Preconditions**: Assert inputs are valid at function start
- **Postconditions**: Assert outputs are valid before return

```python
async def remember(self, text: str) -> list[Entity]:
    # Preconditions
    assert text, "text must not be empty"
    assert len(text) <= 100_000, "text too large"

    # ... implementation ...

    # Postconditions
    assert isinstance(result, list), "must return list"
    return result
```

### Graceful Degradation

LLM calls can fail. Components should:
- Catch `TimeoutError` and `RuntimeError`
- Return fallback values (empty list, None) instead of crashing
- Log failures but don't propagate them to callers

## Architecture Decision Records (ADRs)

Document significant design decisions in `docs/adr/`:
- `009-dual-retrieval.md` - Query rewriting and RRF merging
- `010-entity-extraction.md` - LLM-powered extraction
- `011-evolution-tracking.md` - Memory relationship detection

Create new ADRs for architectural changes. Format: `NNN-short-name.md`

## Build & Test Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"
# or with uv
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest umi/tests/test_memory.py

# Run with verbose output
pytest -v

# Linting
ruff check .
ruff check --fix .

# Type checking
mypy umi/
```

## Code Style

- Python 3.10+
- Async/await throughout
- Ruff for linting (line-length 100)
- Type hints on all public functions
- Docstrings with examples for public APIs

## Directory Structure

```
umi/
├── __init__.py          # Public API exports
├── memory.py            # Main Memory class
├── extraction.py        # EntityExtractor
├── retrieval.py         # DualRetriever
├── evolution.py         # EvolutionTracker
├── storage.py           # SimStorage (Entity dataclass)
├── faults.py            # FaultConfig for testing
├── providers/
│   ├── base.py          # LLMProvider protocol
│   ├── sim.py           # SimLLMProvider (testing)
│   ├── anthropic.py     # Anthropic Claude
│   └── openai.py        # OpenAI
└── tests/
    ├── test_memory.py
    ├── test_extraction.py
    ├── test_retrieval.py
    ├── test_evolution.py
    └── test_providers.py
```

## Current Limitations

- **Storage is in-memory only** - SimStorage doesn't persist
- **No vector search** - substring matching only
- **No real database backends** - Postgres/Qdrant planned

## Git Workflow

- Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`
- Run tests before pushing
- Update ADRs for architectural changes
