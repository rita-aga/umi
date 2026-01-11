# Umi Vision

## What Umi Is

A memory library for AI agents that handles:
1. Extracting structured entities from conversation text
2. Retrieving relevant memories with smart query expansion
3. Tracking when information changes or contradicts previous memories

## Core Principles

### Simulation-First

Every component has a simulation mode (`seed=N`) that:
- Makes zero external API calls
- Produces deterministic results
- Enables reliable, reproducible testing

This is non-negotiable. No new component ships without a simulation implementation.

### Graceful Degradation

LLM calls fail. The library should:
- Return sensible fallbacks (empty lists, None)
- Never crash due to LLM timeouts or errors
- Work in degraded mode rather than not at all

### Simple API

Users should be able to:
```python
memory = Memory(seed=42)  # or provider="anthropic"
await memory.remember("Alice works at Acme")
results = await memory.recall("Who works at Acme?")
```

That's it. The complexity lives inside.

## What's Built

- [x] Entity extraction from text
- [x] Dual retrieval (fast + LLM-expanded search)
- [x] Evolution tracking (update/extend/contradict detection)
- [x] Temporal metadata (document_time vs event_time)
- [x] SimLLMProvider for deterministic testing
- [x] SimStorage for in-memory storage
- [x] FaultConfig for failure injection testing

## What's Not Built Yet

- [ ] Persistent storage (Postgres backend)
- [ ] Vector search (Qdrant/pgvector backend)
- [ ] PyPI package publishing
- [ ] Rust core integration (umi-core via PyO3)

## Constraints

- Python 3.10+ only
- No required dependencies (LLM providers are optional extras)
- Must work without any API keys (simulation mode)
- Tests must pass without network access
