# Umi Vision

## What Umi Is

A memory library for AI agents with two layers:

1. **Rust core**: Memory tiers, storage backends, deterministic simulation
2. **Python layer**: LLM integration, entity extraction, smart retrieval

## Core Principles

### Simulation-First

Every component has a simulation mode that:
- Makes zero external API calls
- Produces deterministic results (same seed = same output)
- Enables reliable, reproducible testing

This applies to both Rust (`SimConfig`) and Python (`seed=N`).

### Graceful Degradation

LLM calls fail. The Python layer should:
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

## What's Built

### Rust (`umi-core/`)
- [x] DST framework (SimConfig, DeterministicRng, SimClock)
- [x] CoreMemory (32KB, bounded)
- [x] WorkingMemory (1MB, TTL-based eviction)
- [x] ArchivalMemory (unlimited, uses storage backend)
- [x] SimStorageBackend for testing
- [x] SimVectorBackend for vector search simulation
- [x] Entity with temporal metadata
- [x] EvolutionRelation types

### Python (`umi/`)
- [x] Entity extraction from text
- [x] Dual retrieval (fast + LLM-expanded search)
- [x] Evolution tracking (update/extend/contradict detection)
- [x] Temporal metadata (document_time vs event_time)
- [x] SimLLMProvider for deterministic testing
- [x] SimStorage for in-memory storage
- [x] FaultConfig for failure injection testing

## What's Not Built Yet

- [ ] PyO3 bindings connecting Python to Rust
- [ ] Persistent storage (Postgres backend)
- [ ] Vector search (Qdrant/pgvector backend)
- [ ] PyPI package publishing
- [ ] Crates.io publishing

## Constraints

- Rust: stable toolchain, no nightly features
- Python: 3.10+ only
- No required dependencies (LLM providers are optional extras)
- Must work without any API keys (simulation mode)
- Tests must pass without network access

## Test Counts

- Rust: 232 tests
- Python: 145 tests

## Origin & Inspiration

Umi was extracted from **RikaiOS** (Personal Context Operating System) to be a standalone, reusable memory library.

### Inspiration Sources
- **memU** - Dual retrieval architecture (fast vector + LLM semantic search)
- **Mem0** - Entity extraction and evolution tracking
- **Supermemory** - Temporal metadata (when something was said vs when it happened)
- **TigerBeetle's TigerStyle** - Assertion-based programming, explicit limits

### Why Standalone?
Memory is the most reusable component of a context system. By making it standalone:
- Any AI agent framework can use it (not just RikaiOS)
- Clearer boundaries and responsibilities
- Easier to test, maintain, and contribute to
- Can evolve independently of RikaiOS

## Architecture Decisions

Key decisions documented in the codebase:
- **ADR-007**: SimLLMProvider design (deterministic simulation)
- **ADR-008**: Memory class architecture (remember/recall API)
- **ADR-009**: Dual retrieval strategy (RRF merging)
- **ADR-010**: Entity extraction (confidence scoring, fallbacks)
- **ADR-011**: Evolution tracking (update/extend/contradict/derive)
