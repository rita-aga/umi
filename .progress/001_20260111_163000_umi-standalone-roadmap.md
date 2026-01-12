# Umi Standalone Roadmap

## Status: Active

## Overview

Umi is a memory library for AI agents with hybrid Rust+Python architecture. The core features are built and tested. This roadmap covers productionization and integration work.

---

## What's Already Built

### Python Layer (`umi/`)
- **Memory class** - Main interface with `remember()` and `recall()`
- **EntityExtractor** - LLM-powered entity extraction from text
- **DualRetriever** - Fast vector + LLM query rewriting with RRF merging
- **EvolutionTracker** - Detects updates, contradictions, extensions between memories
- **SimLLMProvider** - Deterministic simulation for testing (no API calls)
- **SimStorage** - In-memory storage for testing
- **FaultConfig** - Fault injection for reliability testing
- **145 tests passing**

### Rust Core (`umi-core/`)
- **Memory tiers** - CoreMemory (32KB), WorkingMemory (1MB), ArchivalMemory
- **Storage backends** - SimStorageBackend with DST support
- **DST infrastructure** - Deterministic Simulation Testing
- **232 tests passing**

### PyO3 Bindings (`umi-py/`)
- **Compiles** but not wired to Python layer yet

---

## Phase 1: PyPI Publication

**Goal**: Make Umi installable via `pip install umi-memory`

### Tasks
- [ ] Choose package name (check availability: `umi-memory`, `umi-agent-memory`)
- [ ] Update pyproject.toml with correct metadata
- [ ] Add classifiers, keywords, project URLs
- [ ] Create PyPI account and API token
- [ ] Set up GitHub Actions for automatic publishing on release
- [ ] Create v0.1.0 release

### Files to Modify
- `pyproject.toml` - Package metadata, URLs
- `.github/workflows/publish.yml` - New workflow for PyPI publishing

---

## Phase 2: Real Storage Backends

**Goal**: Production-ready storage beyond SimStorage

### Phase 2a: PostgreSQL Backend
- [ ] Create `umi/storage/postgres.py`
- [ ] Implement async connection pool (asyncpg)
- [ ] Schema: entities table with all fields including temporal
- [ ] Full-text search using tsvector
- [ ] Migration support

### Phase 2b: Vector Storage (Qdrant)
- [ ] Create `umi/storage/vectors.py`
- [ ] Implement QdrantStorage class
- [ ] Hybrid search (vector + metadata filters)
- [ ] Batch operations for efficiency

### Phase 2c: Unified Storage Interface
- [ ] Create `umi/storage/base.py` with Storage protocol
- [ ] Factory function: `create_storage(backend="postgres|qdrant|sim")`
- [ ] Connection string configuration

### New Dependencies (optional extras)
```toml
[project.optional-dependencies]
postgres = ["asyncpg>=0.29"]
qdrant = ["qdrant-client>=1.7"]
```

---

## Phase 3: Wire PyO3 Bindings

**Goal**: Use Rust core from Python for performance-critical paths

### Tasks
- [ ] Expose CoreMemory to Python via PyO3
- [ ] Expose WorkingMemory and ArchivalMemory
- [ ] Create Python wrapper that uses Rust when available, falls back to pure Python
- [ ] Benchmark Rust vs Python performance
- [ ] Update Memory class to optionally use Rust backend

### Architecture
```
Memory (Python)
    ├── uses SimStorage (Python) for testing
    ├── uses PostgresStorage (Python) for production
    └── uses RustMemory (PyO3) for high-performance local storage
```

### Files to Create/Modify
- `umi-py/src/lib.rs` - PyO3 module definitions
- `umi/rust_backend.py` - Python wrapper for Rust bindings
- `umi/memory.py` - Add `backend="rust"` option

---

## Phase 4: Advanced Features

### 4a: Embedding Support
- [ ] Add embedding generation to EntityExtractor
- [ ] Support multiple embedding providers (Voyage, OpenAI, local)
- [ ] Store embeddings in Entity metadata or separate field

### 4b: Batch Operations
- [ ] `remember_batch()` for multiple texts
- [ ] `recall_batch()` for multiple queries
- [ ] Parallel processing with asyncio.gather

### 4c: Memory Consolidation
- [ ] Periodic consolidation of related memories
- [ ] Summary generation for memory clusters
- [ ] Importance decay over time

---

## Alternative Path: Full Rust Umi

Based on feasibility analysis (see `docs/findings-summary.md`), a full Rust implementation is viable and may be preferred for agent-native ecosystems.

### Why Full Rust?

| Factor | Benefit |
|--------|---------|
| Single language | No PyO3 complexity |
| DST native | Deterministic testing built-in |
| Deployment | Single binary, no Python runtime |
| Agent development | Compiler feedback accelerates coding agents |

### Full Rust Roadmap

```
Phase R0: SimLLM Foundation ✅ COMPLETE
├── SimLLM with deterministic responses
├── Entity extraction routing
├── Query rewrite routing
├── Evolution detection routing
└── Fault injection support

Phase R1: Port Python Layer to Rust (IN PROGRESS)
├── [✅] LLM Provider Trait (ADR-013)
│   ├── LLMProvider trait (generic, async)
│   ├── SimLLMProvider (wraps SimLLM)
│   ├── AnthropicProvider (feature-gated)
│   └── OpenAIProvider (feature-gated)
├── [✅] EntityExtractor (ADR-014)
│   ├── EntityExtractor<P: LLMProvider>
│   ├── ExtractionResult, ExtractedEntity, ExtractedRelation
│   ├── Graceful degradation (fallback on LLM failure)
│   └── Type-safe EntityType and RelationType enums
├── [ ] DualRetriever (string matching + HTTP)
├── [ ] EvolutionTracker (comparison + scoring)
└── [ ] Memory class (orchestrates all components)

Phase R2: Storage Backends in Rust
├── PostgreSQL (sqlx)
├── Qdrant (qdrant-client)
└── Unified StorageBackend trait

Phase R3: Publish to crates.io
├── umi-core crate
├── umi-memory crate (full API)
└── Documentation
```

### Progress Log

| Date | Phase | Work Done | Tests |
|------|-------|-----------|-------|
| 2026-01-11 | R0 | SimLLM with DST | 232 → 263 |
| 2026-01-11 | R1 | LLM Provider Trait | 263 → 263 |
| 2026-01-11 | R1 | EntityExtractor | 263 → 301 |

### When to Choose Full Rust

- Primary consumers are Rust agents (e.g., Letta-rs)
- Edge/embedded deployment needed
- DST is critical requirement
- Want single-language codebase

### When to Keep Hybrid

- Primary consumers are Python frameworks
- Need Python ML library integration
- Rapid prototyping priority

See `docs/letta-rust-feasibility.md` and `docs/isotopes-rust-replication.md` for integration analysis with Letta and aidnn-style systems.

---

## Implementation Order

```
Phase 1: PyPI ──────────────────────────────────────────────┐
                                                            │
Phase 2a: Postgres ─────┐                                   │
                        ├──→ Phase 2c: Unified Interface ───┼──→ Phase 4
Phase 2b: Qdrant ───────┘                                   │
                                                            │
Phase 3: PyO3 Bindings ─────────────────────────────────────┘
```

---

## Testing Strategy

Each phase must maintain:
- All existing tests pass
- New tests for new functionality
- Simulation mode always available (no external deps for testing)
- CI passes before merge

---

## Current CI Status

| Workflow | Status |
|----------|--------|
| Python CI | ✅ Passing (lint, type check, tests) |
| Rust CI | ✅ Passing (lint, test, build) |

---

## History

### Origin
Umi was extracted from RikaiOS (Personal Context Operating System) to be a standalone, reusable memory library. The concepts are inspired by:
- **memU** - Dual retrieval (fast + LLM semantic)
- **Mem0** - Entity extraction and evolution tracking
- **Supermemory** - Temporal metadata (document_time, event_time)

### Key Design Decisions
1. **Simulation-first** - Every component has a sim implementation for deterministic testing
2. **TigerStyle assertions** - Preconditions/postconditions with explicit limits
3. **Graceful degradation** - LLM failures return fallbacks, not crashes
4. **Hybrid architecture** - Rust for performance-critical paths, Python for flexibility

### Completed Work (from RikaiOS)
- ADR-007: SimLLMProvider design
- ADR-008: Memory class architecture
- ADR-009: Dual retrieval strategy
- ADR-010: Entity extraction
- ADR-011: Evolution tracking
