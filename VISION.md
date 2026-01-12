# Umi Vision

## What Umi Is

**Umi** is an open-source memory library for AI agents with entity extraction, dual retrieval, and evolution tracking.

It provides infrastructure for building agents that:

- **Extract structured knowledge** from unstructured text via LLM-powered entity extraction
- **Retrieve intelligently** using dual-mode search (fast vector + LLM semantic expansion)
- **Track evolution** by detecting when memories update, extend, or contradict each other
- **Test deterministically** with simulation testing for reproducible results

## Core Principles

### 1. Simulation-First (Mandatory)

Every component MUST have a simulation implementation:

```rust
// Rust - Deterministic simulation
use umi_memory::{Memory, SimLLMProvider, SimStorageBackend, SimConfig};

let config = SimConfig::with_seed(42);
let llm = SimLLMProvider::new();
let storage = SimStorageBackend::new();
```

```python
# Python - Deterministic simulation
from umi import Memory

memory = Memory(seed=42)  # Same seed = same results
```

**Why?** Same seed = same results = reproducible tests and bugs.

### 2. Graceful Degradation

LLM calls can fail. Components should:
- Return sensible fallbacks (empty lists, None)
- Never crash due to LLM timeouts or errors
- Log errors but continue operation
- Work in degraded mode rather than not at all

### 3. Simple API

Users should be able to:
```python
memory = Memory(seed=42)  # or provider="anthropic"
await memory.remember("Alice works at Acme")
results = await memory.recall("Who works at Acme?")
```

### 4. TigerStyle Safety

Following TigerBeetle's engineering principles:
- Explicit constants with units: `MEMORY_SIZE_BYTES_MAX`, `TTL_SECONDS_DEFAULT`
- Debug assertions for invariants
- No silent truncation or implicit conversions
- Deterministic simulation testing (DST)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Python: pip install umi                         │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │   Memory     │  │EntityExtract │  │  DualRetriever           │ │
│  │  (Main API)  │  │  (LLM-based) │  │  (Fast + LLM semantic)   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘ │
│         │                 │                      │                 │
│  ┌──────┴─────────────────┴──────────────────────┴───────────────┐ │
│  │              LLMProvider (Sim | Anthropic | OpenAI)            │ │
│  └────────────────────────────────┬────────────────────────────────┘ │
└───────────────────────────────────┼──────────────────────────────────┘
                                    │ PyO3 (planned)
┌───────────────────────────────────┴──────────────────────────────────┐
│                     Rust: umi-memory                                 │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ CoreMemory   │  │WorkingMemory │  │  ArchivalMemory          │  │
│  │  (32KB ctx)  │  │  (1MB TTL)   │  │  (Unlimited vector)      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                 │                      │                  │
│  ┌──────┴─────────────────┴──────────────────────┴───────────────┐  │
│  │         StorageBackend (Sim | Postgres | Lance)               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Memory Hierarchy

| Tier | Purpose | Size | Persistence | Access Pattern |
|------|---------|------|-------------|----------------|
| **Core** | Always in LLM context | ~32KB | Per-session | Read/Write via Memory |
| **Working** | Session KV store | ~1MB | TTL-based | Fast key-value ops |
| **Archival** | Long-term semantic search | Unlimited | Persistent | Vector similarity |

---

## Components

| Component | Description | Status | Lines |
|-----------|-------------|--------|-------|
| `umi-memory` | Rust core (memory tiers, DST) | Complete | ~3,500 |
| `umi-py` | PyO3 bindings | Planned | - |
| EntityExtractor | LLM-powered entity extraction | Complete | ~400 |
| DualRetriever | Fast + LLM semantic search | Complete | ~350 |
| EvolutionTracker | Memory relationship detection | Complete | ~300 |
| SimLLMProvider | Deterministic LLM simulation | Complete | ~200 |
| LanceVectorBackend | Production vector storage | Complete | ~600 |
| PostgresVectorBackend | Postgres-based storage | Complete | ~400 |

---

## What Umi Is NOT

### Not a Complete Agent Framework
Umi provides memory, not agent orchestration. Use LangChain, CrewAI, or similar for agent workflows.

### Not a Vector Database
Umi's archival memory is for agent state, not general-purpose vector search. Use Pinecone, Weaviate, Qdrant for that.

### Not a Prompt Framework
Umi doesn't provide prompt templates, chains, or RAG pipelines. Use DSPy, Guidance, or similar for that.

### Not a Model Router
Umi doesn't route between LLM providers or manage API keys. Use LiteLLM, OpenRouter for that.

---

## Use Cases

### 1. Agent Memory System
Build agents that remember context across sessions:
```python
# Session 1
await memory.remember("User prefers dark mode")
# Session 2 (days later)
prefs = await memory.recall("user preferences")
```

### 2. Evolving Knowledge Base
Track how information changes over time:
```python
await memory.remember("Alice works at Acme Corp")
# Later...
await memory.remember("Alice now works at Initech")
# Evolution tracker detects UPDATE relationship
```

### 3. Semantic Entity Search
Find structured entities across conversations:
```python
await memory.remember("Met Bob at the AI conference")
await memory.remember("Discussed transformers with Carol")
# Later...
entities = await memory.recall("who did I meet at conferences?")
```

### 4. Deterministic Testing
Test AI agents without API calls:
```python
# Same seed = same results = reproducible tests
memory = Memory(seed=42)
result = await memory.remember("test input")
assert len(result) == 3  # Always true with seed 42
```

---

## Implementation Status

### Complete

| Feature | Status | Tests | Notes |
|---------|--------|-------|-------|
| Memory Tiers | ✅ Complete | 85 | Core/Working/Archival |
| DST Framework | ✅ Complete | 62 | SimConfig, DeterministicRng |
| Entity Extraction | ✅ Complete | 38 | LLM-powered, fallback support |
| Dual Retrieval | ✅ Complete | 42 | Fast vector + LLM expansion |
| Evolution Tracking | ✅ Complete | 31 | Update/Extend/Contradict |
| Sim Providers | ✅ Complete | 45 | SimLLM, SimStorage, SimVector |
| Lance Backend | ✅ Complete | 28 | Production vector storage |
| Postgres Backend | ✅ Complete | 24 | Sim-first implementation |
| LLM Providers | ✅ Complete | 22 | Anthropic, OpenAI |
| Temporal Metadata | ✅ Complete | 18 | document_time vs event_time |

### Planned

| Feature | Priority | Notes |
|---------|----------|-------|
| PyO3 Bindings | P0 | Connect Python to Rust core |
| Persistent Storage | P0 | Real Postgres backend |
| PyPI Publishing | P1 | Public release |
| Crates.io Publishing | P1 | Rust crate distribution |
| Qdrant Backend | P2 | Alternative vector DB |
| Local Embeddings | P2 | fastembed integration |

---

## Performance Targets

| Metric | Target | Current | Notes |
|--------|--------|---------|-------|
| Entity extraction | <2s | ~1.5s | With real LLM (Claude) |
| Vector search (10K) | <100ms | ~50ms | In-memory SimVector |
| Vector search (1M) | <500ms | ~300ms | Lance backend |
| Memory write | <10ms | ~5ms | Core/Working memory |
| DST test suite | <30s | ~25s | 232 Rust + 145 Python tests |
| Dual retrieval | <3s | ~2s | Fast + LLM combined |
| Evolution tracking | <1s | ~800ms | Per memory relationship |

---

## Test Coverage

| Category | Test Count | Status | Coverage |
|----------|------------|--------|----------|
| **Rust Core** | **232** | ✅ Passing | **~85%** |
| Memory Tiers | 85 | ✅ Passing | High |
| DST Framework | 62 | ✅ Passing | High |
| Storage Backends | 52 | ✅ Passing | High |
| Entity Types | 33 | ✅ Passing | High |
| **Python Layer** | **145** | ✅ Passing | **~78%** |
| Entity Extraction | 38 | ✅ Passing | High |
| Dual Retrieval | 42 | ✅ Passing | High |
| Evolution Tracking | 31 | ✅ Passing | High |
| LLM Providers | 22 | ✅ Passing | Medium |
| Integration | 12 | ✅ Passing | Medium |
| **Total** | **377** | ✅ All Passing | **~82%** |

### DST Test Coverage

Critical paths with deterministic simulation testing:
- [x] Entity extraction with SimLLM
- [x] Dual retrieval with SimVector
- [x] Evolution tracking with SimStorage
- [x] Memory tier operations
- [x] Storage backend fault injection
- [x] LLM provider failures
- [x] Vector search edge cases

---

## Roadmap

See [CLAUDE.md](./CLAUDE.md) for development guidelines.

### Phase 1: Production Foundation (Current)
- [x] Complete Rust memory tiers
- [x] Complete Python LLM integration
- [x] Lance vector backend
- [x] Postgres backend (sim-first)
- [ ] **PyO3 bindings** (P0 - Critical)
- [ ] **Real Postgres persistence** (P0 - Critical)
- [ ] **PyPI package** (P1)

### Phase 2: Enhanced Backends
- [ ] Qdrant vector backend (P2)
- [ ] Local embeddings via fastembed (P2)
- [ ] Redis backend for Working memory (P2)
- [ ] S3 backend for Archival (P3)

### Phase 3: Advanced Features
- [ ] Multi-agent shared memory
- [ ] Memory compression strategies
- [ ] Incremental entity updates
- [ ] Advanced evolution detection

### Phase 4: Distribution
- [ ] Crates.io publishing
- [ ] Documentation site
- [ ] Example projects
- [ ] Community contributions

---

## Engineering Principles

Umi follows **TigerStyle** (Safety > Performance > DX):

- Explicit constants with units: `MEMORY_BYTES_MAX`, `TTL_SECONDS_DEFAULT`
- 2+ assertions per non-trivial function
- No silent truncation or implicit conversions
- Deterministic simulation testing for critical paths
- Every component has a simulation implementation

See [CLAUDE.md](./CLAUDE.md) for detailed guidelines.

---

## Origin & Inspiration

Umi was extracted from **RikaiOS** (Personal Context Operating System) to be a standalone, reusable memory library.

### Inspiration Sources

- **[memU](https://github.com/mem-u/memu)** - Dual retrieval architecture (fast vector + LLM semantic search)
- **[Mem0](https://github.com/mem0ai/mem0)** - Entity extraction and evolution tracking
- **[Supermemory](https://github.com/supermemoryai/supermemory)** - Temporal metadata (when something was said vs when it happened)
- **[TigerBeetle](https://github.com/tigerbeetle/tigerbeetle)** - TigerStyle engineering (assertions, explicit limits)
- **[FoundationDB](https://www.foundationdb.org/)** - Deterministic simulation testing

### Why Standalone?

Memory is the most reusable component of a context system. By making it standalone:
- Any AI agent framework can use it (not just RikaiOS)
- Clearer boundaries and responsibilities
- Easier to test, maintain, and contribute to
- Can evolve independently of RikaiOS

---

## Architecture Decisions

Key decisions documented in `docs/adr/`:

- **[ADR-013](./docs/adr/013-llm-provider-trait.md)** - LLM abstraction layer
- **[ADR-014](./docs/adr/014-entity-extractor.md)** - Entity extraction strategy
- **[ADR-015](./docs/adr/015-dual-retriever.md)** - Dual retrieval with RRF merging
- **[ADR-016](./docs/adr/016-evolution-tracker.md)** - Memory relationship detection
- **[ADR-017](./docs/adr/017-memory-class.md)** - Main Memory API design
- **[ADR-018](./docs/adr/018-lance-storage-backend.md)** - LanceDB integration

---

## Contributing

See [CLAUDE.md](./CLAUDE.md) for development guidelines.

Key principles:
1. **Simulation-first** - Every component has a Sim implementation
2. **Tests must pass** - All 377 tests before commit
3. **TigerStyle** - Explicit limits, assertions, no silent failures
4. **No stubs** - Complete implementations or don't merge
5. **Graceful degradation** - Handle LLM failures elegantly

---

## License

MIT - See [LICENSE](./LICENSE) for details
