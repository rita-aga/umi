# Umi

[![Crates.io](https://img.shields.io/crates/v/umi-memory.svg)](https://crates.io/crates/umi-memory)
[![Documentation](https://docs.rs/umi-memory/badge.svg)](https://docs.rs/umi-memory)
[![Rust CI](https://github.com/rita-aga/umi/actions/workflows/ci-rust.yml/badge.svg)](https://github.com/rita-aga/umi/actions/workflows/ci-rust.yml)
[![Python CI](https://github.com/rita-aga/umi/actions/workflows/ci.yml/badge.svg)](https://github.com/rita-aga/umi/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Memory for AI agents - entity extraction, dual retrieval, and evolution tracking.

## Overview

Umi provides infrastructure for building AI agents that:

- **Extract structured knowledge** from unstructured text via LLM-powered entity extraction
- **Retrieve intelligently** using dual-mode search (fast vector + LLM semantic expansion)
- **Track evolution** by detecting when memories update, extend, or contradict each other
- **Test deterministically** with simulation testing for reproducible results

## Quick Start

### Python

**Status**: 🔶 Experimental - Basic functionality works, but incomplete. See [PYTHON.md](PYTHON.md) for full details.

```bash
# Install from source (not yet on PyPI)
cd umi-py
pip install maturin
maturin develop  # Build and install locally
```

```python
from umi import Memory

# Simulation mode (deterministic, no API calls)
memory = Memory.sim(42)

# Remember information
memory.remember("Alice is a software engineer at Acme Corp")

# Recall memories
results = memory.recall("Who works at Acme?")
for entity in results:
    print(f"Found: {entity.name} - {entity.content}")
```

**What works**: Basic `Memory.sim()`, `remember()`, `recall()`
**What's missing**: Options classes, real LLM providers, type stubs, async support

See [PYTHON.md](PYTHON.md) for roadmap and contributing guide.

### Rust

```bash
cargo add umi-memory
```

```rust
use umi_memory::umi::{Memory, RememberOptions, RecallOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simulation mode (deterministic, no API calls)
    let mut memory = Memory::sim(42);

    // Remember information
    let result = memory
        .remember("Alice works at Acme Corp", RememberOptions::default())
        .await?;
    println!("Stored {} entities", result.entity_count());

    // Recall memories
    let results = memory
        .recall("Alice", RecallOptions::default())
        .await?;

    for entity in results {
        println!("  - {}: {}", entity.name, entity.content);
    }

    Ok(())
}
```

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

## Features

### Memory Hierarchy

| Tier | Purpose | Size | Persistence | Access Pattern |
|------|---------|------|-------------|----------------|
| **Core** | Always in LLM context | ~32KB | Per-session | Read/Write via Memory |
| **Working** | Session KV store | ~1MB | TTL-based | Fast key-value ops |
| **Archival** | Long-term semantic search | Unlimited | Persistent | Vector similarity |

### Component Status

| Component | Description | Status | Tests |
|-----------|-------------|--------|-------|
| **umi-memory** | Rust core (memory tiers, DST) | ✅ Complete | 470 |
| **umi-py** | PyO3 bindings | 🚧 Planned | - |
| Memory API | Main orchestrator with remember/recall | ✅ Complete | 12 |
| MemoryBuilder | Builder pattern for Memory construction | ✅ Complete | 11 |
| MemoryConfig | Global configuration system | ✅ Complete | 13 |
| EntityExtractor | LLM-powered entity extraction | ✅ Complete | 38 |
| DualRetriever | Fast + LLM semantic search | ✅ Complete | 42 |
| EvolutionTracker | Memory relationship detection | ✅ Complete | 31 |
| SimLLMProvider | Deterministic LLM simulation | ✅ Complete | 22 |
| LanceVectorBackend | Production vector storage | ✅ Complete | 28 |
| PostgresVectorBackend | Postgres-based storage | ✅ Complete | 24 |

### LLM Providers

```python
# Simulation (deterministic, no API)
memory = Memory(seed=42)

# Anthropic Claude
memory = Memory(provider="anthropic")

# OpenAI GPT
memory = Memory(provider="openai")
```

### Storage Backends

```rust
use umi_memory::umi::{Memory, MemoryBuilder};
use umi_memory::embedding::SimEmbeddingProvider;
use umi_memory::llm::SimLLMProvider;
use umi_memory::storage::{SimStorageBackend, SimVectorBackend};

// Quick start with simulation (all components)
let mut memory = Memory::sim(42);

// Or use builder for explicit configuration
let memory = Memory::builder()
    .with_llm(SimLLMProvider::with_seed(42))
    .with_embedder(SimEmbeddingProvider::with_seed(42))
    .with_vector(SimVectorBackend::new(42))
    .with_storage(SimStorageBackend::new(/* config */))
    .build();

// Production with LanceDB (requires 'lance' feature)
#[cfg(feature = "lance")]
{
    use umi_memory::storage::LanceVectorBackend;

    let lance = LanceVectorBackend::connect("./lance_db").await?;
    let memory = Memory::builder()
        .with_llm(/* your LLM provider */)
        .with_embedder(/* your embedding provider */)
        .with_vector(lance)
        .with_storage(/* your storage backend */)
        .build();
}
```

## Production Mode

```python
import os
from umi import Memory

# Set API keys
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
# or
os.environ["OPENAI_API_KEY"] = "sk-..."

# Create memory with real LLM
memory = Memory(provider="anthropic")

# Use normally - LLM calls are made
entities = await memory.remember("Important information")
results = await memory.recall("search query")
```

## Configuration

### Memory Configuration (Rust)

```rust
use umi_memory::umi::{Memory, MemoryConfig};
use std::time::Duration;

// Default configuration
let mut memory = Memory::sim(42);

// Custom configuration
let config = MemoryConfig::default()
    .with_core_memory_bytes(64 * 1024)  // 64KB core memory
    .with_recall_limit(20)              // Return up to 20 results
    .without_embeddings();              // Disable embedding generation

let mut memory = Memory::sim_with_config(42, config);

// Configuration options:
// - core_memory_bytes: Size of core memory (default: 32KB)
// - working_memory_bytes: Size of working memory (default: 1MB)
// - working_memory_ttl: TTL for working memory (default: 1 hour)
// - generate_embeddings: Enable/disable embeddings (default: true)
// - embedding_batch_size: Batch size for embedding generation (default: 100)
// - default_recall_limit: Default limit for recall (default: 10)
// - semantic_search_enabled: Enable semantic search (default: true)
// - query_expansion_enabled: Enable query expansion (default: true)
```

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key | - | For Claude |
| `OPENAI_API_KEY` | OpenAI API key | - | For GPT |
| `UMI_LOG_LEVEL` | Logging level | `info` | No |
| `DST_SEED` | Deterministic test seed | Random | No |

### Python Package Extras

| Extra | Purpose | Install |
|-------|---------|---------|
| `anthropic` | Anthropic Claude support | `pip install "umi[anthropic]"` |
| `openai` | OpenAI GPT support | `pip install "umi[openai]"` |
| `dev` | Development tools | `pip install "umi[dev]"` |
| `all` | All extras | `pip install "umi[all]"` |

### Rust Crate Features

| Feature | Description | Enable |
|---------|-------------|--------|
| `lance` | LanceDB vector backend | `features = ["lance"]` |
| `postgres` | Postgres storage backend | `features = ["postgres"]` |
| `anthropic` | Anthropic LLM provider | `features = ["anthropic"]` |
| `openai` | OpenAI LLM provider | `features = ["openai"]` |

## Testing

### Deterministic Simulation Testing (DST)

Umi uses deterministic simulation for reliable, reproducible tests:

```python
# Python - Same seed = same results
memory = Memory(seed=42)
result = await memory.remember("test input")
assert len(result) == 3  # Always true with seed 42
```

```rust
// Rust - Deterministic simulation
use umi_memory::SimConfig;

let config = SimConfig::with_seed(42);
// All operations deterministic with this seed
```

### Running Tests

```bash
# Python tests (145 tests)
pip install -e ".[dev]"
pytest -v

# Rust tests (470 tests)
cargo test -p umi-memory --features lance

# All tests with specific DST seed
DST_SEED=12345 cargo test --all-features

# Rust with coverage
cargo tarpaulin --all-features --out Html
```

### Fault Injection

```python
from umi import Memory, FaultConfig, FaultType

# Test with simulated failures
memory = Memory(
    seed=42,
    fault_config=FaultConfig(
        fault_type=FaultType.STORAGE_WRITE_FAIL,
        probability=0.1  # 10% chance of failure
    )
)
```

### Test Coverage

| Category | Tests | Status | Coverage |
|----------|-------|--------|----------|
| **Rust Core** | 470 | ✅ Passing | ~85% |
| **Python Layer** | 145 | ✅ Passing | ~78% |
| **Total** | 615 | ✅ All Passing | ~82% |

## Performance

| Metric | Target | Current |
|--------|--------|---------|
| Entity extraction | <2s | ~1.5s |
| Vector search (10K) | <100ms | ~50ms |
| Vector search (1M) | <500ms | ~300ms |
| Memory write | <10ms | ~5ms |
| Dual retrieval | <3s | ~2s |

## Development

### Python Development

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run linter
ruff check .
ruff format .

# Run type checker
mypy umi/

# Run tests
pytest -v
```

### Rust Development

```bash
# Build
cargo build --all-features

# Run tests
cargo test --all-features

# Run linter
cargo clippy --all-features -- -D warnings

# Format
cargo fmt --check

# Run benchmarks
cargo bench -p umi-memory
```

### Pre-commit Checklist

Before every commit:

```bash
# Python
ruff check . && ruff format . && mypy umi/ && pytest

# Rust
cargo fmt && cargo clippy --all-features -- -D warnings && cargo test --all-features
```

## Project Structure

```
umi/
├── umi-memory/             # Rust core library
│   ├── src/
│   │   ├── lib.rs          # Public exports
│   │   ├── umi/            # Memory orchestrator
│   │   ├── extraction/     # EntityExtractor
│   │   ├── retrieval/      # DualRetriever
│   │   ├── evolution/      # EvolutionTracker
│   │   ├── storage/        # Sim + Lance backends
│   │   ├── llm/            # LLM providers
│   │   ├── memory/         # Memory tiers
│   │   ├── dst/            # Deterministic simulation
│   │   └── constants.rs    # TigerStyle limits
│   └── benches/            # Benchmarks
│
├── umi-py/                 # PyO3 bindings (planned)
│   └── src/lib.rs
│
├── docs/                   # Documentation
│   ├── adr/                # Architecture Decision Records
│   └── kelpie-learnings.md # Project improvement analysis
│
├── .github/workflows/      # CI/CD
│   ├── ci.yml              # Python CI
│   └── ci-rust.yml         # Rust CI
│
├── VISION.md               # Project vision and roadmap
├── CLAUDE.md               # Development guidelines
├── README.md               # This file
├── Cargo.toml              # Rust workspace
└── pyproject.toml          # Python package
```

## Roadmap

See [VISION.md](./VISION.md) for detailed roadmap.

**Next priorities:**

1. **PyO3 bindings** (P0) - Connect Python to Rust core
2. **Real Postgres backend** (P0) - Persistent storage
3. **PyPI publishing** (P1) - Public release
4. **Crates.io publishing** (P1) - Rust crate distribution

## Engineering Principles

Umi follows **TigerStyle** (Safety > Performance > DX):

- Explicit constants with units: `MEMORY_BYTES_MAX`, `TTL_SECONDS_DEFAULT`
- 2+ assertions per non-trivial function
- No silent truncation or implicit conversions
- Deterministic simulation testing for critical paths
- Every component has a simulation implementation

See [CLAUDE.md](./CLAUDE.md) for detailed development guidelines.

## Inspiration

- **[memU](https://github.com/mem-u/memu)** - Dual retrieval architecture
- **[Mem0](https://github.com/mem0ai/mem0)** - Entity extraction and evolution tracking
- **[Supermemory](https://github.com/supermemoryai/supermemory)** - Temporal metadata
- **[TigerBeetle](https://github.com/tigerbeetle/tigerbeetle)** - TigerStyle engineering
- **[FoundationDB](https://www.foundationdb.org/)** - Deterministic simulation testing

## Contributing

See [CLAUDE.md](./CLAUDE.md) for development guidelines.

Key principles:
1. **Simulation-first** - Every component has a Sim implementation
2. **Tests must pass** - All 615 tests before commit
3. **TigerStyle** - Explicit limits, assertions, no silent failures
4. **No stubs** - Complete implementations or don't merge
5. **Graceful degradation** - Handle LLM failures elegantly

## License

MIT - See [LICENSE](./LICENSE) for details
