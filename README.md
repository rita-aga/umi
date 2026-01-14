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

**Status**: 🔶 Experimental - Low-level memory primitives only. See [PYTHON.md](PYTHON.md) for full details.

```bash
# Install from source (not yet on PyPI)
cd umi-py
pip install maturin
maturin develop  # Build and install locally
```

```python
import umi

# Core Memory (32KB, always in LLM context)
core = umi.CoreMemory()
core.set_block("system", "You are a helpful assistant.")
core.set_block("human", "User prefers concise responses.")
context = core.render()  # XML for LLM context

# Working Memory (1MB KV store with TTL)
working = umi.WorkingMemory()
working.set("session_id", b"abc123")
value = working.get("session_id")

# Entities for storage
entity = umi.Entity("person", "Alice", "Software engineer at Acme")
print(f"{entity.name}: {entity.content}")
```

**What works**: `CoreMemory`, `WorkingMemory`, `Entity`, `EvolutionRelation`
**What's missing**: High-level `Memory` class, real LLM providers, async support

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
│                   Python: pip install umi (PyO3 bindings)           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         Memory (thin wrapper around Rust Memory API)        │   │
│  └─────────────────────────┬───────────────────────────────────┘   │
└────────────────────────────┼─────────────────────────────────────────┘
                             │ PyO3 (exposing Rust to Python)
┌────────────────────────────┴─────────────────────────────────────────┐
│                          Rust: umi-memory                            │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Memory     │  │EntityExtract │  │  DualRetriever           │  │
│  │  (Main API)  │  │  (LLM-based) │  │  (Fast + LLM semantic)   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                 │                      │                  │
│  ┌──────┴─────────────────┴──────────────────────┴───────────────┐  │
│  │       EvolutionTracker (Memory relationship detection)        │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┴───────────────────────────────────┐  │
│  │           LLMProvider (Sim | Anthropic | OpenAI)             │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
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
| **umi-memory** | Rust core (memory tiers, DST) | ✅ Complete | ~813 |
| **umi-py** | PyO3 bindings (CoreMemory, WorkingMemory, Entity) | 🔶 Partial | - |
| Memory API | Main orchestrator with remember/recall | ✅ Complete | ✓ |
| MemoryBuilder | Builder pattern for Memory construction | ✅ Complete | ✓ |
| MemoryConfig | Global configuration system | ✅ Complete | ✓ |
| EntityExtractor | LLM-powered entity extraction | ✅ Complete | ✓ |
| DualRetriever | Fast + LLM semantic search | ✅ Complete | ✓ |
| EvolutionTracker | Memory relationship detection | ✅ Complete | ✓ |
| SimLLMProvider | Deterministic LLM simulation | ✅ Complete | ✓ |
| LanceVectorBackend | Production vector storage | ✅ Complete | ✓ |
| PostgresVectorBackend | Postgres-based storage | ✅ Complete | ✓ |

### LLM Providers (Rust)

```rust
use umi_memory::llm::{SimLLMProvider, LLMProvider};

// Simulation (deterministic, no API)
let llm = SimLLMProvider::with_seed(42);

// Anthropic Claude (requires 'anthropic' feature)
#[cfg(feature = "anthropic")]
let llm = umi_memory::llm::AnthropicProvider::new(api_key);

// OpenAI GPT (requires 'openai' feature)
#[cfg(feature = "openai")]
let llm = umi_memory::llm::OpenAIProvider::new(api_key);
```

*Note: Python bindings do not yet support LLM providers. See [PYTHON.md](PYTHON.md).*

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

## Production Mode (Rust)

```rust
use umi_memory::umi::{Memory, MemoryBuilder};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Production with real LLM (requires 'anthropic' feature)
    #[cfg(feature = "anthropic")]
    {
        use umi_memory::llm::AnthropicProvider;

        let api_key = env::var("ANTHROPIC_API_KEY")?;
        let llm = AnthropicProvider::new(&api_key);

        // Build memory with real provider
        // let memory = MemoryBuilder::new()
        //     .with_llm(llm)
        //     .build();
    }

    Ok(())
}
```

*Note: Python bindings currently only support low-level primitives (`CoreMemory`, `WorkingMemory`). High-level `Memory` class with LLM integration is planned for v0.3.0.*

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

```rust
// Rust - Deterministic simulation
use umi_memory::dst::SimConfig;
use umi_memory::umi::Memory;

// Same seed = same results
let mut memory1 = Memory::sim(42);
let mut memory2 = Memory::sim(42);

// Both produce identical results
```

### Running Tests

```bash
# Rust tests (~813 tests)
cargo test -p umi-memory --all-features

# All tests with specific DST seed
DST_SEED=12345 cargo test --all-features

# Rust with coverage
cargo tarpaulin --all-features --out Html

# Python tests (not yet implemented)
# pip install -e ".[dev]"
# pytest -v
```

### Fault Injection (Rust)

```rust
use umi_memory::dst::{Simulation, SimConfig, FaultConfig, FaultType};

// Test with simulated failures
let sim = Simulation::new(SimConfig::with_seed(42))
    .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.1));  // 10% failure

sim.run(|env| async move {
    let mut memory = env.create_memory();
    // Operations may fail due to fault injection
    Ok(())
}).await.unwrap();
```

*Note: Python bindings do not yet expose fault injection. See [PYTHON.md](PYTHON.md).*

### Test Coverage

| Category | Tests | Status |
|----------|-------|--------|
| **Rust Core** | ~813 | ✅ Passing |
| **Python Layer** | 0 | ⚠️ Not Implemented |

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

*Note: Python tests not yet implemented. PyO3 bindings can be built with:*

```bash
cd umi-py
pip install maturin
maturin develop  # Build and install locally

# Test the bindings work
python -c "import umi; print(umi.CoreMemory())"
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
# Rust (required)
cargo fmt && cargo clippy --all-features -- -D warnings && cargo test --all-features

# Python (when Python tests are implemented)
# ruff check . && ruff format . && pytest
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
│   └── adr/                # Architecture Decision Records
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
2. **Tests must pass** - All tests before commit
3. **TigerStyle** - Explicit limits, assertions, no silent failures
4. **No stubs** - Complete implementations or don't merge
5. **Graceful degradation** - Handle LLM failures elegantly

## License

MIT - See [LICENSE](./LICENSE) for details
