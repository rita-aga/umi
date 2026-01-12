# Umi

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

```bash
# Install from source (not yet on PyPI)
pip install git+https://github.com/rita-aga/umi.git

# With LLM provider support
pip install "umi[anthropic] @ git+https://github.com/rita-aga/umi.git"
pip install "umi[openai] @ git+https://github.com/rita-aga/umi.git"
```

```python
from umi import Memory

# Simulation mode (deterministic, no API calls)
memory = Memory(seed=42)

# Remember information
entities = await memory.remember("I met Alice at Acme Corp")
print(f"Stored {len(entities)} entities")

# Recall memories
results = await memory.recall("Who do I know at Acme?")
for entity in results:
    print(f"  - {entity.name}: {entity.content}")
```

### Rust

```bash
cargo add umi-memory --features lance
```

```rust
use umi_memory::{CoreMemory, SimConfig};

// Create with deterministic seed
let config = SimConfig::with_seed(42);
let memory = CoreMemory::new(32 * 1024, config);

// Store and retrieve
memory.write(b"important context")?;
let data = memory.read_all()?;
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
| **umi-memory** | Rust core (memory tiers, DST) | ✅ Complete | 232 |
| **umi-py** | PyO3 bindings | 🚧 Planned | - |
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
// Simulation backend (in-memory)
use umi_memory::{SimStorageBackend, SimVectorBackend};

// LanceDB backend (persistent, production)
use umi_memory::LanceVectorBackend;

// Postgres backend (with pgvector)
use umi_memory::PostgresVectorBackend;
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

# Rust tests (232 tests)
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
| **Rust Core** | 232 | ✅ Passing | ~85% |
| **Python Layer** | 145 | ✅ Passing | ~78% |
| **Total** | 377 | ✅ All Passing | ~82% |

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
2. **Tests must pass** - All 377 tests before commit
3. **TigerStyle** - Explicit limits, assertions, no silent failures
4. **No stubs** - Complete implementations or don't merge
5. **Graceful degradation** - Handle LLM failures elegantly

## License

MIT - See [LICENSE](./LICENSE) for details
