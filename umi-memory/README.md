# umi-memory

[![Crates.io](https://img.shields.io/crates/v/umi-memory.svg)](https://crates.io/crates/umi-memory)
[![Documentation](https://docs.rs/umi-memory/badge.svg)](https://docs.rs/umi-memory)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A production-ready memory library for AI agents with deterministic simulation testing.

## Features

- **🧠 Smart Memory Management**: Core, working, and archival memory tiers with automatic eviction
- **🔍 Dual Retrieval**: Fast vector search + LLM-powered semantic query expansion
- **🔄 Evolution Tracking**: Automatically detect updates, contradictions, and derived insights
- **✅ Graceful Degradation**: System continues operating even when LLM/storage components fail
- **🎯 Deterministic Testing**: Full DST (Deterministic Simulation Testing) for reproducible fault injection
- **🚀 Production Backends**: LanceDB for embedded vectors, Postgres for persistence

## Quick Start

```toml
[dependencies]
umi-memory = "0.2"
```

### Basic Usage

```rust
use umi_memory::umi::{Memory, RememberOptions, RecallOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create memory with simulation providers (deterministic, seed 42)
    let mut memory = Memory::sim(42);

    // Remember information
    memory.remember(
        "Alice is a software engineer at Acme Corp",
        RememberOptions::default()
    ).await?;

    // Recall information (with optional limit: 1-100, default 10)
    let results = memory.recall(
        "Who works at Acme?",
        RecallOptions::default().with_limit(20)?
    ).await?;

    for entity in results {
        println!("Found: {} - {}", entity.name, entity.content);
    }

    Ok(())
}
```

### Production Setup

```rust
use umi_memory::umi::{Memory, MemoryBuilder, MemoryConfig};
use umi_memory::llm::AnthropicProvider;
use umi_memory::embedding::OpenAIEmbedding;
use umi_memory::storage::LanceStorageBackend;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure production memory settings
    let config = MemoryConfig::default()
        .with_core_memory_bytes(128 * 1024)        // 128 KB
        .with_working_memory_bytes(10 * 1024 * 1024) // 10 MB
        .with_working_memory_ttl(Duration::from_secs(3600 * 4)) // 4 hours
        .with_recall_limit(50);

    // Create with real providers
    let llm = AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY")?);
    let embedder = OpenAIEmbedding::new(std::env::var("OPENAI_API_KEY")?);
    let storage = LanceStorageBackend::connect("./lance_db").await?;

    let mut memory = MemoryBuilder::new()
        .with_llm(llm)
        .with_embedder(embedder)
        .with_storage(storage)
        .with_config(config)
        .build();

    // Use memory...
    Ok(())
}
```

## Feature Flags

- **`lance`** - LanceDB storage backend (persistent, embedded vector database)
- **`postgres`** - PostgreSQL storage backend (persistent, external database)
- **`anthropic`** - Anthropic LLM provider (Claude)
- **`openai`** - OpenAI LLM provider (GPT, embeddings)
- **`llm-providers`** - All LLM providers (convenience flag)
- **`embedding-providers`** - All embedding providers (convenience flag)
- **`opentelemetry`** - OpenTelemetry distributed tracing and metrics export

```toml
[dependencies]
umi-memory = { version = "0.2", features = ["lance", "anthropic", "openai"] }
```

## Architecture

Umi follows **TigerStyle** principles:
- ✅ Simulation-first: Every component has deterministic simulation
- ✅ Graceful degradation: System continues operating despite component failures
- ✅ Assertions everywhere: `debug_assert!` validates invariants
- ✅ Same seed = same results: Reproducible testing and debugging

### Core Components

1. **EntityExtractor** - Extracts structured entities from text using LLM
2. **DualRetriever** - Fast vector search + LLM query rewriting with RRF merging
3. **EvolutionTracker** - Detects updates, contradictions, extensions, and derivations
4. **Memory Orchestrator** - Coordinates all components with graceful degradation

### Memory Tiers

- **CoreMemory**: Small (32 KB), persistent, always loaded
- **WorkingMemory**: Medium (1 MB), TTL-based eviction, in-memory cache
- **ArchivalMemory**: Large (unlimited), persistent storage + vector search

## Examples

See the [`examples/`](https://github.com/rita-aga/umi/tree/main/umi-memory/examples) directory:

- `quick_start.rs` - Basic remember/recall workflow
- `production_setup.rs` - Production configuration with real providers
- `configuration.rs` - Customize memory behavior
- `test_anthropic.rs` - Test Anthropic integration (requires API key)
- `test_openai.rs` - Test OpenAI integration (requires API key)

Run examples:
```bash
cargo run --example quick_start
cargo run --example production_setup --features lance,anthropic
```

## Testing

Umi has comprehensive test coverage:
- **462 unit tests** - All passing
- **44 DST tests** - Fault injection with deterministic simulation
- **Benchmarks** - Criterion-based performance tracking

```bash
# Run all tests
cargo test -p umi-memory --all-features

# Run fault injection tests
cargo test -p umi-memory --lib dst_tests

# Run benchmarks
cargo bench -p umi-memory
```

## Documentation

- **[Architecture Decision Records](https://github.com/rita-aga/umi/tree/main/docs/adr)** - Design decisions and rationale
- **[API Documentation](https://docs.rs/umi-memory)** - Full API reference
- **[CLAUDE.md](https://github.com/rita-aga/umi/blob/main/CLAUDE.md)** - Development guide for AI assistants
- **[Testing Guide](https://github.com/rita-aga/umi/blob/main/docs/testing)** - Testing philosophy and practices

## Development Philosophy

### Simulation-First

Every component MUST have a simulation implementation:

```rust
use umi_memory::{Memory, SimLLMProvider, SimStorageBackend, SimConfig};

// Deterministic - same seed = same results
let config = SimConfig::with_seed(42);
let memory = Memory::sim(42);
```

**Why?** Same seed = same results = reproducible tests and bugs.

### Understanding SimLLM

`SimLLMProvider` returns **deterministic placeholder data** by design for testing:

- **Entity Names**: Rotates through "Alice", "Bob", "Eve", "Charlie", etc.
- **Content**: Generic text like "Information about X"
- **Purpose**: Enable reproducible testing without API costs

**This is correct behavior!** SimLLM is designed for:
- ✅ Unit tests that need consistent results
- ✅ Development without API keys
- ✅ CI/CD pipelines
- ✅ Fault injection testing

**For real content extraction**, use production LLM providers:
```rust
use umi_memory::llm::AnthropicProvider;

let llm = AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY")?);
let memory = MemoryBuilder::new().with_llm(llm).build();
```

See [`examples/test_anthropic.rs`](https://github.com/rita-aga/umi/blob/main/umi-memory/examples/test_anthropic.rs) and [`examples/test_openai.rs`](https://github.com/rita-aga/umi/blob/main/umi-memory/examples/test_openai.rs) for production setups.

### Graceful Degradation

Components handle failures gracefully:
- **EntityExtractor**: Creates fallback entities (type=Note, confidence=0.5)
- **DualRetriever**: Falls back to fast search when LLM fails
- **EvolutionTracker**: Skips detection (Ok(None)) when LLM fails

### TigerStyle Assertions

Use `debug_assert!` for invariants:
```rust
fn store(&mut self, data: &[u8]) -> Result<()> {
    debug_assert!(!data.is_empty(), "data must not be empty");
    debug_assert!(data.len() <= self.capacity, "data exceeds capacity");
    // ...
}
```

## Python Bindings

Python bindings are available but experimental. See [PYTHON.md](https://github.com/rita-aga/umi/blob/main/PYTHON.md) for details.

```python
from umi import Memory

memory = Memory.sim(42)
memory.remember("Alice is an engineer")
results = memory.recall("Who is Alice?")
```

**Status**: Basic functionality works, but incomplete type system. Rust API is primary.

## Contributing

Contributions welcome! Please:
1. Read [CLAUDE.md](https://github.com/rita-aga/umi/blob/main/CLAUDE.md) for development guidelines
2. Ensure all tests pass: `cargo test --all-features`
3. Add tests for new features
4. Follow TigerStyle principles (simulation-first, graceful degradation)

## License

MIT License - see [LICENSE](https://github.com/rita-aga/umi/blob/main/LICENSE) for details.

## Origin

Umi was extracted from RikaiOS to be a standalone library. Inspired by:
- **memU** - Dual retrieval (fast + LLM semantic)
- **Mem0** - Entity extraction and evolution tracking
- **Supermemory** - Temporal metadata
- **TigerStyle** - Assertion-based programming

## Versioning

Umi follows semantic versioning (SemVer):
- **0.1.x** - Beta release, core features stable
- **0.2.x** - Python bindings complete
- **1.0.0** - Production-ready, API stable

Breaking changes are expected in 0.x versions.
