# CLAUDE.md

Instructions for Claude Code when working in the Umi repository.

## What Umi Is

Umi is a memory library for AI agents, fully implemented in Rust:

**Rust crate** (`umi-memory/`):
- Memory class - Main interface with `remember()` and `recall()`
- EntityExtractor - LLM-powered entity extraction
- DualRetriever - Fast vector + LLM query rewriting with RRF merging
- EvolutionTracker - Detects updates, contradictions, extensions
- Storage backends - SimStorageBackend (testing) + LanceStorageBackend (production)
- LLM providers - SimLLMProvider, AnthropicProvider, OpenAIProvider
- DST (Deterministic Simulation Testing)
- Memory tiers (CoreMemory, WorkingMemory, ArchivalMemory)

**Python bindings** (`umi-py/`):
- PyO3 bindings exposing Rust types to Python
- Built with maturin for PyPI distribution

## Development Philosophy

### Simulation-First (Mandatory)

Every component MUST have a simulation implementation:

```rust
use umi_memory::{Memory, SimLLMProvider, SimStorageBackend, SimConfig};

// Deterministic - same seed = same results
let config = SimConfig::with_seed(42);
let llm = SimLLMProvider::new();
let storage = SimStorageBackend::new();
let memory = Memory::new(llm, storage);
```

**Why?** Same seed = same results = reproducible tests and bugs.

### TigerStyle Assertions

Use `debug_assert!` for invariants:
```rust
fn store(&mut self, data: &[u8]) -> Result<()> {
    debug_assert!(!data.is_empty(), "data must not be empty");
    debug_assert!(data.len() <= self.capacity, "data exceeds capacity");
    // ...
}
```

### Graceful Degradation

LLM calls can fail. Components should:
- Return fallback values instead of crashing
- Log errors but continue operation

## Build & Test Commands

### Rust

```bash
cargo test -p umi-memory --features lance   # 405 tests
cargo clippy --all-features
cargo fmt --check
cargo build --release
```

### Python Bindings (via maturin)

```bash
pip install maturin
maturin develop                              # Build and install locally
python -c "import umi; print(umi.__version__)"
```

## Architecture Decision Records (ADRs)

Document significant design decisions in `docs/adr/`:
- `013-llm-provider-trait.md` - LLM abstraction
- `014-entity-extractor.md` - Entity extraction
- `015-dual-retriever.md` - Query rewriting and RRF
- `016-evolution-tracker.md` - Memory relationships
- `017-memory-class.md` - Main orchestrator
- `018-lance-storage-backend.md` - LanceDB storage

Create new ADRs for architectural changes. Format: `NNN-short-name.md`

## Directory Structure

```
umi/
├── umi-memory/             # Rust crate (full implementation)
│   └── src/
│       ├── lib.rs          # Public exports
│       ├── umi/            # Memory orchestrator
│       ├── extraction/     # EntityExtractor
│       ├── retrieval/      # DualRetriever
│       ├── evolution/      # EvolutionTracker
│       ├── storage/        # Sim + Lance backends
│       ├── llm/            # LLM providers
│       ├── memory/         # Memory tiers
│       ├── dst/            # Deterministic simulation
│       └── constants.rs    # TigerStyle limits
│
├── umi-py/                 # PyO3 bindings
│   └── src/lib.rs          # Python module
│
├── Cargo.toml              # Rust workspace
├── pyproject.toml          # Python package (maturin)
└── docs/adr/               # Architecture decisions
```

## Features

The `umi-memory` crate has optional features:

```toml
[dependencies]
umi-memory = { version = "0.1", features = ["lance"] }
```

| Feature | Description |
|---------|-------------|
| `lance` | LanceDB storage backend (persistent, vector search) |
| `anthropic` | Anthropic LLM provider |
| `openai` | OpenAI LLM provider |
| `llm-providers` | All LLM providers |

## Git Workflow

- Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`
- Run tests before pushing: `cargo test -p umi-memory --features lance`
- Update ADRs for architectural changes
- Always commit and push after completing work

## Planning & Progress

**Before starting work**, check these directories:

- `.vision/umi-vision.md` - Core principles and constraints
- `.progress/` - Active plans and roadmaps

## Next Steps

1. **Benchmarking** - Set up criterion benchmarks to validate performance
2. **PyO3 bindings** - Expose full Memory API to Python
3. **crates.io** - Publish `umi-memory` crate
4. **PyPI** - Publish Python package via maturin

## Origin

Umi was extracted from RikaiOS to be a standalone library. Inspired by:
- **memU** - Dual retrieval (fast + LLM semantic)
- **Mem0** - Entity extraction and evolution tracking
- **Supermemory** - Temporal metadata
- **TigerStyle** - Assertion-based programming
