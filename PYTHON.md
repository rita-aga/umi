# Umi Python Bindings

Python bindings for Umi Memory using PyO3.

## Status: Feature Complete (v0.2.0)

✅ **All Phases 1-10 Complete** - Full Python bindings with async support!
⚠️  **Phase 12 Pending** - Real provider injection (requires type erasure)

### What Works

- ✅ **All Provider Classes Exposed** (Phase 2)
  - LLM: `AnthropicProvider`, `OpenAIProvider`, `SimLLMProvider`
  - Embedding: `OpenAIEmbeddingProvider`, `SimEmbeddingProvider`
  - Storage: `LanceStorageBackend`, `PostgresStorageBackend`, `SimStorageBackend`
  - Vector: `LanceVectorBackend`, `PostgresVectorBackend`, `SimVectorBackend`

- ✅ **Options Types** (Phase 3)
  - `RememberOptions` - Control entity extraction, evolution tracking, embeddings
  - `RecallOptions` - Control search limits, deep search, time ranges
  - `MemoryConfig` - Memory-wide configuration

- ✅ **Result Types** (Phase 4)
  - `RememberResult` - Contains stored entities and evolution relationships

- ✅ **Memory Class with Full API** (Phase 5-6)
  - `Memory.sim(seed)` - Create with Sim providers (deterministic testing)
  - **Async API**: `remember()`, `recall()`, `forget()`, `get()`, `count()`
  - **Sync API**: `remember_sync()`, `recall_sync()`, `forget_sync()`, `get_sync()`, `count_sync()`

- ✅ **Error Handling** (Phase 7)
  - Python exception types: `UmiError`, `EmptyTextError`, `StorageError`, etc.

- ✅ **Type Stubs** (Phase 8)
  - `umi.pyi` for IDE autocomplete and type checking

- ✅ **Examples** (Phase 9)
  - Basic sync usage
  - Options demo
  - Determinism demo

- ✅ **Tests** (Phase 10)
  - Basic unit tests with pytest

### What's Not Yet Implemented

- ⚠️ **Real Provider Integration in Memory** - Memory class currently only works with Sim providers
  - All real provider classes ARE exposed and functional
  - Just not wired into Memory constructor yet (needs enum-based type erasure)

## Installation

```bash
# Development mode (from umi root)
pip install maturin
cd umi-py
maturin develop

# Or build wheel
maturin build --release
pip install target/wheels/umi-*.whl
```

## Quick Start

### Async API (Recommended)

```python
import asyncio
import umi

async def main():
    # Create memory with deterministic seed
    memory = umi.Memory.sim(seed=42)

    # Store information
    options = umi.RememberOptions()
    result = await memory.remember("Alice works at Acme Corp", options)
    print(f"Stored {result.entity_count()} entities")

    # Retrieve information
    entities = await memory.recall("Who works at Acme?", umi.RecallOptions())
    for entity in entities:
        print(f"- {entity.name}: {entity.content}")

    # Count total entities
    total = await memory.count()
    print(f"Total: {total} entities")

asyncio.run(main())
```

### Sync API (For REPL/Simple Scripts)

```python
import umi

memory = umi.Memory.sim(seed=42)
result = memory.remember_sync("Alice works at Acme", umi.RememberOptions())
entities = memory.recall_sync("Alice", umi.RecallOptions())
print(f"Found {len(entities)} entities")
```

## Provider Classes

All provider classes are exposed and functional:

```python
# LLM Providers
llm = umi.AnthropicProvider(api_key="sk-ant-...")
llm = umi.OpenAIProvider(api_key="sk-...")
llm = umi.SimLLMProvider(seed=42)

# Embedding Providers
embedder = umi.OpenAIEmbeddingProvider(api_key="sk-...")
embedder = umi.SimEmbeddingProvider(seed=42)

# Storage Backends
storage = umi.LanceStorageBackend.connect(path="./umi_db")
storage = umi.PostgresStorageBackend.connect(url="postgresql://...")
storage = umi.SimStorageBackend(seed=42)

# Vector Backends
vector = umi.LanceVectorBackend.connect(path="./umi_db")
vector = umi.PostgresVectorBackend.connect(url="postgresql://...")
vector = umi.SimVectorBackend(seed=42)
```

**Note**: Real providers are exposed but not yet integrated into Memory constructor. Memory currently only supports `Memory.sim(seed)` constructor.

## Options API

### RememberOptions

```python
# Default options
options = umi.RememberOptions()

# Disable entity extraction (store as raw text)
options = umi.RememberOptions().without_extraction()

# Set importance score
options = umi.RememberOptions().with_importance(0.9)

# Disable embeddings
options = umi.RememberOptions().without_embeddings()

# Chain multiple options
options = (umi.RememberOptions()
    .without_evolution()
    .with_importance(0.8)
    .with_embeddings())
```

### RecallOptions

```python
# Default options
options = umi.RecallOptions()

# Set result limit
options = umi.RecallOptions().with_limit(20)

# Enable deep search (LLM query rewrite)
options = umi.RecallOptions().with_deep_search()

# Fast text-only search
options = umi.RecallOptions().fast_only()

# Time range filter (Unix timestamps in milliseconds)
options = umi.RecallOptions().with_time_range(start_ms=0, end_ms=9999999999)
```

## Deterministic Testing (DST)

Sim providers enable deterministic testing:

```python
# Same seed = same results
memory1 = umi.Memory.sim(seed=42)
memory2 = umi.Memory.sim(seed=42)

result1 = memory1.remember_sync("Alice works at Acme", umi.RememberOptions())
result2 = memory2.remember_sync("Alice works at Acme", umi.RememberOptions())

assert result1.entity_count() == result2.entity_count()  # ✅ Always true
```

## Type Checking

Type stubs are provided in `umi.pyi`:

```bash
# Install mypy
pip install mypy

# Type check your code
mypy your_script.py
```

## Examples

See `umi-py/examples/` for complete examples:

- `01_basic_sync_sim.py` - Basic sync usage
- `02_options_demo.py` - Options API
- `03_deterministic_demo.py` - DST demonstration
- `04_async_demo.py` - Native async/await support

## Testing

```bash
# Install pytest and pytest-asyncio
pip install pytest pytest-asyncio

# Run all tests (unit tests, no API keys needed)
pytest umi-py/tests/

# Run with verbose output
pytest umi-py/tests/ -v

# Run only async tests
pytest umi-py/tests/test_memory_async.py -v

# Skip integration tests (default)
pytest umi-py/tests/ -m "not integration"
```

## Implementation Notes

### Current Limitations

1. **Memory only supports Sim providers** - Real providers (Anthropic, OpenAI, Lance, Postgres) are exposed as classes but not yet integrated into Memory constructor. Requires enum-based type erasure to support all provider combinations.

### Why Sim Providers Only in Memory?

The Rust `Memory` struct is generic over 4 provider types:
```rust
Memory<L: LLMProvider, E: EmbeddingProvider, S: StorageBackend, V: VectorBackend>
```

Supporting all combinations in Python requires type erasure (enum or trait objects), which adds significant complexity. The current implementation prioritizes getting a working Memory API with full DST capabilities. Real provider support in Memory is planned for a future release.

### Future Work (v0.3.0)

- [ ] Add enum-based type erasure for Memory to support all provider combinations
- [ ] Integration tests for real providers (Anthropic, OpenAI, LanceDB, Postgres)
- [ ] Builder pattern for Memory construction
- [ ] Comprehensive documentation with all provider examples

## Architecture

```
umi-py/
├── src/
│   └── lib.rs           # PyO3 bindings (1500+ lines)
├── examples/            # Example scripts
├── tests/               # Pytest tests
├── umi.pyi              # Type stubs
└── Cargo.toml           # Dependencies

Key components:
- Provider wrappers (10 classes)
- Options types (3 classes)
- Result types (1 class)
- Memory class (sync API)
- Exception types (8 classes)
```

## Contributing

See [CLAUDE.md](./CLAUDE.md) for development guidelines, including:

- TigerStyle engineering principles
- DST (Deterministic Simulation Testing)
- Commit policy (only working software)
- Pre-commit verification

## License

MIT
