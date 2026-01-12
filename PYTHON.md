# Python Bindings for Umi Memory

**Status**: 🔶 Experimental - Basic functionality works, but incomplete

The Python bindings for Umi are built with PyO3 and provide a Python interface to the Rust library. However, they are currently **incomplete** and should be considered **experimental**.

## What Works ✅

- **Basic Memory Operations**:
  - `Memory.sim(seed)` - Create deterministic memory
  - `memory.remember(text)` - Store information
  - `memory.recall(query)` - Retrieve information
  - Returns lists of entities with basic fields

## What's Missing ❌

- **Type System**: Most Rust types not exposed to Python
  - No `RememberOptions` class
  - No `RecallOptions` class
  - No `MemoryConfig` class
  - No `Entity` class with full fields
  - No `EvolutionRelation` type

- **Advanced Features**:
  - Cannot customize memory configuration
  - Cannot use real LLM providers (Anthropic, OpenAI)
  - Cannot use production storage backends (LanceDB, Postgres)
  - Limited error handling

- **Documentation**:
  - No type stubs (`.pyi` files)
  - No docstrings in Python
  - No Python examples

## Installation

**Note**: Not published to PyPI yet. Install from source:

```bash
cd umi-py
pip install maturin
maturin develop  # Build and install locally
```

## Usage

```python
from umi import Memory

# Create deterministic memory (simulation only)
memory = Memory.sim(42)

# Remember information
memory.remember("Alice is a software engineer at Acme Corp")
memory.remember("Bob is the CTO at TechCo")

# Recall information
results = memory.recall("Who works at Acme?")

for entity in results:
    print(f"Found: {entity.name} - {entity.content}")
```

## Current Architecture

```
umi-py/
├── src/
│   └── lib.rs          # PyO3 bindings (incomplete)
├── Cargo.toml          # Python package config
└── pyproject.toml      # Maturin config
```

The bindings expose a minimal `Memory` class but don't expose the full Rust type system.

## Roadmap

### Version 0.2.0 - Full Options Classes
**Target**: Q1 2026

- [ ] Expose `MemoryConfig` to Python
- [ ] Expose `RememberOptions` and `RecallOptions`
- [ ] Expose `Entity` with all fields
- [ ] Expose `EvolutionRelation` and evolution types
- [ ] Add Python type stubs (`.pyi` files)
- [ ] Add docstrings for all classes and methods

### Version 0.3.0 - Real Providers
**Target**: Q2 2026

- [ ] Support Anthropic provider from Python
- [ ] Support OpenAI provider from Python
- [ ] Support LanceDB storage from Python
- [ ] Async support (Python `async`/`await`)
- [ ] Proper error handling with Python exceptions

### Version 0.4.0 - Production Ready
**Target**: Q3 2026

- [ ] Complete type system parity with Rust
- [ ] Full documentation and examples
- [ ] PyPI publication
- [ ] Wheels for major platforms (Linux, macOS, Windows)
- [ ] Comprehensive test suite

### Version 1.0.0 - Feature Parity
**Target**: 2026

- [ ] All Rust features available in Python
- [ ] Performance benchmarks
- [ ] Production deployment guide
- [ ] Integration with popular Python ML frameworks

## Why Rust First?

Umi prioritizes the **Rust API** for several reasons:

1. **Performance**: Rust provides zero-cost abstractions and predictable performance
2. **Type Safety**: Catch bugs at compile time, not runtime
3. **Deterministic Testing**: DST framework is built in Rust
4. **Memory Safety**: No garbage collector, explicit ownership
5. **Production Ready**: Rust is ideal for high-performance backend services

Python bindings are a "nice to have" but not the primary focus.

## Contributing to Python Bindings

If you'd like to help complete the Python bindings:

1. **Expose More Types**: Add PyO3 wrappers in `umi-py/src/lib.rs`
2. **Add Type Stubs**: Create `.pyi` files for IDE support
3. **Write Examples**: Add Python examples in `umi-py/examples/`
4. **Test Coverage**: Add Python tests in `umi-py/tests/`

See [CONTRIBUTING.md](https://github.com/rita-aga/umi/blob/main/CONTRIBUTING.md) for guidelines.

## Example: What Full Bindings Should Look Like

**Goal** (not yet implemented):

```python
from umi import (
    Memory,
    MemoryConfig,
    MemoryBuilder,
    RememberOptions,
    RecallOptions,
    AnthropicProvider,
    OpenAIEmbeddingProvider,
    LanceStorageBackend,
)
import os
from datetime import timedelta

# Configure production memory
config = MemoryConfig(
    core_memory_bytes=128 * 1024,
    working_memory_bytes=10 * 1024 * 1024,
    working_memory_ttl=timedelta(hours=4),
    recall_limit=50,
)

# Create with real providers
memory = MemoryBuilder() \
    .with_llm(AnthropicProvider(os.environ["ANTHROPIC_API_KEY"])) \
    .with_embedder(OpenAIEmbeddingProvider(os.environ["OPENAI_API_KEY"])) \
    .with_storage(LanceStorageBackend.connect("./lance_db")) \
    .with_config(config) \
    .build()

# Remember with options
result = await memory.remember(
    "Alice is a software engineer at Acme Corp",
    RememberOptions(
        extract_entities=True,
        generate_embeddings=True,
        detect_evolutions=True,
    )
)

print(f"Stored {result.entity_count} entities")
if result.has_evolutions:
    for evolution in result.evolutions:
        print(f"Evolution: {evolution.type} - {evolution.reason}")

# Recall with options
results = await memory.recall(
    "Who works at Acme?",
    RecallOptions(limit=10, use_vector_search=True)
)

for entity in results:
    print(f"{entity.name}: {entity.content} (confidence: {entity.confidence})")
```

## Current vs. Future State

| Feature | v0.1 (Current) | v1.0 (Goal) |
|---------|----------------|-------------|
| Basic remember/recall | ✅ | ✅ |
| Options classes | ❌ | ✅ |
| Real LLM providers | ❌ | ✅ |
| Storage backends | ❌ | ✅ |
| Async support | ❌ | ✅ |
| Type stubs | ❌ | ✅ |
| Documentation | ❌ | ✅ |
| PyPI package | ❌ | ✅ |
| Error handling | ⚠️ Basic | ✅ Full |

## Questions?

- **Q: When will Python bindings be complete?**
  A: Version 0.2.0 (Q1 2026) will have full options classes. Version 1.0.0 (2026) will have feature parity.

- **Q: Should I use Python or Rust?**
  A: Use **Rust** for production. Python bindings are experimental and incomplete.

- **Q: Can I help complete the bindings?**
  A: Yes! See the Contributing section above. PRs welcome.

- **Q: Will Python bindings be maintained long-term?**
  A: Yes, but Rust is the primary API. Python will always lag slightly behind.

## License

Same as Umi: MIT License

## Resources

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Documentation](https://www.maturin.rs/)
- [Umi Rust API Documentation](https://docs.rs/umi-memory)
