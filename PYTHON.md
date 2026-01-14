# Python Bindings for Umi Memory

**Status**: 🔶 Experimental - Low-level primitives only

The Python bindings for Umi are built with PyO3 and provide a Python interface to the Rust library. However, they are currently **incomplete** and should be considered **experimental**.

## What Works ✅

- **Low-Level Memory Primitives**:
  - `CoreMemory` - 32KB structured memory for LLM context
  - `WorkingMemory` - 1MB KV store with TTL
  - `Entity` - Entity objects for archival storage
  - `EvolutionRelation` - Memory evolution relationships

## What's Missing ❌

- **High-Level API**:
  - No `Memory` class (the main orchestrator)
  - No `remember()` / `recall()` high-level methods
  - No `RememberOptions` or `RecallOptions` classes
  - No `MemoryConfig` class

- **Advanced Features**:
  - Cannot use real LLM providers (Anthropic, OpenAI)
  - Cannot use production storage backends (LanceDB, Postgres)
  - No async support
  - Limited error handling

- **Documentation**:
  - No type stubs (`.pyi` files)
  - No Python examples directory
  - No Python tests

## Installation

**Note**: Not published to PyPI yet. Install from source:

```bash
cd umi-py
pip install maturin
maturin develop  # Build and install locally
```

## Usage

```python
import umi

# Core Memory (32KB, always in LLM context)
core = umi.CoreMemory()
core.set_block("system", "You are a helpful assistant.")
core.set_block("human", "User prefers concise responses.")
core.set_block("facts", "User's name is Alice.")

# Render to XML for LLM context
context = core.render()
print(context)

# Check usage
print(f"Used: {core.used_bytes}/{core.max_bytes} bytes ({core.utilization*100:.1f}%)")

# Working Memory (1MB KV store with TTL)
working = umi.WorkingMemory()
working.set("session_id", b"abc123")
working.set("user_prefs", b'{"theme": "dark"}', ttl_secs=3600)  # 1 hour TTL

value = working.get("session_id")  # Returns bytes
if value:
    print(f"Session: {value.decode()}")

# Entity for archival storage
entity = umi.Entity("person", "Alice", "Software engineer at Acme Corp")
print(f"{entity.name}: {entity.content}")
print(f"Type: {entity.entity_type}, ID: {entity.id}")

# Evolution relation (memory updates/contradictions)
relation = umi.EvolutionRelation(
    source_id="entity-1",
    target_id="entity-2",
    evolution_type="update",
    reason="Job change",
    confidence=0.95
)
print(f"Evolution: {relation.evolution_type} ({relation.confidence})")
```

## Current Architecture

```
umi-py/
├── src/
│   └── lib.rs          # PyO3 bindings
├── Cargo.toml          # Rust package config
└── pyproject.toml      # Maturin config
```

The bindings expose low-level memory primitives but NOT the high-level `Memory` class.

## Roadmap

### Version 0.2.0 - High-Level Memory Class
**Target**: Q1 2026

- [ ] Expose `Memory` class to Python (main orchestrator)
- [ ] Expose `MemoryConfig` to Python
- [ ] Expose `RememberOptions` and `RecallOptions`
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
| CoreMemory | ✅ | ✅ |
| WorkingMemory | ✅ | ✅ |
| Entity | ✅ | ✅ |
| EvolutionRelation | ✅ | ✅ |
| Memory class | ❌ | ✅ |
| remember/recall | ❌ | ✅ |
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
  A: Version 0.2.0 (Q1 2026) will add the high-level `Memory` class. Version 1.0.0 (2026) will have feature parity.

- **Q: Should I use Python or Rust?**
  A: Use **Rust** for production. Python bindings currently only offer low-level primitives.

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
