# CLAUDE.md - Umi Development Guide

This document provides guidance for AI assistants (and human developers) contributing to Umi.

## Project Overview

Umi is a memory library for AI agents with entity extraction, dual retrieval, and evolution tracking. Built with DST-first (Deterministic Simulation Testing) and TigerStyle engineering principles.

**Key Architecture**:
- **Rust core** (`umi-memory/`): Memory tiers, storage backends, DST
- **Python layer** (planned `umi-py/`): LLM integration, entity extraction, smart retrieval

## Quick Commands

```bash
# Build the entire workspace
cargo build --all-features

# Run all tests
cargo test --all-features
pytest  # Python tests

# Run tests with DST seed for reproduction
DST_SEED=12345 cargo test -p umi-memory

# Format code
cargo fmt
ruff format .

# Run linter
cargo clippy --all-features -- -D warnings
ruff check .

# Type check Python
mypy umi/

# Run benchmarks
cargo bench -p umi-memory

# Coverage
cargo tarpaulin --all-features --out Html
```

## Architecture

```
umi/
├── umi-memory/           # Rust core
│   ├── src/
│   │   ├── lib.rs        # Public exports
│   │   ├── umi/          # Memory orchestrator
│   │   ├── extraction/   # EntityExtractor
│   │   ├── retrieval/    # DualRetriever
│   │   ├── evolution/    # EvolutionTracker
│   │   ├── storage/      # Sim + Lance + Postgres backends
│   │   ├── llm/          # LLM providers (Sim, Anthropic, OpenAI)
│   │   ├── memory/       # Memory tiers (Core, Working, Archival)
│   │   ├── dst/          # Deterministic Simulation Testing
│   │   └── constants.rs  # TigerStyle constants
│   └── benches/          # Criterion benchmarks
│
├── umi-py/               # PyO3 bindings (planned)
│
├── docs/adr/             # Architecture Decision Records
│
├── .github/workflows/    # CI/CD
│   ├── ci.yml            # Python CI
│   └── ci-rust.yml       # Rust CI
│
└── tests/                # Integration tests
```

---

## TigerStyle Engineering Principles

Umi follows TigerBeetle's TigerStyle: **Safety > Performance > DX**

### 1. Explicit Constants with Units

All limits are named constants with units in the name:

```rust
// Good - unit in name, explicit limit
pub const MEMORY_SIZE_BYTES_MAX: usize = 32 * 1024;  // 32KB
pub const WORKING_MEMORY_BYTES_MAX: usize = 1024 * 1024;  // 1MB
pub const TTL_SECONDS_DEFAULT: u64 = 3600;  // 1 hour
pub const ENTITY_NAME_LENGTH_BYTES_MAX: usize = 256;

// Bad - unclear units, magic number
pub const MAX_SIZE: usize = 32768;
const TIMEOUT: u64 = 3600;
```

### 2. Big-Endian Naming

Name identifiers from big to small concept:

```rust
// Good - big to small
memory_size_bytes_max
storage_write_latency_ms_base
entity_extraction_timeout_seconds

// Bad - small to big
max_memory_size_bytes
base_latency_ms_storage_write
timeout_seconds_entity_extraction
```

### 3. Assertions (2+ per Function)

Every non-trivial function should have at least 2 assertions:

```rust
pub fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
    // Preconditions
    assert!(!text.is_empty(), "text must not be empty");
    assert!(text.len() <= TEXT_LENGTH_BYTES_MAX, "text exceeds maximum");

    let entities = self.llm_provider.extract(text)?;

    // Postconditions
    assert!(entities.len() <= ENTITIES_COUNT_MAX);
    assert!(entities.iter().all(|e| !e.name.is_empty()));

    Ok(entities)
}
```

### 4. Prefer u64 Over usize for Sizes

Use fixed-width integers for portability:

```rust
// Good - portable across platforms
pub fn size_bytes(&self) -> u64;
pub fn entity_count(&self) -> u64;

// Bad - varies by platform (32-bit vs 64-bit)
pub fn size_bytes(&self) -> usize;
```

### 5. No Silent Truncation

Avoid implicit conversions that could truncate:

```rust
// Good - explicit conversion with assertion
let size: u64 = data.len() as u64;
assert!(size <= u32::MAX as u64, "size too large for u32");
let size_u32: u32 = size as u32;

// Bad - silent truncation on 64-bit platforms
let size: u32 = data.len() as u32;
```

### 6. Explicit Error Handling

No `unwrap()` or `expect()` in production code (only tests):

```rust
// Good - explicit error handling
let entities = self.extractor.extract(text)?;
let result = self.storage.get(key)
    .map_err(|e| Error::StorageReadFailed { key: key.to_string(), reason: e.to_string() })?;

// Bad - panics in production
let entities = self.extractor.extract(text).unwrap();
let result = self.storage.get(key).expect("storage read failed");
```

### 7. Debug Assertions for Expensive Checks

Use `debug_assert!` for checks that are too expensive for release:

```rust
pub fn insert(&mut self, entity: Entity) {
    // Cheap check - always run
    assert!(entity.name.len() <= ENTITY_NAME_LENGTH_BYTES_MAX);

    // Expensive check - debug only
    debug_assert!(self.validate_entity_uniqueness(&entity));
    debug_assert!(self.check_memory_consistency());

    self.entities.push(entity);
}
```

---

## DST (Deterministic Simulation Testing)

### Core Principles

1. **All randomness flows from a single seed** - set `DST_SEED` to reproduce
2. **Simulated time** - `SimClock` replaces wall clock
3. **Explicit fault injection** - `FaultConfig` with configurable probability
4. **Deterministic LLM** - `SimLLMProvider` returns predictable results

### Running DST Tests

```bash
# Run with random seed (logged for reproduction)
cargo test -p umi-memory

# Reproduce specific run
DST_SEED=12345 cargo test -p umi-memory

# Python DST
pytest --seed=42
```

### Writing DST Tests (Rust)

```rust
use umi_memory::{SimConfig, SimLLMProvider, SimStorageBackend};

#[test]
fn test_entity_extraction_deterministic() {
    let config = SimConfig::with_seed(42);
    let llm = SimLLMProvider::new(config);

    let entities = llm.extract_entities("Alice works at Acme").unwrap();

    // With seed 42, always extract exactly 2 entities
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].name, "Alice");
    assert_eq!(entities[1].name, "Acme");
}

#[test]
fn test_with_fault_injection() {
    let mut config = SimConfig::with_seed(42);
    config.set_fault(FaultType::StorageWriteFail, 0.1);  // 10% failure rate

    let storage = SimStorageBackend::new(config);

    // Test handles write failures gracefully
    for i in 0..100 {
        let result = storage.write(&format!("key{}", i), b"value");
        // Should handle failures without panicking
    }
}
```

### Writing DST Tests (Python)

```python
from umi import Memory

def test_memory_deterministic():
    # Same seed = same results
    memory1 = Memory(seed=42)
    entities1 = await memory1.remember("Alice works at Acme")

    memory2 = Memory(seed=42)
    entities2 = await memory2.remember("Alice works at Acme")

    assert entities1 == entities2  # Identical results

def test_with_fault_injection():
    from umi import FaultConfig, FaultType

    memory = Memory(
        seed=42,
        fault_config=FaultConfig(
            fault_type=FaultType.STORAGE_WRITE_FAIL,
            probability=0.1
        )
    )

    # Should handle failures gracefully
    result = await memory.remember("test")
    assert result is not None  # Fallback value
```

### Fault Types

| Category | Fault Types | Description |
|----------|-------------|-------------|
| **Storage** | `StorageWriteFail` | Write operation fails |
| | `StorageReadFail` | Read operation fails |
| | `StorageCorruption` | Data corrupted on read |
| | `StorageLatency` | Artificial delay |
| **LLM** | `LLMTimeout` | LLM request times out |
| | `LLMRateLimitLLM` | Rate limit exceeded |
| | `LLMInvalidResponse` | Malformed response |
| **Network** | `NetworkTimeout` | Network call times out |
| | `NetworkPartition` | Simulated network split |

### Memory Integration with Fault Injection

When testing Memory operations with fault injection, use `SimEnvironment::create_memory()` to create a Memory instance with providers connected to the simulation's fault injector:

```rust
use umi_memory::dst::{Simulation, SimConfig, FaultConfig, FaultType};
use umi_memory::umi::RememberOptions;

#[tokio::test]
async fn test_memory_with_fault_injection() {
    let sim = Simulation::new(SimConfig::with_seed(42))
        .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.1));

    sim.run(|env| async move {
        // Create Memory with providers connected to the simulation's FaultInjector
        let mut memory = env.create_memory();

        // Now all memory operations have fault injection applied
        match memory.remember("Alice works at Acme", RememberOptions::default()).await {
            Ok(result) => println!("Stored {} entities", result.entities.len()),
            Err(e) => println!("Failed due to fault: {}", e),  // May fail due to 10% fault rate
        }

        Ok::<(), umi_memory::umi::MemoryError>(())
    }).await.unwrap();
}
```

**Important Notes:**
- Use `env.create_memory()` instead of `Memory::sim(seed)` when you want fault injection
- `Memory::sim(seed)` still works for simple tests without fault injection (backward compatible)
- Fault injection is global across all providers - an LLM fault might cause storage operations to fail
- This is expected behavior since the FaultInjector is shared

---

## Code Style

### Module Organization

```rust
//! Module-level documentation with TigerStyle note
//!
//! TigerStyle: Brief description of the module's invariants.

// Imports grouped by: std, external crates, internal crates, local modules
use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::dst::SimConfig;
use crate::constants::*;

use super::entity::Entity;
```

### Struct Layout

```rust
/// Brief description
///
/// Longer description if needed.
///
/// # Invariants
/// - `name` is never empty
/// - `confidence` is in [0.0, 1.0]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    // Public fields at top with documentation
    /// The entity's name
    pub name: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    // Private fields below
    internal_id: u64,
    metadata: HashMap<String, String>,
}
```

### Function Signatures

```rust
/// Extract entities from text using LLM.
///
/// # Arguments
/// * `text` - The input text to extract entities from
///
/// # Returns
/// A vector of extracted entities with confidence scores
///
/// # Errors
/// Returns `Error::LLMFailed` if the LLM call fails
/// Returns `Error::TextTooLong` if text exceeds maximum length
pub async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
    // Preconditions
    assert!(!text.is_empty(), "text cannot be empty");
    assert!(text.len() <= TEXT_LENGTH_BYTES_MAX);

    // Implementation...

    let entities = self.llm.call(text).await?;

    // Postconditions
    assert!(entities.len() <= ENTITIES_COUNT_MAX);

    Ok(entities)
}
```

---

## Testing Guidelines

### Test Naming

```rust
#[test]
fn test_entity_extraction_valid() { }           // Positive case
#[test]
fn test_entity_extraction_empty_text() { }      // Edge case
#[test]
fn test_entity_extraction_text_too_long() { }   // Error case
```

### Property-Based Testing

Use proptest for invariant testing:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_entity_name_roundtrip(name in "[a-zA-Z]{1,100}") {
        let entity = Entity::new(&name, 1.0).unwrap();
        let serialized = serde_json::to_string(&entity).unwrap();
        let deserialized: Entity = serde_json::from_str(&serialized).unwrap();
        assert_eq!(entity.name, deserialized.name);
    }
}
```

### DST Test Coverage

Every critical path must have DST coverage:
- [x] Entity extraction with SimLLM
- [x] Dual retrieval with SimVector
- [x] Evolution tracking with SimStorage
- [x] Memory tier operations
- [x] Storage backend fault injection
- [x] LLM provider failures

---

## Error Handling

### Error Types

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("entity name too long: {length} bytes (max {max})")]
    EntityNameTooLong { length: usize, max: usize },

    #[error("LLM call failed: {reason}")]
    LLMFailed { reason: String },

    #[error("storage read failed for key '{key}': {reason}")]
    StorageReadFailed { key: String, reason: String },
}
```

### Result Type

```rust
// All fallible operations return Result
pub type Result<T> = std::result::Result<T, Error>;
```

### Graceful Degradation

```rust
// Good - returns fallback on LLM failure
pub async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
    match self.llm.extract(text).await {
        Ok(entities) => Ok(entities),
        Err(e) => {
            tracing::warn!("LLM extraction failed: {}, returning empty", e);
            Ok(Vec::new())  // Fallback to empty list
        }
    }
}

// Bad - propagates error
pub async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
    self.llm.extract(text).await  // Fails if LLM fails
}
```

---

## Performance Guidelines

### Allocation

- Prefer stack allocation for small, fixed-size data
- Use `Bytes` for byte buffers (zero-copy slicing)
- Pool allocations for hot paths

### Async

- Use `tokio` runtime with `multi_thread` flavor for production
- Avoid blocking operations in async contexts
- Use channels for cross-task communication

### Benchmarking

```bash
# Run all benchmarks
cargo bench -p umi-memory

# Run specific benchmark
cargo bench -p umi-memory -- entity_extraction

# With baseline comparison
cargo bench -p umi-memory --bench vector_backends -- --save-baseline main
```

---

## Commit Policy: Only Working Software

**Never commit broken code.** Every commit must represent working software.

### Pre-Commit Verification

Before every commit, you MUST verify the code works:

```bash
# Rust - Required before EVERY commit
cargo fmt --all                                  # Format
cargo clippy --all-features -- -D warnings       # Lint
cargo test --all-features                        # All tests pass

# Python - Required before EVERY commit
ruff format .                                    # Format
ruff check .                                     # Lint
mypy umi/                                        # Type check
pytest -v                                        # All tests pass
```

### Why This Matters

- Every commit is a potential rollback point
- Broken commits make `git bisect` useless
- CI should never be the first place code is tested
- Other developers should be able to checkout any commit

### Commit Checklist

Before running `git commit`:

1. **Run `cargo test --all-features`** - All Rust tests must pass (currently 232 tests)
2. **Run `pytest`** - All Python tests must pass (currently 145 tests)
3. **Run `cargo clippy`** - Fix any warnings
4. **Run `ruff check .`** - Fix any lint errors
5. **Review changes** - `git diff` to verify what's being committed
6. **Write clear message** - Describe what and why, not how

### If Tests Fail

Do NOT commit. Instead:
1. Fix the failing tests
2. If the fix is complex, consider `git stash` to save work
3. Never use `--no-verify` to skip pre-commit hooks
4. Never commit with `// TODO: fix this test` comments

---

## Acceptance Criteria: No Stubs, Verification First

**Every feature must have real implementation and empirical verification.**

### No Stubs Policy

Code must be functional, not placeholder:

```rust
// FORBIDDEN - stub implementation
fn extract_entities(&self, text: &str) -> Vec<Entity> {
    Vec::new()  // TODO: implement
}

// FORBIDDEN - TODO comments as implementation
async fn recall(&self, query: &str) -> Result<Vec<Entity>> {
    // TODO: implement dual retrieval
    Ok(Vec::new())
}

// REQUIRED - real implementation or don't merge
fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
    let prompt = format!("Extract entities from: {}", text);
    let response = self.llm.call(&prompt).await?;
    self.parse_entities(&response)
}
```

### Verification-First Development

You must **empirically prove** features work before considering them done:

1. **Unit tests** - Function-level correctness
2. **Integration tests** - Component interaction
3. **Manual verification** - Actually run it and see it work
4. **DST coverage** - Behavior under faults

### Verification Checklist

Before marking any feature complete:

| Check | How to Verify |
|-------|---------------|
| Code compiles | `cargo build --all-features` |
| Tests pass | `cargo test --all-features && pytest` |
| No warnings | `cargo clippy && ruff check .` |
| Actually works | Run manual examples, see real output |
| Edge cases handled | Test with empty input, large input, malformed input |
| Errors are meaningful | Trigger errors, verify messages are actionable |

### Example: Verifying Entity Extraction

Don't just write the code. Prove it works:

```bash
# 1. Write a test script
cat > test_extraction.py <<EOF
from umi import Memory

async def main():
    memory = Memory(provider="anthropic")
    entities = await memory.remember("Alice works at Acme Corp")
    print(f"Extracted {len(entities)} entities:")
    for e in entities:
        print(f"  - {e.name} ({e.confidence:.2f})")

import asyncio
asyncio.run(main())
EOF

# 2. Run it with real LLM
export ANTHROPIC_API_KEY=sk-ant-...
python test_extraction.py

# 3. Verify output is real, not "stub response"
# Output should be: "Extracted 2 entities: Alice (0.95), Acme Corp (0.90)"

# 4. Test with edge cases
python -c "import asyncio; from umi import Memory; asyncio.run(Memory(provider='anthropic').remember(''))"  # Should handle gracefully

# 5. Test with simulation
python -c "import asyncio; from umi import Memory; print(asyncio.run(Memory(seed=42).remember('test')))"  # Should be deterministic
```

### What "Done" Means

A feature is done when:

- [ ] Implementation is complete (no TODOs, no stubs)
- [ ] Unit tests exist and pass
- [ ] Integration test exists and passes
- [ ] You have personally run it and seen it work
- [ ] Error paths have been tested
- [ ] DST coverage exists
- [ ] Documentation updated if needed

### Current Codebase Audit

Run this evaluation periodically:

```bash
# Find stubs and TODOs
grep -r "TODO" --include="*.rs" --include="*.py" umi-memory/ umi/
grep -r "unimplemented!" --include="*.rs" umi-memory/
grep -r "stub" --include="*.rs" --include="*.py" umi-memory/ umi/

# Verify all tests pass
cargo test --all-features && pytest

# Check test coverage
cargo tarpaulin --all-features --out Html
pytest --cov=umi --cov-report=html
```

---

## Documentation

### ADRs (Architecture Decision Records)

All significant architectural decisions are documented in `docs/adr/`:

- `013-llm-provider-trait.md` - LLM abstraction layer
- `014-entity-extractor.md` - Entity extraction strategy
- `015-dual-retriever.md` - Dual retrieval with RRF merging
- `016-evolution-tracker.md` - Memory relationship detection
- `017-memory-class.md` - Main Memory API design
- `018-lance-storage-backend.md` - LanceDB integration

Create new ADRs for architectural changes. Format: `NNN-short-name.md`

### Code Documentation

- All public items must have doc comments
- Include examples for complex APIs
- Document invariants and safety requirements

```rust
/// Extract entities from text using LLM.
///
/// # Examples
///
/// ```
/// use umi_memory::{EntityExtractor, SimLLMProvider, SimConfig};
///
/// let config = SimConfig::with_seed(42);
/// let extractor = EntityExtractor::new(SimLLMProvider::new(config));
/// let entities = extractor.extract("Alice works at Acme").await?;
/// assert_eq!(entities.len(), 2);
/// ```
///
/// # Errors
///
/// Returns `Error::LLMFailed` if the LLM call fails.
pub async fn extract(&self, text: &str) -> Result<Vec<Entity>> {
    // ...
}
```

---

## Git Workflow

When working on a task:

1. **Create branch** from `main`
2. **Make changes** following this guide
3. **Run full test suite**: `cargo test --all-features && pytest`
4. **Run linters**: `cargo clippy && ruff check .`
5. **Run formatters**: `cargo fmt && ruff format .`
6. **Manual verification** - Run examples, check output
7. **Commit and push** with clear message
8. **Create PR** with description

### Commit Message Format

```
feat: Add LanceDB vector backend

- Implement LanceVectorBackend with similarity search
- Add DST tests with SimConfig
- Benchmarked at ~300ms for 1M vectors
- Closes #42

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

Use conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `refactor:` for code refactoring
- `docs:` for documentation changes
- `test:` for test additions/fixes
- `chore:` for maintenance tasks

---

## Features

The `umi-memory` crate has optional features:

```toml
[dependencies]
umi-memory = { version = "0.1", features = ["lance", "postgres"] }
```

| Feature | Description |
|---------|-------------|
| `lance` | LanceDB storage backend (persistent, vector search) |
| `postgres` | Postgres storage backend |
| `anthropic` | Anthropic LLM provider |
| `openai` | OpenAI LLM provider |
| `llm-providers` | All LLM providers |

---

## Contributing

1. Read [VISION.md](./VISION.md) to understand project goals
2. Follow the guidelines in this document
3. Check [GitHub Issues](https://github.com/rita-aga/umi/issues) for open tasks
4. Create a branch, implement, test, and submit PR
5. Ensure all tests pass and linters are clean

---

## References

- [TigerStyle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md) - Engineering philosophy
- [FoundationDB Testing](https://www.foundationdb.org/files/fdb-paper.pdf) - Deterministic simulation testing
- [memU](https://github.com/mem-u/memu) - Dual retrieval inspiration
- [Mem0](https://github.com/mem0ai/mem0) - Entity extraction patterns
