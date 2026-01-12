# Phase 6: Testing & Validation (FINAL PHASE)

**Task**: Comprehensive testing and documentation for production readiness
**Status**: Planning
**Started**: 2026-01-12
**Plan File**: `006_20260112_210000_phase6-testing-validation.md`

---

## Context

Phases 1-5 completed the core implementation:
- ✅ Phase 1: Embedding Foundation
- ✅ Phase 2: Integrate Embeddings into Memory Flow
- ✅ Phase 3: True Dual Retrieval
- ✅ Phase 4: Storage Backend Upgrades (LanceDB + PostgreSQL/pgvector)
- ✅ Phase 5: Configuration & API Polish

Phase 6 is the **final validation phase** - comprehensive testing and documentation to ensure production readiness.

**Critical Principle**: PRAGMATIC TESTING
- Integration tests for key workflows (semantic search, evolution tracking)
- Benchmarks for performance validation
- Documentation for users and maintainers
- Not DST-first (these are integration/validation tests, not unit tests)

---

## Current State Analysis

### What Works

1. **Memory API**
   - `Memory::sim(seed)` - Deterministic testing
   - `Memory::sim_with_config(seed, config)` - Configured testing
   - `Memory::builder()` - Flexible construction
   - `remember()` and `recall()` - Core operations

2. **Test Coverage**
   - 470 tests passing (446 lib + 24 DST)
   - Unit tests for all components
   - DST tests for builder and config
   - Phase-specific DST tests (Lance, Postgres)

3. **Backends**
   - SimVectorBackend (testing)
   - LanceVectorBackend (production)
   - PostgresVectorBackend (production)

### What's Missing

1. **Integration Tests** - End-to-end workflows
2. **Benchmarks** - Performance comparison of search strategies
3. **Documentation** - Usage guides and ADRs

---

## Objectives

1. Add integration tests for key Memory workflows
2. Add benchmarks comparing search strategies
3. Update documentation (README, ADRs, examples)
4. Validate production readiness

---

## Implementation Plan

### Phase 6.1: Integration Tests

**File**: `umi-memory/tests/integration_memory.rs`

**Tests to Add**:
```rust
// Test 1: Semantic search finds similar content
#[tokio::test]
async fn test_semantic_search_finds_similar_content() {
    let mut memory = Memory::sim(42);

    // Store related information
    memory.remember("Alice is a software engineer at Acme Corp", RememberOptions::default()).await.unwrap();
    memory.remember("Bob works as a developer at TechCo", RememberOptions::default()).await.unwrap();
    memory.remember("The weather today is sunny", RememberOptions::default()).await.unwrap();

    // Semantic search should find both engineers, not weather
    let results = memory.recall("Who are the programmers?", RecallOptions::default()).await.unwrap();

    // Should find engineer-related entities
    assert!(!results.is_empty(), "Should find engineer-related results");
}

// Test 2: Memory full workflow (remember -> recall -> evolution)
#[tokio::test]
async fn test_memory_full_workflow() {
    let mut memory = Memory::sim(42);

    // Remember initial fact
    let result1 = memory.remember("Alice works at Acme Corp", RememberOptions::default()).await.unwrap();
    assert!(!result1.entities.is_empty());

    // Remember updated fact
    let result2 = memory.remember("Alice now works at TechCo", RememberOptions::default()).await.unwrap();
    assert!(!result2.entities.is_empty());

    // Should detect evolution
    // (May or may not depending on LLM - check if evolutions exist)

    // Recall should work
    let results = memory.recall("Alice", RecallOptions::default()).await.unwrap();
    assert!(!results.is_empty());
}

// Test 3: Multiple entities in single remember
#[tokio::test]
async fn test_multiple_entities_extraction() {
    let mut memory = Memory::sim(42);

    let result = memory.remember(
        "Alice and Bob work together at Acme Corp on the new project",
        RememberOptions::default()
    ).await.unwrap();

    // Should extract multiple entities
    assert!(result.entity_count() >= 2, "Should extract Alice, Bob, or Acme Corp");
}

// Test 4: Config affects behavior
#[tokio::test]
async fn test_config_without_embeddings() {
    let config = MemoryConfig::default().without_embeddings();
    let mut memory = Memory::sim_with_config(42, config);

    // Should still work (graceful degradation)
    let result = memory.remember("Test entity", RememberOptions::default()).await.unwrap();
    assert!(!result.entities.is_empty());
}
```

**Tasks**:
- [ ] Create `tests/integration_memory.rs`
- [ ] Add test for semantic search
- [ ] Add test for full workflow (remember -> recall -> evolution)
- [ ] Add test for multiple entity extraction
- [ ] Add test for config effects
- [ ] Verify all integration tests pass

---

### Phase 6.2: Benchmark Suite

**File**: `umi-memory/benches/memory.rs`

**Benchmarks to Add**:
```rust
use criterion::{criterion_group, criterion_main, Criterion};
use tokio::runtime::Runtime;
use umi_memory::umi::{Memory, RecallOptions, RememberOptions};

fn bench_remember_single_entity(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("remember_single_entity", |b| {
        b.iter(|| {
            let mut memory = Memory::sim(42);
            rt.block_on(async {
                memory.remember("Alice is an engineer", RememberOptions::default())
                    .await
                    .unwrap()
            })
        });
    });
}

fn bench_remember_multiple_entities(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("remember_multiple_entities", |b| {
        b.iter(|| {
            let mut memory = Memory::sim(42);
            rt.block_on(async {
                memory.remember(
                    "Alice and Bob work at Acme Corp on the new project",
                    RememberOptions::default()
                ).await.unwrap()
            })
        });
    });
}

fn bench_recall_with_results(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Setup: pre-populate memory
    let mut memory = Memory::sim(42);
    rt.block_on(async {
        for i in 0..100 {
            memory.remember(
                &format!("Person {} is a software engineer", i),
                RememberOptions::default()
            ).await.unwrap();
        }
    });

    c.bench_function("recall_with_100_entities", |b| {
        b.to_async(&rt).iter(|| async {
            memory.recall("engineer", RecallOptions::default())
                .await
                .unwrap()
        });
    });
}

criterion_group!(
    benches,
    bench_remember_single_entity,
    bench_remember_multiple_entities,
    bench_recall_with_results
);
criterion_main!(benches);
```

**Tasks**:
- [ ] Create `benches/memory.rs`
- [ ] Add benchmark for remember (single entity)
- [ ] Add benchmark for remember (multiple entities)
- [ ] Add benchmark for recall with pre-populated data
- [ ] Run benchmarks and document results

---

### Phase 6.3: Documentation Updates

**Tasks**:

1. **Update README.md**
   - [ ] Add quick start example with `Memory::sim()`
   - [ ] Add production example with LanceDB
   - [ ] Add configuration example
   - [ ] Document all public APIs

2. **Create ADRs**
   - [ ] ADR-019: Embedding Provider Trait (if not exists)
   - [ ] ADR-020: Memory Configuration System
   - [ ] Update existing ADRs if needed

3. **Code Documentation**
   - [ ] Ensure all public APIs have doc comments
   - [ ] Add examples to doc comments
   - [ ] Verify `cargo doc` builds without warnings

4. **Examples**
   - [ ] Create `examples/basic_usage.rs`
   - [ ] Create `examples/production_setup.rs`
   - [ ] Create `examples/configuration.rs`

---

## Success Criteria

- [ ] Integration tests pass (4+ key workflow tests)
- [ ] Benchmarks run successfully and document performance
- [ ] README updated with examples
- [ ] All public APIs documented
- [ ] Examples compile and run
- [ ] `cargo doc` builds without warnings
- [ ] All existing tests still pass (470+)

---

## File Manifest

**New Files**:
- `umi-memory/tests/integration_memory.rs` - Integration tests
- `umi-memory/benches/memory.rs` - Memory benchmarks
- `umi-memory/examples/basic_usage.rs` - Basic example
- `umi-memory/examples/production_setup.rs` - Production example
- `umi-memory/examples/configuration.rs` - Config example
- `docs/adr/019-embedding-provider-trait.md` - ADR (if needed)
- `docs/adr/020-memory-configuration.md` - ADR

**Modified**:
- `README.md` - Updated with examples and API docs
- Existing ADRs - Updated references if needed

---

## Notes

- Phase 6 is **validation**, not new features
- Focus on user-facing documentation and examples
- Benchmarks document performance, not optimize
- Integration tests validate key workflows
- This is the **FINAL PHASE** of production readiness
