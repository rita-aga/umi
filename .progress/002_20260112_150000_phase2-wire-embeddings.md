# Phase 2: Wire Embeddings into Memory Flow

**Task**: Add embedding generation to Memory.remember() - simulation-first
**Status**: ✅ COMPLETED
**Started**: 2026-01-12
**Completed**: 2026-01-12
**Plan File**: `002_20260112_150000_phase2-wire-embeddings.md`

---

## Context

Phase 1 built the EmbeddingProvider infrastructure. Phase 2 wires it into the Memory orchestrator so that `remember()` generates embeddings for entities.

**Critical Principle**: SIMULATION-FIRST
- Test ONLY with `SimEmbeddingProvider`
- Zero production provider integration yet
- Verify complete determinism
- All changes must be testable without external APIs

---

## Objectives

1. Add `EmbeddingProvider` generic parameter to `Memory<L, S>`
2. Generate embeddings in `remember()` for each extracted entity
3. Store embeddings with entities in storage backend
4. Add `generate_embeddings` option to `RememberOptions`
5. Implement graceful degradation on embedding failures
6. Test everything with simulation stack only

---

## Architecture Changes

### Current Memory Struct
```rust
pub struct Memory<L: LLMProvider, S: StorageBackend> {
    storage: S,
    extractor: EntityExtractor<L>,
    retriever: DualRetriever<L, S>,
    evolution: EvolutionTracker<L, S>,
}
```

### Target Memory Struct
```rust
pub struct Memory<L: LLMProvider, E: EmbeddingProvider, S: StorageBackend> {
    storage: S,
    extractor: EntityExtractor<L>,
    retriever: DualRetriever<L, S>,  // Phase 3 will add E here
    evolution: EvolutionTracker<L, S>,
    embedder: E,  // NEW
}
```

---

## Implementation Plan

### Phase 2.1: Update Memory Struct

**File**: `umi-memory/src/umi/mod.rs`

**Tasks**:
- [ ] Add `EmbeddingProvider` generic parameter
- [ ] Add `embedder: E` field to Memory struct
- [ ] Update `Memory::new()` to accept embedder
- [ ] Create helper: `Memory::with_sim()` for testing
- [ ] Update all internal methods to handle new generic

**Design Notes**:
- Keep existing `Memory::new(llm, storage)` signature? NO - breaking change is OK for pre-1.0
- Add `Memory::new(llm, embedder, storage)` - three parameters
- Helper: `Memory::with_sim(seed)` returns fully-configured simulation stack

### Phase 2.2: Update RememberOptions

**File**: `umi-memory/src/umi/mod.rs`

**Tasks**:
- [ ] Add `generate_embeddings: bool` field (default: true)
- [ ] Add `with_embeddings()` / `without_embeddings()` builders
- [ ] Document when embeddings are skipped

**Default Behavior**:
- `generate_embeddings: true` - always generate unless explicitly disabled
- Graceful degradation: if embedding fails, log warning and continue

### Phase 2.3: Wire Embedding Generation in remember()

**File**: `umi-memory/src/umi/mod.rs`

**Tasks**:
- [ ] After entity extraction, generate embeddings for each entity
- [ ] Use batch embedding when possible (if >1 entity)
- [ ] Set embedding on each entity via `entity.set_embedding()`
- [ ] Handle errors gracefully (log warning, continue without embedding)
- [ ] Store entities with embeddings in storage backend

**Algorithm**:
```rust
pub async fn remember(&mut self, text: &str, options: RememberOptions) -> Result<RememberResult> {
    // 1. Extract entities (existing)
    let entities = if options.extract_entities {
        self.extractor.extract(text, extraction_opts).await?
    } else {
        // ...
    };

    // 2. Generate embeddings (NEW)
    if options.generate_embeddings && !entities.is_empty() {
        let contents: Vec<&str> = entities.iter().map(|e| e.content.as_str()).collect();

        match self.embedder.embed_batch(&contents).await {
            Ok(embeddings) => {
                for (entity, embedding) in entities.iter_mut().zip(embeddings) {
                    entity.set_embedding(embedding);
                }
            }
            Err(e) => {
                // Graceful degradation: log warning, continue without embeddings
                tracing::warn!("Failed to generate embeddings: {}", e);
            }
        }
    }

    // 3. Track evolution (existing)
    // 4. Store entities (existing - now with embeddings!)
}
```

**Graceful Degradation Rules**:
- Embedding failure NEVER fails the entire remember() operation
- Log warnings for debugging
- Entity still gets stored (just without embedding)
- Text search still works

### Phase 2.4: Update Tests

**Tasks**:
- [ ] Update all Memory tests to use 3-parameter constructor
- [ ] Add tests for embedding generation
- [ ] Add tests for graceful degradation (embedding failure)
- [ ] Add determinism tests (same seed = same embeddings)
- [ ] Verify existing tests still pass with simulation

**New Test Cases**:
1. `test_remember_generates_embeddings()` - verify entities have embeddings
2. `test_remember_embeddings_deterministic()` - same seed = same embeddings
3. `test_remember_without_embeddings()` - options.generate_embeddings = false
4. `test_remember_embedding_failure_graceful()` - embedding fails, entity still stored
5. `test_remember_batch_embeddings()` - multiple entities use batch API

---

## Testing Strategy

### Unit Tests
- [ ] Memory::new() with SimEmbeddingProvider
- [ ] Memory::with_sim(seed) helper
- [ ] RememberOptions defaults and builders

### Integration Tests
- [ ] Full remember() flow with embeddings
- [ ] Embeddings stored in SimStorageBackend
- [ ] Embeddings retrievable from storage
- [ ] Batch embedding for multiple entities

### Property Tests
- [ ] Determinism: same seed = same embeddings in storage
- [ ] Graceful degradation: embedding failure doesn't crash
- [ ] All existing properties still hold

---

## Success Criteria

- [ ] All existing Memory tests pass (with 3-param constructor update)
- [ ] New embedding tests pass (5+ tests)
- [ ] Determinism verified: same seed = same embeddings stored
- [ ] Graceful degradation works: embedding failure logged, not fatal
- [ ] Zero production provider usage in tests
- [ ] No clippy warnings
- [ ] Documentation updated

---

## File Manifest

**Modified**:
- `umi-memory/src/umi/mod.rs` - Add embedder, update remember()

**No new files** - all changes in existing Memory module

---

## Dependencies

**None!** Everything needed exists:
- `EmbeddingProvider` trait (Phase 1)
- `SimEmbeddingProvider` (Phase 1)
- `Entity.set_embedding()` (already exists)
- `SimStorageBackend` (already exists)

---

## Phase Tracking

### Instance Log
- **Instance 1** (Primary): Working on Phase 2.1-2.4

### The 2-Action Rule
After every 2 significant operations, update this file with findings.

---

## Findings & Notes

### Pre-Implementation Review

1. **Current Memory Constructor** (umi/mod.rs:327)
   ```rust
   pub fn new(llm: L, storage: S) -> Self
   ```
   - Takes 2 parameters
   - Will need to add `embedder: E`
   - Breaking change is acceptable (pre-1.0)

2. **RememberOptions** (umi/mod.rs:114)
   ```rust
   pub struct RememberOptions {
       pub extract_entities: bool,
       pub track_evolution: bool,
       pub importance: f32,
   }
   ```
   - Will add `generate_embeddings: bool`
   - Default: true

3. **Entity Already Ready** (storage/entity.rs:294)
   ```rust
   pub fn set_embedding(&mut self, embedding: Vec<f32>)
   ```
   - Method exists and works
   - Updates timestamp automatically

4. **SimStorageBackend Ready**
   - Already stores embeddings
   - No changes needed

---

## Blockers

- None currently

---

## Notes

- Strictly simulation-first - NO production providers
- Graceful degradation is critical
- Breaking change to Memory constructor is OK
- Phase 3 will wire embeddings into DualRetriever

---

## Completion Summary

### What Was Built

1. **Memory Struct Updated**
   - Added `EmbeddingProvider` generic parameter
   - Updated constructor: `Memory::new(llm, embedder, storage)`
   - Embedder field added to struct

2. **RememberOptions Extended**
   - Added `generate_embeddings: bool` field (default: true)
   - Added `with_embeddings()` / `without_embeddings()` builders

3. **Embedding Generation in remember()**
   - Batch embedding after entity extraction
   - Graceful degradation: warnings on failure, continues without embeddings
   - Sets embedding on each entity before storage

4. **Fault Injection Infrastructure**
   - Added 5 new `FaultType` enum variants:
     - `EmbeddingTimeout`
     - `EmbeddingRateLimit`
     - `EmbeddingContextOverflow`
     - `EmbeddingInvalidResponse`
     - `EmbeddingServiceUnavailable`
   - Updated `SimEmbeddingProvider` with fault injection support
   - Added `with_faults()` constructor

5. **DST Tests (7 new tests)**
   - `test_remember_with_embedding_timeout` - 100% failure rate
   - `test_remember_with_embedding_rate_limit` - 50% failure rate  
   - `test_remember_without_embeddings_option` - disabled embeddings
   - `test_remember_embeddings_deterministic` - same seed verification
   - `test_remember_embeddings_stored` - storage and retrieval
   - `test_remember_batch_embeddings` - multiple entities
   - `test_remember_with_service_unavailable` - service down

### Test Results

- **Total tests**: 435 passing (up from 428 - added 7 DST tests)
- **DST tests**: All 7 passing
- **Determinism verified**: Same seed = same embeddings
- **Graceful degradation verified**: Embedding failures don't crash remember()
- **Zero regressions**: All existing tests still pass

### Key Findings from DST

1. **Graceful Degradation Works**
   - Embedding timeout doesn't fail remember()
   - Entity stored without embedding on failure
   - Warning logged (via tracing::warn!)

2. **Fault Injection Effective**
   - 100% failure rate: all embeddings fail gracefully
   - 50% failure rate: mix of success/failure handled correctly
   - Service unavailable: continues without embeddings

3. **Determinism Verified**
   - Same seed produces identical embeddings
   - Embeddings stored correctly in storage
   - Retrieved embeddings match generated ones

4. **Batch Embedding Works**
   - Multiple entities use batch API
   - All entities get embeddings (or none on failure)

### Verification Checklist

- [x] Memory struct updated with EmbeddingProvider generic
- [x] RememberOptions extended with generate_embeddings
- [x] Embedding generation wired into remember()
- [x] Graceful degradation on embedding failures
- [x] All existing tests updated and passing
- [x] 7 new DST tests with fault injection
- [x] Determinism verified across multiple seeds
- [x] Zero production provider integration (simulation-first!)
- [x] No clippy warnings

### Files Modified

```
umi-memory/src/umi/mod.rs          - Memory struct, remember(), 7 DST tests
umi-memory/src/embedding/sim.rs    - Fault injection support
umi-memory/src/dst/fault.rs        - 5 new FaultType variants
```

**Total changes**: ~250 lines added (including 180 lines of DST tests)

---

## DST Lessons Learned

### What DST Found

1. **Initial attempt forgot imports** - DST compile errors caught missing EMBEDDING_DIMENSIONS_COUNT
2. **Storage access pattern** - DST revealed Memory doesn't expose get_entity, must use memory.storage
3. **Fault injection return type** - should_inject_fault returns Option<FaultType>, not Option<&str>

### What DST Validated

1. **Graceful degradation** - Embedding failures logged and handled, never crash
2. **Determinism** - Same seed always produces same embeddings in storage
3. **Partial failures** - 50% failure rate correctly produces mix of embedded/non-embedded entities
4. **Batch handling** - Multiple entities correctly use batch API

### TigerStyle Validation

✅ **Simulation-First**: All tests use SimEmbeddingProvider
✅ **Fault Injection**: Comprehensive coverage of failure modes
✅ **Determinism**: Verified with multiple seeds
✅ **Graceful Degradation**: No crashes on embedding failures
✅ **Zero Production Dependencies**: No OpenAI API calls in tests

---

## Notes

- Strictly simulation-first - NO production providers used
- DST torture testing found zero crashes
- Graceful degradation works as designed
- Ready for Phase 3: Vector search in DualRetriever
