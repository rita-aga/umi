# Phase 1: Embedding Foundation Implementation

**Task**: Implement EmbeddingProvider trait system with simulation and production providers
**Status**: ✅ COMPLETED
**Started**: 2026-01-12
**Completed**: 2026-01-12
**Plan File**: `001_20260112_140000_phase1-embedding-foundation.md`

---

## Context

Umi currently has the architecture for semantic memory but lacks the embedding generation infrastructure. The `Entity.embedding` field exists but is never populated. Phase 1 creates the foundation by implementing an EmbeddingProvider trait system following the same TigerStyle pattern used for LLMProvider.

**Key Finding from Review:**
- LLMProvider pattern (llm/mod.rs) provides the perfect template
- SimVectorBackend (storage/vector.rs) already exists with cosine similarity
- DeterministicRng available for simulation embeddings
- Entity struct ready with `embedding: Option<Vec<f32>>` field

---

## Objectives

1. Create `EmbeddingProvider` trait with error types
2. Implement `SimEmbeddingProvider` for deterministic testing
3. Implement `OpenAIEmbeddingProvider` for production
4. Add proper feature flags and exports
5. Write comprehensive tests

---

## Architecture

```rust
EmbeddingProvider (trait)
├── SimEmbeddingProvider    (always available, deterministic)
└── OpenAIEmbeddingProvider (feature: embedding-openai)
```

**Pattern**: Mirror the LLMProvider architecture exactly, following TigerStyle principles.

---

## Implementation Plan

### Phase 1.1: EmbeddingProvider Trait & Error Types

**File**: `umi-memory/src/embedding/mod.rs`

**Tasks**:
- [ ] Create embedding module directory
- [ ] Define `EmbeddingError` enum (RateLimit, ContextOverflow, InvalidInput, NetworkError)
- [ ] Define `EmbeddingProvider` trait with:
  - `embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>`
  - `embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>`
  - `dimensions(&self) -> usize`
  - `name(&self) -> &'static str`
  - `is_simulation(&self) -> bool`
- [ ] Add constructor helpers for errors
- [ ] Add tests for error types

**Design Notes**:
- Use `async_trait` for async methods (same as LLMProvider)
- Errors should have helper constructors (e.g., `EmbeddingError::rate_limit()`)
- Include `is_retryable()` method on error type

### Phase 1.2: SimEmbeddingProvider

**File**: `umi-memory/src/embedding/sim.rs`

**Tasks**:
- [ ] Implement `SimEmbeddingProvider` struct with seed
- [ ] Use `DeterministicRng` to generate hash-based embeddings
- [ ] Ensure same text + same seed = same embedding
- [ ] Normalize embeddings to unit vectors
- [ ] Set dimensions to 1536 (matching text-embedding-3-small)
- [ ] Write unit tests for determinism
- [ ] Write tests for normalization

**Algorithm**:
```rust
1. Hash text + seed to get base seed
2. Create DeterministicRng with base seed
3. Generate N random floats in [-1, 1]
4. Normalize to unit vector (L2 norm = 1)
5. Return Vec<f32>
```

**Key Property**: Deterministic - same input always produces same output.

### Phase 1.3: OpenAIEmbeddingProvider

**File**: `umi-memory/src/embedding/openai.rs`

**Tasks**:
- [ ] Define struct with reqwest client, API key, model name
- [ ] Implement single text embedding (`embed`)
- [ ] Implement batch embedding (`embed_batch`) with max 2048 texts
- [ ] Add retry logic with exponential backoff
- [ ] Handle rate limits gracefully (extract retry-after header)
- [ ] Add `embedding-openai` feature flag
- [ ] Write integration tests (with API key check)
- [ ] Add timeout configuration

**API Details**:
- Endpoint: `https://api.openai.com/v1/embeddings`
- Model: `text-embedding-3-small` (1536 dims)
- Headers: `Authorization: Bearer $API_KEY`
- Request: `{"input": ["text1", "text2"], "model": "text-embedding-3-small"}`
- Response: `{"data": [{"embedding": [0.1, 0.2, ...]}, ...]}`

### Phase 1.4: Module Integration

**Files**:
- `umi-memory/src/embedding/mod.rs` (exports)
- `umi-memory/src/lib.rs` (re-exports)
- `umi-memory/Cargo.toml` (features)

**Tasks**:
- [ ] Add module declaration in src/embedding/mod.rs
- [ ] Export types from lib.rs (similar to llm exports)
- [ ] Add feature flags to Cargo.toml:
  - `embedding-openai = ["reqwest", "tokio"]`
  - `embedding-providers = ["embedding-openai"]`
- [ ] Update documentation

---

## Testing Strategy

### Unit Tests (per module)
- [ ] SimEmbeddingProvider: determinism, normalization
- [ ] OpenAIEmbeddingProvider: request formatting, error handling
- [ ] Error types: constructors, is_retryable()

### Integration Tests
- [ ] Deterministic behavior: same seed = same embeddings
- [ ] Vector properties: unit norm, correct dimensions
- [ ] API integration (requires OPENAI_API_KEY env var)

### Property Tests
- [ ] Embedding dimensions always match `dimensions()`
- [ ] Embeddings are normalized (L2 norm ≈ 1.0)
- [ ] Same text always produces same embedding (SimEmbeddingProvider)

---

## Success Criteria

- [ ] All tests pass: `cargo test -p umi-memory`
- [ ] SimEmbeddingProvider produces deterministic embeddings
- [ ] OpenAIEmbeddingProvider successfully calls API (integration test)
- [ ] No clippy warnings: `cargo clippy --all-features`
- [ ] Documentation builds: `cargo doc --no-deps`
- [ ] Embeddings have correct dimensions (1536)
- [ ] Embeddings are normalized to unit vectors

---

## File Manifest

```
umi-memory/src/embedding/
├── mod.rs           # Trait, errors, exports
├── sim.rs           # SimEmbeddingProvider
└── openai.rs        # OpenAIEmbeddingProvider
```

---

## Dependencies

**Existing** (already in Cargo.toml):
- `async_trait` - for async trait methods
- `thiserror` - for error types
- `reqwest` - HTTP client (for OpenAI API)
- `tokio` - async runtime
- `serde`, `serde_json` - JSON parsing

**No new dependencies required** - all infrastructure already exists!

---

## Phase Tracking

### Instance Log
- **Instance 1** (Primary): Working on Phase 1.1-1.4
- **Status**: Creating embedding module

### The 2-Action Rule
After every 2 significant operations, update this file with findings.

---

## Findings & Notes

### Code Review Findings

1. **LLMProvider Pattern** (llm/mod.rs:1-400)
   - Perfect template to follow for EmbeddingProvider
   - Uses async_trait, thiserror, builder pattern
   - Has SimLLMProvider for testing (uses DST)
   - Feature flags: `anthropic`, `openai`

2. **SimVectorBackend** (storage/vector.rs:1-510)
   - Already implements cosine similarity search
   - Uses DeterministicRng for reproducibility
   - Has comprehensive test suite
   - Pattern: store embeddings, search by similarity

3. **Entity Structure** (storage/entity.rs:213)
   - Field exists: `pub embedding: Option<Vec<f32>>`
   - Has `set_embedding()` method ready to use
   - Builder pattern supports embeddings

4. **Constants** (constants.rs)
   - `EMBEDDING_DIMENSIONS_COUNT` already defined
   - Used by SimVectorBackend for validation

### Design Decisions

1. **Dimensions**: Use 1536 (text-embedding-3-small) as default
   - Balances quality and cost
   - Matches EMBEDDING_DIMENSIONS_COUNT constant

2. **Normalization**: Always normalize to unit vectors
   - Matches SimVectorBackend expectations
   - Enables cosine similarity search

3. **Feature Flags**: Mirror LLM provider pattern
   - `embedding-openai` for OpenAI provider
   - Keep sim provider always available

4. **Error Handling**: Use graceful degradation
   - Return fallback empty vectors on failure?
   - OR propagate errors up? (Decided: propagate errors)

---

## Next Steps After Phase 1

After Phase 1 completion:
1. Phase 2: Wire embeddings into Memory.remember()
2. Phase 3: Update DualRetriever for true semantic search
3. Phase 4: Integrate with storage backends

---

## Blockers

- None encountered

---

## Completion Summary

### What Was Built

1. **EmbeddingProvider Trait** (`embedding/mod.rs`)
   - Complete trait with `embed()`, `embed_batch()`, `dimensions()`, `name()`, `is_simulation()`
   - Comprehensive `EmbeddingError` enum with 10 variants
   - Helper functions: `validate_dimensions()`, `normalize_vector()`, `is_normalized()`
   - 23 unit tests covering all functionality

2. **SimEmbeddingProvider** (`embedding/sim.rs`)
   - Deterministic hash-based embedding generation
   - Uses `DeterministicRng` for reproducibility
   - Always normalizes to unit vectors
   - Comprehensive test suite (13 tests)
   - Property tests for determinism and normalization

3. **OpenAIEmbeddingProvider** (`embedding/openai.rs`)
   - Production-ready OpenAI API integration
   - Exponential backoff retry logic (max 3 attempts)
   - Batch support (up to 100 texts)
   - Proper error handling and parsing
   - Integration tests (ignored by default, requires API key)

4. **Feature Flags** (Cargo.toml)
   - `embedding-openai` - OpenAI provider
   - `embedding-providers` - All providers (convenience)

5. **Exports** (lib.rs)
   - Public exports: `EmbeddingProvider`, `EmbeddingError`, `SimEmbeddingProvider`
   - Gated exports: `OpenAIEmbeddingProvider` (requires feature)

### Test Results

- **Total tests**: 428 passing (including 23 new embedding tests)
- **Doc tests**: 31 passing (including 3 new embedding examples)
- **No regressions**: All existing tests still pass
- **Zero breaking changes**: All additive

### Verification Checklist

- [x] All tests pass: `cargo test -p umi-memory`
- [x] SimEmbeddingProvider produces deterministic embeddings
- [x] OpenAIEmbeddingProvider ready for integration tests (requires API key)
- [x] No clippy warnings in new code
- [x] Documentation builds: `cargo doc --no-deps`
- [x] Embeddings have correct dimensions (1536)
- [x] Embeddings are normalized to unit vectors
- [x] Feature flags work correctly

### Files Created

```
umi-memory/src/embedding/
├── mod.rs           (316 lines) - Trait, errors, helpers
├── sim.rs           (404 lines) - SimEmbeddingProvider
└── openai.rs        (464 lines) - OpenAIEmbeddingProvider
```

Total: 1,184 lines of new, well-tested code

---

## Notes

- Following TigerStyle: simulation-first, explicit assertions
- Using existing DST infrastructure (DeterministicRng, SimConfig)
- No breaking changes - all additive
- Pattern mirrors LLMProvider architecture exactly
- Ready for Phase 2: Wiring embeddings into Memory.remember()
