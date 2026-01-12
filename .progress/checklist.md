# UMI Production Readiness Checklist

Quick reference for tracking implementation progress.

---

## Phase 1: Embedding Foundation

### 1.1 EmbeddingProvider Trait
- [ ] Create `umi-memory/src/embedding/mod.rs`
- [ ] Define `EmbeddingProvider` trait
- [ ] Define `EmbeddingError` error types
- [ ] Update `lib.rs` exports

### 1.2 SimEmbeddingProvider
- [ ] Create `umi-memory/src/embedding/sim.rs`
- [ ] Implement deterministic hash-based embeddings
- [ ] Ensure same text + seed = same embedding
- [ ] Write unit tests

### 1.3 OpenAIEmbeddingProvider
- [ ] Create `umi-memory/src/embedding/openai.rs`
- [ ] Implement single text embedding
- [ ] Implement batch embedding (up to 2048 texts)
- [ ] Add retry logic with exponential backoff
- [ ] Handle rate limits gracefully
- [ ] Add `embedding-openai` feature flag
- [ ] Write integration tests (requires API key)

---

## Phase 2: Wire Embeddings into Memory

### 2.1 Update Memory Struct
- [ ] Add `EmbeddingProvider` generic parameter
- [ ] Update `Memory::new()` signature
- [ ] Create `Memory::sim(seed)` helper

### 2.2 Generate Embeddings in remember()
- [ ] Add `generate_embeddings` to `RememberOptions`
- [ ] Generate embedding for each entity's content
- [ ] Implement batch embedding for efficiency
- [ ] Add graceful degradation on embedding failure
- [ ] Write integration tests

### 2.3 Update API
- [ ] Update `RememberOptions` struct
- [ ] Document new options
- [ ] Ensure backward compatibility

---

## Phase 3: True Dual Retrieval

### 3.1 VectorSearchable Trait
- [ ] Create trait in `storage/mod.rs`
- [ ] Define `search_vector()` method
- [ ] Define `has_vector_index()` method

### 3.2 Redesign DualRetriever
- [ ] Add `EmbeddingProvider` to DualRetriever
- [ ] Implement text search path (existing)
- [ ] Implement vector search path (new)
- [ ] Update RRF merge for three sources
- [ ] Add `semantic_search` to `SearchOptions`
- [ ] Add `query_expansion` to `SearchOptions`

### 3.3 Update recall() Flow
- [ ] Generate query embedding
- [ ] Run parallel text + vector search
- [ ] Merge results with RRF
- [ ] Write integration tests

---

## Phase 4: Storage Backend Upgrades

### 4.1 SimStorageBackend + Vectors
- [ ] Add `SimVectorBackend` field
- [ ] Implement `VectorSearchable` trait
- [ ] Store embeddings on `store_entity()`
- [ ] Update existing tests

### 4.2 LanceDB Vector Search
- [ ] Implement `VectorSearchable` trait
- [ ] Use `table.search(embedding)` API
- [ ] Create vector index on embedding column
- [ ] Support hybrid search (vector + filter)
- [ ] Write persistence tests

### 4.3 PostgreSQL + pgvector
- [ ] Add pgvector extension requirement
- [ ] Update schema: `embedding vector(1536)`
- [ ] Create IVFFlat or HNSW index
- [ ] Implement `VectorSearchable` trait
- [ ] Use `<=>` cosine distance operator
- [ ] Write integration tests (requires PostgreSQL)

---

## Phase 5: Configuration & Polish

### 5.1 Memory Builder
- [ ] Create `MemoryBuilder` struct
- [ ] Implement builder pattern methods
- [ ] Add sensible defaults
- [ ] Document usage

### 5.2 MemoryConfig
- [ ] Define configuration struct
- [ ] Wire config to all components
- [ ] Add environment variable overrides
- [ ] Document all options

### 5.3 Error Handling
- [ ] Add `EmbeddingFailed` error
- [ ] Add `VectorSearchUnavailable` error
- [ ] Add `DimensionMismatch` error
- [ ] Update error documentation

---

## Phase 6: Testing & Documentation

### 6.1 Integration Tests
- [ ] Semantic search finds similar content
- [ ] Synonym search works
- [ ] Hybrid beats text-only
- [ ] Graceful degradation tests
- [ ] Determinism tests (same seed = same results)

### 6.2 Benchmarks
- [ ] Text search benchmark
- [ ] Vector search benchmark
- [ ] Hybrid search benchmark
- [ ] Embedding generation benchmark

### 6.3 Documentation
- [ ] Update README.md
- [ ] Create ADR-019: Embedding Provider
- [ ] Create ADR-020: True Dual Retrieval
- [ ] API documentation
- [ ] "Getting Started" guide

---

## Success Criteria

### Functional
- [ ] Embeddings generated for all entities
- [ ] Semantic search works by default
- [ ] Synonym search succeeds
- [ ] Hybrid outperforms text-only
- [ ] Graceful degradation works

### Performance
- [ ] Embedding: < 100ms/entity (batched)
- [ ] Vector search: < 50ms for 10K entities
- [ ] Hybrid search: < 100ms total

### Quality
- [ ] 95%+ test coverage on new code
- [ ] All existing tests pass
- [ ] Deterministic simulation mode
- [ ] No breaking API changes

---

## Notes

_Use this section to track blockers, decisions, and learnings._

### Blockers
- None currently

### Decisions Made
- TBD

### Open Questions
1. text-embedding-3-small vs large?
2. Embedding cache TTL?
3. RRF weight ratio?
4. Optimal batch size?
5. Fallback behavior on partial failure?
