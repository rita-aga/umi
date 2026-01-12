# UMI Production Readiness Plan

> A thorough plan to address architectural gaps and make UMI production-ready for semantic memory retrieval.

## Current State Assessment

### What Works
- [x] Entity extraction via LLM (Anthropic/OpenAI)
- [x] Evolution tracking (update/extend/contradict detection)
- [x] Memory tiers architecture (Core/Working/Archival)
- [x] Deterministic simulation testing (DST) framework
- [x] SimLLMProvider for testing
- [x] SimStorageBackend for testing
- [x] LanceDB persistence (text search only)
- [x] PostgreSQL backend (text search only)
- [x] VectorBackend trait with cosine similarity

### Critical Gaps
- [ ] **No embedding generation** - Entity.embedding field exists but is never populated
- [ ] **No semantic search** - All search is substring/LIKE matching
- [ ] **VectorBackend not integrated** - Exists but not wired to Memory/DualRetriever
- [ ] **"Dual retrieval" is misleading** - Just multiple text searches, not text + semantic
- [ ] **LanceDB vector search unused** - Has embedding column but searches via LIKE
- [ ] **PostgreSQL lacks pgvector** - No vector index or similarity search

---

## Phase 1: Embedding Foundation

**Goal**: Create the infrastructure to generate and store embeddings.

### 1.1 EmbeddingProvider Trait

Create `umi-memory/src/embedding/mod.rs`:

```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for a single text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    
    /// Generate embeddings for multiple texts (batch).
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    
    /// Get the embedding dimensions (e.g., 1536 for text-embedding-3-small).
    fn dimensions(&self) -> usize;
    
    /// Provider name for logging.
    fn name(&self) -> &'static str;
    
    /// Is this a simulation provider?
    fn is_simulation(&self) -> bool;
}
```

**Files to create**:
- [ ] `umi-memory/src/embedding/mod.rs` - Trait and error types
- [ ] `umi-memory/src/embedding/sim.rs` - SimEmbeddingProvider (deterministic)
- [ ] `umi-memory/src/embedding/openai.rs` - OpenAI text-embedding-3-small/large
- [ ] `umi-memory/src/embedding/cohere.rs` - Cohere embed-v3 (optional)

**Tasks**:
- [ ] Define `EmbeddingError` with variants: RateLimit, ContextOverflow, InvalidInput, NetworkError
- [ ] Implement `SimEmbeddingProvider` with deterministic hash-based embeddings
- [ ] Implement `OpenAIEmbeddingProvider` with batching and retry logic
- [ ] Add feature flags: `embedding-openai`, `embedding-cohere`
- [ ] Update `lib.rs` exports
- [ ] Write tests for each provider

### 1.2 SimEmbeddingProvider (Testing)

Deterministic embeddings for reproducible tests:

```rust
pub struct SimEmbeddingProvider {
    seed: u64,
    dimensions: usize,
}

impl SimEmbeddingProvider {
    pub fn with_seed(seed: u64) -> Self {
        Self { seed, dimensions: 1536 }
    }
}

impl EmbeddingProvider for SimEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Hash text + seed for deterministic embedding
        let mut rng = DeterministicRng::new(hash(text) ^ self.seed);
        let embedding: Vec<f32> = (0..self.dimensions)
            .map(|_| rng.next_float() as f32 * 2.0 - 1.0)
            .collect();
        
        // Normalize to unit vector
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(embedding.into_iter().map(|x| x / norm).collect())
    }
}
```

**Key property**: Same text + same seed = same embedding (deterministic).

### 1.3 OpenAIEmbeddingProvider (Production)

```rust
pub struct OpenAIEmbeddingProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,  // "text-embedding-3-small" or "text-embedding-3-large"
}

impl OpenAIEmbeddingProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, "text-embedding-3-small")
    }
    
    pub fn with_model(api_key: impl Into<String>, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.to_string(),
        }
    }
}
```

**API Details**:
- Endpoint: `https://api.openai.com/v1/embeddings`
- Model: `text-embedding-3-small` (1536 dims) or `text-embedding-3-large` (3072 dims)
- Batch size: Up to 2048 inputs per request
- Rate limits: Handle with exponential backoff

---

## Phase 2: Integrate Embeddings into Memory Flow

**Goal**: Generate embeddings when storing entities, use them for search.

### 2.1 Update Memory Struct

```rust
pub struct Memory<L: LLMProvider, E: EmbeddingProvider, S: StorageBackend> {
    storage: S,
    extractor: EntityExtractor<L>,
    retriever: DualRetriever<L, E, S>,  // Now includes embedding provider
    evolution: EvolutionTracker<L, S>,
    embedder: E,  // New field
}
```

**Tasks**:
- [ ] Add `EmbeddingProvider` generic parameter to `Memory`
- [ ] Update `Memory::new()` to accept embedding provider
- [ ] Create `Memory::new_with_sim()` helper for testing

### 2.2 Generate Embeddings in remember()

```rust
pub async fn remember(&mut self, text: &str, options: RememberOptions) -> Result<RememberResult, MemoryError> {
    // ... existing extraction logic ...
    
    // NEW: Generate embeddings for each entity
    if options.generate_embeddings {
        for entity in &mut entities {
            match self.embedder.embed(&entity.content).await {
                Ok(embedding) => entity.set_embedding(embedding),
                Err(e) => {
                    // Graceful degradation: log warning, continue without embedding
                    tracing::warn!("Failed to generate embedding for {}: {}", entity.id, e);
                }
            }
        }
    }
    
    // ... store entities ...
}
```

**Tasks**:
- [ ] Add `generate_embeddings: bool` to `RememberOptions` (default: true)
- [ ] Implement embedding generation in `remember()`
- [ ] Handle batch embedding for efficiency (multiple entities at once)
- [ ] Add graceful degradation (continue without embedding on failure)

### 2.3 Update RememberOptions

```rust
pub struct RememberOptions {
    pub extract_entities: bool,      // existing
    pub track_evolution: bool,       // existing
    pub importance: f32,             // existing
    pub generate_embeddings: bool,   // NEW - default true
}
```

---

## Phase 3: True Dual Retrieval

**Goal**: Implement actual semantic search alongside text search.

### 3.1 Redesign DualRetriever

Current (broken):
```
Query → LLM rewrites → Multiple text searches → RRF merge
```

Target (correct):
```
Query → [Text Search] + [Vector Search] → RRF merge
         ↓                    ↓
    BM25/keyword          Embedding similarity
```

### 3.2 New DualRetriever Architecture

```rust
pub struct DualRetriever<L: LLMProvider, E: EmbeddingProvider, S: StorageBackend + VectorSearchable> {
    llm: L,
    embedder: E,
    storage: S,
}

impl<L, E, S> DualRetriever<L, E, S> {
    pub async fn search(&self, query: &str, options: SearchOptions) -> Result<SearchResult, RetrievalError> {
        // 1. Fast path: Text search (always runs)
        let text_results = self.storage.search_text(query, options.limit * 2).await?;
        
        // 2. Semantic path: Vector search (if enabled)
        let vector_results = if options.semantic_search {
            let query_embedding = self.embedder.embed(query).await?;
            self.storage.search_vector(&query_embedding, options.limit * 2).await?
        } else {
            vec![]
        };
        
        // 3. LLM query expansion (optional, for complex queries)
        let expanded_results = if options.query_expansion && needs_expansion(query) {
            let variations = self.rewrite_query(query).await;
            self.search_variations(&variations, options.limit).await
        } else {
            vec![]
        };
        
        // 4. Merge using Reciprocal Rank Fusion
        let merged = self.merge_rrf(&[&text_results, &vector_results, &expanded_results]);
        
        Ok(SearchResult::new(merged, query, options))
    }
}
```

**Tasks**:
- [ ] Add `EmbeddingProvider` to `DualRetriever`
- [ ] Implement `search_text()` (existing logic)
- [ ] Implement `search_vector()` using embeddings
- [ ] Update RRF merge to handle three result sources
- [ ] Add `semantic_search: bool` to `SearchOptions` (default: true)
- [ ] Add `query_expansion: bool` to `SearchOptions` (default: auto)

### 3.3 VectorSearchable Trait

Extend storage backends with vector search capability:

```rust
#[async_trait]
pub trait VectorSearchable: StorageBackend {
    /// Search by vector similarity.
    async fn search_vector(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> StorageResult<Vec<(Entity, f32)>>;  // (entity, similarity_score)
    
    /// Check if vector search is available (has indexed embeddings).
    async fn has_vector_index(&self) -> bool;
}
```

---

## Phase 4: Storage Backend Upgrades

### 4.1 SimStorageBackend + Vector Search

Integrate existing `SimVectorBackend`:

```rust
pub struct SimStorageBackend {
    entities: Arc<RwLock<HashMap<String, Entity>>>,
    vectors: SimVectorBackend,  // NEW: for similarity search
    // ... existing fields ...
}

impl VectorSearchable for SimStorageBackend {
    async fn search_vector(&self, embedding: &[f32], limit: usize) -> StorageResult<Vec<(Entity, f32)>> {
        let vector_results = self.vectors.search(embedding, limit).await?;
        
        let entities = self.entities.read().unwrap();
        let results: Vec<(Entity, f32)> = vector_results
            .into_iter()
            .filter_map(|vr| {
                entities.get(&vr.id).map(|e| (e.clone(), vr.score))
            })
            .collect();
        
        Ok(results)
    }
}
```

**Tasks**:
- [ ] Add `SimVectorBackend` to `SimStorageBackend`
- [ ] Implement `VectorSearchable` for `SimStorageBackend`
- [ ] Update `store_entity()` to also store embedding in vector backend
- [ ] Update tests

### 4.2 LanceDB Vector Search

LanceDB natively supports vector search - just need to use it:

```rust
impl VectorSearchable for LanceStorageBackend {
    async fn search_vector(&self, embedding: &[f32], limit: usize) -> StorageResult<Vec<(Entity, f32)>> {
        let table = self.get_table().await?;
        
        // Use LanceDB's native vector search
        let results = table
            .search(embedding)
            .limit(limit)
            .execute()
            .await?;
        
        // Convert to entities with scores
        // ...
    }
}
```

**Tasks**:
- [ ] Implement `VectorSearchable` for `LanceStorageBackend`
- [ ] Use `table.search(embedding)` instead of `LIKE` filter
- [ ] Create vector index on embedding column for performance
- [ ] Add hybrid search (vector + metadata filter)

### 4.3 PostgreSQL + pgvector

Add pgvector support for production semantic search:

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Update schema
ALTER TABLE entities 
    ALTER COLUMN embedding TYPE vector(1536) 
    USING embedding::vector(1536);

-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_entities_embedding 
    ON entities USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

```rust
impl VectorSearchable for PostgresBackend {
    async fn search_vector(&self, embedding: &[f32], limit: usize) -> StorageResult<Vec<(Entity, f32)>> {
        let rows = sqlx::query(
            r#"
            SELECT *, 1 - (embedding <=> $1::vector) as similarity
            FROM entities
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            "#,
        )
        .bind(embedding)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        
        // Convert to entities with scores
        // ...
    }
}
```

**Tasks**:
- [ ] Add `pgvector` to PostgreSQL requirements
- [ ] Update schema migration to use `vector(1536)` type
- [ ] Create IVFFlat or HNSW index on embedding column
- [ ] Implement `VectorSearchable` for `PostgresBackend`
- [ ] Add `<=>` (cosine distance) search queries

---

## Phase 5: Configuration & API Polish

### 5.1 Memory Builder Pattern

Make construction cleaner:

```rust
let memory = Memory::builder()
    .with_llm(AnthropicProvider::new(api_key))
    .with_embedder(OpenAIEmbeddingProvider::new(embedding_key))
    .with_storage(LanceStorageBackend::connect("./data").await?)
    .build();
```

**Tasks**:
- [ ] Create `MemoryBuilder` struct
- [ ] Add sensible defaults
- [ ] Add `Memory::sim(seed)` for quick test setup

### 5.2 Memory Tier Configuration

```rust
pub struct MemoryConfig {
    // Core memory (always in LLM context)
    pub core_memory_bytes: usize,      // default: 32KB
    
    // Working memory (session state with TTL)
    pub working_memory_bytes: usize,   // default: 1MB
    pub working_memory_ttl: Duration,  // default: 1 hour
    
    // Archival memory (storage backend)
    pub generate_embeddings: bool,     // default: true
    pub embedding_batch_size: usize,   // default: 100
    
    // Retrieval
    pub default_recall_limit: usize,   // default: 10
    pub semantic_search_enabled: bool, // default: true
    pub query_expansion_enabled: bool, // default: auto
}
```

**Tasks**:
- [ ] Create `MemoryConfig` struct
- [ ] Wire config through to all components
- [ ] Add environment variable overrides
- [ ] Document configuration options

### 5.3 Improved Error Types

```rust
#[derive(Debug, Error)]
pub enum MemoryError {
    // Existing errors...
    
    // New embedding-related errors
    #[error("embedding generation failed: {message}")]
    EmbeddingFailed { message: String },
    
    #[error("vector search unavailable: {reason}")]
    VectorSearchUnavailable { reason: String },
    
    #[error("embedding dimensions mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}
```

---

## Phase 6: Testing & Validation

### 6.1 Integration Tests

```rust
#[tokio::test]
async fn test_semantic_search_finds_similar_content() {
    let memory = Memory::sim(42);
    
    // Store related information
    memory.remember("Alice is a software engineer at Acme Corp", default()).await?;
    memory.remember("Bob works as a developer at TechCo", default()).await?;
    memory.remember("The weather today is sunny", default()).await?;
    
    // Semantic search should find both engineers, not weather
    let results = memory.recall("Who are the programmers?", default()).await?;
    
    assert!(results.iter().any(|e| e.name.contains("Alice")));
    assert!(results.iter().any(|e| e.name.contains("Bob")));
    assert!(!results.iter().any(|e| e.content.contains("weather")));
}

#[tokio::test]
async fn test_hybrid_search_beats_text_only() {
    let memory = Memory::sim(42);
    
    // Store with synonym that won't match text search
    memory.remember("The automobile needs repair", default()).await?;
    
    // Text search: "car" won't match "automobile"
    let text_only = memory.recall("car", RecallOptions::new().text_only()).await?;
    
    // Semantic search: "car" ≈ "automobile"
    let semantic = memory.recall("car", RecallOptions::new().with_semantic()).await?;
    
    assert!(text_only.is_empty());
    assert!(!semantic.is_empty());
}
```

### 6.2 Benchmark Suite

```rust
// benches/retrieval.rs

fn bench_text_vs_semantic_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let memory = rt.block_on(setup_memory_with_1000_entities());
    
    c.bench_function("text_search_100", |b| {
        b.iter(|| rt.block_on(memory.recall("engineer", text_only())))
    });
    
    c.bench_function("semantic_search_100", |b| {
        b.iter(|| rt.block_on(memory.recall("engineer", semantic())))
    });
    
    c.bench_function("hybrid_search_100", |b| {
        b.iter(|| rt.block_on(memory.recall("engineer", hybrid())))
    });
}
```

**Tasks**:
- [ ] Add integration tests for semantic search
- [ ] Add property-based tests for embedding determinism
- [ ] Add benchmarks comparing search strategies
- [ ] Add load tests for embedding API rate limits

### 6.3 Documentation

- [ ] Update README with semantic search examples
- [ ] Create ADR-019: Embedding Provider Trait
- [ ] Create ADR-020: True Dual Retrieval
- [ ] Add API documentation for new types
- [ ] Create "Getting Started" guide with production setup

---

## Implementation Order

### Sprint 1: Foundation (Week 1-2)
1. [ ] Phase 1.1: EmbeddingProvider trait
2. [ ] Phase 1.2: SimEmbeddingProvider
3. [ ] Phase 1.3: OpenAIEmbeddingProvider
4. [ ] Phase 2.1: Update Memory struct
5. [ ] Phase 2.2: Generate embeddings in remember()

### Sprint 2: Search (Week 3-4)
6. [ ] Phase 3.1-3.2: Redesign DualRetriever
7. [ ] Phase 3.3: VectorSearchable trait
8. [ ] Phase 4.1: SimStorageBackend + vectors
9. [ ] Phase 6.1: Integration tests (sim only)

### Sprint 3: Production Storage (Week 5-6)
10. [ ] Phase 4.2: LanceDB vector search
11. [ ] Phase 4.3: PostgreSQL + pgvector
12. [ ] Phase 5.1: Memory builder pattern
13. [ ] Phase 5.2: Configuration

### Sprint 4: Polish (Week 7-8)
14. [ ] Phase 5.3: Error types
15. [ ] Phase 6.2: Benchmarks
16. [ ] Phase 6.3: Documentation
17. [ ] Release candidate testing

---

## Success Criteria

### Functional
- [ ] `memory.remember()` generates embeddings for all entities
- [ ] `memory.recall()` uses semantic similarity by default
- [ ] Synonym search works ("car" finds "automobile")
- [ ] Hybrid search outperforms text-only on semantic queries
- [ ] Graceful degradation when embedding service unavailable

### Performance
- [ ] Embedding generation: < 100ms per entity (batched)
- [ ] Vector search: < 50ms for 10K entities
- [ ] Hybrid search: < 100ms total latency
- [ ] Memory: < 10MB overhead for 1K entities

### Quality
- [ ] 95%+ test coverage on new code
- [ ] All existing tests pass
- [ ] Deterministic behavior in simulation mode
- [ ] Zero breaking changes to existing API (additive only)

---

## Dependencies

### Crates to Add
```toml
[dependencies]
# Embedding providers
reqwest = { version = "0.11", features = ["json"] }  # Already have

# pgvector support
pgvector = "0.3"  # For PostgreSQL vector type
```

### External Services
- OpenAI API key (for embeddings)
- PostgreSQL 15+ with pgvector extension (for production)

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| OpenAI rate limits | High | Medium | Implement batching, caching, exponential backoff |
| Embedding cost | Medium | Medium | Use text-embedding-3-small, cache embeddings |
| pgvector unavailable | Low | High | LanceDB as fallback, clear error messages |
| Breaking existing API | Medium | High | Additive changes only, deprecation warnings |
| Performance regression | Medium | Medium | Benchmark suite, lazy embedding generation |

---

## Open Questions

1. **Embedding model choice**: text-embedding-3-small (1536 dims) vs large (3072 dims)?
   - Recommendation: Default to small, make configurable

2. **Embedding caching**: Should we cache query embeddings?
   - Recommendation: Yes, with configurable TTL

3. **Hybrid search weights**: How to balance text vs semantic scores in RRF?
   - Recommendation: Configurable, default 0.5/0.5

4. **Batch size for embedding API**: What's optimal?
   - Recommendation: 100 texts per batch, configurable

5. **Fallback behavior**: What if embeddings fail for some entities?
   - Recommendation: Store without embedding, text search still works
