# ADR-015: DualRetriever - Rust Port

## Status

Accepted

## Context

Porting the Python `DualRetriever` to Rust as part of the full Rust migration.
The retriever combines fast substring search with LLM-assisted query rewriting
for improved recall on complex queries.

### Requirements

1. **Simulation-first**: Works with `SimLLMProvider` and `SimStorageBackend`
2. **Generic over providers**: `DualRetriever<L: LLMProvider, S: StorageBackend>`
3. **Graceful degradation**: Returns fast results on LLM failure
4. **TigerStyle**: Preconditions, postconditions, explicit limits
5. **Parity with Python**: Same heuristics, RRF algorithm, prompt format

### Python Reference

The Python implementation provides:
- `DualRetriever` with storage, llm, seed
- `search()` with fast + deep paths
- `needs_deep_search()` heuristic routing
- `rewrite_query()` via LLM
- `merge_rrf()` Reciprocal Rank Fusion

## Decision

Create a `retrieval` module with types and retriever following Rust idioms.

### Architecture

```
umi-core/src/retrieval/
├── mod.rs       # DualRetriever + re-exports
├── types.rs     # SearchOptions, SearchResult, trigger word sets
└── prompts.rs   # Query rewrite prompt template
```

### Type Design

```rust
/// Options for search operation.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub limit: usize,
    pub deep_search: bool,
    pub time_range: Option<(u64, u64)>,  // (start_ms, end_ms)
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: RETRIEVAL_RESULTS_COUNT_DEFAULT,
            deep_search: true,
            time_range: None,
        }
    }
}

/// Result from a search operation.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub entities: Vec<Entity>,
    pub query: String,
    pub deep_search_used: bool,
    pub query_variations: Vec<String>,
}
```

### DualRetriever

Generic over both LLM and storage for sim/production flexibility:

```rust
pub struct DualRetriever<L: LLMProvider, S: StorageBackend> {
    llm: L,
    storage: S,
}

impl<L: LLMProvider, S: StorageBackend> DualRetriever<L, S> {
    pub fn new(llm: L, storage: S) -> Self;

    pub async fn search(
        &self,
        query: &str,
        options: SearchOptions,
    ) -> Result<SearchResult, RetrievalError>;

    pub fn needs_deep_search(&self, query: &str) -> bool;

    pub async fn rewrite_query(
        &self,
        query: &str,
    ) -> Vec<String>;

    pub fn merge_rrf(&self, result_lists: &[Vec<&Entity>]) -> Vec<Entity>;
}
```

### Deep Search Heuristics

Static word sets for O(1) lookup:

```rust
lazy_static! {
    static ref QUESTION_WORDS: HashSet<&'static str> = {
        ["who", "what", "when", "where", "why", "how"].into_iter().collect()
    };
    static ref RELATIONSHIP_TERMS: HashSet<&'static str> = {
        ["related", "about", "regarding", "involving", "connected"].into_iter().collect()
    };
    static ref TEMPORAL_TERMS: HashSet<&'static str> = {
        ["yesterday", "today", "last", "recent", "before", "after", "week", "month", "year"].into_iter().collect()
    };
    static ref ABSTRACT_TERMS: HashSet<&'static str> = {
        ["similar", "like", "connections", "associated", "linked"].into_iter().collect()
    };
}
```

### RRF Algorithm

Reciprocal Rank Fusion implementation:

```rust
/// Merge results using RRF.
/// Score(doc) = Σ 1 / (k + rank)
pub fn merge_rrf(&self, result_lists: &[Vec<&Entity>]) -> Vec<Entity> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut entities: HashMap<String, Entity> = HashMap::new();

    for list in result_lists {
        for (rank, entity) in list.iter().enumerate() {
            *scores.entry(entity.id.clone()).or_default() += 1.0 / (RRF_K as f64 + rank as f64);
            entities.entry(entity.id.clone()).or_insert_with(|| (*entity).clone());
        }
    }

    // Sort by score descending
    let mut sorted: Vec<_> = scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    sorted.into_iter()
        .filter_map(|(id, _)| entities.remove(&id))
        .collect()
}
```

### Graceful Degradation

On LLM failure:
1. Catch `ProviderError` from rewrite_query
2. Return original query only (no variations)
3. Deep search proceeds with just the original

This matches Python behavior and ensures search always returns results.

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum RetrievalError {
    #[error("Query is empty")]
    EmptyQuery,

    #[error("Query too long: {len} bytes (max {max})")]
    QueryTooLong { len: usize, max: usize },

    #[error("Invalid limit: {value} (must be 1-{max})")]
    InvalidLimit { value: usize, max: usize },

    #[error("Storage error: {0}")]
    Storage(#[from] storage::StorageError),
}
```

Note: LLM errors are NOT propagated - we degrade gracefully instead.

### Query Rewrite Prompt

Same structure as Python for compatibility:

```
Rewrite this search query into 2-3 variations that would help find relevant memories.

Query: {query}

Return as JSON array of strings. Example: ["variation 1", "variation 2", "variation 3"]
Only return the JSON array, nothing else.
```

### Constants

Add to `constants.rs`:

```rust
// Retrieval limits
pub const RETRIEVAL_RESULTS_COUNT_MAX: usize = 100;
pub const RETRIEVAL_RESULTS_COUNT_DEFAULT: usize = 10;
pub const RETRIEVAL_QUERY_BYTES_MAX: usize = 10_000;
pub const RETRIEVAL_QUERY_REWRITE_COUNT_MAX: usize = 3;
pub const RETRIEVAL_RRF_K: usize = 60;
```

## Consequences

### Positive

- **Type-safe**: Rust generics ensure correct provider usage
- **Testable**: Works with SimLLMProvider + SimStorageBackend
- **Resilient**: Graceful degradation on LLM failure
- **Consistent**: Same behavior as Python implementation
- **Fast heuristics**: O(1) word set lookups

### Negative

- **Two generic parameters**: More complex type signatures
- **Storage trait coupling**: Need Entity type from storage module

### Mitigations

1. Type aliases for common combinations
2. Comprehensive tests with various query types
3. Document the generic pattern clearly

## References

- Python `umi/retrieval.py`
- ADR-009: Dual Retrieval (Python design)
- ADR-013: LLM Provider Trait
- ADR-014: EntityExtractor (similar pattern)
