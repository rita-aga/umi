# ADR-017: Memory Class - Rust Implementation

## Status

Accepted

## Context

The Python Memory class orchestrates all Umi components:
- EntityExtractor for extracting entities from text
- DualRetriever for searching memories
- EvolutionTracker for detecting memory relationships
- Storage backend for persistence

As the final component of Phase R1, we need a Rust Memory class that:
1. Composes all existing components
2. Is generic over `LLMProvider` and `StorageBackend` traits
3. Provides simple `remember()` and `recall()` API
4. Follows simulation-first design

### Requirements

1. **Generic design**: `Memory<L: LLMProvider, S: StorageBackend>`
2. **Sim-first**: All tests use `SimLLMProvider` and `SimStorageBackend`
3. **Graceful degradation**: Component failures don't crash the orchestrator
4. **TigerStyle**: Preconditions, postconditions, explicit limits
5. **Async API**: All operations are async

## Decision

Create `umi-core/src/umi/mod.rs` module with:

### Core Types

```rust
/// Options for remember operations.
pub struct RememberOptions {
    /// Whether to extract entities using LLM (default: true)
    pub extract_entities: bool,
    /// Whether to track evolution with existing memories (default: true)
    pub track_evolution: bool,
    /// Importance score 0.0-1.0 (default: 0.5)
    pub importance: f32,
}

/// Options for recall operations.
pub struct RecallOptions {
    /// Maximum results (default: 10)
    pub limit: usize,
    /// Use LLM for deep search (default: auto based on query)
    pub deep_search: Option<bool>,
    /// Time range filter
    pub time_range: Option<(u64, u64)>,
}

/// Result of a remember operation.
pub struct RememberResult {
    /// Stored entities
    pub entities: Vec<Entity>,
    /// Evolution relations detected (if any)
    pub evolutions: Vec<EvolutionRelation>,
}
```

### Memory Class

```rust
pub struct Memory<L: LLMProvider, S: StorageBackend> {
    storage: S,
    extractor: EntityExtractor<L>,
    retriever: DualRetriever<L, S>,
    evolution: EvolutionTracker<L, S>,
}

impl<L: LLMProvider + Clone, S: StorageBackend + Clone> Memory<L, S> {
    /// Create new Memory with all components.
    pub fn new(llm: L, storage: S) -> Self;

    /// Store information in memory.
    pub async fn remember(
        &mut self,
        text: &str,
        options: RememberOptions,
    ) -> Result<RememberResult, MemoryError>;

    /// Retrieve memories matching query.
    pub async fn recall(
        &self,
        query: &str,
        options: RecallOptions,
    ) -> Result<Vec<Entity>, MemoryError>;

    /// Delete entity by ID.
    pub async fn forget(&mut self, entity_id: &str) -> Result<bool, MemoryError>;

    /// Get entity by ID.
    pub async fn get(&self, entity_id: &str) -> Result<Option<Entity>, MemoryError>;

    /// Count total entities.
    pub async fn count(&self) -> Result<usize, MemoryError>;
}
```

### Workflow

```
remember("Alice left Acme, now at StartupX")
    │
    ├─► EntityExtractor.extract()
    │       └─► [Entity { name: "Alice", content: "left Acme..." }]
    │
    ├─► For each entity:
    │       └─► Storage.store_entity()
    │       └─► EvolutionTracker.detect() (if track_evolution)
    │               └─► EvolutionRelation { type: Update, ... }
    │
    └─► RememberResult { entities: [...], evolutions: [...] }
```

### Graceful Degradation

Each component failure is handled independently:

```rust
pub async fn remember(&mut self, text: &str, options: RememberOptions)
    -> Result<RememberResult, MemoryError>
{
    let mut entities = Vec::new();
    let mut evolutions = Vec::new();

    // Extract entities (graceful: fallback to raw text)
    let extracted = if options.extract_entities {
        match self.extractor.extract(text, ExtractionOptions::default()).await {
            Ok(result) => result.entities,
            Err(_) => vec![], // Extraction failed, will use fallback
        }
    } else {
        vec![]
    };

    // If extraction returned nothing, create fallback Note
    let to_store = if extracted.is_empty() {
        vec![create_fallback_entity(text)]
    } else {
        extracted.into_iter().map(convert_to_entity).collect()
    };

    // Store each entity
    for entity in to_store {
        let stored = self.storage.store_entity(&entity).await?;

        // Track evolution (graceful: skip on failure)
        if options.track_evolution {
            if let Ok(Some(detection)) = self.evolution.detect(...).await {
                evolutions.push(detection.relation);
            }
        }

        entities.push(stored);
    }

    Ok(RememberResult { entities, evolutions })
}
```

### TigerStyle Compliance

```rust
pub async fn remember(&mut self, text: &str, options: RememberOptions)
    -> Result<RememberResult, MemoryError>
{
    // Preconditions
    debug_assert!(!text.is_empty(), "text must not be empty");
    debug_assert!(text.len() <= TEXT_BYTES_MAX, "text exceeds limit");
    debug_assert!(
        (0.0..=1.0).contains(&options.importance),
        "importance must be 0.0-1.0"
    );

    // ... implementation ...

    // Postconditions
    debug_assert!(!result.entities.is_empty(), "must store at least one entity");

    Ok(result)
}
```

## Constants

Add to `constants.rs`:

```rust
/// Maximum text size for remember operations
pub const MEMORY_TEXT_BYTES_MAX: usize = 100_000;

/// Maximum results for recall operations
pub const MEMORY_RECALL_LIMIT_MAX: usize = 100;

/// Default results for recall operations
pub const MEMORY_RECALL_LIMIT_DEFAULT: usize = 10;

/// Default importance for entities
pub const MEMORY_IMPORTANCE_DEFAULT: f32 = 0.5;
```

## Testing Strategy

All tests use simulation:

```rust
#[tokio::test]
async fn test_remember_and_recall() {
    let llm = SimLLMProvider::with_seed(42);
    let storage = SimStorageBackend::new(SimConfig::with_seed(42));
    let mut memory = Memory::new(llm, storage);

    // Remember
    let result = memory.remember(
        "Alice works at Acme Corp as an engineer",
        RememberOptions::default(),
    ).await.unwrap();

    assert!(!result.entities.is_empty());

    // Recall
    let found = memory.recall("Alice", RecallOptions::default()).await.unwrap();
    assert!(!found.is_empty());
}
```

## Consequences

### Positive

- Clean orchestration of all components
- Same API pattern as Python for familiarity
- Full DST support via simulation providers
- Type-safe, compile-time verified

### Negative

- Requires Clone bounds on providers/storage for component initialization
- More complex than simple storage-only solution

### Mitigations

- Clone bounds are already satisfied by Sim implementations
- Complexity is encapsulated; users see simple API

## References

- Python Memory class: `umi/memory.py`
- ADR-014: EntityExtractor
- ADR-015: DualRetriever
- ADR-016: EvolutionTracker
