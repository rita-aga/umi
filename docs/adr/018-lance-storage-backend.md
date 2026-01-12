# ADR-018: LanceDB Storage Backend

## Status

Accepted

## Context

Phase R1 completed the Memory orchestrator with SimStorageBackend for testing. Phase R2 requires a production-ready storage backend that:

1. Persists entities to disk
2. Supports vector similarity search (when embeddings available)
3. Supports text/metadata filtering
4. Is embeddable (no separate service)
5. Works well with Rust

### Why LanceDB?

| Feature | LanceDB | PostgreSQL+pgvector | Qdrant |
|---------|---------|---------------------|--------|
| Embedded | Yes | No (server) | No (server) |
| Rust-native | Yes | Via sqlx | Via client |
| Vector search | Native | Extension | Native |
| Zero-config | Yes | No | No |
| Local-first | Yes | No | No |

LanceDB is the best fit for Umi's local-first, agent-native design.

## Decision

Create `LanceStorageBackend` implementing `StorageBackend` trait.

### Schema Design

Map Entity to Lance table using Arrow schema:

```rust
Schema::new(vec![
    Field::new("id", DataType::Utf8, false),
    Field::new("entity_type", DataType::Utf8, false),
    Field::new("name", DataType::Utf8, false),
    Field::new("content", DataType::Utf8, false),
    Field::new("metadata", DataType::Utf8, true),  // JSON string
    Field::new("embedding", DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        EMBEDDING_DIMENSIONS_COUNT as i32,
    ), true),
    Field::new("created_at", DataType::Int64, false),  // ms since epoch
    Field::new("updated_at", DataType::Int64, false),
    Field::new("document_time", DataType::Int64, true),
    Field::new("event_time", DataType::Int64, true),
])
```

### Feature Gate

```toml
[features]
lance = ["dep:lancedb", "dep:arrow-array", "dep:arrow-schema"]
```

### API

```rust
use umi_core::storage::LanceStorageBackend;

// Create with path
let storage = LanceStorageBackend::connect("./data/umi.lance").await?;

// Use with Memory
let memory = Memory::new(llm, storage);
```

### Implementation

```rust
pub struct LanceStorageBackend {
    db: lancedb::Database,
    table: lancedb::Table,
}

impl LanceStorageBackend {
    /// Connect to or create a LanceDB database.
    pub async fn connect(path: &str) -> Result<Self, StorageError>;

    /// Connect with custom table name.
    pub async fn connect_with_table(
        path: &str,
        table_name: &str
    ) -> Result<Self, StorageError>;
}

#[async_trait]
impl StorageBackend for LanceStorageBackend {
    // Implement all trait methods
}
```

### Search Strategy

1. **With embedding**: Vector similarity search using `nearest_to()`
2. **Without embedding**: Full-text filter on name/content fields
3. **Hybrid**: Combine vector search with metadata filters

For initial implementation, use text filtering. Vector search requires:
- Embeddings to be set on entities
- Index creation on embedding column

### Error Handling

Map LanceDB errors to `StorageError`:

```rust
impl From<lancedb::Error> for StorageError {
    fn from(err: lancedb::Error) -> Self {
        StorageError::lance(err.to_string())
    }
}
```

## Consequences

### Positive

- Production-ready persistent storage
- No external services required
- Native vector search capability
- Rust-native (same language as umi-core)
- Apache Arrow format (efficient, interoperable)

### Negative

- Adds ~5MB to binary size
- Arrow dependency increases compile time
- Learning curve for Arrow data types

### Mitigations

- Feature-gated (only included when `lance` feature enabled)
- SimStorageBackend remains available for testing
- Conversion utilities hide Arrow complexity

## Testing Strategy

1. Run existing `StorageBackend` trait tests against `LanceStorageBackend`
2. Add Lance-specific tests (persistence, index creation)
3. Integration test with Memory class

```rust
#[tokio::test]
async fn test_lance_persistence() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.lance");

    // Create and store
    {
        let storage = LanceStorageBackend::connect(path.to_str().unwrap()).await.unwrap();
        let entity = Entity::new(EntityType::Person, "Alice".to_string(), "Friend".to_string());
        storage.store_entity(&entity).await.unwrap();
    }

    // Reopen and verify
    {
        let storage = LanceStorageBackend::connect(path.to_str().unwrap()).await.unwrap();
        let count = storage.count_entities(None).await.unwrap();
        assert_eq!(count, 1);
    }
}
```

## References

- [LanceDB Rust Docs](https://docs.rs/lancedb/latest/lancedb/)
- [Apache Arrow](https://arrow.apache.org/)
- ADR-017: Memory Class
