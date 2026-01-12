//! LanceDB Storage Backend
//!
//! TigerStyle: Production storage using LanceDB (embedded vector database).
//!
//! # Feature Gate
//!
//! This module requires the `lance` feature:
//! ```toml
//! [dependencies]
//! umi-core = { version = "0.1", features = ["lance"] }
//! ```

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
    FixedSizeListArray,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use vectordb::Connection;

use super::backend::StorageBackend;
use super::entity::{Entity, EntityType};
use super::error::{StorageError, StorageResult};
use crate::constants::EMBEDDING_DIMENSIONS_COUNT;

// =============================================================================
// Constants
// =============================================================================

const TABLE_NAME: &str = "entities";

// =============================================================================
// LanceStorageBackend
// =============================================================================

/// Storage backend using LanceDB (embedded vector database).
///
/// TigerStyle:
/// - Persistent storage to disk
/// - Vector similarity search when embeddings available
/// - Text search via metadata filtering
///
/// # Example
///
/// ```ignore
/// use umi_memory::storage::LanceStorageBackend;
///
/// let storage = LanceStorageBackend::connect("./data/umi.lance").await?;
/// ```
pub struct LanceStorageBackend {
    db: std::sync::Arc<dyn Connection>,
    table_name: String,
}

impl std::fmt::Debug for LanceStorageBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceStorageBackend")
            .field("table_name", &self.table_name)
            .finish_non_exhaustive()
    }
}

impl LanceStorageBackend {
    /// Connect to or create a LanceDB database.
    ///
    /// # Arguments
    /// - `path`: Path to the database directory
    ///
    /// # Errors
    /// Returns error if connection fails.
    pub async fn connect(path: &str) -> StorageResult<Self> {
        Self::connect_with_table(path, TABLE_NAME).await
    }

    /// Connect with a custom table name.
    ///
    /// # Arguments
    /// - `path`: Path to the database directory
    /// - `table_name`: Name of the table to use
    ///
    /// # Errors
    /// Returns error if connection fails.
    pub async fn connect_with_table(path: &str, table_name: &str) -> StorageResult<Self> {
        let db = vectordb::connect(path)
            .await
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))?;

        Ok(Self {
            db,
            table_name: table_name.to_string(),
        })
    }

    /// Check if the table exists.
    async fn table_exists(&self) -> StorageResult<bool> {
        let tables = self
            .db
            .table_names()
            .await
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))?;
        Ok(tables.contains(&self.table_name))
    }

    /// Create the table with an initial entity.
    async fn create_table_with_entity(&self, entity: &Entity) -> StorageResult<()> {
        let batch = Self::entity_to_batch(entity)?;
        let schema = Self::entity_schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        self.db
            .create_table(&self.table_name, Box::new(batches), None)
            .await
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))?;

        Ok(())
    }

    /// Get the Arrow schema for entities.
    fn entity_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("entity_type", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("metadata_json", DataType::Utf8, true),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIMENSIONS_COUNT as i32,
                ),
                true,
            ),
            Field::new("created_at_ms", DataType::Int64, false),
            Field::new("updated_at_ms", DataType::Int64, false),
            Field::new("document_time_ms", DataType::Int64, true),
            Field::new("event_time_ms", DataType::Int64, true),
            // Source reference for multimedia content (JSON serialized)
            Field::new("source_ref_json", DataType::Utf8, true),
        ]))
    }

    /// Convert an Entity to a RecordBatch.
    fn entity_to_batch(entity: &Entity) -> StorageResult<RecordBatch> {
        let schema = Self::entity_schema();

        let id: StringArray = vec![Some(entity.id.clone())].into_iter().collect();
        let entity_type: StringArray = vec![Some(entity.entity_type.as_str().to_string())]
            .into_iter()
            .collect();
        let name: StringArray = vec![Some(entity.name.clone())].into_iter().collect();
        let content: StringArray = vec![Some(entity.content.clone())].into_iter().collect();

        let metadata_json: StringArray = vec![Some(
            serde_json::to_string(&entity.metadata)
                .map_err(|e| StorageError::SerializationError(e.to_string()))?,
        )]
        .into_iter()
        .collect();

        // Handle embedding - pad with zeros if not present or wrong size
        let embedding_vec: Vec<f32> = match &entity.embedding {
            Some(emb) if emb.len() == EMBEDDING_DIMENSIONS_COUNT => emb.clone(),
            _ => vec![0.0; EMBEDDING_DIMENSIONS_COUNT],
        };
        let embedding_field = Arc::new(Field::new("item", DataType::Float32, true));
        let embedding_values: ArrayRef = Arc::new(Float32Array::from(embedding_vec));
        let embedding = FixedSizeListArray::new(
            embedding_field,
            EMBEDDING_DIMENSIONS_COUNT as i32,
            embedding_values,
            None,
        );

        let created_at_ms: Int64Array =
            vec![Some(entity.created_at.timestamp_millis())].into_iter().collect();
        let updated_at_ms: Int64Array =
            vec![Some(entity.updated_at.timestamp_millis())].into_iter().collect();
        let document_time_ms: Int64Array = vec![entity.document_time.map(|dt| dt.timestamp_millis())]
            .into_iter()
            .collect();
        let event_time_ms: Int64Array = vec![entity.event_time.map(|dt| dt.timestamp_millis())]
            .into_iter()
            .collect();

        // Serialize source_ref as JSON (None becomes null)
        let source_ref_json: StringArray = vec![entity
            .source_ref
            .as_ref()
            .map(|sr| serde_json::to_string(sr))
            .transpose()
            .map_err(|e| StorageError::SerializationError(e.to_string()))?]
        .into_iter()
        .collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id),
                Arc::new(entity_type),
                Arc::new(name),
                Arc::new(content),
                Arc::new(metadata_json),
                Arc::new(embedding),
                Arc::new(created_at_ms),
                Arc::new(updated_at_ms),
                Arc::new(document_time_ms),
                Arc::new(event_time_ms),
                Arc::new(source_ref_json),
            ],
        )
        .map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    /// Convert a RecordBatch row to an Entity.
    fn batch_row_to_entity(batch: &RecordBatch, row: usize) -> StorageResult<Entity> {
        let id = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StorageError::DeserializationError("id column".to_string()))?
            .value(row)
            .to_string();

        let entity_type_str = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StorageError::DeserializationError("entity_type column".to_string()))?
            .value(row);
        let entity_type = EntityType::from_str(entity_type_str)
            .ok_or_else(|| StorageError::DeserializationError(format!("unknown entity type: {entity_type_str}")))?;

        let name = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StorageError::DeserializationError("name column".to_string()))?
            .value(row)
            .to_string();

        let content = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StorageError::DeserializationError("content column".to_string()))?
            .value(row)
            .to_string();

        let metadata_json = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StorageError::DeserializationError("metadata_json column".to_string()))?
            .value(row);
        let metadata: std::collections::HashMap<String, String> =
            serde_json::from_str(metadata_json)
                .map_err(|e| StorageError::DeserializationError(e.to_string()))?;

        // Extract embedding if present and non-zero
        let embedding = {
            let embedding_col = batch
                .column(5)
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeListArray>()
                .ok_or_else(|| StorageError::DeserializationError("embedding column".to_string()))?;

            if !embedding_col.is_null(row) {
                let values = embedding_col
                    .value(row)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .map(|arr| arr.values().to_vec());

                // Check if embedding is all zeros (placeholder)
                values.filter(|v| v.iter().any(|&x| x != 0.0))
            } else {
                None
            }
        };

        let created_at_ms = batch
            .column(6)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| StorageError::DeserializationError("created_at_ms column".to_string()))?
            .value(row);
        let created_at = chrono::DateTime::from_timestamp_millis(created_at_ms)
            .ok_or_else(|| StorageError::DeserializationError("invalid created_at timestamp".to_string()))?;

        let updated_at_ms = batch
            .column(7)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| StorageError::DeserializationError("updated_at_ms column".to_string()))?
            .value(row);
        let updated_at = chrono::DateTime::from_timestamp_millis(updated_at_ms)
            .ok_or_else(|| StorageError::DeserializationError("invalid updated_at timestamp".to_string()))?;

        let document_time_ms_col = batch
            .column(8)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| StorageError::DeserializationError("document_time_ms column".to_string()))?;
        let document_time = if document_time_ms_col.is_null(row) {
            None
        } else {
            chrono::DateTime::from_timestamp_millis(document_time_ms_col.value(row))
        };

        let event_time_ms_col = batch
            .column(9)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| StorageError::DeserializationError("event_time_ms column".to_string()))?;
        let event_time = if event_time_ms_col.is_null(row) {
            None
        } else {
            chrono::DateTime::from_timestamp_millis(event_time_ms_col.value(row))
        };

        // Deserialize source_ref from JSON (null becomes None)
        let source_ref = {
            let source_ref_col = batch
                .column(10)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| StorageError::DeserializationError("source_ref_json column".to_string()))?;

            if source_ref_col.is_null(row) {
                None
            } else {
                let json_str = source_ref_col.value(row);
                if json_str.is_empty() {
                    None
                } else {
                    Some(
                        serde_json::from_str(json_str)
                            .map_err(|e| StorageError::DeserializationError(e.to_string()))?,
                    )
                }
            }
        };

        Ok(Entity {
            id,
            entity_type,
            name,
            content,
            metadata,
            embedding,
            created_at,
            updated_at,
            document_time,
            event_time,
            source_ref,
        })
    }

    /// Get the table, creating it if necessary.
    async fn get_table(&self) -> StorageResult<std::sync::Arc<dyn vectordb::Table>> {
        self.db
            .open_table(&self.table_name)
            .await
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))
    }
}

#[async_trait]
impl StorageBackend for LanceStorageBackend {
    async fn store_entity(&self, entity: &Entity) -> StorageResult<String> {
        // Lazy table creation: create table with first entity if it doesn't exist
        if !self.table_exists().await? {
            self.create_table_with_entity(entity).await?;
            return Ok(entity.id.clone());
        }

        let table = self.get_table().await?;

        // Delete existing entity with same ID (upsert behavior)
        let _ = table
            .delete(&format!("id = '{}'", entity.id))
            .await;

        // Insert new entity
        let batch = Self::entity_to_batch(entity)?;
        let schema = Self::entity_schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        table
            .add(Box::new(batches), None)
            .await
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(entity.id.clone())
    }

    async fn get_entity(&self, id: &str) -> StorageResult<Option<Entity>> {
        // Return None if table doesn't exist yet
        if !self.table_exists().await? {
            return Ok(None);
        }

        let table = self.get_table().await?;

        let results: Vec<RecordBatch> = table
            .query()
            .filter(format!("id = '{id}'"))
            .execute_stream()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        if results.is_empty() || results[0].num_rows() == 0 {
            return Ok(None);
        }

        let entity = Self::batch_row_to_entity(&results[0], 0)?;
        Ok(Some(entity))
    }

    async fn delete_entity(&self, id: &str) -> StorageResult<bool> {
        // Return false if table doesn't exist yet
        if !self.table_exists().await? {
            return Ok(false);
        }

        let table = self.get_table().await?;

        // Check if entity exists first
        let exists = self.get_entity(id).await?.is_some();

        if exists {
            table
                .delete(&format!("id = '{id}'"))
                .await
                .map_err(|e| StorageError::WriteFailed(e.to_string()))?;
        }

        Ok(exists)
    }

    async fn search(&self, query: &str, limit: usize) -> StorageResult<Vec<Entity>> {
        // Return empty if table doesn't exist yet
        if !self.table_exists().await? {
            return Ok(Vec::new());
        }

        let table = self.get_table().await?;

        // Text search via filter on name and content
        // LanceDB doesn't have built-in full-text search, so we use LIKE
        let filter = format!(
            "name LIKE '%{query}%' OR content LIKE '%{query}%'"
        );

        let results: Vec<RecordBatch> = table
            .query()
            .filter(filter)
            .limit(limit)
            .execute_stream()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        let mut entities = Vec::new();
        for batch in results {
            for row in 0..batch.num_rows() {
                entities.push(Self::batch_row_to_entity(&batch, row)?);
            }
        }

        Ok(entities)
    }

    async fn list_entities(
        &self,
        entity_type: Option<EntityType>,
        limit: usize,
        offset: usize,
    ) -> StorageResult<Vec<Entity>> {
        // Return empty if table doesn't exist yet
        if !self.table_exists().await? {
            return Ok(Vec::new());
        }

        let table = self.get_table().await?;

        let mut query = table.query();

        if let Some(et) = entity_type {
            query = query.filter(format!("entity_type = '{}'", et.as_str()));
        }

        // Note: LanceDB doesn't have native offset support, so we fetch more and skip
        let results: Vec<RecordBatch> = query
            .limit(limit + offset)
            .execute_stream()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        let mut entities = Vec::new();
        let mut skipped = 0;

        for batch in results {
            for row in 0..batch.num_rows() {
                if skipped < offset {
                    skipped += 1;
                    continue;
                }
                if entities.len() >= limit {
                    break;
                }
                entities.push(Self::batch_row_to_entity(&batch, row)?);
            }
        }

        Ok(entities)
    }

    async fn count_entities(&self, entity_type: Option<EntityType>) -> StorageResult<usize> {
        // Return 0 if table doesn't exist yet
        if !self.table_exists().await? {
            return Ok(0);
        }

        let table = self.get_table().await?;

        let mut query = table.query();

        if let Some(et) = entity_type {
            query = query.filter(format!("entity_type = '{}'", et.as_str()));
        }

        let results: Vec<RecordBatch> = query
            .execute_stream()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        let count: usize = results.iter().map(|b| b.num_rows()).sum();
        Ok(count)
    }

    async fn clear(&self) -> StorageResult<()> {
        // Drop table if it exists (will be recreated lazily on next store)
        if self.table_exists().await? {
            self.db
                .drop_table(&self.table_name)
                .await
                .map_err(|e| StorageError::WriteFailed(e.to_string()))?;
        }
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::entity::SourceRef;
    use tempfile::tempdir;

    async fn create_test_storage() -> (LanceStorageBackend, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.lance");
        let storage = LanceStorageBackend::connect(path.to_str().unwrap())
            .await
            .unwrap();
        (storage, dir)
    }

    #[tokio::test]
    async fn test_connect() {
        let (storage, _dir) = create_test_storage().await;
        assert_eq!(storage.table_name, TABLE_NAME);
    }

    #[tokio::test]
    async fn test_store_and_get() {
        let (storage, _dir) = create_test_storage().await;

        let entity = Entity::new(
            EntityType::Person,
            "Alice".to_string(),
            "My friend Alice".to_string(),
        );
        let id = entity.id.clone();

        storage.store_entity(&entity).await.unwrap();

        let retrieved = storage.get_entity(&id).await.unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.name, "Alice");
        assert_eq!(retrieved.content, "My friend Alice");
        assert_eq!(retrieved.entity_type, EntityType::Person);
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let (storage, _dir) = create_test_storage().await;

        let result = storage.get_entity("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_delete() {
        let (storage, _dir) = create_test_storage().await;

        let entity = Entity::new(
            EntityType::Note,
            "Test".to_string(),
            "Test content".to_string(),
        );
        let id = entity.id.clone();

        storage.store_entity(&entity).await.unwrap();
        assert!(storage.get_entity(&id).await.unwrap().is_some());

        let deleted = storage.delete_entity(&id).await.unwrap();
        assert!(deleted);

        assert!(storage.get_entity(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        let (storage, _dir) = create_test_storage().await;

        let deleted = storage.delete_entity("nonexistent").await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_search() {
        let (storage, _dir) = create_test_storage().await;

        let alice = Entity::new(
            EntityType::Person,
            "Alice".to_string(),
            "Software engineer".to_string(),
        );
        let bob = Entity::new(
            EntityType::Person,
            "Bob".to_string(),
            "Data scientist".to_string(),
        );

        storage.store_entity(&alice).await.unwrap();
        storage.store_entity(&bob).await.unwrap();

        let results = storage.search("Alice", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Alice");

        let results = storage.search("scientist", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Bob");
    }

    #[tokio::test]
    async fn test_count() {
        let (storage, _dir) = create_test_storage().await;

        assert_eq!(storage.count_entities(None).await.unwrap(), 0);

        let entity1 = Entity::new(EntityType::Person, "Alice".to_string(), "A".to_string());
        let entity2 = Entity::new(EntityType::Note, "Note".to_string(), "B".to_string());

        storage.store_entity(&entity1).await.unwrap();
        storage.store_entity(&entity2).await.unwrap();

        assert_eq!(storage.count_entities(None).await.unwrap(), 2);
        assert_eq!(
            storage.count_entities(Some(EntityType::Person)).await.unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn test_clear() {
        let (storage, _dir) = create_test_storage().await;

        let entity = Entity::new(EntityType::Note, "Test".to_string(), "Content".to_string());
        storage.store_entity(&entity).await.unwrap();

        assert_eq!(storage.count_entities(None).await.unwrap(), 1);

        storage.clear().await.unwrap();

        assert_eq!(storage.count_entities(None).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_upsert() {
        let (storage, _dir) = create_test_storage().await;

        let mut entity = Entity::new(
            EntityType::Note,
            "Test".to_string(),
            "Original content".to_string(),
        );
        let id = entity.id.clone();

        storage.store_entity(&entity).await.unwrap();

        entity.content = "Updated content".to_string();
        storage.store_entity(&entity).await.unwrap();

        // Should still be 1 entity (upsert, not insert)
        assert_eq!(storage.count_entities(None).await.unwrap(), 1);

        let retrieved = storage.get_entity(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Updated content");
    }

    #[tokio::test]
    async fn test_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("persist.lance");
        let path_str = path.to_str().unwrap();

        // Create and store
        {
            let storage = LanceStorageBackend::connect(path_str).await.unwrap();
            let entity = Entity::new(
                EntityType::Person,
                "Alice".to_string(),
                "Persisted".to_string(),
            );
            storage.store_entity(&entity).await.unwrap();
        }

        // Reopen and verify
        {
            let storage = LanceStorageBackend::connect(path_str).await.unwrap();
            let count = storage.count_entities(None).await.unwrap();
            assert_eq!(count, 1);

            let results = storage.search("Alice", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].content, "Persisted");
        }
    }

    #[tokio::test]
    async fn test_list_with_offset() {
        let (storage, _dir) = create_test_storage().await;

        for i in 0..5 {
            let entity = Entity::new(
                EntityType::Note,
                format!("Note {i}"),
                format!("Content {i}"),
            );
            storage.store_entity(&entity).await.unwrap();
        }

        let results = storage.list_entities(None, 2, 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_source_ref_roundtrip() {
        let (storage, _dir) = create_test_storage().await;

        // Create entity with source reference
        let source_ref = SourceRef::new("file:///photos/meeting.jpg".to_string())
            .with_mime_type("image/jpeg".to_string())
            .with_size_bytes(1024)
            .with_checksum("sha256:abc123".to_string());

        let mut entity = Entity::new(
            EntityType::Note,
            "Meeting Notes".to_string(),
            "Summary of the meeting from the photo".to_string(),
        );
        entity.set_source_ref(source_ref);
        let id = entity.id.clone();

        storage.store_entity(&entity).await.unwrap();

        // Retrieve and verify source_ref is preserved
        let retrieved = storage.get_entity(&id).await.unwrap().unwrap();
        assert!(retrieved.has_source_ref());

        let sr = retrieved.source_ref().unwrap();
        assert_eq!(sr.uri, "file:///photos/meeting.jpg");
        assert_eq!(sr.mime_type, Some("image/jpeg".to_string()));
        assert_eq!(sr.size_bytes, Some(1024));
        assert_eq!(sr.checksum, Some("sha256:abc123".to_string()));
    }

    #[tokio::test]
    async fn test_source_ref_none_roundtrip() {
        let (storage, _dir) = create_test_storage().await;

        // Create entity without source reference
        let entity = Entity::new(
            EntityType::Person,
            "Alice".to_string(),
            "Friend Alice".to_string(),
        );
        let id = entity.id.clone();

        storage.store_entity(&entity).await.unwrap();

        // Retrieve and verify source_ref is None
        let retrieved = storage.get_entity(&id).await.unwrap().unwrap();
        assert!(!retrieved.has_source_ref());
        assert!(retrieved.source_ref().is_none());
    }
}
