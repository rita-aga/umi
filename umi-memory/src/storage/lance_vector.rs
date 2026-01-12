//! LanceDB Vector Backend
//!
//! TigerStyle: Production vector storage using LanceDB's native vector search.
//!
//! # Feature Gate
//!
//! This module requires the `lance` feature:
//! ```toml
//! [dependencies]
//! umi-memory = { version = "0.1", features = ["lance"] }
//! ```

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use vectordb::Connection;

use super::error::{StorageError, StorageResult};
use super::vector::{VectorBackend, VectorSearchResult};
use crate::constants::EMBEDDING_DIMENSIONS_COUNT;

// =============================================================================
// Constants
// =============================================================================

const TABLE_NAME: &str = "embeddings";

// =============================================================================
// LanceVectorBackend
// =============================================================================

/// Vector storage backend using LanceDB's native vector search.
///
/// TigerStyle:
/// - Persistent vector storage to disk
/// - Native ANN (Approximate Nearest Neighbor) search
/// - Efficient batch operations
///
/// # Example
///
/// ```ignore
/// use umi_memory::storage::LanceVectorBackend;
///
/// let backend = LanceVectorBackend::connect("./data/vectors.lance").await?;
/// backend.store("entity1", &embedding).await?;
/// let results = backend.search(&query_embedding, 10).await?;
/// ```
pub struct LanceVectorBackend {
    db: Arc<dyn Connection>,
    table_name: String,
}

impl std::fmt::Debug for LanceVectorBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceVectorBackend")
            .field("table_name", &self.table_name)
            .finish_non_exhaustive()
    }
}

impl LanceVectorBackend {
    /// Connect to or create a LanceDB vector database.
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
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        Ok(tables.iter().any(|t| t == &self.table_name))
    }

    /// Get the table (does not create if missing - use create_table_with_data instead).
    async fn get_table(&self) -> StorageResult<Arc<dyn vectordb::Table>> {
        self.db
            .open_table(&self.table_name)
            .await
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))
    }

    /// Create table with first data batch (avoids empty batch statistics issues).
    async fn create_table_with_data(&self, id: &str, embedding: &[f32]) -> StorageResult<()> {
        let schema = self.create_schema();
        let batch = self.create_batch_from_embedding(id, embedding, &schema)?;
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        self.db
            .create_table(&self.table_name, Box::new(batches), None)
            .await
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }

    /// Create the Arrow schema for the embeddings table.
    fn create_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIMENSIONS_COUNT as i32,
                ),
                true,  // Make nullable to avoid Lance statistics collector issues
            ),
        ]))
    }

    /// Create a record batch from a single embedding.
    fn create_batch_from_embedding(
        &self,
        id: &str,
        embedding: &[f32],
        schema: &Arc<Schema>,
    ) -> StorageResult<RecordBatch> {
        // Precondition
        assert_eq!(
            embedding.len(),
            EMBEDDING_DIMENSIONS_COUNT,
            "embedding must have {} dimensions",
            EMBEDDING_DIMENSIONS_COUNT
        );

        let id_array = StringArray::from(vec![id.to_string()]);

        let embedding_field = Arc::new(Field::new("item", DataType::Float32, true));
        let embedding_values: ArrayRef = Arc::new(Float32Array::from(embedding.to_vec()));
        let embedding_array = FixedSizeListArray::new(
            embedding_field,
            EMBEDDING_DIMENSIONS_COUNT as i32,
            embedding_values,
            None,
        );

        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_array), Arc::new(embedding_array)],
        )
        .map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    /// Parse a row from a search result.
    fn parse_search_result(batch: &RecordBatch, row: usize) -> StorageResult<VectorSearchResult> {
        // Extract ID
        let id_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StorageError::DeserializationError("id column".to_string()))?;

        let id = id_col
            .value(row)
            .to_string();

        // Extract distance/score
        // LanceDB returns _distance column in search results
        let score = if let Some(distance_col) = batch.column_by_name("_distance") {
            let distances = distance_col
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| StorageError::DeserializationError("_distance column".to_string()))?;

            // Convert distance to similarity score (1 - distance for cosine)
            1.0 - distances.value(row)
        } else {
            // If no distance column, default to 1.0 (exact match)
            1.0
        };

        Ok(VectorSearchResult { id, score })
    }
}

#[async_trait]
impl VectorBackend for LanceVectorBackend {
    async fn store(&self, id: &str, embedding: &[f32]) -> StorageResult<()> {
        // Preconditions
        assert!(!id.is_empty(), "id must not be empty");
        assert_eq!(
            embedding.len(),
            EMBEDDING_DIMENSIONS_COUNT,
            "embedding must have {} dimensions, got {}",
            EMBEDDING_DIMENSIONS_COUNT,
            embedding.len()
        );

        // Lazy table creation: create with first data to avoid empty batch issues
        if !self.table_exists().await? {
            self.create_table_with_data(id, embedding).await?;
            return Ok(());
        }

        // Retry logic for optimistic concurrency control
        // LanceDB uses optimistic concurrency (like Git commits)
        const MAX_RETRIES: u32 = 10;
        let mut retry_count = 0;

        loop {
            let table = self.get_table().await?;

            // Delete existing entry if present (upsert behavior)
            // LanceDB .add() creates duplicates, so we must delete first
            let filter = format!("id = '{}'", id);
            let _ = table.delete(&filter).await; // Ignore error if doesn't exist

            // Add new entry
            let schema = self.create_schema();
            let batch = self.create_batch_from_embedding(id, embedding, &schema)?;
            let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

            match table.add(Box::new(batches), None).await {
                Ok(_) => return Ok(()),
                Err(e) => {
                    let err_msg = e.to_string();

                    // Retry on commit conflicts (optimistic concurrency control)
                    if err_msg.contains("Commit conflict") && retry_count < MAX_RETRIES {
                        retry_count += 1;
                        // Exponential backoff: 2^retry_count milliseconds
                        let delay_ms = 2u64.pow(retry_count);
                        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                        continue;
                    }

                    // Non-retryable error or max retries exceeded
                    return Err(StorageError::WriteFailed(err_msg));
                }
            }
        }
    }

    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> StorageResult<Vec<VectorSearchResult>> {
        // Preconditions
        assert_eq!(
            embedding.len(),
            EMBEDDING_DIMENSIONS_COUNT,
            "query embedding must have {} dimensions, got {}",
            EMBEDDING_DIMENSIONS_COUNT,
            embedding.len()
        );
        assert!(limit > 0, "limit must be positive");

        // Return empty if table doesn't exist
        if !self.table_exists().await? {
            return Ok(Vec::new());
        }

        let table = self.get_table().await?;

        // Use LanceDB's native vector search
        let results: Vec<RecordBatch> = table
            .search(embedding)
            .limit(limit)
            .execute_stream()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        let mut search_results = Vec::new();
        for batch in results {
            for row in 0..batch.num_rows() {
                search_results.push(Self::parse_search_result(&batch, row)?);
            }
        }

        // Postcondition
        debug_assert!(
            search_results.len() <= limit,
            "results must not exceed limit"
        );

        Ok(search_results)
    }

    async fn delete(&self, id: &str) -> StorageResult<()> {
        // Precondition
        assert!(!id.is_empty(), "id must not be empty");

        // Return early if table doesn't exist
        if !self.table_exists().await? {
            return Ok(());
        }

        let table = self.get_table().await?;

        // Delete by filter
        let filter = format!("id = '{}'", id);
        table
            .delete(&filter)
            .await
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }

    async fn exists(&self, id: &str) -> StorageResult<bool> {
        // Precondition
        assert!(!id.is_empty(), "id must not be empty");

        // Return false if table doesn't exist
        if !self.table_exists().await? {
            return Ok(false);
        }

        let table = self.get_table().await?;

        // Query by filter
        let filter = format!("id = '{}'", id);
        let results: Vec<RecordBatch> = table
            .query()
            .filter(filter)
            .limit(1)
            .execute_stream()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        Ok(!results.is_empty() && results[0].num_rows() > 0)
    }

    async fn get(&self, id: &str) -> StorageResult<Option<Vec<f32>>> {
        // Precondition
        assert!(!id.is_empty(), "id must not be empty");

        // Return None if table doesn't exist
        if !self.table_exists().await? {
            return Ok(None);
        }

        let table = self.get_table().await?;

        // Query by filter
        let filter = format!("id = '{}'", id);
        let results: Vec<RecordBatch> = table
            .query()
            .filter(filter)
            .limit(1)
            .execute_stream()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        if results.is_empty() || results[0].num_rows() == 0 {
            return Ok(None);
        }

        // Extract embedding
        let batch = &results[0];
        let embedding_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| StorageError::DeserializationError("embedding column".to_string()))?;

        if !embedding_col.is_null(0) {
            let values = embedding_col
                .value(0)
                .as_any()
                .downcast_ref::<Float32Array>()
                .map(|arr| arr.values().to_vec());

            return Ok(values);
        }

        Ok(None)
    }

    async fn count(&self) -> StorageResult<usize> {
        // Return 0 if table doesn't exist
        if !self.table_exists().await? {
            return Ok(0);
        }

        let table = self.get_table().await?;

        // Count rows (no filter = count all)
        let count = table
            .count_rows(None)
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        Ok(count)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_backend() -> (LanceVectorBackend, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceVectorBackend::connect(temp_dir.path().to_str().unwrap())
            .await
            .unwrap();
        (backend, temp_dir)
    }

    #[tokio::test]
    async fn test_lance_vector_store_and_search() {
        let (backend, _temp) = create_test_backend().await;

        // Store some embeddings
        let emb1 = vec![1.0; EMBEDDING_DIMENSIONS_COUNT];
        let emb2 = vec![0.5; EMBEDDING_DIMENSIONS_COUNT];

        backend.store("entity1", &emb1).await.unwrap();
        backend.store("entity2", &emb2).await.unwrap();

        // Search for similar
        let query = vec![0.9; EMBEDDING_DIMENSIONS_COUNT];
        let results = backend.search(&query, 10).await.unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 2);
    }

    #[tokio::test]
    async fn test_lance_vector_empty_search() {
        let (backend, _temp) = create_test_backend().await;

        // Search in empty table
        let query = vec![1.0; EMBEDDING_DIMENSIONS_COUNT];
        let results = backend.search(&query, 10).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_lance_vector_update() {
        let (backend, _temp) = create_test_backend().await;

        // Store initial embedding
        let emb1 = vec![1.0; EMBEDDING_DIMENSIONS_COUNT];
        backend.store("entity1", &emb1).await.unwrap();

        // Update with new embedding
        let emb2 = vec![0.5; EMBEDDING_DIMENSIONS_COUNT];
        backend.store("entity1", &emb2).await.unwrap();

        // Search should find updated version
        let results = backend.search(&emb2, 10).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    #[should_panic(expected = "id must not be empty")]
    async fn test_lance_vector_store_empty_id() {
        let (backend, _temp) = create_test_backend().await;
        let emb = vec![1.0; EMBEDDING_DIMENSIONS_COUNT];
        backend.store("", &emb).await.unwrap();
    }

    #[tokio::test]
    #[should_panic(expected = "embedding must have")]
    async fn test_lance_vector_store_wrong_dimensions() {
        let (backend, _temp) = create_test_backend().await;
        let emb = vec![1.0; 100]; // Wrong dimensions
        backend.store("entity1", &emb).await.unwrap();
    }
}
