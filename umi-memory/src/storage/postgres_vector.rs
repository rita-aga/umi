//! PostgreSQL + pgvector Vector Backend
//!
//! TigerStyle: Production vector storage using PostgreSQL with pgvector extension.
//!
//! # Feature Gate
//!
//! This module requires the `postgres` feature:
//! ```toml
//! [dependencies]
//! umi-memory = { version = "0.1", features = ["postgres"] }
//! ```
//!
//! # Setup
//!
//! Requires PostgreSQL with pgvector extension:
//! ```sql
//! CREATE EXTENSION IF NOT EXISTS vector;
//! ```

use async_trait::async_trait;
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;

use super::error::{StorageError, StorageResult};
use super::vector::{VectorBackend, VectorSearchResult};
use crate::constants::EMBEDDING_DIMENSIONS_COUNT;

// =============================================================================
// Constants
// =============================================================================

const TABLE_NAME: &str = "embeddings";

// =============================================================================
// PostgresVectorBackend
// =============================================================================

/// Vector storage backend using PostgreSQL + pgvector extension.
///
/// TigerStyle:
/// - Persistent vector storage with ACID guarantees
/// - Native similarity search via pgvector operators
/// - Transactional consistency
/// - Horizontal scalability
///
/// # Example
///
/// ```ignore
/// use umi_memory::storage::PostgresVectorBackend;
///
/// let backend = PostgresVectorBackend::connect("postgres://localhost/umi").await?;
/// backend.store("entity1", &embedding).await?;
/// let results = backend.search(&query_embedding, 10).await?;
/// ```
pub struct PostgresVectorBackend {
    pool: PgPool,
}

impl std::fmt::Debug for PostgresVectorBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresVectorBackend")
            .finish_non_exhaustive()
    }
}

impl PostgresVectorBackend {
    /// Connect to PostgreSQL database and initialize pgvector.
    ///
    /// # Arguments
    /// - `database_url`: PostgreSQL connection string
    ///
    /// # Errors
    /// Returns error if connection fails or pgvector is not installed.
    pub async fn connect(database_url: &str) -> StorageResult<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))?;

        let backend = Self { pool };

        // Initialize table and extension
        backend.init_table().await?;

        Ok(backend)
    }

    /// Initialize pgvector extension and embeddings table.
    async fn init_table(&self) -> StorageResult<()> {
        // Enable pgvector extension
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await
            .map_err(|e| {
                StorageError::WriteFailed(format!("Failed to create vector extension: {}", e))
            })?;

        // Create embeddings table
        let create_table_sql = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                id TEXT PRIMARY KEY,
                embedding vector({}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            "#,
            TABLE_NAME, EMBEDDING_DIMENSIONS_COUNT
        );

        sqlx::query(&create_table_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::WriteFailed(format!("Failed to create table: {}", e)))?;

        // Create index for fast similarity search (IVFFlat)
        // Note: Index creation is idempotent with IF NOT EXISTS
        let create_index_sql = format!(
            r#"
            CREATE INDEX IF NOT EXISTS idx_{}_vector
            ON {} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            "#,
            TABLE_NAME, TABLE_NAME
        );

        sqlx::query(&create_index_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::WriteFailed(format!("Failed to create index: {}", e)))?;

        Ok(())
    }
}

#[async_trait]
impl VectorBackend for PostgresVectorBackend {
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

        // Convert embedding to pgvector format (array string)
        let embedding_str = format!(
            "[{}]",
            embedding
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        // Upsert: INSERT ... ON CONFLICT DO UPDATE
        let sql = format!(
            r#"
            INSERT INTO {} (id, embedding)
            VALUES ($1, $2::vector)
            ON CONFLICT (id)
            DO UPDATE SET embedding = EXCLUDED.embedding
            "#,
            TABLE_NAME
        );

        sqlx::query(&sql)
            .bind(id)
            .bind(&embedding_str)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
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

        // Convert embedding to pgvector format
        let embedding_str = format!(
            "[{}]",
            embedding
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        // Use pgvector cosine distance operator (<=>)
        // Score = 1 - cosine_distance (to match SimVectorBackend: higher = more similar)
        let sql = format!(
            r#"
            SELECT id, 1 - (embedding <=> $1::vector) as score
            FROM {}
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            "#,
            TABLE_NAME
        );

        let rows = sqlx::query(&sql)
            .bind(&embedding_str)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        let results = rows
            .into_iter()
            .map(|row| {
                let id: String = row.get("id");
                let score: f32 = row.get::<f64, _>("score") as f32;
                VectorSearchResult { id, score }
            })
            .collect();

        Ok(results)
    }

    async fn delete(&self, id: &str) -> StorageResult<()> {
        // Precondition
        assert!(!id.is_empty(), "id must not be empty");

        let sql = format!("DELETE FROM {} WHERE id = $1", TABLE_NAME);

        sqlx::query(&sql)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }

    async fn exists(&self, id: &str) -> StorageResult<bool> {
        // Precondition
        assert!(!id.is_empty(), "id must not be empty");

        let sql = format!("SELECT EXISTS(SELECT 1 FROM {} WHERE id = $1)", TABLE_NAME);

        let exists: bool = sqlx::query_scalar(&sql)
            .bind(id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        Ok(exists)
    }

    async fn get(&self, id: &str) -> StorageResult<Option<Vec<f32>>> {
        // Precondition
        assert!(!id.is_empty(), "id must not be empty");

        let sql = format!("SELECT embedding FROM {} WHERE id = $1", TABLE_NAME);

        let row = sqlx::query(&sql)
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        match row {
            Some(row) => {
                // pgvector returns vector as string "[1.0, 2.0, ...]"
                let embedding_str: String = row.get("embedding");

                // Parse vector string to Vec<f32>
                let embedding = parse_pgvector_string(&embedding_str)
                    .map_err(|e| StorageError::DeserializationError(e))?;

                Ok(Some(embedding))
            }
            None => Ok(None),
        }
    }

    async fn count(&self) -> StorageResult<usize> {
        let sql = format!("SELECT COUNT(*) FROM {}", TABLE_NAME);

        let count: i64 = sqlx::query_scalar(&sql)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?;

        Ok(count as usize)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parse pgvector string format "[1.0, 2.0, ...]" to Vec<f32>.
fn parse_pgvector_string(s: &str) -> Result<Vec<f32>, String> {
    // Remove brackets and split by comma
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(format!("Invalid pgvector format: {}", s));
    }

    let inner = &s[1..s.len() - 1];
    let values: Result<Vec<f32>, _> = inner.split(',').map(|v| v.trim().parse::<f32>()).collect();

    values.map_err(|e| format!("Failed to parse pgvector values: {}", e))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pgvector_string() {
        let s = "[1.0, 2.0, 3.0]";
        let vec = parse_pgvector_string(s).unwrap();
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);

        let s = "[ 1.5 ,  2.5 , 3.5  ]";
        let vec = parse_pgvector_string(s).unwrap();
        assert_eq!(vec, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_parse_pgvector_string_invalid() {
        let s = "1.0, 2.0, 3.0"; // Missing brackets
        assert!(parse_pgvector_string(s).is_err());

        let s = "[1.0, 2.0, abc]"; // Invalid number
        assert!(parse_pgvector_string(s).is_err());
    }
}
