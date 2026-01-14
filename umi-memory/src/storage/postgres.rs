//! PostgresBackend - Production Storage
//!
//! TigerStyle: Real database storage with pgvector support.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     PostgresBackend                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Pool: sqlx::PgPool (connection pooling)                     │
//! │  Table: entities (id, type, name, content, metadata, ...)   │
//! │  Index: GIN on metadata, btree on entity_type               │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Schema
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS entities (
//!     id TEXT PRIMARY KEY,
//!     entity_type TEXT NOT NULL,
//!     name TEXT NOT NULL,
//!     content TEXT NOT NULL,
//!     metadata JSONB NOT NULL DEFAULT '{}',
//!     embedding REAL[],
//!     created_at TIMESTAMPTZ NOT NULL,
//!     updated_at TIMESTAMPTZ NOT NULL,
//!     document_time TIMESTAMPTZ,  -- ADR-006: When source was created
//!     event_time TIMESTAMPTZ      -- ADR-006: When event occurred
//! );
//!
//! CREATE TABLE IF NOT EXISTS evolution_relations (
//!     id TEXT PRIMARY KEY,
//!     source_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
//!     target_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
//!     evolution_type TEXT NOT NULL,
//!     reason TEXT NOT NULL DEFAULT '',
//!     confidence REAL NOT NULL,
//!     created_at TIMESTAMPTZ NOT NULL
//! );
//! ```

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use sqlx::Row;

use super::backend::StorageBackend;
use super::entity::{Entity, EntityType};
use super::error::{StorageError, StorageResult};
use super::evolution::{EvolutionRelation, EvolutionType};

// =============================================================================
// PostgresBackend
// =============================================================================

/// PostgreSQL storage backend for production use.
///
/// TigerStyle: Connection pooling, explicit schema, proper error handling.
#[derive(Clone, Debug)]
pub struct PostgresBackend {
    pool: PgPool,
}

impl PostgresBackend {
    /// Create a new PostgresBackend with connection string.
    ///
    /// # Arguments
    /// * `connection_string` - PostgreSQL connection URL
    ///
    /// # Errors
    /// Returns error if connection fails or pool cannot be created.
    ///
    /// # Example
    /// ```ignore
    /// let backend = PostgresBackend::new("postgres://user:pass@localhost/umi").await?;
    /// ```
    pub async fn new(connection_string: &str) -> StorageResult<Self> {
        // Preconditions
        assert!(
            !connection_string.is_empty(),
            "connection string cannot be empty"
        );
        assert!(
            connection_string.starts_with("postgres://")
                || connection_string.starts_with("postgresql://"),
            "connection string must be postgres URL"
        );

        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(connection_string)
            .await
            .map_err(|e| StorageError::connection(format!("failed to connect: {e}")))?;

        let backend = Self { pool };

        // Initialize schema
        backend.init_schema().await?;

        Ok(backend)
    }

    /// Create from an existing pool.
    ///
    /// Useful when sharing a pool across multiple backends.
    pub async fn from_pool(pool: PgPool) -> StorageResult<Self> {
        let backend = Self { pool };
        backend.init_schema().await?;
        Ok(backend)
    }

    /// Initialize database schema.
    async fn init_schema(&self) -> StorageResult<()> {
        // Create entities table with temporal fields (ADR-006)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}',
                embedding REAL[],
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                document_time TIMESTAMPTZ,
                event_time TIMESTAMPTZ
            );
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entities_metadata ON entities USING GIN(metadata);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_updated ON entities(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_entities_event_time ON entities(event_time);
            CREATE INDEX IF NOT EXISTS idx_entities_document_time ON entities(document_time);
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::internal(format!("failed to create entities schema: {e}")))?;

        // Add temporal columns if they don't exist (migration for existing DBs)
        sqlx::query(
            r#"
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='entities' AND column_name='document_time') THEN
                    ALTER TABLE entities ADD COLUMN document_time TIMESTAMPTZ;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='entities' AND column_name='event_time') THEN
                    ALTER TABLE entities ADD COLUMN event_time TIMESTAMPTZ;
                END IF;
            END $$;
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::internal(format!("failed to migrate temporal columns: {e}")))?;

        // Create evolution_relations table (ADR-006)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS evolution_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                evolution_type TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_evolution_source ON evolution_relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_evolution_target ON evolution_relations(target_id);
            CREATE INDEX IF NOT EXISTS idx_evolution_type ON evolution_relations(evolution_type);
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::internal(format!("failed to create evolution schema: {e}")))?;

        Ok(())
    }

    /// Get the connection pool.
    #[must_use]
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Close all connections in the pool.
    pub async fn close(&self) {
        self.pool.close().await;
    }
}

// =============================================================================
// Row Mapping
// =============================================================================

/// Parse a database row into an Entity.
fn row_to_entity(row: &PgRow) -> StorageResult<Entity> {
    let id: String = row
        .try_get("id")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let entity_type_str: String = row
        .try_get("entity_type")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let entity_type = EntityType::from_str(&entity_type_str)
        .ok_or_else(|| StorageError::internal(format!("invalid entity type: {entity_type_str}")))?;

    let name: String = row
        .try_get("name")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let content: String = row
        .try_get("content")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let metadata_json: serde_json::Value = row
        .try_get("metadata")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let metadata: HashMap<String, String> = serde_json::from_value(metadata_json)
        .map_err(|e| StorageError::internal(format!("failed to parse metadata: {e}")))?;

    let embedding: Option<Vec<f32>> = row
        .try_get("embedding")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let created_at: DateTime<Utc> = row
        .try_get("created_at")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let updated_at: DateTime<Utc> = row
        .try_get("updated_at")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    // Temporal fields (ADR-006)
    let document_time: Option<DateTime<Utc>> = row
        .try_get("document_time")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let event_time: Option<DateTime<Utc>> = row
        .try_get("event_time")
        .map_err(|e| StorageError::internal(e.to_string()))?;

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
        source_ref: None, // TODO: Add source_ref column to postgres schema
    })
}

// =============================================================================
// StorageBackend Implementation
// =============================================================================

#[async_trait]
impl StorageBackend for PostgresBackend {
    /// Store an entity, upserting if ID exists.
    async fn store_entity(&self, entity: &Entity) -> StorageResult<String> {
        // Preconditions
        assert!(!entity.id.is_empty(), "entity must have id");
        assert!(!entity.name.is_empty(), "entity must have name");

        let metadata_json = serde_json::to_value(&entity.metadata)
            .map_err(|e| StorageError::internal(format!("failed to serialize metadata: {e}")))?;

        sqlx::query(
            r#"
            INSERT INTO entities (id, entity_type, name, content, metadata, embedding, created_at, updated_at, document_time, event_time)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                entity_type = EXCLUDED.entity_type,
                name = EXCLUDED.name,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = EXCLUDED.updated_at,
                document_time = EXCLUDED.document_time,
                event_time = EXCLUDED.event_time
            "#,
        )
        .bind(&entity.id)
        .bind(entity.entity_type.as_str())
        .bind(&entity.name)
        .bind(&entity.content)
        .bind(&metadata_json)
        .bind(&entity.embedding)
        .bind(entity.created_at)
        .bind(entity.updated_at)
        .bind(entity.document_time)
        .bind(entity.event_time)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::write(format!("failed to store entity: {e}")))?;

        // Postcondition
        assert!(!entity.id.is_empty(), "stored entity must have id");

        Ok(entity.id.clone())
    }

    /// Get an entity by ID.
    async fn get_entity(&self, id: &str) -> StorageResult<Option<Entity>> {
        // Precondition
        assert!(!id.is_empty(), "id cannot be empty");

        let row = sqlx::query("SELECT * FROM entities WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StorageError::read(format!("failed to get entity: {e}")))?;

        match row {
            Some(row) => {
                let entity = row_to_entity(&row)?;
                // Postcondition
                assert_eq!(entity.id, id, "returned entity must match requested id");
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Delete an entity by ID.
    async fn delete_entity(&self, id: &str) -> StorageResult<bool> {
        // Precondition
        assert!(!id.is_empty(), "id cannot be empty");

        let result = sqlx::query("DELETE FROM entities WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::write(format!("failed to delete entity: {e}")))?;

        Ok(result.rows_affected() > 0)
    }

    /// Search entities by query (matches name or content).
    async fn search(&self, query: &str, limit: usize) -> StorageResult<Vec<Entity>> {
        // Preconditions
        assert!(limit > 0, "limit must be positive");
        assert!(limit <= 1000, "limit cannot exceed 1000");

        let pattern = format!("%{query}%");

        let rows = sqlx::query(
            r#"
            SELECT * FROM entities
            WHERE name ILIKE $1 OR content ILIKE $1
            ORDER BY updated_at DESC
            LIMIT $2
            "#,
        )
        .bind(&pattern)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::read(format!("failed to search: {e}")))?;

        let mut entities = Vec::with_capacity(rows.len());
        for row in &rows {
            entities.push(row_to_entity(row)?);
        }

        // Postcondition
        assert!(
            entities.len() <= limit,
            "result count {} exceeds limit {}",
            entities.len(),
            limit
        );

        Ok(entities)
    }

    /// List entities with optional type filter.
    async fn list_entities(
        &self,
        entity_type: Option<EntityType>,
        limit: usize,
        offset: usize,
    ) -> StorageResult<Vec<Entity>> {
        // Preconditions
        assert!(limit > 0, "limit must be positive");
        assert!(limit <= 1000, "limit cannot exceed 1000");

        let rows = match entity_type {
            Some(et) => {
                sqlx::query(
                    r#"
                    SELECT * FROM entities
                    WHERE entity_type = $1
                    ORDER BY updated_at DESC
                    LIMIT $2 OFFSET $3
                    "#,
                )
                .bind(et.as_str())
                .bind(limit as i64)
                .bind(offset as i64)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query(
                    r#"
                    SELECT * FROM entities
                    ORDER BY updated_at DESC
                    LIMIT $1 OFFSET $2
                    "#,
                )
                .bind(limit as i64)
                .bind(offset as i64)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| StorageError::read(format!("failed to list entities: {e}")))?;

        let mut entities = Vec::with_capacity(rows.len());
        for row in &rows {
            entities.push(row_to_entity(row)?);
        }

        // Postcondition
        assert!(
            entities.len() <= limit,
            "result count {} exceeds limit {}",
            entities.len(),
            limit
        );

        Ok(entities)
    }

    /// Count entities with optional type filter.
    async fn count_entities(&self, entity_type: Option<EntityType>) -> StorageResult<usize> {
        let count: i64 = match entity_type {
            Some(et) => {
                sqlx::query_scalar("SELECT COUNT(*) FROM entities WHERE entity_type = $1")
                    .bind(et.as_str())
                    .fetch_one(&self.pool)
                    .await
            }
            None => {
                sqlx::query_scalar("SELECT COUNT(*) FROM entities")
                    .fetch_one(&self.pool)
                    .await
            }
        }
        .map_err(|e| StorageError::read(format!("failed to count entities: {e}")))?;

        // Postcondition
        assert!(count >= 0, "count cannot be negative");

        Ok(count as usize)
    }

    /// Clear all entities.
    async fn clear(&self) -> StorageResult<()> {
        // Clear evolution relations first (due to foreign key references)
        sqlx::query("DELETE FROM evolution_relations")
            .execute(&self.pool)
            .await
            .map_err(|e| {
                StorageError::write(format!("failed to clear evolution relations: {e}"))
            })?;

        sqlx::query("DELETE FROM entities")
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::write(format!("failed to clear entities: {e}")))?;

        // Postcondition: verify empty
        let count = self.count_entities(None).await?;
        assert_eq!(count, 0, "table should be empty after clear");

        Ok(())
    }
}

// =============================================================================
// Evolution Relations (ADR-006)
// =============================================================================

impl PostgresBackend {
    /// Store an evolution relation.
    pub async fn store_evolution(&self, relation: &EvolutionRelation) -> StorageResult<String> {
        // Preconditions
        assert!(!relation.id.is_empty(), "relation must have id");
        assert!(
            !relation.source_id.is_empty(),
            "relation must have source_id"
        );
        assert!(
            !relation.target_id.is_empty(),
            "relation must have target_id"
        );

        sqlx::query(
            r#"
            INSERT INTO evolution_relations (id, source_id, target_id, evolution_type, reason, confidence, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (id) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                target_id = EXCLUDED.target_id,
                evolution_type = EXCLUDED.evolution_type,
                reason = EXCLUDED.reason,
                confidence = EXCLUDED.confidence
            "#,
        )
        .bind(&relation.id)
        .bind(&relation.source_id)
        .bind(&relation.target_id)
        .bind(relation.evolution_type.as_str())
        .bind(&relation.reason)
        .bind(relation.confidence)
        .bind(relation.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::write(format!("failed to store evolution: {e}")))?;

        Ok(relation.id.clone())
    }

    /// Get evolutions for a source entity (memories that evolved FROM this one).
    pub async fn get_evolutions_from(
        &self,
        source_id: &str,
    ) -> StorageResult<Vec<EvolutionRelation>> {
        // Precondition
        assert!(!source_id.is_empty(), "source_id cannot be empty");

        let rows = sqlx::query(
            "SELECT * FROM evolution_relations WHERE source_id = $1 ORDER BY created_at DESC",
        )
        .bind(source_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::read(format!("failed to get evolutions: {e}")))?;

        let mut relations = Vec::with_capacity(rows.len());
        for row in &rows {
            relations.push(row_to_evolution(row)?);
        }

        Ok(relations)
    }

    /// Get evolutions for a target entity (what this memory evolved FROM).
    pub async fn get_evolutions_to(
        &self,
        target_id: &str,
    ) -> StorageResult<Vec<EvolutionRelation>> {
        // Precondition
        assert!(!target_id.is_empty(), "target_id cannot be empty");

        let rows = sqlx::query(
            "SELECT * FROM evolution_relations WHERE target_id = $1 ORDER BY created_at DESC",
        )
        .bind(target_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::read(format!("failed to get evolutions: {e}")))?;

        let mut relations = Vec::with_capacity(rows.len());
        for row in &rows {
            relations.push(row_to_evolution(row)?);
        }

        Ok(relations)
    }

    /// Get all contradictions (high-confidence conflicts that need resolution).
    pub async fn get_contradictions(&self) -> StorageResult<Vec<EvolutionRelation>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM evolution_relations
            WHERE evolution_type = 'contradict' AND confidence >= 0.8
            ORDER BY created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::read(format!("failed to get contradictions: {e}")))?;

        let mut relations = Vec::with_capacity(rows.len());
        for row in &rows {
            relations.push(row_to_evolution(row)?);
        }

        Ok(relations)
    }

    /// Delete an evolution relation.
    pub async fn delete_evolution(&self, id: &str) -> StorageResult<bool> {
        // Precondition
        assert!(!id.is_empty(), "id cannot be empty");

        let result = sqlx::query("DELETE FROM evolution_relations WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::write(format!("failed to delete evolution: {e}")))?;

        Ok(result.rows_affected() > 0)
    }
}

/// Parse a database row into an EvolutionRelation.
fn row_to_evolution(row: &PgRow) -> StorageResult<EvolutionRelation> {
    let id: String = row
        .try_get("id")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let source_id: String = row
        .try_get("source_id")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let target_id: String = row
        .try_get("target_id")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let evolution_type_str: String = row
        .try_get("evolution_type")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let evolution_type = EvolutionType::from_str(&evolution_type_str).ok_or_else(|| {
        StorageError::internal(format!("invalid evolution type: {evolution_type_str}"))
    })?;

    let reason: String = row
        .try_get("reason")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let confidence: f32 = row
        .try_get("confidence")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    let created_at: DateTime<Utc> = row
        .try_get("created_at")
        .map_err(|e| StorageError::internal(e.to_string()))?;

    Ok(EvolutionRelation {
        id,
        source_id,
        target_id,
        evolution_type,
        reason,
        confidence,
        created_at,
    })
}

// =============================================================================
// Tests (require running Postgres)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Get test database URL from environment.
    fn test_db_url() -> Option<String> {
        env::var("TEST_POSTGRES_URL").ok()
    }

    /// Skip test if no database available.
    macro_rules! require_db {
        () => {
            match test_db_url() {
                Some(url) => url,
                None => {
                    eprintln!("Skipping test: TEST_POSTGRES_URL not set");
                    return;
                }
            }
        };
    }

    #[tokio::test]
    async fn test_postgres_backend_connection() {
        let url = require_db!();

        let backend = PostgresBackend::new(&url).await;
        assert!(backend.is_ok(), "should connect to database");

        let backend = backend.unwrap();
        backend.close().await;
    }

    #[tokio::test]
    async fn test_postgres_crud() {
        let url = require_db!();
        let backend = PostgresBackend::new(&url).await.unwrap();

        // Clear existing data
        backend.clear().await.unwrap();

        // Store
        let entity = Entity::new(
            EntityType::Person,
            "Alice".to_string(),
            "My friend Alice".to_string(),
        );
        let id = backend.store_entity(&entity).await.unwrap();
        assert_eq!(id, entity.id);

        // Get
        let retrieved = backend.get_entity(&id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.name, "Alice");
        assert_eq!(retrieved.content, "My friend Alice");

        // Delete
        let deleted = backend.delete_entity(&id).await.unwrap();
        assert!(deleted);

        // Verify deleted
        let gone = backend.get_entity(&id).await.unwrap();
        assert!(gone.is_none());

        backend.close().await;
    }

    #[tokio::test]
    async fn test_postgres_search() {
        let url = require_db!();
        let backend = PostgresBackend::new(&url).await.unwrap();
        backend.clear().await.unwrap();

        // Store test entities
        let e1 = Entity::new(
            EntityType::Person,
            "Alice".to_string(),
            "Software engineer".to_string(),
        );
        let e2 = Entity::new(
            EntityType::Person,
            "Bob".to_string(),
            "Data scientist".to_string(),
        );
        let e3 = Entity::new(
            EntityType::Project,
            "Umi".to_string(),
            "Memory system".to_string(),
        );

        backend.store_entity(&e1).await.unwrap();
        backend.store_entity(&e2).await.unwrap();
        backend.store_entity(&e3).await.unwrap();

        // Search by name
        let results = backend.search("Alice", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Alice");

        // Search by content
        let results = backend.search("engineer", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Alice");

        // Search no match
        let results = backend.search("nonexistent", 10).await.unwrap();
        assert!(results.is_empty());

        backend.close().await;
    }

    #[tokio::test]
    async fn test_postgres_list_and_count() {
        let url = require_db!();
        let backend = PostgresBackend::new(&url).await.unwrap();
        backend.clear().await.unwrap();

        // Store entities of different types
        for i in 0..5 {
            let entity = Entity::new(
                EntityType::Person,
                format!("Person{i}"),
                format!("Content{i}"),
            );
            backend.store_entity(&entity).await.unwrap();
        }
        for i in 0..3 {
            let entity = Entity::new(
                EntityType::Project,
                format!("Project{i}"),
                format!("Content{i}"),
            );
            backend.store_entity(&entity).await.unwrap();
        }

        // Count all
        let count = backend.count_entities(None).await.unwrap();
        assert_eq!(count, 8);

        // Count by type
        let person_count = backend
            .count_entities(Some(EntityType::Person))
            .await
            .unwrap();
        assert_eq!(person_count, 5);

        let project_count = backend
            .count_entities(Some(EntityType::Project))
            .await
            .unwrap();
        assert_eq!(project_count, 3);

        // List with limit and offset
        let page1 = backend.list_entities(None, 3, 0).await.unwrap();
        assert_eq!(page1.len(), 3);

        let page2 = backend.list_entities(None, 3, 3).await.unwrap();
        assert_eq!(page2.len(), 3);

        // List by type
        let persons = backend
            .list_entities(Some(EntityType::Person), 10, 0)
            .await
            .unwrap();
        assert_eq!(persons.len(), 5);

        backend.close().await;
    }

    #[tokio::test]
    async fn test_postgres_upsert() {
        let url = require_db!();
        let backend = PostgresBackend::new(&url).await.unwrap();
        backend.clear().await.unwrap();

        // Store initial
        let mut entity = Entity::new(
            EntityType::Note,
            "Test Note".to_string(),
            "Original content".to_string(),
        );
        let id = backend.store_entity(&entity).await.unwrap();

        // Update
        entity.update_content("Updated content".to_string());
        backend.store_entity(&entity).await.unwrap();

        // Verify update
        let retrieved = backend.get_entity(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Updated content");

        // Count should still be 1
        let count = backend.count_entities(None).await.unwrap();
        assert_eq!(count, 1);

        backend.close().await;
    }

    #[tokio::test]
    async fn test_postgres_metadata() {
        let url = require_db!();
        let backend = PostgresBackend::new(&url).await.unwrap();
        backend.clear().await.unwrap();

        // Store with metadata
        let entity = Entity::builder(
            EntityType::Task,
            "Complete report".to_string(),
            "Finish the quarterly report".to_string(),
        )
        .with_metadata("priority".to_string(), "high".to_string())
        .with_metadata("due".to_string(), "2026-01-15".to_string())
        .build();

        let id = backend.store_entity(&entity).await.unwrap();

        // Retrieve and verify metadata
        let retrieved = backend.get_entity(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.get_metadata("priority"), Some("high"));
        assert_eq!(retrieved.get_metadata("due"), Some("2026-01-15"));

        backend.close().await;
    }
}
