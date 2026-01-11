//! Storage Backend Trait
//!
//! TigerStyle: Abstract interface for entity storage.
//!
//! # Simulation-First
//!
//! Tests are written using SimStorageBackend before PostgresBackend.
//! All implementations must satisfy the same trait contract.

use async_trait::async_trait;

use super::entity::{Entity, EntityType};
use super::error::StorageResult;

/// Abstract storage backend for entities.
///
/// TigerStyle: All operations are async, return explicit errors.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store or update an entity.
    ///
    /// If entity with same ID exists, it is updated.
    /// Returns the entity ID.
    async fn store_entity(&self, entity: &Entity) -> StorageResult<String>;

    /// Get an entity by ID.
    ///
    /// Returns None if entity does not exist.
    async fn get_entity(&self, id: &str) -> StorageResult<Option<Entity>>;

    /// Delete an entity by ID.
    ///
    /// Returns true if entity existed and was deleted.
    async fn delete_entity(&self, id: &str) -> StorageResult<bool>;

    /// Search entities by text query.
    ///
    /// Simple text matching for SimStorageBackend.
    /// Vector/semantic search for PostgresBackend.
    async fn search(&self, query: &str, limit: usize) -> StorageResult<Vec<Entity>>;

    /// List entities with optional type filter.
    async fn list_entities(
        &self,
        entity_type: Option<EntityType>,
        limit: usize,
        offset: usize,
    ) -> StorageResult<Vec<Entity>>;

    /// Count entities with optional type filter.
    async fn count_entities(&self, entity_type: Option<EntityType>) -> StorageResult<usize>;

    /// Clear all entities.
    ///
    /// Primarily for testing.
    async fn clear(&self) -> StorageResult<()>;
}
