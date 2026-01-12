//! Archival Memory - Tier 3 Long-Term Storage
//!
//! `TigerStyle`: High-level API over `StorageBackend`.
//!
//! # Simulation-First
//!
//! Tests are written BEFORE implementation. This file starts with tests
//! and minimal stubs. Implementation follows to make tests pass.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      ArchivalMemory                          │
//! │  High-level API for Tier 3 operations                       │
//! │  - remember(content, type) → stores with auto-naming        │
//! │  - recall(query, limit) → semantic search                   │
//! │  - forget(id) → delete                                      │
//! └─────────────────────────────────────────────────────────────┘
//!                               │
//!                               ↓ uses
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    StorageBackend Trait                      │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::sync::Arc;

use crate::storage::{Entity, EntityType, StorageBackend, StorageResult};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for archival memory.
#[derive(Debug, Clone)]
pub struct ArchivalMemoryConfig {
    /// Default limit for recall operations
    pub default_recall_limit: usize,
    /// Maximum content length for auto-naming
    pub auto_name_max_chars: usize,
}

impl Default for ArchivalMemoryConfig {
    fn default() -> Self {
        Self {
            default_recall_limit: 10,
            auto_name_max_chars: 50,
        }
    }
}

// =============================================================================
// Archival Memory
// =============================================================================

/// Archival Memory - Tier 3 long-term storage.
///
/// `TigerStyle`:
/// - High-level API over `StorageBackend`
/// - Automatic naming from content
/// - Generic over backend for testing
#[derive(Debug)]
pub struct ArchivalMemory<B: StorageBackend> {
    backend: Arc<B>,
    config: ArchivalMemoryConfig,
}

impl<B: StorageBackend> ArchivalMemory<B> {
    /// Create a new archival memory with the given backend.
    pub fn new(backend: B) -> Self {
        Self {
            backend: Arc::new(backend),
            config: ArchivalMemoryConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(backend: B, config: ArchivalMemoryConfig) -> Self {
        Self {
            backend: Arc::new(backend),
            config,
        }
    }

    /// Remember something - store in long-term memory.
    ///
    /// Automatically generates a name from content if not provided.
    ///
    /// # Arguments
    /// * `content` - The content to remember
    /// * `entity_type` - Type of entity (Note, Person, Project, etc.)
    /// * `name` - Optional name (auto-generated if None)
    ///
    /// # Returns
    /// The stored entity
    pub async fn remember(
        &self,
        content: &str,
        entity_type: EntityType,
        name: Option<&str>,
    ) -> StorageResult<Entity> {
        // Generate name from content if not provided
        let name = match name {
            Some(n) => n.to_string(),
            None => self.auto_name(content),
        };

        // Preconditions
        assert!(!content.is_empty(), "content must not be empty");
        assert!(!name.is_empty(), "name must not be empty");

        let entity = Entity::new(entity_type, name, content.to_string());

        self.backend.store_entity(&entity).await?;

        Ok(entity)
    }

    /// Recall memories matching a query.
    ///
    /// # Arguments
    /// * `query` - Search query
    /// * `limit` - Maximum results (None uses default)
    ///
    /// # Returns
    /// Matching entities
    pub async fn recall(&self, query: &str, limit: Option<usize>) -> StorageResult<Vec<Entity>> {
        let limit = limit.unwrap_or(self.config.default_recall_limit);
        self.backend.search(query, limit).await
    }

    /// Forget a memory by ID.
    ///
    /// # Returns
    /// True if memory existed and was forgotten
    pub async fn forget(&self, id: &str) -> StorageResult<bool> {
        self.backend.delete_entity(id).await
    }

    /// Get a specific memory by ID.
    pub async fn get(&self, id: &str) -> StorageResult<Option<Entity>> {
        self.backend.get_entity(id).await
    }

    /// List memories of a given type.
    pub async fn list(
        &self,
        entity_type: Option<EntityType>,
        limit: usize,
        offset: usize,
    ) -> StorageResult<Vec<Entity>> {
        self.backend.list_entities(entity_type, limit, offset).await
    }

    /// Count memories of a given type.
    pub async fn count(&self, entity_type: Option<EntityType>) -> StorageResult<usize> {
        self.backend.count_entities(entity_type).await
    }

    /// Update an existing memory.
    pub async fn update(&self, entity: &Entity) -> StorageResult<String> {
        self.backend.store_entity(entity).await
    }

    /// Remember with full entity builder control.
    pub async fn remember_entity(&self, entity: Entity) -> StorageResult<Entity> {
        self.backend.store_entity(&entity).await?;
        Ok(entity)
    }

    /// Generate a name from content.
    fn auto_name(&self, content: &str) -> String {
        // Take first line or first N characters
        let first_line = content.lines().next().unwrap_or(content);
        let trimmed = first_line.trim();

        if trimmed.len() <= self.config.auto_name_max_chars {
            trimmed.to_string()
        } else {
            // Truncate at word boundary if possible
            let truncated = &trimmed[..self.config.auto_name_max_chars];
            if let Some(last_space) = truncated.rfind(' ') {
                format!("{}...", &truncated[..last_space])
            } else {
                format!("{truncated}...")
            }
        }
    }

    /// Get the underlying backend (for testing).
    #[cfg(test)]
    #[must_use] pub fn backend(&self) -> &B {
        &self.backend
    }
}

// =============================================================================
// TESTS - Written FIRST (Simulation-First)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dst::SimConfig;
    use crate::storage::SimStorageBackend;

    fn create_memory() -> ArchivalMemory<SimStorageBackend> {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));
        ArchivalMemory::new(backend)
    }

    // =========================================================================
    // Remember Tests
    // =========================================================================

    #[tokio::test]
    async fn test_remember_with_name() {
        let memory = create_memory();

        let entity = memory
            .remember("Alice is my friend", EntityType::Person, Some("Alice"))
            .await
            .unwrap();

        assert_eq!(entity.name, "Alice");
        assert_eq!(entity.content, "Alice is my friend");
        assert_eq!(entity.entity_type, EntityType::Person);
    }

    #[tokio::test]
    async fn test_remember_auto_name() {
        let memory = create_memory();

        let entity = memory
            .remember(
                "This is a note about something important",
                EntityType::Note,
                None,
            )
            .await
            .unwrap();

        assert_eq!(entity.name, "This is a note about something important");
        assert_eq!(entity.entity_type, EntityType::Note);
    }

    #[tokio::test]
    async fn test_remember_auto_name_truncates() {
        let memory = create_memory();

        let long_content = "This is a very long piece of content that exceeds the maximum auto-name length and should be truncated at a word boundary";
        let entity = memory
            .remember(long_content, EntityType::Note, None)
            .await
            .unwrap();

        assert!(entity.name.len() <= 55); // 50 + "..."
        assert!(entity.name.ends_with("..."));
    }

    #[tokio::test]
    async fn test_remember_auto_name_first_line() {
        let memory = create_memory();

        let multiline = "First line title\nSecond line with more content\nThird line";
        let entity = memory
            .remember(multiline, EntityType::Note, None)
            .await
            .unwrap();

        assert_eq!(entity.name, "First line title");
    }

    // =========================================================================
    // Recall Tests
    // =========================================================================

    #[tokio::test]
    async fn test_recall_finds_matching() {
        let memory = create_memory();

        memory
            .remember(
                "Alice is a software engineer",
                EntityType::Person,
                Some("Alice"),
            )
            .await
            .unwrap();
        memory
            .remember("Bob is a designer", EntityType::Person, Some("Bob"))
            .await
            .unwrap();

        let results = memory.recall("software", None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Alice");
    }

    #[tokio::test]
    async fn test_recall_respects_limit() {
        let memory = create_memory();

        for i in 0..10 {
            memory
                .remember(&format!("Note {i} about coding"), EntityType::Note, None)
                .await
                .unwrap();
        }

        let results = memory.recall("coding", Some(3)).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_recall_empty_results() {
        let memory = create_memory();

        memory
            .remember("Something about Rust", EntityType::Note, Some("Rust"))
            .await
            .unwrap();

        let results = memory.recall("Python", None).await.unwrap();
        assert!(results.is_empty());
    }

    // =========================================================================
    // Forget Tests
    // =========================================================================

    #[tokio::test]
    async fn test_forget() {
        let memory = create_memory();

        let entity = memory
            .remember("Temporary note", EntityType::Note, Some("Temp"))
            .await
            .unwrap();

        let forgotten = memory.forget(&entity.id).await.unwrap();
        assert!(forgotten);

        let retrieved = memory.get(&entity.id).await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_forget_nonexistent() {
        let memory = create_memory();

        let forgotten = memory.forget("nonexistent-id").await.unwrap();
        assert!(!forgotten);
    }

    // =========================================================================
    // Get Tests
    // =========================================================================

    #[tokio::test]
    async fn test_get() {
        let memory = create_memory();

        let entity = memory
            .remember("Test content", EntityType::Note, Some("Test"))
            .await
            .unwrap();

        let retrieved = memory.get(&entity.id).await.unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, entity.id);
        assert_eq!(retrieved.name, "Test");
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let memory = create_memory();

        let retrieved = memory.get("nonexistent").await.unwrap();
        assert!(retrieved.is_none());
    }

    // =========================================================================
    // List and Count Tests
    // =========================================================================

    #[tokio::test]
    async fn test_list_all() {
        let memory = create_memory();

        memory
            .remember("Person 1", EntityType::Person, Some("Alice"))
            .await
            .unwrap();
        memory
            .remember("Project 1", EntityType::Project, Some("Umi"))
            .await
            .unwrap();
        memory
            .remember("Note 1", EntityType::Note, Some("Note"))
            .await
            .unwrap();

        let all = memory.list(None, 100, 0).await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn test_list_by_type() {
        let memory = create_memory();

        memory
            .remember("Alice", EntityType::Person, Some("Alice"))
            .await
            .unwrap();
        memory
            .remember("Bob", EntityType::Person, Some("Bob"))
            .await
            .unwrap();
        memory
            .remember("Umi", EntityType::Project, Some("Umi"))
            .await
            .unwrap();

        let people = memory.list(Some(EntityType::Person), 100, 0).await.unwrap();
        assert_eq!(people.len(), 2);

        let projects = memory
            .list(Some(EntityType::Project), 100, 0)
            .await
            .unwrap();
        assert_eq!(projects.len(), 1);
    }

    #[tokio::test]
    async fn test_count() {
        let memory = create_memory();

        memory
            .remember("A", EntityType::Note, Some("A"))
            .await
            .unwrap();
        memory
            .remember("B", EntityType::Note, Some("B"))
            .await
            .unwrap();
        memory
            .remember("C", EntityType::Person, Some("C"))
            .await
            .unwrap();

        assert_eq!(memory.count(None).await.unwrap(), 3);
        assert_eq!(memory.count(Some(EntityType::Note)).await.unwrap(), 2);
        assert_eq!(memory.count(Some(EntityType::Person)).await.unwrap(), 1);
    }

    // =========================================================================
    // Update Tests
    // =========================================================================

    #[tokio::test]
    async fn test_update() {
        let memory = create_memory();

        let mut entity = memory
            .remember("Original content", EntityType::Note, Some("Note"))
            .await
            .unwrap();

        entity.update_content("Updated content".to_string());
        memory.update(&entity).await.unwrap();

        let retrieved = memory.get(&entity.id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Updated content");
    }
}

// =============================================================================
// DST Tests
// =============================================================================

#[cfg(test)]
mod dst_tests {
    use super::*;
    use crate::dst::{FaultConfig, FaultType, SimConfig};
    use crate::storage::SimStorageBackend;

    #[tokio::test]
    async fn test_remember_with_fault_injection() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42))
            .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 1.0).with_filter("store"));
        let memory = ArchivalMemory::new(backend);

        let result = memory
            .remember("Test", EntityType::Note, Some("Test"))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_recall_with_fault_injection() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42))
            .with_faults(FaultConfig::new(FaultType::StorageReadFail, 1.0).with_filter("search"));
        let memory = ArchivalMemory::new(backend);

        let result = memory.recall("test", None).await;
        assert!(result.is_err());
    }
}

// =============================================================================
// Property-Based Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::dst::{
        DeterministicRng, PropertyTest, PropertyTestable, SimClock, SimConfig, TimeAdvanceConfig,
    };
    use crate::storage::SimStorageBackend;

    /// Operations on `ArchivalMemory`
    #[derive(Debug, Clone)]
    enum MemoryOp {
        Remember {
            content: String,
            entity_type: EntityType,
        },
        Recall {
            query: String,
        },
        Forget {
            id: String,
        },
        Get {
            id: String,
        },
        Count,
    }

    struct MemoryWrapper {
        memory: ArchivalMemory<SimStorageBackend>,
        known_ids: Vec<String>,
    }

    impl PropertyTestable for MemoryWrapper {
        type Operation = MemoryOp;

        fn generate_operation(&self, rng: &mut DeterministicRng) -> Self::Operation {
            let op_type = rng.next_usize(0, 4);

            match op_type {
                0 => {
                    let types = EntityType::all();
                    let type_idx = rng.next_usize(0, types.len() - 1);
                    MemoryOp::Remember {
                        content: format!("Content {}", rng.next_usize(0, 999)),
                        entity_type: types[type_idx],
                    }
                }
                1 => MemoryOp::Recall {
                    query: format!("Content {}", rng.next_usize(0, 9)),
                },
                2 => {
                    let id = if !self.known_ids.is_empty() && rng.next_bool(0.7) {
                        let idx = rng.next_usize(0, self.known_ids.len() - 1);
                        self.known_ids[idx].clone()
                    } else {
                        format!("unknown_{}", rng.next_usize(0, 99))
                    };
                    MemoryOp::Forget { id }
                }
                3 => {
                    let id = if !self.known_ids.is_empty() && rng.next_bool(0.7) {
                        let idx = rng.next_usize(0, self.known_ids.len() - 1);
                        self.known_ids[idx].clone()
                    } else {
                        format!("unknown_{}", rng.next_usize(0, 99))
                    };
                    MemoryOp::Get { id }
                }
                _ => MemoryOp::Count,
            }
        }

        fn apply_operation(&mut self, op: &Self::Operation, _clock: &SimClock) {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async {
                match op {
                    MemoryOp::Remember {
                        content,
                        entity_type,
                    } => {
                        if let Ok(entity) = self.memory.remember(content, *entity_type, None).await
                        {
                            self.known_ids.push(entity.id);
                        }
                    }
                    MemoryOp::Recall { query } => {
                        let _ = self.memory.recall(query, None).await;
                    }
                    MemoryOp::Forget { id } => {
                        if self.memory.forget(id).await.unwrap_or(false) {
                            self.known_ids.retain(|i| i != id);
                        }
                    }
                    MemoryOp::Get { id } => {
                        let _ = self.memory.get(id).await;
                    }
                    MemoryOp::Count => {
                        let _ = self.memory.count(None).await;
                    }
                }
            });
        }

        fn check_invariants(&self) -> Result<(), String> {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async {
                let count = self.memory.count(None).await.map_err(|e| e.to_string())?;
                if count != self.known_ids.len() {
                    return Err(format!(
                        "count {} != known_ids.len() {}",
                        count,
                        self.known_ids.len()
                    ));
                }
                Ok(())
            })
        }

        fn describe_state(&self) -> String {
            format!("ArchivalMemory {{ known_ids: {} }}", self.known_ids.len())
        }
    }

    #[test]
    fn test_property_invariants() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));
        let memory = ArchivalMemory::new(backend);

        let wrapper = MemoryWrapper {
            memory,
            known_ids: Vec::new(),
        };

        PropertyTest::new(42)
            .with_max_operations(200)
            .with_time_advance(TimeAdvanceConfig::none())
            .run_and_assert(wrapper);
    }

    #[test]
    fn test_property_multi_seed() {
        for seed in [0, 1, 42, 12345] {
            let backend = SimStorageBackend::new(SimConfig::with_seed(seed));
            let memory = ArchivalMemory::new(backend);

            let wrapper = MemoryWrapper {
                memory,
                known_ids: Vec::new(),
            };

            PropertyTest::new(seed)
                .with_max_operations(100)
                .with_time_advance(TimeAdvanceConfig::none())
                .run_and_assert(wrapper);
        }
    }
}
