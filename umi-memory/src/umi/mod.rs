//! Umi Memory - Main Interface (ADR-017)
//!
//! TigerStyle: Sim-first, deterministic, graceful degradation.
//!
//! # Overview
//!
//! The Memory class orchestrates all Umi components:
//! - EntityExtractor for extracting entities from text
//! - DualRetriever for searching memories
//! - EvolutionTracker for detecting memory relationships
//! - Storage backend for persistence
//!
//! # Example
//!
//! ```rust,ignore
//! use umi_memory::umi::{Memory, RememberOptions, RecallOptions};
//! use umi_memory::{SimLLMProvider, SimStorageBackend, SimConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     let llm = SimLLMProvider::with_seed(42);
//!     let storage = SimStorageBackend::new(SimConfig::with_seed(42));
//!     let mut memory = Memory::new(llm, storage);
//!
//!     // Remember information
//!     let result = memory.remember("Alice works at Acme", RememberOptions::default()).await.unwrap();
//!     println!("Stored {} entities", result.entities.len());
//!
//!     // Recall information
//!     let found = memory.recall("Alice", RecallOptions::default()).await.unwrap();
//!     println!("Found {} results", found.len());
//! }
//! ```

use crate::constants::{
    MEMORY_IMPORTANCE_DEFAULT, MEMORY_IMPORTANCE_MAX, MEMORY_IMPORTANCE_MIN,
    MEMORY_RECALL_LIMIT_DEFAULT, MEMORY_RECALL_LIMIT_MAX, MEMORY_TEXT_BYTES_MAX,
};
use crate::evolution::{DetectionOptions, EvolutionTracker};
use crate::extraction::{EntityExtractor, ExtractionOptions};
use crate::llm::LLMProvider;
use crate::retrieval::{DualRetriever, SearchOptions};
use crate::storage::{Entity, EntityType, EvolutionRelation, StorageBackend};
use thiserror::Error;

// =============================================================================
// Error Types
// =============================================================================

/// Errors from memory operations.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Input text is empty
    #[error("text is empty")]
    EmptyText,

    /// Input text exceeds size limit
    #[error("text too long: {len} bytes (max {max})")]
    TextTooLong {
        /// Actual length
        len: usize,
        /// Maximum allowed
        max: usize,
    },

    /// Query is empty
    #[error("query is empty")]
    EmptyQuery,

    /// Invalid importance value
    #[error("invalid importance: {value} (must be {min}-{max})")]
    InvalidImportance {
        /// Provided value
        value: f32,
        /// Minimum allowed
        min: f32,
        /// Maximum allowed
        max: f32,
    },

    /// Invalid limit value
    #[error("invalid limit: {value} (must be 1-{max})")]
    InvalidLimit {
        /// Provided value
        value: usize,
        /// Maximum allowed
        max: usize,
    },

    /// Storage error
    #[error("storage error: {message}")]
    Storage {
        /// Error message
        message: String,
    },
}

impl From<crate::storage::StorageError> for MemoryError {
    fn from(err: crate::storage::StorageError) -> Self {
        MemoryError::Storage {
            message: err.to_string(),
        }
    }
}

// =============================================================================
// Options Types
// =============================================================================

/// Options for remember operations.
///
/// TigerStyle: Builder pattern with defaults.
#[derive(Debug, Clone)]
pub struct RememberOptions {
    /// Whether to extract entities using LLM (default: true)
    pub extract_entities: bool,

    /// Whether to track evolution with existing memories (default: true)
    pub track_evolution: bool,

    /// Importance score 0.0-1.0 (default: 0.5)
    pub importance: f32,
}

impl RememberOptions {
    /// Create new options with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable entity extraction.
    #[must_use]
    pub fn without_extraction(mut self) -> Self {
        self.extract_entities = false;
        self
    }

    /// Disable evolution tracking.
    #[must_use]
    pub fn without_evolution(mut self) -> Self {
        self.track_evolution = false;
        self
    }

    /// Set importance score.
    ///
    /// # Panics
    /// Panics if importance is not in valid range.
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        debug_assert!(
            (MEMORY_IMPORTANCE_MIN..=MEMORY_IMPORTANCE_MAX).contains(&importance),
            "importance must be {}-{}: got {}",
            MEMORY_IMPORTANCE_MIN,
            MEMORY_IMPORTANCE_MAX,
            importance
        );
        self.importance = importance;
        self
    }
}

impl Default for RememberOptions {
    fn default() -> Self {
        Self {
            extract_entities: true,
            track_evolution: true,
            importance: MEMORY_IMPORTANCE_DEFAULT,
        }
    }
}

/// Options for recall operations.
///
/// TigerStyle: Builder pattern with defaults.
#[derive(Debug, Clone)]
pub struct RecallOptions {
    /// Maximum results (default: 10)
    pub limit: usize,

    /// Use LLM for deep search (default: auto based on query)
    pub deep_search: Option<bool>,

    /// Time range filter (start_ms, end_ms)
    pub time_range: Option<(u64, u64)>,
}

impl RecallOptions {
    /// Create new options with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum results.
    ///
    /// # Panics
    /// Panics if limit is 0 or exceeds maximum.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        debug_assert!(
            limit > 0 && limit <= MEMORY_RECALL_LIMIT_MAX,
            "limit must be 1-{}: got {}",
            MEMORY_RECALL_LIMIT_MAX,
            limit
        );
        self.limit = limit;
        self
    }

    /// Enable deep search.
    #[must_use]
    pub fn with_deep_search(mut self) -> Self {
        self.deep_search = Some(true);
        self
    }

    /// Disable deep search (fast only).
    #[must_use]
    pub fn fast_only(mut self) -> Self {
        self.deep_search = Some(false);
        self
    }

    /// Set time range filter.
    #[must_use]
    pub fn with_time_range(mut self, start_ms: u64, end_ms: u64) -> Self {
        debug_assert!(start_ms <= end_ms, "start_ms must be <= end_ms");
        self.time_range = Some((start_ms, end_ms));
        self
    }
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self {
            limit: MEMORY_RECALL_LIMIT_DEFAULT,
            deep_search: None,
            time_range: None,
        }
    }
}

// =============================================================================
// Result Types
// =============================================================================

/// Result of a remember operation.
#[derive(Debug, Clone)]
pub struct RememberResult {
    /// Stored entities
    pub entities: Vec<Entity>,

    /// Evolution relations detected (if any)
    pub evolutions: Vec<EvolutionRelation>,
}

impl RememberResult {
    /// Create a new remember result.
    #[must_use]
    pub fn new(entities: Vec<Entity>, evolutions: Vec<EvolutionRelation>) -> Self {
        Self {
            entities,
            evolutions,
        }
    }

    /// Get the number of stored entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Check if any evolution relations were detected.
    #[must_use]
    pub fn has_evolutions(&self) -> bool {
        !self.evolutions.is_empty()
    }

    /// Get entities iterator.
    pub fn iter_entities(&self) -> impl Iterator<Item = &Entity> {
        self.entities.iter()
    }
}

// =============================================================================
// Memory Class
// =============================================================================

/// Main interface for Umi memory system.
///
/// Orchestrates all components for a simple remember/recall API.
///
/// # Type Parameters
/// - `L`: LLM provider for extraction, retrieval, evolution (SimLLMProvider for testing)
/// - `S`: Storage backend for persistence (SimStorageBackend for testing)
///
/// # Example
///
/// ```rust,ignore
/// use umi_memory::umi::{Memory, RememberOptions, RecallOptions};
/// use umi_memory::{SimLLMProvider, SimStorageBackend, SimConfig};
///
/// let llm = SimLLMProvider::with_seed(42);
/// let storage = SimStorageBackend::new(SimConfig::with_seed(42));
/// let mut memory = Memory::new(llm, storage);
///
/// // Store and retrieve memories
/// memory.remember("Alice works at Acme", RememberOptions::default()).await?;
/// let results = memory.recall("Alice", RecallOptions::default()).await?;
/// ```
pub struct Memory<L: LLMProvider, S: StorageBackend> {
    storage: S,
    extractor: EntityExtractor<L>,
    retriever: DualRetriever<L, S>,
    evolution: EvolutionTracker<L, S>,
}

impl<L: LLMProvider + Clone, S: StorageBackend + Clone> Memory<L, S> {
    /// Create a new Memory with all components.
    ///
    /// # Arguments
    /// - `llm` - LLM provider (cloned for each component)
    /// - `storage` - Storage backend (cloned for retriever)
    #[must_use]
    pub fn new(llm: L, storage: S) -> Self {
        let extractor = EntityExtractor::new(llm.clone());
        let retriever = DualRetriever::new(llm.clone(), storage.clone());
        let evolution = EvolutionTracker::new(llm);

        Self {
            storage,
            extractor,
            retriever,
            evolution,
        }
    }

    /// Store information in memory.
    ///
    /// Extracts entities from text using LLM and stores them.
    /// Optionally detects evolution relationships with existing memories.
    ///
    /// # Arguments
    /// - `text` - Text to remember
    /// - `options` - Remember options
    ///
    /// # Returns
    /// `Ok(RememberResult)` with stored entities and detected evolutions,
    /// `Err(MemoryError)` for validation errors.
    ///
    /// # Graceful Degradation
    /// - If extraction fails, falls back to storing raw text as Note
    /// - If evolution detection fails, skips without error
    pub async fn remember(
        &mut self,
        text: &str,
        options: RememberOptions,
    ) -> Result<RememberResult, MemoryError> {
        // Preconditions (TigerStyle)
        if text.is_empty() {
            return Err(MemoryError::EmptyText);
        }
        if text.len() > MEMORY_TEXT_BYTES_MAX {
            return Err(MemoryError::TextTooLong {
                len: text.len(),
                max: MEMORY_TEXT_BYTES_MAX,
            });
        }
        if !(MEMORY_IMPORTANCE_MIN..=MEMORY_IMPORTANCE_MAX).contains(&options.importance) {
            return Err(MemoryError::InvalidImportance {
                value: options.importance,
                min: MEMORY_IMPORTANCE_MIN,
                max: MEMORY_IMPORTANCE_MAX,
            });
        }

        let mut entities = Vec::new();
        let mut evolutions = Vec::new();

        // Extract entities (graceful degradation: fallback to raw text)
        let extracted = if options.extract_entities {
            match self
                .extractor
                .extract(text, ExtractionOptions::default())
                .await
            {
                Ok(result) => result.entities,
                Err(_) => vec![], // Extraction failed, will use fallback
            }
        } else {
            vec![]
        };

        // Convert extracted entities to storage entities
        let to_store: Vec<Entity> = if extracted.is_empty() {
            // Fallback: store as single Note entity
            let name = if text.len() > 50 {
                format!("Note: {}...", &text[..47])
            } else {
                format!("Note: {}", text)
            };
            vec![Entity::new(EntityType::Note, name, text.to_string())]
        } else {
            extracted
                .into_iter()
                .map(|e| {
                    let entity_type = convert_entity_type(&e.entity_type);
                    Entity::new(entity_type, e.name, e.content)
                })
                .collect()
        };

        // Store each entity
        for entity in to_store {
            // Store returns the entity ID
            let _stored_id = self.storage.store_entity(&entity).await?;

            // Track evolution (graceful: skip on failure)
            if options.track_evolution {
                // Search for related entities
                if let Ok(existing) = self.storage.search(&entity.name, 5).await {
                    // Filter out the entity we just stored
                    let existing: Vec<Entity> =
                        existing.into_iter().filter(|e| e.id != entity.id).collect();

                    if !existing.is_empty() {
                        if let Ok(Some(detection)) = self
                            .evolution
                            .detect(&entity, &existing, DetectionOptions::default())
                            .await
                        {
                            evolutions.push(detection.relation);
                        }
                    }
                }
            }

            entities.push(entity);
        }

        // Postcondition (TigerStyle)
        debug_assert!(!entities.is_empty(), "must store at least one entity");

        Ok(RememberResult::new(entities, evolutions))
    }

    /// Retrieve memories matching query.
    ///
    /// Uses DualRetriever for smart search:
    /// - Fast path: Direct search in storage
    /// - Deep path: LLM rewrites query into variations, merges results
    ///
    /// # Arguments
    /// - `query` - Search query
    /// - `options` - Recall options
    ///
    /// # Returns
    /// `Ok(Vec<Entity>)` with matching entities,
    /// `Err(MemoryError)` for validation errors.
    pub async fn recall(
        &self,
        query: &str,
        options: RecallOptions,
    ) -> Result<Vec<Entity>, MemoryError> {
        // Preconditions (TigerStyle)
        if query.is_empty() {
            return Err(MemoryError::EmptyQuery);
        }
        if options.limit == 0 || options.limit > MEMORY_RECALL_LIMIT_MAX {
            return Err(MemoryError::InvalidLimit {
                value: options.limit,
                max: MEMORY_RECALL_LIMIT_MAX,
            });
        }

        // Build search options
        let mut search_options = SearchOptions::new().with_limit(options.limit);

        // Apply deep_search setting
        if let Some(deep) = options.deep_search {
            search_options = search_options.with_deep_search(deep);
        }

        // Apply time range if set
        if let Some((start, end)) = options.time_range {
            search_options = search_options.with_time_range(start, end);
        }

        // Use DualRetriever for search
        let result = self
            .retriever
            .search(query, search_options)
            .await
            .map_err(|e| MemoryError::Storage {
                message: e.to_string(),
            })?;

        // Postcondition (TigerStyle)
        debug_assert!(
            result.len() <= options.limit,
            "results exceed limit: {} > {}",
            result.len(),
            options.limit
        );

        Ok(result.entities)
    }

    /// Delete entity by ID.
    ///
    /// # Arguments
    /// - `entity_id` - ID of entity to delete
    ///
    /// # Returns
    /// `Ok(true)` if deleted, `Ok(false)` if not found.
    pub async fn forget(&mut self, entity_id: &str) -> Result<bool, MemoryError> {
        debug_assert!(!entity_id.is_empty(), "entity_id must not be empty");

        self.storage.delete_entity(entity_id).await?;
        Ok(true)
    }

    /// Get entity by ID.
    ///
    /// # Arguments
    /// - `entity_id` - Entity ID
    ///
    /// # Returns
    /// `Ok(Some(Entity))` if found, `Ok(None)` otherwise.
    pub async fn get(&self, entity_id: &str) -> Result<Option<Entity>, MemoryError> {
        debug_assert!(!entity_id.is_empty(), "entity_id must not be empty");

        Ok(self.storage.get_entity(entity_id).await?)
    }

    /// Count total entities in storage.
    pub async fn count(&self) -> Result<usize, MemoryError> {
        Ok(self.storage.count_entities(None).await?)
    }

    /// Get reference to storage backend.
    #[must_use]
    pub fn storage(&self) -> &S {
        &self.storage
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert extraction EntityType to storage EntityType.
fn convert_entity_type(ext_type: &crate::extraction::EntityType) -> EntityType {
    use crate::extraction::EntityType as ExtType;

    match ext_type {
        ExtType::Person => EntityType::Person,
        ExtType::Organization => EntityType::Note, // No direct mapping, use Note
        ExtType::Project => EntityType::Project,
        ExtType::Topic => EntityType::Topic,
        ExtType::Preference => EntityType::Note, // No direct mapping, use Note
        ExtType::Task => EntityType::Task,
        ExtType::Event => EntityType::Note, // No direct mapping, use Note
        ExtType::Note => EntityType::Note,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dst::SimConfig;
    use crate::llm::SimLLMProvider;
    use crate::storage::SimStorageBackend;

    /// Helper to create a Memory with deterministic seed.
    fn create_memory(seed: u64) -> Memory<SimLLMProvider, SimStorageBackend> {
        let llm = SimLLMProvider::with_seed(seed);
        let storage = SimStorageBackend::new(SimConfig::with_seed(seed));
        Memory::new(llm, storage)
    }

    // =========================================================================
    // RememberOptions Tests
    // =========================================================================

    #[test]
    fn test_remember_options_default() {
        let options = RememberOptions::default();

        assert!(options.extract_entities);
        assert!(options.track_evolution);
        assert!((options.importance - MEMORY_IMPORTANCE_DEFAULT).abs() < f32::EPSILON);
    }

    #[test]
    fn test_remember_options_builder() {
        let options = RememberOptions::new()
            .without_extraction()
            .without_evolution()
            .with_importance(0.8);

        assert!(!options.extract_entities);
        assert!(!options.track_evolution);
        assert!((options.importance - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic(expected = "importance must be")]
    fn test_remember_options_invalid_importance() {
        let _ = RememberOptions::new().with_importance(1.5);
    }

    // =========================================================================
    // RecallOptions Tests
    // =========================================================================

    #[test]
    fn test_recall_options_default() {
        let options = RecallOptions::default();

        assert_eq!(options.limit, MEMORY_RECALL_LIMIT_DEFAULT);
        assert!(options.deep_search.is_none());
        assert!(options.time_range.is_none());
    }

    #[test]
    fn test_recall_options_builder() {
        let options = RecallOptions::new()
            .with_limit(20)
            .with_deep_search()
            .with_time_range(1000, 2000);

        assert_eq!(options.limit, 20);
        assert_eq!(options.deep_search, Some(true));
        assert_eq!(options.time_range, Some((1000, 2000)));
    }

    #[test]
    fn test_recall_options_fast_only() {
        let options = RecallOptions::new().fast_only();

        assert_eq!(options.deep_search, Some(false));
    }

    #[test]
    #[should_panic(expected = "limit must be")]
    fn test_recall_options_invalid_limit_zero() {
        let _ = RecallOptions::new().with_limit(0);
    }

    #[test]
    #[should_panic(expected = "limit must be")]
    fn test_recall_options_invalid_limit_too_large() {
        let _ = RecallOptions::new().with_limit(MEMORY_RECALL_LIMIT_MAX + 1);
    }

    // =========================================================================
    // RememberResult Tests
    // =========================================================================

    #[test]
    fn test_remember_result() {
        let entities = vec![Entity::new(
            EntityType::Person,
            "Alice".to_string(),
            "Works at Acme".to_string(),
        )];
        let result = RememberResult::new(entities, vec![]);

        assert_eq!(result.entity_count(), 1);
        assert!(!result.has_evolutions());
    }

    // =========================================================================
    // Memory Creation Tests
    // =========================================================================

    #[test]
    fn test_memory_creation() {
        let memory = create_memory(42);
        // Just verify it compiles and creates without panic
        let _ = memory;
    }

    // =========================================================================
    // Remember Tests
    // =========================================================================

    #[tokio::test]
    async fn test_remember_basic() {
        let mut memory = create_memory(42);

        let result = memory
            .remember("Alice works at Acme Corp", RememberOptions::default())
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.entities.is_empty());
    }

    #[tokio::test]
    async fn test_remember_without_extraction() {
        let mut memory = create_memory(42);

        let result = memory
            .remember(
                "Some text to store",
                RememberOptions::new().without_extraction(),
            )
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.entity_count(), 1);
        // Should be stored as Note
        assert_eq!(result.entities[0].entity_type, EntityType::Note);
    }

    #[tokio::test]
    async fn test_remember_empty_text_error() {
        let mut memory = create_memory(42);

        let result = memory.remember("", RememberOptions::default()).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MemoryError::EmptyText));
    }

    #[tokio::test]
    async fn test_remember_text_too_long_error() {
        let mut memory = create_memory(42);
        let long_text = "a".repeat(MEMORY_TEXT_BYTES_MAX + 1);

        let result = memory.remember(&long_text, RememberOptions::default()).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MemoryError::TextTooLong { .. }));
    }

    // =========================================================================
    // Recall Tests
    // =========================================================================

    #[tokio::test]
    async fn test_recall_basic() {
        let mut memory = create_memory(42);

        // First remember something
        memory
            .remember("Alice works at Acme Corp", RememberOptions::default())
            .await
            .unwrap();

        // Then recall
        let results = memory.recall("Alice", RecallOptions::default()).await;

        assert!(results.is_ok());
    }

    #[tokio::test]
    async fn test_recall_empty_query_error() {
        let memory = create_memory(42);

        let result = memory.recall("", RecallOptions::default()).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MemoryError::EmptyQuery));
    }

    #[tokio::test]
    async fn test_recall_with_limit() {
        let mut memory = create_memory(42);

        // Remember multiple items
        for i in 0..5 {
            memory
                .remember(
                    &format!("Item {} is interesting", i),
                    RememberOptions::new().without_extraction(),
                )
                .await
                .unwrap();
        }

        // Recall with limit
        let results = memory
            .recall("Item", RecallOptions::new().with_limit(2))
            .await
            .unwrap();

        assert!(results.len() <= 2);
    }

    // =========================================================================
    // Get/Forget Tests
    // =========================================================================

    #[tokio::test]
    async fn test_get_entity() {
        let mut memory = create_memory(42);

        let result = memory
            .remember(
                "Test entity",
                RememberOptions::new().without_extraction(),
            )
            .await
            .unwrap();

        let entity_id = &result.entities[0].id;
        let found = memory.get(entity_id).await.unwrap();

        assert!(found.is_some());
        assert_eq!(found.unwrap().id, *entity_id);
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let memory = create_memory(42);

        let found = memory.get("nonexistent-id").await.unwrap();

        assert!(found.is_none());
    }

    #[tokio::test]
    async fn test_forget_entity() {
        let mut memory = create_memory(42);

        let result = memory
            .remember(
                "Test entity",
                RememberOptions::new().without_extraction(),
            )
            .await
            .unwrap();

        let entity_id = &result.entities[0].id;

        // Verify it exists
        assert!(memory.get(entity_id).await.unwrap().is_some());

        // Forget it
        let deleted = memory.forget(entity_id).await.unwrap();
        assert!(deleted);

        // Verify it's gone
        assert!(memory.get(entity_id).await.unwrap().is_none());
    }

    // =========================================================================
    // Count Tests
    // =========================================================================

    #[tokio::test]
    async fn test_count() {
        let mut memory = create_memory(42);

        assert_eq!(memory.count().await.unwrap(), 0);

        memory
            .remember(
                "First item",
                RememberOptions::new().without_extraction(),
            )
            .await
            .unwrap();

        assert_eq!(memory.count().await.unwrap(), 1);

        memory
            .remember(
                "Second item",
                RememberOptions::new().without_extraction(),
            )
            .await
            .unwrap();

        assert_eq!(memory.count().await.unwrap(), 2);
    }

    // =========================================================================
    // Determinism Tests
    // =========================================================================

    #[tokio::test]
    async fn test_deterministic_same_seed() {
        let mut memory1 = create_memory(42);
        let mut memory2 = create_memory(42);

        let text = "Alice works at Acme Corp as an engineer";

        let result1 = memory1
            .remember(text, RememberOptions::default())
            .await
            .unwrap();
        let result2 = memory2
            .remember(text, RememberOptions::default())
            .await
            .unwrap();

        // Same seed should produce same number of entities
        assert_eq!(result1.entity_count(), result2.entity_count());
    }

    // =========================================================================
    // EntityType Conversion Tests
    // =========================================================================

    #[test]
    fn test_convert_entity_type() {
        use crate::extraction::EntityType as ExtType;

        assert_eq!(convert_entity_type(&ExtType::Person), EntityType::Person);
        assert_eq!(convert_entity_type(&ExtType::Project), EntityType::Project);
        assert_eq!(convert_entity_type(&ExtType::Topic), EntityType::Topic);
        assert_eq!(convert_entity_type(&ExtType::Task), EntityType::Task);
        assert_eq!(convert_entity_type(&ExtType::Note), EntityType::Note);

        // Types without direct mapping should become Note
        assert_eq!(
            convert_entity_type(&ExtType::Organization),
            EntityType::Note
        );
        assert_eq!(convert_entity_type(&ExtType::Preference), EntityType::Note);
        assert_eq!(convert_entity_type(&ExtType::Event), EntityType::Note);
    }
}
