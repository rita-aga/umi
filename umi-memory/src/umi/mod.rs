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
//!     let embedder = SimEmbeddingProvider::with_seed(42);
//!     let vector = SimVectorBackend::new(42);
//!     let storage = SimStorageBackend::new(SimConfig::with_seed(42));
//!     let mut memory = Memory::new(llm, embedder, vector, storage);
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

mod builder;

pub use builder::MemoryBuilder;

use crate::constants::{
    MEMORY_IMPORTANCE_DEFAULT, MEMORY_IMPORTANCE_MAX, MEMORY_IMPORTANCE_MIN,
    MEMORY_RECALL_LIMIT_DEFAULT, MEMORY_RECALL_LIMIT_MAX, MEMORY_TEXT_BYTES_MAX,
};
use crate::embedding::EmbeddingProvider;
use crate::evolution::{DetectionOptions, EvolutionTracker};
use crate::extraction::{EntityExtractor, ExtractionOptions};
use crate::llm::LLMProvider;
use crate::retrieval::{DualRetriever, SearchOptions};
use crate::storage::{Entity, EntityType, EvolutionRelation, StorageBackend, VectorBackend};
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

    /// Whether to generate embeddings for entities (default: true)
    pub generate_embeddings: bool,
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

    /// Enable embedding generation (default).
    #[must_use]
    pub fn with_embeddings(mut self) -> Self {
        self.generate_embeddings = true;
        self
    }

    /// Disable embedding generation.
    #[must_use]
    pub fn without_embeddings(mut self) -> Self {
        self.generate_embeddings = false;
        self
    }
}

impl Default for RememberOptions {
    fn default() -> Self {
        Self {
            extract_entities: true,
            track_evolution: true,
            importance: MEMORY_IMPORTANCE_DEFAULT,
            generate_embeddings: true,
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
/// let embedder = SimEmbeddingProvider::with_seed(42);
/// let vector = SimVectorBackend::new(42);
/// let storage = SimStorageBackend::new(SimConfig::with_seed(42));
/// let mut memory = Memory::new(llm, embedder, vector, storage);
///
/// // Store and retrieve memories
/// memory.remember("Alice works at Acme", RememberOptions::default()).await?;
/// let results = memory.recall("Alice", RecallOptions::default()).await?;
/// ```
pub struct Memory<
    L: LLMProvider,
    E: EmbeddingProvider,
    S: StorageBackend,
    V: crate::storage::VectorBackend,
> {
    storage: S,
    extractor: EntityExtractor<L>,
    retriever: DualRetriever<L, E, V, S>,
    evolution: EvolutionTracker<L, S>,
    embedder: E,
    vector: V,
}

impl<
        L: LLMProvider + Clone,
        E: EmbeddingProvider + Clone,
        S: StorageBackend + Clone,
        V: crate::storage::VectorBackend + Clone,
    > Memory<L, E, S, V>
{
    /// Create a new Memory with all components.
    ///
    /// # Arguments
    /// - `llm` - LLM provider (cloned for each component)
    /// - `embedder` - Embedding provider (cloned for retriever)
    /// - `vector` - Vector backend for similarity search
    /// - `storage` - Storage backend (cloned for retriever)
    #[must_use]
    pub fn new(llm: L, embedder: E, vector: V, storage: S) -> Self {
        let extractor = EntityExtractor::new(llm.clone());
        let retriever = DualRetriever::new(
            llm.clone(),
            embedder.clone(),
            vector.clone(),
            storage.clone(),
        );
        let evolution = EvolutionTracker::new(llm);

        Self {
            storage,
            extractor,
            retriever,
            evolution,
            embedder,
            vector,
        }
    }

    /// Create a MemoryBuilder for constructing Memory with builder pattern.
    ///
    /// # Example
    /// ```rust,ignore
    /// let memory = Memory::builder()
    ///     .with_llm(llm)
    ///     .with_embedder(embedder)
    ///     .with_vector(vector)
    ///     .with_storage(storage)
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder() -> MemoryBuilder<L, E, V, S> {
        MemoryBuilder::new()
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
        let mut to_store: Vec<Entity> = if extracted.is_empty() {
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

        // Generate embeddings (NEW - graceful degradation: warn on failure, continue)
        if options.generate_embeddings && !to_store.is_empty() {
            // Collect entity contents for batch embedding
            let contents: Vec<&str> = to_store.iter().map(|e| e.content.as_str()).collect();

            match self.embedder.embed_batch(&contents).await {
                Ok(embeddings) => {
                    // Set embeddings on entities
                    for (entity, embedding) in to_store.iter_mut().zip(embeddings) {
                        entity.set_embedding(embedding);
                    }
                }
                Err(e) => {
                    // Graceful degradation: log warning, continue without embeddings
                    tracing::warn!(
                        "Failed to generate embeddings: {}. Continuing without embeddings.",
                        e
                    );
                }
            }
        }

        // Store each entity
        for entity in to_store {
            // Store returns the entity ID
            let _stored_id = self.storage.store_entity(&entity).await?;

            // Store embedding in vector backend (graceful degradation: warn on failure)
            if let Some(ref embedding) = entity.embedding {
                if let Err(e) = self.vector.store(&entity.id, embedding).await {
                    tracing::warn!(
                        "Failed to store embedding in vector backend for entity {}: {}. Entity searchable by text only.",
                        entity.id, e
                    );
                }
            }

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
    use crate::embedding::SimEmbeddingProvider;
    use crate::llm::SimLLMProvider;
    use crate::storage::{SimStorageBackend, SimVectorBackend};

    /// Helper to create a Memory with deterministic seed.
    fn create_memory(
        seed: u64,
    ) -> Memory<SimLLMProvider, SimEmbeddingProvider, SimStorageBackend, SimVectorBackend> {
        let llm = SimLLMProvider::with_seed(seed);
        let embedder = SimEmbeddingProvider::with_seed(seed);
        let vector = SimVectorBackend::new(seed);
        let storage = SimStorageBackend::new(SimConfig::with_seed(seed));
        Memory::new(llm, embedder, vector, storage)
    }

    // =========================================================================
    // RememberOptions Tests
    // =========================================================================

    #[test]
    fn test_remember_options_default() {
        let options = RememberOptions::default();

        assert!(options.extract_entities);
        assert!(options.track_evolution);
        assert!(options.generate_embeddings);
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

        let result = memory
            .remember(&long_text, RememberOptions::default())
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MemoryError::TextTooLong { .. }
        ));
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
            .remember("Test entity", RememberOptions::new().without_extraction())
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
            .remember("Test entity", RememberOptions::new().without_extraction())
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
            .remember("First item", RememberOptions::new().without_extraction())
            .await
            .unwrap();

        assert_eq!(memory.count().await.unwrap(), 1);

        memory
            .remember("Second item", RememberOptions::new().without_extraction())
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

// =============================================================================
// Sim Constructor
// =============================================================================

impl Memory<
    crate::llm::SimLLMProvider,
    crate::embedding::SimEmbeddingProvider,
    crate::storage::SimStorageBackend,
    crate::storage::SimVectorBackend,
> {
    /// Create a deterministic simulation Memory for testing.
    ///
    /// All components (LLM, embedder, vector, storage) use the same seed
    /// for reproducible behavior.
    ///
    /// TigerStyle: Convenient constructor for tests.
    ///
    /// # Arguments
    /// - `seed` - Random seed for deterministic behavior
    ///
    /// # Example
    /// ```rust
    /// use umi_memory::umi::Memory;
    ///
    /// let memory = Memory::sim(42);
    /// // All operations will be deterministic with same seed
    /// ```
    #[must_use]
    pub fn sim(seed: u64) -> Self {
        use crate::dst::SimConfig;
        use crate::embedding::SimEmbeddingProvider;
        use crate::llm::SimLLMProvider;
        use crate::storage::{SimStorageBackend, SimVectorBackend};

        let llm = SimLLMProvider::with_seed(seed);
        let embedder = SimEmbeddingProvider::with_seed(seed);
        let vector = SimVectorBackend::new(seed);
        let storage = SimStorageBackend::new(SimConfig::with_seed(seed));

        Self::new(llm, embedder, vector, storage)
    }
}

// =============================================================================
// DST Tests - Deterministic Simulation with Fault Injection
// =============================================================================

/// DST tests for Memory with embedding fault injection.
#[cfg(test)]
mod dst_tests {
    use super::*;
    use crate::constants::EMBEDDING_DIMENSIONS_COUNT;
    use crate::dst::{FaultConfig, FaultType, SimConfig, Simulation};
    use crate::embedding::SimEmbeddingProvider;
    use crate::llm::SimLLMProvider;
    use crate::storage::{SimStorageBackend, SimVectorBackend};

    #[tokio::test]
    async fn test_remember_with_embedding_timeout() {
        // Test that embedding timeout is handled gracefully
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::EmbeddingTimeout, 1.0)); // 100% failure rate

        sim.run(|env| async move {
            // Create embedder with fault injector
            let embedder = SimEmbeddingProvider::with_faults(42, env.faults.clone());
            let llm = SimLLMProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, embedder, vector, storage);

            // Remember should succeed even though embeddings fail
            let result = memory
                .remember("Alice works at Acme", RememberOptions::default())
                .await?;

            // Entity should be stored (without embedding)
            assert!(!result.entities.is_empty());
            // Embedding should be None due to failure
            assert!(result.entities[0].embedding.is_none());

            Ok::<(), MemoryError>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_remember_with_embedding_rate_limit() {
        // Test that rate limits are handled gracefully
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::EmbeddingRateLimit, 0.5)); // 50% failure

        sim.run(|env| async move {
            let embedder = SimEmbeddingProvider::with_faults(42, env.faults.clone());
            let llm = SimLLMProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, embedder, vector, storage);

            // Try multiple times, some should succeed, some fail
            let mut successes = 0;
            let mut failures = 0;

            for i in 0..10 {
                let result = memory
                    .remember(&format!("Text {}", i), RememberOptions::default())
                    .await;

                assert!(result.is_ok()); // Should never fail the entire operation
                let res = result.unwrap();

                if res.entities[0].embedding.is_some() {
                    successes += 1;
                } else {
                    failures += 1;
                }
            }

            // With 50% failure rate and 10 tries, should have both successes and failures
            assert!(successes > 0, "Should have some successful embeddings");
            assert!(failures > 0, "Should have some failed embeddings");

            Ok::<(), MemoryError>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_remember_without_embeddings_option() {
        // Test that disabling embeddings works
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|_env| async move {
            let embedder = SimEmbeddingProvider::with_seed(42);
            let llm = SimLLMProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, embedder, vector, storage);

            // Remember with embeddings disabled
            let result = memory
                .remember(
                    "Alice works at Acme",
                    RememberOptions::default().without_embeddings(),
                )
                .await?;

            // Entity stored but no embedding
            assert!(!result.entities.is_empty());
            assert!(result.entities[0].embedding.is_none());

            Ok::<(), MemoryError>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_remember_embeddings_deterministic() {
        // Test that same seed produces same embeddings
        async fn run_with_seed(seed: u64) -> Vec<Vec<f32>> {
            let embedder = SimEmbeddingProvider::with_seed(seed);
            let llm = SimLLMProvider::with_seed(seed);
            let vector = SimVectorBackend::new(seed);
            let storage = SimStorageBackend::new(SimConfig::with_seed(seed));
            let mut memory = Memory::new(llm, embedder, vector, storage);

            let result = memory
                .remember("Alice works at Acme", RememberOptions::default())
                .await
                .unwrap();

            result
                .entities
                .into_iter()
                .filter_map(|e| e.embedding)
                .collect()
        }

        let embeddings1 = run_with_seed(12345).await;
        let embeddings2 = run_with_seed(12345).await;

        assert!(!embeddings1.is_empty());
        assert_eq!(embeddings1.len(), embeddings2.len());

        // Same seed = same embeddings
        for (e1, e2) in embeddings1.iter().zip(embeddings2.iter()) {
            assert_eq!(e1, e2, "Same seed must produce same embeddings");
        }
    }

    #[tokio::test]
    async fn test_remember_embeddings_stored() {
        // Test that embeddings are actually stored and retrievable
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|_env| async move {
            let embedder = SimEmbeddingProvider::with_seed(42);
            let llm = SimLLMProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, embedder, vector, storage);

            // Remember with embeddings
            let result = memory
                .remember("Alice works at Acme", RememberOptions::default())
                .await?;

            assert!(!result.entities.is_empty());
            let entity_id = result.entities[0].id.clone();

            // Retrieve entity and verify embedding exists
            let retrieved = memory.storage.get_entity(&entity_id).await?;
            assert!(retrieved.is_some());

            let entity = retrieved.unwrap();
            assert!(entity.embedding.is_some(), "Embedding should be stored");
            assert_eq!(
                entity.embedding.as_ref().unwrap().len(),
                EMBEDDING_DIMENSIONS_COUNT
            );

            // Verify normalized (L2 norm = 1)
            let embedding = entity.embedding.unwrap();
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");

            Ok::<(), MemoryError>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_remember_batch_embeddings() {
        // Test that multiple entities get batch-embedded
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|_env| async move {
            let embedder = SimEmbeddingProvider::with_seed(42);
            let llm = SimLLMProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, embedder, vector, storage);

            // Remember text that will extract multiple entities
            let result = memory
                .remember(
                    "Alice works at Acme. Bob works at TechCo.",
                    RememberOptions::default(),
                )
                .await?;

            // Should have multiple entities, each with embedding
            if result.entities.len() > 1 {
                for entity in &result.entities {
                    if entity.embedding.is_some() {
                        assert_eq!(
                            entity.embedding.as_ref().unwrap().len(),
                            EMBEDDING_DIMENSIONS_COUNT
                        );
                    }
                }
            }

            Ok::<(), MemoryError>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_remember_with_service_unavailable() {
        // Test graceful degradation with service unavailable
        let sim = Simulation::new(SimConfig::with_seed(42)).with_fault(FaultConfig::new(
            FaultType::EmbeddingServiceUnavailable,
            1.0,
        ));

        sim.run(|env| async move {
            let embedder = SimEmbeddingProvider::with_faults(42, env.faults.clone());
            let llm = SimLLMProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, embedder, vector, storage);

            // Should succeed despite embedding service being down
            let result = memory
                .remember("Alice works at Acme", RememberOptions::default())
                .await?;

            assert!(!result.entities.is_empty());
            // No embedding due to service failure
            assert!(result.entities[0].embedding.is_none());

            Ok::<(), MemoryError>(())
        })
        .await
        .unwrap();
    }

    // =========================================================================
    // Vector Search DST Tests
    // =========================================================================

    #[tokio::test]
    async fn test_recall_with_vector_search() {
        // Test that recall uses vector search when embeddings are available
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|_env| async move {
            let embedder = SimEmbeddingProvider::with_seed(42);
            let llm = SimLLMProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, embedder, vector, storage);

            // Store entities with embeddings
            memory
                .remember("Alice works at Acme Corp", RememberOptions::default())
                .await?;
            memory
                .remember("Bob works at TechCo", RememberOptions::default())
                .await?;

            // Recall should use vector search
            let result = memory
                .recall("Who works at Acme?", RecallOptions::default())
                .await?;

            // Should find relevant results
            assert!(!result.is_empty());
            // Alice should be in results (content similarity)
            assert!(result.iter().any(|e| e.name.contains("Alice")));

            Ok::<(), MemoryError>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_recall_vector_search_timeout() {
        // Test fallback to text search when vector search times out
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::VectorSearchTimeout, 1.0));

        sim.run(|env| async move {
            let embedder = SimEmbeddingProvider::with_seed(42);
            let llm = SimLLMProvider::with_seed(42);
            let vector = SimVectorBackend::with_faults(42, env.faults.clone());
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, embedder, vector, storage);

            // Store entity
            memory
                .remember("Alice works at Acme Corp", RememberOptions::default())
                .await?;

            // Recall should fall back to text search
            let result = memory.recall("Alice", RecallOptions::default()).await;

            // Should still work via text fallback
            assert!(result.is_ok());
            let entities = result.unwrap();
            assert!(!entities.is_empty());

            Ok::<(), MemoryError>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_recall_vector_deterministic() {
        // Test that same seed produces same ranking
        // Disable entity extraction to ensure deterministic entities
        let seed = 42;

        // First run
        let embedder = SimEmbeddingProvider::with_seed(seed);
        let llm = SimLLMProvider::with_seed(seed);
        let vector = SimVectorBackend::new(seed);
        let storage = SimStorageBackend::new(SimConfig::with_seed(seed));
        let mut memory1 = Memory::new(llm, embedder, vector, storage);

        memory1
            .remember(
                "Alice works at Acme Corp",
                RememberOptions::default().without_extraction(),
            )
            .await
            .unwrap();
        memory1
            .remember(
                "Bob works at TechCo",
                RememberOptions::default().without_extraction(),
            )
            .await
            .unwrap();
        memory1
            .remember(
                "Charlie works at DataInc",
                RememberOptions::default().without_extraction(),
            )
            .await
            .unwrap();

        let result1 = memory1
            .recall("works", RecallOptions::default().fast_only())
            .await
            .unwrap();
        let names1: Vec<String> = result1.iter().map(|e| e.name.clone()).collect();

        // Second run with same seed
        let embedder2 = SimEmbeddingProvider::with_seed(seed);
        let llm2 = SimLLMProvider::with_seed(seed);
        let vector2 = SimVectorBackend::new(seed);
        let storage2 = SimStorageBackend::new(SimConfig::with_seed(seed));
        let mut memory2 = Memory::new(llm2, embedder2, vector2, storage2);

        memory2
            .remember(
                "Alice works at Acme Corp",
                RememberOptions::default().without_extraction(),
            )
            .await
            .unwrap();
        memory2
            .remember(
                "Bob works at TechCo",
                RememberOptions::default().without_extraction(),
            )
            .await
            .unwrap();
        memory2
            .remember(
                "Charlie works at DataInc",
                RememberOptions::default().without_extraction(),
            )
            .await
            .unwrap();

        let result2 = memory2
            .recall("works", RecallOptions::default().fast_only())
            .await
            .unwrap();
        let names2: Vec<String> = result2.iter().map(|e| e.name.clone()).collect();

        // Same seed = same ordering
        assert!(!names1.is_empty(), "Should find results");
        assert_eq!(names1, names2, "Same seed must produce same ranking");
    }

    #[tokio::test]
    async fn test_recall_vector_storage_partial_failure() {
        // Test that some embeddings failing to store doesn't break recall
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::VectorStoreFail, 0.5)); // 50% failure

        sim.run(|env| async move {
            let embedder = SimEmbeddingProvider::with_seed(42);
            let llm = SimLLMProvider::with_seed(42);
            let vector = SimVectorBackend::with_faults(42, env.faults.clone());
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, embedder, vector, storage);

            // Store multiple entities (some will fail vector storage)
            // Use without_extraction to ensure deterministic entities
            for i in 0..10 {
                let opts = RememberOptions::default().without_extraction();
                memory
                    .remember(&format!("Entity number {}", i), opts)
                    .await?;
            }

            // Recall with text search should still work (fallback path)
            let result = memory.recall("Entity", RecallOptions::default()).await;

            assert!(result.is_ok());
            let entities = result.unwrap();
            // Should find entities via text search fallback
            assert!(
                !entities.is_empty(),
                "Should find entities even with vector storage failures"
            );

            Ok::<(), MemoryError>(())
        })
        .await
        .unwrap();
    }
}
