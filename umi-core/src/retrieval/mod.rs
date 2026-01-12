//! Dual Retrieval - Fast search + LLM reasoning
//!
//! TigerStyle: Sim-first, deterministic, graceful degradation.
//!
//! See ADR-015 for design rationale.
//!
//! # Architecture
//!
//! ```text
//! DualRetriever<L: LLMProvider, S: StorageBackend>
//! ├── search()          → SearchResult
//! ├── needs_deep_search() → bool (heuristic)
//! ├── rewrite_query()   → Vec<String> (via LLM)
//! └── merge_rrf()       → Vec<Entity> (Reciprocal Rank Fusion)
//! ```
//!
//! # Usage
//!
//! ```rust
//! use umi_core::retrieval::{DualRetriever, SearchOptions};
//! use umi_core::llm::SimLLMProvider;
//! use umi_core::storage::SimStorageBackend;
//! use umi_core::dst::SimConfig;
//!
//! #[tokio::main]
//! async fn main() {
//!     let llm = SimLLMProvider::with_seed(42);
//!     let storage = SimStorageBackend::new(SimConfig::with_seed(42));
//!     let retriever = DualRetriever::new(llm, storage);
//!
//!     let result = retriever.search("Who works at Acme?", SearchOptions::default()).await.unwrap();
//!     println!("Found {} results", result.len());
//! }
//! ```

mod prompts;
mod types;

pub use prompts::build_query_rewrite_prompt;
pub use types::{
    needs_deep_search, SearchOptions, SearchResult, ABSTRACT_TERMS, QUESTION_WORDS,
    RELATIONSHIP_TERMS, TEMPORAL_TERMS,
};

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::constants::{
    RETRIEVAL_QUERY_BYTES_MAX, RETRIEVAL_QUERY_REWRITE_COUNT_MAX, RETRIEVAL_RESULTS_COUNT_MAX,
    RETRIEVAL_RRF_K,
};
use crate::llm::{CompletionRequest, LLMProvider};
use crate::storage::{Entity, StorageBackend};

// =============================================================================
// Error Types
// =============================================================================

/// Errors from retrieval operations.
///
/// Note: LLM errors result in graceful degradation (fast search only),
/// not an error return.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RetrievalError {
    /// Query is empty
    #[error("Query is empty")]
    EmptyQuery,

    /// Query exceeds size limit
    #[error("Query too long: {len} bytes (max {max})")]
    QueryTooLong {
        /// Actual length
        len: usize,
        /// Maximum allowed
        max: usize,
    },

    /// Invalid result limit
    #[error("Invalid limit: {value} (must be 1-{max})")]
    InvalidLimit {
        /// Provided value
        value: usize,
        /// Maximum allowed
        max: usize,
    },

    /// Storage error
    #[error("Storage error: {message}")]
    Storage {
        /// Error message
        message: String,
    },
}

impl From<crate::storage::StorageError> for RetrievalError {
    fn from(err: crate::storage::StorageError) -> Self {
        RetrievalError::Storage {
            message: err.to_string(),
        }
    }
}

// =============================================================================
// DualRetriever
// =============================================================================

/// Dual retriever: fast search + LLM reasoning.
///
/// TigerStyle: Generic over LLM and storage for sim/production flexibility.
///
/// # Example
///
/// ```rust
/// use umi_core::retrieval::{DualRetriever, SearchOptions};
/// use umi_core::llm::SimLLMProvider;
/// use umi_core::storage::SimStorageBackend;
/// use umi_core::dst::SimConfig;
///
/// #[tokio::main]
/// async fn main() {
///     let llm = SimLLMProvider::with_seed(42);
///     let storage = SimStorageBackend::new(SimConfig::with_seed(42));
///     let retriever = DualRetriever::new(llm, storage);
///
///     // Deep search with query rewriting
///     let result = retriever
///         .search("Who works at Acme?", SearchOptions::default())
///         .await
///         .unwrap();
///
///     // Fast search only
///     let fast_result = retriever
///         .search("Alice", SearchOptions::new().fast_only())
///         .await
///         .unwrap();
/// }
/// ```
#[derive(Debug)]
pub struct DualRetriever<L: LLMProvider, S: StorageBackend> {
    llm: L,
    storage: S,
}

impl<L: LLMProvider, S: StorageBackend> DualRetriever<L, S> {
    /// Create a new dual retriever.
    #[must_use]
    pub fn new(llm: L, storage: S) -> Self {
        Self { llm, storage }
    }

    /// Search with dual retrieval strategy.
    ///
    /// # Arguments
    /// - `query` - Search query
    /// - `options` - Search options (limit, deep_search, time_range)
    ///
    /// # Returns
    /// `SearchResult` with entities, query info, and metadata.
    ///
    /// # Errors
    /// Returns `RetrievalError` if query is empty, too long, or limit is invalid.
    ///
    /// # Graceful Degradation
    /// If LLM fails during query rewriting, falls back to fast search only.
    pub async fn search(
        &self,
        query: &str,
        options: SearchOptions,
    ) -> Result<SearchResult, RetrievalError> {
        // TigerStyle: Preconditions
        if query.is_empty() {
            return Err(RetrievalError::EmptyQuery);
        }
        if query.len() > RETRIEVAL_QUERY_BYTES_MAX {
            return Err(RetrievalError::QueryTooLong {
                len: query.len(),
                max: RETRIEVAL_QUERY_BYTES_MAX,
            });
        }
        if options.limit == 0 || options.limit > RETRIEVAL_RESULTS_COUNT_MAX {
            return Err(RetrievalError::InvalidLimit {
                value: options.limit,
                max: RETRIEVAL_RESULTS_COUNT_MAX,
            });
        }

        // 1. Fast search (always runs)
        let fast_results = self.fast_search(query, options.limit * 2).await?;

        // 2. Decide if deep search is needed
        let use_deep = options.deep_search && needs_deep_search(query);

        let (results, deep_search_used, query_variations) = if use_deep {
            // 3. Deep search: rewrite query and search variations
            let variations = self.rewrite_query(query).await;
            let deep_results = self
                .deep_search(&variations, query, options.limit * 2)
                .await;

            // 4. Merge results using RRF
            let merged = self.merge_rrf(&[&fast_results, &deep_results]);
            (merged, true, variations)
        } else {
            (fast_results, false, vec![query.to_string()])
        };

        // 5. Apply time filter if specified
        let results = if let Some((start_ms, end_ms)) = options.time_range {
            results
                .into_iter()
                .filter(|e| {
                    if let Some(event_time) = e.event_time {
                        // Convert DateTime<Utc> to milliseconds for comparison
                        let event_ms = event_time.timestamp_millis() as u64;
                        event_ms >= start_ms && event_ms <= end_ms
                    } else {
                        false
                    }
                })
                .collect()
        } else {
            results
        };

        // 6. Sort by updated_at descending and limit
        let mut results = results;
        results.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        results.truncate(options.limit);

        let result = SearchResult::new(results, query, deep_search_used, query_variations);

        // TigerStyle: Postconditions
        debug_assert!(
            result.len() <= options.limit,
            "results exceed limit: {} > {}",
            result.len(),
            options.limit
        );

        Ok(result)
    }

    /// Rewrite query into search variations using LLM.
    ///
    /// # Arguments
    /// - `query` - Original search query
    ///
    /// # Returns
    /// Vector of query variations (always includes original).
    ///
    /// # Graceful Degradation
    /// Returns only the original query if LLM fails.
    pub async fn rewrite_query(&self, query: &str) -> Vec<String> {
        debug_assert!(!query.is_empty(), "query must not be empty");

        let prompt = build_query_rewrite_prompt(query);
        let request = CompletionRequest::new(&prompt).with_json_mode();

        match self.llm.complete(&request).await {
            Ok(response) => self.parse_variations(&response, query),
            Err(_) => {
                // Graceful degradation: return original query only
                vec![query.to_string()]
            }
        }
    }

    /// Parse LLM response into query variations.
    fn parse_variations(&self, response: &str, original_query: &str) -> Vec<String> {
        // Try to parse as JSON array
        let variations: Vec<String> = match serde_json::from_str(response) {
            Ok(v) => v,
            Err(_) => return vec![original_query.to_string()],
        };

        // Filter to valid strings
        let mut valid: Vec<String> = variations
            .into_iter()
            .filter(|v| !v.trim().is_empty())
            .take(RETRIEVAL_QUERY_REWRITE_COUNT_MAX)
            .collect();

        // Always include original query
        if !valid.iter().any(|v| v == original_query) {
            valid.insert(0, original_query.to_string());
        }

        valid.truncate(RETRIEVAL_QUERY_REWRITE_COUNT_MAX);
        valid
    }

    /// Merge results using Reciprocal Rank Fusion.
    ///
    /// RRF score: sum(1 / (k + rank)) for each list the document appears in.
    /// Documents appearing in multiple lists get higher scores.
    ///
    /// # Arguments
    /// - `result_lists` - Slice of entity vectors to merge
    ///
    /// # Returns
    /// Merged and deduplicated entities, sorted by RRF score.
    #[must_use]
    pub fn merge_rrf(&self, result_lists: &[&Vec<Entity>]) -> Vec<Entity> {
        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut entities: HashMap<String, Entity> = HashMap::new();

        for list in result_lists {
            for (rank, entity) in list.iter().enumerate() {
                // RRF formula: 1 / (k + rank)
                // rank is 0-indexed, so rank=0 gives highest score
                *scores.entry(entity.id.clone()).or_default() +=
                    1.0 / (RETRIEVAL_RRF_K as f64 + rank as f64);
                entities
                    .entry(entity.id.clone())
                    .or_insert_with(|| entity.clone());
            }
        }

        // Sort by score descending
        let mut sorted: Vec<_> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Build result list
        sorted
            .into_iter()
            .filter_map(|(id, _)| entities.remove(&id))
            .collect()
    }

    /// Execute fast substring search.
    async fn fast_search(&self, query: &str, limit: usize) -> Result<Vec<Entity>, RetrievalError> {
        self.storage
            .search(query, limit)
            .await
            .map_err(RetrievalError::from)
    }

    /// Execute deep search with query variations.
    async fn deep_search(
        &self,
        variations: &[String],
        original_query: &str,
        limit: usize,
    ) -> Vec<Entity> {
        let mut all_results = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        for variation in variations {
            // Skip if same as original (already searched in fast path)
            if variation == original_query {
                continue;
            }

            // Search this variation
            match self.storage.search(variation, limit).await {
                Ok(results) => {
                    for entity in results {
                        if seen_ids.insert(entity.id.clone()) {
                            all_results.push(entity);
                        }
                    }
                }
                Err(_) => {
                    // Skip failed searches, continue with others
                    continue;
                }
            }
        }

        all_results
    }

    /// Get a reference to the underlying LLM provider.
    #[must_use]
    pub fn llm(&self) -> &L {
        &self.llm
    }

    /// Get a reference to the underlying storage backend.
    #[must_use]
    pub fn storage(&self) -> &S {
        &self.storage
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
    use crate::storage::{Entity, EntityType, SimStorageBackend, StorageBackend};

    async fn create_test_retriever(seed: u64) -> DualRetriever<SimLLMProvider, SimStorageBackend> {
        let llm = SimLLMProvider::with_seed(seed);
        let storage = SimStorageBackend::new(SimConfig::with_seed(seed));
        DualRetriever::new(llm, storage)
    }

    async fn create_test_retriever_with_data(
        seed: u64,
    ) -> DualRetriever<SimLLMProvider, SimStorageBackend> {
        let llm = SimLLMProvider::with_seed(seed);
        let storage = SimStorageBackend::new(SimConfig::with_seed(seed));

        // Add test entities
        storage
            .store_entity(&Entity::new(
                EntityType::Person,
                "Alice".to_string(),
                "Alice works at Acme Corp".to_string(),
            ))
            .await
            .unwrap();
        storage
            .store_entity(&Entity::new(
                EntityType::Person,
                "Bob".to_string(),
                "Bob is a developer at TechCo".to_string(),
            ))
            .await
            .unwrap();
        storage
            .store_entity(&Entity::new(
                EntityType::Note,
                "Meeting".to_string(),
                "Team meeting about project".to_string(),
            ))
            .await
            .unwrap();
        storage
            .store_entity(&Entity::new(
                EntityType::Project,
                "Acme Project".to_string(),
                "Project at Acme Corp".to_string(),
            ))
            .await
            .unwrap();

        DualRetriever::new(llm, storage)
    }

    #[tokio::test]
    async fn test_basic_search() {
        let retriever = create_test_retriever_with_data(42).await;

        let result = retriever
            .search("Alice", SearchOptions::default())
            .await
            .unwrap();

        assert!(!result.is_empty());
        assert_eq!(result.query, "Alice");
    }

    #[tokio::test]
    async fn test_fast_search_only() {
        let retriever = create_test_retriever_with_data(42).await;

        let result = retriever
            .search("Alice", SearchOptions::new().fast_only())
            .await
            .unwrap();

        assert!(!result.deep_search_used);
        assert_eq!(result.query_variations, vec!["Alice"]);
    }

    #[tokio::test]
    async fn test_deep_search_triggered() {
        let retriever = create_test_retriever_with_data(42).await;

        let result = retriever
            .search("Who works at Acme?", SearchOptions::default())
            .await
            .unwrap();

        // Deep search should be used for question word
        assert!(result.deep_search_used);
        assert!(result.query_variations.len() >= 1);
    }

    #[tokio::test]
    async fn test_empty_query_error() {
        let retriever = create_test_retriever(42).await;

        let result = retriever.search("", SearchOptions::default()).await;

        assert!(matches!(result, Err(RetrievalError::EmptyQuery)));
    }

    #[tokio::test]
    async fn test_query_too_long_error() {
        let retriever = create_test_retriever(42).await;

        let long_query = "x".repeat(RETRIEVAL_QUERY_BYTES_MAX + 1);
        let result = retriever
            .search(&long_query, SearchOptions::default())
            .await;

        assert!(matches!(result, Err(RetrievalError::QueryTooLong { .. })));
    }

    #[tokio::test]
    async fn test_invalid_limit_error() {
        let retriever = create_test_retriever(42).await;

        let options = SearchOptions {
            limit: 0,
            deep_search: false,
            time_range: None,
        };
        let result = retriever.search("test", options).await;

        assert!(matches!(result, Err(RetrievalError::InvalidLimit { .. })));
    }

    #[tokio::test]
    async fn test_rewrite_query() {
        let retriever = create_test_retriever(42).await;

        let variations = retriever.rewrite_query("Acme employees").await;

        // Should include original or variations
        assert!(!variations.is_empty());
        assert!(variations.len() <= RETRIEVAL_QUERY_REWRITE_COUNT_MAX);
    }

    #[test]
    fn test_merge_rrf() {
        let retriever = DualRetriever::new(
            SimLLMProvider::with_seed(42),
            SimStorageBackend::new(SimConfig::with_seed(42)),
        );

        let e1 = Entity::new(EntityType::Note, "A".to_string(), "content A".to_string());
        let e2 = Entity::new(EntityType::Note, "B".to_string(), "content B".to_string());
        let e3 = Entity::new(EntityType::Note, "C".to_string(), "content C".to_string());

        let list1 = vec![e1.clone(), e2.clone()];
        let list2 = vec![e2.clone(), e3.clone()];

        let merged = retriever.merge_rrf(&[&list1, &list2]);

        // B appears in both lists, should be ranked higher
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].name, "B"); // Highest RRF score
    }

    #[test]
    fn test_merge_rrf_empty() {
        let retriever = DualRetriever::new(
            SimLLMProvider::with_seed(42),
            SimStorageBackend::new(SimConfig::with_seed(42)),
        );

        let empty: Vec<Entity> = vec![];
        let merged = retriever.merge_rrf(&[&empty, &empty]);

        assert!(merged.is_empty());
    }

    #[test]
    fn test_parse_variations_valid() {
        let retriever = DualRetriever::new(
            SimLLMProvider::with_seed(42),
            SimStorageBackend::new(SimConfig::with_seed(42)),
        );

        let response = r#"["variation 1", "variation 2"]"#;
        let variations = retriever.parse_variations(response, "original");

        assert!(variations.contains(&"original".to_string()));
        assert!(variations.len() <= RETRIEVAL_QUERY_REWRITE_COUNT_MAX);
    }

    #[test]
    fn test_parse_variations_invalid_json() {
        let retriever = DualRetriever::new(
            SimLLMProvider::with_seed(42),
            SimStorageBackend::new(SimConfig::with_seed(42)),
        );

        let response = "not valid json";
        let variations = retriever.parse_variations(response, "original");

        assert_eq!(variations, vec!["original"]);
    }

    #[test]
    fn test_parse_variations_empty_strings() {
        let retriever = DualRetriever::new(
            SimLLMProvider::with_seed(42),
            SimStorageBackend::new(SimConfig::with_seed(42)),
        );

        let response = r#"["", "  ", "valid"]"#;
        let variations = retriever.parse_variations(response, "original");

        // Empty strings should be filtered out
        assert!(!variations.iter().any(|v| v.trim().is_empty()));
    }

    #[tokio::test]
    async fn test_time_range_filter() {
        use chrono::{TimeZone, Utc};

        let llm = SimLLMProvider::with_seed(42);
        let storage = SimStorageBackend::new(SimConfig::with_seed(42));

        // Add entities with different event times
        let mut e1 = Entity::new(EntityType::Note, "Early".to_string(), "content".to_string());
        e1.event_time = Some(Utc.timestamp_millis_opt(1000).unwrap());
        storage.store_entity(&e1).await.unwrap();

        let mut e2 = Entity::new(
            EntityType::Note,
            "Middle".to_string(),
            "content".to_string(),
        );
        e2.event_time = Some(Utc.timestamp_millis_opt(2000).unwrap());
        storage.store_entity(&e2).await.unwrap();

        let mut e3 = Entity::new(EntityType::Note, "Late".to_string(), "content".to_string());
        e3.event_time = Some(Utc.timestamp_millis_opt(3000).unwrap());
        storage.store_entity(&e3).await.unwrap();

        let retriever = DualRetriever::new(llm, storage);

        let options = SearchOptions::new().with_time_range(1500, 2500).fast_only();

        let result = retriever.search("content", options).await.unwrap();

        // Only "Middle" should be in range
        assert_eq!(result.len(), 1);
        assert_eq!(result.entities[0].name, "Middle");
    }

    #[tokio::test]
    async fn test_determinism() {
        let retriever1 = create_test_retriever_with_data(42).await;
        let retriever2 = create_test_retriever_with_data(42).await;

        let result1 = retriever1
            .search("Who works at Acme?", SearchOptions::default())
            .await
            .unwrap();

        let result2 = retriever2
            .search("Who works at Acme?", SearchOptions::default())
            .await
            .unwrap();

        // Same seed should produce same query variations
        assert_eq!(result1.query_variations, result2.query_variations);
    }

    #[tokio::test]
    async fn test_provider_accessors() {
        let retriever = create_test_retriever(42).await;

        assert!(retriever.llm().is_simulation());
        // Storage accessor exists
        let _ = retriever.storage();
    }
}
