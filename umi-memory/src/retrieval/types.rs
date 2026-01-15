//! Retrieval Types - Search Options and Results
//!
//! `TigerStyle`: Type-safe options, explicit validation.

use crate::constants::{RETRIEVAL_RESULTS_COUNT_DEFAULT, RETRIEVAL_RESULTS_COUNT_MAX};
use crate::storage::Entity;

// =============================================================================
// Search Options
// =============================================================================

/// Options for search operations.
///
/// `TigerStyle`: Builder pattern with validation.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of results to return.
    pub limit: usize,

    /// Whether to use deep search (LLM query rewriting).
    pub deep_search: bool,

    /// Optional time range filter (`start_ms`, `end_ms`).
    pub time_range: Option<(u64, u64)>,
}

impl SearchOptions {
    /// Create new search options with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the result limit.
    ///
    /// # Arguments
    /// * `limit` - Maximum number of results (must be 1-100)
    ///
    /// # Errors
    /// Returns `RetrievalError::InvalidLimit` if limit is 0 or exceeds 100.
    ///
    /// # Example
    /// ```
    /// use umi_memory::retrieval::SearchOptions;
    ///
    /// let options = SearchOptions::default().with_limit(20).unwrap();
    /// ```
    pub fn with_limit(mut self, limit: usize) -> Result<Self, super::RetrievalError> {
        if limit == 0 || limit > RETRIEVAL_RESULTS_COUNT_MAX {
            return Err(super::RetrievalError::InvalidLimit {
                value: limit,
                max: RETRIEVAL_RESULTS_COUNT_MAX,
            });
        }
        debug_assert!(
            limit > 0 && limit <= RETRIEVAL_RESULTS_COUNT_MAX,
            "limit validation failed"
        );
        self.limit = limit;
        Ok(self)
    }

    /// Enable or disable deep search.
    #[must_use]
    pub fn with_deep_search(mut self, deep_search: bool) -> Self {
        self.deep_search = deep_search;
        self
    }

    /// Set time range filter.
    ///
    /// # Arguments
    /// - `start_ms` - Start time in milliseconds since epoch
    /// - `end_ms` - End time in milliseconds since epoch
    ///
    /// # Panics
    /// Panics if `start_ms` > `end_ms`.
    #[must_use]
    pub fn with_time_range(mut self, start_ms: u64, end_ms: u64) -> Self {
        debug_assert!(
            start_ms <= end_ms,
            "start_ms must be <= end_ms: {start_ms} > {end_ms}"
        );
        self.time_range = Some((start_ms, end_ms));
        self
    }

    /// Disable deep search (fast search only).
    #[must_use]
    pub fn fast_only(mut self) -> Self {
        self.deep_search = false;
        self
    }
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: RETRIEVAL_RESULTS_COUNT_DEFAULT,
            deep_search: true,
            time_range: None,
        }
    }
}

// =============================================================================
// Search Result
// =============================================================================

/// Result from a search operation.
///
/// Contains the matched entities along with metadata about the search.
///
/// **DST-First Discovery**: Added `scores` field to fix recall relevance issue.
/// Previously, similarity scores were computed but not tracked, causing results
/// to be sorted by recency instead of relevance.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matched entities, sorted by relevance.
    pub entities: Vec<Entity>,

    /// Similarity scores for each entity (0.0-1.0, same order as entities).
    ///
    /// Higher scores indicate better matches. Scores are computed from:
    /// - Vector similarity (cosine similarity of embeddings)
    /// - Text match scores (fuzzy matching)
    /// - RRF merge scores (reciprocal rank fusion)
    pub scores: Vec<f64>,

    /// The original query.
    pub query: String,

    /// Whether deep search was actually used.
    pub deep_search_used: bool,

    /// Query variations used (includes original if deep search was used).
    pub query_variations: Vec<String>,
}

impl SearchResult {
    /// Create a new search result.
    #[must_use]
    pub fn new(
        entities: Vec<Entity>,
        scores: Vec<f64>,
        query: impl Into<String>,
        deep_search_used: bool,
        query_variations: Vec<String>,
    ) -> Self {
        debug_assert_eq!(
            entities.len(),
            scores.len(),
            "entities and scores must have same length"
        );
        Self {
            entities,
            scores,
            query: query.into(),
            deep_search_used,
            query_variations,
        }
    }

    /// Create a fast-only search result.
    #[must_use]
    pub fn fast_only(entities: Vec<Entity>, scores: Vec<f64>, query: impl Into<String>) -> Self {
        debug_assert_eq!(
            entities.len(),
            scores.len(),
            "entities and scores must have same length"
        );
        let query = query.into();
        Self {
            entities,
            scores,
            query: query.clone(),
            deep_search_used: false,
            query_variations: vec![query],
        }
    }

    /// Get the number of results.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if results are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get the first entity if any.
    #[must_use]
    pub fn first(&self) -> Option<&Entity> {
        self.entities.first()
    }

    /// Iterate over entities.
    pub fn iter(&self) -> impl Iterator<Item = &Entity> {
        self.entities.iter()
    }
}

impl IntoIterator for SearchResult {
    type Item = Entity;
    type IntoIter = std::vec::IntoIter<Entity>;

    fn into_iter(self) -> Self::IntoIter {
        self.entities.into_iter()
    }
}

// =============================================================================
// Deep Search Trigger Words
// =============================================================================

/// Question words that trigger deep search.
pub const QUESTION_WORDS: &[&str] = &["who", "what", "when", "where", "why", "how"];

/// Relationship terms that trigger deep search.
pub const RELATIONSHIP_TERMS: &[&str] =
    &["related", "about", "regarding", "involving", "connected"];

/// Temporal terms that trigger deep search.
pub const TEMPORAL_TERMS: &[&str] = &[
    "yesterday",
    "today",
    "last",
    "recent",
    "before",
    "after",
    "week",
    "month",
    "year",
];

/// Abstract terms that trigger deep search.
pub const ABSTRACT_TERMS: &[&str] = &["similar", "like", "connections", "associated", "linked"];

/// Check if a query contains any trigger words.
///
/// This is the heuristic that determines if deep search would be beneficial.
#[must_use]
pub fn needs_deep_search(query: &str) -> bool {
    debug_assert!(!query.is_empty(), "query must not be empty");

    let query_lower = query.to_lowercase();

    // Check question words
    for word in QUESTION_WORDS {
        if query_lower.contains(word) {
            return true;
        }
    }

    // Check temporal terms
    for word in TEMPORAL_TERMS {
        if query_lower.contains(word) {
            return true;
        }
    }

    // Check abstract terms
    for word in ABSTRACT_TERMS {
        if query_lower.contains(word) {
            return true;
        }
    }

    // Check relationship terms
    for term in RELATIONSHIP_TERMS {
        if query_lower.contains(term) {
            return true;
        }
    }

    false
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::RetrievalError;
    use crate::storage::{Entity, EntityType};

    #[test]
    fn test_search_options_default() {
        let options = SearchOptions::default();

        assert_eq!(options.limit, RETRIEVAL_RESULTS_COUNT_DEFAULT);
        assert!(options.deep_search);
        assert!(options.time_range.is_none());
    }

    #[test]
    fn test_search_options_builder() {
        let options = SearchOptions::new()
            .with_limit(50)
            .unwrap()
            .with_deep_search(false)
            .with_time_range(1000, 2000);

        assert_eq!(options.limit, 50);
        assert!(!options.deep_search);
        assert_eq!(options.time_range, Some((1000, 2000)));
    }

    #[test]
    fn test_search_options_fast_only() {
        let options = SearchOptions::new().fast_only();

        assert!(!options.deep_search);
    }

    #[test]
    fn test_search_options_invalid_limit_zero() {
        let result = SearchOptions::new().with_limit(0);
        assert!(result.is_err());
        match result {
            Err(RetrievalError::InvalidLimit { value, max }) => {
                assert_eq!(value, 0);
                assert_eq!(max, RETRIEVAL_RESULTS_COUNT_MAX);
            }
            _ => panic!("Expected InvalidLimit error"),
        }
    }

    #[test]
    fn test_search_options_invalid_limit_too_large() {
        let result = SearchOptions::new().with_limit(RETRIEVAL_RESULTS_COUNT_MAX + 1);
        assert!(result.is_err());
        match result {
            Err(RetrievalError::InvalidLimit { value, max }) => {
                assert_eq!(value, RETRIEVAL_RESULTS_COUNT_MAX + 1);
                assert_eq!(max, RETRIEVAL_RESULTS_COUNT_MAX);
            }
            _ => panic!("Expected InvalidLimit error"),
        }
    }

    #[test]
    #[should_panic(expected = "start_ms must be")]
    fn test_search_options_invalid_time_range() {
        let _ = SearchOptions::new().with_time_range(2000, 1000);
    }

    #[test]
    fn test_search_result_new() {
        let entities = vec![Entity::new(
            EntityType::Note,
            "test".to_string(),
            "content".to_string(),
        )];
        let result = SearchResult::new(
            entities,
            vec![1.0], // dummy score for testing
            "query",
            true,
            vec!["query".to_string(), "variation".to_string()],
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result.query, "query");
        assert!(result.deep_search_used);
        assert_eq!(result.query_variations.len(), 2);
    }

    #[test]
    fn test_search_result_fast_only() {
        let entities = vec![Entity::new(
            EntityType::Note,
            "test".to_string(),
            "content".to_string(),
        )];
        let result = SearchResult::fast_only(entities, vec![1.0], "query");

        assert_eq!(result.len(), 1);
        assert!(!result.deep_search_used);
        assert_eq!(result.query_variations, vec!["query"]);
    }

    #[test]
    fn test_search_result_empty() {
        let result = SearchResult::fast_only(vec![], vec![], "query");

        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
        assert!(result.first().is_none());
    }

    #[test]
    fn test_search_result_iter() {
        let entities = vec![
            Entity::new(EntityType::Note, "a".to_string(), "content a".to_string()),
            Entity::new(EntityType::Note, "b".to_string(), "content b".to_string()),
        ];
        let result = SearchResult::fast_only(entities, vec![1.0, 1.0], "query");

        let names: Vec<_> = result.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_needs_deep_search_question_words() {
        assert!(needs_deep_search("Who works at Acme?"));
        assert!(needs_deep_search("What is the project about?"));
        assert!(needs_deep_search("When did we meet?"));
        assert!(needs_deep_search("Where is the office?"));
        assert!(needs_deep_search("Why did that happen?"));
        assert!(needs_deep_search("How does it work?"));
    }

    #[test]
    fn test_needs_deep_search_temporal_terms() {
        assert!(needs_deep_search("yesterday's meeting"));
        assert!(needs_deep_search("What happened today?"));
        assert!(needs_deep_search("last week's notes"));
        assert!(needs_deep_search("recent updates"));
        assert!(needs_deep_search("before the deadline"));
        assert!(needs_deep_search("after the conference"));
    }

    #[test]
    fn test_needs_deep_search_abstract_terms() {
        assert!(needs_deep_search("similar projects"));
        assert!(needs_deep_search("something like that"));
        assert!(needs_deep_search("connections to Acme"));
        assert!(needs_deep_search("associated topics"));
        assert!(needs_deep_search("linked issues"));
    }

    #[test]
    fn test_needs_deep_search_relationship_terms() {
        assert!(needs_deep_search("related to Alice"));
        assert!(needs_deep_search("about the meeting"));
        assert!(needs_deep_search("regarding the project"));
        assert!(needs_deep_search("involving customers"));
        assert!(needs_deep_search("connected to sales"));
    }

    #[test]
    fn test_needs_deep_search_simple_query() {
        // Simple queries don't trigger deep search
        assert!(!needs_deep_search("Alice"));
        assert!(!needs_deep_search("Acme Corp"));
        assert!(!needs_deep_search("project status"));
        assert!(!needs_deep_search("meeting notes"));
    }

    #[test]
    fn test_needs_deep_search_case_insensitive() {
        assert!(needs_deep_search("WHO works here?"));
        assert!(needs_deep_search("YESTERDAY"));
        assert!(needs_deep_search("Similar items"));
    }
}
