//! Memory Configuration
//!
//! `TigerStyle`: Sensible defaults, builder pattern, explicit over implicit.
//!
//! Provides global configuration for Memory system behavior.

use std::time::Duration;

// =============================================================================
// MemoryConfig
// =============================================================================

/// Global configuration for Memory system.
///
/// `TigerStyle`:
/// - Sensible defaults via Default impl
/// - Builder pattern for customization
/// - All fields public for transparency
///
/// # Example
///
/// ```rust
/// use umi_memory::umi::MemoryConfig;
///
/// let config = MemoryConfig::default()
///     .with_recall_limit(20)
///     .without_embeddings();
/// ```
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Core memory size in bytes (always in LLM context).
    ///
    /// Default: 32KB
    pub core_memory_bytes: usize,

    /// Working memory size in bytes (session state with TTL).
    ///
    /// Default: 1MB
    pub working_memory_bytes: usize,

    /// Working memory time-to-live duration.
    ///
    /// Default: 1 hour
    pub working_memory_ttl: Duration,

    /// Whether to generate embeddings for entities.
    ///
    /// Default: true
    pub generate_embeddings: bool,

    /// Embedding batch size for bulk operations.
    ///
    /// Default: 100
    pub embedding_batch_size: usize,

    /// Default recall result limit.
    ///
    /// Default: 10
    pub default_recall_limit: usize,

    /// Whether to enable semantic (vector) search.
    ///
    /// Default: true
    pub semantic_search_enabled: bool,

    /// Whether to enable LLM query expansion for retrieval.
    ///
    /// Default: true (auto-enabled when beneficial)
    pub query_expansion_enabled: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            core_memory_bytes: 32 * 1024,         // 32KB
            working_memory_bytes: 1024 * 1024,    // 1MB
            working_memory_ttl: Duration::from_secs(3600), // 1 hour
            generate_embeddings: true,
            embedding_batch_size: 100,
            default_recall_limit: 10,
            semantic_search_enabled: true,
            query_expansion_enabled: true,
        }
    }
}

impl MemoryConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set core memory size in bytes.
    ///
    /// # Arguments
    /// - `bytes` - Core memory size
    #[must_use]
    pub fn with_core_memory_bytes(mut self, bytes: usize) -> Self {
        self.core_memory_bytes = bytes;
        self
    }

    /// Set working memory size in bytes.
    ///
    /// # Arguments
    /// - `bytes` - Working memory size
    #[must_use]
    pub fn with_working_memory_bytes(mut self, bytes: usize) -> Self {
        self.working_memory_bytes = bytes;
        self
    }

    /// Set working memory TTL duration.
    ///
    /// # Arguments
    /// - `ttl` - Time-to-live duration
    #[must_use]
    pub fn with_working_memory_ttl(mut self, ttl: Duration) -> Self {
        self.working_memory_ttl = ttl;
        self
    }

    /// Set default recall limit.
    ///
    /// # Arguments
    /// - `limit` - Default recall result limit
    #[must_use]
    pub fn with_recall_limit(mut self, limit: usize) -> Self {
        self.default_recall_limit = limit;
        self
    }

    /// Set embedding batch size.
    ///
    /// # Arguments
    /// - `size` - Batch size for embedding operations
    #[must_use]
    pub fn with_embedding_batch_size(mut self, size: usize) -> Self {
        self.embedding_batch_size = size;
        self
    }

    /// Disable embedding generation.
    #[must_use]
    pub fn without_embeddings(mut self) -> Self {
        self.generate_embeddings = false;
        self
    }

    /// Disable semantic (vector) search.
    #[must_use]
    pub fn without_semantic_search(mut self) -> Self {
        self.semantic_search_enabled = false;
        self
    }

    /// Disable query expansion.
    #[must_use]
    pub fn without_query_expansion(mut self) -> Self {
        self.query_expansion_enabled = false;
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let config = MemoryConfig::default();

        assert_eq!(config.core_memory_bytes, 32 * 1024);
        assert_eq!(config.working_memory_bytes, 1024 * 1024);
        assert_eq!(config.default_recall_limit, 10);
        assert_eq!(config.embedding_batch_size, 100);
        assert!(config.generate_embeddings);
        assert!(config.semantic_search_enabled);
        assert!(config.query_expansion_enabled);
    }

    #[test]
    fn test_builder_pattern() {
        let config = MemoryConfig::default()
            .with_core_memory_bytes(64 * 1024)
            .with_recall_limit(20)
            .without_embeddings();

        assert_eq!(config.core_memory_bytes, 64 * 1024);
        assert_eq!(config.default_recall_limit, 20);
        assert!(!config.generate_embeddings);
    }

    #[test]
    fn test_method_chaining() {
        let config = MemoryConfig::new()
            .with_core_memory_bytes(32 * 1024)
            .with_working_memory_bytes(2 * 1024 * 1024)
            .with_recall_limit(15)
            .without_query_expansion();

        assert_eq!(config.core_memory_bytes, 32 * 1024);
        assert_eq!(config.working_memory_bytes, 2 * 1024 * 1024);
        assert_eq!(config.default_recall_limit, 15);
        assert!(!config.query_expansion_enabled);
    }
}
