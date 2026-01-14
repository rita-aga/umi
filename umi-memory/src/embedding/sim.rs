//! Simulated Embedding Provider for Deterministic Testing
//!
//! `TigerStyle`: Deterministic, reproducible embeddings for DST.
//!
//! # Overview
//!
//! `SimEmbeddingProvider` generates embeddings deterministically:
//! - Same text + same seed = same embedding (always)
//! - No external API calls
//! - Perfect for testing and reproducibility
//!
//! # Algorithm
//!
//! 1. Hash text + seed to get base seed
//! 2. Use `DeterministicRng` to generate random floats in [-1, 1]
//! 3. Normalize to unit vector (L2 norm = 1)
//! 4. Return consistent 1536-dimensional embedding
//!
//! # Example
//!
//! ```rust
//! use umi_memory::embedding::{EmbeddingProvider, SimEmbeddingProvider};
//!
//! #[tokio::main]
//! async fn main() {
//!     let provider = SimEmbeddingProvider::with_seed(42);
//!
//!     let emb1 = provider.embed("Alice works at Acme").await.unwrap();
//!     let emb2 = provider.embed("Alice works at Acme").await.unwrap();
//!
//!     // Same text = same embedding
//!     assert_eq!(emb1, emb2);
//! }
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use std::sync::Arc;

use async_trait::async_trait;

use super::{EmbeddingError, EmbeddingProvider};
use crate::constants::EMBEDDING_DIMENSIONS_COUNT;
use crate::dst::{DeterministicRng, FaultInjector};

// =============================================================================
// SimEmbeddingProvider
// =============================================================================

/// In-memory embedding provider for deterministic simulation testing.
///
/// Features:
/// - Deterministic: same text + same seed = same embedding
/// - No external dependencies
/// - Fast (no network calls)
/// - Normalized embeddings (unit vectors)
/// - Fault injection support for DST
#[derive(Clone, Debug)]
pub struct SimEmbeddingProvider {
    /// Base seed for RNG
    seed: u64,
    /// Embedding dimensions
    dimensions: usize,
    /// Fault injector (optional for DST)
    fault_injector: Option<Arc<FaultInjector>>,
}

impl SimEmbeddingProvider {
    /// Create a new simulated embedding provider with the given seed.
    ///
    /// # Arguments
    /// * `seed` - Base seed for deterministic generation
    ///
    /// # Example
    ///
    /// ```rust
    /// use umi_memory::embedding::SimEmbeddingProvider;
    ///
    /// let provider = SimEmbeddingProvider::new(42);
    /// ```
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            dimensions: EMBEDDING_DIMENSIONS_COUNT,
            fault_injector: None,
        }
    }

    /// Create with explicit seed (alias for `new`).
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self::new(seed)
    }

    /// Create with fault injection enabled.
    #[must_use]
    pub fn with_faults(seed: u64, fault_injector: Arc<FaultInjector>) -> Self {
        Self {
            seed,
            dimensions: EMBEDDING_DIMENSIONS_COUNT,
            fault_injector: Some(fault_injector),
        }
    }

    /// Check if a fault should be injected.
    fn should_inject_fault(&self, operation: &str) -> bool {
        if let Some(ref injector) = self.fault_injector {
            injector.should_inject(operation).is_some()
        } else {
            false
        }
    }

    /// Hash text to get a deterministic seed.
    ///
    /// Combines the base seed with text hash for consistent results.
    fn hash_text(&self, text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Generate a deterministic embedding for text.
    ///
    /// Algorithm:
    /// 1. Hash text + seed
    /// 2. Generate N random floats in [-1, 1]
    /// 3. Normalize to unit vector
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        // Hash text to get deterministic seed
        let text_seed = self.hash_text(text);
        let mut rng = DeterministicRng::new(text_seed);

        // Generate random values in [-1, 1]
        let mut embedding: Vec<f32> = (0..self.dimensions)
            .map(|_| {
                let val = rng.next_float();
                (val * 2.0 - 1.0) as f32 // Map [0, 1] to [-1, 1]
            })
            .collect();

        // Normalize to unit vector
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        // Postcondition: embedding is normalized
        debug_assert!(
            {
                let check_norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                (check_norm - 1.0).abs() < 0.001
            },
            "embedding must be normalized to unit vector"
        );
        debug_assert_eq!(
            embedding.len(),
            self.dimensions,
            "embedding must have correct dimensions"
        );

        embedding
    }
}

#[async_trait]
impl EmbeddingProvider for SimEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Precondition: text must not be empty
        if text.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Fault injection
        if self.should_inject_fault("embedding_timeout") {
            return Err(EmbeddingError::Timeout);
        }
        if self.should_inject_fault("embedding_rate_limit") {
            return Err(EmbeddingError::rate_limit(Some(60)));
        }
        if self.should_inject_fault("embedding_service_unavailable") {
            return Err(EmbeddingError::service_unavailable("Simulated failure"));
        }

        Ok(self.generate_embedding(text))
    }

    #[tracing::instrument(skip(self, texts), fields(batch_size = texts.len()))]
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // Precondition: batch must not be empty
        if texts.is_empty() {
            return Err(EmbeddingError::invalid_request("batch cannot be empty"));
        }

        // Fault injection (same as embed)
        if self.should_inject_fault("embedding_timeout") {
            return Err(EmbeddingError::Timeout);
        }
        if self.should_inject_fault("embedding_rate_limit") {
            return Err(EmbeddingError::rate_limit(Some(60)));
        }
        if self.should_inject_fault("embedding_service_unavailable") {
            return Err(EmbeddingError::service_unavailable("Simulated failure"));
        }

        // Generate embedding for each text
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            if text.is_empty() {
                return Err(EmbeddingError::EmptyInput);
            }
            embeddings.push(self.generate_embedding(text));
        }

        // Postcondition: same number of embeddings as inputs
        debug_assert_eq!(
            embeddings.len(),
            texts.len(),
            "must return one embedding per input"
        );

        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &'static str {
        "sim-embedding"
    }

    fn is_simulation(&self) -> bool {
        true
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sim_embedding_basic() {
        let provider = SimEmbeddingProvider::new(42);
        let embedding = provider.embed("Alice works at Acme").await.unwrap();

        assert_eq!(embedding.len(), EMBEDDING_DIMENSIONS_COUNT);
    }

    #[tokio::test]
    async fn test_sim_embedding_deterministic() {
        let provider = SimEmbeddingProvider::new(42);

        let emb1 = provider.embed("Alice works at Acme").await.unwrap();
        let emb2 = provider.embed("Alice works at Acme").await.unwrap();

        // Same text should produce identical embeddings
        assert_eq!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_sim_embedding_different_text() {
        let provider = SimEmbeddingProvider::new(42);

        let emb1 = provider.embed("Alice works at Acme").await.unwrap();
        let emb2 = provider.embed("Bob works at TechCo").await.unwrap();

        // Different text should produce different embeddings
        assert_ne!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_sim_embedding_different_seed() {
        let provider1 = SimEmbeddingProvider::new(42);
        let provider2 = SimEmbeddingProvider::new(99);

        let emb1 = provider1.embed("Alice works at Acme").await.unwrap();
        let emb2 = provider2.embed("Alice works at Acme").await.unwrap();

        // Different seed should produce different embeddings
        assert_ne!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_sim_embedding_normalized() {
        let provider = SimEmbeddingProvider::new(42);
        let embedding = provider.embed("Alice works at Acme").await.unwrap();

        // Check that embedding is normalized (L2 norm ≈ 1.0)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001, "embedding must be normalized");
    }

    #[tokio::test]
    async fn test_sim_embedding_empty_text() {
        let provider = SimEmbeddingProvider::new(42);
        let result = provider.embed("").await;

        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }

    #[tokio::test]
    async fn test_sim_embedding_batch() {
        let provider = SimEmbeddingProvider::new(42);
        let texts = vec!["Alice works at Acme", "Bob works at TechCo"];

        let embeddings = provider.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), EMBEDDING_DIMENSIONS_COUNT);
        assert_eq!(embeddings[1].len(), EMBEDDING_DIMENSIONS_COUNT);

        // Should match individual embeds
        let single1 = provider.embed(texts[0]).await.unwrap();
        let single2 = provider.embed(texts[1]).await.unwrap();

        assert_eq!(embeddings[0], single1);
        assert_eq!(embeddings[1], single2);
    }

    #[tokio::test]
    async fn test_sim_embedding_batch_empty() {
        let provider = SimEmbeddingProvider::new(42);
        let texts: Vec<&str> = vec![];

        let result = provider.embed_batch(&texts).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_sim_embedding_batch_with_empty_text() {
        let provider = SimEmbeddingProvider::new(42);
        let texts = vec!["Alice", ""];

        let result = provider.embed_batch(&texts).await;
        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }

    #[tokio::test]
    async fn test_sim_embedding_provider_traits() {
        let provider = SimEmbeddingProvider::new(42);

        assert_eq!(provider.dimensions(), EMBEDDING_DIMENSIONS_COUNT);
        assert_eq!(provider.name(), "sim-embedding");
        assert!(provider.is_simulation());
    }

    // =========================================================================
    // Determinism Property Tests
    // =========================================================================

    #[tokio::test]
    async fn test_determinism_same_seed_same_results() {
        async fn run_with_seed(seed: u64) -> Vec<f32> {
            let provider = SimEmbeddingProvider::new(seed);
            provider.embed("test text").await.unwrap()
        }

        let result1 = run_with_seed(42).await;
        let result2 = run_with_seed(42).await;

        assert_eq!(result1, result2, "same seed must produce same results");
    }

    #[tokio::test]
    async fn test_determinism_different_seed_different_results() {
        let provider1 = SimEmbeddingProvider::new(42);
        let provider2 = SimEmbeddingProvider::new(43);

        let result1 = provider1.embed("test text").await.unwrap();
        let result2 = provider2.embed("test text").await.unwrap();

        assert_ne!(
            result1, result2,
            "different seeds must produce different results"
        );
    }

    #[tokio::test]
    async fn test_batch_determinism() {
        let provider = SimEmbeddingProvider::new(42);
        let texts = vec!["text1", "text2", "text3"];

        let batch1 = provider.embed_batch(&texts).await.unwrap();
        let batch2 = provider.embed_batch(&texts).await.unwrap();

        assert_eq!(batch1, batch2, "batch must be deterministic");
    }

    // =========================================================================
    // Normalization Property Tests
    // =========================================================================

    #[tokio::test]
    async fn test_all_embeddings_normalized() {
        let provider = SimEmbeddingProvider::new(42);
        let texts = vec![
            "short",
            "longer text here",
            "even longer text with more words to test different lengths",
        ];

        for text in texts {
            let embedding = provider.embed(text).await.unwrap();
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.001,
                "embedding for '{text}' must be normalized, got norm {norm}"
            );
        }
    }

    // =========================================================================
    // Hash Function Tests
    // =========================================================================

    #[test]
    fn test_hash_text_deterministic() {
        let provider = SimEmbeddingProvider::new(42);

        let hash1 = provider.hash_text("test");
        let hash2 = provider.hash_text("test");

        assert_eq!(hash1, hash2, "hash must be deterministic");
    }

    #[test]
    fn test_hash_text_different_text() {
        let provider = SimEmbeddingProvider::new(42);

        let hash1 = provider.hash_text("test1");
        let hash2 = provider.hash_text("test2");

        assert_ne!(hash1, hash2, "different text must produce different hashes");
    }

    #[test]
    fn test_hash_text_different_seed() {
        let provider1 = SimEmbeddingProvider::new(42);
        let provider2 = SimEmbeddingProvider::new(99);

        let hash1 = provider1.hash_text("test");
        let hash2 = provider2.hash_text("test");

        assert_ne!(hash1, hash2, "different seed must produce different hashes");
    }
}
