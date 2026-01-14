//! Vector Backend - Embedding Storage and Similarity Search (ADR-006)
//!
//! `TigerStyle`: Trait-based abstraction, simulation-first testing.
//!
//! # Overview
//!
//! Stores vector embeddings and enables similarity search.
//! Used for semantic search in dual retrieval strategy.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    VectorBackend Trait                       │
//! └─────────────────────────────────────────────────────────────┘
//!          ↑                              ↑
//!          │                              │
//! ┌────────┴────────┐           ┌────────┴────────┐
//! │SimVectorBackend │           │ QdrantBackend   │
//! │   (testing)     │           │  (production)   │
//! └─────────────────┘           └─────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::constants::EMBEDDING_DIMENSIONS_COUNT;
use crate::dst::{DeterministicRng, FaultInjector};
use crate::storage::{StorageError, StorageResult};

// =============================================================================
// Vector Backend Trait
// =============================================================================

/// Result of a similarity search.
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// Entity ID
    pub id: String,
    /// Similarity score (0.0 to 1.0, higher = more similar)
    pub score: f32,
}

/// Trait for vector embedding storage backends.
#[async_trait]
pub trait VectorBackend: Send + Sync + std::fmt::Debug + 'static {
    /// Store an embedding for an entity.
    ///
    /// # Arguments
    /// * `id` - Entity ID to associate with the embedding
    /// * `embedding` - Vector embedding (must match EMBEDDING_DIMENSIONS_COUNT)
    async fn store(&self, id: &str, embedding: &[f32]) -> StorageResult<()>;

    /// Search for similar embeddings.
    ///
    /// # Arguments
    /// * `embedding` - Query embedding
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    /// Results sorted by similarity (highest first)
    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> StorageResult<Vec<VectorSearchResult>>;

    /// Delete an embedding.
    async fn delete(&self, id: &str) -> StorageResult<()>;

    /// Check if an embedding exists.
    async fn exists(&self, id: &str) -> StorageResult<bool>;

    /// Get the embedding for an entity.
    async fn get(&self, id: &str) -> StorageResult<Option<Vec<f32>>>;

    /// Get the number of stored embeddings.
    async fn count(&self) -> StorageResult<usize>;
}

// =============================================================================
// Simulated Vector Backend (for DST)
// =============================================================================

/// In-memory vector backend for deterministic simulation testing.
///
/// Features:
/// - Deterministic similarity computation
/// - Fault injection support
/// - No external dependencies
#[derive(Clone, Debug)]
pub struct SimVectorBackend {
    /// Stored embeddings
    embeddings: Arc<std::sync::RwLock<HashMap<String, Vec<f32>>>>,
    /// Fault injector for testing error paths
    fault_injector: Option<Arc<FaultInjector>>,
    /// RNG for deterministic behavior
    _rng: Arc<std::sync::RwLock<DeterministicRng>>,
}

impl SimVectorBackend {
    /// Create a new simulated vector backend.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            embeddings: Arc::new(std::sync::RwLock::new(HashMap::new())),
            fault_injector: None,
            _rng: Arc::new(std::sync::RwLock::new(DeterministicRng::new(seed))),
        }
    }

    /// Create with fault injection enabled.
    #[must_use]
    pub fn with_faults(seed: u64, fault_injector: Arc<FaultInjector>) -> Self {
        Self {
            embeddings: Arc::new(std::sync::RwLock::new(HashMap::new())),
            fault_injector: Some(fault_injector),
            _rng: Arc::new(std::sync::RwLock::new(DeterministicRng::new(seed))),
        }
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        // Preconditions
        assert_eq!(a.len(), b.len(), "vectors must have same length");
        assert!(!a.is_empty(), "vectors must not be empty");

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        let similarity = dot / (norm_a * norm_b);

        // Postcondition: similarity is in [-1, 1], normalize to [0, 1]
        (similarity + 1.0) / 2.0
    }

    /// Check if a fault should be injected.
    fn should_inject_fault(&self, operation: &str) -> bool {
        if let Some(ref injector) = self.fault_injector {
            injector.should_inject(operation).is_some()
        } else {
            false
        }
    }
}

#[async_trait]
impl VectorBackend for SimVectorBackend {
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

        // Fault injection
        if self.should_inject_fault("vector_store_fail") {
            return Err(StorageError::write("Injected fault: vector store failed"));
        }

        let mut embeddings = self.embeddings.write().unwrap();
        embeddings.insert(id.to_string(), embedding.to_vec());

        // Postcondition
        assert!(embeddings.contains_key(id), "embedding must be stored");
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

        // Fault injection
        if self.should_inject_fault("vector_search_timeout") {
            return Err(StorageError::timeout(5000)); // 5 second timeout
        }
        if self.should_inject_fault("vector_search_fail") {
            return Err(StorageError::read("Injected fault: vector search failed"));
        }

        let embeddings = self.embeddings.read().unwrap();

        // Compute similarities
        let mut results: Vec<VectorSearchResult> = embeddings
            .iter()
            .map(|(id, stored)| VectorSearchResult {
                id: id.clone(),
                score: Self::cosine_similarity(embedding, stored),
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Limit results
        results.truncate(limit);

        // Postcondition
        assert!(results.len() <= limit, "results must not exceed limit");
        Ok(results)
    }

    async fn delete(&self, id: &str) -> StorageResult<()> {
        // Precondition
        assert!(!id.is_empty(), "id must not be empty");

        // Fault injection
        if self.should_inject_fault("vector_delete") {
            return Err(StorageError::write("Injected fault: vector delete failed"));
        }

        let mut embeddings = self.embeddings.write().unwrap();
        embeddings.remove(id);

        // Postcondition
        assert!(!embeddings.contains_key(id), "embedding must be deleted");
        Ok(())
    }

    async fn exists(&self, id: &str) -> StorageResult<bool> {
        // Precondition
        assert!(!id.is_empty(), "id must not be empty");

        // Fault injection
        if self.should_inject_fault("vector_exists") {
            return Err(StorageError::read(
                "Injected fault: vector exists check failed",
            ));
        }

        let embeddings = self.embeddings.read().unwrap();
        Ok(embeddings.contains_key(id))
    }

    async fn get(&self, id: &str) -> StorageResult<Option<Vec<f32>>> {
        // Precondition
        assert!(!id.is_empty(), "id must not be empty");

        // Fault injection
        if self.should_inject_fault("vector_get") {
            return Err(StorageError::read("Injected fault: vector get failed"));
        }

        let embeddings = self.embeddings.read().unwrap();
        Ok(embeddings.get(id).cloned())
    }

    async fn count(&self) -> StorageResult<usize> {
        // Fault injection
        if self.should_inject_fault("vector_count") {
            return Err(StorageError::read("Injected fault: vector count failed"));
        }

        let embeddings = self.embeddings.read().unwrap();
        Ok(embeddings.len())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test embedding with specified seed for reproducibility.
    fn make_embedding(seed: u64) -> Vec<f32> {
        let mut rng = DeterministicRng::new(seed);
        (0..EMBEDDING_DIMENSIONS_COUNT)
            .map(|_| (rng.next_float() * 2.0 - 1.0) as f32) // Values in [-1, 1]
            .collect()
    }

    // =========================================================================
    // SimVectorBackend Tests
    // =========================================================================

    #[tokio::test]
    async fn test_sim_vector_store_and_get() {
        let backend = SimVectorBackend::new(42);
        let embedding = make_embedding(1);

        // Store
        backend.store("entity-1", &embedding).await.unwrap();

        // Get
        let retrieved = backend.get("entity-1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), embedding);
    }

    #[tokio::test]
    async fn test_sim_vector_exists() {
        let backend = SimVectorBackend::new(42);
        let embedding = make_embedding(1);

        // Not exists initially
        assert!(!backend.exists("entity-1").await.unwrap());

        // Store
        backend.store("entity-1", &embedding).await.unwrap();

        // Now exists
        assert!(backend.exists("entity-1").await.unwrap());
    }

    #[tokio::test]
    async fn test_sim_vector_delete() {
        let backend = SimVectorBackend::new(42);
        let embedding = make_embedding(1);

        // Store
        backend.store("entity-1", &embedding).await.unwrap();
        assert!(backend.exists("entity-1").await.unwrap());

        // Delete
        backend.delete("entity-1").await.unwrap();
        assert!(!backend.exists("entity-1").await.unwrap());
    }

    #[tokio::test]
    async fn test_sim_vector_count() {
        let backend = SimVectorBackend::new(42);

        assert_eq!(backend.count().await.unwrap(), 0);

        backend.store("e1", &make_embedding(1)).await.unwrap();
        assert_eq!(backend.count().await.unwrap(), 1);

        backend.store("e2", &make_embedding(2)).await.unwrap();
        assert_eq!(backend.count().await.unwrap(), 2);

        backend.delete("e1").await.unwrap();
        assert_eq!(backend.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_sim_vector_search_finds_similar() {
        let backend = SimVectorBackend::new(42);

        // Store some embeddings
        let base = make_embedding(100);
        backend.store("base", &base).await.unwrap();

        // Store a similar embedding (slightly modified)
        let mut similar = base.clone();
        similar[0] += 0.01;
        similar[1] -= 0.01;
        backend.store("similar", &similar).await.unwrap();

        // Store a different embedding
        let different = make_embedding(999);
        backend.store("different", &different).await.unwrap();

        // Search with base embedding
        let results = backend.search(&base, 3).await.unwrap();

        // Should find base first (exact match)
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "base");
        assert!((results[0].score - 1.0).abs() < 0.001); // Exact match = 1.0

        // Similar should be second
        assert_eq!(results[1].id, "similar");
        assert!(results[1].score > 0.99); // Very similar
    }

    #[tokio::test]
    async fn test_sim_vector_search_respects_limit() {
        let backend = SimVectorBackend::new(42);

        // Store 10 embeddings
        for i in 0..10 {
            backend
                .store(&format!("e{i}"), &make_embedding(i))
                .await
                .unwrap();
        }

        // Search with limit 3
        let results = backend.search(&make_embedding(0), 3).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_sim_vector_search_sorted_by_score() {
        let backend = SimVectorBackend::new(42);

        // Store multiple embeddings
        for i in 0..5 {
            backend
                .store(&format!("e{i}"), &make_embedding(i))
                .await
                .unwrap();
        }

        // Search
        let results = backend.search(&make_embedding(0), 5).await.unwrap();

        // Verify sorted by score descending
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "results must be sorted by score descending"
            );
        }
    }

    // =========================================================================
    // Cosine Similarity Tests
    // =========================================================================

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let similarity = SimVectorBackend::cosine_similarity(&v, &v);
        // Normalized to [0, 1], so identical = 1.0
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![-1.0, 0.0, 0.0];
        let similarity = SimVectorBackend::cosine_similarity(&v1, &v2);
        // Opposite vectors = 0.0 after normalization to [0, 1]
        assert!(similarity.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let similarity = SimVectorBackend::cosine_similarity(&v1, &v2);
        // Orthogonal = 0.5 after normalization to [0, 1]
        assert!((similarity - 0.5).abs() < 0.001);
    }

    // =========================================================================
    // Precondition Tests
    // =========================================================================

    #[tokio::test]
    #[should_panic(expected = "id must not be empty")]
    async fn test_sim_vector_store_empty_id() {
        let backend = SimVectorBackend::new(42);
        let _ = backend.store("", &make_embedding(1)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "embedding must have")]
    async fn test_sim_vector_store_wrong_dimensions() {
        let backend = SimVectorBackend::new(42);
        let wrong_size = vec![1.0, 2.0, 3.0]; // Wrong dimension
        let _ = backend.store("entity-1", &wrong_size).await;
    }

    #[tokio::test]
    #[should_panic(expected = "limit must be positive")]
    async fn test_sim_vector_search_zero_limit() {
        let backend = SimVectorBackend::new(42);
        let _ = backend.search(&make_embedding(1), 0).await;
    }

    // =========================================================================
    // Determinism Tests (DST)
    // =========================================================================

    #[tokio::test]
    async fn test_sim_vector_deterministic() {
        // Same seed should produce same results
        async fn run_operations(seed: u64) -> Vec<VectorSearchResult> {
            let backend = SimVectorBackend::new(seed);

            backend.store("e1", &make_embedding(1)).await.unwrap();
            backend.store("e2", &make_embedding(2)).await.unwrap();
            backend.store("e3", &make_embedding(3)).await.unwrap();

            backend.search(&make_embedding(1), 3).await.unwrap()
        }

        let results1 = run_operations(42).await;
        let results2 = run_operations(42).await;

        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.id, r2.id);
            assert!((r1.score - r2.score).abs() < f32::EPSILON);
        }
    }
}
