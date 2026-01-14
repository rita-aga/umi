//! Memory Builder Pattern
//!
//! `TigerStyle`: Clean API, sensible defaults, fail fast.
//!
//! Provides a builder pattern for constructing Memory instances with
//! explicit component configuration.

use super::Memory;
use crate::embedding::EmbeddingProvider;
use crate::llm::LLMProvider;
use crate::storage::{StorageBackend, VectorBackend};

// =============================================================================
// MemoryBuilder
// =============================================================================

/// Builder for constructing Memory instances.
///
/// `TigerStyle`:
/// - Fluent API with method chaining
/// - Panics on `build()` if required components missing (fail fast)
/// - All components required (no defaults)
///
/// # Example
///
/// ```rust,ignore
/// use umi_memory::umi::Memory;
/// use umi_memory::{SimLLMProvider, SimEmbeddingProvider};
/// use umi_memory::storage::{SimVectorBackend, SimStorageBackend, SimConfig};
///
/// let memory = Memory::builder()
///     .with_llm(Box::new(SimLLMProvider::with_seed(42)))
///     .with_embedder(Box::new(SimEmbeddingProvider::with_seed(42)))
///     .with_vector(Box::new(SimVectorBackend::new(42)))
///     .with_storage(Box::new(SimStorageBackend::new(SimConfig::with_seed(42))))
///     .build();
/// ```
pub struct MemoryBuilder {
    llm: Option<Box<dyn LLMProvider>>,
    embedder: Option<Box<dyn EmbeddingProvider>>,
    vector: Option<Box<dyn VectorBackend>>,
    storage: Option<Box<dyn StorageBackend>>,
}

impl MemoryBuilder {
    /// Create a new builder with no components set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            llm: None,
            embedder: None,
            vector: None,
            storage: None,
        }
    }

    /// Set the LLM provider.
    ///
    /// # Arguments
    /// - `llm` - LLM provider for extraction, retrieval, evolution
    #[must_use]
    pub fn with_llm(mut self, llm: Box<dyn LLMProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the embedding provider.
    ///
    /// # Arguments
    /// - `embedder` - Embedding provider for generating vector embeddings
    #[must_use]
    pub fn with_embedder(mut self, embedder: Box<dyn EmbeddingProvider>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set the vector backend.
    ///
    /// # Arguments
    /// - `vector` - Vector backend for similarity search
    #[must_use]
    pub fn with_vector(mut self, vector: Box<dyn VectorBackend>) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Set the storage backend.
    ///
    /// # Arguments
    /// - `storage` - Storage backend for entity persistence
    #[must_use]
    pub fn with_storage(mut self, storage: Box<dyn StorageBackend>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Build the Memory instance.
    ///
    /// # Panics
    /// Panics if any required component is not set (fail fast).
    ///
    /// # Returns
    /// Constructed Memory instance
    #[must_use]
    pub fn build(self) -> Memory {
        let llm = self.llm.expect("LLM provider is required");
        let embedder = self.embedder.expect("Embedder is required");
        let vector = self.vector.expect("Vector backend is required");
        let storage = self.storage.expect("Storage backend is required");

        Memory::from_boxed(llm, embedder, vector, storage)
    }
}

impl Default for MemoryBuilder {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn test_builder_construction() {
        let builder = MemoryBuilder::<
            SimLLMProvider,
            SimEmbeddingProvider,
            SimVectorBackend,
            SimStorageBackend,
        >::new();

        // Builder should be created
        assert!(builder.llm.is_none());
        assert!(builder.embedder.is_none());
        assert!(builder.vector.is_none());
        assert!(builder.storage.is_none());
    }

    #[test]
    fn test_builder_with_methods() {
        let llm = SimLLMProvider::with_seed(42);
        let embedder = SimEmbeddingProvider::with_seed(42);
        let vector = SimVectorBackend::new(42);
        let storage = SimStorageBackend::new(SimConfig::with_seed(42));

        let builder = MemoryBuilder::new()
            .with_llm(llm)
            .with_embedder(embedder)
            .with_vector(vector)
            .with_storage(storage);

        // All components should be set
        assert!(builder.llm.is_some());
        assert!(builder.embedder.is_some());
        assert!(builder.vector.is_some());
        assert!(builder.storage.is_some());
    }

    #[test]
    #[should_panic(expected = "LLM provider is required")]
    fn test_builder_missing_llm() {
        use crate::umi::Memory;
        let _memory: Memory<
            SimLLMProvider,
            SimEmbeddingProvider,
            SimStorageBackend,
            SimVectorBackend,
        > = MemoryBuilder::new()
            .with_embedder(SimEmbeddingProvider::with_seed(42))
            .with_vector(SimVectorBackend::new(42))
            .with_storage(SimStorageBackend::new(SimConfig::with_seed(42)))
            .build();
    }

    #[test]
    #[should_panic(expected = "Embedder is required")]
    fn test_builder_missing_embedder() {
        use crate::umi::Memory;
        let _memory: Memory<
            SimLLMProvider,
            SimEmbeddingProvider,
            SimStorageBackend,
            SimVectorBackend,
        > = MemoryBuilder::new()
            .with_llm(SimLLMProvider::with_seed(42))
            .with_vector(SimVectorBackend::new(42))
            .with_storage(SimStorageBackend::new(SimConfig::with_seed(42)))
            .build();
    }
}
