//! DST Tests for Memory Builder Pattern
//!
//! TigerStyle: Tests written FIRST, then implementation.
//!
//! These tests define the contract for MemoryBuilder before it exists.
//! They will fail to compile until MemoryBuilder is implemented.

use umi_memory::dst::SimConfig;
use umi_memory::embedding::SimEmbeddingProvider;
use umi_memory::llm::SimLLMProvider;
use umi_memory::storage::{SimStorageBackend, SimVectorBackend};
use umi_memory::umi::{Memory, RecallOptions, RememberOptions};

// =============================================================================
// Basic Construction Tests
// =============================================================================

#[tokio::test]
async fn test_builder_basic_construction() {
    // Test that builder pattern works for basic construction
    let mut memory = Memory::builder()
        .with_llm(SimLLMProvider::with_seed(42))
        .with_embedder(SimEmbeddingProvider::with_seed(42))
        .with_vector(SimVectorBackend::new(42))
        .with_storage(SimStorageBackend::new(SimConfig::with_seed(42)))
        .build();

    // Should be able to use memory
    let result = memory
        .remember("test", RememberOptions::default())
        .await
        .unwrap();
    assert!(!result.entities.is_empty());
}

#[tokio::test]
async fn test_builder_behaves_like_new() {
    // Builder and direct construction should behave identically
    let llm1 = SimLLMProvider::with_seed(42);
    let emb1 = SimEmbeddingProvider::with_seed(42);
    let vec1 = SimVectorBackend::new(42);
    let storage1 = SimStorageBackend::new(SimConfig::with_seed(42));

    let llm2 = SimLLMProvider::with_seed(42);
    let emb2 = SimEmbeddingProvider::with_seed(42);
    let vec2 = SimVectorBackend::new(42);
    let storage2 = SimStorageBackend::new(SimConfig::with_seed(42));

    let mut memory_new = Memory::new(llm1, emb1, vec1, storage1);
    let mut memory_builder = Memory::builder()
        .with_llm(llm2)
        .with_embedder(emb2)
        .with_vector(vec2)
        .with_storage(storage2)
        .build();

    // Same operations should work on both
    let result1 = memory_new
        .remember("Alice works at Acme", RememberOptions::default())
        .await
        .unwrap();

    let result2 = memory_builder
        .remember("Alice works at Acme", RememberOptions::default())
        .await
        .unwrap();

    // Both should extract entities
    assert!(!result1.entities.is_empty());
    assert!(!result2.entities.is_empty());
}

#[tokio::test]
async fn test_builder_method_chaining() {
    // Builder should support fluent method chaining
    let llm = SimLLMProvider::with_seed(42);
    let embedder = SimEmbeddingProvider::with_seed(42);
    let vector = SimVectorBackend::new(42);
    let storage = SimStorageBackend::new(SimConfig::with_seed(42));

    let memory = Memory::builder()
        .with_llm(llm)
        .with_embedder(embedder)
        .with_vector(vector)
        .with_storage(storage)
        .build();

    // Should work
    let results = memory
        .recall("test", RecallOptions::default())
        .await
        .unwrap();
    assert!(results.is_empty()); // Nothing stored yet
}

// =============================================================================
// Memory::sim() Constructor Tests
// =============================================================================

#[tokio::test]
async fn test_sim_constructor_basic() {
    // Test that Memory::sim() creates a working memory instance
    let mut memory = Memory::sim(42);

    // Should be able to remember and recall
    memory
        .remember("Alice works at Acme", RememberOptions::default())
        .await
        .unwrap();

    let results = memory
        .recall("Alice", RecallOptions::default())
        .await
        .unwrap();

    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_sim_deterministic() {
    // Same seed should produce deterministic behavior
    let mut memory1 = Memory::sim(42);
    let mut memory2 = Memory::sim(42);

    // Same operations should produce same results
    let result1 = memory1
        .remember("test entity", RememberOptions::default())
        .await
        .unwrap();

    let result2 = memory2
        .remember("test entity", RememberOptions::default())
        .await
        .unwrap();

    // Should extract same number of entities (deterministic LLM)
    assert_eq!(result1.entity_count(), result2.entity_count());
}

#[tokio::test]
async fn test_sim_different_seeds() {
    // Different seeds should produce different behavior
    let mut memory1 = Memory::sim(42);
    let mut memory2 = Memory::sim(99);

    // Store same text
    memory1
        .remember("test", RememberOptions::default())
        .await
        .unwrap();
    memory2
        .remember("test", RememberOptions::default())
        .await
        .unwrap();

    // Both should work (may extract different entities due to different RNG)
    let results1 = memory1
        .recall("test", RecallOptions::default())
        .await
        .unwrap();
    let results2 = memory2
        .recall("test", RecallOptions::default())
        .await
        .unwrap();

    // Both should return results
    assert!(!results1.is_empty());
    assert!(!results2.is_empty());
}

#[tokio::test]
async fn test_sim_full_workflow() {
    // Test complete workflow with sim constructor
    let mut memory = Memory::sim(42);

    // Remember multiple facts
    memory
        .remember("Alice is a software engineer", RememberOptions::default())
        .await
        .unwrap();

    memory
        .remember("Bob works at TechCo", RememberOptions::default())
        .await
        .unwrap();

    memory
        .remember("The weather is sunny", RememberOptions::default())
        .await
        .unwrap();

    // Recall should work
    let results = memory
        .recall("engineer", RecallOptions::default())
        .await
        .unwrap();

    // Should find relevant results
    assert!(!results.is_empty());
}

// =============================================================================
// Builder Error Handling Tests
// =============================================================================

#[tokio::test]
#[should_panic(expected = "LLM provider is required")]
async fn test_builder_missing_llm_panics() {
    // Building without LLM should panic
    let _memory: Memory<SimLLMProvider, SimEmbeddingProvider, SimStorageBackend, SimVectorBackend> =
        Memory::builder()
            .with_embedder(SimEmbeddingProvider::with_seed(42))
            .with_vector(SimVectorBackend::new(42))
            .with_storage(SimStorageBackend::new(SimConfig::with_seed(42)))
            .build();
}

#[tokio::test]
#[should_panic(expected = "Embedder is required")]
async fn test_builder_missing_embedder_panics() {
    // Building without embedder should panic
    let _memory: Memory<SimLLMProvider, SimEmbeddingProvider, SimStorageBackend, SimVectorBackend> =
        Memory::builder()
            .with_llm(SimLLMProvider::with_seed(42))
            .with_vector(SimVectorBackend::new(42))
            .with_storage(SimStorageBackend::new(SimConfig::with_seed(42)))
            .build();
}

#[tokio::test]
#[should_panic(expected = "Vector backend is required")]
async fn test_builder_missing_vector_panics() {
    // Building without vector should panic
    let _memory: Memory<SimLLMProvider, SimEmbeddingProvider, SimStorageBackend, SimVectorBackend> =
        Memory::builder()
            .with_llm(SimLLMProvider::with_seed(42))
            .with_embedder(SimEmbeddingProvider::with_seed(42))
            .with_storage(SimStorageBackend::new(SimConfig::with_seed(42)))
            .build();
}

#[tokio::test]
#[should_panic(expected = "Storage backend is required")]
async fn test_builder_missing_storage_panics() {
    // Building without storage should panic
    let _memory: Memory<SimLLMProvider, SimEmbeddingProvider, SimStorageBackend, SimVectorBackend> =
        Memory::builder()
            .with_llm(SimLLMProvider::with_seed(42))
            .with_embedder(SimEmbeddingProvider::with_seed(42))
            .with_vector(SimVectorBackend::new(42))
            .build();
}
