//! DST Tests for Memory Construction Patterns
//!
//! TigerStyle: Tests for Memory construction using `new()` and `sim()` constructors.
//!
//! Note: MemoryBuilder pattern is currently disabled (see TODO in umi/mod.rs).
//! These tests focus on the available construction methods.

use umi_memory::dst::SimConfig;
use umi_memory::embedding::SimEmbeddingProvider;
use umi_memory::llm::SimLLMProvider;
use umi_memory::storage::{SimStorageBackend, SimVectorBackend};
use umi_memory::umi::{Memory, RecallOptions, RememberOptions};

// =============================================================================
// Memory::new() Constructor Tests
// =============================================================================

#[tokio::test]
async fn test_new_basic_construction() {
    // Test that Memory::new() creates a working memory instance
    let llm = SimLLMProvider::with_seed(42);
    let embedder = SimEmbeddingProvider::with_seed(42);
    let vector = SimVectorBackend::new(42);
    let storage = SimStorageBackend::new(SimConfig::with_seed(42));

    let mut memory = Memory::new(llm, embedder, vector, storage);

    // Should be able to use memory
    let result = memory
        .remember("test", RememberOptions::default())
        .await
        .unwrap();
    assert!(!result.entities.is_empty());
}

#[tokio::test]
async fn test_new_deterministic_with_same_seed() {
    // Same components with same seed should behave identically
    let llm1 = SimLLMProvider::with_seed(42);
    let emb1 = SimEmbeddingProvider::with_seed(42);
    let vec1 = SimVectorBackend::new(42);
    let storage1 = SimStorageBackend::new(SimConfig::with_seed(42));

    let llm2 = SimLLMProvider::with_seed(42);
    let emb2 = SimEmbeddingProvider::with_seed(42);
    let vec2 = SimVectorBackend::new(42);
    let storage2 = SimStorageBackend::new(SimConfig::with_seed(42));

    let mut memory1 = Memory::new(llm1, emb1, vec1, storage1);
    let mut memory2 = Memory::new(llm2, emb2, vec2, storage2);

    // Same operations should produce same results
    let result1 = memory1
        .remember("Alice works at Acme", RememberOptions::default())
        .await
        .unwrap();

    let result2 = memory2
        .remember("Alice works at Acme", RememberOptions::default())
        .await
        .unwrap();

    // Both should extract same number of entities
    assert_eq!(result1.entity_count(), result2.entity_count());
}

#[tokio::test]
async fn test_new_recall_empty_initially() {
    // New memory should have no entities initially
    let llm = SimLLMProvider::with_seed(42);
    let embedder = SimEmbeddingProvider::with_seed(42);
    let vector = SimVectorBackend::new(42);
    let storage = SimStorageBackend::new(SimConfig::with_seed(42));

    let memory = Memory::new(llm, embedder, vector, storage);

    // Should work but return empty
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

#[tokio::test]
async fn test_sim_with_config() {
    // Test Memory::sim_with_config() constructor
    use umi_memory::umi::MemoryConfig;

    let config = MemoryConfig::default()
        .with_recall_limit(5)
        .without_embeddings();

    let mut memory = Memory::sim_with_config(42, config);

    // Should be able to use memory
    let result = memory
        .remember("test entity", RememberOptions::default())
        .await
        .unwrap();
    assert!(!result.entities.is_empty());

    // Config should be applied (though we can't easily verify internal state)
}

// =============================================================================
// Construction Equivalence Tests
// =============================================================================

#[tokio::test]
async fn test_sim_equivalent_to_new_with_sim_providers() {
    // Memory::sim() should behave like Memory::new() with sim providers
    let llm = SimLLMProvider::with_seed(42);
    let embedder = SimEmbeddingProvider::with_seed(42);
    let vector = SimVectorBackend::new(42);
    let storage = SimStorageBackend::new(SimConfig::with_seed(42));

    let mut memory_new = Memory::new(llm, embedder, vector, storage);
    let mut memory_sim = Memory::sim(42);

    // Same operations should work on both
    let result1 = memory_new
        .remember("Alice works at Acme", RememberOptions::default())
        .await
        .unwrap();

    let result2 = memory_sim
        .remember("Alice works at Acme", RememberOptions::default())
        .await
        .unwrap();

    // Both should extract entities
    assert!(!result1.entities.is_empty());
    assert!(!result2.entities.is_empty());

    // Should have same count (deterministic)
    assert_eq!(result1.entity_count(), result2.entity_count());
}
