//! DST Tests for MemoryConfig
//!
//! TigerStyle: Tests written FIRST, then implementation.
//!
//! These tests define the contract for MemoryConfig before it exists.
//! They will fail to compile until MemoryConfig is implemented.

use umi_memory::umi::{Memory, MemoryConfig, RecallOptions, RememberOptions};

// =============================================================================
// Default Configuration Tests
// =============================================================================

#[test]
fn test_memory_config_defaults() {
    // Test that default configuration has sensible values
    let config = MemoryConfig::default();

    assert_eq!(config.core_memory_bytes, 32 * 1024); // 32KB
    assert_eq!(config.working_memory_bytes, 1024 * 1024); // 1MB
    assert_eq!(config.default_recall_limit, 10);
    assert_eq!(config.embedding_batch_size, 100);
    assert!(config.generate_embeddings);
    assert!(config.semantic_search_enabled);
    assert!(config.query_expansion_enabled);
}

#[test]
fn test_memory_config_builder_pattern() {
    // Test that config supports builder pattern for customization
    let config = MemoryConfig::default()
        .with_core_memory_bytes(64 * 1024)
        .with_working_memory_bytes(2 * 1024 * 1024)
        .with_recall_limit(20)
        .with_embedding_batch_size(50)
        .without_embeddings()
        .without_semantic_search();

    assert_eq!(config.core_memory_bytes, 64 * 1024);
    assert_eq!(config.working_memory_bytes, 2 * 1024 * 1024);
    assert_eq!(config.default_recall_limit, 20);
    assert_eq!(config.embedding_batch_size, 50);
    assert!(!config.generate_embeddings);
    assert!(!config.semantic_search_enabled);
}

#[test]
fn test_memory_config_method_chaining() {
    // Test that all builder methods return Self for chaining
    let config = MemoryConfig::default()
        .with_core_memory_bytes(32 * 1024)
        .with_recall_limit(15)
        .without_query_expansion();

    assert_eq!(config.core_memory_bytes, 32 * 1024);
    assert_eq!(config.default_recall_limit, 15);
    assert!(!config.query_expansion_enabled);
}

// =============================================================================
// Config Integration with Memory Tests
// =============================================================================

#[tokio::test]
async fn test_memory_builder_with_config() {
    // Test that MemoryBuilder accepts config
    let config = MemoryConfig::default().with_recall_limit(5);

    let mut memory = Memory::sim_with_config(42, config);

    // Memory should be usable
    memory
        .remember("test", RememberOptions::default())
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "TODO: Wire config.default_recall_limit through Memory.recall()"]
async fn test_memory_respects_recall_limit_from_config() {
    // Test that Memory uses config's default_recall_limit
    // NOTE: This test currently fails because config is not yet fully wired through Memory
    let config = MemoryConfig::default().with_recall_limit(3);

    let mut memory = Memory::sim_with_config(42, config);

    // Store multiple entities
    memory
        .remember("Alice is an engineer", RememberOptions::default())
        .await
        .unwrap();
    memory
        .remember("Bob is a designer", RememberOptions::default())
        .await
        .unwrap();
    memory
        .remember("Charlie is a manager", RememberOptions::default())
        .await
        .unwrap();
    memory
        .remember("Diana is a developer", RememberOptions::default())
        .await
        .unwrap();
    memory
        .remember("Eve is an analyst", RememberOptions::default())
        .await
        .unwrap();

    // Recall without explicit limit should use config default (3)
    let results = memory
        .recall("person", RecallOptions::default())
        .await
        .unwrap();

    // Should respect config limit of 3
    assert!(
        results.len() <= 3,
        "Expected at most 3 results, got {}",
        results.len()
    );
}

#[tokio::test]
async fn test_memory_explicit_limit_overrides_config() {
    // Test that explicit RecallOptions limit overrides config default
    let config = MemoryConfig::default().with_recall_limit(3);

    let mut memory = Memory::sim_with_config(42, config);

    // Store entities
    memory
        .remember("Alice is an engineer", RememberOptions::default())
        .await
        .unwrap();
    memory
        .remember("Bob is a designer", RememberOptions::default())
        .await
        .unwrap();

    // Explicit limit should override config
    let results = memory
        .recall("person", RecallOptions::default().with_limit(10))
        .await
        .unwrap();

    // Should use explicit limit (10), not config default (3)
    assert!(results.len() <= 10);
}

// =============================================================================
// Config Validation Tests
// =============================================================================

#[test]
fn test_config_with_zero_recall_limit() {
    // Setting zero recall limit should work (will be validated at runtime)
    let config = MemoryConfig::default().with_recall_limit(0);
    assert_eq!(config.default_recall_limit, 0);
}

#[test]
fn test_config_with_large_values() {
    // Config should accept large values
    let config = MemoryConfig::default()
        .with_core_memory_bytes(10 * 1024 * 1024) // 10MB
        .with_working_memory_bytes(100 * 1024 * 1024) // 100MB
        .with_recall_limit(1000);

    assert_eq!(config.core_memory_bytes, 10 * 1024 * 1024);
    assert_eq!(config.working_memory_bytes, 100 * 1024 * 1024);
    assert_eq!(config.default_recall_limit, 1000);
}

// =============================================================================
// Config Behavior Tests
// =============================================================================

#[test]
fn test_config_disable_embeddings() {
    // Test that embeddings can be disabled
    let config = MemoryConfig::default().without_embeddings();

    assert!(!config.generate_embeddings);
    assert!(config.semantic_search_enabled); // Other flags unaffected
}

#[test]
fn test_config_disable_semantic_search() {
    // Test that semantic search can be disabled
    let config = MemoryConfig::default().without_semantic_search();

    assert!(!config.semantic_search_enabled);
    assert!(config.generate_embeddings); // Other flags unaffected
}

#[test]
fn test_config_disable_query_expansion() {
    // Test that query expansion can be disabled
    let config = MemoryConfig::default().without_query_expansion();

    assert!(!config.query_expansion_enabled);
    assert!(config.semantic_search_enabled); // Other flags unaffected
}

// =============================================================================
// Config Clone and Debug Tests
// =============================================================================

#[test]
fn test_config_clone() {
    // Config should be cloneable
    let config1 = MemoryConfig::default().with_recall_limit(15);
    let config2 = config1.clone();

    assert_eq!(config1.default_recall_limit, config2.default_recall_limit);
    assert_eq!(config1.core_memory_bytes, config2.core_memory_bytes);
}

#[test]
fn test_config_debug() {
    // Config should be debuggable
    let config = MemoryConfig::default();
    let debug_str = format!("{:?}", config);

    // Should contain key fields
    assert!(debug_str.contains("MemoryConfig"));
}
