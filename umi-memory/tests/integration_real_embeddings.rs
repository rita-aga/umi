//! Integration Tests with Real OpenAI Embeddings
//!
//! These tests use real OpenAI embeddings to verify semantic similarity works correctly.
//! They are marked with `#[ignore]` so they only run when explicitly requested.
//!
//! **Run with:**
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! export ANTHROPIC_API_KEY=sk-ant-...
//! cargo test --test integration_real_embeddings --ignored --features embedding-openai,anthropic
//! ```
//!
//! **Cost:** ~$0.01 per test run (minimal)

#![cfg(all(feature = "embedding-openai", feature = "anthropic"))]

use std::env;
use umi_memory::dst::SimConfig;
use umi_memory::embedding::OpenAIEmbeddingProvider;
use umi_memory::llm::AnthropicProvider;
use umi_memory::storage::{SimStorageBackend, SimVectorBackend};
use umi_memory::umi::{Memory, MemoryConfig, RecallOptions, RememberOptions};

/// Helper to create Memory with real Anthropic LLM + OpenAI embeddings
fn create_memory_with_real_embeddings() -> Memory {
    let openai_key =
        env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set for integration tests");
    let anthropic_key =
        env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set for integration tests");

    let config = MemoryConfig::default();
    let llm = AnthropicProvider::new(anthropic_key);
    let embedder = OpenAIEmbeddingProvider::new(openai_key);
    let storage = SimStorageBackend::new(SimConfig::with_seed(42));
    let vector = SimVectorBackend::new(42);

    Memory::with_config(llm, embedder, vector, storage, config)
}

/// Integration Test 1: Recall Returns Relevant Results with Real Embeddings
///
/// This test verifies that with real semantic embeddings, recall returns
/// relevant results sorted by similarity.
///
/// **Expected:** Top results mention Sarah, NeuralFlow, or ML engineer
#[tokio::test]
#[ignore] // Only run with --ignored flag
async fn test_recall_returns_relevant_results_real_embeddings() {
    let mut memory = create_memory_with_real_embeddings();

    println!("=== INTEGRATION: Storing test data with real embeddings ===");

    // Store test data matching UX report scenario
    let result1 = memory
        .remember(
            "Sarah Chen works at NeuralFlow as an ML engineer",
            RememberOptions::default(),
        )
        .await
        .unwrap();

    println!(
        "  Stored {} entities from first remember()",
        result1.entities.len()
    );
    for e in &result1.entities {
        println!("    - {} ({})", e.name, e.entity_type);
    }

    memory
        .remember(
            "Sarah is learning Rust for the recommendation systems team",
            RememberOptions::default(),
        )
        .await
        .unwrap();

    memory
        .remember(
            "Python is Sarah's main language",
            RememberOptions::default(),
        )
        .await
        .unwrap();

    // Store some unrelated entities
    memory
        .remember(
            "Bob likes coffee and reads about TypeScript",
            RememberOptions::default(),
        )
        .await
        .unwrap();

    println!("=== INTEGRATION: Querying 'What is Sarah's job?' ===");

    // Query: What is Sarah's job?
    let results = memory
        .recall(
            "What is Sarah's job?",
            RecallOptions::default().with_limit(3).unwrap(),
        )
        .await
        .unwrap();

    // Print actual results
    println!("=== INTEGRATION: Top 3 results ===");
    for (i, entity) in results.iter().enumerate() {
        println!(
            "  {}. {} - '{}'",
            i + 1,
            entity.name,
            entity.content.chars().take(60).collect::<String>()
        );
    }

    // With real embeddings, top result should be Sarah-related
    assert!(!results.is_empty(), "Should find at least one result");

    let top_result = &results[0];
    let is_relevant = top_result.name.contains("Sarah")
        || top_result.content.contains("Sarah")
        || top_result.content.contains("NeuralFlow")
        || top_result.content.contains("ML engineer");

    assert!(
        is_relevant,
        "Top result should be Sarah-related with real embeddings, got: '{}' - '{}'",
        top_result.name,
        top_result.content.chars().take(100).collect::<String>()
    );

    println!("✅ PASSED: Recall returns relevant results with real embeddings");
}

/// Integration Test 2: Stress Test with Real Embeddings
///
/// Runs 20 queries to measure relevance consistency with real embeddings.
///
/// **Expected:** Sarah appears in top 3 for 80%+ of queries
#[tokio::test]
#[ignore] // Only run with --ignored flag
async fn test_stress_recall_relevance_real_embeddings() {
    let mut memory = create_memory_with_real_embeddings();

    println!("=== INTEGRATION STRESS: Setting up test data ===");

    // Store 10 entities about Sarah
    for i in 0..10 {
        memory
            .remember(
                &format!(
                    "Sarah fact {}: ML engineer at NeuralFlow working on recommendations",
                    i
                ),
                RememberOptions::default(),
            )
            .await
            .unwrap();
    }

    // Store 10 unrelated entities
    for i in 0..10 {
        memory
            .remember(
                &format!(
                    "Unrelated fact {}: Python and Rust programming languages",
                    i
                ),
                RememberOptions::default(),
            )
            .await
            .unwrap();
    }

    println!("=== INTEGRATION STRESS: Running 20 queries about Sarah ===");

    let mut sarah_in_top3_count = 0;

    // Run 20 queries about Sarah (fewer than DST test due to API cost)
    for i in 0..20 {
        let results = memory
            .recall(
                "What does Sarah do?",
                RecallOptions::default().with_limit(3).unwrap(),
            )
            .await
            .unwrap();

        // Count how many times Sarah appears in top 3
        let sarah_found = results
            .iter()
            .any(|e| e.name.contains("Sarah") || e.content.contains("Sarah"));

        if sarah_found {
            sarah_in_top3_count += 1;
        }

        // Print sample results
        if i % 5 == 0 {
            println!("  Query {}: Found Sarah = {}", i, sarah_found);
        }
    }

    let relevance_percentage = (sarah_in_top3_count as f32 / 20.0) * 100.0;

    println!("=== INTEGRATION STRESS: Results ===");
    println!(
        "  Sarah in top 3: {}/20 ({:.0}%)",
        sarah_in_top3_count, relevance_percentage
    );

    // With real embeddings, should be high relevance (80%+)
    assert!(
        sarah_in_top3_count >= 16,
        "Sarah should appear in top 3 for 80%+ of queries with real embeddings, got {}/20 ({:.0}%)",
        sarah_in_top3_count,
        relevance_percentage
    );

    println!(
        "✅ PASSED: Real embeddings give {:.0}% relevance (16+ expected)",
        relevance_percentage
    );
}

/// Integration Test 3: Non-Existent Entity with Real Embeddings
///
/// Queries for an entity that doesn't exist in memory.
///
/// **Expected:** Empty results or low-relevance filtered results
#[tokio::test]
#[ignore] // Only run with --ignored flag
async fn test_nonexistent_entity_real_embeddings() {
    let mut memory = create_memory_with_real_embeddings();

    println!("=== INTEGRATION: Storing Sarah facts ===");

    // Store facts about Sarah
    memory
        .remember(
            "Sarah works at NeuralFlow as an ML engineer",
            RememberOptions::default(),
        )
        .await
        .unwrap();

    memory
        .remember("Sarah loves Python programming", RememberOptions::default())
        .await
        .unwrap();

    println!("=== INTEGRATION: Querying for non-existent 'John Smith' ===");

    // Query for John Smith (doesn't exist)
    let results = memory
        .recall("Who is John Smith?", RecallOptions::default())
        .await
        .unwrap();

    println!("=== INTEGRATION: Results for non-existent entity ===");
    println!("  Found {} results", results.len());
    for (i, entity) in results.iter().take(3).enumerate() {
        println!("  {}. {}", i + 1, entity.name);
    }

    // With min_score filtering, should return few results
    assert!(
        results.len() <= 2,
        "Query for non-existent entity should return few results with real embeddings, got {}",
        results.len()
    );

    println!("✅ PASSED: Non-existent entity returns few results (min_score filtering works)");
}

/// Integration Test 4: Verify Semantic Similarity Sorting
///
/// Tests that results are sorted by relevance, not recency.
///
/// **Expected:** Most relevant result is first, not most recent
#[tokio::test]
#[ignore] // Only run with --ignored flag
async fn test_semantic_similarity_sorting_real_embeddings() {
    let mut memory = create_memory_with_real_embeddings();

    println!("=== INTEGRATION: Testing semantic similarity sorting ===");

    // Store less relevant fact first
    memory
        .remember(
            "Sarah occasionally mentions Python in meetings",
            RememberOptions::default(),
        )
        .await
        .unwrap();

    // Store highly relevant fact second
    memory
        .remember(
            "Sarah Chen is the Senior ML Engineer at NeuralFlow, leading the recommendation systems team",
            RememberOptions::default(),
        )
        .await
        .unwrap();

    // Store completely irrelevant fact last (most recent)
    memory
        .remember(
            "The coffee machine broke yesterday",
            RememberOptions::default(),
        )
        .await
        .unwrap();

    println!("=== INTEGRATION: Querying 'What is Sarah's job title?' ===");

    let results = memory
        .recall(
            "What is Sarah's job title?",
            RecallOptions::default().with_limit(3).unwrap(),
        )
        .await
        .unwrap();

    println!("=== INTEGRATION: Results (should be sorted by relevance) ===");
    for (i, entity) in results.iter().enumerate() {
        println!(
            "  {}. {} - '{}'",
            i + 1,
            entity.name,
            entity.content.chars().take(80).collect::<String>()
        );
    }

    // Top result should mention "Senior ML Engineer" or "NeuralFlow"
    // NOT the most recent "coffee machine" fact
    assert!(!results.is_empty(), "Should find results");

    let top_result = &results[0];
    let is_job_related = top_result.content.contains("ML Engineer")
        || top_result.content.contains("NeuralFlow")
        || top_result.content.contains("recommendation systems");

    assert!(
        is_job_related,
        "Top result should be job-related (sorted by relevance, not recency), got: '{}'",
        top_result.content.chars().take(100).collect::<String>()
    );

    // Most recent (coffee machine) should NOT be first
    assert!(
        !top_result.content.contains("coffee machine"),
        "Most recent fact should NOT be first (proves not sorted by recency)"
    );

    println!("✅ PASSED: Results sorted by semantic similarity, not recency");
}
