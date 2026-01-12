//! Test Anthropic LLM Provider
//!
//! Manually verify that Anthropic integration works with real API.
//!
//! **IMPORTANT**: This example requires a valid Anthropic API key.
//!
//! Setup:
//!   1. Get API key from https://console.anthropic.com/
//!   2. Export ANTHROPIC_API_KEY environment variable
//!   3. Run: cargo run --example test_anthropic --features anthropic
//!
//! Cost: ~$0.01-0.02 per run (uses Claude Sonnet 3.5)

use umi_memory::extraction::{EntityExtractor, ExtractionOptions};
use umi_memory::llm::AnthropicProvider;
use umi_memory::retrieval::DualRetriever;
use umi_memory::evolution::{EvolutionTracker, DetectionOptions};
use umi_memory::storage::{Entity, EntityType, SimStorageBackend, SimVectorBackend};
use umi_memory::embedding::SimEmbeddingProvider;
use umi_memory::dst::SimConfig;
use std::env;

// Note: extraction::EntityType is separate from storage::EntityType
// For extraction results, use extraction::EntityType::Note for fallback detection
use umi_memory::extraction::EntityType as ExtractionEntityType;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Anthropic LLM Provider ===\n");

    // Get API key from environment
    let api_key = match env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("❌ ERROR: ANTHROPIC_API_KEY environment variable not set");
            eprintln!("\nSetup:");
            eprintln!("  1. Get API key from https://console.anthropic.com/");
            eprintln!("  2. Export ANTHROPIC_API_KEY=<your-key>");
            eprintln!("  3. Run: cargo run --example test_anthropic --features anthropic\n");
            std::process::exit(1);
        }
    };

    println!("✓ API key found ({}...)", &api_key[..8]);
    println!();

    // Create Anthropic LLM provider
    let llm = AnthropicProvider::new(api_key);
    println!("✓ Created AnthropicProvider\n");

    // Test 1: Entity Extraction
    println!("--- Test 1: Entity Extraction ---");
    test_entity_extraction(&llm).await?;
    println!();

    // Test 2: Query Rewriting
    println!("--- Test 2: Query Rewriting ---");
    test_query_rewriting(&llm).await?;
    println!();

    // Test 3: Evolution Detection
    println!("--- Test 3: Evolution Detection ---");
    test_evolution_detection(&llm).await?;
    println!();

    println!("=== All Tests Passed! ===");
    println!("\n✅ Anthropic integration works correctly");
    println!("💰 Approximate cost: $0.01-0.02");
    Ok(())
}

/// Test entity extraction with real Anthropic API
async fn test_entity_extraction(llm: &AnthropicProvider) -> Result<(), Box<dyn std::error::Error>> {
    let extractor = EntityExtractor::new(llm.clone());

    let text = "Alice is a software engineer at Acme Corp. She specializes in Rust and distributed systems.";
    println!("  Input: \"{}\"", text);

    let result = extractor
        .extract(text, ExtractionOptions::default())
        .await?;

    println!("  ✓ Extraction succeeded");
    println!("  Found {} entities:", result.entities.len());

    for entity in &result.entities {
        println!("    - Type: {:?}, Name: {}", entity.entity_type, entity.name);
        println!("      Content: {}", entity.content);
        println!("      Confidence: {:.2}", entity.confidence);
    }

    // Verify: Should have at least one entity
    assert!(
        !result.entities.is_empty(),
        "Expected at least one entity"
    );

    // Verify: Should NOT be fallback (ExtractionEntityType::Note means fallback)
    let has_fallback = result.entities.iter().any(|e| e.entity_type == ExtractionEntityType::Note);
    assert!(
        !has_fallback,
        "Expected real entity extraction, got fallback (ExtractionEntityType::Note)"
    );

    Ok(())
}

/// Test query rewriting with real Anthropic API
async fn test_query_rewriting(llm: &AnthropicProvider) -> Result<(), Box<dyn std::error::Error>> {
    // Create retriever with Anthropic LLM (but sim storage/embedder for this test)
    let embedder = SimEmbeddingProvider::with_seed(42);
    let vector = SimVectorBackend::new(42);
    let storage = SimStorageBackend::new(SimConfig::with_seed(42));
    let retriever = DualRetriever::new(llm.clone(), embedder, vector, storage);

    let query = "Who are the software engineers?";
    println!("  Input query: \"{}\"", query);

    let variations = retriever.rewrite_query(query).await;

    println!("  ✓ Query rewriting succeeded");
    println!("  Generated {} variations:", variations.len());

    for (i, variation) in variations.iter().enumerate() {
        println!("    {}. {}", i + 1, variation);
    }

    // Verify: Should have multiple variations (not just original query)
    assert!(
        variations.len() > 1,
        "Expected query expansion to generate variations, got {} (fallback only)",
        variations.len()
    );

    // Verify: Should include original query
    assert!(
        variations.contains(&query.to_string()),
        "Expected variations to include original query"
    );

    Ok(())
}

/// Test evolution detection with real Anthropic API
async fn test_evolution_detection(llm: &AnthropicProvider) -> Result<(), Box<dyn std::error::Error>> {
    let tracker: EvolutionTracker<AnthropicProvider, SimStorageBackend> = EvolutionTracker::new(llm.clone());

    // Create test entities
    let old_entity = Entity::new(
        EntityType::Person,
        "Alice".to_string(),
        "Works at Acme Corp as a software engineer".to_string(),
    );

    let new_entity = Entity::new(
        EntityType::Person,
        "Alice".to_string(),
        "Left Acme Corp, now CTO at StartupX".to_string(),
    );

    println!("  Old: {}", old_entity.content);
    println!("  New: {}", new_entity.content);

    let result = tracker
        .detect(&new_entity, &[old_entity], DetectionOptions::default())
        .await?;

    match result {
        Some(detection) => {
            println!("  ✓ Evolution detected!");
            println!("    Type: {:?}", detection.evolution_type());
            println!("    Reason: {}", detection.reason());
            println!("    Confidence: {:.2}", detection.confidence());
            println!("    LLM used: {}", detection.llm_used);

            // Verify: LLM should be used (not fallback)
            assert!(
                detection.llm_used,
                "Expected real LLM detection, got fallback"
            );
        }
        None => {
            println!("  ℹ No evolution detected (but LLM call succeeded)");
            println!("    This is valid behavior if LLM determines no clear relationship");
        }
    }

    Ok(())
}
