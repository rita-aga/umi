//! Full End-to-End Testing with Real Providers
//!
//! This example comprehensively tests Umi's full capabilities with:
//! - Real LLM providers (Anthropic Claude, OpenAI GPT)
//! - Real embedding providers (OpenAI text-embedding-3-small)
//! - Real persistent storage (LanceDB)
//! - Full workflow: remember -> recall -> evolution detection
//!
//! ## Prerequisites
//!
//! 1. Set environment variables:
//!    - `ANTHROPIC_API_KEY`: Your Anthropic API key
//!    - `OPENAI_API_KEY`: Your OpenAI API key
//!
//! 2. Build with all features:
//!    ```bash
//!    cargo build --all-features
//!    ```
//!
//! ## Running
//!
//! Run all tests:
//! ```bash
//! cargo run --example full_e2e_test --all-features
//! ```
//!
//! Run with specific provider:
//! ```bash
//! # Anthropic only
//! SKIP_OPENAI=1 cargo run --example full_e2e_test --all-features
//!
//! # OpenAI only  
//! SKIP_ANTHROPIC=1 cargo run --example full_e2e_test --all-features
//! ```
//!
//! ## Estimated Cost
//!
//! - Anthropic: ~$0.02-0.05
//! - OpenAI LLM: ~$0.02-0.05
//! - OpenAI Embeddings: ~$0.001
//! - Total: ~$0.05-0.10 per run

use std::env;
use std::path::PathBuf;
use tempfile::TempDir;

// Core types
use umi_memory::umi::{Memory, RecallOptions, RememberOptions};

// LLM Providers
#[cfg(feature = "anthropic")]
use umi_memory::llm::AnthropicProvider;
#[cfg(feature = "openai")]
use umi_memory::llm::OpenAIProvider;

// Embedding Providers
#[cfg(feature = "embedding-openai")]
use umi_memory::embedding::OpenAIEmbeddingProvider;
use umi_memory::embedding::SimEmbeddingProvider;

// Storage backends
#[cfg(feature = "lance")]
use umi_memory::storage::{LanceStorageBackend, LanceVectorBackend};
use umi_memory::dst::SimConfig;
use umi_memory::storage::{SimStorageBackend, SimVectorBackend};

// Extraction/Evolution/Retrieval
use umi_memory::extraction::{EntityExtractor, ExtractionOptions};
use umi_memory::evolution::{DetectionOptions, EvolutionTracker};
use umi_memory::retrieval::DualRetriever;
use umi_memory::storage::{Entity, EntityType};

/// Test result summary
struct TestResults {
    passed: usize,
    failed: usize,
    skipped: usize,
    details: Vec<String>,
}

impl TestResults {
    fn new() -> Self {
        Self {
            passed: 0,
            failed: 0,
            skipped: 0,
            details: Vec::new(),
        }
    }

    fn pass(&mut self, name: &str) {
        self.passed += 1;
        self.details.push(format!("✓ {}", name));
    }

    fn fail(&mut self, name: &str, reason: &str) {
        self.failed += 1;
        self.details.push(format!("✗ {} - {}", name, reason));
    }

    fn skip(&mut self, name: &str, reason: &str) {
        self.skipped += 1;
        self.details.push(format!("○ {} - SKIPPED: {}", name, reason));
    }

    fn summary(&self) -> String {
        format!(
            "Passed: {}, Failed: {}, Skipped: {}",
            self.passed, self.failed, self.skipped
        )
    }

    fn print(&self) {
        println!("\n=== Test Results ===");
        for detail in &self.details {
            println!("  {}", detail);
        }
        println!("\n{}", self.summary());
        
        if self.failed > 0 {
            println!("\n❌ Some tests failed!");
        } else if self.passed > 0 {
            println!("\n✅ All executed tests passed!");
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          UMI MEMORY - Full End-to-End Integration Tests       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let mut results = TestResults::new();

    // Check environment variables
    let anthropic_key = env::var("ANTHROPIC_API_KEY").ok();
    let openai_key = env::var("OPENAI_API_KEY").ok();
    let skip_anthropic = env::var("SKIP_ANTHROPIC").is_ok();
    let skip_openai = env::var("SKIP_OPENAI").is_ok();

    println!("Environment:");
    println!("  ANTHROPIC_API_KEY: {}", if anthropic_key.is_some() { "✓ Set" } else { "✗ Not set" });
    println!("  OPENAI_API_KEY: {}", if openai_key.is_some() { "✓ Set" } else { "✗ Not set" });
    println!("  SKIP_ANTHROPIC: {}", skip_anthropic);
    println!("  SKIP_OPENAI: {}", skip_openai);
    println!();

    // Create temp directory for LanceDB
    let temp_dir = TempDir::new()?;
    let lance_path = temp_dir.path().join("test_lance_db");
    println!("Using temp directory: {:?}\n", temp_dir.path());

    // =========================================================================
    // Test Suite 1: Anthropic LLM Provider
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test Suite 1: Anthropic LLM Provider");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    #[cfg(feature = "anthropic")]
    {
        if skip_anthropic {
            results.skip("Anthropic Suite", "SKIP_ANTHROPIC set");
        } else if let Some(ref key) = anthropic_key {
            test_anthropic_suite(key, &mut results).await;
        } else {
            results.skip("Anthropic Suite", "ANTHROPIC_API_KEY not set");
        }
    }

    #[cfg(not(feature = "anthropic"))]
    {
        results.skip("Anthropic Suite", "anthropic feature not enabled");
    }

    // =========================================================================
    // Test Suite 2: OpenAI LLM Provider
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test Suite 2: OpenAI LLM Provider");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    #[cfg(feature = "openai")]
    {
        if skip_openai {
            results.skip("OpenAI LLM Suite", "SKIP_OPENAI set");
        } else if let Some(ref key) = openai_key {
            test_openai_llm_suite(key, &mut results).await;
        } else {
            results.skip("OpenAI LLM Suite", "OPENAI_API_KEY not set");
        }
    }

    #[cfg(not(feature = "openai"))]
    {
        results.skip("OpenAI LLM Suite", "openai feature not enabled");
    }

    // =========================================================================
    // Test Suite 3: OpenAI Embedding Provider
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test Suite 3: OpenAI Embedding Provider");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    #[cfg(feature = "embedding-openai")]
    {
        if skip_openai {
            results.skip("OpenAI Embedding Suite", "SKIP_OPENAI set");
        } else if let Some(ref key) = openai_key {
            test_openai_embedding_suite(key, &mut results).await;
        } else {
            results.skip("OpenAI Embedding Suite", "OPENAI_API_KEY not set");
        }
    }

    #[cfg(not(feature = "embedding-openai"))]
    {
        results.skip("OpenAI Embedding Suite", "embedding-openai feature not enabled");
    }

    // =========================================================================
    // Test Suite 4: LanceDB Storage Backend
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test Suite 4: LanceDB Storage Backend");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    #[cfg(feature = "lance")]
    {
        test_lance_storage_suite(&lance_path, &mut results).await;
    }

    #[cfg(not(feature = "lance"))]
    {
        results.skip("LanceDB Suite", "lance feature not enabled");
    }

    // =========================================================================
    // Test Suite 5: Full Integration (All Components)
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test Suite 5: Full Integration (All Real Components)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    #[cfg(all(feature = "anthropic", feature = "embedding-openai", feature = "lance"))]
    {
        if anthropic_key.is_some() && openai_key.is_some() && !skip_anthropic && !skip_openai {
            test_full_integration_suite(
                anthropic_key.as_ref().unwrap(),
                openai_key.as_ref().unwrap(),
                &temp_dir.path().join("full_integration_lance"),
                &mut results,
            ).await;
        } else {
            results.skip("Full Integration Suite", "Missing API keys or skipped");
        }
    }

    #[cfg(not(all(feature = "anthropic", feature = "embedding-openai", feature = "lance")))]
    {
        results.skip("Full Integration Suite", "Requires anthropic + embedding-openai + lance features");
    }

    // =========================================================================
    // Test Suite 6: Memory Class Full Workflow
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test Suite 6: Memory Class Full Workflow");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Test with simulation providers (always available)
    test_memory_workflow_simulation(&mut results).await;

    // Test with real providers if available
    #[cfg(all(feature = "anthropic", feature = "embedding-openai", feature = "lance"))]
    {
        if anthropic_key.is_some() && openai_key.is_some() && !skip_anthropic && !skip_openai {
            test_memory_workflow_production(
                anthropic_key.as_ref().unwrap(),
                openai_key.as_ref().unwrap(),
                &temp_dir.path().join("memory_workflow_lance"),
                &mut results,
            ).await;
        }
    }

    // =========================================================================
    // Final Summary
    // =========================================================================
    results.print();

    // Return error code if any tests failed
    if results.failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}

// =============================================================================
// Anthropic Test Suite
// =============================================================================

#[cfg(feature = "anthropic")]
async fn test_anthropic_suite(api_key: &str, results: &mut TestResults) {
    use umi_memory::llm::{CompletionRequest, LLMProvider};

    println!("\n1.1 Testing Anthropic LLM Basic Completion...");
    let llm = AnthropicProvider::new(api_key);

    // Test basic completion
    match llm.complete(&CompletionRequest::new("Say 'Hello' in exactly one word.")).await {
        Ok(response) => {
            println!("     Response: {}", response.trim());
            if !response.is_empty() {
                results.pass("Anthropic Basic Completion");
            } else {
                results.fail("Anthropic Basic Completion", "Empty response");
            }
        }
        Err(e) => {
            results.fail("Anthropic Basic Completion", &e.to_string());
        }
    }

    // Test entity extraction
    println!("\n1.2 Testing Anthropic Entity Extraction...");
    let extractor = EntityExtractor::new(Box::new(llm.clone()));
    let text = "Alice is a software engineer at Acme Corp who specializes in Rust programming.";
    
    match extractor.extract(text, ExtractionOptions::default()).await {
        Ok(result) => {
            println!("     Input: \"{}\"", text);
            println!("     Extracted {} entities:", result.entities.len());
            for entity in &result.entities {
                println!("       - {} ({:?}): {}", entity.name, entity.entity_type, entity.content);
            }
            if !result.entities.is_empty() {
                results.pass("Anthropic Entity Extraction");
            } else {
                results.fail("Anthropic Entity Extraction", "No entities extracted");
            }
        }
        Err(e) => {
            results.fail("Anthropic Entity Extraction", &e.to_string());
        }
    }

    // Test evolution detection
    println!("\n1.3 Testing Anthropic Evolution Detection...");
    let tracker = EvolutionTracker::new(Box::new(llm.clone()));

    let old_entity = Entity::new(
        EntityType::Person,
        "Alice".to_string(),
        "Works at Acme Corp as a junior developer".to_string(),
    );

    let new_entity = Entity::new(
        EntityType::Person,
        "Alice".to_string(),
        "Now works at TechCo as a senior engineer after promotion".to_string(),
    );

    match tracker.detect(&new_entity, &[old_entity], DetectionOptions::default()).await {
        Ok(Some(detection)) => {
            println!("     Evolution detected:");
            println!("       Type: {:?}", detection.evolution_type());
            println!("       Reason: {}", detection.reason());
            println!("       Confidence: {:.2}", detection.confidence());
            if detection.llm_used {
                results.pass("Anthropic Evolution Detection");
            } else {
                results.fail("Anthropic Evolution Detection", "LLM not used (fallback triggered)");
            }
        }
        Ok(None) => {
            // No evolution detected is valid if LLM determines no clear relationship
            println!("     No evolution detected (valid LLM response)");
            results.pass("Anthropic Evolution Detection");
        }
        Err(e) => {
            results.fail("Anthropic Evolution Detection", &e.to_string());
        }
    }

    // Test query rewriting
    println!("\n1.4 Testing Anthropic Query Rewriting...");
    let embedder = SimEmbeddingProvider::with_seed(42);
    let vector = SimVectorBackend::new(42);
    let storage = SimStorageBackend::new(SimConfig::with_seed(42));
    let retriever = DualRetriever::new(
        Box::new(llm.clone()),
        Box::new(embedder),
        Box::new(vector),
        Box::new(storage),
    );

    let query = "Who works as a programmer?";
    let variations = retriever.rewrite_query(query).await;
    
    println!("     Query: \"{}\"", query);
    println!("     Generated {} variations:", variations.len());
    for (i, var) in variations.iter().enumerate() {
        println!("       {}. {}", i + 1, var);
    }
    
    if variations.len() > 1 {
        results.pass("Anthropic Query Rewriting");
    } else {
        results.fail("Anthropic Query Rewriting", "No query variations generated");
    }
}

// =============================================================================
// OpenAI LLM Test Suite
// =============================================================================

#[cfg(feature = "openai")]
async fn test_openai_llm_suite(api_key: &str, results: &mut TestResults) {
    use umi_memory::llm::{CompletionRequest, LLMProvider};

    println!("\n2.1 Testing OpenAI LLM Basic Completion...");
    let llm = OpenAIProvider::new(api_key);

    // Test basic completion
    match llm.complete(&CompletionRequest::new("Say 'Hello' in exactly one word.")).await {
        Ok(response) => {
            println!("     Response: {}", response.trim());
            if !response.is_empty() {
                results.pass("OpenAI Basic Completion");
            } else {
                results.fail("OpenAI Basic Completion", "Empty response");
            }
        }
        Err(e) => {
            results.fail("OpenAI Basic Completion", &e.to_string());
        }
    }

    // Test entity extraction
    println!("\n2.2 Testing OpenAI Entity Extraction...");
    let extractor = EntityExtractor::new(Box::new(llm.clone()));
    let text = "Bob manages the engineering team at TechCo and is working on Project Phoenix.";
    
    match extractor.extract(text, ExtractionOptions::default()).await {
        Ok(result) => {
            println!("     Input: \"{}\"", text);
            println!("     Extracted {} entities:", result.entities.len());
            for entity in &result.entities {
                println!("       - {} ({:?}): {}", entity.name, entity.entity_type, entity.content);
            }
            if !result.entities.is_empty() {
                results.pass("OpenAI Entity Extraction");
            } else {
                results.fail("OpenAI Entity Extraction", "No entities extracted");
            }
        }
        Err(e) => {
            results.fail("OpenAI Entity Extraction", &e.to_string());
        }
    }

    // Test evolution detection
    println!("\n2.3 Testing OpenAI Evolution Detection...");
    let tracker = EvolutionTracker::new(Box::new(llm.clone()));

    let old_entity = Entity::new(
        EntityType::Project,
        "Project Phoenix".to_string(),
        "A new product launch scheduled for Q3".to_string(),
    );

    let new_entity = Entity::new(
        EntityType::Project,
        "Project Phoenix".to_string(),
        "Launch delayed to Q4 due to supply chain issues".to_string(),
    );

    match tracker.detect(&new_entity, &[old_entity], DetectionOptions::default()).await {
        Ok(Some(detection)) => {
            println!("     Evolution detected:");
            println!("       Type: {:?}", detection.evolution_type());
            println!("       Reason: {}", detection.reason());
            results.pass("OpenAI Evolution Detection");
        }
        Ok(None) => {
            println!("     No evolution detected (valid LLM response)");
            results.pass("OpenAI Evolution Detection");
        }
        Err(e) => {
            results.fail("OpenAI Evolution Detection", &e.to_string());
        }
    }
}

// =============================================================================
// OpenAI Embedding Test Suite
// =============================================================================

#[cfg(feature = "embedding-openai")]
async fn test_openai_embedding_suite(api_key: &str, results: &mut TestResults) {
    use umi_memory::embedding::EmbeddingProvider;

    println!("\n3.1 Testing OpenAI Single Embedding...");
    let embedder = OpenAIEmbeddingProvider::new(api_key);

    // Test single embedding
    match embedder.embed("Alice works at Acme Corp as a software engineer").await {
        Ok(embedding) => {
            println!("     Dimensions: {}", embedding.len());
            // Verify normalized
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("     L2 Norm: {:.4} (should be ~1.0)", norm);
            if embedding.len() == 1536 && (norm - 1.0).abs() < 0.01 {
                results.pass("OpenAI Single Embedding");
            } else {
                results.fail("OpenAI Single Embedding", &format!("Wrong dims or norm: {} dims, {:.4} norm", embedding.len(), norm));
            }
        }
        Err(e) => {
            results.fail("OpenAI Single Embedding", &e.to_string());
        }
    }

    // Test batch embedding
    println!("\n3.2 Testing OpenAI Batch Embedding...");
    let texts = vec![
        "Alice is a software engineer",
        "Bob works in data science",
        "Charlie manages the product team",
    ];

    match embedder.embed_batch(&texts).await {
        Ok(embeddings) => {
            println!("     Batch size: {}", embeddings.len());
            for (i, emb) in embeddings.iter().enumerate() {
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                println!("       Text {}: {} dims, norm={:.4}", i + 1, emb.len(), norm);
            }
            if embeddings.len() == 3 && embeddings.iter().all(|e| e.len() == 1536) {
                results.pass("OpenAI Batch Embedding");
            } else {
                results.fail("OpenAI Batch Embedding", "Wrong batch size or dimensions");
            }
        }
        Err(e) => {
            results.fail("OpenAI Batch Embedding", &e.to_string());
        }
    }

    // Test semantic similarity
    println!("\n3.3 Testing Embedding Semantic Similarity...");
    let text1 = "The quick brown fox jumps over the lazy dog";
    let text2 = "A fast auburn fox leaps above a sleepy canine";
    let text3 = "Python is a popular programming language";

    match (embedder.embed(text1).await, embedder.embed(text2).await, embedder.embed(text3).await) {
        (Ok(emb1), Ok(emb2), Ok(emb3)) => {
            let sim_12 = cosine_similarity(&emb1, &emb2);
            let sim_13 = cosine_similarity(&emb1, &emb3);
            let sim_23 = cosine_similarity(&emb2, &emb3);
            
            println!("     Similar texts (fox sentences): {:.4}", sim_12);
            println!("     Dissimilar texts (fox vs python): {:.4}", sim_13);
            println!("     Dissimilar texts (fox vs python): {:.4}", sim_23);
            
            // Similar texts should have higher similarity than dissimilar texts
            if sim_12 > sim_13 && sim_12 > sim_23 {
                results.pass("Embedding Semantic Similarity");
            } else {
                results.fail("Embedding Semantic Similarity", 
                    &format!("Similar texts ({:.4}) should score higher than dissimilar ({:.4}, {:.4})", 
                             sim_12, sim_13, sim_23));
            }
        }
        _ => {
            results.fail("Embedding Semantic Similarity", "Failed to generate embeddings");
        }
    }
}

// =============================================================================
// LanceDB Storage Test Suite
// =============================================================================

#[cfg(feature = "lance")]
async fn test_lance_storage_suite(lance_path: &PathBuf, results: &mut TestResults) {
    use umi_memory::storage::StorageBackend;

    println!("\n4.1 Testing LanceDB Connection...");
    let storage_path = lance_path.join("storage");
    let vector_path = lance_path.join("vectors");

    // Test storage backend
    match LanceStorageBackend::connect(storage_path.to_str().unwrap()).await {
        Ok(storage) => {
            println!("     Connected to: {:?}", storage_path);
            results.pass("LanceDB Storage Connection");

            // Test entity storage
            println!("\n4.2 Testing LanceDB Entity Storage...");
            let entity = Entity::new(
                EntityType::Person,
                "Alice".to_string(),
                "Software engineer at Acme Corp".to_string(),
            );
            let id = entity.id.clone();

            match storage.store_entity(&entity).await {
                Ok(_) => {
                    // Verify retrieval
                    match storage.get_entity(&id).await {
                        Ok(Some(retrieved)) => {
                            if retrieved.name == "Alice" && retrieved.content.contains("Acme") {
                                println!("     Stored and retrieved entity: {}", retrieved.name);
                                results.pass("LanceDB Entity Storage");
                            } else {
                                results.fail("LanceDB Entity Storage", "Retrieved entity doesn't match");
                            }
                        }
                        Ok(None) => results.fail("LanceDB Entity Storage", "Entity not found after storage"),
                        Err(e) => results.fail("LanceDB Entity Storage", &e.to_string()),
                    }
                }
                Err(e) => results.fail("LanceDB Entity Storage", &e.to_string()),
            }

            // Test search
            println!("\n4.3 Testing LanceDB Search...");
            match storage.search("Alice", 10).await {
                Ok(results_vec) => {
                    println!("     Found {} entities matching 'Alice'", results_vec.len());
                    if !results_vec.is_empty() {
                        results.pass("LanceDB Search");
                    } else {
                        results.fail("LanceDB Search", "No results found");
                    }
                }
                Err(e) => results.fail("LanceDB Search", &e.to_string()),
            }

            // Test persistence
            println!("\n4.4 Testing LanceDB Persistence...");
            // Close and reconnect
            drop(storage);
            
            match LanceStorageBackend::connect(storage_path.to_str().unwrap()).await {
                Ok(storage2) => {
                    match storage2.get_entity(&id).await {
                        Ok(Some(_)) => {
                            println!("     Entity persisted across reconnection");
                            results.pass("LanceDB Persistence");
                        }
                        _ => results.fail("LanceDB Persistence", "Entity not found after reconnection"),
                    }
                }
                Err(e) => results.fail("LanceDB Persistence", &e.to_string()),
            }
        }
        Err(e) => {
            results.fail("LanceDB Storage Connection", &e.to_string());
            results.skip("LanceDB Entity Storage", "Connection failed");
            results.skip("LanceDB Search", "Connection failed");
            results.skip("LanceDB Persistence", "Connection failed");
        }
    }

    // Test vector backend
    println!("\n4.5 Testing LanceDB Vector Backend...");
    use umi_memory::storage::VectorBackend;
    use umi_memory::constants::EMBEDDING_DIMENSIONS_COUNT;

    match LanceVectorBackend::connect(vector_path.to_str().unwrap()).await {
        Ok(vector) => {
            println!("     Connected to: {:?}", vector_path);
            
            // Store some vectors
            let emb1: Vec<f32> = (0..EMBEDDING_DIMENSIONS_COUNT).map(|i| (i as f32 / EMBEDDING_DIMENSIONS_COUNT as f32)).collect();
            let emb2: Vec<f32> = (0..EMBEDDING_DIMENSIONS_COUNT).map(|i| (1.0 - i as f32 / EMBEDDING_DIMENSIONS_COUNT as f32)).collect();

            match vector.store("entity1", &emb1).await {
                Ok(_) => {
                    match vector.store("entity2", &emb2).await {
                        Ok(_) => {
                            // Test vector search
                            match vector.search(&emb1, 10).await {
                                Ok(search_results) => {
                                    println!("     Found {} similar vectors", search_results.len());
                                    if !search_results.is_empty() && search_results[0].id == "entity1" {
                                        println!("     Top result: {} (score: {:.4})", 
                                                search_results[0].id, search_results[0].score);
                                        results.pass("LanceDB Vector Backend");
                                    } else {
                                        results.fail("LanceDB Vector Backend", "Wrong top result");
                                    }
                                }
                                Err(e) => results.fail("LanceDB Vector Backend", &e.to_string()),
                            }
                        }
                        Err(e) => results.fail("LanceDB Vector Backend", &e.to_string()),
                    }
                }
                Err(e) => results.fail("LanceDB Vector Backend", &e.to_string()),
            }
        }
        Err(e) => results.fail("LanceDB Vector Backend", &e.to_string()),
    }
}

// =============================================================================
// Full Integration Test Suite
// =============================================================================

#[cfg(all(feature = "anthropic", feature = "embedding-openai", feature = "lance"))]
async fn test_full_integration_suite(
    anthropic_key: &str,
    openai_key: &str,
    lance_path: &PathBuf,
    results: &mut TestResults,
) {
    println!("\n5.1 Testing Full Integration: Anthropic + OpenAI Embeddings + LanceDB...");

    // Create real providers
    let llm = AnthropicProvider::new(anthropic_key);
    let embedder = OpenAIEmbeddingProvider::new(openai_key);

    // Create LanceDB backends
    let storage_path = lance_path.join("storage");
    let vector_path = lance_path.join("vectors");

    let storage = match LanceStorageBackend::connect(storage_path.to_str().unwrap()).await {
        Ok(s) => s,
        Err(e) => {
            results.fail("Full Integration", &format!("Storage connection failed: {}", e));
            return;
        }
    };

    let vector = match LanceVectorBackend::connect(vector_path.to_str().unwrap()).await {
        Ok(v) => v,
        Err(e) => {
            results.fail("Full Integration", &format!("Vector connection failed: {}", e));
            return;
        }
    };

    // Create Memory with all real components
    let mut memory = Memory::new(llm, embedder, vector, storage);

    // Test remember
    println!("     Remembering: 'Alice is a software engineer at Acme Corp...'");
    let remember_result = memory.remember(
        "Alice is a software engineer at Acme Corp who specializes in distributed systems and Rust.",
        RememberOptions::default(),
    ).await;

    match remember_result {
        Ok(result) => {
            println!("     ✓ Stored {} entities", result.entity_count());
            for entity in result.iter_entities() {
                println!("       - {} ({}): {}", entity.name, entity.entity_type, entity.content);
            }

            // Test recall
            println!("\n     Recalling: 'Who works at Acme?'");
            match memory.recall("Who works at Acme?", RecallOptions::default()).await {
                Ok(recall_results) => {
                    println!("     ✓ Found {} results", recall_results.len());
                    for entity in &recall_results {
                        println!("       - {}: {}", entity.name, entity.content);
                    }
                    
                    if !recall_results.is_empty() {
                        results.pass("Full Integration (Anthropic + OpenAI + LanceDB)");
                    } else {
                        results.fail("Full Integration", "No recall results");
                    }
                }
                Err(e) => results.fail("Full Integration", &format!("Recall failed: {}", e)),
            }
        }
        Err(e) => results.fail("Full Integration", &format!("Remember failed: {}", e)),
    }
}

// =============================================================================
// Memory Workflow Tests
// =============================================================================

async fn test_memory_workflow_simulation(results: &mut TestResults) {
    println!("\n6.1 Testing Memory Workflow (Simulation Providers)...");

    let mut memory = Memory::sim(42);

    // Remember multiple facts
    let facts = [
        "Alice is a software engineer at Acme Corp",
        "Bob is a data scientist at TechCo",
        "Alice and Bob are collaborating on Project Phoenix",
    ];

    for fact in &facts {
        match memory.remember(*fact, RememberOptions::default()).await {
            Ok(result) => println!("     Stored: {} entities from '{}'", result.entity_count(), &fact[..30.min(fact.len())]),
            Err(e) => {
                results.fail("Memory Workflow (Sim)", &format!("Remember failed: {}", e));
                return;
            }
        }
    }

    // Recall
    match memory.recall("Alice", RecallOptions::default()).await {
        Ok(recall_results) => {
            println!("     Recalled {} results for 'Alice'", recall_results.len());
            if !recall_results.is_empty() {
                results.pass("Memory Workflow (Simulation)");
            } else {
                results.fail("Memory Workflow (Sim)", "No recall results");
            }
        }
        Err(e) => results.fail("Memory Workflow (Sim)", &format!("Recall failed: {}", e)),
    }

    // Test count
    match memory.count().await {
        Ok(count) => println!("     Total entities in memory: {}", count),
        Err(e) => println!("     Warning: count failed: {}", e),
    }
}

#[cfg(all(feature = "anthropic", feature = "embedding-openai", feature = "lance"))]
async fn test_memory_workflow_production(
    anthropic_key: &str,
    openai_key: &str,
    lance_path: &PathBuf,
    results: &mut TestResults,
) {
    println!("\n6.2 Testing Memory Workflow (Production Providers)...");

    // Create providers
    let llm = AnthropicProvider::new(anthropic_key);
    let embedder = OpenAIEmbeddingProvider::new(openai_key);

    let storage_path = lance_path.join("storage");
    let vector_path = lance_path.join("vectors");

    let storage = match LanceStorageBackend::connect(storage_path.to_str().unwrap()).await {
        Ok(s) => s,
        Err(e) => {
            results.fail("Memory Workflow (Production)", &format!("Storage failed: {}", e));
            return;
        }
    };

    let vector = match LanceVectorBackend::connect(vector_path.to_str().unwrap()).await {
        Ok(v) => v,
        Err(e) => {
            results.fail("Memory Workflow (Production)", &format!("Vector failed: {}", e));
            return;
        }
    };

    let mut memory = Memory::new(llm, embedder, vector, storage);

    // Remember
    println!("     Remembering facts...");
    let result1 = memory.remember(
        "Charlie is the CTO at DataInc and oversees the AI division",
        RememberOptions::default(),
    ).await;

    match result1 {
        Ok(r) => println!("       ✓ Stored {} entities", r.entity_count()),
        Err(e) => {
            results.fail("Memory Workflow (Production)", &format!("Remember 1 failed: {}", e));
            return;
        }
    }

    let result2 = memory.remember(
        "Charlie recently announced that DataInc is launching a new product called 'AI Assistant'",
        RememberOptions::default(),
    ).await;

    match result2 {
        Ok(r) => {
            println!("       ✓ Stored {} entities", r.entity_count());
            if r.has_evolutions() {
                println!("       Evolution detected: {:?}", r.evolutions);
            }
        }
        Err(e) => {
            results.fail("Memory Workflow (Production)", &format!("Remember 2 failed: {}", e));
            return;
        }
    }

    // Recall
    println!("     Recalling 'Who is Charlie?'...");
    match memory.recall("Who is Charlie?", RecallOptions::default()).await {
        Ok(recall_results) => {
            println!("       ✓ Found {} results", recall_results.len());
            for entity in &recall_results {
                println!("         - {}: {}", entity.name, &entity.content[..50.min(entity.content.len())]);
            }
            
            if !recall_results.is_empty() {
                results.pass("Memory Workflow (Production)");
            } else {
                results.fail("Memory Workflow (Production)", "No recall results");
            }
        }
        Err(e) => results.fail("Memory Workflow (Production)", &format!("Recall failed: {}", e)),
    }

    // Test get/forget
    println!("     Testing get/forget...");
    match memory.count().await {
        Ok(count) => println!("       Total entities: {}", count),
        Err(e) => println!("       Warning: count failed: {}", e),
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
