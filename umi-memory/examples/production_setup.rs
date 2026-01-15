//! Production Memory Setup Example
//!
//! Demonstrates setting up Memory for production use with:
//! - Builder pattern for explicit configuration
//! - LanceDB vector backend (persistent storage)
//! - Custom configuration
//!
//! Note: This example uses simulation components for demonstration.
//! In production, replace with real LLM providers and storage backends.
//!
//! Run with:
//!   cargo run --example production_setup

use std::time::Duration;
use umi_memory::dst::SimConfig;
use umi_memory::embedding::SimEmbeddingProvider;
use umi_memory::llm::SimLLMProvider;
use umi_memory::storage::{SimStorageBackend, SimVectorBackend};
use umi_memory::umi::{Memory, MemoryConfig, RecallOptions, RememberOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Umi Memory: Production Setup ===\n");

    // === Direct Construction with Explicit Components ===
    println!("--- Direct Construction (Explicit Configuration) ---");

    // Create each component explicitly
    let llm = SimLLMProvider::with_seed(42);
    let embedder = SimEmbeddingProvider::with_seed(42);
    let vector_backend = SimVectorBackend::new(42);
    let storage_backend = SimStorageBackend::new(SimConfig::with_seed(42));

    // Assemble with Memory::new()
    let mut memory = Memory::new(llm, embedder, vector_backend, storage_backend);

    println!("✓ Created Memory with direct construction");
    println!("  - LLM: SimLLMProvider (seed: 42)");
    println!("  - Embedder: SimEmbeddingProvider (seed: 42)");
    println!("  - Vector: SimVectorBackend (in-memory)");
    println!("  - Storage: SimStorageBackend (in-memory)");
    println!();

    // Test the memory
    let result = memory
        .remember("Production test entity", RememberOptions::default())
        .await?;
    println!("  Stored {} entities\n", result.entity_count());

    // === Production Configuration ===
    println!("--- Production Configuration ---");

    let production_config = MemoryConfig::default()
        .with_core_memory_bytes(128 * 1024) // 128 KB for larger context
        .with_working_memory_bytes(10 * 1024 * 1024) // 10 MB working memory
        .with_working_memory_ttl(Duration::from_secs(3600 * 4)) // 4 hours
        .with_recall_limit(50) // Return more results
        .with_embedding_batch_size(500); // Larger batches for production
                                         // Note: semantic_search_enabled and query_expansion_enabled are true by default

    println!("✓ Production configuration:");
    println!(
        "  - Core memory: {} KB",
        production_config.core_memory_bytes / 1024
    );
    println!(
        "  - Working memory: {} MB",
        production_config.working_memory_bytes / (1024 * 1024)
    );
    println!(
        "  - Working memory TTL: {} hours",
        production_config.working_memory_ttl.as_secs() / 3600
    );
    println!(
        "  - Recall limit: {}",
        production_config.default_recall_limit
    );
    println!(
        "  - Embedding batch size: {}",
        production_config.embedding_batch_size
    );
    println!(
        "  - Semantic search: {}",
        production_config.semantic_search_enabled
    );
    println!(
        "  - Query expansion: {}",
        production_config.query_expansion_enabled
    );
    println!();

    // === LanceDB Backend (Conceptual) ===
    println!("--- LanceDB Backend (Conceptual Example) ---");
    println!("In production, you would use:");
    println!(
        r#"
    #[cfg(feature = "lance")]
    {{
        use umi_memory::storage::{{LanceVectorBackend, LanceStorageBackend}};

        // Connect to persistent LanceDB
        let vector = LanceVectorBackend::connect("./production_vectors.lance")
            .await
            .expect("Failed to connect to LanceDB");
        let storage = LanceStorageBackend::connect("./production_storage.lance")
            .await
            .expect("Failed to connect to LanceDB");

        let memory = Memory::new(llm, embedder, vector, storage);
    }}
    "#
    );
    println!();

    // === Anthropic Provider (Conceptual) ===
    println!("--- Real LLM Provider (Conceptual Example) ---");
    println!("In production with Anthropic:");
    println!(
        r#"
    #[cfg(all(feature = "anthropic", feature = "embedding-openai", feature = "lance"))]
    {{
        use umi_memory::llm::AnthropicProvider;
        use umi_memory::embedding::OpenAIEmbeddingProvider;
        use umi_memory::storage::{{LanceVectorBackend, LanceStorageBackend}};

        let llm = AnthropicProvider::new(
            std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set")
        );
        let embedder = OpenAIEmbeddingProvider::new(
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set")
        );
        let vector = LanceVectorBackend::connect("./vectors.lance").await?;
        let storage = LanceStorageBackend::connect("./storage.lance").await?;

        let memory = Memory::new(llm, embedder, vector, storage);
    }}
    "#
    );
    println!();

    // === Error Handling ===
    println!("--- Production Error Handling ---");

    let result = memory
        .remember("Test for error handling", RememberOptions::default())
        .await;

    match result {
        Ok(r) => println!("  ✓ Remember succeeded: {} entities", r.entity_count()),
        Err(e) => println!("  ✗ Remember failed: {}", e),
    }

    let query_result = memory.recall("test", RecallOptions::default()).await;

    match query_result {
        Ok(results) => println!("  ✓ Recall succeeded: {} results", results.len()),
        Err(e) => println!("  ✗ Recall failed: {}", e),
    }
    println!();

    // === Graceful Degradation ===
    println!("--- Graceful Degradation ---");

    let degraded_config = MemoryConfig::default()
        .without_embeddings()
        .without_query_expansion();

    let mut degraded_memory = Memory::sim_with_config(42, degraded_config);

    println!("✓ Memory with graceful degradation:");
    println!("  - If embeddings fail: fallback to text search");
    println!("  - If query expansion fails: use query as-is");
    println!("  - System continues operating despite component failures");
    println!();

    degraded_memory
        .remember("Graceful degradation test", RememberOptions::default())
        .await?;

    let degraded_results = degraded_memory
        .recall("graceful", RecallOptions::default())
        .await?;

    println!(
        "  Recall with degraded config: {} results",
        degraded_results.len()
    );
    println!();

    println!("=== Example Complete ===");
    println!("\nProduction Checklist:");
    println!("  [ ] Set environment variables (ANTHROPIC_API_KEY, etc.)");
    println!("  [ ] Configure persistent storage (LanceDB path)");
    println!("  [ ] Set appropriate memory limits for your use case");
    println!("  [ ] Enable semantic search and query expansion");
    println!("  [ ] Implement proper error handling");
    println!("  [ ] Add monitoring and logging");
    println!("  [ ] Test with production-like workloads");

    Ok(())
}
