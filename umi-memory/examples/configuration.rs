//! Memory Configuration Example
//!
//! Demonstrates MemoryConfig for customizing Memory behavior.
//!
//! Run with:
//!   cargo run --example configuration

use std::time::Duration;
use umi_memory::umi::{Memory, MemoryConfig, RecallOptions, RememberOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Umi Memory: Configuration Examples ===\n");

    // === Default Configuration ===
    println!("--- Default Configuration ---");
    let mut default_memory = Memory::sim(42);
    println!("✓ Created Memory with default config");
    println!("  - Core memory: 32 KB");
    println!("  - Working memory: 1 MB, TTL: 1 hour");
    println!("  - Embeddings: enabled");
    println!("  - Recall limit: 10");
    println!();

    // Store some data
    default_memory
        .remember("Test entity", RememberOptions::default())
        .await?;

    // === Custom Configuration ===
    println!("--- Custom Configuration ---");

    let custom_config = MemoryConfig::default()
        .with_core_memory_bytes(64 * 1024) // 64 KB core memory
        .with_working_memory_bytes(2 * 1024 * 1024) // 2 MB working memory
        .with_working_memory_ttl(Duration::from_secs(7200)) // 2 hours
        .with_recall_limit(20) // Return up to 20 results
        .with_embedding_batch_size(200); // Process 200 at a time

    let _custom_memory = Memory::sim_with_config(42, custom_config.clone());
    println!("✓ Created Memory with custom config");
    println!(
        "  - Core memory: {} KB",
        custom_config.core_memory_bytes / 1024
    );
    println!(
        "  - Working memory: {} MB",
        custom_config.working_memory_bytes / (1024 * 1024)
    );
    println!(
        "  - Working memory TTL: {} seconds",
        custom_config.working_memory_ttl.as_secs()
    );
    println!("  - Recall limit: {}", custom_config.default_recall_limit);
    println!(
        "  - Embedding batch size: {}",
        custom_config.embedding_batch_size
    );
    println!();

    // === Disable Embeddings (Graceful Degradation) ===
    println!("--- Configuration: Without Embeddings ---");

    let no_embedding_config = MemoryConfig::default().without_embeddings();

    let mut no_embedding_memory = Memory::sim_with_config(42, no_embedding_config);
    println!("✓ Created Memory with embeddings disabled");
    println!("  (System will use text search instead of vector search)");
    println!();

    // Still works with text search
    no_embedding_memory
        .remember("Alice works at Acme", RememberOptions::default())
        .await?;

    let results = no_embedding_memory
        .recall("Alice", RecallOptions::default())
        .await?;

    println!(
        "  Recall found {} results (using text search)",
        results.len()
    );
    println!();

    // === Disable Query Expansion ===
    println!("--- Configuration: Without Query Expansion ---");

    let no_expansion_config = MemoryConfig::default().without_query_expansion();

    let _no_expansion_memory = Memory::sim_with_config(42, no_expansion_config);
    println!("✓ Created Memory with query expansion disabled");
    println!("  (Queries will be used as-is, no LLM semantic expansion)");
    println!();

    // === Combining Multiple Options ===
    println!("--- Configuration: Combined Options ---");

    let combined_config = MemoryConfig::default()
        .with_core_memory_bytes(128 * 1024) // 128 KB
        .with_recall_limit(50) // Return up to 50 results
        .without_embeddings() // No embeddings
        .without_query_expansion(); // No query expansion

    let mut combined_memory = Memory::sim_with_config(42, combined_config.clone());
    println!("✓ Created Memory with combined config:");
    println!(
        "  - Core memory: {} KB",
        combined_config.core_memory_bytes / 1024
    );
    println!("  - Recall limit: {}", combined_config.default_recall_limit);
    println!(
        "  - Embeddings: {}",
        if combined_config.generate_embeddings {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "  - Query expansion: {}",
        if combined_config.query_expansion_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!();

    // === Runtime Recall Options ===
    println!("--- Runtime Recall Options ---");

    // Store multiple entities
    for i in 0..30 {
        combined_memory
            .remember(
                &format!("Entity {} is a test item", i),
                RememberOptions::default(),
            )
            .await?;
    }

    // Override recall limit at query time
    let limited_results = combined_memory
        .recall("entity", RecallOptions::default().with_limit(5)?)
        .await?;

    println!(
        "  Config default recall limit: {}",
        combined_config.default_recall_limit
    );
    println!(
        "  Actual results returned: {} (using RecallOptions.with_limit(5))",
        limited_results.len()
    );
    println!();

    println!("=== Example Complete ===");
    Ok(())
}
