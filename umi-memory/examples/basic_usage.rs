//! Basic Memory Usage Example
//!
//! Demonstrates the core remember/recall workflow with Memory::sim().
//!
//! Run with:
//!   cargo run --example basic_usage

use umi_memory::umi::{Memory, RecallOptions, RememberOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Umi Memory: Basic Usage ===\n");

    // Create a deterministic memory with seed 42
    // Same seed = same results (perfect for testing)
    let mut memory = Memory::sim(42);
    println!("✓ Created Memory with seed 42\n");

    // === Remember Facts ===
    println!("--- Remembering facts ---");

    let result1 = memory
        .remember("Alice works at Acme Corp as a software engineer", RememberOptions::default())
        .await?;
    println!("  Stored {} entities from first remember", result1.entity_count());

    let result2 = memory
        .remember("Bob is the CTO at TechCo", RememberOptions::default())
        .await?;
    println!("  Stored {} entities from second remember", result2.entity_count());

    let result3 = memory
        .remember("Alice and Bob are collaborating on a new project", RememberOptions::default())
        .await?;
    println!("  Stored {} entities from third remember\n", result3.entity_count());

    // === Recall Memories ===
    println!("--- Recalling memories ---");

    // Query 1: Specific person
    let alice_results = memory
        .recall("Alice", RecallOptions::default())
        .await?;
    println!("  Query: 'Alice'");
    println!("  Found {} results:", alice_results.len());
    for entity in alice_results.iter().take(3) {
        println!("    - {}: {}", entity.name, entity.content);
    }
    println!();

    // Query 2: Company
    let acme_results = memory
        .recall("Acme Corp", RecallOptions::default())
        .await?;
    println!("  Query: 'Acme Corp'");
    println!("  Found {} results:", acme_results.len());
    for entity in acme_results.iter().take(3) {
        println!("    - {}: {}", entity.name, entity.content);
    }
    println!();

    // Query 3: Semantic search
    let engineer_results = memory
        .recall("Who are the engineers?", RecallOptions::default())
        .await?;
    println!("  Query: 'Who are the engineers?'");
    println!("  Found {} results:", engineer_results.len());
    for entity in engineer_results.iter().take(3) {
        println!("    - {}: {}", entity.name, entity.content);
    }
    println!();

    // === Evolution Tracking ===
    println!("--- Testing evolution tracking ---");

    let update_result = memory
        .remember("Alice now works at TechCo", RememberOptions::default())
        .await?;
    println!("  Stored {} entities from update", update_result.entity_count());

    if update_result.has_evolutions() {
        println!("  ✓ Evolution detected! ({} evolutions)", update_result.evolutions.len());
        for evolution in &update_result.evolutions {
            println!("    - Type: {:?}", evolution.evolution_type);
        }
    } else {
        println!("  (No evolution detected - depends on LLM behavior)");
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
