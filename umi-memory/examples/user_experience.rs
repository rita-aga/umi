//! User Experience Test - Using Umi as a Real Developer Would
//!
//! This script simulates a realistic usage scenario: building an AI assistant
//! that remembers information about a user and their projects.

use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Umi Memory - Real User Experience Session");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Check for API keys
    let has_anthropic = env::var("ANTHROPIC_API_KEY").is_ok();
    let has_openai = env::var("OPENAI_API_KEY").is_ok();

    if !has_anthropic || !has_openai {
        println!("⚠️  Missing API keys. Set ANTHROPIC_API_KEY and OPENAI_API_KEY");
        return Ok(());
    }

    // Create a temporary directory for our "production" database
    let temp_dir = tempfile::tempdir()?;
    let db_path = temp_dir.path().join("my_agent_memory");
    std::fs::create_dir_all(&db_path)?;

    println!("📁 Database path: {:?}\n", db_path);

    // =========================================================================
    // SCENARIO: Building a Personal AI Assistant
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("SCENARIO: Building a Personal AI Assistant with Memory");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Step 1: Initialize Memory with real providers
    println!("Step 1: Initializing Memory System...\n");

    use umi_memory::embedding::OpenAIEmbeddingProvider;
    use umi_memory::llm::AnthropicProvider;
    use umi_memory::storage::{LanceStorageBackend, LanceVectorBackend};
    use umi_memory::umi::{Memory, RecallOptions, RememberOptions};

    let llm = AnthropicProvider::new(env::var("ANTHROPIC_API_KEY").unwrap());
    let embedder = OpenAIEmbeddingProvider::new(env::var("OPENAI_API_KEY").unwrap());
    let vector = LanceVectorBackend::connect(db_path.join("vectors").to_str().unwrap()).await?;
    let storage = LanceStorageBackend::connect(db_path.join("entities").to_str().unwrap()).await?;

    let mut memory = Memory::new(llm, embedder, vector, storage);
    println!("   ✓ Memory system initialized\n");

    // =========================================================================
    // Session 1: User introduces themselves
    // =========================================================================

    println!("───────────────────────────────────────────────────────────────");
    println!("Session 1: User Introduction");
    println!("───────────────────────────────────────────────────────────────\n");

    let intro_messages = vec![
        "Hi! My name is Sarah Chen. I'm a machine learning engineer based in San Francisco.",
        "I work at a startup called NeuralFlow, where I lead the recommendation systems team.",
        "I'm currently working on a project called Athena - it's a personalized learning platform.",
        "My favorite programming languages are Python and Rust. I've been learning Rust for about 6 months.",
    ];

    for msg in &intro_messages {
        println!("   User: \"{}\"", msg);
        let result = memory.remember(msg, RememberOptions::default()).await?;
        println!("   → Remembered {} entities\n", result.entities.len());
    }

    // Let's see what we remember
    println!(
        "\n   📊 Memory Status: {} entities stored\n",
        memory.count().await?
    );

    // =========================================================================
    // Session 2: Ask questions about the user
    // =========================================================================

    println!("───────────────────────────────────────────────────────────────");
    println!("Session 2: Testing Recall - What do we know about Sarah?");
    println!("───────────────────────────────────────────────────────────────\n");

    let queries = vec![
        "What is Sarah's job?",
        "Where does Sarah work?",
        "What project is Sarah working on?",
        "What programming languages does Sarah know?",
        "Who is Sarah?", // Broad query
    ];

    for query in &queries {
        println!("   Query: \"{}\"", query);
        let results = memory.recall(query, RecallOptions::default()).await?;
        if results.is_empty() {
            println!("   → No results found\n");
        } else {
            println!("   → Found {} results:", results.len());
            for (i, entity) in results.iter().take(3).enumerate() {
                println!(
                    "      {}. {} ({}): {}",
                    i + 1,
                    entity.name,
                    entity.entity_type.as_str(),
                    truncate(&entity.content, 60)
                );
            }
            println!();
        }
    }

    // =========================================================================
    // Session 3: Updates and Evolution
    // =========================================================================

    println!("───────────────────────────────────────────────────────────────");
    println!("Session 3: Information Updates (Testing Evolution)");
    println!("───────────────────────────────────────────────────────────────\n");

    let updates = vec![
        "Sarah just got promoted to Senior ML Engineer at NeuralFlow!",
        "The Athena project launched successfully last week. It now has 10,000 active users.",
        "Sarah is now also learning Go in addition to Rust.",
    ];

    for update in &updates {
        println!("   User: \"{}\"", update);
        let result = memory.remember(update, RememberOptions::default()).await?;
        println!("   → Stored {} entities", result.entities.len());

        // Check for evolution
        if !result.evolutions.is_empty() {
            println!("   🔄 Evolution detected:");
            for rel in &result.evolutions {
                println!(
                    "      - Type: {:?}, Confidence: {:.2}",
                    rel.evolution_type, rel.confidence
                );
                println!("        Reason: {}", truncate(&rel.reason, 70));
            }
        }
        println!();
    }

    println!(
        "   📊 Memory Status: {} entities stored\n",
        memory.count().await?
    );

    // =========================================================================
    // Session 4: Complex Queries
    // =========================================================================

    println!("───────────────────────────────────────────────────────────────");
    println!("Session 4: Complex Queries");
    println!("───────────────────────────────────────────────────────────────\n");

    let complex_queries = vec![
        "What has Sarah achieved recently?",
        "Tell me about Sarah's technical skills",
        "What is the status of Sarah's projects?",
        "How has Sarah's role changed?",
    ];

    for query in &complex_queries {
        println!("   Query: \"{}\"", query);
        let results = memory
            .recall(query, RecallOptions::default().with_limit(5)?)
            .await?;
        println!("   → Found {} relevant memories:", results.len());
        for entity in results.iter().take(3) {
            println!("      • {}: {}", entity.name, truncate(&entity.content, 50));
        }
        println!();
    }

    // =========================================================================
    // Session 5: Edge Cases
    // =========================================================================

    println!("───────────────────────────────────────────────────────────────");
    println!("Session 5: Edge Cases");
    println!("───────────────────────────────────────────────────────────────\n");

    // Empty query
    println!("   Test: Empty string query");
    match memory.recall("", RecallOptions::default()).await {
        Ok(results) => println!(
            "   → Returned {} results (handled gracefully)\n",
            results.len()
        ),
        Err(e) => println!("   → Error: {} (should this be handled better?)\n", e),
    }

    // Very long input
    println!("   Test: Very long input text");
    let long_text = "Sarah mentioned that ".to_string()
        + &"she really enjoys working on machine learning projects. ".repeat(50);
    match memory
        .remember(&long_text, RememberOptions::default())
        .await
    {
        Ok(result) => println!(
            "   → Handled long text, extracted {} entities\n",
            result.entities.len()
        ),
        Err(e) => println!("   → Error with long text: {}\n", e),
    }

    // Query for non-existent entity
    println!("   Test: Query for entity that doesn't exist");
    let results = memory
        .recall("Who is John Smith?", RecallOptions::default())
        .await?;
    println!(
        "   → Found {} results (expected 0 or low relevance)\n",
        results.len()
    );

    // Unicode handling
    println!("   Test: Unicode in names and text");
    let unicode_result = memory
        .remember(
            "Sarah's colleague 田中太郎 (Tanaka Taro) from Tokyo joined the project.",
            RememberOptions::default(),
        )
        .await?;
    println!(
        "   → Extracted {} entities with unicode\n",
        unicode_result.entities.len()
    );

    // =========================================================================
    // Final Summary
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("FINAL SUMMARY");
    println!("═══════════════════════════════════════════════════════════════\n");

    let total = memory.count().await?;
    println!("   Total entities in memory: {}\n", total);

    // Get all entities and summarize
    println!("   All remembered entities:");
    let all_results = memory
        .recall("*", RecallOptions::default().with_limit(50)?)
        .await?;

    let mut by_type: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for entity in &all_results {
        by_type
            .entry(entity.entity_type.as_str().to_string())
            .or_default()
            .push(entity.name.clone());
    }

    for (entity_type, names) in &by_type {
        println!("      {}: {}", entity_type, names.join(", "));
    }

    println!("\n   ✅ Session complete!\n");

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
