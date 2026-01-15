//! User Experience Test v2 - Deeper investigation
//!
//! Testing specific issues found in v1

use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Umi Memory - Detailed Investigation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let has_anthropic = env::var("ANTHROPIC_API_KEY").is_ok();
    let has_openai = env::var("OPENAI_API_KEY").is_ok();

    if !has_anthropic || !has_openai {
        println!("⚠️  Missing API keys");
        return Ok(());
    }

    let temp_dir = tempfile::tempdir()?;
    let db_path = temp_dir.path().join("test_memory");
    std::fs::create_dir_all(&db_path)?;

    use umi_memory::embedding::OpenAIEmbeddingProvider;
    use umi_memory::llm::AnthropicProvider;
    use umi_memory::storage::{LanceStorageBackend, LanceVectorBackend};
    use umi_memory::umi::{Memory, RecallOptions, RememberOptions};

    let llm = AnthropicProvider::new(env::var("ANTHROPIC_API_KEY").unwrap());
    let embedder = OpenAIEmbeddingProvider::new(env::var("OPENAI_API_KEY").unwrap());
    let vector = LanceVectorBackend::connect(db_path.join("vectors").to_str().unwrap()).await?;
    let storage = LanceStorageBackend::connect(db_path.join("entities").to_str().unwrap()).await?;

    let mut memory = Memory::new(llm, embedder, vector, storage);

    // =========================================================================
    // Test 1: Simple fact storage and immediate recall
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Simple Storage and Recall");
    println!("═══════════════════════════════════════════════════════════════\n");

    let fact = "John Smith is a software developer at Google.";
    println!("   Storing: \"{}\"", fact);
    let result = memory.remember(fact, RememberOptions::default()).await?;
    println!("   → Stored {} entities:", result.entities.len());
    for e in &result.entities {
        println!(
            "      - {} ({}) [id: {}]: {}",
            e.name,
            e.entity_type.as_str(),
            &e.id[..8],
            e.content
        );
    }

    println!("\n   Querying: \"Who is John Smith?\"");
    let recall = memory
        .recall("Who is John Smith?", RecallOptions::default())
        .await?;
    println!("   → Found {} results:", recall.len());
    for e in &recall {
        println!(
            "      - {} ({}) [id: {}]: {}",
            e.name,
            e.entity_type.as_str(),
            &e.id[..8],
            e.content
        );
    }

    println!("\n   Querying: \"John Smith\"");
    let recall2 = memory
        .recall("John Smith", RecallOptions::default())
        .await?;
    println!("   → Found {} results:", recall2.len());
    for e in &recall2 {
        println!(
            "      - {} ({}) [id: {}]",
            e.name,
            e.entity_type.as_str(),
            &e.id[..8]
        );
    }

    println!("\n   Querying: \"Google\"");
    let recall3 = memory.recall("Google", RecallOptions::default()).await?;
    println!("   → Found {} results:", recall3.len());
    for e in &recall3 {
        println!(
            "      - {} ({}) [id: {}]",
            e.name,
            e.entity_type.as_str(),
            &e.id[..8]
        );
    }

    // =========================================================================
    // Test 2: Check what's actually in storage
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: Direct Storage Inspection");
    println!("═══════════════════════════════════════════════════════════════\n");

    let count = memory.count().await?;
    println!("   Total entities in storage: {}", count);

    // Get all with a very broad query
    println!("\n   Trying various recall queries to list all:");

    for query in &[
        "*",
        "all",
        "everything",
        "list all entities",
        "software developer Google John",
    ] {
        let results = memory.recall(query, RecallOptions::default()).await?;
        println!("   Query \"{}\": {} results", query, results.len());
    }

    // =========================================================================
    // Test 3: Test individual entity retrieval by ID
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 3: Get Entity by ID");
    println!("═══════════════════════════════════════════════════════════════\n");

    if !result.entities.is_empty() {
        let first_id = &result.entities[0].id;
        println!("   Fetching entity by ID: {}", first_id);
        match memory.get(first_id).await? {
            Some(entity) => println!(
                "   → Found: {} ({})",
                entity.name,
                entity.entity_type.as_str()
            ),
            None => println!("   → NOT FOUND (this is a bug!)"),
        }
    }

    // =========================================================================
    // Test 4: Check if embeddings are being stored
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 4: Embedding Check");
    println!("═══════════════════════════════════════════════════════════════\n");

    for entity in &result.entities {
        let has_embedding = entity.embedding.is_some();
        let embedding_len = entity.embedding.as_ref().map(|e| e.len()).unwrap_or(0);
        println!(
            "   Entity '{}': embedding={} (dims={})",
            entity.name,
            if has_embedding { "yes" } else { "no" },
            embedding_len
        );
    }

    // =========================================================================
    // Test 5: Duplicate Detection
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 5: Duplicate Entity Detection");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Store the same fact again
    println!("   Storing same fact again: \"{}\"", fact);
    let result2 = memory.remember(fact, RememberOptions::default()).await?;
    println!("   → Stored {} entities", result2.entities.len());

    let count_after = memory.count().await?;
    println!("   Total entities now: {} (was {})", count_after, count);

    if count_after > count {
        println!("   ⚠️  DUPLICATE ISSUE: Count increased (expected deduplication)");
    } else {
        println!("   ✓ No duplicate entities created");
    }

    // =========================================================================
    // Test 6: Recall with different limits
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 6: Recall Limit Behavior");
    println!("═══════════════════════════════════════════════════════════════\n");

    for limit in &[1, 5, 10, 50] {
        let results = memory
            .recall("John", RecallOptions::default().with_limit(*limit)?)
            .await?;
        println!("   Limit {}: {} results", limit, results.len());
    }

    // =========================================================================
    // Test 7: Similarity Threshold
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 7: Does relevance threshold exist?");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Query for completely unrelated topic
    let unrelated_results = memory
        .recall("quantum physics black holes", RecallOptions::default())
        .await?;
    println!(
        "   Query for unrelated topic 'quantum physics black holes': {} results",
        unrelated_results.len()
    );

    if unrelated_results.is_empty() {
        println!("   ✓ Good - no false positives for unrelated query");
    } else {
        println!("   ⚠️  Returned results for completely unrelated query");
        for e in &unrelated_results {
            println!("      - {} ({})", e.name, e.entity_type.as_str());
        }
    }

    println!("\n✅ Investigation complete!\n");

    Ok(())
}
